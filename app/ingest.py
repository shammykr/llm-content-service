"""Ingestion pipeline: preprocess documents, queue them in Redis, process with async workers.

Each document is stored as a Redis hash (ingest:doc:<id>) and its id is pushed onto a
Redis list (ingest:queue). Worker coroutines pop ids and summarize the documents through
ContentService, so they share the response cache and token batching with the API.
"""

import asyncio
import logging
import uuid

from redis.asyncio import Redis

from .preprocess import preprocess
from .service import ContentService

log = logging.getLogger(__name__)

QUEUE_KEY = "ingest:queue"
DOC_TTL_SECONDS = 7 * 24 * 3600


def _doc_key(doc_id: str) -> str:
    return f"ingest:doc:{doc_id}"


class IngestPipeline:
    def __init__(self, redis: Redis, service: ContentService, workers: int):
        self._redis = redis
        self._service = service
        self._n_workers = workers
        self._workers: list[asyncio.Task] = []

    async def enqueue(self, documents: list[dict]) -> list[dict]:
        """Preprocess and queue documents. Returns per-document metadata."""
        # Preprocessing is CPU-bound (language detection); keep it off the event loop.
        processed = await asyncio.to_thread(lambda: [preprocess(d["text"]) for d in documents])
        acks = []
        pipe = self._redis.pipeline(transaction=False)
        for doc, meta in zip(documents, processed):
            doc_id = doc.get("id") or uuid.uuid4().hex
            record = {"id": doc_id, "status": "queued", **meta}
            pipe.hset(_doc_key(doc_id), mapping=record)
            pipe.expire(_doc_key(doc_id), DOC_TTL_SECONDS)
            pipe.lpush(QUEUE_KEY, doc_id)
            acks.append({k: v for k, v in record.items() if k != "text"})
        await pipe.execute()
        return acks

    async def get(self, doc_id: str) -> dict | None:
        record = await self._redis.hgetall(_doc_key(doc_id))
        if not record:
            return None
        for field in ("token_count", "sentence_count"):
            record[field] = int(record[field])
        return record

    def start(self) -> None:
        self._workers = [asyncio.create_task(self._worker(i)) for i in range(self._n_workers)]

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def _worker(self, n: int) -> None:
        while True:
            try:
                popped = await self._redis.brpop([QUEUE_KEY], timeout=1)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("ingest worker %d: queue read failed", n)
                await asyncio.sleep(1)
                continue
            if popped is None:
                continue
            await self._process(popped[1])

    async def _process(self, doc_id: str) -> None:
        key = _doc_key(doc_id)
        text = await self._redis.hget(key, "text")
        if text is None:
            return
        await self._redis.hset(key, "status", "processing")
        try:
            summary, _, _ = await self._service.summarize(text, max_sentences=3)
            await self._redis.hset(key, mapping={"status": "done", "summary": summary})
        except Exception as exc:
            log.exception("ingest: failed to process %s", doc_id)
            await self._redis.hset(key, mapping={"status": "failed", "error": str(exc)})
