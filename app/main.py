import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from redis.asyncio import Redis

from .cache import ResponseCache
from .config import settings
from .ingest import IngestPipeline
from .llm import LLMClient
from .schemas import (IngestDocument, IngestRequest, IngestResponse, IngestStatus,
                      RecommendRequest, RecommendResult, StreamIngestResponse, SummarizeRequest,
                      TextResult, TranslateRequest)
from .service import ContentService, LLMResponseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = Redis.from_url(settings.redis_url, decode_responses=True, max_connections=256)
    llm = LLMClient(settings)
    cache = ResponseCache(redis, settings.cache_ttl_seconds, settings.llm_model)
    app.state.redis = redis
    app.state.service = ContentService(llm, cache, settings)
    app.state.ingest = IngestPipeline(redis, app.state.service, settings.ingest_workers)
    app.state.ingest.start()
    yield
    await app.state.ingest.stop()
    await llm.close()
    await redis.aclose()


app = FastAPI(title="LLM Content Service", version="1.0.0", lifespan=lifespan)


def _cache_headers(response: Response, cached: bool, lookup_ms: float) -> None:
    response.headers["X-Cache"] = "HIT" if cached else "MISS"
    response.headers["X-Cache-Lookup-Ms"] = f"{lookup_ms:.3f}"


@app.get("/health")
async def health():
    await app.state.redis.ping()
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return {task: b.stats for task, b in app.state.service.batchers.items()}


@app.post("/summarize", response_model=TextResult)
async def summarize(req: SummarizeRequest, response: Response):
    result, cached, ms = await app.state.service.summarize(req.text, req.max_sentences)
    _cache_headers(response, cached, ms)
    return TextResult(result=result, cached=cached)


@app.post("/translate", response_model=TextResult)
async def translate(req: TranslateRequest, response: Response):
    result, cached, ms = await app.state.service.translate(req.text, req.target_language)
    _cache_headers(response, cached, ms)
    return TextResult(result=result, cached=cached)


@app.post("/recommend", response_model=RecommendResult)
async def recommend(req: RecommendRequest, response: Response):
    candidates = [c.model_dump() for c in req.candidates]
    try:
        recs, cached, ms = await app.state.service.recommend(req.content, candidates, req.top_k)
    except LLMResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    _cache_headers(response, cached, ms)
    return RecommendResult(recommendations=recs, cached=cached)


@app.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(req: IngestRequest):
    """Batch ingestion: preprocess a list of documents and queue them for summarization."""
    acks = await app.state.ingest.enqueue([d.model_dump() for d in req.documents])
    return IngestResponse(queued=len(acks), documents=acks)


STREAM_CHUNK = 100


@app.post("/ingest/stream", response_model=StreamIngestResponse, status_code=202)
async def ingest_stream(request: Request):
    """Streaming ingestion: NDJSON body, one {"id"?, "text"} object per line.

    Documents are queued in chunks while the body is still arriving, so workers start
    processing before the upload finishes.
    """
    acks, rejected, chunk = [], [], []
    buffer, line_no = b"", 0

    async def flush():
        if chunk:
            acks.extend(await app.state.ingest.enqueue(chunk))
            chunk.clear()

    def parse(line: bytes):
        nonlocal line_no
        line_no += 1
        if not line.strip():
            return
        try:
            chunk.append(IngestDocument.model_validate(json.loads(line)).model_dump())
        except (json.JSONDecodeError, ValidationError) as exc:
            rejected.append({"line": line_no, "error": str(exc).splitlines()[0]})

    async for data in request.stream():
        buffer += data
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            parse(line)
        if len(chunk) >= STREAM_CHUNK:
            await flush()
    parse(buffer)
    await flush()
    return StreamIngestResponse(queued=len(acks), rejected=rejected, documents=acks)


@app.get("/ingest/{doc_id}", response_model=IngestStatus)
async def ingest_status(doc_id: str):
    record = await app.state.ingest.get(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return record
