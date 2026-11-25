import hashlib
import json
import time

from redis.asyncio import Redis


class ResponseCache:
    """Redis-backed cache for LLM results, keyed by a SHA-256 hash of the request."""

    def __init__(self, redis: Redis, ttl_seconds: int, model: str):
        self._redis = redis
        self._ttl = ttl_seconds
        self._model = model

    def key(self, task: str, payload: dict) -> str:
        blob = json.dumps({"task": task, "model": self._model, "input": payload},
                          sort_keys=True, ensure_ascii=False)
        return f"llmcache:{task}:{hashlib.sha256(blob.encode()).hexdigest()}"

    async def get(self, key: str) -> tuple[dict | None, float]:
        """Return (cached value or None, lookup time in ms)."""
        start = time.perf_counter()
        raw = await self._redis.get(key)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return (json.loads(raw) if raw is not None else None), elapsed_ms

    async def set(self, key: str, value: dict) -> None:
        await self._redis.set(key, json.dumps(value, ensure_ascii=False), ex=self._ttl)
