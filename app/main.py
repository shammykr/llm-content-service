from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from redis.asyncio import Redis

from .cache import ResponseCache
from .config import settings
from .llm import LLMClient
from .schemas import (RecommendRequest, RecommendResult, SummarizeRequest, TextResult,
                      TranslateRequest)
from .service import ContentService, LLMResponseError


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = Redis.from_url(settings.redis_url, decode_responses=True, max_connections=256)
    llm = LLMClient(settings)
    cache = ResponseCache(redis, settings.cache_ttl_seconds, settings.llm_model)
    app.state.redis = redis
    app.state.service = ContentService(llm, cache, settings)
    yield
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
