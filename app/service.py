import json
import re

from .cache import ResponseCache
from .llm import LLMClient
from .prompts import recommend_prompt, summarize_prompt, translate_prompt


class LLMResponseError(Exception):
    pass


class ContentService:
    """Cache-first content tasks. Each method returns (result, cached, cache_lookup_ms)."""

    def __init__(self, llm: LLMClient, cache: ResponseCache):
        self.llm = llm
        self.cache = cache

    async def _cached(self, task: str, payload: dict, compute):
        key = self.cache.key(task, payload)
        hit, lookup_ms = await self.cache.get(key)
        if hit is not None:
            return hit["value"], True, lookup_ms
        value = await compute()
        await self.cache.set(key, {"value": value})
        return value, False, lookup_ms

    async def _run_text(self, task: str, user_content: str) -> str:
        return await self.llm.complete(task, user_content)

    async def summarize(self, text: str, max_sentences: int):
        payload = {"text": text, "max_sentences": max_sentences}
        return await self._cached("summarize", payload,
                                  lambda: self._run_text("summarize", summarize_prompt(text, max_sentences)))

    async def translate(self, text: str, target_language: str):
        payload = {"text": text, "target_language": target_language.strip().lower()}
        return await self._cached("translate", payload,
                                  lambda: self._run_text("translate", translate_prompt(text, target_language)))

    async def recommend(self, content: str, candidates: list[dict], top_k: int):
        payload = {"content": content, "candidates": candidates, "top_k": top_k}

        async def compute():
            raw = await self.llm.complete("recommend", recommend_prompt(content, candidates, top_k))
            return _parse_recommendations(raw, {c["id"] for c in candidates}, top_k)

        return await self._cached("recommend", payload, compute)


def _parse_recommendations(raw: str, valid_ids: set[str], top_k: int) -> list[dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        items = json.loads(match.group(0))["recommendations"] if match else None
    except (json.JSONDecodeError, KeyError, TypeError):
        items = None
    if not isinstance(items, list):
        raise LLMResponseError("LLM returned malformed recommendation JSON")
    out, seen = [], set()
    for item in items:
        if isinstance(item, dict) and item.get("id") in valid_ids and item["id"] not in seen:
            seen.add(item["id"])
            out.append({"id": item["id"], "reason": str(item.get("reason", ""))})
    return out[:top_k]
