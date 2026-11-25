import asyncio

from openai import AsyncOpenAI

from .config import Settings
from .prompts import system_message


class LLMClient:
    """Async wrapper around any OpenAI-compatible chat completions API."""

    def __init__(self, settings: Settings):
        self.model = settings.llm_model
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60.0,
            max_retries=2,
        )
        # Bound in-flight upstream calls so bursts don't overrun provider rate limits.
        self._sem = asyncio.Semaphore(settings.llm_max_concurrency)

    async def complete(self, task: str, user_content: str, batched: bool = False) -> str:
        messages = [system_message(task, batched), {"role": "user", "content": user_content}]
        async with self._sem:
            resp = await self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2
            )
        return (resp.choices[0].message.content or "").strip()

    async def close(self) -> None:
        await self._client.close()
