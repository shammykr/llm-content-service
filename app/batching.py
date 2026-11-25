"""Token batching: merge small same-task requests into a single LLM call.

Requests are collected for up to `window_ms`. The batch is flushed early when it
reaches `max_items` or `max_tokens`. The merged prompt numbers each item, and the
model returns a JSON array with one result per item. If the reply can't be parsed,
each item falls back to its own call, so callers always get a result.
"""

import asyncio
import json
import re
from dataclasses import dataclass

from .llm import LLMClient
from .prompts import batch_prompt


@dataclass
class _Pending:
    user_content: str
    tokens: int
    future: asyncio.Future


class TokenBatcher:
    def __init__(self, llm: LLMClient, task: str, window_ms: float, max_items: int, max_tokens: int):
        self._llm = llm
        self._task = task
        self._window = window_ms / 1000
        self._max_items = max_items
        self._max_tokens = max_tokens
        self._pending: list[_Pending] = []
        self._pending_tokens = 0
        self._timer: asyncio.TimerHandle | None = None
        self._tasks: set[asyncio.Task] = set()
        self.stats = {"llm_calls": 0, "items": 0}

    async def submit(self, user_content: str, tokens: int) -> str:
        loop = asyncio.get_running_loop()
        if self._pending and self._pending_tokens + tokens > self._max_tokens:
            self._flush()
        item = _Pending(user_content, tokens, loop.create_future())
        self._pending.append(item)
        self._pending_tokens += tokens
        if len(self._pending) >= self._max_items or self._pending_tokens >= self._max_tokens:
            self._flush()
        elif self._timer is None:
            self._timer = loop.call_later(self._window, self._flush)
        return await item.future

    def _flush(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        items, self._pending, self._pending_tokens = self._pending, [], 0
        if items:
            task = asyncio.create_task(self._run(items))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, items: list[_Pending]) -> None:
        self.stats["items"] += len(items)
        try:
            if len(items) == 1:
                self.stats["llm_calls"] += 1
                results = [await self._llm.complete(self._task, items[0].user_content)]
            else:
                self.stats["llm_calls"] += 1
                raw = await self._llm.complete(
                    self._task, batch_prompt([i.user_content for i in items]), batched=True)
                results = _parse_array(raw, len(items))
                if results is None:
                    self.stats["llm_calls"] += len(items)
                    results = await asyncio.gather(
                        *(self._llm.complete(self._task, i.user_content) for i in items))
            for item, result in zip(items, results):
                if not item.future.done():
                    item.future.set_result(result)
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)


def _parse_array(raw: str, expected: int) -> list[str] | None:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and len(data) == expected:
        return [str(x).strip() for x in data]
    return None
