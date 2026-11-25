"""Instant-response, OpenAI-compatible mock backend used for benchmarking.

Run: uvicorn bench.mock_llm:app --port 9000
Then point the service at it with LLM_BASE_URL=http://localhost:9000/v1
"""

import json
import re
import time

from fastapi import FastAPI, Request

app = FastAPI()
calls = {"total": 0, "batched_items": 0}


def _content(block: str) -> str:
    return block.rsplit("Content:\n", 1)[-1].strip()[:80]


def _answer(system: str, user: str) -> str:
    items = re.split(r"^### Item \d+\n", user, flags=re.MULTILINE)[1:]
    if items:  # batched request: one result per item, as a JSON array
        calls["batched_items"] += len(items)
        return json.dumps([f"[mock] {_content(it)}" for it in items])
    if "recommendation engine" in system:
        ids = re.findall(r"^- id=([^\s|]+)", user, flags=re.MULTILINE)
        top_k = int(m.group(1)) if (m := re.search(r"at most (\d+)", user)) else len(ids)
        return json.dumps({"recommendations": [{"id": i, "reason": "mock relevance"} for i in ids[:top_k]]})
    return f"[mock] {_content(user)}"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    calls["total"] += 1
    msgs = body["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    return {
        "id": f"mock-{calls['total']}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": _answer(system, user)}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/stats")
async def stats():
    return calls
