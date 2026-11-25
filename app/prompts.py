"""System prompts and prompt construction.

Prompt caching: system prompts are static per task and kept byte-identical across
calls. Every per-request parameter (target language, length limit, ...) goes into
the user message, so the system prefix never changes. The prebuilt message dicts are
memoized and reused on every call, which avoids rebuilding them and lets
OpenAI-compatible providers that do automatic prefix caching reuse the cached prefix.
"""

from functools import lru_cache

SYSTEM_PROMPTS = {
    "summarize": (
        "You are a precise summarization engine. Summarize the content faithfully and "
        "concisely, keeping key facts, names and numbers. Respect the requested maximum "
        "length. Output only the summary."
    ),
    "translate": (
        "You are a professional translator. Translate the content into the requested "
        "target language, preserving meaning, tone and formatting. Output only the "
        "translation."
    ),
    "recommend": (
        "You are a content recommendation engine. Given a source document and a list of "
        "candidate items, rank the candidates by how relevant they are to a reader of the "
        "source document. Respond with JSON only, in the form "
        '{"recommendations": [{"id": "<candidate id>", "reason": "<one short sentence>"}]}, '
        "most relevant first, using only ids from the candidate list."
    ),
}

BATCH_INSTRUCTIONS = (
    "\n\nThe user message contains several independent items, each starting with a line "
    "'### Item <n>'. Process every item independently using the rules above and respond "
    "with JSON only: an array of strings with exactly one result per item, in order."
)


@lru_cache(maxsize=None)
def system_message(task: str, batched: bool = False) -> dict:
    content = SYSTEM_PROMPTS[task] + (BATCH_INSTRUCTIONS if batched else "")
    return {"role": "system", "content": content}


def summarize_prompt(text: str, max_sentences: int) -> str:
    return f"Maximum length: {max_sentences} sentences.\n\nContent:\n{text}"


def translate_prompt(text: str, target_language: str) -> str:
    return f"Target language: {target_language}\n\nContent:\n{text}"


def recommend_prompt(content: str, candidates: list[dict], top_k: int) -> str:
    lines = "\n".join(f"- id={c['id']} | {c['title']}" + (f" | {c['description']}" if c.get("description") else "")
                      for c in candidates)
    return f"Return at most {top_k} recommendations.\n\nSource document:\n{content}\n\nCandidates:\n{lines}"


def batch_prompt(user_contents: list[str]) -> str:
    return "\n\n".join(f"### Item {i}\n{c}" for i, c in enumerate(user_contents, 1))
