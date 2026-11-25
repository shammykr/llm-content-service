"""Lightweight NLP preprocessing: normalization, tokenization, language detection."""

import re
import unicodedata

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic language detection

_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text) if s]


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:  # empty / non-linguistic input
        return "unknown"


def preprocess(text: str) -> dict:
    clean = normalize(text)
    return {
        "text": clean,
        "language": detect_language(clean),
        "token_count": len(tokenize(clean)),
        "sentence_count": len(split_sentences(clean)),
    }
