"""Text chunking, search, and merging helpers."""

from __future__ import annotations

import re
from typing import Iterable

_WORD = re.compile(r"[A-Za-z0-9$%.]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "to",
        "in",
        "on",
        "at",
        "and",
        "or",
        "for",
        "by",
        "with",
        "be",
        "this",
        "that",
        "it",
        "as",
        "from",
        "what",
        "which",
        "who",
        "how",
        "why",
        "where",
        "when",
    }
)


def _content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _WORD.findall(text):
        tok = raw.lower()
        if len(tok) < 3 or tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


def chunk_text(text: str, *, max_chars: int = 480) -> list[str]:
    """Split text into roughly sentence-aware chunks bounded by ``max_chars``."""
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
            elif len(current) + len(sentence) + 1 <= max_chars:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
    return chunks


def search_chunks(
    chunks: Iterable[str], query: str, *, limit: int = 5
) -> list[str]:
    """Return up to ``limit`` chunks ranked by query keyword overlap."""
    chunks_list = list(chunks)
    query_tokens = _content_tokens(query)
    if not query_tokens or not chunks_list:
        return chunks_list[:limit]
    scored: list[tuple[int, int, str]] = []
    for i, chunk in enumerate(chunks_list):
        overlap = len(query_tokens & _content_tokens(chunk))
        if overlap:
            scored.append((overlap, i, chunk))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:limit]]


def merge_evidence(spans: Iterable[str]) -> str:
    """Deduplicate while preserving order, then join as a single block."""
    seen: list[str] = []
    for span in spans:
        text = (span or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "\n\n".join(seen)
