"""Deterministic, content-feature based capability profiler.

The profiler must never branch on external dataset names; it only looks at
content features of the incoming request.
"""

from __future__ import annotations

import re
from types import MappingProxyType

from .schema import CapabilityProfile, TaskRequest

CAPABILITIES: tuple[str, ...] = (
    "planning",
    "calculator",
    "web_research",
    "shell_code",
    "doc_research",
    "policy",
    "fact_verify",
    "composition",
)

_SHELL_TOKENS = (
    "bash",
    "shell",
    "python -c",
    "sh -c",
    "$ ",
    "./",
    "/bin/",
    "/usr/",
)

_RESEARCH_TOKENS = (
    "summarize",
    "according to",
    "find",
    "based on",
    "extract",
    "lookup",
    "context:",
    "question:",
    "passage:",
    "document:",
)

_CALC_TOKENS = (
    "calculate",
    "compute",
    "arithmetic",
    "sum",
    "difference",
    "product",
    "quotient",
    "percentage",
    "percent",
    "rounded",
)

_WEB_TOKENS = (
    "academic",
    "advisor",
    "doctoral",
    "lineage",
    "identify",
    "find out",
    "find ",
    "trace",
    "lookup",
    "official",
    "repository",
    "github",
    "profile",
    "source",
    "sources",
    "url",
    "urls",
    "cite",
    "evidence",
    "world championship",
    "patent",
    "spotify",
    "overleaf",
)

_CODE_FENCE = re.compile(r"```")
_URL = re.compile(r"https?://", re.IGNORECASE)
_ARITHMETIC_EXPR = re.compile(r"\d\s*[-+*/%]\s*\d")

_SELECT_THRESHOLD = 0.5


def _count_token_hits(text: str, tokens: tuple[str, ...]) -> int:
    haystack = text.lower()
    return sum(1 for tok in tokens if tok in haystack)


class CapabilityProfiler:
    """Deterministic feature-based capability scorer."""

    def profile(self, request: TaskRequest) -> CapabilityProfile:
        prompt = request.prompt or ""
        context_blob = "\n".join(request.context)
        full_text = f"{prompt}\n{context_blob}"

        shell_hits = _count_token_hits(full_text, _SHELL_TOKENS)
        code_fence_count = len(_CODE_FENCE.findall(full_text))
        shell_score = min(1.0, 0.4 * shell_hits + 0.3 * code_fence_count)

        calc_hits = _count_token_hits(full_text, _CALC_TOKENS)
        calc_expr = 1 if _ARITHMETIC_EXPR.search(full_text) else 0
        calc_score = min(1.0, 0.35 * calc_hits + 0.55 * calc_expr)

        web_hits = _count_token_hits(full_text, _WEB_TOKENS)
        url_count = len(_URL.findall(full_text))
        web_score = min(1.0, 0.25 * web_hits + 0.35 * url_count)

        research_hits = _count_token_hits(full_text, _RESEARCH_TOKENS)
        has_attachments = 1 if request.attachments else 0
        has_context = 1 if request.context else 0
        doc_score = min(
            1.0,
            0.25 * research_hits
            + 0.3 * url_count
            + 0.4 * has_attachments
            + 0.5 * has_context,
        )

        scores = {
            "planning": 1.0,
            "calculator": calc_score,
            "web_research": web_score,
            "shell_code": shell_score,
            "doc_research": doc_score,
            "policy": 0.6,
            "fact_verify": 0.5,
            "composition": 1.0,
        }

        selected = tuple(
            cap for cap in CAPABILITIES if scores[cap] >= _SELECT_THRESHOLD
        )

        return CapabilityProfile(
            scores=MappingProxyType(dict(scores)),
            selected=selected,
        )
