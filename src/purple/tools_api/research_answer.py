"""``research_answer`` tool — LLM-heavy open-web research primitive.

This tool intentionally stays domain-agnostic: it does not know about GitHub,
academic lineages, esports, or benchmark task shapes. It delegates semantic
research to a configured web-search-capable LLM and returns the answer plus the
model's cited/evidence text as transcript spans. Deterministic tools remain
available for fetching, quote checking, and arithmetic, but this primitive is
for the "mostly LLM" path.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..prompts import load_prompt
from ..runtime.tool import ToolContext, ToolResult
from ..tools import WebAnswerer, extract_urls, openai_web_search_from_env


_FAILURE_MARKERS = (
    "couldn't locate",
    "could not locate",
    "couldn't find",
    "could not find",
    "i don't have",
    "not enough information",
    "insufficient information",
    "would need access",
)


class ResearchAnswerTool:
    name = "research_answer"
    description = (
        "Use a web-search-capable LLM to research the open web and draft a "
        "source-cited answer. This is generic and domain-agnostic; use it for "
        "open-web questions before adding bespoke parsing logic."
    )
    arg_schema: Mapping[str, str] = {
        "question": "optional question override; defaults to the user prompt",
        "max_tokens": "optional output token cap (default 1800)",
    }

    def __init__(self, *, web_answerer: WebAnswerer | None = None, use_env: bool = True) -> None:
        self._web_answerer = web_answerer if web_answerer is not None else (
            openai_web_search_from_env() if use_env else None
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            question = ctx.request.prompt or ""
        question = question.strip()
        if not question:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="research_answer requires a question",
                observation="missing question",
                outputs={"answer_candidate": "", "spans": []},
                error="missing question",
            )
        try:
            max_tokens = max(400, min(4000, int(args.get("max_tokens", 1800))))
        except (TypeError, ValueError):
            max_tokens = 1800
        if self._web_answerer is None:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="research_answer has no configured web-search LLM",
                observation="OPENAI_API_KEY/LLM_API_KEY not configured for web search",
                outputs={"answer_candidate": "", "spans": []},
                error="no-web-answerer",
            )

        system = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                "You are a general-purpose open-web research agent. Solve the task by searching public sources. "
                "Do not use benchmark IDs, hidden ground truth, task-specific lookup tables, or private data. "
                "Prefer primary/official sources. Include source URLs next to critical claims. "
                "If a numeric answer requires calculation, show the values used and the arithmetic in prose; do not invent missing values.",
            )
            if chunk
        )
        requirements = ctx.notes.get("requirements") if hasattr(ctx.notes, "get") else {}
        req_text = ""
        if isinstance(requirements, Mapping) and requirements:
            req_text = "\n\nRequired success criteria (from first-pass task analysis):\n" + json.dumps(
                requirements, ensure_ascii=False, indent=2
            )
        answer = await self._web_answerer.answer(
            prompt=(
                "Research and answer this task using public web sources. Return a concise final answer with URLs for critical claims. "
                "Explicitly cover every non-optional required output; if one cannot be supported, say what is missing. "
                "For multi-clue entity questions, first decompose the clues and build a requirement coverage table: "
                "for each clue list the candidate entity, source URL, date if relevant, and a short quoted/visible evidence phrase. "
                "Do candidate-independent discovery before naming a final entity: search the rarest clue/source first, then chain follow-up searches from that source's institution/domain/date. "
                "For paired date clues, prefer this order: find the first dated article/source URL, compute the derived date (for example seven days later), then search that derived date plus the next clue. "
                "Reject mixed-entity chains; all satisfied clues must point to the same entity/institution/domain or be explicitly cross-verified. "
                "Make date arithmetic explicit when the task says things like 'fourth Sunday' or 'seven days after'.\n\n"
                f"Task:\n{question}"
                + req_text
            ),
            system=system,
            max_tokens=max_tokens,
        )
        answer = (answer or "").strip()
        urls_detected = extract_urls(answer, limit=8) if answer else []
        looks_failed = _looks_like_failed(answer)
        return ToolResult(
            tool_call_id="",
            ok=bool(answer),
            summary=(
                "research_answer produced source-cited draft"
                if answer and not looks_failed
                else "research_answer did not find a confident answer"
            ),
            observation=answer[:800] if answer else "(empty)",
            outputs={
                "answer_candidate": answer if answer and not looks_failed else "",
                "spans": [answer[:6000]] if answer else [],
                "source_urls": urls_detected,
                "urls_detected": urls_detected,
                "source": "llm_web_research",
                # Do not mark LLM web research as self-sufficient. The answer may
                # be semantically plausible but still unsupported or scoped
                # wrong, so the normal sufficiency/verifier loop must inspect it.
                "sufficient_alone": False,
            },
            error="" if answer else "empty",
        )


def _looks_like_failed(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _FAILURE_MARKERS)


__all__ = ["ResearchAnswerTool"]
