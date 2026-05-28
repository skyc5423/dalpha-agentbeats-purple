"""``extract_answer`` tool — LLM-driven extraction over already-found spans."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..runtime.tool import ToolContext, ToolResult
from ..tools import chunk_text, extract_json, search_chunks


class ExtractAnswerTool:
    name = "extract_answer"
    description = (
        "Extract one verbatim answer from spans surfaced by search_docs "
        "(uses the injected LLM when available)."
    )
    arg_schema: Mapping[str, str] = {
        "question": "optional override; defaults to the user prompt",
        "spans": "optional explicit list of strings to extract from",
    }

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            question = ctx.request.prompt or ""

        spans: list[str] = []
        raw_spans = args.get("spans")
        if isinstance(raw_spans, list):
            spans = [s for s in raw_spans if isinstance(s, str) and s.strip()]
        if not spans:
            note_spans = ctx.notes.get("spans") if hasattr(ctx.notes, "get") else None
            if isinstance(note_spans, list):
                spans = [s for s in note_spans if isinstance(s, str) and s.strip()]
        if not spans:
            sources: list[str] = list(ctx.request.context)
            for att in ctx.request.attachments:
                if att.text:
                    sources.append(att.text)
            chunks: list[str] = []
            for src in sources:
                chunks.extend(chunk_text(src))
            spans = search_chunks(chunks, question, limit=6)

        if self._llm is None:
            return ToolResult(
                tool_call_id="",
                ok=True,
                summary=f"no llm; surfaced {len(spans)} span(s)",
                observation="LLM not configured; returning spans only",
                outputs={
                    "spans": list(spans),
                    "answer_candidate": "",
                    "source": "no-llm",
                },
            )

        candidate = await self._llm_extract(question, spans)
        if candidate:
            summary = (
                f"extracted {len(spans)} span(s); candidate: {candidate!r}"
            )
        else:
            summary = f"no candidate; surfaced {len(spans)} span(s)"
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=summary,
            observation=candidate or "(no candidate)",
            outputs={
                "spans": list(spans),
                "answer_candidate": candidate,
                "source": "llm" if candidate else "no-candidate",
            },
        )

    async def _llm_extract(self, question: str, spans: list[str]) -> str:
        assert self._llm is not None
        system_text = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                load_prompt("doc_research"),
                load_skill("extraction"),
            )
            if chunk
        )
        user_payload = {
            "user_prompt": question,
            "spans": list(spans),
        }
        user_text = (
            "Inputs (JSON):\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\nReturn the extraction JSON object."
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", system_text),
                    ChatMessage("user", user_text),
                ],
                tag="extract_answer",
                max_tokens=400,
            )
        except Exception:
            return ""
        data = extract_json(text or "")
        if not isinstance(data, dict):
            return ""
        candidate = data.get("answer_candidate")
        if not isinstance(candidate, str):
            return ""
        return candidate.strip()


__all__ = ["ExtractAnswerTool"]
