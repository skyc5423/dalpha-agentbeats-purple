"""Document / research specialist.

Operates only on in-context material:
- ``request.context`` strings
- ``request.attachments[*].text``

It never imports outbound-HTTP libraries; URLs in the prompt are noted but
never fetched. When an :class:`LLMClient` is injected, the specialist asks
the model to extract a single answer candidate from the retrieved spans.
"""

from __future__ import annotations

import json
import re

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import StepRecord
from ..state import WorkingState
from ..tools import chunk_text, extract_json, search_chunks

_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _gather_sources(state: WorkingState) -> list[str]:
    sources: list[str] = []
    prompt = state.request.prompt or ""
    if any(label in prompt.lower() for label in ("context:", "passage:", "document:")):
        sources.append(prompt)
    sources.extend(state.request.context)
    for att in state.request.attachments:
        if att.text:
            sources.append(att.text)
    return sources


class DocResearchSpecialist:
    name = "doc_research"
    capability = "doc_research"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, state: WorkingState) -> StepRecord:
        sources = _gather_sources(state)

        urls = list(_URL.findall(state.request.prompt))
        for src in sources:
            urls.extend(_URL.findall(src))

        chunks: list[str] = []
        for src in sources:
            chunks.extend(chunk_text(src))

        spans = search_chunks(chunks, state.request.prompt, limit=6)
        if not spans and chunks:
            spans = chunks[:3]

        answer_candidate: str | None = None
        used_llm = False
        if self._llm is not None and spans:
            answer_candidate = await self._llm_extract(state, spans)
            used_llm = answer_candidate is not None

        state.set_note("research_spans", tuple(spans))
        if answer_candidate:
            state.set_note("answer_candidate", answer_candidate)
        if urls:
            state.set_note("urls_detected", tuple(urls))

        if not sources:
            summary = "no fetch: no context or attachments to read"
        elif used_llm and answer_candidate:
            summary = (
                f"extracted {len(spans)} span(s); candidate: {answer_candidate!r}"
            )
        else:
            summary = f"extracted {len(spans)} span(s) from {len(sources)} source(s)"

        return StepRecord(
            capability=self.capability,
            summary=summary,
            outputs={
                "spans": list(spans),
                "answer_candidate": answer_candidate or "",
                "urls_detected": list(urls),
                "fetched": False,
            },
        )

    async def _llm_extract(
        self, state: WorkingState, spans: list[str]
    ) -> str | None:
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
            "user_prompt": state.request.prompt,
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
                tag="doc_research",
                max_tokens=400,
            )
        except Exception:
            return None
        data = extract_json(text)
        if not isinstance(data, dict):
            return None
        candidate = data.get("answer_candidate")
        if not isinstance(candidate, str):
            return None
        candidate = candidate.strip()
        return candidate or None
