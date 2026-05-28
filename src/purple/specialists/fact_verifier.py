"""Fact verifier specialist.

Two-mode behaviour:

* Heuristic baseline (always computed): token overlap between the prompt and
  the union of context, attachments, and research spans.
* LLM verification (when an :class:`LLMClient` is injected and an
  ``answer_candidate`` is available): the model returns a JSON object with
  ``confidence``, ``verdict``, and ``concerns``. Its confidence supersedes
  the heuristic when valid.
"""

from __future__ import annotations

import json
import re

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import StepRecord
from ..state import WorkingState
from ..tools import extract_json

_WORD = re.compile(r"[A-Za-z0-9]+")
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
    return {
        w.lower()
        for w in _WORD.findall(text)
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    }


class FactVerifierSpecialist:
    name = "fact_verifier"
    capability = "fact_verify"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, state: WorkingState) -> StepRecord:
        prompt_tokens = _content_tokens(state.request.prompt or "")

        evidence_parts: list[str] = []
        evidence_parts.extend(state.request.context)
        for att in state.request.attachments:
            if att.text:
                evidence_parts.append(att.text)
        spans = state.get_note("research_spans", ())
        evidence_parts.extend(spans)
        evidence_text = "\n".join(evidence_parts)
        evidence_tokens = _content_tokens(evidence_text)

        if not evidence_tokens or not prompt_tokens:
            heuristic_confidence = 0.0
            overlap: set[str] = set()
        else:
            overlap = prompt_tokens & evidence_tokens
            heuristic_confidence = round(
                len(overlap) / max(1, len(prompt_tokens)), 4
            )
            if heuristic_confidence > 1.0:
                heuristic_confidence = 1.0

        candidate = state.get_note("answer_candidate", "")
        verdict = "uncertain"
        concerns: list[str] = []
        confidence = heuristic_confidence
        source = "heuristic"

        if candidate and any(str(span).startswith("Calculation:") and str(candidate) in str(span) for span in spans):
            confidence = 1.0
            verdict = "supported"
            source = "calculator"
        elif self._llm is not None and candidate:
            llm_result = await self._llm_verify(state, str(candidate), list(spans))
            if llm_result is not None:
                confidence = llm_result["confidence"]
                verdict = llm_result["verdict"]
                concerns = llm_result["concerns"]
                source = "llm"

        evidence = {
            "confidence": confidence,
            "verdict": verdict,
            "concerns": list(concerns),
            "overlap_tokens": sorted(overlap),
            "evidence_token_count": len(evidence_tokens),
            "prompt_token_count": len(prompt_tokens),
            "source": source,
            "heuristic_confidence": heuristic_confidence,
        }
        state.set_note("evidence", evidence)

        summary = (
            f"confidence={confidence:.2f} ({source}, verdict={verdict})"
        )
        return StepRecord(
            capability=self.capability,
            summary=summary,
            outputs=evidence,
        )

    async def _llm_verify(
        self,
        state: WorkingState,
        candidate: str,
        spans: list[str],
    ) -> dict | None:
        assert self._llm is not None
        system_text = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                load_prompt("fact_verifier"),
                load_skill("verification"),
            )
            if chunk
        )
        user_payload = {
            "user_prompt": state.request.prompt,
            "answer_candidate": candidate,
            "evidence_spans": spans,
        }
        user_text = (
            "Inputs (JSON):\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\nReturn the verification JSON object."
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", system_text),
                    ChatMessage("user", user_text),
                ],
                tag="fact_verifier",
                max_tokens=300,
            )
        except Exception:
            return None
        data = extract_json(text)
        if not isinstance(data, dict):
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        confidence = max(0.0, min(1.0, confidence))
        verdict = data.get("verdict", "uncertain")
        if verdict not in {"supported", "unsupported", "uncertain"}:
            verdict = "uncertain"
        raw_concerns = data.get("concerns") or []
        concerns: list[str] = []
        if isinstance(raw_concerns, list):
            for item in raw_concerns:
                if isinstance(item, str) and item.strip():
                    concerns.append(item.strip())
        return {
            "confidence": round(confidence, 4),
            "verdict": verdict,
            "concerns": concerns,
        }
