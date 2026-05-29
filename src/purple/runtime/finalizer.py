"""Verifier + composer + rationale wrap-up.

The finalizer runs after the controller loop completes (or is short-circuited
by preflight). It always runs the verifier and the composer and appends the
synthetic ``verify_claim`` and ``compose`` turns to the transcript so the
result's :attr:`steps` reflects the entire flow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import TaskRequest
from ..tools import extract_json
from .tool import ToolCall, ToolResult
from .transcript import Transcript


_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "at", "and", "or", "for", "by", "with", "be", "this", "that", "it",
        "as", "from", "what", "which", "who", "how", "why", "where", "when",
    }
)


def _relevant_excerpt(text: str, query: str, *, limit: int = 3600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return text[:limit].rstrip()
    best_start = 0
    best_score = -1
    window = min(limit, 2200)
    lower = text.lower()
    starts = {0}
    for tok in query_tokens:
        idx = lower.find(tok)
        while idx != -1 and len(starts) < 80:
            starts.add(max(0, idx - window // 2))
            idx = lower.find(tok, idx + len(tok))
    for start in starts:
        excerpt = text[start : start + window]
        score = len(_content_tokens(excerpt) & query_tokens)
        if score > best_score:
            best_score = score
            best_start = start
    excerpt = text[best_start : best_start + window].strip()
    if best_start > 0:
        excerpt = "... " + excerpt
    if best_start + window < len(text):
        excerpt = excerpt.rstrip() + " ..."
    return excerpt[:limit]


def _content_tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD.findall(text)
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    }


def _truncate(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _is_structured_github_commit_metadata(text: str) -> bool:
    lowered = (text or "").lower()
    return all(
        token in lowered
        for token in ("github commit url:", "sha:", "author:", "github author profile:")
    )


def _answer_repeats_rejected_candidate(answer: str, candidate: str) -> bool:
    """Return True when composer merely restates an unsupported candidate."""

    answer_norm = " ".join((answer or "").lower().split())
    candidate_norm = " ".join((candidate or "").lower().split())
    if not answer_norm or not candidate_norm:
        return False
    if len(candidate_norm) >= 12 and candidate_norm in answer_norm:
        return True
    cand_tokens = _content_tokens(candidate_norm)
    ans_tokens = _content_tokens(answer_norm)
    if not cand_tokens:
        return False
    # For short entity-like candidates, repeating most content tokens is enough
    # to treat the answer as the same rejected claim.
    overlap = cand_tokens & ans_tokens
    return len(cand_tokens) <= 5 and len(overlap) / max(1, len(cand_tokens)) >= 0.8


@dataclass(frozen=True)
class FinalizationResult:
    answer: str
    rationale: str
    confidence: float
    verdict: str
    concerns: tuple[str, ...]
    source: str


class Finalizer:
    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(
        self,
        request: TaskRequest,
        transcript: Transcript,
        *,
        flags: tuple[str, ...] = (),
        controller_answer: str = "",
        refusal: str | None = None,
    ) -> FinalizationResult:
        candidate = (controller_answer or "").strip()
        if not candidate:
            note_candidate = transcript.latest_output("answer_candidate")
            if isinstance(note_candidate, str):
                candidate = note_candidate.strip()
        spans = self._evidence_spans(request, transcript, candidate)

        if refusal is None and self._llm is not None and spans and not _is_structured_github_commit_metadata(candidate):
            synthesized = await self._llm_synthesize_candidate(
                request,
                candidate,
                spans,
                transcript.latest_output("requirements") or {},
            )
            if synthesized and synthesized != candidate and not _is_structured_github_commit_metadata(candidate):
                candidate = synthesized
                synth_call = ToolCall(id="final-synthesize", name="synthesize_candidate", args={})
                synth_result = ToolResult(
                    tool_call_id=synth_call.id,
                    ok=True,
                    summary="synthesized candidate from full evidence pool",
                    outputs={
                        "answer_candidate": candidate,
                        "source": "synthesized_from_evidence",
                        "evidence_span_count": len(spans),
                    },
                )
                transcript.append(synth_call, synth_result)
                spans = self._evidence_spans(request, transcript, candidate)

        verification = await self._verify(request, candidate, spans, transcript)
        verify_call = ToolCall(
            id="final-verify",
            name="verify_claim",
            args={"claim": candidate},
        )
        verify_result = ToolResult(
            tool_call_id=verify_call.id,
            ok=True,
            summary=(
                f"confidence={verification['confidence']:.2f} "
                f"({verification['source']}, verdict={verification['verdict']})"
            ),
            outputs={
                "confidence": verification["confidence"],
                "verdict": verification["verdict"],
                "concerns": list(verification["concerns"]),
                "source": verification["source"],
                "heuristic_confidence": verification["heuristic_confidence"],
            },
        )
        transcript.append(verify_call, verify_result)

        if refusal is not None:
            answer = refusal
            source = "policy"
        else:
            answer = ""
            used_llm = False
            if _is_structured_github_commit_metadata(candidate):
                answer = candidate
            elif self._llm is not None and (candidate or spans):
                answer = await self._llm_compose(
                    request,
                    candidate,
                    spans,
                    verification,
                    transcript.latest_output("requirements") or {},
                )
                used_llm = bool(answer)
                if verification["verdict"] == "unsupported" and _answer_repeats_rejected_candidate(answer, candidate):
                    answer = ""
                    used_llm = False
            if not answer:
                answer = self._fallback_answer(candidate, spans, verification["verdict"])
            source = "llm" if used_llm else "fallback"

        compose_call = ToolCall(id="final-compose", name="compose", args={})
        compose_result = ToolResult(
            tool_call_id=compose_call.id,
            ok=True,
            summary=(
                f"composed answer ({source}, confidence={verification['confidence']:.2f})"
            ),
            outputs={
                "answer": answer,
                "source": source,
                "rationale_present": True,
            },
        )
        transcript.append(compose_call, compose_result)

        rationale = self._build_rationale(request, transcript, verification, flags)
        return FinalizationResult(
            answer=answer,
            rationale=rationale,
            confidence=verification["confidence"],
            verdict=verification["verdict"],
            concerns=tuple(verification["concerns"]),
            source=source,
        )

    @staticmethod
    def _evidence_spans(request: TaskRequest, transcript: Transcript, candidate: str) -> list[str]:
        spans = [s for s in transcript.collected("spans") if isinstance(s, str)]
        query_parts = [request.prompt or "", candidate]
        requirements = transcript.latest_output("requirements") or {}
        if requirements:
            try:
                query_parts.append(json.dumps(requirements, ensure_ascii=False))
            except TypeError:
                query_parts.append(str(requirements))
        query = "\n".join(query_parts)
        existing = set(spans)
        for page in transcript.collected("fetched_pages"):
            if not isinstance(page, dict):
                continue
            text = page.get("text")
            url = page.get("url", "")
            if not isinstance(text, str) or not text.strip():
                continue
            excerpt = _relevant_excerpt(text, query, limit=3600)
            span = f"Fetched source {url}: {excerpt}" if url else excerpt
            if span not in existing:
                existing.add(span)
                spans.append(span)
        return spans

    async def _llm_synthesize_candidate(
        self,
        request: TaskRequest,
        current_candidate: str,
        spans: list[str],
        requirements: Any,
    ) -> str:
        assert self._llm is not None
        payload = {
            "user_prompt": request.prompt,
            "requirements": requirements if isinstance(requirements, dict) else {},
            "current_candidate": current_candidate,
            "evidence_spans": list(spans),
        }
        user_text = (
            "Synthesize one complete answer_candidate from the full evidence pool. "
            "Do not choose a single search result title. Use all relevant evidence spans together. "
            "Only include claims supported by the evidence. Return JSON only with key answer_candidate.\n\n"
            "Inputs (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage(
                        "system",
                        "You synthesize verifier-ready answer candidates from collected evidence for a general-purpose agent.",
                    ),
                    ChatMessage("user", user_text),
                ],
                tag="candidate_synthesizer",
                max_tokens=1400,
            )
        except Exception:
            return ""
        data = extract_json(text or "")
        if isinstance(data, dict):
            candidate = data.get("answer_candidate")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return (text or "").strip()

    async def _verify(
        self,
        request: TaskRequest,
        candidate: str,
        spans: list[str],
        transcript: Transcript,
    ) -> dict[str, Any]:
        prompt_tokens = _content_tokens(request.prompt or "")
        evidence_parts: list[str] = []
        evidence_parts.extend(request.context)
        for att in request.attachments:
            if att.text:
                evidence_parts.append(att.text)
        evidence_parts.extend(spans)
        evidence_tokens = _content_tokens("\n".join(evidence_parts))

        if not evidence_tokens or not prompt_tokens:
            heuristic_confidence = 0.0
        else:
            overlap = prompt_tokens & evidence_tokens
            heuristic_confidence = min(
                1.0, round(len(overlap) / max(1, len(prompt_tokens)), 4)
            )

        verdict = "uncertain"
        confidence = heuristic_confidence
        source = "heuristic"
        concerns: list[str] = []

        if candidate and any(
            isinstance(s, str) and s.startswith("Calculation:") and candidate in s
            for s in transcript.collected("spans")
        ):
            confidence = 1.0
            verdict = "supported"
            source = "calculator"
        elif self._llm is not None and candidate:
            llm_result = await self._llm_verify(
                request, candidate, spans, transcript.latest_output("requirements") or {}
            )
            if llm_result is not None:
                confidence = llm_result["confidence"]
                verdict = llm_result["verdict"]
                concerns = llm_result["concerns"]
                source = "llm"

        return {
            "confidence": confidence,
            "verdict": verdict,
            "concerns": concerns,
            "source": source,
            "heuristic_confidence": heuristic_confidence,
        }

    async def _llm_verify(
        self,
        request: TaskRequest,
        candidate: str,
        spans: list[str],
        requirements: Any,
    ) -> dict[str, Any] | None:
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
            "user_prompt": request.prompt,
            "requirements": requirements,
            "answer_candidate": candidate,
            "evidence_spans": list(spans),
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
                max_tokens=800,
            )
        except Exception:
            return None
        data = extract_json(text or "")
        if not isinstance(data, dict):
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        confidence = max(0.0, min(1.0, round(confidence, 4)))
        verdict = data.get("verdict", "uncertain")
        if verdict not in {"supported", "unsupported", "uncertain"}:
            verdict = "uncertain"
        raw_concerns = data.get("concerns") or []
        concerns: list[str] = []
        if isinstance(raw_concerns, list):
            for item in raw_concerns:
                if isinstance(item, str) and item.strip():
                    concerns.append(item.strip())
        return {"confidence": confidence, "verdict": verdict, "concerns": concerns}

    async def _llm_compose(
        self,
        request: TaskRequest,
        candidate: str,
        spans: list[str],
        verification: dict[str, Any],
        requirements: Any,
    ) -> str:
        assert self._llm is not None
        system_text = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                load_prompt("composer"),
                load_skill("composition"),
            )
            if chunk
        )
        verdict = verification["verdict"]
        # Low confidence should not suppress an answer in benchmark settings, but
        # an explicitly unsupported candidate must not be treated as the answer
        # seed. Keep it visible for debugging while forcing the composer to build
        # from evidence + verifier concerns instead of repeating rejected claims.
        user_payload = {
            "user_prompt": request.prompt,
            "requirements": requirements,
            "answer_candidate": candidate if verdict != "unsupported" else "",
            "rejected_candidate": candidate if verdict == "unsupported" else "",
            "confidence": verification["confidence"],
            "verdict": verdict,
            "concerns": list(verification["concerns"]),
            "evidence_spans": list(spans),
            "composition_policy": (
                "Always produce the best benchmark answer from supported evidence. "
                "If rejected_candidate is present, do not use it as a source of truth; "
                "use verifier concerns to identify contradictions, then answer from evidence_spans."
            ),
        }
        user_text = (
            "Inputs (JSON):\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\nWrite the final user-visible answer now."
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", system_text),
                    ChatMessage("user", user_text),
                ],
                tag="composer",
                max_tokens=1200,
            )
        except Exception:
            return ""
        return (text or "").strip()

    @staticmethod
    def _fallback_answer(candidate: str, spans: list[str], verdict: str) -> str:
        candidate = candidate.strip()
        if candidate and verdict != "unsupported":
            return candidate
        if verdict == "unsupported":
            return "Insufficient verified evidence to answer confidently."
        if spans:
            return _truncate(spans[0])
        return "Insufficient information in provided context to answer confidently."

    @staticmethod
    def _build_rationale(
        request: TaskRequest,
        transcript: Transcript,
        verification: dict[str, Any],
        flags: tuple[str, ...],
    ) -> str:
        prompt = (request.prompt or "(no prompt provided)").strip()
        lines = [
            f"Prompt: {_truncate(prompt)}",
            (
                f"Confidence: {verification['confidence']:.2f}"
                f" ({verification['verdict']}, source={verification['source']})"
            ),
        ]
        concerns = verification["concerns"]
        if concerns:
            lines.append("Concerns: " + "; ".join(concerns))
        history = [
            f"- {call.name}: {result.summary}"
            for call, result in transcript.turns
        ]
        if history:
            lines.append("Steps:")
            lines.extend(history)
        if flags:
            lines.append("Flags: " + ", ".join(flags))
        return "\n".join(lines)


__all__ = ["FinalizationResult", "Finalizer"]
