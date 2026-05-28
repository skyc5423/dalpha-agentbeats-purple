"""``verify_claim`` tool — mid-loop verification using the LLM when available."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..runtime.tool import ToolContext, ToolResult
from ..tools import extract_json


class VerifyClaimTool:
    name = "verify_claim"
    description = (
        "Verify an answer candidate against the evidence already on the "
        "transcript. The finalizer runs a verifier regardless; the controller "
        "may invoke this mid-loop to redirect."
    )
    arg_schema: Mapping[str, str] = {
        "claim": "the answer candidate to verify",
        "spans": "optional list of evidence spans (defaults to transcript)",
    }

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        claim = args.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            note_candidate = ctx.notes.get("answer_candidate") if hasattr(ctx.notes, "get") else ""
            claim = str(note_candidate or "").strip()
        if not claim:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="verify_claim requires a non-empty claim",
                observation="no claim provided and none on transcript",
                outputs={"confidence": 0.0, "verdict": "uncertain", "concerns": []},
                error="missing claim",
            )

        spans: list[str] = []
        raw_spans = args.get("spans")
        if isinstance(raw_spans, list):
            spans = [s for s in raw_spans if isinstance(s, str) and s.strip()]
        if not spans:
            note_spans = ctx.notes.get("spans") if hasattr(ctx.notes, "get") else None
            if isinstance(note_spans, list):
                spans = [s for s in note_spans if isinstance(s, str) and s.strip()]

        if self._llm is None:
            return ToolResult(
                tool_call_id="",
                ok=True,
                summary="no llm; verify_claim is inert",
                observation="no llm configured",
                outputs={
                    "confidence": 0.0,
                    "verdict": "uncertain",
                    "concerns": ["LLM not configured"],
                    "source": "no-llm",
                },
            )

        result = await self._llm_verify(ctx.request.prompt or "", claim, spans)
        if result is None:
            return ToolResult(
                tool_call_id="",
                ok=True,
                summary="verify_claim: malformed LLM output",
                observation="LLM returned non-JSON; defaulting to uncertain",
                outputs={
                    "confidence": 0.0,
                    "verdict": "uncertain",
                    "concerns": ["malformed verifier output"],
                    "source": "llm-malformed",
                },
            )
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=(
                f"verify_claim confidence={result['confidence']:.2f} "
                f"({result['verdict']})"
            ),
            observation=f"verdict={result['verdict']} confidence={result['confidence']:.2f}",
            outputs={
                "confidence": result["confidence"],
                "verdict": result["verdict"],
                "concerns": result["concerns"],
                "source": "llm",
            },
        )

    async def _llm_verify(self, prompt: str, claim: str, spans: list[str]) -> dict | None:
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
            "user_prompt": prompt,
            "answer_candidate": claim,
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
                tag="verify_claim",
                max_tokens=300,
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


__all__ = ["VerifyClaimTool"]
