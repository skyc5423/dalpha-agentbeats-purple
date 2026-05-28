"""Answer composer specialist.

With an LLM injected, the composer asks the model to write a concise final
answer from the verified evidence. Without one, it falls back to the
``answer_candidate`` produced by doc_research, or to a short evidence
extract, or to an explicit "insufficient information" sentence. Either way,
the rationale field surfaces the executed pipeline for debugging.
"""

from __future__ import annotations

import json

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import StepRecord
from ..state import WorkingState


def _truncate(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


class AnswerComposerSpecialist:
    name = "composer"
    capability = "composition"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, state: WorkingState) -> StepRecord:
        evidence = state.get_note("evidence", {}) or {}
        confidence = float(evidence.get("confidence", 0.0))
        verdict = evidence.get("verdict", "uncertain")
        concerns = list(evidence.get("concerns", []))
        spans = list(state.get_note("research_spans", ()))
        candidate = str(state.get_note("answer_candidate", "") or "")
        plan = list(state.get_note("plan", ()))
        flags = tuple(state.get_note("flags", ()))

        prompt = state.request.prompt.strip() or "(no prompt provided)"
        plan_summary = " -> ".join(
            ["planning", *plan, "policy", "fact_verify", "composition"]
        )

        answer = ""
        used_llm = False
        if flags:
            answer = self._policy_safe_answer(flags)
        elif self._llm is not None:
            answer = await self._llm_compose(state, candidate, spans, confidence, verdict, concerns)
            used_llm = bool(answer)

        if not answer:
            answer = self._fallback_answer(candidate, spans, verdict)

        # Rationale always references step capabilities so downstream tooling
        # and humans can trace what ran. This is debug metadata, not the
        # user-visible answer.
        rationale_lines = [
            f"Prompt: {_truncate(prompt)}",
            f"Pipeline: {plan_summary}",
            f"Confidence: {confidence:.2f} ({verdict})",
        ]
        if concerns:
            rationale_lines.append("Concerns: " + "; ".join(concerns))
        history_summaries = [
            f"- {step.capability}: {step.summary}" for step in state.history
        ]
        if history_summaries:
            rationale_lines.append("Steps:")
            rationale_lines.extend(history_summaries)
        if flags:
            rationale_lines.append("Flags: " + ", ".join(flags))
        rationale = "\n".join(rationale_lines)

        state.set_note("final_answer", answer)
        state.set_note("final_rationale", rationale)
        state.set_note("final_confidence", confidence)

        return StepRecord(
            capability=self.capability,
            summary=(
                f"composed answer (llm, confidence={confidence:.2f})"
                if used_llm
                else f"composed answer (fallback, confidence={confidence:.2f})"
            ),
            outputs={
                "answer": answer,
                "rationale": rationale,
                "confidence": confidence,
                "source": "llm" if used_llm else "fallback",
            },
        )

    def _policy_safe_answer(self, flags: tuple[str, ...]) -> str:
        if any("destructive_disk_command" in flag for flag in flags):
            return "No — avoid running destructive disk commands because they can permanently delete system data."
        if any("disk_overwrite" in flag for flag in flags):
            return "No — avoid disk-overwrite commands unless you have explicit recovery media and target verification."
        if any("fork_bomb" in flag for flag in flags):
            return "No — avoid fork bombs because they can make the system unusable."
        if any("credential" in flag or "private_key" in flag for flag in flags):
            return "No — I can’t help expose or extract credentials or private keys."
        return "No — that request appears unsafe, so I can’t help perform it."

    def _fallback_answer(
        self, candidate: str, spans: list[str], verdict: str
    ) -> str:
        candidate = candidate.strip()
        if candidate and verdict != "unsupported":
            return candidate
        if spans:
            return _truncate(spans[0])
        return "Insufficient information in provided context to answer confidently."

    async def _llm_compose(
        self,
        state: WorkingState,
        candidate: str,
        spans: list[str],
        confidence: float,
        verdict: str,
        concerns: list[str],
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
        user_payload = {
            "user_prompt": state.request.prompt,
            "answer_candidate": candidate,
            "confidence": confidence,
            "verdict": verdict,
            "concerns": concerns,
            "evidence_spans": spans,
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
                max_tokens=400,
            )
        except Exception:
            return ""
        return (text or "").strip()
