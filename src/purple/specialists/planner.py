"""Planner specialist — picks an ordered sequence of mid-pipeline capabilities.

When an :class:`LLMClient` is injected, the planner asks the model to pick a
plan from the allowed-capability set. The model's response is parsed as JSON
and filtered against the allowed list before use. If no LLM is available, or
the LLM output is malformed, the planner falls back to the deterministic
capability profile.
"""

from __future__ import annotations

import json

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import CapabilityProfile, StepRecord
from ..state import WorkingState
from ..tools import extract_json

# Capabilities the orchestrator handles itself, not the planner.
_RESERVED = frozenset({"planning", "policy", "fact_verify", "composition"})


class PlannerSpecialist:
    name = "planner"
    capability = "planning"

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, state: WorkingState) -> StepRecord:
        profile: CapabilityProfile | None = state.get_note("profile")
        allowed = tuple(
            cap for cap in (profile.selected if profile else ()) if cap not in _RESERVED
        )
        plan = allowed
        source = "profile"

        if self._llm is not None and allowed:
            llm_plan = await self._llm_plan(state, allowed)
            if llm_plan is not None:
                plan = llm_plan
                source = "llm"

        state.set_note("plan", plan)
        summary = (
            f"plan ({source}): {' -> '.join(plan)}" if plan else f"plan ({source}): gates only"
        )
        return StepRecord(
            capability=self.capability,
            summary=summary,
            outputs={"plan": list(plan), "source": source},
        )

    async def _llm_plan(
        self, state: WorkingState, allowed: tuple[str, ...]
    ) -> tuple[str, ...] | None:
        assert self._llm is not None
        system_text = "\n\n".join(
            chunk for chunk in (load_prompt("system"), load_prompt("planner")) if chunk
        )
        user_payload = {
            "user_prompt": state.request.prompt,
            "allowed_capabilities": list(allowed),
            "fallback_plan": list(allowed),
        }
        user_text = (
            "Inputs (JSON):\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\nReturn the plan JSON object."
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", system_text),
                    ChatMessage("user", user_text),
                ],
                tag="planner",
                max_tokens=200,
            )
        except Exception:
            return None
        data = extract_json(text)
        if not isinstance(data, dict):
            return None
        raw_plan = data.get("plan")
        if not isinstance(raw_plan, list):
            return None
        allowed_set = set(allowed)
        filtered: list[str] = []
        for cap in raw_plan:
            if not isinstance(cap, str):
                continue
            if cap in _RESERVED:
                continue
            if cap not in allowed_set:
                continue
            if cap in filtered:
                continue
            filtered.append(cap)
        # Skill text is loaded so it's available to humans reviewing prompts
        # and to specialists that want to reference it; we don't inject it
        # here because the planner has its own scoped prompt.
        _ = load_skill("composition")
        if not filtered:
            return allowed
        return tuple(filtered)
