"""Orchestrator — owns the controller loop, policy gates, and finalisation.

The orchestrator no longer runs a fixed specialist pipeline. It:

1. Runs :meth:`PolicyGate.preflight` against the request; on hit, the loop
   never starts and a refusal is composed.
2. Drives a :class:`ControllerLoop` that lets a controller (LLM-backed or
   rule-based) pick primitive tools from a :class:`ToolRegistry` until it
   emits a final answer or the budget runs out.
3. Runs :class:`Finalizer` to compose + verify the answer, then applies
   :meth:`PolicyGate.postflight` (credential redaction).

It dispatches purely on tool names; benchmark or task identifiers are not
visible to this layer.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .budget import BudgetTracker
from .llm import LLMClient, llm_from_env
from .profiler import CapabilityProfiler
from .registry import ToolRegistry
from .runtime import (
    ControllerLoop,
    Finalizer,
    LLMController,
    PolicyGate,
    RuleBasedController,
    Transcript,
)
from .runtime.controller import Controller
from .schema import TaskRequest, TaskResult


class Orchestrator:
    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        profiler: CapabilityProfiler | None = None,
        llm: LLMClient | None = None,
        controller: Controller | None = None,
        finalizer: Finalizer | None = None,
        policy_gate: PolicyGate | None = None,
        max_steps: int = 20,
        time_limit_s: float | None = 180.0,
        time_source: Callable[[], float] = time.monotonic,
        max_attempts_per_tool: int = 3,
    ) -> None:
        self._llm = llm if llm is not None else llm_from_env()
        self._llm_configured = self._llm is not None
        self._profiler = profiler or CapabilityProfiler()
        if registry is None:
            from .tools_api import default_tools

            registry = default_tools(llm=self._llm)
        self._registry = registry
        self._controller: Controller = controller or self._default_controller()
        self._finalizer = finalizer or Finalizer(llm=self._llm)
        self._policy = policy_gate or PolicyGate()
        self._max_steps = max_steps
        self._time_limit_s = time_limit_s
        self._time_source = time_source
        self._max_attempts_per_tool = max_attempts_per_tool

    def _default_controller(self) -> Controller:
        if self._llm is not None:
            return LLMController(self._llm)
        return RuleBasedController(profiler=self._profiler)

    async def solve(self, request: TaskRequest) -> TaskResult:
        budget = BudgetTracker(
            max_steps=self._max_steps,
            time_limit_s=self._time_limit_s,
            time_source=self._time_source,
        )
        budget.start()

        profile = self._profiler.profile(request)
        transcript = Transcript()
        flags: list[str] = []
        if not self._llm_configured:
            flags.append("llm-unavailable")

        preflight = self._policy.preflight(request)
        for tag in preflight.flags:
            if tag not in flags:
                flags.append(tag)

        scratch: dict[str, Any] = {}
        loop_outcome = None
        refusal: str | None = None
        if preflight.flags:
            refusal = self._policy.refusal_for(preflight.flags)
        else:
            loop = ControllerLoop(
                controller=self._controller,
                registry=self._registry,
                budget=budget,
                max_attempts_per_tool=self._max_attempts_per_tool,
            )
            loop_outcome = await loop.run(request, transcript, scratch)
            for tag in loop_outcome.flags:
                if tag not in flags:
                    flags.append(tag)

        finalization = await self._finalizer.run(
            request,
            transcript,
            flags=tuple(flags),
            controller_answer=(loop_outcome.final_answer if loop_outcome else ""),
            refusal=refusal,
        )

        post = self._policy.postflight(finalization.answer)
        answer = finalization.answer
        if post.redacted:
            for tag in post.flags:
                if tag not in flags:
                    flags.append(tag)
            answer = self._policy.refusal_for(tuple(post.flags))

        return TaskResult(
            answer=answer,
            rationale=finalization.rationale,
            steps=transcript.to_step_records(),
            profile=profile,
            budget=budget.snapshot(),
            confidence=finalization.confidence,
            flags=tuple(flags),
        )
