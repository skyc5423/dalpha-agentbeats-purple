"""Controller loop driver.

The loop owns:
- budget enforcement,
- back-to-back duplicate-call detection,
- per-tool attempt caps,
- unknown-tool handling (records a failed observation, never raises),
- exception isolation (tool exceptions become failed observations).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from ..budget import BudgetTracker
from ..schema import TaskRequest
from .controller import Controller, FinalAnswer, Surrender
from .rule_controller import RuleBasedController
from .tool import ToolContext, ToolResult
from .transcript import Transcript


_UNCAPPED_TOOLS = {"sufficiency_check"}
_MIN_TOOL_ATTEMPTS = {
    "web_fetch": 8,
    "web_search": 4,
}


def _registry_items(registry: Any) -> dict[str, Any]:
    items = getattr(registry, "items", None)
    if callable(items):
        return dict(items())
    return dict(registry)


@dataclass(frozen=True)
class LoopOutcome:
    final_answer: str = ""
    surrendered: bool = False
    surrender_reason: str = ""
    truncated: bool = False
    flags: tuple[str, ...] = ()


class ControllerLoop:
    def __init__(
        self,
        *,
        controller: Controller,
        registry: Any,
        budget: BudgetTracker,
        max_attempts_per_tool: int = 2,
        max_consecutive_duplicates: int = 2,
        step_callback: Callable[[Transcript, BudgetTracker], None] | None = None,
    ) -> None:
        self._controller = controller
        self._registry = registry
        self._budget = budget
        self._max_attempts_per_tool = max_attempts_per_tool
        self._max_consecutive_duplicates = max_consecutive_duplicates
        self._step_callback = step_callback

    async def run(
        self,
        request: TaskRequest,
        transcript: Transcript,
        scratch: dict[str, Any],
    ) -> LoopOutcome:
        flags: list[str] = []
        truncated = False
        surrendered = False
        surrender_reason = ""
        final_answer: str | None = None
        last_call_key: tuple | None = None
        seen_external_call_keys: set[tuple] = set()
        consecutive_duplicates = 0

        tool_map = _registry_items(self._registry)
        fallback_controller = RuleBasedController(max_attempts=self._max_attempts_per_tool)

        while True:
            if not self._budget.can_continue():
                truncated = True
                break

            if "analyze_requirements" in tool_map and "analyze_requirements" not in transcript.names():
                action = dataclasses.replace(
                    fallback_controller._make_call("analyze_requirements"),  # noqa: SLF001 - shared primitive call builder
                    id="requirements-1",
                )
            else:
                action = await self._controller.next_action(request, transcript, tool_map)

            if isinstance(action, FinalAnswer):
                if _final_needs_more_evidence_check(transcript) and self._budget.can_continue():
                    action = await fallback_controller.next_action(request, transcript, tool_map)
                    if isinstance(action, FinalAnswer):
                        if _final_needs_more_evidence_check(transcript):
                            surrendered = True
                            surrender_reason = "evidence insufficient after fallback"
                            break
                        final_answer = action.answer or ""
                        break
                    if isinstance(action, Surrender):
                        surrendered = True
                        surrender_reason = action.reason or ""
                        break
                else:
                    final_answer = action.answer or ""
                    break
            if isinstance(action, Surrender):
                if _final_needs_more_evidence_check(transcript) and self._budget.can_continue():
                    action = await fallback_controller.next_action(request, transcript, tool_map)
                    if isinstance(action, FinalAnswer):
                        if _final_needs_more_evidence_check(transcript):
                            surrendered = True
                            surrender_reason = "evidence insufficient after fallback"
                            break
                        final_answer = action.answer or ""
                        break
                    if not isinstance(action, Surrender):
                        pass
                    else:
                        surrendered = True
                        surrender_reason = action.reason or ""
                        break
                else:
                    surrendered = True
                    surrender_reason = action.reason or ""
                    break

            call = action
            call_key = (call.name, _hashable_args(call.args))
            if call.name in {"web_search", "web_fetch"} and call_key in seen_external_call_keys:
                duplicate_outputs: dict[str, Any] = {}
                if call.name == "web_search":
                    query = call.args.get("query") if isinstance(call.args, Mapping) else None
                    prior = call.args.get("attempted_queries") if isinstance(call.args, Mapping) else None
                    attempted = [str(x) for x in prior] if isinstance(prior, (list, tuple)) else []
                    if isinstance(query, str) and query.strip():
                        duplicate_outputs["query"] = query.strip()
                        duplicate_outputs["skipped_query"] = query.strip()
                        if query.strip() not in attempted:
                            attempted.append(query.strip())
                    duplicate_outputs["attempted_queries"] = attempted
                    duplicate_outputs["results"] = []
                    duplicate_outputs["spans"] = []
                elif call.name == "web_fetch":
                    url = call.args.get("url") if isinstance(call.args, Mapping) else None
                    if isinstance(url, str) and url.strip():
                        duplicate_outputs["url"] = url.strip()
                    duplicate_outputs["fetched"] = False
                result = ToolResult(
                    tool_call_id=call.id,
                    ok=False,
                    summary=f"duplicate {call.name} call skipped",
                    observation=f"duplicate {call.name} call skipped",
                    outputs=duplicate_outputs,
                    error="duplicate-call",
                )
                transcript.append(call, result)
                self._budget.record_step()
                self._emit_step(transcript)
                continue
            if call_key == last_call_key:
                consecutive_duplicates += 1
                if consecutive_duplicates >= self._max_consecutive_duplicates:
                    surrendered = True
                    surrender_reason = "consecutive duplicate calls"
                    break
            else:
                consecutive_duplicates = 0
            last_call_key = call_key

            tool = tool_map.get(call.name)
            if tool is None:
                result = ToolResult(
                    tool_call_id=call.id,
                    ok=False,
                    summary=f"unknown tool: {call.name}",
                    observation=f"unknown tool {call.name!r}",
                    error="unknown-tool",
                )
                transcript.append(call, result)
                self._budget.record_step()
                self._emit_step(transcript)
                continue

            attempts = sum(1 for name in transcript.names() if name == call.name)
            tool_cap = _attempt_cap_for_name(call.name, self._max_attempts_per_tool)
            if tool_cap is not None and attempts >= tool_cap:
                result = ToolResult(
                    tool_call_id=call.id,
                    ok=False,
                    summary=f"{call.name} attempt cap reached",
                    observation=f"{call.name} has been attempted {attempts} time(s) already",
                    error="max-attempts",
                )
                transcript.append(call, result)
                self._budget.record_step()
                self._emit_step(transcript)
                continue

            ctx = ToolContext(
                request=request,
                notes=transcript.notes_view(),
                scratch=scratch,
                steps_remaining=max(0, self._budget.max_steps - self._budget.snapshot().steps_used),
            )
            try:
                result = await tool.run(call.args, ctx)
            except Exception as exc:  # noqa: BLE001 - tool isolation
                result = ToolResult(
                    tool_call_id=call.id,
                    ok=False,
                    summary=f"{type(exc).__name__}: {exc}",
                    observation=f"tool raised: {type(exc).__name__}",
                    error=str(exc),
                )
            if not result.tool_call_id:
                result = dataclasses.replace(result, tool_call_id=call.id)
            if call.name in {"web_search", "web_fetch"}:
                seen_external_call_keys.add(call_key)
            transcript.append(call, result)
            self._budget.record_step()
            self._emit_step(transcript)

        if truncated:
            flags.append("budget-truncated")

        return LoopOutcome(
            final_answer=final_answer or "",
            surrendered=surrendered,
            surrender_reason=surrender_reason,
            truncated=truncated,
            flags=tuple(flags),
        )

    def _emit_step(self, transcript: Transcript) -> None:
        if self._step_callback is None:
            return
        self._step_callback(transcript, self._budget)


def _final_needs_more_evidence_check(transcript: Transcript) -> bool:
    """Block controller final/stop when evidence has not been checked.

    LLM controllers can jump directly from a tool-produced candidate to
    ``FinalAnswer``. For research-style tasks this skips the requirement coverage
    critic, so force the deterministic fallback controller to run
    ``sufficiency_check`` or follow-up search first whenever new evidence exists
    after the latest sufficiency verdict, or the latest verdict is explicitly
    insufficient.
    """

    last_evidence_idx = -1
    last_suff_idx = -1
    latest_sufficient: bool | None = None
    for i, (call, result) in enumerate(transcript.turns):
        if not result.ok:
            continue
        if call.name == "sufficiency_check":
            last_suff_idx = i
            value = result.outputs.get("sufficient")
            if value is True:
                latest_sufficient = True
            elif value is False:
                latest_sufficient = False
        elif _looks_like_evidence_outputs(result.outputs):
            last_evidence_idx = i
    if latest_sufficient is False:
        return True
    return last_evidence_idx >= 0 and last_evidence_idx > last_suff_idx


def _looks_like_evidence_outputs(outputs: Any) -> bool:
    if not isinstance(outputs, dict):
        return False
    if outputs.get("answer_candidate"):
        return True
    spans = outputs.get("spans")
    if isinstance(spans, (list, tuple)) and any(spans):
        return True
    fetched = outputs.get("fetched_pages")
    if isinstance(fetched, (list, tuple)) and any(fetched):
        return True
    results = outputs.get("results")
    if isinstance(results, (list, tuple)) and any(results):
        return True
    return False


def _latest_sufficiency_insufficient(transcript: Transcript) -> bool:
    for call, result in reversed(transcript.turns):
        if call.name == "sufficiency_check" and result.ok:
            return result.outputs.get("sufficient") is False
        if result.ok and result.outputs.get("sufficient_alone") is True:
            return False
    return False


def _attempt_cap_for_name(tool_name: str, default: int) -> int | None:
    if tool_name in _UNCAPPED_TOOLS:
        return None
    return max(default, _MIN_TOOL_ATTEMPTS.get(tool_name, default))


def _hashable_args(args: Any) -> Any:
    if isinstance(args, dict):
        return tuple(sorted((k, _hashable_args(v)) for k, v in args.items()))
    if isinstance(args, (list, tuple)):
        return tuple(_hashable_args(v) for v in args)
    return args


__all__ = ["ControllerLoop", "LoopOutcome"]
