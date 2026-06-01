"""Thin A2A adapter that delegates to the purple orchestrator."""

from __future__ import annotations

import json
import os
import sys

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import new_agent_text_message

from messenger import Messenger
from purple import Orchestrator, a2a_message_to_request, result_to_artifact_parts, result_to_status_message
from purple.protocols import (
    build_structured_tool_response,
    extract_terminal_payload_from_message,
    next_terminal_response,
)


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _log_result_summary(result) -> None:
    """Emit secret-safe one-line diagnostics for GitHub Actions/container logs."""
    tool_counts: dict[str, int] = {}
    for step in result.steps:
        tool_counts[step.capability] = tool_counts.get(step.capability, 0) + 1
    payload = {
        "event": "purple_result",
        "confidence": result.confidence,
        "flags": list(result.flags),
        "answer_chars": len(result.answer or ""),
        "answer_prefix": _truncate(result.answer, 160),
        "budget": {
            "steps_used": result.budget.steps_used,
            "steps_limit": result.budget.steps_limit,
            "elapsed_s": round(result.budget.elapsed_s, 3),
            "time_limit_s": result.budget.time_limit_s,
        },
        "tool_counts": tool_counts,
        "step_summaries": [
            {
                "tool": step.capability,
                "summary": _truncate(step.summary),
                "output_keys": sorted(str(key) for key in step.outputs.keys()),
            }
            for step in result.steps[-12:]
        ],
    }
    print("PURPLE_TRACE " + json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


class Agent:
    def __init__(self, orchestrator: Orchestrator | None = None) -> None:
        self.messenger = Messenger()
        self._orchestrator = orchestrator or Orchestrator()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Run the orchestrator against an incoming A2A message.

        Args:
            message: The incoming A2A message.
            updater: Reports progress (``update_status``) and results
                (``add_artifact``).

        Outbound calls to peer agents are available via ``self.messenger`` but
        are not used by the default in-context pipeline.
        """
        terminal_payload = extract_terminal_payload_from_message(message)
        if terminal_payload is not None:
            response_text = next_terminal_response(terminal_payload, last_result=None)
            await updater.complete(new_agent_text_message(response_text))
            return

        request = a2a_message_to_request(message)
        await updater.update_status(
            TaskState.working, new_agent_text_message("Profiling task...")
        )
        result = await self._orchestrator.solve(request)
        if _env_enabled("PURPLE_TRACE_RESULTS", True):
            _log_result_summary(result)

        structured_response = build_structured_tool_response(message, fallback_text=result.answer)
        if structured_response is not None:
            await updater.complete(structured_response)
            return

        if _env_enabled("PURPLE_ENABLE_DEBUG_ARTIFACT", False):
            await updater.add_artifact(
                parts=result_to_artifact_parts(result),
                name="Purple Agent Debug",
            )
        # Some green agents read only task.status.message/raw_message and ignore
        # artifacts. Put the final benchmark answer only in status by default;
        # debug artifacts are opt-in because answer-only evaluators may
        # concatenate artifact text/data into the scored response.
        await updater.complete(result_to_status_message(result))
