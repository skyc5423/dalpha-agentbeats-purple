"""Thin A2A adapter that delegates to the purple orchestrator."""

from __future__ import annotations

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

        structured_response = build_structured_tool_response(message, fallback_text=result.answer)
        if structured_response is not None:
            await updater.complete(structured_response)
            return

        await updater.add_artifact(
            parts=result_to_artifact_parts(result),
            name="Purple Agent Answer",
        )
        # Some green agents read only task.status.message/raw_message and ignore
        # artifacts. Always provide the final text there as well.
        await updater.complete(result_to_status_message(result))
