"""Thin A2A adapter that delegates to the purple orchestrator."""

from __future__ import annotations

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import new_agent_text_message

from messenger import Messenger
from purple import Orchestrator, a2a_message_to_request, result_to_artifact_parts


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
        request = a2a_message_to_request(message)
        await updater.update_status(
            TaskState.working, new_agent_text_message("Profiling task...")
        )
        result = await self._orchestrator.solve(request)
        await updater.add_artifact(
            parts=result_to_artifact_parts(result),
            name="Purple Agent Answer",
        )
