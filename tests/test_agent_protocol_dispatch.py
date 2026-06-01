import json

import pytest
from a2a.types import DataPart, Message, Part, Role, TaskState, TextPart

from agent import Agent


class FakeUpdater:
    def __init__(self):
        self.events = []
        self._terminal_state_reached = False

    async def update_status(self, state, message=None, final=False, **kwargs):
        self.events.append((state, message, final))
        if state in {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}:
            self._terminal_state_reached = True

    async def add_artifact(self, **kwargs):
        self.events.append(("artifact", kwargs, False))

    async def complete(self, message=None):
        await self.update_status(TaskState.completed, message, final=True)


def _message(parts):
    return Message(kind="message", role=Role.user, message_id="m", parts=parts)


@pytest.mark.asyncio
async def test_agent_returns_terminal_protocol_status_message_without_orchestrator():
    agent = Agent()
    updater = FakeUpdater()
    msg = _message([
        Part(root=TextPart(kind="text", text=json.dumps({
            "kind": "task",
            "protocol": "terminal-bench-shell-v1",
            "instruction": "inspect files",
        })))
    ])

    await agent.run(msg, updater)

    completed = [event for event in updater.events if event[0] == TaskState.completed]
    assert completed
    payload = json.loads(completed[-1][1].parts[0].root.text)
    assert payload["kind"] == "exec_request"


@pytest.mark.asyncio
async def test_agent_returns_tool_call_status_message_for_structured_tool_request():
    agent = Agent()
    updater = FakeUpdater()
    msg = _message([
        Part(root=TextPart(kind="text", text="Can you help with the car?")),
        Part(root=DataPart(kind="data", data={
            "tools": [{
                "name": "respond",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            }]
        })),
    ])

    await agent.run(msg, updater)

    completed = [event for event in updater.events if event[0] == TaskState.completed]
    assert completed
    data = completed[-1][1].parts[0].root.data
    assert data["tool_calls"][0]["function"]["name"] == "respond"
