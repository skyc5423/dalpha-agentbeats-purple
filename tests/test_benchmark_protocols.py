import json

from a2a.types import DataPart, Message, Part, Role, TextPart

from purple.io_adapter import result_to_status_message
from purple.protocols import (
    build_structured_tool_response,
    decode_terminal_payload,
    next_terminal_response,
)
from purple.schema import BudgetSnapshot, CapabilityProfile, TaskResult


def _result(answer: str) -> TaskResult:
    return TaskResult(
        answer=answer,
        rationale="test",
        steps=(),
        profile=CapabilityProfile(scores={}, selected=()),
        budget=BudgetSnapshot(steps_used=0, steps_limit=1, elapsed_s=0.0, time_limit_s=None),
        confidence=1.0,
    )


def test_task_result_is_available_as_status_message_for_raw_message_clients():
    message = result_to_status_message(_result("final answer"))

    assert message.parts[0].root.text == "final answer"
    assert message.role == Role.agent


def test_terminal_bench_task_payload_returns_exec_request_json():
    payload = decode_terminal_payload(json.dumps({
        "kind": "task",
        "protocol": "terminal-bench-shell-v1",
        "instruction": "create /tmp/hello.txt",
    }))

    response = json.loads(next_terminal_response(payload, last_result=None))

    assert response["kind"] == "exec_request"
    assert isinstance(response["command"], str)
    assert response["command"].strip()
    assert isinstance(response["timeout"], int)


def test_terminal_bench_exec_result_payload_returns_final_json():
    payload = decode_terminal_payload(json.dumps({
        "kind": "exec_result",
        "exit_code": 0,
        "stdout": "ok",
        "stderr": "",
    }))

    response = json.loads(next_terminal_response(payload, last_result=None))

    assert response["kind"] == "final"
    assert "ok" in response["output"]


def test_structured_tool_response_emits_data_part_tool_calls_for_car_or_pi_tools():
    message = Message(
        kind="message",
        role=Role.user,
        message_id="m1",
        parts=[
            Part(root=TextPart(kind="text", text="User asks for help")),
            Part(root=DataPart(kind="data", data={
                "tools": [
                    {
                        "name": "respond",
                        "parameters": {
                            "type": "object",
                            "properties": {"content": {"type": "string"}},
                            "required": ["content"],
                        },
                    }
                ]
            })),
        ],
    )

    response = build_structured_tool_response(message, fallback_text="I can help.")

    assert response is not None
    data = response.parts[0].root.data
    assert data["tool_calls"][0]["function"]["name"] == "respond"
    assert json.loads(data["tool_calls"][0]["function"]["arguments"])["content"]
