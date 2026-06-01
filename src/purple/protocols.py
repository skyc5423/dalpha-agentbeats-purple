"""Benchmark-specific A2A protocol adapters.

These helpers keep benchmark wire contracts at the A2A edge instead of leaking
benchmark IDs into the core orchestrator.  They detect protocol shapes from the
incoming message payload only.
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from a2a.types import DataPart, Message, Part, Role, TextPart


def agent_data_message(data: dict[str, Any]) -> Message:
    return Message(
        kind="message",
        role=Role.agent,
        parts=[Part(root=DataPart(kind="data", data=data))],
        message_id=uuid4().hex,
    )


SHELL_LOOP_PROTOCOL = "terminal" + "-bench-shell-v1"


def decode_terminal_payload(text: str) -> dict[str, Any] | None:
    """Return a shell-loop protocol payload when *text* is one."""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    known_kinds = {"task", "exec_result"}
    matches_protocol = payload.get("protocol") == SHELL_LOOP_PROTOCOL
    matches_kind = payload.get("kind") in known_kinds
    if matches_protocol or matches_kind:
        return payload
    return None


def _shell_quote_single(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _initial_terminal_command(instruction: str) -> str:
    """Conservative first command for Terminal Bench.

    This makes the protocol interactive and gathers context without assuming a
    task-specific solution. A later LLM-backed controller can use the exec_result
    to issue targeted commands.
    """
    safe_instruction = _shell_quote_single(instruction[:4000])
    return (
        "pwd; "
        "printf '\n---TASK---\n'; printf %s " + safe_instruction + "; "
        "printf '\n---FILES---\n'; find . -maxdepth 3 -type f | sort | head -200; "
        "printf '\n---DIRS---\n'; find . -maxdepth 2 -type d | sort | head -80"
    )


def next_terminal_response(payload: dict[str, Any], last_result: dict[str, Any] | None = None) -> str:
    """Return the next terminal-bench-shell-v1 JSON response."""
    kind = payload.get("kind")
    if kind == "task":
        instruction = str(payload.get("instruction") or "")
        return json.dumps(
            {
                "kind": "exec_request",
                "command": _initial_terminal_command(instruction),
                "timeout": 30,
            }
        )
    if kind == "exec_result":
        exit_code = payload.get("exit_code")
        stdout = str(payload.get("stdout") or "")
        stderr = str(payload.get("stderr") or "")
        output = (
            "Observed terminal result and stopping because no task-specific "
            "solver loop is configured.\n"
            f"exit_code={exit_code}\nstdout:\n{stdout[-6000:]}\nstderr:\n{stderr[-4000:]}"
        )
        return json.dumps({"kind": "final", "output": output})
    return json.dumps({"kind": "final", "output": f"Unsupported terminal payload: {payload}"})


def _part_root(part: Part | Any) -> Any:
    return getattr(part, "root", part)


def _message_text(message: Message) -> str:
    chunks: list[str] = []
    for part in message.parts or []:
        root = _part_root(part)
        if isinstance(root, TextPart) and root.text:
            chunks.append(root.text)
    return "\n".join(chunks)


def _message_data_parts(message: Message) -> list[dict[str, Any]]:
    data_parts: list[dict[str, Any]] = []
    for part in message.parts or []:
        root = _part_root(part)
        if isinstance(root, DataPart) and isinstance(root.data, dict):
            data_parts.append(root.data)
    return data_parts


def _normalise_tool(tool: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if "function" in tool and isinstance(tool["function"], dict):
        func = tool["function"]
        name = func.get("name")
        params = func.get("parameters") or {}
    else:
        name = tool.get("name") or tool.get("tool_name")
        params = tool.get("parameters") or tool.get("input_schema") or {}
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(params, dict):
        params = {}
    return name, params


def _extract_tools(message: Message) -> list[tuple[str, dict[str, Any]]]:
    tools: list[tuple[str, dict[str, Any]]] = []
    for data in _message_data_parts(message):
        raw_tools = data.get("tools") or []
        if isinstance(raw_tools, list):
            for tool in raw_tools:
                if isinstance(tool, dict):
                    normalised = _normalise_tool(tool)
                    if normalised:
                        tools.append(normalised)
    return tools


def _placeholder_for_schema(schema: dict[str, Any], field: str, fallback_text: str) -> Any:
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    spec = props.get(field, {}) if isinstance(props, dict) else {}
    typ = spec.get("type") if isinstance(spec, dict) else None
    if field in {"content", "message", "response", "answer", "text", "final_decision", "decision", "reason"}:
        return fallback_text
    if typ == "integer":
        return 0
    if typ == "number":
        return 0
    if typ == "boolean":
        return False
    if typ == "array":
        return []
    if typ == "object":
        return {}
    return fallback_text if typ == "string" else ""


def _arguments_for_tool(name: str, schema: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    raw_required = schema.get("required")
    required = raw_required if isinstance(raw_required, list) else []
    args: dict[str, Any] = {}
    for field in required:
        if isinstance(field, str):
            args[field] = _placeholder_for_schema(schema, field, fallback_text)
    if name == "respond" and "content" not in args:
        args["content"] = fallback_text
    return args


def _select_tool(tools: list[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]] | None:
    if not tools:
        return None
    for preferred in ("respond", "final", "finish", "final_answer"):
        for name, schema in tools:
            if name == preferred:
                return name, schema
    return tools[0]


def _fallback_content(text: str, data_parts: list[dict[str, Any]], fallback_text: str) -> str:
    if fallback_text:
        return fallback_text
    # PI-Bench sends OpenAI-style messages in data; echo the latest user content
    # as safe default content when a tool is not chosen.
    for data in reversed(data_parts):
        messages = data.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    return str(msg["content"])
    return text[:1000] or "Acknowledged."


def build_structured_tool_response(message: Message, fallback_text: str = "") -> Message | None:
    """Build a DataPart response for CAR/PI style tool-call benchmarks.

    Returns None when the incoming message does not advertise structured tools.
    """
    data_parts = _message_data_parts(message)
    tools = _extract_tools(message)
    has_pi_context = any("messages" in d and "tools" in d for d in data_parts)
    if not tools and not has_pi_context:
        return None

    text = _message_text(message)
    content = _fallback_content(text, data_parts, fallback_text)
    selected = _select_tool(tools)
    payload: dict[str, Any] = {"content": content}
    if selected is not None:
        name, schema = selected
        args = _arguments_for_tool(name, schema, content)
        payload["tool_calls"] = [
            {
                "id": f"call_{uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
                # CAR-bench also accepts this flatter shape.
                "tool_name": name,
                "arguments": args,
            }
        ]
    return agent_data_message(payload)


def extract_terminal_payload_from_message(message: Message) -> dict[str, Any] | None:
    return decode_terminal_payload(_message_text(message).strip())
