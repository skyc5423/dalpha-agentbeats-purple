"""Adapter between the A2A wire types and the purple ``TaskRequest`` schema.

Strips context/task identifiers — the solver layer must not be able to key
behaviour off external IDs.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    TextPart,
)
from a2a.utils import new_agent_text_message

from .schema import Attachment, TaskRequest, TaskResult


def _stringify_data(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, (list, tuple)):
        return "\n".join(_stringify_data(item) for item in data)
    if isinstance(data, dict):
        # Render shallow key:value pairs; deeper objects are JSON-ish.
        try:
            import json

            return json.dumps(data, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(data)
    return str(data)


def a2a_message_to_request(message: Message) -> TaskRequest:
    prompt_chunks: list[str] = []
    context_chunks: list[str] = []
    attachments: list[Attachment] = []

    for part in message.parts or ():
        root = getattr(part, "root", part)
        if isinstance(root, TextPart):
            text = root.text or ""
            if text:
                prompt_chunks.append(text)
        elif isinstance(root, DataPart):
            rendered = _stringify_data(root.data)
            if rendered:
                context_chunks.append(rendered)
        elif isinstance(root, FilePart):
            file = root.file
            if isinstance(file, FileWithBytes):
                data_bytes = file.bytes if isinstance(file.bytes, (bytes, bytearray)) else None
                text_payload: str | None = None
                if data_bytes is not None and (file.mime_type or "").startswith("text/"):
                    try:
                        text_payload = data_bytes.decode("utf-8", errors="replace")
                    except Exception:
                        text_payload = None
                attachments.append(
                    Attachment(
                        name=file.name or "attachment",
                        mime_type=file.mime_type or "application/octet-stream",
                        text=text_payload,
                        data=bytes(data_bytes) if data_bytes is not None else None,
                    )
                )
            elif isinstance(file, FileWithUri):
                attachments.append(
                    Attachment(
                        name=file.name or "attachment",
                        mime_type=file.mime_type or "application/octet-stream",
                        text=None,
                        data=None,
                    )
                )

    prompt = "\n".join(chunk for chunk in prompt_chunks if chunk).strip()
    context = tuple(chunk for chunk in context_chunks if chunk)

    return TaskRequest(
        prompt=prompt,
        context=context,
        attachments=tuple(attachments),
    )


def _result_payload(result: TaskResult) -> dict[str, Any]:
    return {
        "rationale": result.rationale,
        "confidence": result.confidence,
        "flags": list(result.flags),
        "profile": {
            "scores": dict(result.profile.scores),
            "selected": list(result.profile.selected),
        },
        "budget": asdict(result.budget),
        "steps": [
            {
                "capability": step.capability,
                "summary": step.summary,
                "outputs": dict(step.outputs),
            }
            for step in result.steps
        ],
    }


def result_to_artifact_parts(result: TaskResult) -> list[Part]:
    return [
        Part(root=TextPart(kind="text", text=result.answer)),
        Part(root=DataPart(kind="data", data=_result_payload(result))),
    ]


def result_to_status_message(result: TaskResult) -> Message:
    return new_agent_text_message(result.answer)


__all__ = [
    "a2a_message_to_request",
    "result_to_artifact_parts",
    "result_to_status_message",
]
