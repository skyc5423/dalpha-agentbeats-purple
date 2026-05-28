"""Tool protocol, call/result records, and the read-only context handle.

Tools are the primitive units the controller can invoke. They never reach
into orchestrator internals; everything they need lives in :class:`ToolContext`
which is built fresh per call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from ..schema import TaskRequest


def _empty_any_map() -> Mapping[str, Any]:
    return MappingProxyType({})


def _empty_str_map() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: Mapping[str, Any] = field(default_factory=_empty_any_map)


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    ok: bool
    summary: str
    observation: str = ""
    outputs: Mapping[str, Any] = field(default_factory=_empty_any_map)
    error: str = ""


class ToolContext:
    """Read-only handle threaded to every :meth:`Tool.run` call.

    Tools see the task request, a view of the transcript's accumulated notes
    (spans, candidates, urls), and a per-request scratch dict that they may
    use to share lightweight state across turns. The orchestrator owns the
    transcript itself.
    """

    __slots__ = ("_request", "_notes", "_scratch", "_steps_remaining")

    def __init__(
        self,
        *,
        request: TaskRequest,
        notes: Mapping[str, Any],
        scratch: dict[str, Any],
        steps_remaining: int,
    ) -> None:
        self._request = request
        self._notes = notes
        self._scratch = scratch
        self._steps_remaining = steps_remaining

    @property
    def request(self) -> TaskRequest:
        return self._request

    @property
    def notes(self) -> Mapping[str, Any]:
        return self._notes

    @property
    def scratch(self) -> dict[str, Any]:
        return self._scratch

    @property
    def steps_remaining(self) -> int:
        return self._steps_remaining


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    arg_schema: Mapping[str, str]

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult: ...


__all__ = [
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolResult",
]
