"""Controller protocol + action records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Union, runtime_checkable

from ..schema import TaskRequest
from .tool import Tool, ToolCall
from .transcript import Transcript


@dataclass(frozen=True)
class FinalAnswer:
    answer: str = ""


@dataclass(frozen=True)
class Surrender:
    reason: str = ""


Action = Union[ToolCall, FinalAnswer, Surrender]


@runtime_checkable
class Controller(Protocol):
    async def next_action(
        self,
        request: TaskRequest,
        transcript: Transcript,
        tools: Mapping[str, Tool],
    ) -> Action: ...


__all__ = [
    "Action",
    "Controller",
    "FinalAnswer",
    "Surrender",
]
