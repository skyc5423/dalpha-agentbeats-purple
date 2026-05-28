"""Specialist protocol — every specialist must satisfy this shape."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import StepRecord
from ..state import WorkingState


@runtime_checkable
class Specialist(Protocol):
    name: str
    capability: str

    async def run(self, state: WorkingState) -> StepRecord:
        ...
