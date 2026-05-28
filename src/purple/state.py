"""Mutable working state passed between specialists during one solve()."""

from __future__ import annotations

from typing import Any, Mapping

from .budget import BudgetTracker
from .schema import StepRecord, TaskRequest


class WorkingState:
    def __init__(self, request: TaskRequest, budget: BudgetTracker) -> None:
        self._request = request
        self._budget = budget
        self._history: list[StepRecord] = []
        self._notes: dict[str, Any] = {}

    @property
    def request(self) -> TaskRequest:
        return self._request

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    @property
    def history(self) -> tuple[StepRecord, ...]:
        return tuple(self._history)

    @property
    def notes(self) -> Mapping[str, Any]:
        return self._notes

    def append_step(self, step: StepRecord) -> None:
        self._history.append(step)

    def set_note(self, key: str, value: Any) -> None:
        self._notes[key] = value

    def get_note(self, key: str, default: Any = None) -> Any:
        return self._notes.get(key, default)
