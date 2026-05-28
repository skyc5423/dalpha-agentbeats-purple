"""Step + wall-clock budget tracker."""

from __future__ import annotations

import time
from typing import Callable

from .schema import BudgetSnapshot


class BudgetTracker:
    def __init__(
        self,
        *,
        max_steps: int = 8,
        time_limit_s: float | None = 90.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if time_limit_s is not None and time_limit_s < 0:
            raise ValueError("time_limit_s must be non-negative or None")
        self._max_steps = max_steps
        self._time_limit_s = time_limit_s
        self._time_source = time_source
        self._steps_used = 0
        self._started_at: float | None = None

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def time_limit_s(self) -> float | None:
        return self._time_limit_s

    def start(self) -> None:
        self._started_at = self._time_source()
        self._steps_used = 0

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._time_source() - self._started_at)

    def can_continue(self) -> bool:
        if self._steps_used >= self._max_steps:
            return False
        if self._time_limit_s is not None and self._elapsed() >= self._time_limit_s:
            return False
        return True

    def record_step(self) -> None:
        self._steps_used += 1

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            steps_used=self._steps_used,
            steps_limit=self._max_steps,
            elapsed_s=self._elapsed(),
            time_limit_s=self._time_limit_s,
        )
