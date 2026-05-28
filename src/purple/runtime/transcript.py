"""Append-only transcript of (ToolCall, ToolResult) turns."""

from __future__ import annotations

from typing import Any

from ..schema import StepRecord
from .tool import ToolCall, ToolResult


class Transcript:
    def __init__(self) -> None:
        self._turns: list[tuple[ToolCall, ToolResult]] = []

    @property
    def turns(self) -> tuple[tuple[ToolCall, ToolResult], ...]:
        return tuple(self._turns)

    def append(self, call: ToolCall, result: ToolResult) -> None:
        self._turns.append((call, result))

    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c, _ in self._turns)

    def calls_by_name(self, name: str) -> tuple[ToolCall, ...]:
        return tuple(c for c, _ in self._turns if c.name == name)

    def last(self) -> tuple[ToolCall, ToolResult] | None:
        return self._turns[-1] if self._turns else None

    def latest_output(self, key: str) -> Any:
        for _, result in reversed(self._turns):
            if not result.ok:
                continue
            value = result.outputs.get(key)
            if value:
                return value
        return None

    def collected(self, key: str) -> list[Any]:
        """Concatenate list-valued ``outputs[key]`` across successful turns."""
        out: list[Any] = []
        seen: set[str] = set()
        for _, result in self._turns:
            if not result.ok:
                continue
            value = result.outputs.get(key)
            if not value:
                continue
            if isinstance(value, (list, tuple)):
                items = list(value)
            else:
                items = [value]
            for item in items:
                key_str = item if isinstance(item, str) else repr(item)
                if key_str in seen:
                    continue
                seen.add(key_str)
                out.append(item)
        return out

    def notes_view(self) -> dict[str, Any]:
        """Cheap dict view used to populate :class:`ToolContext`."""
        return {
            "spans": self.collected("spans"),
            "answer_candidate": self.latest_output("answer_candidate") or "",
            "urls_detected": self.collected("urls_detected"),
            "source_urls": self.collected("source_urls"),
            "results": self.collected("results"),
            "fetched_urls": self.collected("fetched_urls"),
            "fetched_pages": self.collected("fetched_pages"),
            "requirements": self.latest_output("requirements") or {},
            "required_outputs": self.latest_output("required_outputs") or [],
            "minimum_success_condition": self.latest_output("minimum_success_condition") or "",
            "missing_or_weak_points": self.latest_output("missing_or_weak_points") or [],
            "requirement_coverage": self.latest_output("requirement_coverage") or [],
            "next_queries": self.latest_output("next_queries") or [],
        }

    def to_step_records(self) -> tuple[StepRecord, ...]:
        records: list[StepRecord] = []
        for call, result in self._turns:
            summary = result.summary or (result.error if not result.ok else call.name)
            records.append(
                StepRecord(
                    capability=call.name,
                    summary=summary,
                    outputs=dict(result.outputs),
                )
            )
        return tuple(records)


__all__ = ["Transcript"]
