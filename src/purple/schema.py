"""Frozen dataclasses for the purple orchestrator.

The schema deliberately omits any task or context identifier — the orchestrator
must not be reachable through external IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _empty_str_map() -> Mapping[str, str]:
    return MappingProxyType({})


def _empty_any_map() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True)
class Attachment:
    name: str
    mime_type: str
    text: str | None = None
    data: bytes | None = None


@dataclass(frozen=True)
class TaskRequest:
    prompt: str
    context: tuple[str, ...] = ()
    attachments: tuple[Attachment, ...] = ()
    hints: Mapping[str, str] = field(default_factory=_empty_str_map)


@dataclass(frozen=True)
class CapabilityProfile:
    scores: Mapping[str, float]
    selected: tuple[str, ...]


@dataclass(frozen=True)
class StepRecord:
    capability: str
    summary: str
    outputs: Mapping[str, Any] = field(default_factory=_empty_any_map)


@dataclass(frozen=True)
class BudgetSnapshot:
    steps_used: int
    steps_limit: int
    elapsed_s: float
    time_limit_s: float | None


@dataclass(frozen=True)
class TaskResult:
    answer: str
    rationale: str
    steps: tuple[StepRecord, ...]
    profile: CapabilityProfile
    budget: BudgetSnapshot
    confidence: float
    flags: tuple[str, ...] = ()
