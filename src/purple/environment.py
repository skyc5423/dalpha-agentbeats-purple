"""Read-only text environment over the request's in-context material.

This intentionally does no I/O. Outbound research, sandbox shells, etc. must
be supplied by a deployer-injected environment, not by the default build.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import TaskRequest


@runtime_checkable
class Environment(Protocol):
    name: str

    async def read(self, query: str) -> str:
        ...


class TextEnvironment:
    """Returns substrings of the request's context that mention the query."""

    name = "text"

    def __init__(self, request: TaskRequest) -> None:
        self._request = request

    async def read(self, query: str) -> str:
        needle = (query or "").strip().lower()
        if not needle:
            return ""
        hits: list[str] = []
        for src in self._request.context:
            if needle in src.lower():
                hits.append(src)
        for att in self._request.attachments:
            if att.text and needle in att.text.lower():
                hits.append(att.text)
        return "\n".join(hits)
