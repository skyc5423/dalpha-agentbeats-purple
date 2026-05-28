"""Tool registry for the controller-loop runtime.

The controller dispatches against a :class:`ToolRegistry` of primitive
``Tool`` objects. The legacy ``CapabilityRegistry`` (specialist-based) is no
longer wired into the orchestrator; tool wrappers in
:mod:`purple.tools_api` replace it. ``default_registry`` is kept as an alias
of :func:`purple.tools_api.default_tools` so existing callers continue to
work and now receive a :class:`ToolRegistry`.
"""

from __future__ import annotations

from typing import Callable

from .llm import LLMClient
from .runtime.tool import Tool
from .tools import WebAnswerer, WebClient


class ToolRegistry:
    """Name → Tool registry. Tool names must be unique."""

    def __init__(self) -> None:
        self._items: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("tool must declare a non-empty name string")
        if name in self._items:
            raise ValueError(f"tool {name!r} already registered")
        self._items[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._items.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._items.keys())

    def capabilities(self) -> tuple[str, ...]:
        """Compatibility shim: same as :meth:`names`."""
        return self.names()

    def items(self) -> dict[str, Tool]:
        return dict(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __len__(self) -> int:
        return len(self._items)


def default_registry(
    *,
    llm: LLMClient | None = None,
    shell_runner: Callable[[str], str] | None = None,
    web_client: WebClient | None = None,
    web_answerer: WebAnswerer | None = None,
    use_env_web_answerer: bool = True,
) -> ToolRegistry:
    """Backward-compatible alias of :func:`purple.tools_api.default_tools`."""

    from .tools_api import default_tools

    return default_tools(
        llm=llm,
        shell_runner=shell_runner,
        web_client=web_client,
        web_answerer=web_answerer,
        use_env_web_answerer=use_env_web_answerer,
    )


__all__ = ["ToolRegistry", "default_registry"]
