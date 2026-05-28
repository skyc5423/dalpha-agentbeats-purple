"""``shell_exec`` tool — inert by default; opt-in via injected runner.

This module deliberately never imports ``subprocess`` or any process-spawning
module at top level. If a deployer wants real execution, they pass a
``runner`` callable to the constructor.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..runtime.tool import ToolContext, ToolResult


class ShellExecTool:
    name = "shell_exec"
    description = (
        "Execute a shell command. Disabled in the public default build; a "
        "deployer-injected runner enables it."
    )
    arg_schema: Mapping[str, str] = {
        "command": "shell command to run; defaults to the user prompt",
    }

    def __init__(self, *, runner: Callable[[str], str] | None = None) -> None:
        self._runner = runner

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        if self._runner is None:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="shell execution disabled in public build",
                observation="shell_exec is inert without a deployer-injected runner",
                outputs={"executed": False, "reason": "no runner injected"},
                error="execution disabled",
            )
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            command = ctx.request.prompt or ""
        try:
            output = self._runner(command)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary=f"runner raised: {type(exc).__name__}",
                observation=str(exc)[:600],
                outputs={"executed": False},
                error=str(exc),
            )
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="runner returned output",
            observation=(output or "")[:600],
            outputs={"executed": True, "output": output},
        )


__all__ = ["ShellExecTool"]
