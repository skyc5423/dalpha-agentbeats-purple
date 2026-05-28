"""``finish`` tool — sentinel the controller can call to end the loop.

The controller can also emit a ``final`` action directly; the tool form is
provided so a controller that only knows how to call tools can still terminate.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..runtime.tool import ToolContext, ToolResult


class FinishTool:
    name = "finish"
    description = "Emit the final user-visible answer and stop the loop."
    arg_schema: Mapping[str, str] = {
        "answer": "the final user-visible answer (string)",
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        answer = args.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="finish requested by controller",
            observation=answer[:600],
            outputs={
                "answer_candidate": answer,
                "final": True,
                "sufficient_alone": True,
            },
        )


__all__ = ["FinishTool"]
