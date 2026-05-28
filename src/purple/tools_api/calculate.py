"""``calculate`` tool — safe arithmetic eval over an expression."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..runtime.tool import ToolContext, ToolResult
from ..tools import safe_eval

_EXPR_CHARS = re.compile(r"[(0-9][0-9\s+\-*/().%]*[0-9)]")
_DANGEROUS = re.compile(r"[A-Za-z_=`;{}\[\]\\]")


def normalise_expression(text: str) -> str:
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("percent", "%")
    candidates: list[str] = []
    for match in _EXPR_CHARS.finditer(text):
        expr = match.group(0).strip()
        if _DANGEROUS.search(expr):
            continue
        if any(op in expr for op in ("+", "-", "*", "/", "%")):
            candidates.append(expr)
    if not candidates:
        return ""
    return max(candidates, key=len).rstrip("% ")


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class CalculateTool:
    name = "calculate"
    description = "Evaluate a single arithmetic expression with +, -, *, /, %, **, parentheses."
    arg_schema: Mapping[str, str] = {
        "expression": "arithmetic expression; if omitted, extracted from the user prompt",
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        expr = args.get("expression")
        if not isinstance(expr, str) or not expr.strip():
            expr = normalise_expression(ctx.request.prompt or "")
        expr = (expr or "").strip()
        if not expr:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="no arithmetic expression detected",
                observation="missing expression",
                outputs={"calculated": False},
                error="empty expression",
            )
        try:
            value = safe_eval(expr)
        except Exception as exc:  # noqa: BLE001 - return as observation
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="calculator rejected expression",
                observation=f"safe_eval failed: {exc}",
                outputs={"calculated": False, "expression": expr},
                error=str(exc),
            )
        answer = _format_number(value)
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=f"calculated {expr} = {answer}",
            observation=f"{expr} = {answer}",
            outputs={
                "calculated": True,
                "expression": expr,
                "answer_candidate": answer,
                "spans": [f"Calculation: {expr} = {answer}"],
                # Calculator output is deterministic and complete by
                # construction; no sufficiency_check needed before finalising.
                "sufficient_alone": True,
            },
        )


__all__ = ["CalculateTool", "normalise_expression"]
