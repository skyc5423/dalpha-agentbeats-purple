"""Deterministic calculator specialist for simple arithmetic prompts."""

from __future__ import annotations

import re

from ..schema import StepRecord
from ..state import WorkingState
from ..tools import safe_eval

_EXPR_CHARS = re.compile(r"[(0-9][0-9\s+\-*/().%]*[0-9)]")
_DANGEROUS = re.compile(r"[A-Za-z_=`;{}\[\]\\]")


def _normalise_expression(text: str) -> str:
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("percent", "%")
    candidates = []
    for match in _EXPR_CHARS.finditer(text):
        expr = match.group(0).strip()
        if _DANGEROUS.search(expr):
            continue
        if any(op in expr for op in ("+", "-", "*", "/", "%")):
            candidates.append(expr)
    if not candidates:
        return ""
    expr = max(candidates, key=len)
    # Python's modulo operator is allowed by safe_eval. A trailing percent sign
    # in natural language is not an arithmetic expression, so leave it out.
    return expr.rstrip("% ")


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class CalculatorSpecialist:
    name = "calculator"
    capability = "calculator"

    async def run(self, state: WorkingState) -> StepRecord:
        prompt = state.request.prompt or ""
        expression = _normalise_expression(prompt)
        if not expression:
            return StepRecord(
                capability=self.capability,
                summary="no arithmetic expression detected",
                outputs={"calculated": False},
            )
        try:
            value = safe_eval(expression)
        except Exception as exc:
            return StepRecord(
                capability=self.capability,
                summary="calculator rejected expression",
                outputs={"calculated": False, "expression": expression, "error": str(exc)},
            )

        answer = _format_number(value)
        spans = list(state.get_note("research_spans", ()))
        spans.append(f"Calculation: {expression} = {answer}")
        state.set_note("research_spans", tuple(spans))
        state.set_note("answer_candidate", answer)
        return StepRecord(
            capability=self.capability,
            summary=f"calculated {expression} = {answer}",
            outputs={"calculated": True, "expression": expression, "answer_candidate": answer},
        )
