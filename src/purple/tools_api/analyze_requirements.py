"""``analyze_requirements`` tool — first-pass task success criteria.

This tool is deliberately generic: before any research/search step, it asks the
LLM to turn the user task into an explicit checklist of required outputs and
minimum evidence conditions. Later tools/verifiers can then track coverage of
that checklist instead of relying on an implicit plan.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt
from ..runtime.tool import ToolContext, ToolResult
from ..tools import extract_json


class AnalyzeRequirementsTool:
    name = "analyze_requirements"
    description = (
        "Analyze the user task before research and produce a generic checklist "
        "of required outputs, evidence requirements, minimum success condition, "
        "common failure modes, and initial search hints."
    )
    arg_schema: Mapping[str, str] = {
        "task": "optional task override; defaults to the user prompt",
    }

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            task = ctx.request.prompt or ""
        task = task.strip()
        if not task:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="analyze_requirements requires a task",
                observation="missing task",
                outputs={"requirements": {}},
                error="missing task",
            )
        data = await self._llm_analyze(task) if self._llm is not None else None
        if data is None:
            data = self._fallback_requirements(task)
        data = _normalise_requirements(data)
        required_count = len(data.get("required_outputs", []))
        missing_preview = "; ".join(
            str(item.get("description", item.get("id", "")))
            for item in data.get("required_outputs", [])[:5]
            if isinstance(item, dict)
        )
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=f"analyzed task requirements ({required_count} required output(s))",
            observation=(
                f"Minimum success: {data.get('minimum_success_condition', '')}\n"
                f"Required outputs: {missing_preview}"
            )[:1200],
            outputs={
                "requirements": data,
                "required_outputs": data.get("required_outputs", []),
                "minimum_success_condition": data.get("minimum_success_condition", ""),
                "initial_search_hints": data.get("initial_search_hints", []),
            },
        )

    async def _llm_analyze(self, task: str) -> dict[str, Any] | None:
        assert self._llm is not None
        system = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                "You are the first-pass requirements analyst for a general-purpose agent. "
                "Do not solve the task yet. Do not use hidden ground truth or benchmark IDs. "
                "Convert the user's task into explicit, checkable success criteria that later research and verification must satisfy. "
                "Separate required outputs from search plan. Requirements are what a final answer must contain, not tool steps.",
            )
            if chunk
        )
        user_text = (
            "Analyze this task and return JSON only with this shape:\n"
            "{\n"
            "  \"task_type\": \"short generic type\",\n"
            "  \"required_outputs\": [\n"
            "    {\"id\": \"snake_case\", \"description\": \"what must be answered\", \"evidence_required\": \"what evidence would satisfy it\", \"optional\": false}\n"
            "  ],\n"
            "  \"minimum_success_condition\": \"what must be true before finalizing\",\n"
            "  \"common_failure_modes\": [\"specific ways an agent might answer incompletely or with wrong scope\"],\n"
            "  \"initial_search_hints\": [\"generic useful search hints, not hidden answers\"]\n"
            "}\n\n"
            "Rules:\n"
            "- Include every explicit user-requested field/qualifier.\n"
            "- If the user asks for evidence/citations, say which claims/relations need evidence.\n"
            "- If the task is multi-hop, represent each hop/edge requirement generically.\n"
            "- Do not add benchmark-specific or answer-specific knowledge not implied by the task.\n\n"
            f"Task:\n{task}"
        )
        try:
            text = await self._llm.complete(
                messages=[ChatMessage("system", system), ChatMessage("user", user_text)],
                tag="analyze_requirements",
                max_tokens=2000,
            )
        except Exception:
            return None
        data = extract_json(text or "")
        return data if isinstance(data, dict) else None

    @staticmethod
    def _fallback_requirements(task: str) -> dict[str, Any]:
        return {
            "task_type": "general_task",
            "required_outputs": [
                {
                    "id": "answer",
                    "description": "Answer every explicit part of the user task",
                    "evidence_required": "Use provided context or retrieved public evidence where the task asks for factual claims",
                    "optional": False,
                }
            ],
            "minimum_success_condition": "The final answer must address every explicit field, qualifier, and evidence/citation request in the task.",
            "common_failure_modes": ["Answering only part of the task", "Ignoring requested evidence or scope qualifiers"],
            "initial_search_hints": [task[:240]],
        }


def _normalise_requirements(data: Mapping[str, Any]) -> dict[str, Any]:
    required: list[dict[str, Any]] = []
    raw_required = data.get("required_outputs")
    if isinstance(raw_required, list):
        for i, item in enumerate(raw_required, 1):
            if not isinstance(item, Mapping):
                continue
            rid = item.get("id")
            desc = item.get("description")
            ev = item.get("evidence_required")
            required.append(
                {
                    "id": str(rid or f"requirement_{i}")[:80],
                    "description": str(desc or rid or f"Requirement {i}")[:500],
                    "evidence_required": str(ev or "Evidence sufficient to support this output")[:500],
                    "optional": bool(item.get("optional", False)),
                }
            )
    if not required:
        required = AnalyzeRequirementsTool._fallback_requirements("")["required_outputs"]
    def _str_list(value: Any, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(x)[:300] for x in value if str(x).strip()][:limit]
    return {
        "task_type": str(data.get("task_type") or "general_task")[:120],
        "required_outputs": required[:20],
        "minimum_success_condition": str(data.get("minimum_success_condition") or "All non-optional required outputs must be satisfied before finalizing.")[:1000],
        "common_failure_modes": _str_list(data.get("common_failure_modes"), 10),
        "initial_search_hints": _str_list(data.get("initial_search_hints"), 8),
    }


__all__ = ["AnalyzeRequirementsTool"]
