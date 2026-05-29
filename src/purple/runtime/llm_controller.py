"""LLM-backed controller.

Sends a tool catalog + transcript tail to the model each turn, parses a
single JSON action, and falls back to ``Surrender`` on any malformed output.
The orchestrator's loop is responsible for budgeting; this class is stateless
between calls.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Mapping

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt, load_skill
from ..schema import TaskRequest
from ..tools import extract_json
from .controller import Action, FinalAnswer, Surrender
from .rule_controller import RuleBasedController
from .tool import Tool, ToolCall
from .transcript import Transcript


_INFO_GATHERING_TOOLS = {
    "web_search",
    "web_fetch",
    "research_answer",
    "search_docs",
    "extract_answer",
}
_BLOCKING_STATUSES = {"missing", "weak", "contradicted"}


class LLMController:
    def __init__(
        self,
        llm: LLMClient,
        *,
        max_tokens: int = 320,
        transcript_tail: int = 6,
        fallback: RuleBasedController | None = None,
    ) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._transcript_tail = transcript_tail
        self._counter = itertools.count(1)
        self._fallback = fallback or RuleBasedController()

    async def next_action(
        self,
        request: TaskRequest,
        transcript: Transcript,
        tools: Mapping[str, Tool],
    ) -> Action:
        system_text = self._build_system(tools)
        user_text = self._build_user(request, transcript)
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", system_text),
                    ChatMessage("user", user_text),
                ],
                tag="controller",
                max_tokens=self._max_tokens,
            )
        except Exception:
            return await self._fallback.next_action(request, transcript, tools)

        data = extract_json(text or "")
        if not isinstance(data, dict):
            return await self._fallback.next_action(request, transcript, tools)

        action = (data.get("action") or "").strip().lower()
        if _must_continue_from_missing_requirements(transcript):
            if action in {"final", "stop"}:
                return await self._fallback.next_action(request, transcript, tools)
            if action == "call_tool":
                proposed = data.get("name", "")
                if not isinstance(proposed, str) or proposed not in _INFO_GATHERING_TOOLS:
                    return await self._fallback.next_action(request, transcript, tools)
                if proposed == "web_search" and _has_unfetched_url(transcript):
                    return await self._fallback.next_action(request, transcript, tools)
        if action == "final":
            answer = data.get("answer", "")
            return FinalAnswer(answer=answer if isinstance(answer, str) else "")
        if action == "stop":
            return Surrender(reason=str(data.get("reason", "")))
        if action == "call_tool":
            name = data.get("name", "")
            args = data.get("args", {})
            if not isinstance(name, str) or not name:
                return await self._fallback.next_action(request, transcript, tools)
            if not isinstance(args, dict):
                args = {}
            return ToolCall(
                id=f"ctl-{next(self._counter)}",
                name=name,
                args=dict(args),
            )
        return await self._fallback.next_action(request, transcript, tools)

    def _build_system(self, tools: Mapping[str, Tool]) -> str:
        catalog_lines = []
        for name, tool in tools.items():
            schema = ", ".join(
                f"{k}: {v}" for k, v in (tool.arg_schema or {}).items()
            )
            schema_text = f" args({schema})" if schema else ""
            description = (tool.description or "").strip().splitlines()[0:1]
            desc = description[0] if description else ""
            catalog_lines.append(f"- {name}{schema_text}: {desc}")
        catalog = "\n".join(catalog_lines)
        chunks = [
            load_prompt("system"),
            load_prompt("controller"),
            load_skill("tool_use"),
            "Available tools:\n" + catalog,
            (
                "Hard requirement-coverage rule: if notes.requirement_coverage or "
                "notes.missing_or_weak_points show any missing/weak/contradicted "
                "non-optional requirement, do not return final or stop. Select an "
                "information-gathering tool such as web_search, web_fetch, "
                "research_answer, search_docs, or extract_answer to address that "
                "specific missing requirement."
            ),
        ]
        return "\n\n".join(chunk for chunk in chunks if chunk).strip()

    def _build_user(self, request: TaskRequest, transcript: Transcript) -> str:
        notes = transcript.notes_view()
        payload: dict[str, Any] = {
            "user_prompt": request.prompt,
            "context_items": list(request.context),
            "attachments": [
                {"name": a.name, "mime_type": a.mime_type, "has_text": bool(a.text)}
                for a in request.attachments
            ],
            "transcript_tail": self._transcript_summary(transcript),
            "notes": _truncate_notes(notes),
            "controller_constraints": _controller_constraints(notes),
        }
        return (
            "Decide the next action as JSON.\n\nInputs (JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )

    def _transcript_summary(self, transcript: Transcript) -> list[dict[str, Any]]:
        turns = transcript.turns[-self._transcript_tail :]
        out: list[dict[str, Any]] = []
        for call, result in turns:
            out.append(
                {
                    "tool": call.name,
                    "args": dict(call.args),
                    "ok": result.ok,
                    "summary": result.summary,
                    "outputs": _shrink_outputs(result.outputs),
                    "error": result.error,
                }
            )
        return out


def _controller_constraints(notes: Mapping[str, Any]) -> dict[str, Any]:
    blocking = _blocking_requirement_points(notes)
    if not blocking:
        return {}
    return {
        "must_continue": True,
        "final_forbidden": True,
        "stop_forbidden": True,
        "allowed_tool_types": sorted(_INFO_GATHERING_TOOLS),
        "blocking_requirements": blocking[:8],
    }


def _must_continue_from_missing_requirements(transcript: Transcript) -> bool:
    return bool(_blocking_requirement_points(transcript.notes_view()))


def _blocking_requirement_points(notes: Mapping[str, Any]) -> list[str]:
    points: list[str] = []
    raw_coverage = notes.get("requirement_coverage")
    if isinstance(raw_coverage, list):
        for item in raw_coverage:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or "").lower()
            if status not in _BLOCKING_STATUSES:
                continue
            rid = str(item.get("requirement_id") or item.get("id") or "requirement")
            reason = str(item.get("reason") or status)
            points.append(f"{rid}: {reason}")
    return points


def _has_unfetched_url(transcript: Transcript) -> bool:
    fetched: set[str] = set()
    candidates: list[str] = []
    for _, result in transcript.turns:
        if not result.ok:
            continue
        for value in result.outputs.get("fetched_urls") or []:
            if isinstance(value, str):
                fetched.add(value)
        for value in result.outputs.get("source_urls") or []:
            if isinstance(value, str):
                candidates.append(value)
        for value in result.outputs.get("urls_detected") or []:
            if isinstance(value, str):
                candidates.append(value)
        results = result.outputs.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, Mapping) and isinstance(item.get("url"), str):
                    candidates.append(str(item["url"]))
    return any(url.startswith(("http://", "https://")) and url not in fetched for url in candidates)


def _truncate_notes(notes: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in notes.items():
        if isinstance(value, list):
            out[key] = [_shrink_text(v) for v in value[:5]]
        elif isinstance(value, str):
            out[key] = _shrink_text(value)
        else:
            out[key] = value
    return out


def _shrink_outputs(outputs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in outputs.items():
        if isinstance(value, list):
            out[key] = [_shrink_text(v) for v in value[:3]]
        elif isinstance(value, str):
            out[key] = _shrink_text(value)
        else:
            out[key] = value
    return out


def _shrink_text(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 320:
        return value[:317] + "..."
    return value


__all__ = ["LLMController"]
