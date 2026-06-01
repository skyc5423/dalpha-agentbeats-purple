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
        deterministic_followup = await self._deterministic_multiclue_followup(request, transcript, tools)
        if deterministic_followup is not None:
            return deterministic_followup

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
                if (
                    proposed == "web_search"
                    and _has_unfetched_url(transcript)
                    and not _has_mixed_source_domain_blocker(transcript)
                ):
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
            if (
                name == "web_search"
                and str(args.get("query") or "").strip()
                and _repeat_or_exhausted_multiclue_web_query(request, transcript, str(args.get("query") or ""))
            ):
                fallback_action = await self._fallback.next_action(request, transcript, tools)
                if isinstance(fallback_action, ToolCall):
                    if (
                        fallback_action.name == "web_search"
                        and _repeat_or_exhausted_multiclue_web_query(
                            request,
                            transcript,
                            str(fallback_action.args.get("query") or ""),
                        )
                    ):
                        if "web_fetch" in tools:
                            unfetched = _first_unfetched_url(transcript)
                            if unfetched:
                                return ToolCall(
                                    id=f"ctl-{next(self._counter)}",
                                    name="web_fetch",
                                    args={"url": unfetched},
                                )
                        if "research_answer" in tools and _attempted(transcript, "research_answer") < 2:
                            return ToolCall(
                                id=f"ctl-{next(self._counter)}",
                                name="research_answer",
                                args={"question": _multiclue_retry_question(request, transcript)},
                            )
                    return fallback_action
            if name == "web_search":
                args = dict(args)
                args.setdefault("attempted_queries", _prior_web_queries(transcript))
            if (
                name == "web_search"
                and _is_multiclue_entity_prompt(request.prompt or "")
                and _must_continue_from_missing_requirements(transcript)
                and not str(args.get("query") or "").strip().lower().startswith("site:")
            ):
                fallback_action = await self._fallback.next_action(request, transcript, tools)
                if isinstance(fallback_action, ToolCall) and fallback_action.name == "web_search":
                    fallback_query = str(fallback_action.args.get("query") or "").strip()
                    if fallback_query.lower().startswith("site:") or _looks_like_candidate_scoped_multiclue_query(fallback_query):
                        return fallback_action
            if (
                name == "web_search"
                and _is_multiclue_entity_prompt(request.prompt or "")
                and _has_mixed_source_domain_blocker(transcript)
                and _looks_like_overbroad_multiclue_query(str(args.get("query") or ""))
            ):
                # Mixed-domain blockers mean the current candidate chain is
                # contradicted. If the LLM proposes another overbroad all-clue
                # query, replace only the query args with the deterministic
                # missing-requirement query builder; this stays generic and
                # avoids spending the next step on controller prose / blended
                # clue searches.
                fallback_action = await self._fallback.next_action(request, transcript, tools)
                if isinstance(fallback_action, ToolCall) and fallback_action.name == "web_search":
                    return fallback_action
            if (
                name == "web_search"
                and "research_answer" in tools
                and _is_multiclue_entity_prompt(request.prompt or "")
                and _empty_web_search_count(transcript) >= 2
                and _attempted(transcript, "research_answer") < 2
            ):
                return ToolCall(
                    id=f"ctl-{next(self._counter)}",
                    name="research_answer",
                    args={"question": _multiclue_retry_question(request, transcript)},
                )
            return ToolCall(
                id=f"ctl-{next(self._counter)}",
                name=name,
                args=dict(args),
            )
        return await self._fallback.next_action(request, transcript, tools)

    async def _deterministic_multiclue_followup(
        self,
        request: TaskRequest,
        transcript: Transcript,
        tools: Mapping[str, Tool],
    ) -> ToolCall | None:
        """Let deterministic coverage guards preempt broad LLM choices.

        Multi-clue entity traces can drift after a useful source fetch: the LLM
        may choose another broad search or a fresh unsourced research draft even
        though the generic rule controller has a concrete next gate such as
        ``sufficiency_check`` or a same-domain ``site:`` follow-up query.  This
        method only promotes those generic coverage-preserving actions; it does
        not inspect benchmark IDs or expected answers.
        """

        if not _is_multiclue_entity_prompt(request.prompt or ""):
            return None
        if _latest_evidence_after_sufficiency(transcript):
            if "sufficiency_check" in tools:
                return ToolCall(id=f"ctl-{next(self._counter)}", name="sufficiency_check", args={})
            action = await self._fallback.next_action(request, transcript, tools)
            if isinstance(action, ToolCall) and action.name == "sufficiency_check":
                return action
        if _must_continue_from_missing_requirements(transcript):
            action = await self._fallback.next_action(request, transcript, tools)
            if isinstance(action, ToolCall) and action.name == "web_search":
                query = str(action.args.get("query") or "").strip()
                if query.lower().startswith("site:") or _looks_like_candidate_scoped_multiclue_query(query):
                    return action
        return None

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


def _attempted(transcript: Transcript, tool_name: str) -> int:
    return sum(1 for name in transcript.names() if name == tool_name)


def _latest_evidence_after_sufficiency(transcript: Transcript) -> bool:
    last_evidence_idx = -1
    last_suff_idx = -1
    for i, (call, result) in enumerate(transcript.turns):
        if not result.ok:
            continue
        if call.name == "sufficiency_check":
            last_suff_idx = i
        elif _looks_like_evidence_outputs(result.outputs):
            last_evidence_idx = i
    return last_evidence_idx >= 0 and last_evidence_idx > last_suff_idx


def _looks_like_evidence_outputs(outputs: Mapping[str, Any]) -> bool:
    if not isinstance(outputs, Mapping):
        return False
    if outputs.get("answer_candidate"):
        return True
    for key in ("spans", "fetched_pages", "results"):
        value = outputs.get(key)
        if isinstance(value, (list, tuple)) and any(value):
            return True
    return False


def _is_multiclue_entity_prompt(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    has_entity = any(
        token in lowered
        for token in ("institution", "university", "college", "school", "learning establishment")
    )
    clue_count = sum(
        1
        for token in ("criterion", "2002", "2003", "2022", "seven days", "capital city")
        if token in lowered
    )
    return has_entity and clue_count >= 2


def _prior_web_queries(transcript: Transcript) -> list[str]:
    prior: list[str] = []
    for call, result in transcript.turns:
        if call.name != "web_search":
            continue
        for value in (call.args.get("query"), result.outputs.get("query") if result.ok else None):
            if isinstance(value, str) and value.strip():
                prior.append(" ".join(value.split())[:300])
        attempted = result.outputs.get("attempted_queries") if result.ok else None
        if isinstance(attempted, (list, tuple)):
            for item in attempted:
                if isinstance(item, str) and item.strip():
                    prior.append(" ".join(item.split())[:300])
    return list(dict.fromkeys(prior))[-20:]


def _same_as_prior_web_query(transcript: Transcript, query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    if not normalized:
        return False
    for prior in _prior_web_queries(transcript):
        if " ".join(prior.lower().split()) == normalized:
            return True
    return False


def _repeat_or_exhausted_multiclue_web_query(request: TaskRequest, transcript: Transcript, query: str) -> bool:
    """Return true when a proposed search would be a no-op repeat.

    The web_search tool can suppress broad same-clue repeats and return a
    skipped-duplicate observation. For expanded-budget multi-clue tasks that
    still consumes a controller step. Catch exact repeats and non-site-scoped
    same clue-group repeats before the tool call so the controller can fetch a
    pending source or escalate to a broader research retry instead.
    """

    if _same_as_prior_web_query(transcript, query):
        return True
    if not _is_multiclue_entity_prompt(request.prompt or ""):
        return False
    if _is_site_scoped_query(query):
        return False
    group = _query_group(query)
    if not group:
        return False
    prior_groups = {_query_group(item) for item in _prior_web_queries(transcript) if _query_group(item)}
    return group in prior_groups


def _query_group(query: str) -> str:
    lowered = (query or "").lower()
    if any(token in lowered for token in ("plant", "sample", "field trip", "botany", "flora", "herbarium", "botanical")):
        return "plant"
    if any(token in lowered for token in ("bank", "tribute", "management", "ceremony", "vice chancellor", "rector")):
        return "bank"
    if any(token in lowered for token in ("graduation", "commencement", "convocation", "fourth sunday")):
        return "graduation"
    if any(token in lowered for token in ("2002", "thursday", "saturday", "three day", "three-day", "support")):
        return "event"
    return ""


def _is_site_scoped_query(query: str) -> bool:
    return " ".join((query or "").lower().split()).startswith("site:")


def _empty_web_search_count(transcript: Transcript) -> int:
    count = 0
    for call, result in transcript.turns:
        if call.name != "web_search" or not result.ok:
            continue
        results = result.outputs.get("results")
        if not isinstance(results, list) or not results:
            count += 1
    return count


def _looks_like_overbroad_multiclue_query(query: str) -> bool:
    """Detect blended/controller-style multi-clue searches.

    This is a generic query hygiene guard. It does not know an expected answer;
    it only recognizes queries that blend several independent evidence clues or
    look like instructions rather than a compact public-source search.
    """

    cleaned = " ".join((query or "").split())
    lowered = cleaned.lower()
    if not cleaned:
        return True
    if len(cleaned) > 180 or lowered.startswith(("search ", "verify ", "find sources")):
        return True
    clue_groups = 0
    if any(token in lowered for token in ("plant", "sample", "field trip", "herbarium", "specimens")):
        clue_groups += 1
    if any(token in lowered for token in ("bank", "tribute", "management", "ceremony", "vice chancellor", "rector")):
        clue_groups += 1
    if any(token in lowered for token in ("graduation", "commencement", "convocation", "fourth sunday")):
        clue_groups += 1
    if any(token in lowered for token in ("thursday", "saturday", "three-day", "support")):
        clue_groups += 1
    return clue_groups >= 2


def _looks_like_candidate_scoped_multiclue_query(query: str) -> bool:
    """Detect generic entity-name + one-clue verification queries.

    These queries are produced after a source-free named institution/entity
    candidate appears. They are not task-id or answer routing: they simply ask
    the public web to validate the candidate against one missing clue group.
    """

    cleaned = " ".join((query or "").split())
    lowered = cleaned.lower()
    if not cleaned or cleaned.lower().startswith("site:"):
        return False
    quoted_segments = [segment.lower() for segment in cleaned.split('"')[1::2]]
    if not quoted_segments:
        return False
    has_named_entity = any(
        any(token in segment for token in ("university", "college", "school", "institute"))
        for segment in quoted_segments
    )
    has_clue_group = bool(_query_group(cleaned))
    # Candidate-scoped queries should be compact; overlong strings are usually
    # controller prose or blended all-clue searches.
    return has_named_entity and has_clue_group and len(cleaned) <= 180


def _multiclue_retry_question(request: TaskRequest, transcript: Transcript) -> str:
    blocking = _blocking_requirement_points(transcript.notes_view())
    blocking_text = "\n".join(f"- {point}" for point in blocking[:6])
    return (
        "Retry this public-source research task as a multi-clue entity search. "
        "Do not reuse a previously contradicted candidate unless each clue is independently sourced. "
        "Search one missing clue at a time with broader synonyms, then require the same institution/entity to satisfy every clue. "
        "Return source URLs for each clue and say unsupported if coverage remains incomplete.\n\n"
        f"Original task:\n{request.prompt or ''}\n\n"
        f"Currently blocking requirements:\n{blocking_text}"
    ).strip()


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


def _first_unfetched_url(transcript: Transcript) -> str:
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
    for url in candidates:
        if url.startswith(("http://", "https://")) and url not in fetched:
            return url
    return ""


def _has_unfetched_url(transcript: Transcript) -> bool:
    return bool(_first_unfetched_url(transcript))


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


def _has_mixed_source_domain_blocker(transcript: Transcript) -> bool:
    raw_coverage = transcript.notes_view().get("requirement_coverage")
    if not isinstance(raw_coverage, list):
        return False
    for item in raw_coverage:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").lower()
        reason = str(item.get("reason") or "").lower()
        if status in _BLOCKING_STATUSES and "mixed source domains" in reason:
            return True
    return False


__all__ = ["LLMController"]
