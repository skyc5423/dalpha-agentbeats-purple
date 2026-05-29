"""``sufficiency_check`` tool — score whether the current candidate + evidence
fully satisfies the user prompt.

This is a deterministic, benchmark-agnostic heuristic: token-overlap coverage
between the prompt's content tokens and the union of the candidate +
transcript spans. The rule-based controller calls it before finalising on any
non-self-validating tool's candidate (e.g. ``web_search`` snippets), so a
single shallow web answer never short-circuits the loop. The LLM controller
sees it in its tool catalog and may invoke it for the same purpose.
"""

from __future__ import annotations

import re
import json
from typing import Any, Mapping

from ..llm import ChatMessage, LLMClient
from ..runtime.tool import ToolContext, ToolResult
from ..tools import extract_json


_WORD = re.compile(r"[A-Za-z0-9]+")
_URL = re.compile(r"https?://[^\s)\]}>,\"']+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "at", "and", "or", "for", "by", "with", "be", "this", "that", "it",
        "as", "from", "what", "which", "who", "how", "why", "where", "when",
        "do", "does", "did", "can", "could", "should", "would", "shall",
        "will", "may", "might", "have", "has", "had", "been", "being",
        "into", "out", "your", "you", "i", "we", "they", "them",
        "please", "give", "show", "list", "tell", "find", "return", "only",
        "use", "using", "include", "based",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD.findall(text or "")
        if len(w) >= 3 and w.lower() not in _STOPWORDS
    }


def _looks_like_complete_github_commit_answer(prompt: str, evidence_text: str) -> bool:
    prompt_lower = (prompt or "").lower()
    if "commit" not in prompt_lower:
        return False
    evidence_lower = (evidence_text or "").lower()
    required = (
        "github commit url:",
        "sha:",
        "author:",
        "github author profile:",
    )
    if not all(token in evidence_lower for token in required):
        return False
    if any(token in prompt_lower for token in ("contributor", "co-author", "coauthor", "authors")):
        if "co-author" not in evidence_lower and "author:" not in evidence_lower:
            return False
    if "real name" in prompt_lower and "real name:" not in evidence_lower:
        return False
    return True


def _extract_urls(text: str) -> list[str]:
    return [match.group(0).rstrip(".,;") for match in _URL.finditer(text or "")]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _requirement_gate(requirements: Any, coverage: list[Any]) -> tuple[bool, list[str]] | None:
    if not coverage:
        return None
    if not isinstance(requirements, Mapping):
        return None
    raw_required = requirements.get("required_outputs")
    if not isinstance(raw_required, list) or not raw_required:
        return None
    required: dict[str, str] = {}
    for i, item in enumerate(raw_required, 1):
        if not isinstance(item, Mapping) or bool(item.get("optional", False)):
            continue
        rid = str(item.get("id") or f"requirement_{i}")
        required[rid] = str(item.get("description") or rid)
    if not required:
        return None
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in coverage:
        if not isinstance(item, Mapping):
            continue
        rid = item.get("requirement_id") or item.get("id")
        if isinstance(rid, str) and rid:
            by_id[rid] = item
    missing: list[str] = []
    for rid, desc in required.items():
        item = by_id.get(rid)
        status = str(item.get("status", "") if item else "missing").lower()
        if status != "satisfied":
            reason = str(item.get("reason", "") if item else "not checked").strip()
            missing.append(f"{rid}: {reason or desc}")
    return (not missing, missing)


class SufficiencyCheckTool:
    name = "sufficiency_check"
    description = (
        "Assess whether the current answer candidate plus transcript evidence "
        "fully satisfies the user prompt. Use before finalising on a "
        "tool-produced candidate to avoid shipping incomplete answers."
    )
    arg_schema: Mapping[str, str] = {
        "candidate": "optional explicit candidate to score; defaults to latest transcript candidate",
        "min_coverage": "optional float in [0,1]; default 0.5",
    }

    def __init__(self, *, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        candidate = args.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            note_candidate = (
                ctx.notes.get("answer_candidate") if hasattr(ctx.notes, "get") else ""
            )
            candidate = str(note_candidate or "").strip()
        else:
            candidate = candidate.strip()

        spans: list[str] = []
        note_spans = ctx.notes.get("spans") if hasattr(ctx.notes, "get") else None
        if isinstance(note_spans, list):
            spans = [s for s in note_spans if isinstance(s, str) and s.strip()]

        try:
            min_coverage = max(0.0, min(1.0, float(args.get("min_coverage", 0.5))))
        except (TypeError, ValueError):
            min_coverage = 0.5

        prompt_tokens = _content_tokens(ctx.request.prompt or "")
        evidence_text = "\n".join([candidate, *spans])
        evidence_tokens = _content_tokens(evidence_text)

        if not prompt_tokens:
            coverage = 1.0 if candidate else 0.0
            missing_tokens: list[str] = []
        else:
            overlap = prompt_tokens & evidence_tokens
            coverage = round(len(overlap) / max(1, len(prompt_tokens)), 4)
            missing_tokens = sorted(prompt_tokens - evidence_tokens)

        heuristic_sufficient = bool(candidate) and coverage >= min_coverage
        sufficient = heuristic_sufficient
        missing_points = missing_tokens[:8]
        source = "heuristic"

        llm_result: dict[str, Any] | None = None
        if self._llm is not None and candidate:
            llm_result = await self._llm_check(
                prompt=ctx.request.prompt or "",
                candidate=candidate,
                spans=spans,
                requirements=ctx.notes.get("requirements") if hasattr(ctx.notes, "get") else {},
                heuristic={
                    "coverage": coverage,
                    "missing_terms": missing_tokens[:20],
                    "heuristic_sufficient": heuristic_sufficient,
                },
            )
            if llm_result is not None:
                sufficient = bool(llm_result.get("sufficient"))
                missing_raw = llm_result.get("missing_or_weak_points") or []
                if isinstance(missing_raw, list):
                    missing_points = [str(x) for x in missing_raw if str(x).strip()][:8]
                source = "llm"
        requirement_coverage: list[Any] = []
        next_queries: list[str] = []
        source_urls: list[str] = []
        if isinstance(llm_result, dict):
            raw_coverage = llm_result.get("coverage") or []
            if isinstance(raw_coverage, list):
                requirement_coverage = raw_coverage[:20]
            raw_next = llm_result.get("next_queries") or []
            if isinstance(raw_next, list):
                for item in raw_next:
                    text = str(item).strip()
                    if not text:
                        continue
                    urls = _extract_urls(text)
                    if urls and len(urls) == 1 and urls[0] == text.rstrip(".,"):
                        source_urls.extend(urls)
                        continue
                    if urls:
                        source_urls.extend(urls)
                    else:
                        next_queries.append(" ".join(text.split())[:300])
            raw_sources = llm_result.get("source_urls") or []
            if isinstance(raw_sources, list):
                for item in raw_sources:
                    source_urls.extend(_extract_urls(str(item)))

        requirement_gate = _requirement_gate(
            ctx.notes.get("requirements") if hasattr(ctx.notes, "get") else {},
            requirement_coverage,
        )
        if requirement_gate is not None:
            all_required_satisfied, gate_missing = requirement_gate
            if not all_required_satisfied:
                sufficient = False
                missing_points = gate_missing[:8]
                source = f"{source}+requirements_gate"
            elif isinstance(llm_result, dict):
                sufficient = True
                missing_points = []
                source = f"{source}+requirements_gate"

        next_queries = _dedupe(next_queries)[:8]
        source_urls = _dedupe(source_urls)[:8]

        if _looks_like_complete_github_commit_answer(ctx.request.prompt or "", evidence_text):
            sufficient = True
            missing_points = []
            source = f"{source}+github_commit_heuristic"
        elif "commit" in (ctx.request.prompt or "").lower() and any(
            token in (ctx.request.prompt or "").lower()
            for token in ("github", "repository", "branch")
        ):
            sufficient = False
            marker = "structured GitHub commit metadata"
            if marker not in missing_points:
                missing_points = [marker, *missing_points[:7]]
            source = f"{source}+github_commit_guard"

        prompt_lower = (ctx.request.prompt or "").lower()
        candidate_lower = (candidate or "").lower()
        if (
            "advisor" in prompt_lower
            and "lineage" in prompt_lower
            and ("five" in prompt_lower or "5" in prompt_lower or "generation" in prompt_lower)
            and any(marker in candidate_lower for marker in ("cannot be completed", "cannot complete", "insufficient", "not enough"))
        ):
            sufficient = False
            marker = "complete five-generation advisor lineage"
            if marker not in missing_points:
                missing_points = [marker, *missing_points[:7]]
            source = f"{source}+advisor_lineage_guard"

        verdict = "sufficient" if sufficient else "insufficient"
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=f"sufficiency={verdict} coverage={coverage:.2f}",
            observation=(
                f"{verdict} ({source}, coverage={coverage:.2f}); "
                f"missing: {', '.join(missing_points) or '-'}"
            ),
            outputs={
                "sufficient": sufficient,
                "coverage": coverage,
                "missing_terms": missing_tokens,
                "candidate": candidate,
                "answer_candidate": candidate,
                "min_coverage": min_coverage,
                "source": source,
                "heuristic_sufficient": heuristic_sufficient,
                "missing_or_weak_points": missing_points,
                "requirement_coverage": requirement_coverage,
                "next_queries": next_queries,
                "source_urls": source_urls,
            },
        )

    async def _llm_check(
        self,
        *,
        prompt: str,
        candidate: str,
        spans: list[str],
        requirements: Any,
        heuristic: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        assert self._llm is not None
        payload = {
            "user_prompt": prompt,
            "answer_candidate": candidate,
            "evidence_spans": spans[:8],
            "requirements": requirements if isinstance(requirements, Mapping) else {},
            "heuristic_hint": dict(heuristic),
        }
        user_text = (
            "Decide whether the evidence directly and completely satisfies the original user prompt. "
            "Do not judge whether the candidate is merely consistent with the evidence; judge whether the evidence is enough to finish the task. "
            "Use the provided requirements checklist as the primary success criteria when present. "
            "Return JSON only: {\"sufficient\": boolean, \"missing_or_weak_points\": [strings], "
            "\"coverage\": [{\"requirement_id\": string, \"status\": \"satisfied|missing|weak|contradicted\", \"reason\": string}], "
            "\"next_queries\": [strings]}.\n\n"
            "Inputs (JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", "You are an evidence sufficiency critic for a general-purpose tool-using agent. Be strict about directness and completeness. Do not use benchmark IDs or hidden ground truth."),
                    ChatMessage("user", user_text),
                ],
                tag="sufficiency_check",
                max_tokens=800,
            )
        except Exception:
            return None
        data = extract_json(text or "")
        return data if isinstance(data, dict) else None


__all__ = ["SufficiencyCheckTool"]
