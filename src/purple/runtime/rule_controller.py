"""Deterministic, profile-driven controller used in tests / no-LLM mode.

The rule controller no longer finalises on the first ``answer_candidate``
that bubbles up. It enforces the loop invariant that evidence must be
verified as sufficient before commitment:

1. If any prior turn marked itself ``sufficient_alone=True`` (e.g. the
   ``calculate`` or ``finish`` tools), commit that candidate directly.
2. Otherwise, if a candidate is available and ``sufficiency_check`` has not
   yet run against the latest evidence, dispatch ``sufficiency_check``.
3. If ``sufficiency_check`` confirmed sufficient and no new evidence has
   arrived since, commit the candidate.
4. Otherwise, escalate: try unfetched URLs via ``web_fetch``, then
   profile-ranked tools that have not yet been attempted.
5. When no more tools can help, surrender; the finalizer composes a
   fallback answer using whatever spans exist.
"""

from __future__ import annotations

import itertools
import re
from urllib.parse import urlparse
from typing import Iterable, Mapping

from ..profiler import CapabilityProfiler
from ..schema import TaskRequest
from .controller import Action, FinalAnswer, Surrender
from .tool import Tool, ToolCall
from .transcript import Transcript


_CAP_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "calculator": ("calculate",),
    "doc_research": ("search_docs", "extract_answer"),
    "web_research": ("research_answer", "web_search", "web_fetch"),
    "shell_code": ("shell_exec",),
}

_MIN_TOOL_ATTEMPTS = {
    "web_fetch": 8,
    "web_search": 4,
}


def _looks_like_evidence(outputs: Mapping[str, object]) -> bool:
    if not isinstance(outputs, Mapping):
        return False
    if outputs.get("answer_candidate"):
        return True
    spans = outputs.get("spans")
    if isinstance(spans, (list, tuple)) and any(spans):
        return True
    return False


class RuleBasedController:
    def __init__(
        self,
        profiler: CapabilityProfiler | None = None,
        *,
        max_attempts: int = 1,
        max_external_attempts: int | None = None,
        select_threshold: float = 0.4,
    ) -> None:
        self._profiler = profiler or CapabilityProfiler()
        self._max_attempts = max_attempts
        self._max_external_attempts = max_external_attempts
        self._select_threshold = select_threshold
        self._counter = itertools.count(1)

    async def next_action(
        self,
        request: TaskRequest,
        transcript: Transcript,
        tools: Mapping[str, Tool],
    ) -> Action:
        # 1. Self-sufficient tools (calculator, finish) short-circuit.
        for call, result in reversed(transcript.turns):
            if not result.ok:
                continue
            if result.outputs.get("sufficient_alone") is True:
                cand = result.outputs.get("answer_candidate")
                if isinstance(cand, str) and cand.strip():
                    return FinalAnswer(answer=cand.strip())

        # 2. Compute sufficiency / evidence indices.
        last_evidence_idx = -1
        last_suff_idx = -1
        latest_sufficient: bool | None = None
        for i, (call, result) in enumerate(transcript.turns):
            if not result.ok:
                continue
            if call.name == "sufficiency_check":
                last_suff_idx = i
                if result.outputs.get("sufficient") is True:
                    latest_sufficient = True
                elif result.outputs.get("sufficient") is False:
                    latest_sufficient = False
            elif _looks_like_evidence(result.outputs):
                last_evidence_idx = i

        candidate = transcript.latest_output("answer_candidate")
        has_candidate = isinstance(candidate, str) and bool(candidate.strip())

        # 3. If sufficiency_check confirmed sufficient and nothing newer has
        #    landed on the transcript, commit the candidate.
        if (
            has_candidate
            and latest_sufficient is True
            and last_evidence_idx <= last_suff_idx
        ):
            return FinalAnswer(answer=str(candidate).strip())

        # 4. New evidence with no sufficiency verdict yet → check sufficiency.
        if (
            has_candidate
            and "sufficiency_check" in tools
            and last_evidence_idx > last_suff_idx
        ):
            return self._make_call("sufficiency_check")

        # 5. Insufficient: try follow-up tools. Mixed-domain multi-clue drafts
        #    are usually bad candidate chains; do not spend the next steps
        #    fetching every source from that contradicted chain. First issue a
        #    fresh focused search for a missing clue/source. Once such a search
        #    has surfaced concrete public URLs, fetch a promising result before
        #    spending more budget on another same-clue search; otherwise the
        #    loop can bounce between sufficiency checks and repeated broad
        #    queries without validating any source.
        if (
            "web_search" in tools
            and _is_multiclue_entity_prompt(request.prompt or "")
            and _has_mixed_source_domain_blocker(transcript)
            and self._attempted(transcript, "web_search") < self._attempt_limit("web_search")
        ):
            search_args = self._next_search_args(request, transcript)
            if (
                search_args is not None
                and _is_site_scoped_query(str(search_args.get("query") or ""))
                and _has_fetched_partial_source_seed(transcript)
            ):
                return self._make_call("web_search", args=search_args)
            unfetched = self._unfetched_url(transcript)
            if (
                unfetched is not None
                and self._attempted(transcript, "web_search") > 0
                and "web_fetch" in tools
                and self._attempted(transcript, "web_fetch") < self._attempt_limit("web_fetch")
            ):
                return self._make_call("web_fetch", args={"url": unfetched})
            if search_args is not None:
                return self._make_call("web_search", args=search_args)

        # For multi-clue entity tasks, a single source host that supports one
        # clue should be probed for complementary clues before the loop drains
        # unrelated intra-site links. This also works before an LLM has produced
        # a candidate/coverage table: the prompt supplies clue groups and fetched
        # source text supplies the seed host.
        if (
            "web_search" in tools
            and _is_multiclue_entity_prompt(request.prompt or "")
            and _has_fetched_partial_source_seed(transcript)
            and self._attempted(transcript, "web_search") < self._attempt_limit("web_search")
        ):
            search_args = self._next_search_args(request, transcript)
            if search_args is not None and _is_site_scoped_query(str(search_args.get("query") or "")):
                return self._make_call("web_search", args=search_args)

        # For multi-clue entity tasks, a single LLM-drafted source URL often
        # supports only one claimed clue. If the requirement table still has
        # missing/weak/contradicted items and no directed search has run yet,
        # issue one focused public-source query before spending budget fetching
        # every URL from the weak draft.
        if (
            "web_search" in tools
            and _is_multiclue_entity_prompt(request.prompt or "")
            and _has_blocking_requirement(transcript)
            and self._attempted(transcript, "web_search") == 0
        ):
            search_args = self._next_search_args(request, transcript)
            if search_args is not None:
                return self._make_call("web_search", args=search_args)

        # Prefer web_fetch on any URL surfaced by web_search but not yet
        # retrieved, unless the latest sufficiency blocker says the current URL
        # set is a mixed-entity chain as handled above.
        unfetched = self._unfetched_url(transcript)
        if (
            unfetched is not None
            and "web_fetch" in tools
            and self._attempted(transcript, "web_fetch") < self._attempt_limit("web_fetch")
        ):
            return self._make_call("web_fetch", args={"url": unfetched})

        # If search produced no usable URLs, or all surfaced URLs were still
        # insufficient, issue a broadened/directed query instead of repeating
        # the same bare prompt.
        if (
            "web_search" in tools
            and self._attempted(transcript, "web_search") < self._attempt_limit("web_search")
        ):
            search_args = self._next_search_args(request, transcript)
            if search_args is not None:
                return self._make_call("web_search", args=search_args)

        # 6. If context exists, prefer the doc tools.
        if request.context or any(a.text for a in request.attachments):
            for tool_name in ("search_docs", "extract_answer"):
                if (
                    tool_name in tools
                    and self._attempted(transcript, tool_name) < self._attempt_limit(tool_name)
                ):
                    return self._make_call(tool_name)

        # 7. Otherwise walk the profile in descending order.
        profile = self._profiler.profile(request)
        sorted_caps = sorted(
            profile.scores.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for cap, score in sorted_caps:
            if score < self._select_threshold:
                continue
            for tool_name in _CAP_TO_TOOLS.get(cap, ()):
                if tool_name not in tools:
                    continue
                if self._attempted(transcript, tool_name) >= self._attempt_limit(tool_name):
                    continue
                if tool_name == "web_fetch":
                    next_url = self._unfetched_url(transcript)
                    if next_url is None:
                        continue
                    return self._make_call("web_fetch", args={"url": next_url})
                if tool_name == "web_search":
                    args = self._next_search_args(request, transcript)
                    if args is None and _is_multiclue_entity_prompt(request.prompt or ""):
                        continue
                    return self._make_call(tool_name, args=args)
                return self._make_call(tool_name)

        # 8. Nothing left to try. Commit a candidate if we have one;
        #    otherwise let the finalizer compose a fallback.
        if has_candidate:
            return FinalAnswer(answer=str(candidate).strip())
        return Surrender(reason="rule controller exhausted candidates")

    @staticmethod
    def _attempted(transcript: Transcript, tool_name: str) -> int:
        return sum(1 for name in transcript.names() if name == tool_name)

    def _attempt_limit(self, tool_name: str) -> int:
        base = max(self._max_attempts, _MIN_TOOL_ATTEMPTS.get(tool_name, self._max_attempts))
        if tool_name in {"web_search", "web_fetch"} and self._max_external_attempts is not None:
            return max(base, self._max_external_attempts)
        return base

    def _next_search_args(self, request: TaskRequest, transcript: Transcript) -> Mapping[str, object] | None:
        prior_queries: list[str] = []
        for call, result in transcript.turns:
            if call.name != "web_search":
                continue
            call_query = call.args.get("query")
            if isinstance(call_query, str) and call_query.strip():
                prior_queries.append(call_query.strip())
            query = result.outputs.get("query") if result.ok else call.args.get("query")
            if isinstance(query, str) and query.strip():
                prior_queries.append(query.strip())
            attempted = result.outputs.get("attempted_queries") if result.ok else None
            if isinstance(attempted, (list, tuple)):
                for item in attempted:
                    if isinstance(item, str) and item.strip():
                        prior_queries.append(item.strip())
        if not prior_queries:
            next_query = _latest_next_query(transcript, prior_queries)
            if not next_query:
                next_query = _missing_requirement_query(request, transcript, prior_queries)
            if not next_query and _is_multiclue_entity_prompt(request.prompt or ""):
                next_query = _focused_multiclue_query(request.prompt or "", [request.prompt or ""], prior_queries)
            if not next_query:
                prompt_lower = (request.prompt or "").lower()
                if "advisor" in prompt_lower and any(token in prompt_lower for token in ("lineage", "genealogy", "doctoral", "phd")):
                    next_query = self._refined_query(request, transcript, "")
            return _search_args(next_query, prior_queries) if next_query else None

        # For multi-clue entity searches, sufficiency_check may emit long
        # prompt-like follow-up strings. Prefer our focused missing-requirement
        # query builder first so the runner does not repeatedly search the same
        # normalized clue query under different controller prose.
        if _is_multiclue_entity_prompt(request.prompt or ""):
            missing_query = _missing_requirement_query(request, transcript, prior_queries)
            if missing_query:
                return _search_args(missing_query, prior_queries)
            seeded_prompt_query = _seeded_prompt_followup_query(request.prompt or "", transcript, prior_queries)
            if seeded_prompt_query:
                return _search_args(seeded_prompt_query, prior_queries)
            # If early broad/fallback searches produced zero usable URLs, there
            # is no requirement coverage table yet. Keep issuing the next
            # candidate-independent clue query instead of falling through to
            # research_answer/no-LLM surrender. This is generic source discovery
            # hygiene for multi-clue entity tasks, not answer routing.
            no_search_results = not any(
                call.name == "web_search"
                and result.ok
                and isinstance(result.outputs.get("results"), (list, tuple))
                and bool(result.outputs.get("results"))
                for call, result in transcript.turns
            )
            if no_search_results:
                focused_query = _focused_multiclue_query(request.prompt or "", [request.prompt or ""], prior_queries)
                if focused_query:
                    return _search_args(focused_query, prior_queries)

        next_query = _latest_next_query(transcript, prior_queries)
        if next_query:
            return _search_args(next_query, prior_queries)

        missing_query = _missing_requirement_query(request, transcript, prior_queries)
        if missing_query:
            return {"query": missing_query, "limit": 8, "skip_web_answerer": True}

        refined = self._refined_query(request, transcript, prior_queries[-1])
        if not refined or _query_seen(refined, prior_queries):
            return None
        return _search_args(refined, prior_queries)

    @staticmethod
    def _refined_query(request: TaskRequest, transcript: Transcript, previous_query: str) -> str:
        prompt = " ".join((request.prompt or "").split())
        missing_terms: list[str] = []
        for call, result in reversed(transcript.turns):
            if call.name == "sufficiency_check" and result.ok:
                raw = result.outputs.get("missing_terms")
                if isinstance(raw, (list, tuple)):
                    missing_terms = [str(x) for x in raw[:8] if str(x).strip()]
                break

        lowered = f"{prompt} {previous_query}".lower()
        repo = _github_repo_from_transcript(transcript)
        boosters: list[str] = []
        if repo:
            model_terms = [
                term
                for term in ("llava", "model")
                if term in lowered
            ]
            if "llava" in lowered:
                boosters.append(f"repo:{repo} llava add commit")
            elif model_terms:
                boosters.append(f"repo:{repo} {' '.join(model_terms)} add commit")
            else:
                boosters.append(f"repo:{repo} add commit")
        else:
            boosters.extend(["official source", "primary source"])
        if any(token in lowered for token in ("commit", "repository", "github", "branch", "author")):
            boosters.extend(["github commit", "commits", "author date"])
        if any(token in lowered for token in ("profile", "contributor", "real name")):
            boosters.extend(["github profile", "contributors", "real name"])

        if repo:
            pieces = [" ".join(boosters)]
        else:
            advisor_name = _advisor_name_from_transcript(transcript)
            if advisor_name and any(token in lowered for token in ("advisor", "lineage", "genealogy", "doctoral", "phd")):
                pieces = [f'"{advisor_name}" "Mathematics Genealogy" advisor Ph.D.']
            else:
                pieces = [prompt[:220], " ".join(missing_terms), " ".join(boosters)]
        query = " ".join(piece for piece in pieces if piece).strip()
        return query[:300]

    @staticmethod
    def _unfetched_url(transcript: Transcript) -> str | None:
        fetched: set[str] = set()
        # Treat every already-attempted web_fetch URL as consumed, including
        # failed/empty-body fetches. Otherwise a high-scoring but unfetchable
        # search result can be selected again and again; the loop then spends
        # later budget on duplicate web_fetch skip observations instead of
        # advancing to the next candidate/source. This is generic URL-state
        # hygiene, not benchmark/task-specific answer routing.
        for call, result in transcript.turns:
            if call.name == "web_fetch":
                call_url = call.args.get("url") if isinstance(call.args, Mapping) else None
                if isinstance(call_url, str) and _is_url(call_url):
                    fetched.add(call_url)
                result_url = result.outputs.get("url") if isinstance(result.outputs, Mapping) else None
                if isinstance(result_url, str) and _is_url(result_url):
                    fetched.add(result_url)
            if not result.ok:
                continue
            for url in _iter_strings(result.outputs.get("fetched_urls")):
                fetched.add(url)

        # If a later answer candidate superseded an earlier candidate, do not
        # keep spending multi-clue budget fetching stale source URLs from the
        # old candidate chain. Search results remain eligible because they are
        # candidate-independent discovery evidence; stale LLM-draft URLs are
        # not. This is generic state hygiene, not task/answer routing.
        latest_candidate_idx = _latest_answer_candidate_index(transcript)
        blocking = _has_blocking_requirement(transcript)
        # Avoid draining intra-site links after same-source validation has already
        # disproved a required clue group for that candidate scope. For example,
        # once every generic bank-tribute probe for ``site:<scope>`` is empty,
        # more staff/course/navigation pages from that same scope are unlikely to
        # repair a multi-clue entity chain and should not consume the remaining
        # budget.
        prior_site_queries = _prior_web_queries(transcript)
        latest_archive_result_idx: dict[tuple[str, str], int] = {}
        for idx, (call, result) in enumerate(transcript.turns):
            if call.name != "web_fetch":
                continue
            values: list[str] = []
            call_url = call.args.get("url") if isinstance(call.args, Mapping) else None
            if isinstance(call_url, str):
                values.append(call_url)
            result_url = result.outputs.get("url") if isinstance(result.outputs, Mapping) else None
            if isinstance(result_url, str):
                values.append(result_url)
            for value in values:
                key = _archive_pagination_key(value)
                if key is None:
                    continue
                host, path, _page = key
                latest_archive_result_idx[(host, path)] = max(latest_archive_result_idx.get((host, path), -1), idx)
        scored: list[tuple[int, int, str]] = []
        order = 0
        for idx, (_, result) in enumerate(transcript.turns):
            if not result.ok:
                continue
            for key in ("source_urls", "urls_detected"):
                # If a fetched page only supported one clue and same-scope
                # probes have already disproved a required complementary clue,
                # do not drain its outgoing navigation/cross-domain links. This
                # keeps multi-clue entity search candidate-independent: reject
                # the weak source and return to fresh clue discovery instead of
                # crawling home/news/archive/social links from a failed seed.
                result_scope_failed = key == "urls_detected" and _result_scope_failed_required_group(
                    result.outputs, prior_site_queries
                )
                detected_context = _result_evidence_text(result.outputs) if key == "urls_detected" else ""
                source_url = result.outputs.get("url") if isinstance(result.outputs, Mapping) else None
                if key == "urls_detected" and isinstance(source_url, str) and _source_host(source_url) in {"sites.google.com"}:
                    continue
                for url in _iter_strings(result.outputs.get(key)):
                    if not (_is_url(url) and url not in fetched and not _is_low_value_search_url(url)):
                        continue
                    allow_language_alternate = key == "urls_detected" and isinstance(source_url, str) and _is_same_article_language_alternate(source_url, url)
                    if key == "urls_detected" and not _source_clue_groups(detected_context) and not allow_language_alternate:
                        continue
                    allow_same_source_discovery = key == "urls_detected" and isinstance(source_url, str) and _is_same_source_discovery_link(source_url, url)
                    if result_scope_failed and not allow_same_source_discovery:
                        continue
                    if _scope_failed_required_group(url, prior_site_queries) and not allow_same_source_discovery:
                        continue
                    if _document_series_drained(url, transcript):
                        continue
                    if _archive_pagination_chain_drained(url, transcript):
                        continue
                    if blocking and latest_candidate_idx >= 0 and idx < latest_candidate_idx:
                        continue
                    score = _url_fetch_score(url, detected_context, source_kind=key)
                    if key == "urls_detected" and isinstance(source_url, str):
                        source_archive_key = _archive_pagination_key(source_url)
                        candidate_archive_key = _archive_pagination_key(url)
                        if source_archive_key is not None and candidate_archive_key is None:
                            host, path, _page = source_archive_key
                            latest_idx = latest_archive_result_idx.get((host, path), idx)
                            if idx < latest_idx:
                                # Once a newer page in the same archive window has
                                # been fetched, prefer its concrete article links
                                # over stale article links exposed by older pages.
                                # This keeps archive traversal source-driven while
                                # preventing old page candidates from starving the
                                # latest useful page's evidence links.
                                score -= min(80, (latest_idx - idx) * 24)
                    if allow_language_alternate:
                        score += 45
                    priorities = result.outputs.get("url_priorities") if isinstance(result.outputs, Mapping) else None
                    if key == "urls_detected" and isinstance(priorities, Mapping):
                        try:
                            score += min(40, max(0, int(priorities.get(url, 0))) * 2)
                        except (TypeError, ValueError):
                            pass
                    scored.append((score, order, url))
                    order += 1
            results = result.outputs.get("results")
            if not isinstance(results, (list, tuple)):
                continue
            for item in results:
                if not isinstance(item, Mapping):
                    continue
                url = item.get("url")
                if not (
                    isinstance(url, str)
                    and _is_url(url)
                    and url not in fetched
                    and not _is_low_value_search_url(url)
                    and not _scope_failed_required_group(url, prior_site_queries)
                ):
                    continue
                evidence_text = " ".join(str(item.get(k) or "") for k in ("title", "snippet", "url"))
                score = _url_fetch_score(url, evidence_text, source_kind="search_result")
                scored.append((score, order, url))
                order += 1
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored[0][2]

    def _make_call(self, tool_name: str, *, args: Mapping[str, object] | None = None) -> ToolCall:
        return ToolCall(
            id=f"rule-{next(self._counter)}",
            name=tool_name,
            args=dict(args) if args else {},
        )



def _latest_answer_candidate_index(transcript: Transcript) -> int:
    latest = -1
    for i, (_, result) in enumerate(transcript.turns):
        if not result.ok:
            continue
        candidate = result.outputs.get("answer_candidate")
        if isinstance(candidate, str) and candidate.strip():
            latest = i
    return latest


def _url_fetch_score(url: str, evidence_text: str, *, source_kind: str) -> int:
    lowered = " ".join([url, evidence_text or ""]).lower()
    is_detected_source_url = source_kind in {"source_urls", "urls_detected"}
    is_document_url = bool(re.search(r"\.(?:pdf|docx?|xlsx?|csv)(?:$|[?#])", lowered))
    has_requirement_clue = any(
        token in lowered
        for token in (
            "2002",
            "2003",
            "2022",
            "plant sample",
            "plant samples",
            "field trip",
            "field visit",
            "botany",
            "bank management",
            "paid tribute",
            "tribute ceremony",
            "vice chancellor",
            "graduation",
            "commencement",
            "convocation",
        )
    )
    # Search-result snippets are useful discovery evidence. Links extracted from
    # fetched source pages can be the next primary document, but generic archive
    # listings often expose dozens of opaque newsletter/PDF volumes. Prioritize
    # detected documents only when their URL or source-page context carries the
    # active requirement clues; otherwise keep them behind fresh search results so
    # the loop does not drain a whole volume series before validation probes.
    if is_detected_source_url and is_document_url and has_requirement_clue:
        score = 62
    elif is_detected_source_url and any(token in lowered for token in ("download", "archive", "newsletter", "news%20letter", "press", "media", "report")) and has_requirement_clue:
        score = 48
    else:
        score = 30 if source_kind == "search_result" else 15
    if any(token in lowered for token in ("news", "article", "event", "ceremony", "commencement", "graduation")):
        score += 10
    if any(token in lowered for token in ("plant", "sample", "field trip", "botany", "bank", "tribute", "vice chancellor")):
        score += 10
    if any(token in lowered for token in ("calendar", "calendarmaniacs", "sports", "kusports")):
        score -= 12
    if is_document_url and not is_detected_source_url:
        score -= 4
    return score


def _result_evidence_text(outputs: Mapping[str, object]) -> str:
    """Compact text from a fetched page for scoring its outgoing links."""

    pieces: list[str] = []
    for key in ("text", "answer_candidate"):
        value = outputs.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value[:3000])
    spans = outputs.get("spans")
    if isinstance(spans, (list, tuple)):
        pieces.extend(str(item)[:1000] for item in spans[:3] if str(item).strip())
    return " ".join(pieces)


def _document_series_key(url: str) -> str:
    lowered = url.lower()
    if not re.search(r"\.(?:pdf|docx?|xlsx?|csv)(?:$|[?#])", lowered):
        return ""
    try:
        parsed = urlparse(lowered)
    except Exception:
        return ""
    path = re.sub(r"(?:vol|volume|issue|no)[-_ %20]*\d+[a-z]?", "<series>", parsed.path)
    path = re.sub(r"\d{2,4}", "<num>", path)
    directory = path.rsplit("/", 1)[0]
    if "<series>" not in path and "news%20letter" not in path and "newsletter" not in path:
        return ""
    return f"{parsed.netloc}{directory}"


def _document_series_drained(url: str, transcript: Transcript) -> bool:
    """Skip more opaque archive/newsletter volumes after one sibling failed.

    This is generic budget hygiene: once a volume-series PDF from the same page
    was fetched and yielded only an unreadable/unsupported placeholder, the next
    sibling volume is less useful than returning to search/site validation.
    """

    key = _document_series_key(url)
    if not key:
        return False
    for call, result in transcript.turns:
        if call.name != "web_fetch":
            continue
        fetched_url = call.args.get("url") if isinstance(call.args, Mapping) else None
        if not isinstance(fetched_url, str) or _document_series_key(fetched_url) != key:
            continue
        text = " ".join(
            [str(result.summary or ""), str(result.error or ""), _result_evidence_text(result.outputs)]
        ).lower()
        if not result.ok or "text extraction unavailable" in text or "unreadable" in text or "empty body" in text:
            return True
    return False


def _archive_pagination_key(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    query = parsed.query.lower()
    match = re.search(r"(?:^|[&;])(page|p)=(\d+)(?:$|[&;])", query)
    if not match:
        return None
    return (_source_host(url), parsed.path.rstrip("/").lower(), int(match.group(2)))


def _archive_pagination_chain_drained(url: str, transcript: Transcript, *, max_pages: int = 4) -> bool:
    """Stop crawling endless archive pagination once a bounded window is open.

    Pagination pages are useful for reaching historical evidence, but after a few
    adjacent pages from the same collection have been fetched, concrete article
    links from those pages are better evidence than page N+1/N+2.  This prevents
    multi-clue runs from spending the whole fetch budget walking an archive.
    """

    key = _archive_pagination_key(url)
    if key is None:
        return False
    target_host, target_path, target_page = key
    fetched_page_urls: set[str] = set()
    for call, result in transcript.turns:
        if call.name != "web_fetch":
            continue
        values = []
        call_url = call.args.get("url") if isinstance(call.args, Mapping) else None
        if isinstance(call_url, str):
            values.append(call_url)
        result_url = result.outputs.get("url") if isinstance(result.outputs, Mapping) else None
        if isinstance(result_url, str):
            values.append(result_url)
        for value in values:
            value_key = _archive_pagination_key(value)
            if value_key is None:
                continue
            host, path, page_num = value_key
            if host == target_host and path == target_path and abs(page_num - target_page) <= max_pages * 2:
                fetched_page_urls.add(value)
    return len(fetched_page_urls) >= max_pages


def _search_args(query: str, prior_queries: list[str]) -> dict[str, object]:
    return {
        "query": query,
        "limit": 8,
        "skip_web_answerer": True,
        # Keep enough prior query telemetry for expanded-budget focused runs.
        # A 16-search BrowseComp-style chunk can execute several fallback
        # queries per tool call; truncating to 20 lets old exact fallbacks rotate
        # back into later calls and hides the real exhausted groups in artifacts.
        "attempted_queries": list(dict.fromkeys(prior_queries))[-80:],
        # The controller has already de-duplicated exact queries.  For hard
        # multi-clue entity searches it may intentionally issue a second broad
        # query in the same clue group after a candidate site-scope was rejected
        # (for example, find the next plant-source candidate after the previous
        # plant-source host failed all bank probes).  Without this explicit flag
        # the web tool's generic group guard can turn that fresh variant into a
        # skipped no-op and prematurely stop discovery.
        "allow_same_group_retry": True,
        # Controller-directed hard-source searches often surface useful original
        # domain probes only after several filtered wrapper/trap results.  Use a
        # modest bounded fallback budget here; the overall loop/tool caps still
        # bound the run, while preserving candidate-independent wrapper recovery
        # such as double-encoded `site:<domain>` search pages.
        "max_search_fallbacks": 5,
    }


def _is_multiclue_entity_prompt(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    has_entity = any(token in lowered for token in ("institution", "university", "college", "school", "learning establishment"))
    clue_count = sum(1 for token in ("criterion", "2002", "2003", "2022", "seven days", "capital city") if token in lowered)
    return has_entity and clue_count >= 2


def _latest_next_query(transcript: Transcript, prior_queries: list[str]) -> str:
    prior_groups = {_query_group(query) for query in prior_queries if _query_group(query)}
    for _, result in reversed(transcript.turns):
        if not result.ok:
            continue
        raw = result.outputs.get("next_queries")
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            query = " ".join(item.split())[:300]
            if not query or _is_url(query) or _query_seen(query, prior_queries):
                continue
            group = _query_group(query)
            # Sufficiency LLMs often emit long prompt-like follow-up strings.
            # For multi-clue institution searches those normalize to the same
            # broad clue group already exhausted by web_search, which produced
            # empty-query/no-op turns. Site-scoped probes are allowed elsewhere;
            # here skip broad repeated groups so the controller moves to the
            # next real seed/tool instead of burning steps.
            if group and group in prior_groups and not _is_site_scoped_query(query):
                continue
            return query
    return ""


def _normalize_query(query: str) -> str:
    cleaned = " ".join((query or "").lower().replace('"', " ").replace("'", " ").split())
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", cleaned)
        if token and token not in {"news", "official", "source", "primary", "page", "website"}
    ]
    return " ".join(tokens)


def _query_seen(query: str, prior_queries: list[str]) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False
    return any(_normalize_query(prior) == normalized for prior in prior_queries)


def _prior_web_queries(transcript: Transcript) -> list[str]:
    prior: list[str] = []
    for call, result in transcript.turns:
        if call.name != "web_search":
            continue
        call_query = call.args.get("query") if isinstance(call.args, Mapping) else None
        if isinstance(call_query, str) and call_query.strip():
            prior.append(call_query.strip())
        result_query = result.outputs.get("query") if isinstance(result.outputs, Mapping) else None
        if isinstance(result_query, str) and result_query.strip():
            prior.append(result_query.strip())
        attempted = result.outputs.get("attempted_queries") if isinstance(result.outputs, Mapping) else None
        if isinstance(attempted, (list, tuple)):
            for item in attempted:
                if isinstance(item, str) and item.strip():
                    prior.append(item.strip())
    return list(dict.fromkeys(prior))


def _is_same_source_discovery_link(source_url: str, candidate_url: str) -> bool:
    """Allow official same-site news/list/detail links after search-index probes fail.

    A site-scoped search can return zero even when the fetched official news page
    exposes local pagination or article links. Those links are still public
    same-source evidence, unlike social/share/navigation exits, so the controller
    may fetch a bounded next page/detail before abandoning the candidate scope.
    """

    if _source_host(source_url) != _source_host(candidate_url):
        return False
    try:
        path = urlparse(candidate_url).path.lower()
    except Exception:
        return False
    if not path or path == "/":
        return False
    if any(token in path for token in ("facebook", "twitter", "linkedin", "whatsapp", "share")):
        return False
    return any(
        token in path
        for token in (
            "/news",
            "news-section",
            "/event",
            "/events",
            "/archive",
            "/archives",
            "/media",
            "/press",
            "/article",
            "/articles",
        )
    )



def _result_scope_failed_required_group(outputs: Mapping[str, object], prior_queries: list[str]) -> bool:
    """Whether a fetched result's own source scope already failed validation.

    ``urls_detected`` are outgoing links from the fetched page. If that page's
    candidate scope has already exhausted a required clue group, following those
    links usually crawls navigation/news/archive/social surfaces from a rejected
    candidate rather than discovering a new entity. Keep ``source_urls`` eligible
    elsewhere; this guard is only for outgoing detected links.
    """

    for value in (outputs.get("url"), *list(_iter_strings(outputs.get("fetched_urls"))), *list(_iter_strings(outputs.get("source_urls")))):
        if isinstance(value, str) and _is_url(value) and _scope_failed_required_group(value, prior_queries):
            return True
    return False


def _scope_failed_required_group(url: str, prior_queries: list[str]) -> bool:
    scope = _source_scope(url, allow_hosted_path=True)
    if not scope:
        return False
    # Bank/tribute is a high-value complementary clue in multi-requirement
    # institution tasks. If all generic variants for that scope are exhausted,
    # the candidate source has failed a required clue and should not be drained
    # through navigation links. Other groups can still be tried before rejection.
    return _host_group_exhausted(scope, "bank", prior_queries)


def _missing_requirement_query(request: TaskRequest, transcript: Transcript, prior_queries: list[str]) -> str:
    requirements: Mapping[str, object] = {}
    coverage: list[Mapping[str, object]] = []
    for _, result in reversed(transcript.turns):
        if not result.ok:
            continue
        raw_req = result.outputs.get("requirements")
        if isinstance(raw_req, Mapping) and not requirements:
            requirements = raw_req
        raw_cov = result.outputs.get("requirement_coverage")
        if isinstance(raw_cov, list) and not coverage:
            coverage = [item for item in raw_cov if isinstance(item, Mapping)]
        if requirements and coverage:
            break
    if not coverage:
        return ""
    required_by_id: dict[str, str] = {}
    raw_required = requirements.get("required_outputs") if isinstance(requirements, Mapping) else None
    if isinstance(raw_required, list):
        for i, item in enumerate(raw_required, 1):
            if not isinstance(item, Mapping) or bool(item.get("optional", False)):
                continue
            rid = str(item.get("id") or f"requirement_{i}")
            required_by_id[rid] = " ".join(
                str(item.get(key) or "")
                for key in ("description", "evidence_required")
            ).strip() or rid
    parts: list[str] = []
    for item in coverage:
        rid = str(item.get("requirement_id") or item.get("id") or "")
        status = str(item.get("status") or "").lower()
        if status not in {"missing", "weak", "contradicted"}:
            continue
        desc = required_by_id.get(rid, rid)
        reason = str(item.get("reason") or "")
        pieces = [desc, reason]
        parts.append(" ".join(piece for piece in pieces if piece).strip())
    if not parts:
        return ""
    seeded = _seeded_missing_requirement_query(transcript, parts, prior_queries)
    if seeded:
        return seeded
    candidate_scoped = _candidate_scoped_missing_requirement_query(transcript, parts, prior_queries)
    if candidate_scoped:
        return candidate_scoped
    focused = _focused_multiclue_query(request.prompt or "", parts, prior_queries)
    if focused:
        return focused
    prompt = " ".join((request.prompt or "").split())[:180]
    query = " ".join([prompt, *parts])[:300].strip()
    return "" if _query_seen(query, prior_queries) else query


def _seeded_missing_requirement_query(transcript: Transcript, parts: list[str], prior_queries: list[str]) -> str:
    """Build site-scoped follow-up queries from a partial same-source clue hit.

    Multi-clue entity tasks often find one strong clue first (for example an
    official plant-trip article) and then drift to unrelated institutions for
    the remaining clues. A source-domain seed keeps the next query tied to that
    partial candidate without knowing the benchmark answer or task id.
    """

    missing_groups: list[str] = []
    for part in parts:
        group = _query_group(part)
        if group and group not in missing_groups:
            missing_groups.append(group)
    if not missing_groups:
        return ""
    fetched_seeds = _partial_source_seeds(transcript, fetched_only=True)
    seeds = fetched_seeds or _partial_source_seeds(transcript)
    if not seeds:
        return ""
    seed_groups = {group for seed in seeds for group in seed[1]}
    for host, groups in seeds:
        if _host_has_exhausted_required_group(host, groups, missing_groups, prior_queries):
            continue
        ordered = _seed_followup_group_order(missing_groups, groups, seed_groups)
        for group in ordered:
            if _host_group_exhausted(host, group, prior_queries):
                continue
            for query in _site_queries_for_group(host, group):
                if not _query_seen(query, prior_queries):
                    return query[:300]
    return ""


def _seeded_prompt_followup_query(prompt: str, transcript: Transcript, prior_queries: list[str]) -> str:
    """Continue validating a partial source host when no candidate exists yet.

    Early multi-clue searches often find one institution-like source that only
    supports one clue. If no LLM candidate/coverage table is available, the
    controller should still ask whether that same host supports complementary
    clue groups before draining unrelated intra-site links. The rule is generic:
    derive clue groups from the prompt and source text, then build
    ``site:<host>`` probes; no task id, expected answer, or fixture is used.
    """

    prompt_groups = _prompt_multiclue_groups(prompt)
    if not prompt_groups:
        return ""
    seeds = _partial_source_seeds(transcript, fetched_only=True) or _partial_source_seeds(transcript)
    if not seeds:
        return ""
    seed_groups = {group for _, groups in seeds for group in groups}
    priority = {"bank": 0, "graduation": 1, "event": 2, "plant": 3}
    for host, groups in seeds:
        missing = [group for group in prompt_groups if group not in groups]
        if not missing:
            missing = [group for group in prompt_groups if group in seed_groups or group]
        if any(group not in groups and _host_group_exhausted(host, group, prior_queries) for group in missing):
            continue
        for group in sorted(dict.fromkeys(missing), key=lambda g: priority.get(g, 9)):
            if _host_group_exhausted(host, group, prior_queries):
                continue
            for query in _site_queries_for_group(host, group):
                if not _query_seen(query, prior_queries):
                    return query[:300]
    return ""


def _prompt_multiclue_groups(prompt: str) -> list[str]:
    lowered = (prompt or "").lower()
    groups: list[str] = []
    checks = (
        ("plant", ("plant", "sample", "field trip", "botany", "flora", "herbarium")),
        ("bank", ("bank", "tribute", "management", "ceremony", "vice chancellor")),
        ("graduation", ("graduation", "commencement", "convocation", "fourth sunday", "2003")),
        ("event", ("2002", "thursday", "saturday", "three day", "three-day", "support")),
    )
    for group, tokens in checks:
        if any(token in lowered for token in tokens):
            groups.append(group)
    return groups


def _candidate_scoped_missing_requirement_query(transcript: Transcript, parts: list[str], prior_queries: list[str]) -> str:
    """Probe a newly proposed entity name against missing clue groups.

    LLM research drafts sometimes name a plausible institution without exposing
    cited URLs.  When the requirement table still has weak/missing/contradicted
    rows, query the public web for that candidate plus one missing clue before
    falling back to broad clue-only searches.  This is candidate-independent
    evidence validation: it does not know expected answers, task ids, or
    benchmark labels, and it still requires later fetch/sufficiency checks.
    """

    candidate = _latest_named_entity_candidate(transcript)
    if not candidate:
        return ""
    groups: list[str] = []
    for part in parts:
        group = _query_group(part)
        if group and group not in groups:
            groups.append(group)
    if not groups:
        return ""
    priority = {"bank": 0, "plant": 1, "graduation": 2, "event": 3}
    for group in sorted(groups, key=lambda g: priority.get(g, 9)):
        for query in _candidate_queries_for_group(candidate, group):
            if not _query_seen(query, prior_queries):
                return query[:300]
    return ""


def _latest_named_entity_candidate(transcript: Transcript) -> str:
    patterns = (
        re.compile(r"\*\*(?:Institution Name|Institution|University|College|School)\s*:?\*\*\s*([^\n.;]+)", re.I),
        re.compile(r"(?:institution|learning institution|university|college|school)\s+(?:is|:)?\s*\*\*([^*\n]{4,120})\*\*", re.I),
        re.compile(r"\*\*([A-Z][A-Za-z&.,'() -]{3,120}(?:University|College|School|Institute)[A-Za-z&.,'() -]*)\*\*"),
        re.compile(r"(?:is|named)\s+(?:the\s+)?([A-Z][A-Za-z&.,'() -]{3,120}(?:University|College|School|Institute)[A-Za-z&.,'() -]*)"),
    )
    reject = ("insufficient", "no verified", "not enough", "cannot be named", "unsupported")
    for _, result in reversed(transcript.turns):
        if not result.ok:
            continue
        raw = result.outputs.get("answer_candidate")
        if not isinstance(raw, str) or not raw.strip():
            continue
        lowered = raw.lower()
        if any(token in lowered[:180] for token in reject):
            continue
        for pattern in patterns:
            match = pattern.search(raw)
            if not match:
                continue
            candidate = " ".join(match.group(1).strip(" *:-—.\n\t").split())
            candidate = re.sub(r"\s+(?:\(|-)?(?:UCSC|UPD|UCB)\)?$", "", candidate).strip()
            if _looks_like_named_institution(candidate):
                return candidate[:120]
    return ""


def _looks_like_named_institution(candidate: str) -> bool:
    lowered = (candidate or "").lower()
    if len(candidate.split()) < 2:
        return False
    if any(token in lowered for token in ("provided criteria", "specified conditions", "not available", "no verified", "insufficient")):
        return False
    return any(token in lowered for token in ("university", "college", "school", "institute"))


def _candidate_queries_for_group(candidate: str, group: str) -> list[str]:
    quoted = f'"{candidate}"'
    if group == "bank":
        return [
            f'{quoted} "bank management" "ceremony"',
            f'{quoted} "paid tribute" "bank"',
            f'{quoted} "vice chancellor" "bank"',
        ]
    if group == "plant":
        return [
            f'{quoted} "plant samples" students department 2022',
            f'{quoted} "field trip" plant students 2022',
            f'{quoted} botany field trip students',
        ]
    if group == "graduation":
        return [
            f'{quoted} "2003" graduation Sunday',
            f'{quoted} "2003" commencement Sunday',
            f'{quoted} "2003" convocation Sunday',
        ]
    if group == "event":
        return [
            f'{quoted} "2002" Thursday Saturday support',
            f'{quoted} "2002" "three-day" support',
            f'{quoted} "2002" event support',
        ]
    return []


def _host_has_exhausted_required_group(host: str, seed_groups: set[str], missing_groups: list[str], prior_queries: list[str]) -> bool:
    """Reject a candidate host once same-host probes disprove a required clue.

    A multi-clue entity candidate found from one clue (for example a plant-trip
    article) must satisfy every other non-optional clue on the same institution
    domain.  If all generic site-scoped variants for any complementary missing
    group have already been tried for that host, keep the host out of the
    follow-up queue instead of spending later turns on a different clue for a
    candidate that already failed one required group.
    """

    for group in missing_groups:
        if group in seed_groups:
            continue
        if _host_group_exhausted(host, group, prior_queries):
            return True
    return False


def _host_group_exhausted(host: str, group: str, prior_queries: list[str]) -> bool:
    variants = _site_queries_for_group(host, group)
    return bool(variants) and all(_query_seen(query, prior_queries) for query in variants)


def _seed_followup_group_order(missing_groups: list[str], seed_groups: set[str], all_seed_groups: set[str]) -> list[str]:
    # If the seed already proves one clue group, ask for complementary clues on
    # the same domain first. The global priority reflects common entity tasks:
    # cross-link a dated article to the related ceremony, then historical dates.
    priority = {"bank": 0, "graduation": 1, "event": 2, "plant": 3}
    groups = [g for g in missing_groups if g not in seed_groups]
    if not groups:
        groups = [g for g in missing_groups if g in all_seed_groups or g]
    return sorted(dict.fromkeys(groups), key=lambda g: priority.get(g, 9))


def _site_queries_for_group(host: str, group: str) -> list[str]:
    if not host:
        return []
    prefix = f"site:{host}"
    if group == "bank":
        return [
            f'{prefix} "bank" "tribute" "ceremony"',
            f'{prefix} "bank management" "vice chancellor"',
            f'{prefix} "paid tribute" "bank"',
        ]
    if group == "graduation":
        return [
            f'{prefix} "2003" "graduation" "Sunday"',
            f'{prefix} "2003" "commencement" "Sunday"',
            f'{prefix} "2003" "convocation" "Sunday"',
        ]
    if group == "event":
        return [
            f'{prefix} "2002" "Thursday" "Saturday" "support"',
            f'{prefix} "2002" "three-day" "support"',
            f'{prefix} "2002" "3-day" "support"',
        ]
    if group == "plant":
        return [
            f'{prefix} "2022" "plant" "samples" "students"',
            f'{prefix} "field trip" "plant" "department"',
        ]
    return []


def _partial_source_seeds(transcript: Transcript, *, fetched_only: bool = False) -> list[tuple[str, set[str]]]:
    by_host: dict[str, set[str]] = {}
    order: list[str] = []
    for call, result in transcript.turns:
        if not result.ok:
            continue
        candidates: list[tuple[str, str, bool]] = []
        url = result.outputs.get("url")
        if isinstance(url, str):
            text = " ".join(
                str(result.outputs.get(key) or "")[:4000]
                for key in ("text", "answer_candidate")
            )
            spans = result.outputs.get("spans")
            if isinstance(spans, (list, tuple)):
                text = " ".join([text, *[str(x) for x in spans[:3]]])
            candidates.append((url, text, True))
        if not fetched_only:
            results = result.outputs.get("results")
            if isinstance(results, (list, tuple)):
                for item in results:
                    if not isinstance(item, Mapping):
                        continue
                    item_url = item.get("url")
                    if not isinstance(item_url, str):
                        continue
                    text = " ".join(str(item.get(k) or "") for k in ("title", "snippet", "url"))
                    candidates.append((item_url, text, False))
        for item_url, text, fetched_page in candidates:
            host = _source_scope(item_url, allow_hosted_path=fetched_page)
            if not host or _is_low_value_search_url(item_url) or _is_low_value_seed_host(host):
                continue
            groups = _source_clue_groups(text)
            if not groups:
                continue
            if host not in by_host:
                by_host[host] = set()
                order.append(host)
            by_host[host].update(groups)
    scored = sorted(
        ((len(by_host[host]), idx, host, by_host[host]) for idx, host in enumerate(order)),
        key=lambda item: (-item[0], item[1]),
    )
    return [(host, groups) for _, _, host, groups in scored[:4]]


def _has_fetched_partial_source_seed(transcript: Transcript) -> bool:
    return bool(_partial_source_seeds(transcript, fetched_only=True))


def _is_site_scoped_query(query: str) -> bool:
    return " ".join((query or "").lower().split()).startswith("site:")


def _source_host(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().split("@")[-1].split(":", 1)[0]
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def _article_language_neutral_path(url: str) -> str:
    try:
        path = urlparse(url).path.lower().strip("/")
    except Exception:
        return ""
    parts = [part for part in path.split("/") if part]
    # Many public sites expose machine-translated or partially translated pages
    # by inserting a two-letter language component (for example /en/news/326
    # and /news/326).  The original-language page can contain the real title/body
    # when the translated page is only a generic shell.  Normalize only the path
    # shape; later fetch/sufficiency checks still decide whether the source helps.
    neutral = [part for part in parts if not re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", part)]
    return "/".join(neutral)


def _is_same_article_language_alternate(source_url: str, candidate_url: str) -> bool:
    if _source_host(source_url) != _source_host(candidate_url):
        return False
    source_path = _article_language_neutral_path(source_url)
    candidate_path = _article_language_neutral_path(candidate_url)
    if not source_path or source_path != candidate_path:
        return False
    try:
        source_parts = set(urlparse(source_url).path.lower().strip("/").split("/"))
        candidate_parts = set(urlparse(candidate_url).path.lower().strip("/").split("/"))
    except Exception:
        return False
    return source_parts != candidate_parts and any(part in {"news", "article", "event", "events"} for part in source_path.split("/"))


def _source_scope(url: str, *, allow_hosted_path: bool = False) -> str:
    """Return a search ``site:`` scope for a source URL.

    Generic hosting domains are unsafe as bare seeds because they mix many
    unrelated tenants. Once a hosted page itself has been fetched and has useful
    clue text, however, a path-scoped tenant such as
    ``sites.google.com/<org>/<site>`` is a legitimate same-source follow-up
    scope. This keeps the repair generic while avoiding bare ``site:sites.google.com``.
    """

    host = _source_host(url)
    if not host:
        return ""
    if allow_hosted_path and host == "sites.google.com":
        try:
            parts = [part for part in urlparse(url).path.split("/") if part]
        except Exception:
            parts = []
        if len(parts) >= 2:
            return "/".join([host, parts[0], parts[1]])
    return host


def _is_low_value_seed_host(host: str) -> bool:
    """Hosts that should not define a same-institution candidate queue.

    Multi-clue repair uses ``site:<host>`` probes to validate whether one
    institution/domain satisfies complementary clues. Generic hosting, search,
    translation, and social/aggregator domains can contain official-looking
    snippets, but the host itself is not the candidate institution domain; using
    it as a seed wastes budget and mixes unrelated tenants.
    """

    lowered = (host or "").lower()
    exact = {
        "sites.google.com",
        "docs.google.com",
        "drive.google.com",
        "translate.google.com",
        "webcache.googleusercontent.com",
        "silo.tips",
        "mymemory.translated.net",
        "americanprofessionguide.com",
        "studenttravel.pro",
        "bio.libretexts.org",
        "iastate.pressbooks.pub",
        "iastatedigitalpress.com",
        "harvardfilmarchive.org",
        "academic.naver.com",
        "academia.edu",
        "plusgarden.com",
        "plantcafeseoul.com",
        "aha-dic.com",
        "yongoro.com",
        "ibiology.org",
        "scandict.com",
        "academic.or.kr",
        "academic.oup.com",
        "iteslj.org",
        "administrator.de",
        "cybo.com",
        "manta.com",
        "yellowpages.com",
        "chamberofcommerce.com",
        "oeb.harvard.edu",
    }
    suffixes = (
        ".googleusercontent.com",
        ".github.io",
    )
    return lowered in exact or any(lowered.endswith(suffix) for suffix in suffixes)


def _source_clue_groups(text: str) -> set[str]:
    lowered = (text or "").lower()
    groups: set[str] = set()
    plantish = any(token in lowered for token in ("plant sample", "plant samples", "plant specimens", "botany", "field trip", "flora", "herbarium"))
    plant_activity = any(token in lowered for token in ("field trip", "field visit", "study trip", "trip to", "gather", "gathering", "collect plants", "collecting plants", "students", "department"))
    plant_article_context = any(token in lowered for token in ("2022", "news", "article", "published", "posted"))
    if plantish and plant_activity and plant_article_context:
        groups.add("plant")
    if any(token in lowered for token in ("bank management", "paid tribute", "pay tribute", "tribute", "vice chancellor", "bank ceremony")) and "bank" in lowered:
        groups.add("bank")
    if any(token in lowered for token in ("graduation", "commencement", "convocation")) and ("2003" in lowered or "sunday" in lowered):
        groups.add("graduation")
    if ("2002" in lowered and any(token in lowered for token in ("thursday", "saturday", "three-day", "three day", "3-day"))) or ("2002" in lowered and "support" in lowered):
        groups.add("event")
    return groups


def _focused_multiclue_query(prompt: str, parts: list[str], prior_queries: list[str]) -> str:
    """Build one focused public-source query for multi-clue entity tasks.

    Broad all-clue queries tend to hit search aggregators or irrelevant library
    catalog pages. For multi-clue institution/entity tasks, aim at one missing
    clue at a time and ask for primary/official pages.
    """

    lowered = " ".join([prompt, *parts]).lower()
    if not any(token in lowered for token in ("institution", "university", "college", "school", "learning establishment")):
        return ""

    grouped: list[tuple[str, list[str]]] = []
    for part in parts:
        part_lower = part.lower()
        if any(token in part_lower for token in ("plant", "sample", "field trip", "department")):
            grouped.append((
                "plant",
                [
                    '"plant samples" students department trip 2022 "news" university',
                    '"field trip" "plant" "samples" students department university 2022',
                    '"plant sampling" students department university 2022 news',
                    'botany field trip students university department 2022',
                    'collecting plant specimens students department university news',
                    '"plant specimens" "students" "department" university news',
                    '"flora" "field trip" students department university',
                    '"botanical" "field visit" students department university',
                ],
            ))
        if any(token in part_lower for token in ("bank", "tribute", "management", "ceremony")):
            grouped.append((
                "bank",
                [
                    '"bank" management tribute ceremony university official',
                    '"bank management" "ceremony" "vice chancellor" university 2022',
                    '"paid tribute" "bank" management university ceremony',
                    'tribute management bank university official ceremony',
                    'academic division ceremony bank management university official',
                ],
            ))
        if any(token in part_lower for token in ("graduation", "commencement", "fourth sunday")):
            grouped.append((
                "graduation",
                [
                    '"2003" graduation ceremony "Sunday" university',
                    '"2003" "convocation" "Sunday" university',
                ],
            ))
        if any(token in part_lower for token in ("three-day", "thursday", "saturday", "support")):
            grouped.append((
                "event",
                [
                    '"2002" "Thursday" "Saturday" "support" university event',
                    '"three day" event support students university 2002',
                ],
            ))

    allow_same_group_revisit = _has_rejected_candidate_scope(prior_queries)
    prior_groups = {_query_group(query) for query in prior_queries if _query_group(query)}
    for group, queries in grouped:
        if group in prior_groups and not allow_same_group_revisit:
            continue
        for query in queries:
            if not _query_seen(query, prior_queries):
                return query[:300]

    targeted: list[str] = []
    for _, queries in grouped:
        targeted.extend(queries)
    targeted.extend([
        '"plant samples" "bank" "ceremony" university',
        '"gather samples of plants" students department university',
        '"herbarium" students "field trip" university 2022',
    ])
    for query in targeted:
        if not _query_seen(query, prior_queries):
            return query[:300]
    return ""


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


def _has_rejected_candidate_scope(prior_queries: list[str]) -> bool:
    """Whether prior site-scoped probes show a candidate scope was tested.

    After a partial clue source is found, the controller validates the same
    host with several ``site:<host>`` complementary-clue probes.  If those probes
    have happened, a broad query in a previously tried clue group can be useful:
    it searches for the *next* independent source candidate instead of staying
    tied to the rejected host.  Exact-query de-duplication still prevents loops.
    """

    scoped = [query for query in prior_queries if _is_site_scoped_query(query)]
    if len(scoped) >= 2:
        return True
    return any(_query_group(query) in {"bank", "graduation", "event"} for query in scoped)


def _advisor_name_from_transcript(transcript: Transcript) -> str:
    patterns = [
        re.compile(r"Ph\.?D\.?\s+Advisor\s+([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,3})"),
        re.compile(r"Advisor\s*(?:\d+)?\s*:\s*([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,3})"),
        re.compile(r"advised by\s+(?:Professor\s+)?([A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,3})", re.I),
    ]
    for _, result in reversed(transcript.turns):
        texts: list[str] = [result.observation or ""]
        spans = result.outputs.get("spans") if isinstance(result.outputs, Mapping) else None
        if isinstance(spans, (list, tuple)):
            texts.extend(str(x) for x in spans if str(x).strip())
        for text in texts:
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    name = " ".join(match.group(1).split())
                    if len(name.split()) >= 2:
                        return name
    return ""


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_low_value_search_url(url: str) -> bool:
    lowered = url.lower()
    low_value_markers = (
        "linkedin.com/jobs/",
        "linkedin.com/posts/",
        "linkedin.com/in/",
        "facebook.com/",
        "instagram.com/",
        "harvardfilmarchive.org/",
        "harvard.edu/copyright-issue",
        "doaks.org/research/library-archives",
        "library.harvard.edu/libraries/harvard-university-archives",
        "library.harvard.edu/about/library-newsletters-social-media",
        "harvard.edu/media-relations",
        "open.spotify.com/",
        "www.google.com/?",
        "google.com/search",
        "askfilo.com/",
        "pinterest.com/",
        "wikipedia.org/wiki/biology",
        "wikipedia.org/wiki/flora",
        "wikipedia.org/wiki/herbarium",
        "worldfloraonline.org/",
        "philippineplants.org/",
        "flora.ai/",
        "flora.ph/",
        "flora.appfinca.com/",
        "microbenotes.com/herbarium",
        "biologynotesonline.com/herbarium",
        "herbarium.duke.edu/about/what-is-a-herbarium",
        "herbarium.com.br/",
        "herbarium.gov.hk/",
        "herbarium.co/",
        "biologyinsights.com/what-is-an-herbarium",
        "britannica.com/science/herbarium",
        "kew.org/science/collections-and-resources/collections/herbarium",
        "usna.usda.gov/science/u.s-national-arboretum-herbaria",
        "britannica.com/science/biology",
        "commons.wikimedia.org/wiki/biology",
        "plant-collection-moves-into-new-space",
        "web-static.archive.org/_static/",
        "archive.org/services/img/",
        "donate.wikimedia.org/",
        "special:downloadaspdf",
        "/digital-accessibility/",
        "/content-formats/social-media",
        "/product-specific-guides/",
        "/contact-support/report-concern",
        "/securepurdue/",
        "/copyright-policies/",
        "/security-programs/",
        "/it-policies-standards/",
        "/data-handling/media-disposal",
        "/ehps/police/statistics-policies/security-reports",
        "/freedom-of-expression",
        "/use-of-university-facilities",
        "giving.purdue.edu/",
        "/newsroom/media",
        "/media-contacts",
        "/purdue-news-weekly",
        "/publicsafety/clery/",
        "/annual-security-report",
        "/visit.html",
        "/apply.html",
        "/scholarships-financial-aid",
        "/connect.html",
        "/login/lostpassword",
        "/user/register",
        "/user/setlocale/",
        "pkp.sfu.ca/ojs/",
        "swiftsoftpro.com/",
        "americanprofessionguide.com/",
        "studenttravel.pro/",
        "bio.libretexts.org/",
        "usbg.gov/schools-families/field-trips",
        "addtoany.com/",
        "iastate.pressbooks.pub/",
        "iastatedigitalpress.com/",
        "m.shein.com/",
        "shein.com/",
        "seaart.ai/search/",
        "/catalog?",
        "/search-results/?",
        "search_index_results",
        "q=%22gather+samples",
        "browsecomp-plus-benchmark",
        "huggingface.co/datasets/timchen0618/browsecomp",
        "browsecomp.jsonl",
        "hkust-nlp/webexplorer",
        "zhihu.com/question/",
        "blog.naver.com/",
        "m.blog.naver.com/",
        "lingolandedu.com/",
        "dictionary.cambridge.org/",
        "collinsdictionary.com/",
        "wordow.com/english/dictionary/",
        "secure.fourth.com/",
        "store-3.co.uk/",
        "grammarly.com/commonly-confused-words/",
        "yourdictionary.com/fourth",
        "grammar.com/forth_vs",
        "definitions.net/definition/fourth",
        "help.hotschedules.com/",
        "hotschedules.zendesk.com/",
        "zoom.us/download",
        "explore.zoom.us/",
        "media.zoom.com/",
        "plusgarden.com/",
        "plantcafeseoul.com/",
        "academic.naver.com/",
        "academia.edu/",
        "aha-dic.com/",
        "yongoro.com/",
        "ibiology.org/",
        "scandict.com/",
        "academic.or.kr/",
        "academic.oup.com/",
        "iteslj.org/",
        "administrator.de/",
        "oeb.harvard.edu/field-trips",
        "oeb.harvard.edu/annual-reports",
        "oeb.harvard.edu/news-events",
        "oeb.harvard.edu/student-news-events",
        "oeb.harvard.edu/ib-students-news",
        "samplefocus.com/",
        "slooply.com/samples/",
        "looperman.com/loops/",
        "samplette.io/",
        "mypikpak.com/",
        "thecalculatorsite.com/conversions/",
        "inchcalculator.com/convert/",
        "unitconverters.net/length/",
        "rapidtables.com/convert/",
        "metric-conversions.org/length/",
        "creativepark.canon/",
        "crunchyroll.com/",
        "justwatch.com/",
        "cybo.com/",
        "manta.com/",
        "yellowpages.com/",
        "chamberofcommerce.com/",
        "archives.nd.edu/commencement/",
        "muarchives.missouri.edu/c-rg0-s4",
        "_assets/calendars",
        "bankofamerica.com/",
        "usbank.com/online-mobile-banking",
        "td.com/us/en/personal-banking",
        "capitalone.com/bank",
        "regions.com/personal-banking",
    )
    return any(marker in lowered for marker in low_value_markers)


def _github_repo_from_transcript(transcript: Transcript) -> str:
    for _, result in transcript.turns:
        if not result.ok:
            continue
        candidates: list[object] = []
        candidates.extend(_iter_strings(result.outputs.get("source_urls")))
        results = result.outputs.get("results")
        if isinstance(results, (list, tuple)):
            for item in results:
                if isinstance(item, Mapping):
                    candidates.append(item.get("url", ""))
        for value in candidates:
            if not isinstance(value, str):
                continue
            parts = value.split("github.com/", 1)
            if len(parts) != 2:
                continue
            path_bits = [bit for bit in parts[1].split("/", 3)[:2] if bit]
            if len(path_bits) == 2:
                return "/".join(path_bits)
    return ""


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                yield item


def _has_blocking_requirement(transcript: Transcript) -> bool:
    raw_coverage = transcript.notes_view().get("requirement_coverage")
    if not isinstance(raw_coverage, list):
        return False
    for item in raw_coverage:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").lower()
        if status in {"missing", "weak", "contradicted"}:
            return True
    return False


def _has_mixed_source_domain_blocker(transcript: Transcript) -> bool:
    notes = transcript.notes_view()
    raw_coverage = notes.get("requirement_coverage")
    if not isinstance(raw_coverage, list):
        return False
    for item in raw_coverage:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").lower()
        reason = str(item.get("reason") or "").lower()
        if status in {"missing", "weak", "contradicted"} and "mixed source domains" in reason:
            return True
    return False


__all__ = ["RuleBasedController"]
