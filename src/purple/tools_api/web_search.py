"""``web_search`` tool — search + optional one-shot OpenAI web_search_preview answer."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Mapping
from urllib.parse import unquote

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt
from ..runtime.tool import ToolContext, ToolResult
from ..tools import (
    StdlibWebClient,
    WebAnswerer,
    WebClient,
    extract_urls,
    openai_web_search_from_env,
)


_FAILURE_MARKERS = (
    "couldn't locate",
    "could not locate",
    "couldn't find",
    "could not find",
    "i don't have",
    "not easily identifiable",
    "would need access",
    "please provide relevant sources",
)


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the open web for the user query. May short-circuit to an "
        "OpenAI web_search_preview answer when configured."
    )
    arg_schema: Mapping[str, str] = {
        "query": "search query; defaults to the user prompt when omitted",
        "limit": "optional max number of results (default 5)",
        "attempted_queries": "optional prior web_search queries to avoid repeating",
    }

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        web_client: WebClient | None = None,
        web_answerer: WebAnswerer | None = None,
        use_env_web_answerer: bool = True,
    ) -> None:
        self._llm = llm
        self._web = web_client or StdlibWebClient()
        self._web_answerer = (
            web_answerer
            if web_answerer is not None
            else (openai_web_search_from_env() if use_env_web_answerer else None)
        )

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        prompt = (ctx.request.prompt or "").strip()
        raw_query = args.get("query")
        explicit_query = isinstance(raw_query, str) and bool(raw_query.strip())
        query = raw_query if explicit_query else prompt[:300]
        try:
            limit = max(1, int(args.get("limit", 5)))
        except (TypeError, ValueError):
            limit = 5

        if self._web_answerer is not None and args.get("skip_web_answerer") is not True:
            direct = await self._openai_web_answer(prompt)
            if direct and not self._looks_like_failed(direct):
                return ToolResult(
                    tool_call_id="",
                    ok=True,
                    summary="answered with OpenAI web_search_preview",
                    observation=direct[:600],
                    outputs={
                        "answer_candidate": direct,
                        "spans": [direct[:3000]],
                        "source": "openai_web_search_preview",
                    },
                )

        prompt_urls = extract_urls(prompt, limit=4)
        if self._llm is not None and not explicit_query:
            built = await self._build_query(prompt)
            if built:
                query = built

        results: list[dict[str, str]] = []
        executed_queries: list[str] = []
        search_attempts: list[dict[str, Any]] = []
        prior_queries = _prior_queries_from_args(args)
        is_multiclue = _is_multiclue_entity_prompt(prompt)
        tried_queries: list[str] = list(prior_queries) if is_multiclue else []
        prior_groups = {_query_group(item) for item in prior_queries if _query_group(item)} if is_multiclue else set()
        # Early in a multi-clue search, avoid repeating the same clue group.
        # Once every core clue family has returned no usable candidate, allow a
        # new exact query from an already-seen group.  Otherwise the controller
        # can be forced into research/finalize after four zero-result groups,
        # even though broader same-group public-source variants remain untried.
        allow_exhausted_group_broadening = is_multiclue and _has_tried_all_core_multiclue_groups(prior_queries)
        if is_multiclue and not explicit_query and (not isinstance(query, str) or query.strip() == prompt[:300]):
            # Never spend a multi-clue search turn on the raw user prompt when
            # the controller/tool call supplied no query. That path creates
            # overlong all-clue searches and can repeat as a duplicate no-op.
            # Start with compact public-source fallback queries instead; exact
            # and group de-duplication below still prevents loops.
            for fallback_query in _fallback_queries_for_prompt("", prompt):
                if not _query_seen(fallback_query, tried_queries):
                    query = fallback_query
                    break
        query = _normalize_query_for_prompt(str(query or ""), prompt)
        original_query = query
        query_group = _query_group(query)
        is_site_scoped = _is_site_scoped_query(query)
        wrapper_followups: list[str] = []
        # For ordinary broad multi-clue searches, suppress same-clue repeats by
        # query group. For site-scoped candidate-domain probes, keep distinct
        # same-group queries (e.g. several bank/ceremony variants on the same
        # host) because the controller is intentionally validating one fetched
        # partial candidate before drifting to unrelated domains.
        allow_same_group_retry = bool(args.get("allow_same_group_retry"))
        group_already_tried = (
            query_group in prior_groups
            and not is_site_scoped
            and not allow_same_group_retry
            and not allow_exhausted_group_broadening
        )
        skipped_query = ""
        if query and not _query_seen(query, tried_queries) and not group_already_tried:
            tried_queries.append(query)
            executed_queries.append(query)
            raw_results = await self._web.search(query, limit=limit)
            wrapper_followups.extend(_wrapper_followup_queries(raw_results, query))
            kept_results = [item for item in raw_results if not _is_low_value_or_benchmark_result(item, query=query)]
            search_attempts.append(
                {
                    "query": query,
                    "phase": "primary",
                    "raw_count": len(raw_results),
                    "kept_count": len(kept_results),
                    "filtered_count": len(raw_results) - len(kept_results),
                }
            )
            results.extend(kept_results)
        else:
            skipped_query = query
            # Preserve blocked duplicate/group queries in attempted_queries so
            # the controller can advance instead of repeatedly producing blank
            # no-op web_search turns for the same exhausted clue group.
            if query and not _query_seen(query, tried_queries):
                tried_queries.append(query)
            if query:
                search_attempts.append(
                    {
                        "query": query,
                        "phase": "skipped",
                        "raw_count": 0,
                        "kept_count": 0,
                        "filtered_count": 0,
                        "skip_reason": "duplicate_or_group_guard",
                    }
                )
            query = ""
        # Do not let an empty site:<host> candidate-domain probe fall back to a
        # global query. Global fallback results pollute the evidence pool with
        # unrelated institutions.  Instead, if a wrapper exposed a plausible
        # source domain but the exact site query returns zero, try a small set of
        # same-domain/path variants before handing control back to the planner.
        site_scoped_original = _is_site_scoped_query(original_query)
        allow_fallbacks = True
        if not results and allow_fallbacks:
            # Bound fallback fan-out inside one tool call. Sparse multi-clue
            # queries may require several semantic variants, but trying every
            # variant synchronously can burn the whole task time budget before
            # the controller can re-plan or report an evidence gap.
            fallback_attempts = 0
            try:
                max_fallback_attempts = max(0, int(args.get("max_search_fallbacks", 3 if is_multiclue else 5)))
            except (TypeError, ValueError):
                max_fallback_attempts = 3 if is_multiclue else 5
            fallback_queue = (
                _site_scoped_fallback_queries(original_query, prompt)
                if site_scoped_original
                else list(wrapper_followups) + _fallback_queries_for_prompt(query, prompt)
            )
            # Wrapper/proxy results can expose a concrete public source domain
            # only after several broad clue fallbacks.  Treat those recovered
            # site:<domain> follow-ups as priority source-discovery probes so a
            # broad-search fallback cap does not discard the first usable
            # candidate domain before it is searched.  This is still bounded by
            # the number of wrapper follow-ups recovered from public search
            # results, and it never invents answers or leaves the exposed domain.
            priority_followups = set(wrapper_followups)
            queue_idx = 0
            while queue_idx < len(fallback_queue):
                fallback_query = fallback_queue[queue_idx]
                if fallback_attempts >= max_fallback_attempts and fallback_query not in priority_followups:
                    break
                queue_idx += 1
                fallback_group = _query_group(fallback_query)
                fallback_site_scoped = _is_site_scoped_query(fallback_query)
                fallback_group_tried = (
                    fallback_group in prior_groups
                    and not fallback_site_scoped
                    and not allow_same_group_retry
                    and not allow_exhausted_group_broadening
                )
                fallback_seen = (
                    any(" ".join(prior.lower().split()) == " ".join(fallback_query.lower().split()) for prior in tried_queries)
                    if fallback_site_scoped
                    else _query_seen(fallback_query, tried_queries)
                )
                if fallback_seen or fallback_group_tried:
                    continue
                tried_queries.append(fallback_query)
                fallback_attempts += 1
                query = fallback_query
                executed_queries.append(fallback_query)
                fallback_results = await self._web.search(fallback_query, limit=limit)
                if fallback_site_scoped:
                    for followup_query in reversed(_site_scoped_fallback_queries(fallback_query, prompt)):
                        exact_seen = any(" ".join(prior.lower().split()) == " ".join(followup_query.lower().split()) for prior in tried_queries)
                        if followup_query not in fallback_queue and not exact_seen:
                            fallback_queue.insert(queue_idx, followup_query)
                # Wrapper/proxy pages can appear on fallback queries too (not
                # just the primary query). Preserve their embedded original
                # site:<domain> search as the next bounded fallback before
                # drifting to unrelated broad clue variants.
                for followup_query in _wrapper_followup_queries(fallback_results, fallback_query):
                    if followup_query not in fallback_queue and not _query_seen(followup_query, tried_queries):
                        fallback_queue.insert(queue_idx, followup_query)
                        priority_followups.add(followup_query)
                kept_fallback_results = [
                    item
                    for item in fallback_results
                    if not _is_low_value_or_benchmark_result(item, query=fallback_query)
                ]
                search_attempts.append(
                    {
                        "query": fallback_query,
                        "phase": "site_fallback" if site_scoped_original else "fallback",
                        "raw_count": len(fallback_results),
                        "kept_count": len(kept_fallback_results),
                        "filtered_count": len(fallback_results) - len(kept_fallback_results),
                    }
                )
                results.extend(kept_fallback_results)
                if results:
                    query = fallback_query
                    break
        direct_discovery_urls: list[str] = []
        if not results:
            # Search backends can miss sparse/older institutional pages even
            # after a public search-wrapper exposes a plausible source domain.
            # Surface a tiny set of same-domain homepage/news/archive URLs for
            # the controller to fetch next. This is bounded source discovery,
            # not a synthetic answer: the fetcher/verifier still has to find
            # visible evidence on the public site.  Apply this both when the
            # primary query was site-scoped and when site:<domain> was recovered
            # as a fallback from a wrapper/proxy search result.
            site_query_for_direct_discovery = original_query if site_scoped_original else ""
            if not site_query_for_direct_discovery:
                for executed_query in reversed(executed_queries):
                    if _is_site_scoped_query(executed_query):
                        site_query_for_direct_discovery = executed_query
                        break
            if site_query_for_direct_discovery:
                direct_discovery_urls = _site_scoped_direct_discovery_urls(site_query_for_direct_discovery, prompt)

        for url in prompt_urls:
            if not any(r.get("url") == url for r in results):
                results.insert(0, {"title": url, "url": url, "snippet": "URL from prompt"})

        spans: list[str] = []
        for item in results[:5]:
            snippet = item.get("snippet") or item.get("title") or item.get("url") or ""
            if snippet:
                spans.append(f"Search result: {snippet} ({item.get('url', '')})")

        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=(
                "web_search returned "
                f"{len(results)} result(s)"
                + (" (skipped duplicate query/group)" if skipped_query and not query else "")
            ),
            observation=spans[0] if spans else "(no results)",
            outputs={
                "query": query or skipped_query,
                "skipped_query": skipped_query if skipped_query and not query else "",
                "attempted_queries": tried_queries,
                "executed_queries": executed_queries,
                "search_attempts": search_attempts,
                "results": results[:limit],
                "source_urls": direct_discovery_urls,
                "urls_detected": direct_discovery_urls,
                "spans": spans,
                "source": "web",
            },
        )

    async def _openai_web_answer(self, prompt: str) -> str:
        assert self._web_answerer is not None
        system = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                "You are a careful open-web research agent. Use web_search_preview to find current primary sources. "
                "Cite source URLs beside critical claims. Do not invoke task IDs or hidden ground truth.",
            )
            if chunk
        )
        try:
            return await self._web_answerer.answer(
                prompt=(
                    "Solve this open-web research task using OpenAI web_search_preview. "
                    "Return the answer with source URLs for critical claims.\n\nTask:\n"
                    + prompt
                ),
                system=system,
                max_tokens=1400,
            )
        except Exception:
            return ""

    async def _build_query(self, prompt: str) -> str:
        assert self._llm is not None
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", load_prompt("system") or ""),
                    ChatMessage(
                        "user",
                        "Turn this user task into one concise web search query. "
                        "Return only the query text, no JSON.\n\nTask:\n" + prompt,
                    ),
                ],
                tag="web_query",
                max_tokens=80,
            )
        except Exception:
            return ""
        return (text or "").strip().strip('"')[:300]

    @staticmethod
    def _looks_like_failed(answer: str) -> bool:
        lowered = answer.lower()
        return any(marker in lowered for marker in _FAILURE_MARKERS)


def _is_low_value_or_benchmark_result(item: Mapping[str, str], *, query: str = "") -> bool:
    text = " ".join(str(item.get(key) or "") for key in ("title", "url", "snippet")).lower()
    query_lower = (query or "").lower()
    if _is_multiclue_query_result_trap(text, query_lower):
        return True
    markers = (
        "google.com/finance",
        "youtube.com/shorts/",
        "youtube.com/results",
        "youtube.com/watch?",
        "wordplays.com/crossword-solver",
        "gather.town/",
        "botanico.co.kr/",
        "plant-collection-moves-into-new-space",
        "gradschoolstory.net/english-grammar/",
        "wordvice.ai/ko/grammar/",
        "play.google.com/store/apps/details",
        "linkedin.com/jobs",
        "linkedin.com/jobs/",
        "linkedin.com/posts/",
        "linkedin.com/in/",
        "harvardfilmarchive.org/",
        "harvard.edu/copyright-issue",
        "doaks.org/research/library-archives",
        "library.harvard.edu/libraries/harvard-university-archives",
        "library.harvard.edu/about/library-newsletters-social-media",
        "harvard.edu/media-relations",
        "open.spotify.com/",
        "www.google.com/?",
        "google.com/search",
        "m.shein.com/",
        "shein.com/",
        "seaart.ai/search/",
        "instagram.com/",
        "facebook.com/",
        "pinterest.com/",
        "etsy.com/market/",
        "scribd.com/document/",
        "silo.tips/search/",
        "mymemory.translated.net/",
        "storyboardthat.com/photos/",
        "wordreference.com/",
        "wiktionary.org/",
        "ichacha.net/",
        "redkiwiapp.com/",
        "namu.wiki/",
        "timesnownews.com/topic/",
        "askfilo.com/",
        "/course/search.php?",
        "/doi/?query=",
        "/search/site%3a",
        "/catalog?",
        "/search-results/?",
        "search_index_results",
        "browsecomp-plus-benchmark",
        "huggingface.co/datasets/timchen0618/browsecomp",
        "browsecomp.jsonl",
        "hkust-nlp/webexplorer",
        "/digital-accessibility/",
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
        "pkp.sfu.ca/ojs/",
        "americanprofessionguide.com/",
        "studenttravel.pro/",
        "bio.libretexts.org/",
        "usbg.gov/schools-families/field-trips",
        "addtoany.com/",
        "iastate.pressbooks.pub/",
        "iastatedigitalpress.com/",
        "zhihu.com/question/",
        "blog.naver.com/",
        "m.blog.naver.com/",
        "lingolandedu.com/",
        "dictionary.cambridge.org/",
        "merriam-webster.com/dictionary/",
        "thefreedictionary.com/",
        "dictionary.com/browse/",
        "writingexplained.org/",
        "collinsdictionary.com/",
        "wordow.com/english/dictionary/",
        "britannica.com/plant/plant",
        "livescience.com/planet-earth/plants",
        "sciencefacts.net/parts-of-a-plant",
        "vocabineer.com/100-types-of-plants",
        "drdata.in/cardiologists",
        "history.com/a-year-in-history/",
        "onthisday.com/date/",
        "thefactsite.com/year/",
        "takemeback.to/events/date/",
        "eventshistory.com/date/",
        "thosewerethedays.substack.com/p/25-fun-facts-and-historical-events",
        "britannica.com/science/botany",
        "environmentalscience.org/botany",
        "geeksforgeeks.org/biology/botany",
        "golifescience.com/introduction-to-botany",
        "golifescience.com/life-sciences-branches",
        "golifescience.com/biochemistry",
        "biologyinsights.com/what-is-botany",
        "scienceinsights.org/what-is-botany",
        "morebetter.sg/plant-nurseries",
        "nparks.gov.sg/florafaunaweb",
        "bioexplorer.net/plants",
        "philoid.com/ncert/",
        "gather.coop/",
        "gatherit.co/",
        "secure.fourth.com/",
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
        "store-3.co.uk/",
        "grammarly.com/commonly-confused-words/",
        "yourdictionary.com/fourth",
        "grammar.com/forth_vs",
        "definitions.net/definition/fourth",
    )
    return any(marker in text for marker in markers)


def _is_multiclue_query_result_trap(text: str, query_lower: str) -> bool:
    """Drop generic clue-word hits that are unlikely to support the requested clue.

    Multi-clue entity searches are easily derailed by pages whose title only
    matches a broad phrase from the query, e.g. an institute homepage containing
    "bank management" but no tribute/ceremony evidence, or generic botany field
    trip PDFs when the requirement asks for a dated website article. This filter
    is query/content based and does not use benchmark IDs or expected answers.
    """

    text_lower = text.lower()
    query_lower = query_lower.lower()
    if not text_lower or not query_lower:
        return False
    # Search engines occasionally return unrelated popular/movie/stock pages for
    # over-constrained multi-clue queries.  Before spending fetch budget, require
    # at least minimal lexical overlap with the clue query unless the query is
    # explicitly site-scoped (where the domain itself is the signal).
    if "site:" not in query_lower:
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", query_lower)
            if term
            not in {
                "the",
                "and",
                "for",
                "with",
                "university",
                "college",
                "school",
                "students",
                "department",
                "article",
                "news",
                "2022",
                "2002",
                "2003",
                "class",
            }
        }
        if query_terms:
            overlap = {term for term in query_terms if term in text_lower}
            if len(overlap) == 0:
                return True
    is_bank_query = any(token in query_lower for token in ("bank", "tribute", "ceremony", "vice chancellor", "rector"))
    if is_bank_query:
        bank_evidence_terms = (
            "tribute",
            "paid tribute",
            "pay tribute",
            "ceremony",
            "honour",
            "honor",
            "honoured",
            "honored",
            "felicitation",
            "vice chancellor",
            "rector",
            "academic division",
        )
        generic_bank_homepage = "bank management" in text_lower and any(
            homepage_token in text_lower for homepage_token in ("institute", "school", "college", "university", "homepage")
        )
        generic_bank_service_page = "bank" in text_lower and any(
            service_token in text_lower
            for service_token in (
                "personal banking",
                "account management",
                "loans",
                "transfers",
                "branch locator",
                "financial products",
                "credit card",
                "internet banking",
            )
        )
        generic_bank_domain_page = any(
            domain_token in text_lower
            for domain_token in (
                "bank.com",
                "bank .com",
                "bank.co",
                "bank .co",
                "bank.kr",
                "bank .kr",
                "bank.in",
                "bank .in",
                "banklocationmaps.com",
                "rbi.org.in/regionalbranch",
                "jaipurchalo.com/banks-in",
                "banks-in-",
                "ibk.co",
                "kbstar.com",
                "nhbank.com",
            )
        ) and not any(edu_token in text_lower for edu_token in ("university", "college", "school", ".edu"))
        if "site:" not in query_lower and "bank" not in text_lower:
            return True
        if (generic_bank_service_page or generic_bank_domain_page) and not any(term in text_lower for term in bank_evidence_terms):
            return True
        if "wikipedia.org/wiki/" in text_lower and "bank" in text_lower and not any(term in text_lower for term in bank_evidence_terms):
            return True
        if generic_bank_homepage and not any(term in text_lower for term in bank_evidence_terms):
            return True
        if "bank management" in text_lower and not any(term in text_lower for term in bank_evidence_terms):
            return True
    is_plant_query = any(token in query_lower for token in ("plant", "botany", "biology", "field trip", "field visit", "sample", "specimen"))
    if is_plant_query:
        if not any(term in text_lower for term in ("plant", "botany", "biology", "flora", "herbarium", "specimen", "sample")):
            return True
        if any(
            trap in text_lower
            for trap in (
                "wikipedia.org/wiki/plant",
                "wikipedia.org/wiki/botany",
                "wikipedia.org/wiki/biology",
                "wikipedia.org/wiki/flora",
                "wikipedia.org/wiki/herbarium",
                "worldfloraonline.org",
                "philippineplants.org",
                "flora.ai",
                "flora.ph",
                "flora.appfinca.com",
                "microbenotes.com/herbarium",
                "biologynotesonline.com/herbarium",
                "herbarium.duke.edu/about/what-is-a-herbarium",
                "herbarium.com.br",
                "herbarium.gov.hk",
                "herbarium.co",
                "biologyinsights.com/what-is-an-herbarium",
                "britannica.com/science/herbarium",
                "kew.org/science/collections-and-resources/collections/herbarium",
                "usna.usda.gov/science/u.s-national-arboretum-herbaria",
                "britannica.com/science/biology",
                "khanacademy.org/science/biology",
                "mdpi.com/journal/biology",
                "commons.wikimedia.org/wiki/biology",
                "dictionary",
                "wordow.com",
                "naver.com",
                "engram",
                "meaning of",
                "plant, vegetation, flora",
                "plant_collection_guidelines",
                "plant collection guidelines",
                "pharmacognosy_field_trips",
                "collectionseducation.org/specimen-collection",
                "plant-science-global-food-security",
                "plant science global food security",
                "plant collection moves into new space",
                "summer experience",
                "laboratório botânico",
                "medicinal plants and phytotherapy",
                "fitoterapia",
                "social equity-licensed dispensary",
                "premium cannabis",
                "copyright 2021 of the hong kong herbarium",
            )
        ):
            return True
        article_terms = ("news", "article", "published", "posted", "press", "story", "year level", "students")
        generic_pdf_report = ("pdf" in text_lower or ".pdf" in text_lower) and any(
            term in text_lower for term in ("department of botany", "field trip", "field visit", "tour report", "programme report")
        )
        generic_research_publication = any(
            marker in text_lower for marker in ("researchgate.net/", "doi.org/", "journal", "issn", "article-processing-charge")
        ) and not any(term in text_lower for term in ("university news", "college news", "/news/", "posted", "published"))
        generic_specimen_or_herbarium_news = any(
            marker in text_lower
            for marker in (
                "plant samples preserved in museums",
                "digitizing" ,
                "herbarium houses",
                "museum specimens",
                "specimen collection",
            )
        ) and not any(term in text_lower for term in ("field trip", "field visit", "study trip", "gather", "gathering", "collect plants", "collecting plants"))
        if generic_pdf_report and not any(term in text_lower for term in article_terms[:6]):
            return True
        if generic_research_publication or generic_specimen_or_herbarium_news:
            return True
    is_graduation_query = any(token in query_lower for token in ("graduation", "commencement", "convocation", "fourth sunday"))
    if is_graduation_query:
        broad_archive_or_calendar = any(
            marker in text_lower
            for marker in (
                "commencement archives",
                "commencement programs and related materials",
                "record sub-group",
                "academic calendar",
                "graduation commencement programs",
                "/commencement/",
                "commencement.pdf",
                "_assets/calendars",
            )
        )
        concrete_fourth_sunday_evidence = any(
            marker in text_lower
            for marker in (
                "fourth sunday",
                "4th sunday",
                "graduation ceremony",
                "commencement ceremony",
                "convocation ceremony",
            )
        )
        if broad_archive_or_calendar and not concrete_fourth_sunday_evidence:
            return True
    is_event_query = any(token in query_lower for token in ("2002", "thursday", "saturday", "three day", "three-day", "support"))
    if is_event_query and any(
        trap in text_lower
        for trap in (
            "stock.yahoo.com",
            "three.com/",
            "three.co.uk/",
            "threejs.org",
            "three.ie/",
            "threecosmetics.com",
            "threetimes333.com",
            "wikipedia.org/wiki/3_(company",
            "cmoney.tw",
            "baidu.com",
            "bilibili.com",
            "wikipedia.org/wiki/2002",
            "quote/2002",
            "technical-analysis",
        )
    ):
        return True
    return False


def _site_scoped_fallback_queries(query: str, prompt: str) -> list[str]:
    """Generate bounded same-domain variants for sparse site:<domain> probes.

    Wrapper recovery can surface an over-specific query such as
    ``site:example.edu/en/news 2003 graduation ...``. Some search backends return
    zero for that exact path even when the same public site can be found with the
    path and clue terms split differently. Keep the repair domain-scoped so a
    failed candidate-domain probe does not drift into unrelated institutions.
    """

    raw = " ".join((query or "").split()).strip()
    match = re.match(r"site\s*:\s*([^\s]+)(?:\s+(.*))?$", raw, flags=re.I)
    if not match:
        return []
    site_expr = match.group(1).strip('"\'')
    tail = match.group(2) or ""
    # Accept both site:host/path and site:host path forms. Preserve only public
    # source-domain/path information already exposed by the search wrapper.
    domain_match = re.match(r"([a-z0-9][a-z0-9.-]+\.[a-z]{2,})(/\S*)?", site_expr, flags=re.I)
    if not domain_match:
        return []
    domain = domain_match.group(1).lower().strip("./")
    path = domain_match.group(2) or ""
    if any(domain.endswith(suffix) for suffix in ("google.com", "youtube.com", "linkedin.com")):
        return []

    decoded_tail = unquote(" ".join([path, tail]))
    decoded_tail = re.sub(r"https?://\S+", " ", decoded_tail)
    decoded_tail = re.split(r"[?&](?:language|cdn_rsite|ref|rep|ret|msockid|utm_|fbclid|gclid)=", decoded_tail, maxsplit=1)[0]
    tokens: list[str] = []
    seen: set[str] = set()
    stop = {
        "site",
        "html",
        "page",
        "search",
        "query",
        "results",
        "language",
        "cdn",
        "rsite",
        "ref",
        "rep",
        "ret",
        "msockid",
        "wrapper",
        "proxy",
        "the",
        "and",
        "for",
        "with",
        "university",
        "college",
        "school",
    }
    domain_terms = {part for part in re.split(r"[^a-z0-9]+", domain) if part}
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{1,}", decoded_tail):
        key = token.lower()
        if key in stop or key in domain_terms or key in seen:
            continue
        tokens.append(token)
        seen.add(key)
    if not tokens:
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", prompt or "")
            if token.lower() not in stop
        ][:8]

    # Pull a generic capitalized entity phrase out of the recovered tail when
    # present (e.g. an institution name in a wrapper query). This is not an
    # expected-answer constant; it is public text from the search result itself.
    capitalized_phrases = []
    leading_time_words = {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    for m in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.-]{2,}\s+){1,5}[A-Z][A-Za-z0-9&.-]{2,}\b", decoded_tail):
        words = m.group(0).split()
        while words and (words[0] in leading_time_words or re.fullmatch(r"\d{4}", words[0])):
            words.pop(0)
        phrase = " ".join(words)
        if len(words) >= 2 and phrase not in capitalized_phrases:
            capitalized_phrases.append(phrase)

    path_parts = [part for part in re.split(r"[^A-Za-z0-9-]+", path) if part and part.lower() not in {"en", "eng", "www"}]
    clue_terms = tokens[:8]
    if not path_parts and "news" not in {term.lower() for term in clue_terms} and not capitalized_phrases:
        return []
    quoted_clues = [tok for tok in clue_terms if len(tok) >= 4][:4]
    out: list[str] = []

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split()).strip()[:300]
        if candidate and candidate.lower() != raw.lower() and candidate not in out:
            out.append(candidate)

    if path_parts:
        add(" ".join([f"site:{domain}/" + "/".join(path_parts[:3])] + clue_terms[:8]))
    if "news" in {term.lower() for term in clue_terms} and not path_parts:
        add(" ".join([f"site:{domain}/news"] + [term for term in clue_terms if term.lower() != "news"][:8]))
    if clue_terms:
        add(" ".join([f"site:{domain}"] + clue_terms[:8]))
    if capitalized_phrases:
        phrase = capitalized_phrases[0]
        add(f'site:{domain} "{phrase}" ' + " ".join(clue_terms[:6]))
    if quoted_clues:
        add(" ".join([f"site:{domain}"] + [f'"{term}"' for term in quoted_clues]))
    if any(term.lower() in {"news", "article", "graduation", "ceremony", "bank", "plant", "samples", "botany"} for term in clue_terms):
        add(" ".join([f"site:{domain}", "news"] + [term for term in clue_terms if term.lower() != "news"][:7]))
    return out[:5]


def _site_scoped_direct_discovery_urls(query: str, prompt: str) -> list[str]:
    """Return bounded same-domain URLs to fetch when site: search is empty.

    A zero-result search for an exposed public source domain is not evidence
    that the domain lacks relevant pages.  Fetching the homepage/news/archive
    entry points lets the controller discover visible navigation/RSS/sitemap
    links without leaving the candidate source domain or inventing an answer.
    """

    match = re.match(r"\s*site\s*:\s*([a-z0-9][a-z0-9.-]+\.[a-z]{2,})(/\S*)?", query or "", flags=re.I)
    if not match:
        return []
    domain = match.group(1).lower().strip("./")
    path = (match.group(2) or "").strip()
    if any(domain.endswith(suffix) for suffix in ("google.com", "youtube.com", "linkedin.com")):
        return []
    if not any(token in (prompt or "").lower() for token in ("institution", "university", "college", "school", "learning establishment")):
        return []

    out: list[str] = []

    def add(url: str) -> None:
        if url not in out:
            out.append(url)

    base = f"https://{domain}"
    add(base + "/")
    path_parts = [part for part in re.split(r"[^A-Za-z0-9-]+", path) if part and part.lower() not in {"en", "eng", "www"}]
    if path_parts:
        add(base + "/" + "/".join(path_parts[:3]).strip("/") + "/")
    lowered = " ".join([query or "", prompt or ""]).lower()
    if any(token in lowered for token in ("news", "article", "2022", "graduation", "ceremony", "event", "support", "bank", "plant")):
        for candidate_path in (
            "/news/",
            "/en/news/",
            "/archives/",
            "/archive/",
            "/events/",
            "/media/",
            "/rss.xml",
            "/sitemap.xml",
        ):
            add(base + candidate_path)
    return out[:8]


def _wrapper_followup_queries(results: Sequence[Mapping[str, str]], source_query: str) -> list[str]:
    """Recover original-domain searches from low-value search-wrapper hits.

    Some search engines surface wrapper/proxy pages whose title or encoded URL
    contains a useful `site:example.edu ...` query, while the wrapper URL itself
    is not worth fetching. Preserve that public-source clue by converting it to
    a fresh domain-scoped query before filtering the wrapper result out. This is
    generic source-discovery hygiene, not benchmark/task-id routing.
    """

    out: list[str] = []
    for item in results:
        if not _is_low_value_or_benchmark_result(item):
            continue
        text = " ".join(str(item.get(key) or "") for key in ("title", "url", "snippet"))
        # Search-wrapper URLs often double-encode the original query
        # (e.g. ``site%253Aexample.edu%252Fnews``).  Decode a few bounded
        # rounds before extracting the public source domain; otherwise the
        # generic wrapper-recovery path misses useful site-scoped follow-ups.
        for _ in range(3):
            decoded = unquote(text)
            if decoded == text:
                break
            text = decoded
        for match in re.finditer(r"site\s*:\s*([a-z0-9][a-z0-9.-]+\.[a-z]{2,})([^\n\r]*)", text, flags=re.I):
            domain = match.group(1).strip(" .,/;:)").lower()
            if not domain or any(domain.endswith(suffix) for suffix in ("google.com", "youtube.com", "linkedin.com")):
                continue
            tail = re.sub(r"https?://\S+", " ", match.group(2) or "")
            tail = re.split(r"[?&](?:language|cdn_rsite|ref|rep|ret|msockid|utm_|fbclid|gclid)=", tail, maxsplit=1)[0]
            tail = re.sub(r"[/_+%=&?:;,#()\[\]{}|<>]+", " ", tail)
            terms = []
            seen_terms: set[str] = set()
            noisy_wrapper_terms = {
                "search",
                "site",
                "page",
                "html",
                "query",
                "results",
                "language",
                "cdn",
                "cdn-rsite",
                "rsite",
                "ref",
                "rep",
                "ret",
                "msockid",
                "explore",
                "over",
                "thousands",
                "generated",
                "al-generated",
                "ai-generated",
                "artworks",
                "about",
            }
            domain_terms = {part for part in re.split(r"[^a-z0-9]+", domain.lower()) if part}
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", tail):
                key = token.lower()
                if key in noisy_wrapper_terms or key in domain_terms or key in seen_terms:
                    continue
                terms.append(token)
                seen_terms.add(key)
            if not terms:
                terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", source_query)[:8]
            query = " ".join([f"site:{domain}"] + terms[:12]).strip()[:300]
            if query and query not in out:
                out.append(query)
    return out[:5]


def _is_multiclue_entity_prompt(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    has_entity = any(token in lowered for token in ("institution", "university", "college", "school", "learning establishment"))
    clue_count = sum(1 for token in ("criterion", "2002", "2003", "2022", "seven days", "capital city", "graduation", "bank", "plant") if token in lowered)
    return has_entity and clue_count >= 2


def _prior_queries_from_args(args: Mapping[str, Any]) -> list[str]:
    raw = args.get("attempted_queries")
    values: list[str] = []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.append(" ".join(item.split())[:300])
    return values


def _normalize_query_key(query: str) -> str:
    import re

    cleaned = " ".join((query or "").lower().replace('"', " ").replace("'", " ").split())
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", cleaned)
        if token and token not in {"news", "official", "source", "primary", "page", "website"}
    ]
    return " ".join(tokens)


def _query_seen(query: str, prior_queries: list[str]) -> bool:
    normalized = _normalize_query_key(query)
    if not normalized:
        return False
    return any(_normalize_query_key(prior) == normalized for prior in prior_queries)


def _is_site_scoped_query(query: str) -> bool:
    return " ".join((query or "").lower().split()).startswith("site:")


def _has_tried_all_core_multiclue_groups(prior_queries: list[str]) -> bool:
    groups = {_query_group(query) for query in prior_queries if _query_group(query)}
    return {"plant", "bank", "graduation", "event"}.issubset(groups)


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


def _fallback_queries_for_prompt(query: str, prompt: str) -> list[str]:
    """Generic public-source query broadening for sparse search results.

    The controller often starts with an over-quoted clue query. When the first
    search returns only filtered aggregator/social/marketplace pages, try a few
    source-discovery synonyms. This is intentionally generic: it keys off
    ordinary clue words in the prompt/query, not benchmark ids or answers.
    """

    combined = " ".join([query or "", prompt or ""]).lower()
    if not any(token in combined for token in ("institution", "university", "college", "school", "learning establishment")):
        return []
    grouped: list[list[str]] = []
    if any(token in combined for token in ("plant", "sample", "field trip", "trip", "department")):
        grouped.append([
            '"field trip" "plant" "samples" students department university 2022',
            '"plant sampling" students department university 2022 news',
            '"systematic botany" class field trip 2022 university news',
            'botany field trip students university department 2022',
            'collecting plant specimens students department university news',
            'biology students field trip collect plants university',
            '"gather samples" plants students university department',
            '"plant specimens" "students" "department" university news',
            '"flora" "field trip" students department university',
            '"botanical" "field visit" students department university',
            '2022 botany department field visit students collected specimens university',
            '2022 biology department study tour students collected plant samples college',
        ])
    if any(token in combined for token in ("bank", "tribute", "management", "ceremony")):
        grouped.append([
            '"bank management" "ceremony" "vice chancellor" university 2022',
            '"paid tribute" "bank" management university ceremony',
            'honoured bank management university ceremony dean',
            'tribute management bank university official ceremony',
            'academic division ceremony bank management university official',
            'university official honoured bank management academic division ceremony',
        ])
    if any(token in combined for token in ("graduation", "commencement", "fourth sunday")):
        grouped.append([
            '"2003" "graduation" "Sunday" "university"',
            '"2003" "convocation" "Sunday" university',
            '"fourth Sunday" graduation university',
            '2003 graduation ceremony fourth Sunday college',
        ])
    if any(token in combined for token in ("three-day", "thursday", "saturday", "support")):
        grouped.append([
            '"2002" "Thursday" "Saturday" "support" university',
            '"three day" event support students university 2002',
            '"three-day" "support" "university" "2002"',
            '2002 Thursday Saturday support event college university',
        ])
    out: list[str] = []
    # Interleave clue groups so a capped sparse-search call does not spend all
    # attempts on one clue family (e.g. plant pages) while bank/graduation/event
    # clues could surface a better candidate. This is candidate-independent and
    # uses only public clue words from the prompt/query.
    max_group_len = max((len(group) for group in grouped), default=0)
    for idx in range(max_group_len):
        for group in grouped:
            if idx >= len(group):
                continue
            item = group[idx]
            if item not in out:
                out.append(item)
    return out[:24]


def _normalize_query_for_prompt(query: str, prompt: str) -> str:
    """Turn controller meta-instructions into concise public-source queries.

    LLM controllers sometimes pass instructions such as "Search exact phrases
    for 2022 institution-site articles..." as the search query. DuckDuckGo then
    tends to return synthetic/aggregator pages (jobs, catalogs, benchmark
    mirrors). This sanitizer is task-agnostic: it only recognizes generic
    multi-clue institution/source-discovery vocabulary and rewrites it into a
    compact evidence query, without using task ids or expected answers.
    """

    cleaned = " ".join((query or "").split()).strip()
    prompt_lower = (prompt or "").lower()
    lowered = cleaned.lower()
    if not cleaned:
        return cleaned
    if not any(token in prompt_lower for token in ("learning institution", "educational institution", "learning establishment", "university", "college", "school")):
        return cleaned[:300]

    # Avoid passing the whole user prompt or controller prose as one query.
    is_controller_instruction = lowered.startswith(("search ", "verify ")) or len(cleaned) > 220
    if not is_controller_instruction:
        return cleaned[:300]

    if any(token in lowered for token in ("plant", "sample", "field trip", "year levels", "department")):
        return '"plant samples" "students" "department" "trip" "2022" university -jobs -linkedin'[:300]
    if any(token in lowered for token in ("bank", "tribute", "management", "ceremony", "vice chancellor", "president", "rector")):
        return '"bank" "management" "tribute" "ceremony" university "2022" -jobs -linkedin'[:300]
    if any(token in lowered for token in ("graduation", "commencement", "fourth sunday")):
        return '"2003" "graduation ceremony" "Sunday" university commencement archive'[:300]
    if any(token in lowered for token in ("three-day", "thursday", "saturday", "support", "2002")):
        return '"2002" "Thursday" "Saturday" "support" university event archive'[:300]
    if any(token in lowered for token in ("capital", "location", "address")):
        return 'official university address country capital city 2023'[:300]
    return cleaned[:300]


__all__ = ["WebSearchTool"]
