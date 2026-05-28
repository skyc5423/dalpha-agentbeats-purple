"""Open-web research specialist.

Searches the public web, fetches a small number of source pages, and asks the
LLM to synthesize an answer with source URLs. It is benchmark-agnostic: routing
is based on open-web research features, not task or benchmark IDs.
"""

from __future__ import annotations

import json

from ..llm import ChatMessage, LLMClient
from ..prompts import load_prompt
from ..schema import StepRecord
from ..state import WorkingState
from ..tools import (
    StdlibWebClient,
    WebAnswerer,
    WebClient,
    extract_json,
    extract_urls,
    openai_web_search_from_env,
)


class WebResearchSpecialist:
    name = "web_research"
    capability = "web_research"

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
        self._web_answerer = web_answerer if web_answerer is not None else (openai_web_search_from_env() if use_env_web_answerer else None)

    async def run(self, state: WorkingState) -> StepRecord:
        prompt = state.request.prompt.strip()
        if self._web_answerer is not None:
            direct_answer = await self._openai_web_answer(state)
            if direct_answer and not self._looks_like_failed_search(direct_answer):
                state.set_note("answer_candidate", direct_answer)
                state.set_note("research_spans", (*state.get_note("research_spans", ()), direct_answer[:3000]))
                return StepRecord(
                    capability=self.capability,
                    summary="answered with OpenAI web_search_preview",
                    outputs={"answer_candidate": direct_answer, "source": "openai_web_search_preview"},
                )

        prompt_urls = extract_urls(prompt, limit=4)
        query = await self._build_query(state) if self._llm is not None else ""
        if not query:
            query = prompt[:300]

        results: list[dict[str, str]] = []
        if query:
            results.extend(await self._web.search(query, limit=5))
        for url in prompt_urls:
            if not any(r.get("url") == url for r in results):
                results.insert(0, {"title": url, "url": url, "snippet": "URL from prompt"})

        fetched: list[dict[str, str]] = []
        source_urls: list[str] = []
        for item in results[:4]:
            url = item.get("url", "")
            if not url or url in source_urls:
                continue
            text = await self._web.fetch_text(url, limit_chars=6000)
            if text:
                fetched.append({"url": url, "title": item.get("title", ""), "text": text[:6000]})
                source_urls.append(url)

        spans = list(state.get_note("research_spans", ()))
        for item in results[:5]:
            snippet = item.get("snippet") or item.get("title") or item.get("url") or ""
            if snippet:
                spans.append(f"Search result: {snippet} ({item.get('url', '')})")
        for page in fetched:
            spans.append(f"Fetched source {page['url']}: {page['text'][:1200]}")
        state.set_note("research_spans", tuple(spans))
        existing_sources = list(state.get_note("source_urls", ()))
        for url in source_urls:
            if url not in existing_sources:
                existing_sources.append(url)
        state.set_note("source_urls", tuple(existing_sources))

        answer = ""
        used_llm = False
        if self._llm is not None and (results or fetched):
            answer = await self._llm_answer(state, results, fetched)
            used_llm = bool(answer)
        if answer:
            state.set_note("answer_candidate", answer)
        elif results:
            # Keep a minimal candidate so the composer has something grounded
            # even when no LLM is configured.
            state.set_note("answer_candidate", results[0].get("snippet") or results[0].get("title") or "")

        return StepRecord(
            capability=self.capability,
            summary=f"searched web ({len(results)} result(s), fetched {len(fetched)} page(s))",
            outputs={
                "query": query,
                "results": results[:5],
                "fetched_urls": [p["url"] for p in fetched],
                "answer_candidate": state.get_note("answer_candidate", ""),
                "source": "llm" if used_llm else "web",
            },
        )

    @staticmethod
    def _looks_like_failed_search(answer: str) -> bool:
        lowered = answer.lower()
        failure_markers = (
            "couldn't locate",
            "could not locate",
            "couldn't find",
            "could not find",
            "i could not locate",
            "i don't have",
            "not easily identifiable",
            "would need access",
            "please provide relevant sources",
        )
        return any(marker in lowered for marker in failure_markers)

    async def _openai_web_answer(self, state: WorkingState) -> str:
        assert self._web_answerer is not None
        system = "\n\n".join(
            chunk
            for chunk in (
                load_prompt("system"),
                "You are a careful Mind2Web-style web research agent. Use web_search_preview to find current primary sources. Answer in extraction-friendly markdown tables when fields are requested. Put source URLs beside critical claims. Do not use hidden evaluator ground truth or task IDs.",
            )
            if chunk
        )
        prompt = (
            "Solve this web research task using OpenAI web_search_preview. "
            "Return only the final answer, with source URLs for each critical field.\n\n"
            f"Task:\n{state.request.prompt}"
        )
        try:
            return await self._web_answerer.answer(prompt=prompt, system=system, max_tokens=1400)
        except Exception:
            return ""

    async def _build_query(self, state: WorkingState) -> str:
        assert self._llm is not None
        user_text = (
            "Turn this benchmark task into one concise web search query. "
            "Return only the query text, no JSON.\n\nTask:\n" + state.request.prompt
        )
        try:
            text = await self._llm.complete(
                messages=[ChatMessage("system", load_prompt("system")), ChatMessage("user", user_text)],
                tag="web_query",
                max_tokens=80,
            )
        except Exception:
            return ""
        return (text or "").strip().strip('"')[:300]

    async def _llm_answer(
        self,
        state: WorkingState,
        results: list[dict[str, str]],
        fetched: list[dict[str, str]],
    ) -> str:
        assert self._llm is not None
        source_payload = {
            "search_results": results[:5],
            "fetched_pages": [
                {"url": page["url"], "title": page.get("title", ""), "text": page["text"][:3500]}
                for page in fetched[:4]
            ],
        }
        user_text = (
            "Use the provided web search results and fetched source text to answer the task. "
            "Include source URLs in the answer when the task asks for provenance. "
            "If evidence is insufficient, say what is missing.\n\n"
            f"Task:\n{state.request.prompt}\n\nSources JSON:\n"
            + json.dumps(source_payload, ensure_ascii=False, indent=2)
        )
        try:
            text = await self._llm.complete(
                messages=[
                    ChatMessage("system", "\n\n".join(chunk for chunk in (load_prompt("system"), "You are a careful web research agent. Never invent facts not supported by fetched sources.") if chunk)),
                    ChatMessage("user", user_text),
                ],
                tag="web_research",
                max_tokens=900,
            )
        except Exception:
            return ""
        data = extract_json(text)
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            # Tests and future structured prompts may return JSON; preserve only
            # the user-visible answer as candidate.
            return data["answer"].strip()
        return (text or "").strip()
