"""``web_search`` tool — search + optional one-shot OpenAI web_search_preview answer."""

from __future__ import annotations

from typing import Any, Mapping

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
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            query = prompt[:300]
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
        if self._llm is not None and (not args.get("query")):
            built = await self._build_query(prompt)
            if built:
                query = built

        results: list[dict[str, str]] = []
        if query:
            results.extend(await self._web.search(query, limit=limit))
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
            summary=f"web_search returned {len(results)} result(s)",
            observation=spans[0] if spans else "(no results)",
            outputs={
                "query": query,
                "results": results[:limit],
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


__all__ = ["WebSearchTool"]
