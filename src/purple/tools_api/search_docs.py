"""``search_docs`` tool — keyword search over context + attachments."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..runtime.tool import ToolContext, ToolResult
from ..tools import chunk_text, search_chunks

_URL = re.compile(r"https?://\S+", re.IGNORECASE)


def _gather_sources(ctx: ToolContext, include_prompt: bool) -> list[str]:
    sources: list[str] = []
    prompt = ctx.request.prompt or ""
    if include_prompt and any(
        label in prompt.lower() for label in ("context:", "passage:", "document:")
    ):
        sources.append(prompt)
    sources.extend(ctx.request.context)
    for att in ctx.request.attachments:
        if att.text:
            sources.append(att.text)
    return sources


class SearchDocsTool:
    name = "search_docs"
    description = (
        "Find evidence spans in the user-provided context and attachments. "
        "No outbound HTTP."
    )
    arg_schema: Mapping[str, str] = {
        "query": "optional override; defaults to the user prompt",
        "limit": "optional integer cap on returned spans (default 6)",
    }

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            query = ctx.request.prompt or ""
        limit_value = args.get("limit", 6)
        try:
            limit = max(1, int(limit_value))
        except (TypeError, ValueError):
            limit = 6

        sources = _gather_sources(ctx, include_prompt=True)
        urls: list[str] = []
        for src in [ctx.request.prompt or "", *sources]:
            urls.extend(_URL.findall(src))
        urls = list(dict.fromkeys(urls))

        chunks: list[str] = []
        for src in sources:
            chunks.extend(chunk_text(src))

        spans = search_chunks(chunks, query, limit=limit)
        if not spans and chunks:
            spans = chunks[: min(3, limit)]

        if not sources:
            summary = "no fetch: no context or attachments to read"
        else:
            summary = (
                f"extracted {len(spans)} span(s) from {len(sources)} source(s)"
            )
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=summary,
            observation=" || ".join(spans)[:600] if spans else "(no matching spans)",
            outputs={
                "spans": list(spans),
                "urls_detected": list(urls),
                "fetched": False,
            },
        )


__all__ = ["SearchDocsTool"]
