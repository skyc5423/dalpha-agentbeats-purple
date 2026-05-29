"""``web_fetch`` tool — fetch the visible text of a URL with a byte cap."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from ..runtime.tool import ToolContext, ToolResult
from ..tools import StdlibWebClient, WebClient, extract_urls


_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset({"the", "and", "for", "with", "what", "which", "who", "how", "was", "were", "from", "that", "this", "into", "page", "source"})


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if len(w) >= 3 and w.lower() not in _STOPWORDS}


def _context_query(ctx: ToolContext) -> str:
    parts: list[str] = [ctx.request.prompt or ""]
    if hasattr(ctx.notes, "get"):
        for key in ("answer_candidate", "minimum_success_condition"):
            value = ctx.notes.get(key)
            if isinstance(value, str):
                parts.append(value)
        requirements = ctx.notes.get("requirements") or ctx.notes.get("required_outputs")
        if requirements:
            try:
                parts.append(json.dumps(requirements, ensure_ascii=False))
            except TypeError:
                parts.append(str(requirements))
    return "\n".join(parts)


def relevant_excerpt(text: str, query: str, *, limit: int = 3600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    query_tokens = _tokens(query)
    if not query_tokens:
        return text[:limit].rstrip()
    best_start = 0
    best_score = -1
    window = min(limit, 2200)
    lower = text.lower()
    starts = {0}
    for tok in query_tokens:
        idx = lower.find(tok)
        while idx != -1 and len(starts) < 80:
            starts.add(max(0, idx - window // 2))
            idx = lower.find(tok, idx + len(tok))
    for start in starts:
        excerpt = text[start : start + window]
        score = len(_tokens(excerpt) & query_tokens)
        if score > best_score:
            best_score = score
            best_start = start
    excerpt = text[best_start : best_start + window].strip()
    if best_start > 0:
        excerpt = "... " + excerpt
    if best_start + window < len(text):
        excerpt = excerpt.rstrip() + " ..."
    return excerpt[:limit]


class WebFetchTool:
    name = "web_fetch"
    description = "Fetch a single URL and return its visible text, byte-capped."
    arg_schema: Mapping[str, str] = {
        "url": "the URL to fetch",
        "limit_chars": "optional byte cap on returned text (default 6000)",
    }

    def __init__(self, *, web_client: WebClient | None = None) -> None:
        self._web = web_client or StdlibWebClient()

    async def run(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary="web_fetch requires an absolute http(s) url",
                observation="missing url",
                outputs={"fetched": False},
                error="missing url",
            )
        try:
            limit_chars = max(256, int(args.get("limit_chars", 6000)))
        except (TypeError, ValueError):
            limit_chars = 6000

        text = await self._web.fetch_text(url, limit_chars=limit_chars)
        if not text:
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary=f"web_fetch returned empty body for {url}",
                observation="(empty)",
                outputs={"fetched": False, "url": url},
                error="empty",
            )
        excerpt = relevant_excerpt(text, _context_query(ctx), limit=3600)
        span = f"Fetched source {url}: {excerpt}"
        outputs = {
            "fetched": True,
            "url": url,
            "text": text,
            "fetched_urls": [url],
            "source_urls": [url],
            "urls_detected": [u for u in extract_urls(text, limit=12) if u != url],
            "spans": [span],
            "fetched_pages": [{"url": url, "text": text[:limit_chars]}],
        }
        if "github.com/" in url and "/commit/" in url:
            metadata_lines: list[str] = []
            for line in text.splitlines():
                if line.strip() == "Message:":
                    break
                if line.strip():
                    metadata_lines.append(line)
            outputs["answer_candidate"] = "\n".join(metadata_lines)[:1800]
        elif "github.com/" in url and "/pull/" in url:
            outputs["answer_candidate"] = text[:2400]
        elif "api.github.com/repos/" in url and "/commits" in url:
            outputs["answer_candidate"] = text[:2400]
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary=f"fetched {len(text)} chars from {url}",
            observation=text[:600],
            outputs=outputs,
        )


__all__ = ["WebFetchTool"]
