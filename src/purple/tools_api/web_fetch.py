"""``web_fetch`` tool — fetch the visible text of a URL with a byte cap."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

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


def _looks_like_binary_or_garbled_text(text: str) -> bool:
    """Reject decoded binary/compressed payloads before treating them as evidence.

    Some HTTP/PDF endpoints return compressed bytes decoded as latin/control
    characters by a simple fetcher.  Those blobs should be consumed as failed
    fetches, not passed to verifier/finalizer as source text.
    """

    sample = (text or "")[:2400]
    if not sample.strip():
        return True
    control = sum(1 for ch in sample if (ord(ch) < 32 and ch not in "\n\r\t"))
    if control / max(1, len(sample)) > 0.02:
        return True
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    if printable / max(1, len(sample)) < 0.88:
        return True
    words = _WORD.findall(sample)
    unicode_letters = sum(1 for ch in sample if ch.isalpha())
    # Non-English public institutional sites may be Arabic/CJK/etc.  Do not
    # classify them as binary just because the ASCII-word extractor sees few
    # Latin tokens; rely on printable/control checks plus Unicode letters.
    if len(sample) > 500 and len(words) < 20 and unicode_letters < 80:
        return True
    # Extracted PDFs often contain table-of-contents dotted leaders or page
    # separators.  They are source text, not binary garbage, when they have
    # many ordinary words and no control-character damage.  Do not reject them
    # only because punctuation lowers the alnum/space ratio.
    if len(words) >= 50:
        return False
    alpha_num = sum(1 for ch in sample if ch.isalnum() or ch.isspace())
    if len(sample) > 500 and alpha_num / max(1, len(sample)) < 0.45:
        return True
    return False


def _rank_detected_urls(urls: list[str], *, source_url: str, context: str, limit: int = 16) -> list[str]:
    """Rank extracted source-page links before the controller spends fetch budget.

    HTML link preservation can expose many anchors from one page: share widgets,
    newest news items, pagination, archives, and documents.  A raw first-N URL
    slice tends to waste the multi-clue budget on social links or latest items,
    while older/public archive pages that may contain year-specific evidence are
    pushed past the cap.  Keep this generic: rank only by URL/source-domain shape
    and words/years present in the public task/context, never by benchmark ids or
    expected answers.
    """

    if not urls:
        return []
    source_host = _host(source_url)
    context_lower = (context or "").lower()
    context_tokens = _tokens(context_lower)
    old_year_requested = any(int(year) <= 2015 for year in re.findall(r"\b(19\d{2}|20\d{2})\b", context_lower))
    source_query = parse_qs(urlparse(source_url).query)
    source_page_values = source_query.get("page") or source_query.get("p") or []
    source_page_num = 0
    if source_page_values:
        try:
            source_page_num = int(str(source_page_values[0]))
        except (TypeError, ValueError):
            source_page_num = 0
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, candidate in enumerate(urls):
        if candidate in seen:
            continue
        seen.add(candidate)
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        lowered = candidate.lower()
        if any(token in lowered for token in ("facebook.com/sharer", "twitter.com/intent", "api.whatsapp.com/send", "linkedin.com/sharing", "wa.me/?text=", "mailto:")):
            continue
        score = 0
        if source_host and _host(candidate) == source_host:
            score += 40
        if any(token in lowered for token in ("news", "archive", "archives", "event", "media", "press", "report", "document", "download")):
            score += 20
        if re.search(r"\.(?:pdf|docx?|xlsx?|csv)(?:$|[?#])", lowered):
            score += 24
        is_numbered_article = bool(re.search(r"/(?:news|article|event|press|media)/\d+(?:$|[?#])", lowered))
        if is_numbered_article:
            score += 12
            if source_page_num:
                # Once an archive/list page has been opened, follow its concrete
                # article links before crawling sibling pagination pages.
                score += 30
        query = parse_qs(parsed.query)
        page_values = query.get("page") or query.get("p") or []
        page_num = 0
        if page_values:
            try:
                page_num = int(str(page_values[0]))
            except (TypeError, ValueError):
                page_num = 0
        if page_num:
            score += 8
            if old_year_requested and not source_page_num:
                # Older clue years often live behind archive pagination.  Prefer
                # bounded late-page probes from a top-level listing over draining
                # only the newest articles; after a page is opened, concrete
                # article links from that page outrank more sibling pages.
                score += min(25, page_num)
            elif old_year_requested and source_page_num:
                # Archive/list pages can themselves be only a stepping stone: a
                # page around the current cursor may be closer to the requested
                # historical year than the newest detail links on the current
                # page.  Keep a bounded window of neighbouring/following pages in
                # the detected URL set so the controller can reach the right
                # archive page, but keep concrete article links ahead of generic
                # pagination once that page has been opened.
                delta = page_num - source_page_num
                if 1 <= delta <= 3:
                    score += 36
        overlap = len(_tokens(lowered.replace("%20", " ")) & context_tokens)
        score += min(18, overlap * 3)
        if any(year in lowered for year in re.findall(r"\b(?:19\d{2}|20\d{2})\b", context_lower)):
            score += 12
        scored.append((score, order, candidate))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [url for _, _, url in scored[:limit]]


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


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
        if _looks_like_binary_or_garbled_text(text):
            return ToolResult(
                tool_call_id="",
                ok=False,
                summary=f"web_fetch returned non-text/gibberish body for {url}",
                observation="(non-text body)",
                outputs={"fetched": False, "url": url},
                error="non-text-body",
            )
        excerpt = relevant_excerpt(text, _context_query(ctx), limit=3600)
        span = f"Fetched source {url}: {excerpt}"
        detected_urls = _rank_detected_urls(
            [u for u in extract_urls(text, limit=80) if u != url],
            source_url=url,
            context=_context_query(ctx),
            limit=16,
        )
        outputs = {
            "fetched": True,
            "url": url,
            "text": text,
            "fetched_urls": [url],
            "source_urls": [url],
            "urls_detected": detected_urls,
            "url_priorities": {candidate: len(detected_urls) - idx for idx, candidate in enumerate(detected_urls)},
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
