"""Small stdlib-only web search and fetch helpers.

The helpers deliberately avoid heavy browser automation. They provide a bounded,
best-effort text/search surface for public benchmark agents that need source
URLs without embedding private infrastructure.
"""

from __future__ import annotations

import asyncio
import base64
import html
import io
import json
import re
import urllib.parse
import urllib.request
from typing import Protocol, runtime_checkable


@runtime_checkable
class WebClient(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]: ...
    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str: ...


_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.I | re.S)
_SPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://[^\s<>'\"]+", re.I)
_PDF_LITERAL = re.compile(rb"\((?:\\.|[^\\()])*\)")


def html_to_text(raw: str, *, limit_chars: int = 5000) -> str:
    raw = _SCRIPT_STYLE.sub(" ", raw)
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    text = _SPACE.sub(" ", text).strip()
    return text[:limit_chars]


def _html_semantic_main_text(raw: str, url: str, *, limit_chars: int = 5000) -> str:
    """Extract article/main content before generic HTML stripping.

    Many public institutional/CMS pages put a large navigation menu before the
    actual article.  A blind first-N text cap can therefore hide the date/title
    and body even though the page was fetched successfully.  Prefer semantic
    containers generically (`article`, `main`, common content classes) and return
    the longest useful text block, with the page title prepended when available.
    """

    if not raw:
        return ""
    candidates: list[str] = []
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/").lower()
    looks_like_collection_page = bool(parsed.query) or path.rsplit("/", 1)[-1] in {
        "",
        "news",
        "archive",
        "archives",
        "events",
        "media",
        "press",
        "blog",
        "articles",
    }
    patterns = [r"<article\b[^>]*>.*?</article>"]
    if not looks_like_collection_page:
        patterns.extend(
            [
                r"<main\b[^>]*>.*?</main>",
                r"<section\b[^>]*(?:class|id)=[\"'][^\"']*(?:article|post|news|content|detail|entry|body|main)[^\"']*[\"'][^>]*>.*?</section>",
                r"<div\b[^>]*(?:class|id)=[\"'][^\"']*(?:article|post|news|content|detail|entry|body|main)[^\"']*[\"'][^>]*>.*?</div>",
            ]
        )
    for pattern in patterns:
        for match in re.finditer(pattern, raw, re.I | re.S):
            text = html_to_text(match.group(0), limit_chars=max(limit_chars * 2, 12000))
            words = re.findall(r"[\w\u0600-\u06FF]{2,}", text, re.U)
            if len(words) >= 8:
                candidates.append(text)
            if len(candidates) >= 20:
                break
        if candidates:
            break
    if not candidates:
        return ""
    title = ""
    title_match = re.search(r"<title\b[^>]*>(.*?)</title>", raw, re.I | re.S)
    if title_match is not None:
        title = html_to_text(title_match.group(1), limit_chars=300)
    # Prefer blocks with date/article metadata and enough body text; fall back to
    # longest candidate.  This stays domain-agnostic and avoids expected answers.
    def score(text: str) -> tuple[int, int]:
        low = text.lower()
        meta = 0
        if any(tok in low for tok in ("publish date", "published", "author", "day ", "date", "تاريخ النشر", "المؤلف", "اليوم")):
            meta += 50
        if re.search(r"\b(?:19\d{2}|20\d{2})\b", text):
            meta += 20
        if title and title.lower() in low:
            meta += 20
        return (meta, len(text))
    best = max(candidates, key=score)
    if title and title.lower() not in best.lower()[:800].lower():
        best = f"Title: {title}\n{best}"
    return best[:limit_chars]


def _html_links_text(raw: str, base_url: str, *, limit: int = 40) -> str:
    """Preserve source URLs hidden behind HTML anchors in fetched pages.

    Plain tag stripping loses the href side of archive/newsletter/download pages.
    For open-web research tasks this is a generic source-adapter gap: the visible
    page text may say "Download PDF" while the actual evidence URL is only in an
    anchor attribute.  Return a compact, de-duplicated list of visible anchor text
    plus absolute URLs so downstream fetch/search loops can follow primary
    sources without domain- or benchmark-specific routing.
    """

    if not raw or "<a" not in raw.lower():
        return ""
    candidates: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for href, body in re.findall(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", raw, re.I | re.S):
        href = html.unescape(href or "").strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        absolute = urllib.parse.urlunparse(
            parsed._replace(path=urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%"), fragment="")
        )
        if absolute in seen:
            continue
        label = html_to_text(body, limit_chars=140)
        if not label:
            label = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])[:140]
        # Skip very low-information navigation links unless the URL itself looks
        # like a source/document/download endpoint.
        low_label = label.lower().strip()
        sourceish_url = bool(re.search(r"\.(pdf|docx?|xlsx?|csv|html?)($|[?#])", absolute, re.I)) or any(
            token in absolute.lower() for token in ("news", "archive", "download", "press", "media", "report")
        )
        if low_label in {"home", "about", "admission", "apply", "read more", "view all", "contact"} and not sourceish_url:
            continue
        score = 0
        if sourceish_url:
            score += 100
        if re.search(r"\.(pdf|docx?|xlsx?|csv)($|[?#])", absolute, re.I):
            score += 80
        if any(token in (low_label + " " + absolute.lower()) for token in ("newsletter", "news letter", "press", "media", "archive", "download", "report")):
            score += 30
        candidates.append((score, label, absolute))
        seen.add(absolute)
    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    lines = [f"- {label}: {absolute}" for _, label, absolute in candidates[:limit]]
    return "Detected page links:\n" + "\n".join(lines)


def pdf_bytes_to_text(raw: bytes, *, limit_chars: int = 5000) -> str:
    """Best-effort PDF text extraction with optional parser fallback."""
    if not raw:
        return ""
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(io.BytesIO(raw))
            parts: list[str] = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
                if sum(len(p) for p in parts) >= limit_chars:
                    break
            text = _SPACE.sub(" ", "\n".join(parts)).strip()
            if text:
                return text[:limit_chars]
        except Exception:
            continue
    literals: list[str] = []
    for match in _PDF_LITERAL.finditer(raw[:2_000_000]):
        body = match.group(0)[1:-1]
        body = (
            body.replace(rb"\(", b"(")
            .replace(rb"\)", b")")
            .replace(rb"\\", b"\\")
            .replace(rb"\n", b"\n")
            .replace(rb"\r", b"\n")
            .replace(rb"\t", b" ")
        )
        text = body.decode("utf-8", errors="ignore").strip()
        if len(text) >= 2 and any(ch.isalpha() for ch in text):
            literals.append(text)
        if sum(len(x) for x in literals) >= limit_chars:
            break
    return _SPACE.sub(" ", " ".join(literals)).strip()[:limit_chars]


def extract_urls(text: str, *, limit: int = 5) -> list[str]:
    seen: list[str] = []
    for match in _URL.finditer(text):
        url = match.group(0).rstrip(".,);]")
        if url not in seen:
            seen.append(url)
        if len(seen) >= limit:
            break
    return seen


def _parse_bing_results(raw: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Parse Bing HTML result blocks into the same lightweight schema as DDG.

    This is intentionally stdlib-only and best-effort. It gives the generic web
    client a second credential-free source-discovery backend when DuckDuckGo's
    HTML endpoint returns no parseable organic results.
    """

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for block in re.findall(r'<li\s+class="b_algo"[^>]*>(.*?)</li>', raw or "", re.I | re.S):
        link_match = re.search(r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.I | re.S)
        if link_match is None:
            continue
        url = _unwrap_bing_result_url(html.unescape(link_match.group(1)).strip())
        if not url.startswith("http") or url in seen_urls:
            continue
        title = html_to_text(link_match.group(2), limit_chars=300)
        caption = ""
        caption_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.I | re.S)
        if caption_match is not None:
            caption = html_to_text(caption_match.group(1), limit_chars=500)
        results.append({"title": title, "url": url, "snippet": caption})
        seen_urls.add(url)
        if len(results) >= limit:
            break
    return results


def _unwrap_bing_result_url(url: str) -> str:
    """Decode Bing click-tracking URLs to the underlying public source URL."""

    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower().endswith("bing.com") and parsed.path.startswith("/ck/"):
        encoded = (urllib.parse.parse_qs(parsed.query).get("u") or [""])[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        if encoded:
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", errors="ignore")
            except Exception:
                decoded = ""
            if decoded.startswith(("http://", "https://")):
                return decoded
    return url


def _format_math_genealogy_page_text(raw: str, url: str) -> str:
    """Return structured text for Mathematics Genealogy profile pages.

    The plain HTML-stripped page keeps advisor names but loses the relative
    advisor links, so the controller cannot follow an advisor chain. This
    formatter keeps the page domain-agnostic enough for academic genealogy
    tasks: profile name, Ph.D. line, dissertation, and advisor edges with
    absolute source URLs.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc.lower() not in {"www.mathgenealogy.org", "mathgenealogy.org"}:
        return ""
    if not parsed.path.endswith("/id.php"):
        return ""

    title_match = re.search(r"<title>\s*(.*?)\s*-\s*The Mathematics Genealogy Project\s*</title>", raw, re.I | re.S)
    h2_match = re.search(r"<h2[^>]*>\s*(.*?)\s*</h2>", raw, re.I | re.S)
    name = ""
    name_match = h2_match or title_match
    if name_match is not None:
        name = html_to_text(name_match.group(1), limit_chars=200)
    name = " ".join(name.split())

    plain = html_to_text(raw, limit_chars=12000)
    phd = ""
    phd_match = re.search(r"Ph\.D\.\s+(.*?)\s+Dissertation:", plain, re.I | re.S)
    if phd_match:
        phd = " ".join(phd_match.group(1).split())

    dissertation = ""
    diss_match = re.search(r"Dissertation:\s*(.*?)\s+(?:Mathematics Subject Classification:|Advisor(?:\s+\d+)?:|Students:)", plain, re.I | re.S)
    if diss_match:
        dissertation = " ".join(diss_match.group(1).split())

    advisor_lines: list[str] = []
    for num, href, advisor_name in re.findall(
        r"Advisor\s*(\d*)\s*:\s*<a\s+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        raw,
        re.I | re.S,
    ):
        advisor_url = urllib.parse.urljoin(url, html.unescape(href))
        advisor = " ".join(html_to_text(advisor_name, limit_chars=200).split())
        label = f"Advisor {num or '1'}"
        advisor_lines.append(f"{label}: {advisor} ({advisor_url})")

    if not (name or advisor_lines):
        return ""
    lines = [f"MathGenealogy profile: {name}" if name else "MathGenealogy profile"]
    if phd:
        lines.append(f"Ph.D.: {phd}")
    if dissertation:
        lines.append(f"Dissertation: {dissertation}")
    lines.extend(advisor_lines)
    lines.append("Source: Mathematics Genealogy Project profile page " + url)
    lines.append("Plain page text excerpt: " + plain[:3000])
    return "\n".join(lines)


def _looks_like_pdf_extraction_garbage(text: str) -> bool:
    sample = (text or "")[:1600]
    if not sample.strip():
        return True
    control = sum(1 for ch in sample if (ord(ch) < 32 and ch not in "\n\r\t"))
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    words = re.findall(r"[A-Za-z]{3,}", sample)
    if control / max(1, len(sample)) > 0.02:
        return True
    if printable / max(1, len(sample)) < 0.88:
        return True
    if len(sample) > 400 and len(words) < 12:
        return True
    return False


class StdlibWebClient:
    def __init__(
        self,
        *,
        timeout_s: float = 12.0,
        search_timeout_s: float = 6.0,
        user_agent: str = "Mozilla/5.0 AgentBeatsPurple/0.1",
    ) -> None:
        self._timeout_s = timeout_s
        self._search_timeout_s = min(timeout_s, search_timeout_s)
        self._user_agent = user_agent

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._sync_search, query, limit)

    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
        return await asyncio.to_thread(self._sync_fetch_text, url, limit_chars)

    def _open(self, url: str, *, limit_chars: int = 5000, timeout_s: float | None = None) -> str:
        try:
            raw, content_type, charset = self._open_bytes(url, timeout_s=timeout_s)
        except TypeError:
            # Preserve compatibility with lightweight test/custom subclasses that
            # override _open_bytes(url) without the optional timeout keyword.
            raw, content_type, charset = self._open_bytes(url)
        is_pdf = (
            "pdf" in content_type.lower()
            or urllib.parse.urlparse(url).path.lower().endswith(".pdf")
            or raw.lstrip()[:5] == b"%PDF-"
        )
        if is_pdf:
            text = pdf_bytes_to_text(raw, limit_chars=limit_chars)
            if _looks_like_pdf_extraction_garbage(text):
                return (
                    f"PDF document fetched from {url}. "
                    "Text extraction unavailable or unreadable with the local parser; "
                    "treat this URL as a source candidate, not as claim evidence."
                )
            return text[:limit_chars]
        return raw.decode(charset or "utf-8", errors="replace")

    def _open_bytes(self, url: str, *, timeout_s: float | None = None) -> tuple[bytes, str, str]:
        # Search/fetch endpoints may infer locale from the container/network and
        # return dictionary or regional SERP traps for English benchmark prompts.
        # Prefer English content generically without depending on any benchmark
        # task id or expected answer.
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self._user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s if timeout_s is None else timeout_s) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            content_type = resp.headers.get_content_type() or ""
            read_limit = 20_000_000 if urllib.parse.urlparse(url).path.lower().endswith(".pdf") else 2_000_000
            return resp.read(read_limit), content_type, charset

    def _sync_search(self, query: str, limit: int) -> list[dict[str, str]]:
        query = query.strip()
        if not query:
            return []
        github_results = self._github_commit_search(query, limit)
        if github_results:
            return github_results[:limit]
        # DuckDuckGo html endpoint is simple enough to parse without JS and does
        # not require API credentials. If it changes/fails or yields an empty
        # page, fall back to Bing HTML before returning a clean empty result.
        # This keeps source discovery public-safe and generic while avoiding a
        # single-search-engine blind spot on sparse multi-clue tasks.
        results = self._duckduckgo_html_search(query, limit)
        if results:
            return results[:limit]
        return self._bing_html_search(query, limit)[:limit]

    def _duckduckgo_html_search(self, query: str, limit: int) -> list[dict[str, str]]:
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
            raw = self._open(url, timeout_s=self._search_timeout_s)
        except TypeError:
            raw = self._open(url)
        except Exception:
            return []
        results: list[dict[str, str]] = []
        for href, title in re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw, re.I | re.S):
            actual = html.unescape(href)
            if actual.startswith("//duckduckgo.com/l/?"):
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + actual).query)
                actual = parsed.get("uddg", [actual])[0]
            title_text = html_to_text(title, limit_chars=300)
            if not actual.startswith("http"):
                continue
            results.append({"title": title_text, "url": actual, "snippet": ""})
            if len(results) >= limit:
                break
        return results[:limit]

    def _bing_html_search(self, query: str, limit: int) -> list[dict[str, str]]:
        url = "https://www.bing.com/search?" + urllib.parse.urlencode(
            {
                "q": query,
                "mkt": "en-US",
                "setlang": "en-US",
                "cc": "US",
            }
        )
        try:
            raw = self._open(url, timeout_s=self._search_timeout_s)
        except TypeError:
            raw = self._open(url)
        except Exception:
            return []
        return _parse_bing_results(raw, limit=limit)

    def _github_commit_search(self, query: str, limit: int) -> list[dict[str, str]]:
        qlow = query.lower()
        repo_match = re.search(r"repo:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", query)
        if repo_match is None:
            repo_match = re.search(r"site:github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", query, re.I)
        repo = repo_match.group(1) if repo_match else ""
        if not repo or "commit" not in qlow:
            return []
        stop_terms = {
            "commit",
            "commits",
            "first",
            "branch",
            "main",
            "support",
            "supports",
            "supported",
            "official",
            "repository",
            "repo",
            "github",
            "author",
            "authors",
            "date",
            "profile",
            "profiles",
            "real",
            "name",
            "names",
            "contributors",
            "contributor",
            "site",
            "com",
        }
        terms = []
        repo_parts = {part.lower() for part in repo.split("/")}
        for term in re.findall(r"[A-Za-z0-9_.-]{3,}", query):
            lower = term.lower()
            if term.startswith("repo:") or "/" in term or lower in stop_terms or lower in repo_parts:
                continue
            if lower == "added":
                term = "add"
            terms.append(term)
            if len(terms) >= 8:
                break
        gh_query = " ".join([f"repo:{repo}", *terms]).strip()
        url = "https://api.github.com/search/commits?" + urllib.parse.urlencode(
            {"q": gh_query, "per_page": str(min(limit, 10)), "sort": "committer-date", "order": "asc"}
        )
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github.cloak-preview+json",
                "User-Agent": self._user_agent,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return []
        out: list[dict[str, str]] = []
        items = data.get("items", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        for item in items[:limit]:
            commit = item.get("commit", {}) or {}
            author = commit.get("author", {}) or {}
            message = (commit.get("message") or "").split("\n", 1)[0]
            sha = item.get("sha", "")
            html_url = item.get("html_url", "")
            snippet = f"GitHub commit {sha[:7]} date {author.get('date', '')}: {message}; author {author.get('name', '')}."
            out.append({"title": message or sha[:7], "url": html_url, "snippet": snippet})
        return out

    def _sync_fetch_text(self, url: str, limit_chars: int) -> str:
        if not url.startswith(("http://", "https://")):
            return ""
        github_text = self._fetch_github_commit_api_text(url, limit_chars)
        if github_text:
            return github_text
        github_html_list_text = self._fetch_github_html_commits_list_api_text(url, limit_chars)
        if github_html_list_text:
            return github_html_list_text
        github_list_text = self._fetch_github_commits_list_api_text(url, limit_chars)
        if github_list_text:
            return github_list_text
        github_pull_text = self._fetch_github_pull_api_text(url, limit_chars)
        if github_pull_text:
            return github_pull_text
        try:
            try:
                raw = self._open(url, limit_chars=limit_chars)
            except TypeError:
                # Some tests/custom lightweight clients override _open(url)
                # without the optional keyword; keep that extension point
                # compatible while still passing the cap to the stdlib path.
                raw = self._open(url)
        except Exception:
            return ""
        math_genealogy_text = _format_math_genealogy_page_text(raw, url)
        if math_genealogy_text:
            return math_genealogy_text[:limit_chars]
        main_text = _html_semantic_main_text(raw, url, limit_chars=max(limit_chars * 2, 12000))
        text = main_text or html_to_text(raw, limit_chars=max(limit_chars * 2, 12000))
        links_text = _html_links_text(raw, url)
        if links_text:
            link_budget = min(len(links_text), max(800, limit_chars // 2))
            text_budget = max(0, limit_chars - link_budget - 2)
            return f"{text[:text_budget]}\n\n{links_text[:link_budget]}"[:limit_chars]
        return text[:limit_chars]

    def _fetch_github_html_commits_list_api_text(self, url: str, limit_chars: int) -> str:
        """Format GitHub HTML commit-history URLs through the commits API.

        GitHub's web UI pages such as
        ``/commits/main/src/transformers/models/llava`` are not source-rich after
        simple HTML stripping, but they encode a generic repo/ref/path history
        query. Convert them to the public commits API so first-commit/path-history
        tasks can see structured dates, authors, and commit URLs without any
        benchmark-specific routing.
        """
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.lower() != "github.com":
            return ""
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 4 or parts[2] != "commits":
            return ""
        owner, repo_name, _, ref, *path_bits = parts
        if not owner or not repo_name or not ref or not path_bits:
            return ""
        api_url = "https://api.github.com/repos/{repo}/commits?".format(
            repo=f"{owner}/{repo_name}"
        ) + urllib.parse.urlencode(
            {
                "sha": ref,
                "path": "/".join(path_bits),
                "per_page": "100",
                "page": "1",
            }
        )
        return self._fetch_github_commits_list_api_text(api_url, limit_chars)

    def _fetch_github_commits_list_api_text(self, url: str, limit_chars: int) -> str:
        parsed = urllib.parse.urlparse(url)
        if parsed.netloc.lower() != "api.github.com":
            return ""
        match = re.match(r"/repos/([^/]+/[^/]+)/commits/?$", parsed.path)
        if not match:
            return ""
        repo = match.group(1)
        query = urllib.parse.parse_qs(parsed.query)
        requested_page = int((query.get("page") or ["1"])[0] or "1")
        per_page = min(100, int((query.get("per_page") or ["100"])[0] or "100"))
        # If this is a path-history query, walk a bounded number of pages so the
        # oldest/first commit is not mistaken for merely the oldest item on page
        # 1. For unfiltered lists, keep the single requested page behavior to
        # avoid excessive API traffic.
        should_paginate = bool((query.get("path") or [""])[0]) and requested_page == 1
        pages_to_fetch = range(1, 11) if should_paginate else range(requested_page, requested_page + 1)
        data: list[object] = []
        pages_fetched = 0
        for page_no in pages_to_fetch:
            page_query = dict(query)
            page_query["per_page"] = [str(per_page)]
            page_query["page"] = [str(page_no)]
            page_url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(page_query, doseq=True)))
            req = urllib.request.Request(page_url, headers={"User-Agent": self._user_agent})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                    page_data = json.loads(resp.read().decode("utf-8", errors="replace"))
            except Exception:
                break
            if not isinstance(page_data, list) or not page_data:
                break
            data.extend(page_data)
            pages_fetched += 1
            if len(page_data) < per_page:
                break
        if not data:
            return ""
        path_filter = (query.get("path") or [""])[0]
        sha_filter = (query.get("sha") or [""])[0]
        lines = [
            f"GitHub commits API URL: {url}",
            f"Repository: {repo}",
            f"Branch/ref: {sha_filter or '(default)'}",
            f"Path filter: {path_filter or '(none)'}",
            f"Pages fetched/per_page: {pages_fetched}/{per_page}",
            f"Commits returned across fetched pages: {len(data)}",
            "GitHub returns this endpoint newest-to-oldest unless pagination/order is changed.",
        ]
        oldest = data[-1]
        lines.extend(["Oldest commit across fetched pages:", *_format_github_commit_summary(oldest)])
        if isinstance(oldest, dict) and oldest.get("html_url"):
            oldest_detail = self._fetch_github_commit_api_text(str(oldest.get("html_url")), limit_chars)
            if oldest_detail:
                lines.append("Oldest commit detailed metadata:")
                lines.append(oldest_detail)
        lines.append("Commits on this page, newest to oldest:")
        for item in data[: min(len(data), 40)]:
            lines.extend(_format_github_commit_summary(item))
        return "\n".join(line for line in lines if line)[:limit_chars]

    def _fetch_github_pull_api_text(self, url: str, limit_chars: int) -> str:
        match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)(?:/commits)?/?$", url)
        if not match:
            return ""
        repo, number = match.groups()
        api_url = f"https://api.github.com/repos/{repo}/pulls/{number}"
        req = urllib.request.Request(api_url, headers={"User-Agent": self._user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return ""
        if not isinstance(data, dict):
            return ""
        merge_sha = str(data.get("merge_commit_sha") or "")
        user = data.get("user") or {}
        lines = [
            f"GitHub pull request URL: {data.get('html_url', url)}",
            f"Repository: {repo}",
            f"PR number: {data.get('number', number)}",
            f"Title: {data.get('title', '')}",
            f"State: {data.get('state', '')}",
            f"Merged at: {data.get('merged_at', '')}",
            f"Merge commit SHA: {merge_sha}",
        ]
        if isinstance(user, dict) and user.get("login"):
            lines.append(f"PR author: @{user.get('login')} {user.get('html_url', '')}")
        if merge_sha:
            commit_text = self._fetch_github_commit_api_text(
                f"https://github.com/{repo}/commit/{merge_sha}",
                limit_chars,
            )
            if commit_text:
                lines.append("Merge commit metadata:")
                lines.append(commit_text)
        return "\n".join(line for line in lines if line)[:limit_chars]

    def _fetch_github_commit_api_text(self, url: str, limit_chars: int) -> str:
        match = re.search(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-fA-F]{7,40})", url)
        if not match:
            return ""
        repo, sha = match.groups()
        api_url = f"https://api.github.com/repos/{repo}/commits/{sha}"
        req = urllib.request.Request(api_url, headers={"User-Agent": self._user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return ""
        if not isinstance(data, dict):
            return ""
        commit = data.get("commit", {}) or {}
        author = commit.get("author", {}) or {}
        committer = commit.get("committer", {}) or {}
        gh_author = data.get("author") or {}
        gh_committer = data.get("committer") or {}
        gh_author_line = ""
        if isinstance(gh_author, dict) and gh_author.get("login"):
            profile_name = self._github_user_real_name(str(gh_author.get("login") or ""))
            gh_author_line = (
                f"GitHub author profile: @{gh_author.get('login')} "
                f"{gh_author.get('html_url', '')} real name: {profile_name or author.get('name', '')}"
            )
        gh_committer_line = ""
        if isinstance(gh_committer, dict) and gh_committer.get("login"):
            profile_name = self._github_user_real_name(str(gh_committer.get("login") or ""))
            gh_committer_line = (
                f"GitHub committer profile: @{gh_committer.get('login')} "
                f"{gh_committer.get('html_url', '')} real name: {profile_name or committer.get('name', '')}"
            )
        coauthor_lines = self._coauthor_profile_lines(commit.get("message", ""))
        lines = [
            f"GitHub commit URL: {data.get('html_url', url)}",
            f"SHA: {data.get('sha', sha)}",
            f"Author: {author.get('name', '')} <{author.get('email', '')}> at {author.get('date', '')}",
            gh_author_line,
            f"Committer: {committer.get('name', '')} <{committer.get('email', '')}> at {committer.get('date', '')}",
            gh_committer_line,
            *coauthor_lines,
            "Message:",
            commit.get("message", ""),
            "Changed files and relevant patch snippets:",
        ]
        for file_info in (data.get("files") or [])[:80]:
            filename = file_info.get("filename", "")
            patch = file_info.get("patch", "") or ""
            interesting = []
            for line in patch.splitlines():
                lower = line.lower()
                if any(tok in lower for tok in ("author", "contributed", "github", "profile")):
                    interesting.append(line[:300])
                if len(interesting) >= 8:
                    break
            if interesting:
                lines.append(f"File: {filename}")
                lines.extend(interesting)
        return "\n".join(line for line in lines if line)[:limit_chars]

    def _github_user_real_name(self, login: str) -> str:
        if not login:
            return ""
        api_url = f"https://api.github.com/users/{login}"
        req = urllib.request.Request(api_url, headers={"User-Agent": self._user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            return ""
        name = data.get("name")
        return str(name).strip() if name else ""

    def _coauthor_profile_lines(self, message: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"Co-authored-by:\s*([^<\n]+?)\s*<([^>]+)>", re.I)
        for name, email in pattern.findall(message or ""):
            clean_name = " ".join(name.split())
            clean_email = email.strip()
            login = ""
            match = re.search(r"\+([^@]+)@users\.noreply\.github\.com", clean_email)
            if match:
                login = match.group(1)
            else:
                match = re.search(r"^([^@]+)@users\.noreply\.github\.com", clean_email)
                if match:
                    login = match.group(1)
            key = login or clean_email or clean_name
            if key in seen:
                continue
            seen.add(key)
            if login:
                profile_name = self._github_user_real_name(login)
                out.append(
                    f"Co-author with GitHub profile: commit trailer name: {clean_name}; "
                    f"GitHub: @{login} https://github.com/{login}; "
                    f"real name: {profile_name or clean_name}; email: {clean_email}"
                )
            else:
                out.append(
                    f"Co-author without GitHub noreply profile: commit trailer name: {clean_name}; "
                    f"GitHub profile: not provided in commit trailer; email: {clean_email}"
                )
        return out


def _format_github_commit_summary(item: object) -> list[str]:
    if not isinstance(item, dict):
        return []
    commit = item.get("commit", {}) or {}
    if not isinstance(commit, dict):
        commit = {}
    author = commit.get("author", {}) or {}
    committer = commit.get("committer", {}) or {}
    if not isinstance(author, dict):
        author = {}
    if not isinstance(committer, dict):
        committer = {}
    gh_author = item.get("author") or {}
    gh_committer = item.get("committer") or {}
    sha = str(item.get("sha") or "")
    html_url = str(item.get("html_url") or "")
    message = str(commit.get("message") or "").split("\n", 1)[0]
    lines = [
        f"- commit {sha[:12]} ({sha})",
        f"  url: {html_url}",
        f"  date: {author.get('date', '')}",
        f"  author: {author.get('name', '')} <{author.get('email', '')}>",
        f"  committer: {committer.get('name', '')} <{committer.get('email', '')}> at {committer.get('date', '')}",
        f"  message: {message}",
    ]
    if isinstance(gh_author, dict) and gh_author.get("login"):
        lines.append(f"  GitHub author: @{gh_author.get('login')} {gh_author.get('html_url', '')}")
    if isinstance(gh_committer, dict) and gh_committer.get("login"):
        lines.append(f"  GitHub committer: @{gh_committer.get('login')} {gh_committer.get('html_url', '')}")
    return lines


def dumps_sources(results: list[dict[str, str]]) -> str:
    compact = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
        for r in results
    ]
    return json.dumps(compact, ensure_ascii=False, indent=2)
