"""Small stdlib-only web search and fetch helpers.

The helpers deliberately avoid heavy browser automation. They provide a bounded,
best-effort text/search surface for public benchmark agents that need source
URLs without embedding private infrastructure.
"""

from __future__ import annotations

import asyncio
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


class StdlibWebClient:
    def __init__(self, *, timeout_s: float = 12.0, user_agent: str = "Mozilla/5.0 AgentBeatsPurple/0.1") -> None:
        self._timeout_s = timeout_s
        self._user_agent = user_agent

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._sync_search, query, limit)

    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
        return await asyncio.to_thread(self._sync_fetch_text, url, limit_chars)

    def _open(self, url: str) -> str:
        raw, content_type, charset = self._open_bytes(url)
        if "pdf" in content_type.lower() or urllib.parse.urlparse(url).path.lower().endswith(".pdf"):
            return pdf_bytes_to_text(raw, limit_chars=5000)
        return raw.decode(charset or "utf-8", errors="replace")

    def _open_bytes(self, url: str) -> tuple[bytes, str, str]:
        req = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
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
        # not require API credentials. If it changes/fails, callers still get a
        # clean empty result rather than an exception.
        url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
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
        if not results:
            return []
        return results[:limit]

    def _github_commit_search(self, query: str, limit: int) -> list[dict[str, str]]:
        qlow = query.lower()
        repo_match = re.search(r"repo:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", query)
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
        for item in data.get("items", [])[:limit]:
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
        try:
            raw = self._open(url)
        except Exception:
            return ""
        return html_to_text(raw, limit_chars=limit_chars)

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


def dumps_sources(results: list[dict[str, str]]) -> str:
    compact = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")}
        for r in results
    ]
    return json.dumps(compact, ensure_ascii=False, indent=2)
