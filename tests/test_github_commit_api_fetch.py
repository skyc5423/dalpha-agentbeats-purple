import json

import pytest

from purple.schema import TaskRequest
from purple.runtime.tool import ToolContext
from purple.tools import StdlibWebClient
from purple.tools_api.web_fetch import WebFetchTool


PAYLOAD = [
    {
        "sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "html_url": "https://github.com/org/repo/commit/bbbbbbb",
        "commit": {
            "author": {"name": "New Author", "email": "new@example.com", "date": "2024-02-02T00:00:00Z"},
            "committer": {"name": "New Committer", "email": "newc@example.com", "date": "2024-02-02T00:00:01Z"},
            "message": "newer change",
        },
        "author": {"login": "newauthor", "html_url": "https://github.com/newauthor"},
    },
    {
        "sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "html_url": "https://github.com/org/repo/commit/aaaaaaaa",
        "commit": {
            "author": {"name": "Old Author", "email": "old@example.com", "date": "2024-01-01T00:00:00Z"},
            "committer": {"name": "Old Committer", "email": "oldc@example.com", "date": "2024-01-01T00:00:01Z"},
            "message": "add model support",
        },
        "author": {"login": "oldauthor", "html_url": "https://github.com/oldauthor"},
    },
]


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(PAYLOAD).encode()


def _fake_urlopen(req, timeout):
    return FakeResponse()


@pytest.mark.asyncio
async def test_web_fetch_promotes_github_commits_api_to_answer_candidate(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    tool = WebFetchTool(web_client=StdlibWebClient())
    ctx = ToolContext(
        request=TaskRequest(prompt="Find the first commit for a path"),
        notes={},
        scratch={},
        steps_remaining=10,
    )

    result = await tool.run(
        {"url": "https://api.github.com/repos/org/repo/commits?sha=main&path=src/model&per_page=100&page=1"},
        ctx,
    )

    assert result.ok
    assert result.outputs["answer_candidate"]
    # The generic fetch path should surface the oldest returned commit, which is
    # the key clue for first-commit/path-history questions.
    assert "aaaaaaaa" in result.outputs["answer_candidate"]
    assert "add model support" in result.outputs["answer_candidate"]


def test_stdlib_github_commits_api_formatter(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    text = StdlibWebClient()._fetch_github_commits_list_api_text(
        "https://api.github.com/repos/org/repo/commits?sha=main&path=src/model&per_page=100&page=1",
        5000,
    )

    assert "Repository: org/repo" in text
    assert "Path filter: src/model" in text
    assert "Oldest commit on this returned page" in text
    assert "aaaaaaaaaaaa" in text
    assert "@oldauthor" in text


PULL_PAYLOAD = {
    "html_url": "https://github.com/org/repo/pull/123",
    "number": 123,
    "title": "Add model support",
    "state": "closed",
    "merged_at": "2024-01-01T00:00:00Z",
    "merge_commit_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "user": {"login": "oldauthor", "html_url": "https://github.com/oldauthor"},
}


def _fake_pr_urlopen(req, timeout):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            url = getattr(req, "full_url", str(req))
            if "/pulls/123" in url:
                return json.dumps(PULL_PAYLOAD).encode()
            if "/commits/aaaaaaaa" in url:
                return json.dumps(PAYLOAD[-1]).encode()
            return json.dumps({"name": "Old Author"}).encode()
    return Resp()


def test_stdlib_github_pull_commits_page_promotes_merge_commit(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _fake_pr_urlopen)
    text = StdlibWebClient()._sync_fetch_text(
        "https://github.com/org/repo/pull/123/commits",
        5000,
    )

    assert "GitHub pull request URL: https://github.com/org/repo/pull/123" in text
    assert "Merge commit SHA: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in text
    assert "@oldauthor" in text
    assert "add model support" in text


MATH_GENEALOGY_HTML = """
<html><head><title>Xifeng Yan - The Mathematics Genealogy Project</title></head>
<body>
<h2>Xifeng  Yan </h2>
<div>Ph.D. <span>University of Illinois at Urbana-Champaign</span> 2006</div>
<div>Dissertation:</div><span id="thesisTitle">Mining, Indexing and Similarity Search in Large Graph Data Sets</span>
<p>Advisor 1: <a href="id.php?id=72247">Jiawei  Han</a><br /></p>
</body></html>
"""


class FakeMathGenealogyClient(StdlibWebClient):
    def _open(self, url: str) -> str:
        return MATH_GENEALOGY_HTML


def test_stdlib_math_genealogy_fetch_structures_advisor_links():
    text = FakeMathGenealogyClient()._sync_fetch_text(
        "https://www.mathgenealogy.org/id.php?id=279264",
        5000,
    )

    assert "MathGenealogy profile: Xifeng Yan" in text
    assert "Advisor 1: Jiawei Han (https://www.mathgenealogy.org/id.php?id=72247)" in text
    assert "Ph.D.: University of Illinois at Urbana-Champaign 2006" in text


class FakeTextWebClient:
    async def search(self, query: str, *, limit: int = 5):
        return []

    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
        return "Advisor source: https://www.mathgenealogy.org/id.php?id=72247"


@pytest.mark.asyncio
async def test_web_fetch_exposes_detected_urls_for_followup_fetches():
    tool = WebFetchTool(web_client=FakeTextWebClient())
    ctx = ToolContext(
        request=TaskRequest(prompt="Trace an advisor lineage"),
        notes={},
        scratch={},
        steps_remaining=10,
    )

    result = await tool.run({"url": "https://example.test/profile"}, ctx)

    assert result.ok
    assert "https://www.mathgenealogy.org/id.php?id=72247" in result.outputs["urls_detected"]


@pytest.mark.asyncio
async def test_sufficiency_rejects_unstructured_github_commit_candidate():
    from purple.tools_api.sufficiency_check import SufficiencyCheckTool

    tool = SufficiencyCheckTool()
    ctx = ToolContext(
        request=TaskRequest(
            prompt=(
                "Identify the first commit on the main branch of the official Hugging Face "
                "transformers repository that added support for the LLaVA model. Please provide "
                "the short commit ID, date, contributors/authors, GitHub profiles and real names."
            )
        ),
        notes={"answer_candidate": "First commit: d00f1ca by Shruthi42 on October 21, 2024."},
        scratch={},
        steps_remaining=10,
    )

    result = await tool.run({}, ctx)

    assert result.outputs["sufficient"] is False
    assert "structured GitHub commit metadata" in result.outputs["missing_or_weak_points"]


@pytest.mark.asyncio
async def test_sufficiency_rejects_incomplete_five_generation_lineage_candidate():
    from purple.tools_api.sufficiency_check import SufficiencyCheckTool

    candidate = (
        "Supported lineage: Yu Su → Xifeng Yan. "
        "A five-generation upward doctoral-advisor lineage cannot be completed from the provided evidence."
    )
    result = await SufficiencyCheckTool().run(
        {"candidate": candidate},
        ToolContext(
            request=TaskRequest(
                prompt=(
                    "Trace OSU Professor Yu Su's doctoral advisor lineage upward for five generations. "
                    "Provide the lineage names in order and cite advisor-advisee evidence."
                )
            ),
            notes={"answer_candidate": candidate, "spans": ["PhD Advisor Xifeng Yan"]},
            scratch={},
            steps_remaining=10,
        ),
    )

    assert result.outputs["sufficient"] is False
    assert "complete five-generation advisor lineage" in result.outputs["missing_or_weak_points"]
