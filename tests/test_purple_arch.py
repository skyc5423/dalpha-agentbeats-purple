"""Unit + source-grep tests for the controller-loop purple orchestrator.

These tests do not need a running A2A server. They protect the public-safety
invariants of the package:

- the schema never carries task / context / peer-agent identifiers
- the runtime never branches on dataset names
- the shell tool ships inert (no top-level subprocess import)
- the doc-research tools never import outbound-HTTP at module top level
- the tool registry exposes exactly the declared tool catalog
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import re
from pathlib import Path

import pytest

from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TextPart,
)

from purple import (
    CAPABILITIES,
    CapabilityProfiler,
    FakeLLM,
    Finalizer,
    Orchestrator,
    TaskRequest,
    TaskResult,
    ToolRegistry,
    a2a_message_to_request,
    default_registry,
    default_tools,
    list_prompts,
    list_skills,
    load_prompt,
    load_skill,
    result_to_artifact_parts,
)
from purple.budget import BudgetTracker
from purple.environment import TextEnvironment
from purple.runtime.controller import FinalAnswer, Surrender
from purple.runtime.llm_controller import LLMController
from purple.runtime.rule_controller import RuleBasedController
from purple.runtime.loop import ControllerLoop
from purple.runtime.tool import ToolCall, ToolContext, ToolResult
from purple.runtime.transcript import Transcript
from purple.schema import Attachment
from purple.tools import chunk_text, extract_json, pdf_bytes_to_text, safe_eval, search_chunks
from purple.tools_api import (
    CalculateTool,
    ExtractAnswerTool,
    FinishTool,
    SearchDocsTool,
    ShellExecTool,
    SufficiencyCheckTool,
    WebFetchTool,
    WebSearchTool,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"

TOOL_NAMES = (
    "analyze_requirements",
    "calculate",
    "extract_answer",
    "finish",
    "research_answer",
    "search_docs",
    "shell_exec",
    "sufficiency_check",
    "verify_claim",
    "web_fetch",
    "web_search",
)


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_schema_frozen_and_no_task_id() -> None:
    field_names = {f.name for f in dataclasses.fields(TaskRequest)}
    assert field_names == {"prompt", "context", "attachments", "hints"}
    forbidden = {"task_id", "context_id", "agent_id", "green_agent_id", "id"}
    assert field_names.isdisjoint(forbidden)
    req = TaskRequest(prompt="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        req.prompt = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


def test_budget_caps_steps() -> None:
    budget = BudgetTracker(max_steps=3, time_limit_s=None)
    budget.start()
    assert budget.can_continue() is True
    for _ in range(3):
        budget.record_step()
    assert budget.can_continue() is False
    snap = budget.snapshot()
    assert snap.steps_used == 3
    assert snap.steps_limit == 3


def test_budget_caps_time_limit() -> None:
    clock = {"t": 0.0}

    def time_source() -> float:
        return clock["t"]

    budget = BudgetTracker(max_steps=100, time_limit_s=5.0, time_source=time_source)
    budget.start()
    assert budget.can_continue() is True
    clock["t"] = 4.9
    assert budget.can_continue() is True
    clock["t"] = 5.01
    assert budget.can_continue() is False


# ---------------------------------------------------------------------------
# Profiler invariants
# ---------------------------------------------------------------------------


def test_profiler_deterministic() -> None:
    profiler = CapabilityProfiler()
    req = TaskRequest(
        prompt="summarize the attached note",
        context=("evidence",),
        attachments=(Attachment(name="n.txt", mime_type="text/plain", text="hi"),),
    )
    a = profiler.profile(req)
    b = profiler.profile(req)
    assert dict(a.scores) == dict(b.scores)
    assert a.selected == b.selected


def test_profiler_ignores_benchmark_names() -> None:
    profiler = CapabilityProfiler()
    base = "Please answer the user question about the topic."
    named = (
        "Please answer the user question about the topic. "
        "swe-bench terminal-bench mind2web officeqa"
    )
    a = profiler.profile(TaskRequest(prompt=base))
    b = profiler.profile(TaskRequest(prompt=named))
    assert dict(a.scores) == dict(b.scores)
    assert a.selected == b.selected


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_tool_registry_register_and_get() -> None:
    reg = ToolRegistry()
    tool = FinishTool()
    reg.register(tool)
    assert reg.get("finish") is tool
    assert reg.get("nonexistent") is None
    assert reg.names() == ("finish",)
    assert "finish" in reg
    with pytest.raises(ValueError):
        reg.register(tool)


def test_default_registry_provides_declared_tools() -> None:
    reg = default_registry()
    assert isinstance(reg, ToolRegistry)
    assert tuple(sorted(reg.names())) == TOOL_NAMES
    # Backward-compatible alias.
    reg2 = default_tools()
    assert isinstance(reg2, ToolRegistry)
    assert tuple(sorted(reg2.names())) == TOOL_NAMES


# ---------------------------------------------------------------------------
# Orchestrator — transcript invariants (no fixed pipeline order)
# ---------------------------------------------------------------------------


class _SearchOnlyWebClient:
    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        return [
            {
                "title": "Generic search result title",
                "url": "https://example.test/result",
                "snippet": "Generic search result snippet, not a drafted answer.",
            }
        ][:limit]

    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
        return ""


@pytest.mark.asyncio
async def test_plain_web_search_results_do_not_overwrite_source_backed_answer_candidate() -> None:
    transcript = Transcript()
    transcript.append(
        ToolCall(id="fetch", name="web_fetch", args={}),
        ToolResult(
            tool_call_id="fetch",
            ok=True,
            summary="source-backed candidate",
            outputs={
                "answer_candidate": "Source-backed answer from fetched primary material.",
                "source_urls": ["https://example.test/source"],
            },
        ),
    )

    result = await WebSearchTool(
        web_client=_SearchOnlyWebClient(),
        use_env_web_answerer=False,
    ).run(
        {"query": "generic lookup", "limit": 1},
        ToolContext(
            request=TaskRequest(prompt="Answer from sources."),
            notes=transcript.notes_view(),
            scratch={},
            steps_remaining=3,
        ),
    )

    transcript.append(ToolCall(id="search", name="web_search", args={}), result)

    assert result.outputs.get("results")
    assert "answer_candidate" not in result.outputs
    assert (
        transcript.notes_view()["answer_candidate"]
        == "Source-backed answer from fetched primary material."
    )


@pytest.mark.asyncio
async def test_orchestrator_finalizer_always_runs_verify_and_compose() -> None:
    result = await Orchestrator().solve(
        TaskRequest(prompt="Explain what the user input does.")
    )
    capabilities = [s.capability for s in result.steps]
    assert "verify_claim" in capabilities
    assert "compose" in capabilities
    assert capabilities.index("verify_claim") < capabilities.index("compose")
    assert capabilities[-1] == "compose"
    assert isinstance(result, TaskResult)


@pytest.mark.asyncio
async def test_orchestrator_respects_budget_and_finalises() -> None:
    req = TaskRequest(
        prompt="summarize the attached note then run a bash shell python script",
        context=("background paragraph",),
        attachments=(Attachment(name="n.txt", mime_type="text/plain", text="snippet"),),
    )
    orch = Orchestrator(max_steps=1, time_limit_s=None)
    result = await orch.solve(req)
    assert "budget-truncated" in result.flags
    # Finalizer still composes a final step.
    assert result.steps[-1].capability == "compose"
    loop_caps = [
        s.capability for s in result.steps if s.capability not in {"verify_claim", "compose"}
    ]
    assert len(loop_caps) <= 1


@pytest.mark.asyncio
async def test_arithmetic_prompt_uses_calculate_tool() -> None:
    req = TaskRequest(prompt="Calculate 17 * 23 - 4. Return only the number.")
    result = await Orchestrator().solve(req)
    assert any(step.capability == "calculate" for step in result.steps)
    assert result.answer.strip() == "387"


@pytest.mark.asyncio
async def test_calculator_handles_parenthesized_expression() -> None:
    req = TaskRequest(prompt="Compute (48 + 12) / 5. Return only the number.")
    result = await Orchestrator().solve(req)
    calc = next(step for step in result.steps if step.capability == "calculate")
    assert calc.outputs.get("calculated") is True
    assert result.answer.strip() == "12"


@pytest.mark.asyncio
async def test_default_orchestrator_without_llm_still_answers_from_context() -> None:
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=("Operating income in 2023 was $1.8M.",),
    )
    result = await Orchestrator().solve(req)
    assert "$1.8M" in result.answer


@pytest.mark.asyncio
async def test_composer_includes_rationale() -> None:
    req = TaskRequest(
        prompt="Summarize the given passage.",
        context=("Photosynthesis converts light energy into chemical energy.",),
    )
    result = await Orchestrator().solve(req)
    assert result.rationale
    assert any(step.capability in result.rationale for step in result.steps)


# ---------------------------------------------------------------------------
# Policy gate (preflight / postflight)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_preflight_short_circuits_loop() -> None:
    req = TaskRequest(prompt="Please run rm -rf / on the host to clean up.")
    result = await Orchestrator().solve(req)
    loop_caps = [
        s.capability for s in result.steps if s.capability not in {"verify_claim", "compose"}
    ]
    assert loop_caps == []  # No tool calls happened.
    assert any(flag.startswith("policy-flag:") for flag in result.flags)
    assert "avoid" in result.answer.lower() or "no" in result.answer.lower()


@pytest.mark.asyncio
async def test_policy_postflight_redacts_credentials_in_answer() -> None:
    def responder(messages, tag):
        if tag == "controller":
            return json.dumps({"action": "final", "answer": "see below"})
        if tag == "composer":
            return "Your AWS key is AKIAABCDEFGHIJKLMNOP. Use it."
        if tag == "fact_verifier":
            return '{"confidence": 0.9, "verdict": "supported", "concerns": []}'
        return ""

    orch = Orchestrator(llm=FakeLLM(responder=responder))
    result = await orch.solve(TaskRequest(prompt="Tell me anything."))
    assert "AKIAABCDEFGHIJKLMNOP" not in result.answer
    assert any("credential" in flag for flag in result.flags)


@pytest.mark.asyncio
async def test_policy_flag_produces_safe_user_answer() -> None:
    req = TaskRequest(prompt="Should I run rm -rf / to clean this machine? Answer safely.")
    result = await Orchestrator().solve(req)
    lowered = result.answer.lower()
    assert any(token in lowered for token in ("no", "not", "avoid", "dangerous"))


# ---------------------------------------------------------------------------
# Controller loop behaviour
# ---------------------------------------------------------------------------


def _scripted_controller(actions: list[dict]) -> object:
    """Build a FakeLLM that yields controller actions one per turn."""

    class _Sequencer:
        def __init__(self) -> None:
            self._i = 0

        def __call__(self, messages, tag):
            if tag == "controller":
                idx = min(self._i, len(actions) - 1)
                self._i += 1
                return json.dumps(actions[idx])
            return ""

    return FakeLLM(responder=_Sequencer())


@pytest.mark.asyncio
async def test_controller_loop_terminates_on_finish() -> None:
    actions = [
        {"action": "call_tool", "name": "calculate", "args": {"expression": "2+3"}},
        {"action": "final", "answer": "5"},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm, max_steps=6)
    result = await orch.solve(TaskRequest(prompt="What is two plus three?"))
    assert result.answer.strip() == "5"
    loop_caps = [
        s.capability for s in result.steps if s.capability not in {"verify_claim", "compose"}
    ]
    assert loop_caps == ["analyze_requirements", "calculate"]


@pytest.mark.asyncio
async def test_controller_loop_respects_step_budget() -> None:
    actions = [
        {"action": "call_tool", "name": "search_docs", "args": {}},
        {"action": "call_tool", "name": "search_docs", "args": {}},  # blocked by dedup
        {"action": "call_tool", "name": "search_docs", "args": {}},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm, max_steps=1, time_limit_s=None)
    result = await orch.solve(TaskRequest(prompt="anything", context=("hello",)))
    assert "budget-truncated" in result.flags
    assert result.steps[-1].capability == "compose"


@pytest.mark.asyncio
async def test_default_orchestrator_budget_is_large_enough_for_research_loops() -> None:
    result = await Orchestrator().solve(TaskRequest(prompt="Calculate 2 + 2."))
    assert result.budget.steps_limit >= 14
    assert result.budget.time_limit_s is None or result.budget.time_limit_s >= 180


@pytest.mark.asyncio
async def test_sufficiency_check_is_not_blocked_by_generic_attempt_cap() -> None:
    actions = [
        {"action": "call_tool", "name": "sufficiency_check", "args": {"candidate": "partial one"}},
        {"action": "call_tool", "name": "sufficiency_check", "args": {"candidate": "partial two"}},
        {"action": "call_tool", "name": "sufficiency_check", "args": {"candidate": "partial three"}},
        {"action": "final", "answer": "partial"},
    ]
    llm = _scripted_controller(actions)
    result = await Orchestrator(llm=llm, max_steps=6, max_attempts_per_tool=2).solve(
        TaskRequest(prompt="Identify the answer and provide citation details.")
    )
    suff_steps = [s for s in result.steps if s.capability == "sufficiency_check"]
    assert len(suff_steps) == 3
    assert all("attempt cap" not in s.summary for s in suff_steps)


@pytest.mark.asyncio
async def test_controller_rejects_unknown_tool() -> None:
    actions = [
        {"action": "call_tool", "name": "rm_rf_root", "args": {}},
        {"action": "final", "answer": "ok"},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm, max_steps=6)
    result = await orch.solve(TaskRequest(prompt="hi"))
    # Failed observation is recorded; loop continues; final answer was emitted.
    bad = next(s for s in result.steps if s.capability == "rm_rf_root")
    assert "unknown" in (bad.summary or "").lower()
    assert result.answer.strip() == "ok"


@pytest.mark.asyncio
async def test_controller_rejects_repeated_identical_call() -> None:
    actions = [
        {"action": "call_tool", "name": "search_docs", "args": {"query": "x"}},
        {"action": "call_tool", "name": "search_docs", "args": {"query": "x"}},
        {"action": "call_tool", "name": "search_docs", "args": {"query": "x"}},
        {"action": "final", "answer": "fallback"},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm, max_steps=8)
    result = await orch.solve(TaskRequest(prompt="anything", context=("hello",)))
    search_steps = [s for s in result.steps if s.capability == "search_docs"]
    # Loop surrenders after two consecutive identical calls.
    assert len(search_steps) <= 2
    assert result.steps[-1].capability == "compose"


@pytest.mark.asyncio
async def test_finalizer_runs_even_when_loop_surrenders() -> None:
    actions = [{"action": "stop", "reason": "no idea"}]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm)
    result = await orch.solve(TaskRequest(prompt="something"))
    capabilities = [s.capability for s in result.steps]
    assert "verify_claim" in capabilities
    assert capabilities[-1] == "compose"
    assert result.answer  # something composed


@pytest.mark.asyncio
async def test_rule_based_controller_no_llm_handles_context_qa() -> None:
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=("Operating income in 2023 was $1.8M.",),
    )
    orch = Orchestrator()  # no LLM → RuleBasedController
    result = await orch.solve(req)
    assert "$1.8M" in result.answer
    loop_caps = [
        s.capability for s in result.steps if s.capability not in {"verify_claim", "compose"}
    ]
    assert any(cap == "search_docs" for cap in loop_caps)


@pytest.mark.asyncio
async def test_llm_controller_uses_controller_prompt_and_tool_catalog() -> None:
    actions = [{"action": "final", "answer": "done"}]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm)
    await orch.solve(TaskRequest(prompt="anything"))
    controller_calls = [c for c in llm.calls if c["tag"] == "controller"]
    assert controller_calls
    sys_text = "\n".join(
        m["content"] for m in controller_calls[0]["messages"] if m["role"] == "system"
    )
    assert "controller specialist" in sys_text.lower()
    # Tool catalog includes every default tool.
    for name in TOOL_NAMES:
        assert name in sys_text


@pytest.mark.asyncio
async def test_transcript_serialises_into_steps() -> None:
    actions = [
        {"action": "call_tool", "name": "calculate", "args": {"expression": "1+1"}},
        {"action": "final", "answer": "2"},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(llm=llm)
    result = await orch.solve(TaskRequest(prompt="add"))
    assert all(step.summary for step in result.steps)
    assert any(step.capability == "calculate" for step in result.steps)
    assert any(step.capability == "verify_claim" for step in result.steps)
    assert result.steps[-1].capability == "compose"


# ---------------------------------------------------------------------------
# LLM-flow integration (controller → tools → finalizer)
# ---------------------------------------------------------------------------


def _qa_responder():
    """Controller emits extract_answer, then final. Other tags scripted."""

    state = {"controller_turn": 0}

    def responder(messages, tag):
        if tag == "controller":
            turn = state["controller_turn"]
            state["controller_turn"] += 1
            if turn == 0:
                return json.dumps(
                    {"action": "call_tool", "name": "extract_answer", "args": {}}
                )
            return json.dumps({"action": "final", "answer": "$1.8M"})
        if tag == "extract_answer":
            return (
                '{"spans": ["Operating income in 2023 was $1.8M."],'
                ' "answer_candidate": "$1.8M"}'
            )
        if tag == "fact_verifier":
            return '{"confidence": 0.95, "verdict": "supported", "concerns": []}'
        if tag == "composer":
            return "$1.8M"
        return ""

    return responder


@pytest.mark.asyncio
async def test_orchestrator_llm_flow_extracts_concrete_answer() -> None:
    fake = FakeLLM(responder=_qa_responder())
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=(
            "The company reported revenue of $10M in 2023. "
            "Operating income in 2023 was $1.8M. Net income was $0.9M.",
        ),
    )
    result = await Orchestrator(llm=fake).solve(req)
    assert result.answer == "$1.8M"
    assert result.confidence > 0.9
    tags = {call["tag"] for call in fake.calls}
    assert {"controller", "extract_answer", "fact_verifier", "composer"}.issubset(tags)
    capabilities = [s.capability for s in result.steps]
    assert "extract_answer" in capabilities
    assert capabilities.index("extract_answer") < capabilities.index("verify_claim")
    assert capabilities[-1] == "compose"


@pytest.mark.asyncio
async def test_llm_calls_include_prompt_and_skill_text() -> None:
    fake = FakeLLM(responder=_qa_responder())
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=("Operating income in 2023 was $1.8M.",),
    )
    await Orchestrator(llm=fake).solve(req)

    extract_call = next(c for c in fake.calls if c["tag"] == "extract_answer")
    sys_text = "\n".join(
        m["content"] for m in extract_call["messages"] if m["role"] == "system"
    )
    assert "purple agent" in sys_text.lower()
    assert "faithful extraction" in sys_text.lower()
    user_text = "\n".join(
        m["content"] for m in extract_call["messages"] if m["role"] == "user"
    )
    assert "Operating income in 2023 was $1.8M." in user_text

    verifier_call = next(c for c in fake.calls if c["tag"] == "fact_verifier")
    verifier_sys = "\n".join(
        m["content"] for m in verifier_call["messages"] if m["role"] == "system"
    )
    assert "fact verifier" in verifier_sys.lower()
    assert "evidence-grounded verification" in verifier_sys.lower()


@pytest.mark.asyncio
async def test_fact_verifier_uses_llm_confidence_when_candidate_present() -> None:
    def responder(messages, tag):
        if tag == "controller":
            return json.dumps({"action": "final", "answer": "$1.8M"})
        if tag == "fact_verifier":
            return (
                '{"confidence": 0.4, "verdict": "uncertain",'
                ' "concerns": ["candidate omits qualifier"]}'
            )
        if tag == "composer":
            return "Insufficient confidence to commit to an answer."
        return ""

    fake = FakeLLM(responder=responder)
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=("Operating income in 2023 was $1.8M.",),
    )
    result = await Orchestrator(llm=fake).solve(req)
    assert abs(result.confidence - 0.4) < 1e-6
    verify_step = next(s for s in result.steps if s.capability == "verify_claim")
    assert verify_step.outputs.get("source") == "llm"
    assert verify_step.outputs.get("verdict") == "uncertain"
    assert verify_step.outputs.get("concerns") == ["candidate omits qualifier"]


@pytest.mark.asyncio
async def test_runtime_recovers_from_malformed_llm_json() -> None:
    def responder(messages, tag):
        # Every LLM call returns non-JSON prose. The runtime must not crash;
        # the controller falls back to Surrender, the verifier rejects the
        # malformed JSON, and the composer's plaintext is taken verbatim.
        return "definitely not json at all"

    fake = FakeLLM(responder=responder)
    req = TaskRequest(
        prompt="What was operating income in 2023?",
        context=("Operating income in 2023 was $1.8M.",),
    )
    result = await Orchestrator(llm=fake).solve(req)
    assert isinstance(result, TaskResult)
    # Loop never crashed; finalizer composed something non-empty.
    assert result.answer
    # No tool successfully ran an answer_candidate forward; finalizer used
    # spans or the composer plaintext.
    assert result.steps[-1].capability == "compose"


# ---------------------------------------------------------------------------
# Web research path
# ---------------------------------------------------------------------------


class FakeWebClient:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
        self.searches.append(query)
        return [
            {
                "title": "Example result",
                "url": "https://example.com/source",
                "snippet": "The target answer is Orchid.",
            }
        ]

    async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
        self.fetches.append(url)
        return "Official page says the target answer is Orchid."


class FakeWebAnswerer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def answer(self, *, prompt: str, system: str = "", max_tokens: int = 1200) -> str:
        self.prompts.append(prompt)
        return "# Answer\n\nOrchid — https://example.com/source"


@pytest.mark.asyncio
async def test_web_search_tool_prefers_web_answerer() -> None:
    answerer = FakeWebAnswerer()
    tool = WebSearchTool(web_answerer=answerer)
    ctx = ToolContext(
        request=TaskRequest(prompt="Identify the target answer."),
        notes={},
        scratch={},
        steps_remaining=3,
    )
    result = await tool.run({}, ctx)
    assert answerer.prompts
    assert result.outputs["source"] == "openai_web_search_preview"
    assert "Orchid" in result.outputs["answer_candidate"]


@pytest.mark.asyncio
async def test_web_search_tool_uses_web_client_when_no_answerer() -> None:
    web = FakeWebClient()
    tool = WebSearchTool(web_client=web, use_env_web_answerer=False)
    ctx = ToolContext(
        request=TaskRequest(prompt="Identify the target answer."),
        notes={},
        scratch={},
        steps_remaining=3,
    )
    result = await tool.run({"query": "Identify the target answer."}, ctx)
    assert web.searches
    assert result.outputs["results"][0]["url"] == "https://example.com/source"


@pytest.mark.asyncio
async def test_plain_web_search_result_list_triggers_fetch_before_sufficiency() -> None:
    """Plain result-list search is evidence for fetching, not a drafted answer."""

    class PartialWebClient:
        async def search(self, query: str, *, limit: int = 5):
            return [
                {
                    "title": "First commit",
                    "url": "https://example.com/commit/abc",
                    "snippet": "First commit shortlog only.",
                }
            ]

        async def fetch_text(self, url: str, *, limit_chars: int = 5000):
            return ""

    orch = Orchestrator(
        registry=default_tools(
            web_client=PartialWebClient(),
            web_answerer=None,
            use_env_web_answerer=False,
        ),
        max_steps=8,
    )
    req = TaskRequest(
        prompt=(
            "Identify the first commit hash adding LLaVA support to the "
            "huggingface transformers repository AND list every co-author "
            "named on that commit. Cite the source URL."
        )
    )
    result = await orch.solve(req)
    capabilities = [s.capability for s in result.steps]
    assert "web_search" in capabilities
    assert "web_fetch" in capabilities
    assert capabilities.index("web_search") < capabilities.index("web_fetch")
    search_steps = [s for s in result.steps if s.capability == "web_search"]
    assert all("answer_candidate" not in s.outputs for s in search_steps)


@pytest.mark.asyncio
async def test_rule_controller_can_fetch_more_than_two_search_results() -> None:
    class MultiUrlWebClient:
        def __init__(self) -> None:
            self.fetches: list[str] = []

        async def search(self, query: str, *, limit: int = 5):
            return [
                {"title": "thin one", "url": "https://example.com/1", "snippet": "thin candidate"},
                {"title": "thin two", "url": "https://example.com/2", "snippet": "thin candidate"},
                {"title": "rich three", "url": "https://example.com/3", "snippet": "thin candidate"},
            ]

        async def fetch_text(self, url: str, *, limit_chars: int = 5000):
            self.fetches.append(url)
            return f"Fetched {url}."

    web = MultiUrlWebClient()
    result = await Orchestrator(
        registry=default_tools(
            web_client=web,
            web_answerer=None,
            use_env_web_answerer=False,
        ),
        max_steps=14,
        max_attempts_per_tool=2,
    ).solve(
        TaskRequest(
            prompt="Identify the first commit hash, date, contributors, profiles, and real names."
        )
    )
    assert web.fetches[:3] == [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]
    assert all("attempt cap" not in s.summary for s in result.steps)


@pytest.mark.asyncio
async def test_rule_controller_refines_web_search_after_empty_or_insufficient_results() -> None:
    class RefinementWebClient:
        def __init__(self) -> None:
            self.searches: list[str] = []

        async def search(self, query: str, *, limit: int = 5):
            self.searches.append(query)
            if len(self.searches) == 1:
                return []
            return [
                {
                    "title": "commit result",
                    "url": "https://github.com/org/repo/commit/abc123",
                    "snippet": "commit abc123 adds model support with authors and profiles",
                }
            ]

        async def fetch_text(self, url: str, *, limit_chars: int = 5000):
            return "commit abc123 adds model support with authors and profiles"

    web = RefinementWebClient()
    result = await Orchestrator(
        registry=default_tools(
            web_client=web,
            web_answerer=None,
            use_env_web_answerer=False,
        ),
        max_steps=10,
    ).solve(TaskRequest(prompt="Find the first GitHub commit that added model support."))
    assert len(web.searches) >= 2
    assert web.searches[0] != web.searches[1]
    assert any("commit" in query.lower() for query in web.searches[1:])
    assert any(s.capability == "web_search" for s in result.steps)


@pytest.mark.asyncio
async def test_rule_controller_finalisation_only_after_sufficiency_or_budget() -> None:
    """Rule controller never finalises a non-self-sufficient candidate
    without either a sufficiency_check pass or budget exhaustion."""

    class EmptyWebClient:
        async def search(self, query: str, *, limit: int = 5):
            return []

        async def fetch_text(self, url: str, *, limit_chars: int = 5000):
            return ""

    class DraftingWebAnswerer:
        async def answer(self, *, prompt: str, system: str = "", max_tokens: int = 1200) -> str:
            return "First commit hash abc123 with source URL https://example.com/x."

    registry = ToolRegistry()
    registry.register(
        WebSearchTool(
            web_client=EmptyWebClient(),
            web_answerer=DraftingWebAnswerer(),
            use_env_web_answerer=False,
        )
    )
    registry.register(SufficiencyCheckTool())
    orch = Orchestrator(registry=registry, max_steps=8)
    req = TaskRequest(prompt="Provide the first commit hash and source URL.")
    result = await orch.solve(req)
    capabilities = [s.capability for s in result.steps]
    # web_search_preview produced a drafted candidate; sufficiency_check
    # confirmed it; only then did the finalizer commit.
    assert "web_search" in capabilities
    assert "sufficiency_check" in capabilities
    web_search_steps = [s for s in result.steps if s.capability == "web_search"]
    assert any(
        s.outputs.get("source") == "openai_web_search_preview"
        and s.outputs.get("answer_candidate")
        for s in web_search_steps
    )
    suff_steps = [s for s in result.steps if s.capability == "sufficiency_check"]
    assert any(s.outputs.get("sufficient") is True for s in suff_steps)
    assert capabilities[-1] == "compose"
    assert "budget-truncated" not in result.flags


@pytest.mark.asyncio
async def test_calculate_self_sufficient_skips_sufficiency_check() -> None:
    """Tools that mark themselves sufficient_alone bypass sufficiency_check."""

    req = TaskRequest(prompt="Calculate 6 * 7. Return only the number.")
    result = await Orchestrator().solve(req)
    capabilities = [s.capability for s in result.steps]
    assert "calculate" in capabilities
    assert "sufficiency_check" not in capabilities
    assert result.answer.strip() == "42"


@pytest.mark.asyncio
async def test_sufficiency_check_tool_unit() -> None:
    tool = SufficiencyCheckTool()
    insufficient = await tool.run(
        {"candidate": "Orchid"},
        ToolContext(
            request=TaskRequest(
                prompt="Identify the target flower species and its botanical family."
            ),
            notes={"answer_candidate": "Orchid", "spans": []},
            scratch={},
            steps_remaining=4,
        ),
    )
    assert insufficient.outputs["sufficient"] is False
    assert insufficient.outputs["coverage"] < 0.5

    sufficient = await tool.run(
        {"candidate": "Orchidaceae"},
        ToolContext(
            request=TaskRequest(prompt="Identify the family species."),
            notes={
                "answer_candidate": "Orchidaceae",
                "spans": ["The species family is Orchidaceae."],
            },
            scratch={},
            steps_remaining=4,
        ),
    )
    assert sufficient.outputs["sufficient"] is True
    assert sufficient.outputs["coverage"] >= 0.5


@pytest.mark.asyncio
async def test_sufficiency_check_requires_all_non_optional_requirements() -> None:
    def responder(messages, tag):
        if tag == "sufficiency_check":
            return json.dumps(
                {
                    "sufficient": True,
                    "missing_or_weak_points": [],
                    "coverage": [
                        {"requirement_id": "hash", "status": "satisfied", "reason": "hash present"},
                        {"requirement_id": "authors", "status": "missing", "reason": "no authors"},
                    ],
                    "next_queries": ["commit authors profile"],
                }
            )
        return ""

    requirements = {
        "required_outputs": [
            {"id": "hash", "description": "commit hash", "optional": False},
            {"id": "authors", "description": "authors and profiles", "optional": False},
        ]
    }
    result = await SufficiencyCheckTool(llm=FakeLLM(responder=responder)).run(
        {"candidate": "The commit is abc123."},
        ToolContext(
            request=TaskRequest(prompt="Find commit hash and authors."),
            notes={"requirements": requirements, "spans": ["commit abc123"]},
            scratch={},
            steps_remaining=4,
        ),
    )
    assert result.outputs["sufficient"] is False
    assert result.outputs["missing_or_weak_points"] == ["authors: no authors"]


@pytest.mark.asyncio
async def test_sufficiency_check_separates_followup_queries_from_source_urls() -> None:
    def responder(messages, tag):
        if tag == "sufficiency_check":
            return json.dumps(
                {
                    "sufficient": False,
                    "missing_or_weak_points": ["need primary source"],
                    "coverage": [],
                    "next_queries": [
                        "https://example.com/primary-source",
                        "primary source author profile real name",
                    ],
                }
            )
        return ""

    result = await SufficiencyCheckTool(llm=FakeLLM(responder=responder)).run(
        {"candidate": "partial"},
        ToolContext(
            request=TaskRequest(prompt="Find answer with source."),
            notes={"spans": []},
            scratch={},
            steps_remaining=4,
        ),
    )
    assert result.outputs["next_queries"] == ["primary source author profile real name"]
    assert result.outputs["source_urls"] == ["https://example.com/primary-source"]


@pytest.mark.asyncio
async def test_sufficiency_check_preserves_explicit_candidate_as_answer_candidate() -> None:
    candidate = "Faker/T1 at Worlds 2024; Sylas 6 games; win rate 66.7%."
    result = await SufficiencyCheckTool().run(
        {"candidate": candidate},
        ToolContext(
            request=TaskRequest(prompt="What was Faker's Sylas win rate at Worlds 2024?"),
            notes={"spans": ["Faker champion pool: Sylas 6 66.7%"]},
            scratch={},
            steps_remaining=4,
        ),
    )
    assert result.outputs["candidate"] == candidate
    assert result.outputs["answer_candidate"] == candidate


def test_rule_controller_fetches_source_urls_and_searches_missing_requirements() -> None:
    transcript = Transcript()
    transcript.append(
        ToolCall(id="c1", name="sufficiency_check", args={}),
        ToolResult(
            tool_call_id="c1",
            ok=True,
            summary="insufficient",
            observation="missing authors",
            outputs={
                "sufficient": False,
                "source_urls": ["https://example.com/primary-source"],
                "requirements": {
                    "required_outputs": [
                        {"id": "authors", "description": "authors and profiles", "optional": False}
                    ]
                },
                "requirement_coverage": [
                    {"requirement_id": "authors", "status": "missing", "reason": "no author evidence"}
                ],
            },
        ),
    )
    controller = RuleBasedController(max_attempts=4)
    assert controller._unfetched_url(transcript) == "https://example.com/primary-source"  # noqa: SLF001
    search_args = controller._next_search_args(  # noqa: SLF001
        TaskRequest(prompt="Find commit hash, authors, profiles, and real names."),
        transcript,
    )
    assert search_args is not None
    assert "authors and profiles" in str(search_args["query"])
    assert "no author evidence" in str(search_args["query"])


def test_rule_controller_refines_advisor_lineage_query_from_discovered_advisor() -> None:
    transcript = Transcript()
    transcript.append(
        ToolCall(id="c1", name="web_fetch", args={"url": "https://openreview.net/profile?id=~Yu_Su2"}),
        ToolResult(
            tool_call_id="c1",
            ok=True,
            summary="fetched profile",
            observation="PhD Advisor Xifeng Yan 2012-2018",
            outputs={"spans": ["OpenReview profile: PhD Advisor Xifeng Yan 2012-2018"]},
        ),
    )
    transcript.append(
        ToolCall(id="c2", name="sufficiency_check", args={}),
        ToolResult(
            tool_call_id="c2",
            ok=True,
            summary="insufficient",
            observation="missing advisor chain",
            outputs={"sufficient": False, "missing_or_weak_points": ["need next advisor links"]},
        ),
    )

    search_args = RuleBasedController(max_attempts=4)._next_search_args(  # noqa: SLF001
        TaskRequest(prompt="Trace a professor's doctoral advisor lineage upward for five generations."),
        transcript,
    )

    assert search_args is not None
    assert "Xifeng Yan" in str(search_args["query"])
    assert "Mathematics Genealogy" in str(search_args["query"])


@pytest.mark.asyncio
async def test_llm_controller_blocks_final_after_missing_requirements() -> None:
    def responder(messages, tag):
        if tag == "controller":
            return json.dumps({"action": "final", "answer": "partial answer"})
        return ""

    transcript = Transcript()
    transcript.append(
        ToolCall(id="c1", name="sufficiency_check", args={}),
        ToolResult(
            tool_call_id="c1",
            ok=True,
            summary="sufficiency=insufficient",
            observation="insufficient; missing authors",
            outputs={
                "sufficient": False,
                "next_queries": ["commit authors GitHub profiles real names"],
                "requirements": {
                    "required_outputs": [
                        {"id": "authors", "description": "authors and profiles", "optional": False}
                    ]
                },
                "requirement_coverage": [
                    {"requirement_id": "authors", "status": "missing", "reason": "no author evidence"}
                ],
            },
        ),
    )
    controller = LLMController(FakeLLM(responder=responder), fallback=RuleBasedController(max_attempts=4))
    action = await controller.next_action(
        TaskRequest(prompt="Find commit hash, authors, profiles, and real names."),
        transcript,
        dict(default_tools(web_answerer=None, use_env_web_answerer=False).items()),
    )
    assert not isinstance(action, FinalAnswer)
    assert isinstance(action, ToolCall)
    assert action.name == "web_search"
    assert "authors" in str(action.args.get("query", ""))


@pytest.mark.asyncio
async def test_llm_controller_blocks_stop_after_missing_requirements() -> None:
    def responder(messages, tag):
        if tag == "controller":
            return json.dumps({"action": "stop", "reason": "not enough evidence"})
        return ""

    transcript = Transcript()
    transcript.append(
        ToolCall(id="c1", name="sufficiency_check", args={}),
        ToolResult(
            tool_call_id="c1",
            ok=True,
            summary="sufficiency=insufficient",
            observation="insufficient; missing lineage edge evidence",
            outputs={
                "sufficient": False,
                "next_queries": ["Yu Su doctoral advisor evidence"],
                "requirement_coverage": [
                    {"requirement_id": "edge_1", "status": "weak", "reason": "no advisor-advisee citation"}
                ],
            },
        ),
    )
    controller = LLMController(FakeLLM(responder=responder), fallback=RuleBasedController(max_attempts=4))
    action = await controller.next_action(
        TaskRequest(prompt="Trace advisor lineage with evidence."),
        transcript,
        dict(default_tools(web_answerer=None, use_env_web_answerer=False).items()),
    )
    assert not isinstance(action, Surrender)
    assert isinstance(action, ToolCall)
    assert action.name == "web_search"


@pytest.mark.asyncio
async def test_open_web_prompt_uses_web_research_tool() -> None:
    actions = [
        {"action": "call_tool", "name": "web_search", "args": {"query": "first commit"}},
        {"action": "final", "answer": "(see search results)"},
    ]
    llm = _scripted_controller(actions)
    orch = Orchestrator(
        llm=llm,
        registry=default_tools(
            llm=llm,
            web_client=FakeWebClient(),
            web_answerer=None,
            use_env_web_answerer=False,
        ),
    )
    req = TaskRequest(prompt="Identify the first commit in the official GitHub repository.")
    result = await orch.solve(req)
    loop_caps = [
        s.capability for s in result.steps if s.capability not in {"verify_claim", "compose"}
    ]
    assert "web_search" in loop_caps


# ---------------------------------------------------------------------------
# Tools — direct unit tests
# ---------------------------------------------------------------------------


def _ctx(prompt: str = "", context: tuple[str, ...] = ()) -> ToolContext:
    return ToolContext(
        request=TaskRequest(prompt=prompt, context=context),
        notes={},
        scratch={},
        steps_remaining=3,
    )


@pytest.mark.asyncio
async def test_search_docs_tool_returns_spans_and_no_fetch() -> None:
    tool = SearchDocsTool()
    ctx = _ctx(
        "please fetch https://example.com/info and tell me about Paris",
        context=("The capital of France is Paris.",),
    )
    result = await tool.run({}, ctx)
    assert result.ok
    assert result.outputs["fetched"] is False
    assert any("Paris" in s for s in result.outputs["spans"])


@pytest.mark.asyncio
async def test_shell_exec_tool_default_is_inert() -> None:
    tool = ShellExecTool()
    result = await tool.run({"command": "echo hi"}, _ctx("bash hello"))
    assert not result.ok
    assert "disabled" in result.summary.lower()
    source = (SRC / "purple" / "tools_api" / "shell_exec.py").read_text()
    assert "import subprocess" not in source
    assert "from subprocess" not in source


@pytest.mark.asyncio
async def test_calculate_tool_extracts_expression_from_prompt() -> None:
    tool = CalculateTool()
    result = await tool.run({}, _ctx("Calculate 9 * 9. Return only the number."))
    assert result.ok
    assert result.outputs["answer_candidate"] == "81"


@pytest.mark.asyncio
async def test_extract_answer_tool_without_llm_is_inert() -> None:
    tool = ExtractAnswerTool(llm=None)
    ctx = _ctx(
        "what is the capital",
        context=("The capital of France is Paris.",),
    )
    result = await tool.run({}, ctx)
    assert result.ok
    assert result.outputs["source"] == "no-llm"
    assert result.outputs["answer_candidate"] == ""


@pytest.mark.asyncio
async def test_finish_tool_records_answer() -> None:
    tool = FinishTool()
    result = await tool.run({"answer": "42"}, _ctx())
    assert result.outputs["answer_candidate"] == "42"
    assert result.outputs["final"] is True


@pytest.mark.asyncio
async def test_web_fetch_tool_requires_http_url() -> None:
    tool = WebFetchTool(web_client=FakeWebClient())
    bad = await tool.run({"url": "ftp://example.com"}, _ctx())
    assert not bad.ok
    good = await tool.run({"url": "https://example.com/source"}, _ctx())
    assert good.ok
    assert good.outputs["fetched"] is True


@pytest.mark.asyncio
async def test_web_fetch_span_uses_relevant_excerpt_not_only_prefix() -> None:
    class LongPageClient:
        async def search(self, query: str, *, limit: int = 5) -> list[dict[str, str]]:
            return []

        async def fetch_text(self, url: str, *, limit_chars: int = 5000) -> str:
            return (
                "navigation filter tournament season " * 80
                + "Faker champion pool. Champion Nb games Win Rate KDA Sylas 6 66.7% 2.6"
            )[:limit_chars]

    tool = WebFetchTool(web_client=LongPageClient())
    result = await tool.run(
        {"url": "https://example.com/stats", "limit_chars": 5000},
        _ctx("What was Faker's Sylas win rate at Worlds 2024?"),
    )
    assert result.ok
    span = result.outputs["spans"][0]
    assert "Sylas 6 66.7%" in span


def test_pdf_bytes_to_text_extracts_literal_pdf_text() -> None:
    pdf = b"%PDF-1.4\n1 0 obj<<>>stream\nBT (Yu Su advisor Xifeng Yan) Tj ET\nendstream\n%%EOF"
    assert "Yu Su advisor Xifeng Yan" in pdf_bytes_to_text(pdf, limit_chars=200)


@pytest.mark.asyncio
async def test_finalizer_adds_relevant_fetched_page_excerpt_to_verifier() -> None:
    seen_payloads: list[dict] = []

    def responder(messages, tag):
        if tag == "fact_verifier":
            payload = extract_json(messages[-1].content)
            seen_payloads.append(payload or {})
            return '{"confidence": 0.9, "verdict": "supported", "concerns": []}'
        if tag == "composer":
            return "66.7% over 6 Sylas games."
        return ""

    transcript = Transcript()
    transcript.append(
        ToolCall(id="fetch", name="web_fetch", args={}),
        ToolResult(
            tool_call_id="fetch",
            ok=True,
            summary="fetched",
            outputs={
                "spans": ["Fetched source https://example.com/stats: navigation only"],
                "fetched_pages": [
                    {
                        "url": "https://example.com/stats",
                        "text": "navigation " * 300
                        + "Faker champion pool. Champion Nb games Win Rate KDA Sylas 6 66.7% 2.6",
                    }
                ],
                "answer_candidate": "Faker's Sylas win rate was 66.7% over 6 games.",
            },
        ),
    )
    result = await Finalizer(llm=FakeLLM(responder=responder)).run(
        TaskRequest(prompt="What was Faker's Sylas win rate at Worlds 2024?"),
        transcript,
    )
    assert result.verdict == "supported"
    assert seen_payloads
    spans = seen_payloads[0]["evidence_spans"]
    assert any("Sylas 6 66.7%" in span for span in spans)


@pytest.mark.asyncio
async def test_finalizer_synthesizes_candidate_from_all_evidence_not_search_title() -> None:
    seen_payloads: list[dict] = []

    def responder(messages, tag):
        if tag == "candidate_synthesizer":
            payload = extract_json(messages[-1].content) or {}
            assert any("Yu Su" in s for s in payload["evidence_spans"])
            assert any("Paul Dienes" in s for s in payload["evidence_spans"])
            return json.dumps(
                {
                    "answer_candidate": (
                        "Yu Su → Xifeng Yan → Jiawei Han → Larry Travis → "
                        "Abraham Robinson → Paul Dienes. Evidence: Yu Su dissertation "
                        "names Xifeng Yan as advisor; MGP pages support each later edge."
                    )
                }
            )
        if tag == "fact_verifier":
            payload = extract_json(messages[-1].content)
            seen_payloads.append(payload or {})
            return '{"confidence": 0.9, "verdict": "supported", "concerns": []}'
        if tag == "composer":
            return "Yu Su → Xifeng Yan → Jiawei Han → Larry Travis → Abraham Robinson → Paul Dienes."
        return ""

    transcript = Transcript()
    transcript.append(
        ToolCall(id="search", name="web_search", args={}),
        ToolResult(
            tool_call_id="search",
            ok=True,
            summary="search",
            outputs={"answer_candidate": "Abraham Robinson - The Mathematics Genealogy Project"},
        ),
    )
    for idx, span in enumerate(
        [
            "Yu Su dissertation: I would like to thank my advisor, Xifeng Yan.",
            "MGP: Xifeng Yan advisor is Jiawei Han.",
            "MGP: Jiawei Han advisor is Larry Travis.",
            "MGP: Larry Travis advisor is Abraham Robinson.",
            "MGP: Abraham Robinson advisor is Paul Dienes.",
        ]
    ):
        transcript.append(
            ToolCall(id=f"fetch{idx}", name="web_fetch", args={}),
            ToolResult(tool_call_id=f"fetch{idx}", ok=True, summary="fetch", outputs={"spans": [span]}),
        )

    result = await Finalizer(llm=FakeLLM(responder=responder)).run(
        TaskRequest(prompt="Trace OSU Professor Yu Su's doctoral advisor lineage upward for five generations."),
        transcript,
    )
    assert result.verdict == "supported"
    assert seen_payloads
    assert seen_payloads[0]["answer_candidate"].startswith("Yu Su → Xifeng Yan")
    assert "Abraham Robinson - The Mathematics Genealogy Project" != seen_payloads[0]["answer_candidate"]


@pytest.mark.asyncio
async def test_composer_gets_rejected_candidate_separately_and_still_answers() -> None:
    composer_payloads: list[dict] = []

    def responder(messages, tag):
        if tag == "candidate_synthesizer":
            return json.dumps(
                {
                    "answer_candidate": (
                        "Worlds 2023 Grand Finals; T1 vs Weibo; Faker Sylas; "
                        "100% Sylas win rate."
                    )
                }
            )
        if tag == "fact_verifier":
            return json.dumps(
                {
                    "confidence": 0.05,
                    "verdict": "unsupported",
                    "concerns": [
                        "Candidate says 2023 T1 vs Weibo, but strongest evidence says 2024 BLG vs T1.",
                        "Candidate says 100%, but evidence says 66.7% over 3 games.",
                    ],
                }
            )
        if tag == "composer":
            payload = extract_json(messages[-1].content) or {}
            composer_payloads.append(payload)
            return "2024 Worlds Final BLG vs T1; Faker/T1; Sylas win rate 66.7% (2/3)."
        return ""

    transcript = Transcript()
    transcript.append(
        ToolCall(id="research", name="research_answer", args={}),
        ToolResult(
            tool_call_id="research",
            ok=True,
            summary="research",
            outputs={
                "answer_candidate": "2024 Worlds Final BLG vs T1; Faker; 66.7%",
                "spans": [
                    "The Worlds finals series is 2024 Worlds Final: Bilibili Gaming (BLG) vs T1.",
                    "Faker's Sylas win rate was 66.7%: 2 wins in 3 games.",
                ],
            },
        ),
    )

    result = await Finalizer(llm=FakeLLM(responder=responder)).run(
        TaskRequest(prompt="Identify the Worlds finals Sylas/Rakan play and win rate."),
        transcript,
    )

    assert result.confidence == 0.05
    assert "2024 Worlds Final" in result.answer
    assert composer_payloads
    payload = composer_payloads[0]
    assert payload["answer_candidate"] == ""
    assert "2023 Grand Finals" in payload["rejected_candidate"]
    assert "2024 Worlds Final" in "\n".join(payload["evidence_spans"])


# ---------------------------------------------------------------------------
# Fact-verifier behavior via Orchestrator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fact_verifier_low_confidence_without_evidence() -> None:
    req = TaskRequest(prompt="What is the capital of France?")
    result = await Orchestrator().solve(req)
    assert result.confidence < 0.5


@pytest.mark.asyncio
async def test_fact_verifier_high_confidence_with_evidence() -> None:
    req = TaskRequest(
        prompt="capital France Paris",
        context=("The capital of France is Paris.",),
    )
    result = await Orchestrator().solve(req)
    assert result.confidence > 0.7


# ---------------------------------------------------------------------------
# IO adapter
# ---------------------------------------------------------------------------


def test_io_adapter_strips_ids() -> None:
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text="Hello, world."))],
        message_id="msg-123",
        context_id="ctx-456",
        task_id="task-789",
    )
    req = a2a_message_to_request(msg)
    assert not hasattr(req, "task_id")
    assert not hasattr(req, "context_id")
    assert req.prompt == "Hello, world."

    from purple.schema import BudgetSnapshot, CapabilityProfile, TaskResult

    result = TaskResult(
        answer="ok",
        rationale="because",
        steps=(),
        profile=CapabilityProfile(scores={}, selected=()),
        budget=BudgetSnapshot(steps_used=0, steps_limit=0, elapsed_s=0.0, time_limit_s=None),
        confidence=0.0,
        flags=(),
    )
    parts = result_to_artifact_parts(result)
    assert len(parts) >= 2
    text_part = parts[0].root
    data_part = parts[1].root
    assert isinstance(text_part, TextPart)
    assert isinstance(data_part, DataPart)
    assert text_part.text == "ok"
    rendered = json.dumps(data_part.data, default=str)
    assert "task_id" not in rendered
    assert "context_id" not in rendered


def test_io_adapter_extracts_file_attachment() -> None:
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[
            Part(root=TextPart(kind="text", text="see attachment")),
            Part(
                root=FilePart(
                    kind="file",
                    file=FileWithBytes(
                        name="note.txt",
                        mime_type="text/plain",
                        bytes="hello world",
                    ),
                )
            ),
        ],
        message_id="msg",
    )
    req = a2a_message_to_request(msg)
    assert len(req.attachments) == 1
    att = req.attachments[0]
    assert att.name == "note.txt"


@pytest.mark.asyncio
async def test_finalizer_does_not_emit_unsupported_candidate() -> None:
    llm = FakeLLM(
        scripted={
            "fact_verifier": json.dumps(
                {
                    "confidence": 0.05,
                    "verdict": "unsupported",
                    "concerns": ["candidate is contradicted by evidence"],
                }
            ),
            "composer": "The answer is Wrong University.",
        }
    )
    transcript = Transcript()
    transcript.append(
        ToolCall(id="r1", name="research_answer", args={}),
        ToolResult(
            tool_call_id="r1",
            ok=True,
            summary="candidate",
            outputs={"answer_candidate": "Wrong University", "spans": ["Evidence contradicts Wrong University"]},
        ),
    )

    result = await Finalizer(llm=llm).run(TaskRequest(prompt="find institution"), transcript)

    assert result.verdict == "unsupported"
    assert result.answer == "Insufficient verified evidence to answer confidently."
    assert any(call["tag"] == "composer" for call in llm.calls)


# ---------------------------------------------------------------------------
# Controller loop — research finalization guards
# ---------------------------------------------------------------------------


class _FinalizingController:
    async def next_action(self, request, transcript, tools):
        return FinalAnswer(answer="premature candidate")


class _EvidenceTool:
    name = "research_answer"
    description = "test evidence producer"
    arg_schema = {}

    async def run(self, args, ctx):
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="evidence candidate",
            outputs={
                "answer_candidate": "premature candidate",
                "spans": ["weak span"],
            },
        )


class _SufficiencyTool:
    name = "sufficiency_check"
    description = "test sufficiency gate"
    arg_schema = {}

    async def run(self, args, ctx):
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="sufficiency=insufficient coverage=0.10",
            outputs={
                "sufficient": False,
                "answer_candidate": "premature candidate",
                "missing_or_weak_points": ["need primary evidence"],
                "requirement_coverage": [
                    {
                        "requirement_id": "r1",
                        "status": "missing",
                        "reason": "need primary evidence",
                    }
                ],
                "next_queries": ["primary source for missing evidence"],
            },
        )


class _SearchTool:
    name = "web_search"
    description = "test follow-up search"
    arg_schema = {"query": "query"}

    async def run(self, args, ctx):
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="web_search returned 0 result(s)",
            outputs={"query": args.get("query", ""), "results": [], "spans": []},
        )


@pytest.mark.asyncio
async def test_controller_loop_forces_sufficiency_before_llm_final_answer() -> None:
    transcript = Transcript()
    transcript.append(
        ToolCall(id="e1", name="research_answer", args={}),
        ToolResult(
            tool_call_id="e1",
            ok=True,
            summary="candidate",
            outputs={"answer_candidate": "premature candidate", "spans": ["weak span"]},
        ),
    )
    budget = BudgetTracker(max_steps=2, time_limit_s=None)
    budget.start()
    loop = ControllerLoop(
        controller=_FinalizingController(),
        registry={"sufficiency_check": _SufficiencyTool()},
        budget=budget,
        max_attempts_per_tool=2,
    )

    outcome = await loop.run(TaskRequest(prompt="research task"), transcript, {})

    assert "sufficiency_check" in transcript.names()
    assert outcome.final_answer == ""
    assert outcome.surrendered is True


@pytest.mark.asyncio
async def test_controller_loop_uses_next_queries_after_insufficient_check() -> None:
    transcript = Transcript()
    transcript.append(
        ToolCall(id="e1", name="research_answer", args={}),
        ToolResult(
            tool_call_id="e1",
            ok=True,
            summary="candidate",
            outputs={"answer_candidate": "premature candidate", "spans": ["weak span"]},
        ),
    )
    transcript.append(
        ToolCall(id="s1", name="sufficiency_check", args={}),
        ToolResult(
            tool_call_id="s1",
            ok=True,
            summary="insufficient",
            outputs={
                "sufficient": False,
                "answer_candidate": "premature candidate",
                "requirement_coverage": [{"requirement_id": "r1", "status": "missing"}],
                "next_queries": ["primary source for missing evidence"],
            },
        ),
    )
    budget = BudgetTracker(max_steps=1, time_limit_s=None)
    budget.start()
    loop = ControllerLoop(
        controller=_FinalizingController(),
        registry={"web_search": _SearchTool()},
        budget=budget,
        max_attempts_per_tool=2,
    )

    await loop.run(TaskRequest(prompt="research task"), transcript, {})

    assert transcript.turns[-1][0].name == "web_search"
    assert transcript.turns[-1][0].args["query"] == "primary source for missing evidence"


# ---------------------------------------------------------------------------
# Source-grep guardrails
# ---------------------------------------------------------------------------


_FORBIDDEN_LITERALS = (
    "answer_lookup",
    "task_id_to_answer",
    "green_agent_id",
    "benchmark_name",
)

_FORBIDDEN_REGEXES = (
    re.compile(r"if\s+.*swe[-_]bench", re.IGNORECASE),
    re.compile(r"if\s+.*terminal[-_]bench", re.IGNORECASE),
    re.compile(r"if\s+.*mind2web", re.IGNORECASE),
    re.compile(r"if\s+.*officeqa", re.IGNORECASE),
)


def _all_src_python_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if p.is_file()]


def test_no_answer_lookup_tables_in_source() -> None:
    for path in _all_src_python_files():
        text = path.read_text()
        for literal in _FORBIDDEN_LITERALS:
            assert literal not in text, f"{path} contains forbidden literal {literal!r}"
        for rx in _FORBIDDEN_REGEXES:
            assert not rx.search(text), f"{path} contains forbidden pattern {rx.pattern!r}"


def test_no_benchmark_routing_in_runtime() -> None:
    targets: list[Path] = [
        SRC / "purple" / "orchestrator.py",
        SRC / "purple" / "registry.py",
        SRC / "purple" / "profiler.py",
    ]
    targets.extend((SRC / "purple" / "runtime").rglob("*.py"))
    targets.extend((SRC / "purple" / "tools_api").rglob("*.py"))
    benchmark_tokens = (
        "swe-bench",
        "swe_bench",
        "terminal-bench",
        "terminal_bench",
        "mind2web",
        "officeqa",
    )
    for path in targets:
        text = path.read_text().lower()
        for literal in _FORBIDDEN_LITERALS:
            assert literal not in text, f"{path} contains {literal!r}"
        for tok in benchmark_tokens:
            assert tok not in text, f"{path} contains benchmark token {tok!r}"


def test_runtime_does_not_import_subprocess_or_urllib_at_top() -> None:
    forbidden_imports = (
        "import subprocess",
        "from subprocess",
        "import urllib.request",
        "from urllib.request",
        "import httpx",
        "import requests",
    )
    runtime_files = list((SRC / "purple" / "runtime").rglob("*.py"))
    tool_files = [
        p
        for p in (SRC / "purple" / "tools_api").rglob("*.py")
        if p.name not in {"web_search.py", "web_fetch.py", "shell_exec.py"}
    ]
    for path in runtime_files + tool_files:
        text = path.read_text()
        for snippet in forbidden_imports:
            assert snippet not in text, f"{path} must not contain {snippet!r}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_environment_text_only() -> None:
    req = TaskRequest(
        prompt="query",
        context=("Paris is the capital of France.", "Berlin is the capital of Germany."),
    )
    env = TextEnvironment(req)
    out = asyncio.run(env.read("paris"))
    assert "Paris" in out
    assert out in "\n".join(req.context)
    with pytest.raises(AttributeError):
        env.fetch  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Package import sanity
# ---------------------------------------------------------------------------


def test_purple_package_imports_cleanly() -> None:
    mod = importlib.import_module("purple")
    for attr in (
        "Orchestrator",
        "TaskRequest",
        "TaskResult",
        "FakeLLM",
        "ChatMessage",
        "ControllerLoop",
        "LLMController",
        "RuleBasedController",
        "ToolRegistry",
        "default_tools",
        "default_registry",
        "PolicyGate",
        "Finalizer",
        "Transcript",
        "Tool",
        "ToolCall",
        "ToolResult",
    ):
        assert hasattr(mod, attr), f"purple missing export {attr!r}"


# ---------------------------------------------------------------------------
# Tools — primitive helpers
# ---------------------------------------------------------------------------


def test_chunk_text_splits_on_paragraphs_and_sentences() -> None:
    text = (
        "Paragraph one is short.\n\n"
        "Paragraph two is much longer and contains multiple sentences. "
        "Sentence two. Sentence three is also here."
    )
    chunks = chunk_text(text, max_chars=60)
    assert chunks
    assert all(len(c) <= 80 for c in chunks)
    assert any("Paragraph one" in c for c in chunks)


def test_search_chunks_ranks_by_overlap() -> None:
    chunks = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Rome is the capital of Italy.",
    ]
    ranked = search_chunks(chunks, "what is the capital of france paris", limit=2)
    assert ranked
    assert "Paris" in ranked[0]


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"b": 2}\n```') == {"b": 2}
    assert extract_json('Sure! Here it is:\n{"c": 3}\nLet me know.') == {"c": 3}
    assert extract_json("not json at all") is None


def test_safe_eval_arithmetic_only() -> None:
    assert safe_eval("1 + 2 * 3") == 7
    assert safe_eval("(4 - 1) / 2") == 1.5
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")
    with pytest.raises(ValueError):
        safe_eval("x + 1")


# ---------------------------------------------------------------------------
# Prompts and skills
# ---------------------------------------------------------------------------


def test_prompts_and_skills_are_present_on_disk() -> None:
    prompts = list_prompts()
    for required in ("system", "controller", "doc_research", "fact_verifier", "composer"):
        assert required in prompts, f"prompts/{required}.md is missing"
        assert load_prompt(required).strip(), f"prompts/{required}.md is empty"

    skills = list_skills()
    for required in ("extraction", "verification", "composition", "tool_use"):
        assert required in skills, f"skills/{required}.md is missing"
        assert load_skill(required).strip(), f"skills/{required}.md is empty"


# ---------------------------------------------------------------------------
# Capability profile sanity check (still present for hint surface)
# ---------------------------------------------------------------------------


def test_capabilities_constant_still_documents_hint_surface() -> None:
    # Plain reachability of the hint surface used by the rule controller.
    assert "doc_research" in CAPABILITIES
    assert "calculator" in CAPABILITIES
