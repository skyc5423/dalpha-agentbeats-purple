# dalpha-agentbeats-purple

A public AgentBeats **purple agent**: a general-purpose A2A agent built
around a Dalpha-Harness-style **controller loop**. An LLM controller picks
primitive tools turn-by-turn, observes each result on a shared transcript,
and replans until it commits a final answer or exhausts its budget. Safety
gates (policy preflight, postflight credential redaction, verifier-aware
finalisation) wrap the loop so the controller cannot bypass them.

It is intentionally public-safe: there is no dataset-specific routing and no
task-to-answer lookup table inside the solver. A deterministic
`RuleBasedController` is used when no LLM is configured, and a `FakeLLM`
makes the full controller loop testable without a network.

This repository was originally seeded from
[`RDI-Foundation/agent-template`](https://github.com/RDI-Foundation/agent-template).

## Architecture

```
A2A request ─▶ Executor ─▶ Agent ─▶ Orchestrator.solve()
                                     │
                                     ├─ PolicyGate.preflight   (hard gate; refusal on hit)
                                     ├─ ControllerLoop
                                     │    repeat until stop:
                                     │      1. controller.next_action(transcript)
                                     │            → ToolCall | Final | Surrender
                                     │      2. ToolRegistry.get(name)
                                     │      3. observation = await tool.run(args, ctx)
                                     │      4. transcript.append(call, observation)
                                     │    stops on: Final | budget | duplicate-call | surrender
                                     │
                                     ├─ Finalizer
                                     │    ├─ verify_claim      (LLM or heuristic; always)
                                     │    └─ compose answer    (LLM or fallback; always)
                                     │
                                     └─ PolicyGate.postflight  (credential redaction)
```

Key properties:

- The controller picks **what to do next**; the orchestrator owns safety and
  termination. Policy + finalize cannot be skipped.
- `CapabilityProfile` survives as a *hint surface* the rule-based controller
  uses to rank candidate tools; nothing branches on dataset names.
- All capabilities are exposed as primitive **tools** behind a single
  `Tool` protocol. The default catalog is generic: `search_docs`,
  `extract_answer`, `calculate`, `web_search`, `web_fetch`, `shell_exec`
  (inert by default), `verify_claim`, `finish`.

**The runtime never branches on benchmark names or task identifiers. It
dispatches purely on tool names and transcript observations.**

## Public-safety constraints

- No dataset-name (e.g. SWE-Bench, Terminal-Bench, Mind2Web, OfficeQA) routing
  in any code path. The runtime sees tool names and transcript observations
  only.
- No task-ID → answer or green-agent-ID → answer lookup tables. The
  `TaskRequest` schema does not carry task or context identifiers into the
  solver; the A2A adapter strips them at the boundary.
- The `shell_exec` tool ships inert. Its module does not import
  `subprocess` (or any process-spawning module) at top level. Deployers may
  inject a runner callable.
- `search_docs` / `extract_answer` operate only on context and attachments
  supplied in the task. Their modules do not import `httpx`, `requests`, or
  `urllib.request` and perform no outbound HTTP.
- `PolicyGate.preflight` flags potentially dangerous instruction patterns and
  short-circuits the controller loop; `PolicyGate.postflight` redacts
  credential-shaped substrings from any composed answer.
- Secrets are supplied via environment variables or AgentBeats / GitHub Actions
  secrets. The repo carries no credentials.

These constraints are enforced by source-scanning tests in
`tests/test_purple_arch.py`.

## Project structure

```text
src/
├─ server.py            # A2A server + agent card
├─ executor.py          # A2A request handling
├─ agent.py             # Thin adapter: A2A message ↔ Orchestrator
├─ messenger.py         # Outbound A2A messaging utility
└─ purple/              # Controller-loop orchestrator package
   ├─ schema.py         # Frozen dataclasses (no IDs)
   ├─ profiler.py       # CapabilityProfiler (hint surface only)
   ├─ budget.py         # Step + time budget tracker
   ├─ state.py          # WorkingState (legacy; runtime uses Transcript)
   ├─ registry.py       # ToolRegistry + default_registry alias
   ├─ orchestrator.py   # Orchestrator.solve()  → controller loop + finalizer
   ├─ io_adapter.py     # A2A message ↔ TaskRequest, TaskResult ↔ Parts
   ├─ environment.py    # Read-only TextEnvironment over context
   ├─ llm.py            # LLMClient, FakeLLM, OpenAICompatibleLLM
   ├─ prompts.py        # Prompt + skill file loader
   ├─ tools/            # chunk_text, search_chunks, extract_json, safe_eval
   ├─ runtime/          # Controller loop, finalizer, policy gate
   │  ├─ controller.py
   │  ├─ llm_controller.py
   │  ├─ rule_controller.py
   │  ├─ loop.py
   │  ├─ finalizer.py
   │  ├─ policy.py
   │  ├─ tool.py
   │  └─ transcript.py
   ├─ tools_api/        # Primitive Tool implementations
   │  ├─ search_docs.py
   │  ├─ extract_answer.py
   │  ├─ calculate.py
   │  ├─ web_search.py
   │  ├─ web_fetch.py
   │  ├─ shell_exec.py
   │  ├─ verify_claim.py
   │  └─ finish.py
   └─ specialists/      # Legacy specialist bodies (not wired into solve())
prompts/                # On-disk prompt artifacts loaded at runtime
├─ system.md
├─ controller.md        # Controller specialist prompt
├─ doc_research.md
├─ fact_verifier.md
├─ composer.md
└─ planner.md           # Deprecated; kept on disk for migration
skills/                 # Reusable skill snippets
├─ extraction.md
├─ verification.md
├─ composition.md
└─ tool_use.md          # Rules for one-tool-per-turn dispatch
tests/
├─ conftest.py
├─ test_agent.py        # A2A conformance smoke tests (needs a live server)
└─ test_purple_arch.py  # Unit + source-grep guardrails + LLM flow
Dockerfile
pyproject.toml
amber-manifest.json5
```

## LLM configuration

LLM wiring is **opt-in**. `default_registry()` accepts an `llm=` parameter,
and `llm_from_env()` builds an `OpenAICompatibleLLM` if any of the following
env vars are set:

| Variable | Default |
|----------|---------|
| `OPENAI_API_KEY` *or* `LLM_API_KEY` | _(required to enable LLM mode)_ |
| `OPENAI_BASE_URL` *or* `LLM_BASE_URL` | `https://api.openai.com/v1` |
| `OPENAI_MODEL` *or* `LLM_MODEL` | `gpt-4o-mini` |

The transport is stdlib `urllib.request` — no extra dependencies. If no key
is configured, every specialist falls back to deterministic behaviour so
the public build still works as a safe skeleton.

```python
from purple import Orchestrator, default_registry, llm_from_env

orchestrator = Orchestrator(registry=default_registry(llm=llm_from_env()))
```

## Prompts and skills

`prompts/*.md` and `skills/*.md` are loaded at runtime through
`purple.prompts.load_prompt(name)` and `purple.prompts.load_skill(name)`.
Edit them directly to tune specialist behaviour without touching code; tests
assert that all required files exist and are non-empty.

## Local development

```bash
uv sync --extra test

# Unit tests for the orchestrator package (no server required)
uv run pytest tests/test_purple_arch.py -v

# A2A conformance: start the server, then run the conformance tests
uv run src/server.py --host 0.0.0.0 --port 9009 &
uv run pytest tests/test_agent.py --agent-url http://localhost:9009 -v
```

## Docker

```bash
docker build -t dalpha-agentbeats-purple .
docker run --rm -p 9009:9009 dalpha-agentbeats-purple
```

## Extending the agent

Register a new primitive tool on the default catalog:

```python
from purple import Orchestrator, default_tools
from purple.runtime.tool import ToolContext, ToolResult


class MyTool:
    name = "my_tool"
    description = "Short description of what this tool does."
    arg_schema = {"query": "input string"}

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        # ... do something deterministic, return a ToolResult
        return ToolResult(
            tool_call_id="",
            ok=True,
            summary="my_tool ran",
            observation="...",
            outputs={"spans": ["evidence span"]},
        )


registry = default_tools()
registry.register(MyTool())
orchestrator = Orchestrator(registry=registry)
```

A deployer with their own sandbox can opt `shell_exec` back in by passing
`shell_runner=` to `default_tools` (or constructing `ShellExecTool` directly).
Tool semantics for the model live entirely in `prompts/controller.md` and the
`arg_schema` advertised by each tool; never hard-code benchmark routes.

## Roadmap

- [x] Wire optional LLM backends behind explicit specialist constructors.
- [x] Load prompts and skills from on-disk artifacts.
- [ ] Harden the planner (richer ordered plans, dependency awareness).
- [ ] Add a structured note / scratch memory across calls.
- [ ] Tighten policy patterns and expand verifier evidence sources.
- [ ] Add a vetted sandbox runner so `ShellCodeSpecialist` can opt in.

## Safety / openness

Do not commit private prompts, internal credentials, `.env` files, customer
data, or internal endpoints. All secrets must be supplied via environment
variables or AgentBeats / GitHub Actions secrets.
