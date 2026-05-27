# dalpha-agentbeats-purple

A public AgentBeats **purple agent** prototype for experimenting with A2A-compatible benchmark submissions.

This repository was created from [`RDI-Foundation/agent-template`](https://github.com/RDI-Foundation/agent-template). The current implementation is intentionally minimal: it exposes a valid A2A server and returns a deterministic smoke-test echo. Benchmark-specific reasoning loops will be added behind the same A2A interface.

## Goals

1. Keep the competition submission code public and reproducible.
2. Avoid depending on private Dalpha Harness/runtime code or internal credentials.
3. Reuse Dalpha Harness design ideas where safe: config-driven agents, tools, prompts/skills, verifier loops, and benchmark-specific profiles.
4. Start with AgentBeats packaging/conformance, then iterate on benchmark performance.

## Project structure

```text
src/
├─ server.py      # A2A server and agent card
├─ executor.py    # A2A request handling
├─ agent.py       # Agent implementation entry point
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # A2A conformance smoke tests
Dockerfile
pyproject.toml
amber-manifest.json5
```

## Local development

```bash
uv sync --extra test
uv run src/server.py --host 0.0.0.0 --port 9009
```

In another shell:

```bash
uv run pytest --agent-url http://localhost:9009
```

## Docker

```bash
docker build -t dalpha-agentbeats-purple .
docker run --rm -p 9009:9009 dalpha-agentbeats-purple
```

## Roadmap

- [ ] Verify A2A conformance and Docker build locally.
- [ ] Register the smoke-test purple agent on AgentBeats.
- [ ] Add a benchmark profile system.
- [ ] Add Terminal Bench 2.0 shell-loop prototype.
- [ ] Add SWE-bench style repository-edit loop.
- [ ] Add web/research profile for Mind2Web2 or OfficeQA.

## Safety / openness

Do not commit private Dalpha Harness code, internal prompts, `.env` files, credentials, customer data, or internal endpoints. All secrets must be supplied via environment variables or AgentBeats/GitHub Actions secrets.
