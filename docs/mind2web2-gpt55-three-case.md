# Mind2Web2 GPT-5.5 3-case direct smoke

This repo is the implementation and test surface for the purple agent. The smoke uses representative public Mind2Web2-style research tasks directly through `scripts/run_mind2web2_three_smoke_gpt55_budget.py`; it does not require modifying benchmark wrapper repositories.

## Command shape

```bash
OPENAI_MODEL=gpt-5.5 LLM_MODEL=gpt-5.5 PYTHONPATH=src \
  uv run --extra test python scripts/run_mind2web2_three_smoke_gpt55_budget.py
```

Budget in the runner:

- `max_steps=40`
- `time_limit_s=600`
- `max_attempts_per_tool=6`

## Latest verified artifact

Ignored local artifact path:

```text
experiment_logs/mind2web2_three_smoke_gpt55_budget/20260529T012020Z/summary.json
```

Probe result: `3/3 PASS_PROBE`.

| case | probe result | relevant generic fix |
| --- | --- | --- |
| `llava_commit` | pass, 5/5 expected markers | GitHub commits/PR API metadata extraction and commit sufficiency guard |
| `yu_lineage` | pass, 4/5 marker threshold with full chain answer | MathGenealogy structured fetch, URL discovery, advisor lineage guard/query refinement |
| `lol_sylas` | pass, 5/5 expected markers | plain web-search result lists no longer overwrite source-backed candidates; unsupported candidates are not emitted |

These are regression probes, not an official AgentX leaderboard score.
