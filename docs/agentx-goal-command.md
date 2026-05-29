# AgentX / AgentBeats Sprint 4 Hermes `/goal` command

Use this from the Discord/Hermes chat when you want Hermes to keep working autonomously on the AgentX purple-agent improvement loop.

## Current recommended command

```text
/goal Keep working on AgentX/AgentBeats Sprint 4 until the goal is actually achieved; do not voluntarily stop after bounded iterations. The implementation and test workspace is /nas-2/code/sangmin/benchmark/dalpha-agentbeats-purple only. Do not modify/fork/create other benchmark wrapper repositories. For each target benchmark, download or define public benchmark samples/testsets inside dalpha-agentbeats-purple and run the purple agent directly from this repo, saving traces and results under experiment_logs/. The goal is to produce one public-safe general-purpose A2A purple-agent submission candidate. Treat COMPLETE only when: (1) one public-safe repo/image/card/README is ready; (2) at least 5 green agents across at least 3 categories have valid direct-smoke/submission/eval artifacts recorded in agentx_sprint4_matrix.{csv,jsonl}; (3) each selected green agent is either submission_candidate or blocked_external with exact evidence, and none remain unknown/setup_unverified/agent_failure_without_root_cause; (4) anti-cheating/public-safety scan passes with no task/benchmark/green-agent-id routing, answer lookup, evaluator/rubric leakage, private Dalpha code/prompts/credentials/data/endpoints, or credential leakage; (5) cross-benchmark regression smoke passes after shared controller/verifier/finalizer/tool changes; (6) a final Korean submit/not-submit report with artifact paths, before→after, scores/status, blockers, and recommendation exists. If COMPLETE, say Goal complete and do not propose a next improvement step. If incomplete, immediately continue by launching the next bounded work chunk, worker batch, background run, or watchdog; bounded iterations are execution chunks, not stopping points. Do not stop on one blocker if other tracks can still move. Stop as BLOCKED only if all remaining meaningful tracks are externally blocked, and report exact evidence plus required external action. Use gpt-5.5 only for 1–3 hard-case ablations with recorded purpose/case IDs/call budget/stop condition, then recheck general fixes on default models.
```

## Resume/check commands

```text
/goal status
/goal pause
/goal resume
/goal clear
```

## State files to maintain

```text
experiment_logs/agentx_sprint4_goal_loop/
experiment_logs/agentx_sprint4_matrix.csv
experiment_logs/agentx_sprint4_matrix.jsonl
```

## Repo-local execution rule

- Implement agent code only in `dalpha-agentbeats-purple`.
- Test agent behavior only from `dalpha-agentbeats-purple`.
- Bring public benchmark samples/tasks into this repo instead of modifying benchmark wrapper repos.
- Other repos may be read for public schema/sample understanding only; no branch/push/modification there.
