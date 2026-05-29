#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from purple.orchestrator import Orchestrator
from purple.schema import TaskRequest


def _score(answer: str, markers: list[str]) -> dict[str, Any]:
    if not markers:
        return {"markers": [], "hits": [], "hit_count": 0, "expected_count": 0, "ok": None}
    lower = answer.lower()
    hits = [m for m in markers if m.lower() in lower]
    # Direct smokes are coverage probes, not official scorers. Require at least
    # half the expected public markers, with a minimum of one marker.
    threshold = max(1, (len(markers) + 1) // 2)
    return {
        "markers": markers,
        "hits": hits,
        "hit_count": len(hits),
        "expected_count": len(markers),
        "threshold": threshold,
        "ok": len(hits) >= threshold,
    }


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "(default/no-llm)"


def _shrink(value: Any, limit: int = 1200) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    if isinstance(value, list):
        return [_shrink(item, limit) for item in value[:8]]
    if isinstance(value, dict):
        return {str(k): _shrink(v, limit) for k, v in list(value.items())[:20]}
    return value


async def _run_case(case: dict[str, Any], orch: Orchestrator, out_dir: Path) -> dict[str, Any]:
    result = await orch.solve(TaskRequest(prompt=str(case["prompt"])))
    row = {
        "benchmark": case.get("benchmark", ""),
        "category": case.get("category", ""),
        "sample_id": case.get("sample_id", ""),
        "model": _model_name(),
        "prompt": case.get("prompt", ""),
        "answer": result.answer,
        "confidence": result.confidence,
        "flags": list(result.flags),
        "budget": {
            "steps_used": result.budget.steps_used,
            "steps_limit": result.budget.steps_limit,
            "elapsed_s": result.budget.elapsed_s,
            "time_limit_s": result.budget.time_limit_s,
        },
        "capabilities": [step.capability for step in result.steps],
        "summaries": [step.summary for step in result.steps],
        "trace": [
            {
                "step": i + 1,
                "capability": step.capability,
                "summary": step.summary,
                "outputs": _shrink(dict(step.outputs)),
            }
            for i, step in enumerate(result.steps)
        ],
        "score_probe": _score(result.answer, list(case.get("expected_markers") or [])),
    }
    case_path = out_dir / f"{row['sample_id']}.json"
    case_path.write_text(json.dumps(row, ensure_ascii=False, indent=2))
    row["artifact_path"] = str(case_path)
    print(json.dumps({k: row[k] for k in ["benchmark", "sample_id", "model", "confidence", "flags", "score_probe", "artifact_path"]}, ensure_ascii=False), flush=True)
    return row


def _write_matrix(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "benchmark",
        "category",
        "sample_id",
        "model",
        "artifact_path",
        "status",
        "result_summary",
        "steps_used",
        "elapsed_s",
        "updated_at",
    ]
    matrix_rows = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for r in rows:
        score = r.get("score_probe") or {}
        ok = score.get("ok")
        if ok is True:
            status = "improved_direct_smoke"
        elif ok is False:
            status = "agent_failure_without_root_cause"
        else:
            status = "direct_smoke_unscored"
        matrix_rows.append(
            {
                "benchmark": r.get("benchmark", ""),
                "category": r.get("category", ""),
                "sample_id": r.get("sample_id", ""),
                "model": r.get("model", ""),
                "artifact_path": r.get("artifact_path", ""),
                "status": status,
                "result_summary": f"markers {score.get('hit_count')}/{score.get('expected_count')} confidence={r.get('confidence')}",
                "steps_used": (r.get("budget") or {}).get("steps_used", ""),
                "elapsed_s": (r.get("budget") or {}).get("elapsed_s", ""),
                "updated_at": stamp,
            }
        )
    with (out_dir / "direct_smoke_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(matrix_rows)
    with (out_dir / "direct_smoke_matrix.jsonl").open("w") as f:
        for r in matrix_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run repo-local AgentX direct smoke samples through the purple orchestrator.")
    parser.add_argument("--samples", default="benchmark_samples/agentx_direct_smoke_v1.json")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated sample_id filter for focused reruns.",
    )
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--time-limit-s", type=float, default=240.0)
    parser.add_argument("--max-attempts-per-tool", type=int, default=4)
    parser.add_argument("--out-root", default="experiment_logs/agentx_direct_smoke")
    args = parser.parse_args()

    samples = json.loads(Path(args.samples).read_text())
    if args.only.strip():
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        samples = [case for case in samples if str(case.get("sample_id")) in wanted]
        missing = wanted - {str(case.get("sample_id")) for case in samples}
        if missing:
            raise SystemExit(f"Unknown sample_id(s) for --only: {', '.join(sorted(missing))}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "started_at": stamp,
        "samples": args.samples,
        "model": _model_name(),
        "max_steps": args.max_steps,
        "time_limit_s": args.time_limit_s,
        "max_attempts_per_tool": args.max_attempts_per_tool,
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "note": "Repo-local direct smoke; not an official AgentX leaderboard score.",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2))

    orch = Orchestrator(
        max_steps=args.max_steps,
        time_limit_s=args.time_limit_s,
        max_attempts_per_tool=args.max_attempts_per_tool,
    )
    rows = []
    for case in samples:
        rows.append(await _run_case(case, orch, out_dir))
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    _write_matrix(rows, out_dir)

    passed = sum(1 for r in rows if (r.get("score_probe") or {}).get("ok") is True)
    scored = sum(1 for r in rows if (r.get("score_probe") or {}).get("ok") is not None)
    print(f"SUMMARY_PATH={out_dir / 'summary.json'}")
    print(f"DIRECT_SMOKE_MARKER_PASS={passed}/{scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
