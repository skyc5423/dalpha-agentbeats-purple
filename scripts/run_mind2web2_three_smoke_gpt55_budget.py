#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from purple.orchestrator import Orchestrator
from purple.schema import TaskRequest

CASES = [
    {
        "id": "llava_commit",
        "prompt": "Identify the first commit on the main branch of the official Hugging Face transformers repository that added support for the LLaVA model. Please provide the short commit ID, date, contributors/authors, GitHub profiles and real names.",
        "expect_any": ["44b5506", "2023-12-07", "younesbelkada", "ArthurZucker", "haotian-liu"],
    },
    {
        "id": "yu_lineage",
        "prompt": "Trace OSU Professor Yu Su's doctoral advisor lineage upward for five generations. Provide the lineage names in order and cite advisor-advisee evidence.",
        "expect_any": ["Xifeng Yan", "Jiawei Han", "Larry E. Travis", "Abraham Robinson", "Paul Dienes"],
    },
    {
        "id": "lol_sylas",
        "prompt": "Identify the Worlds finals Sylas/Rakan play and report the player's Sylas win rate during that S-series. Include the player/team and source-backed evidence.",
        "expect_any": ["Faker", "T1", "4", "2", "Sylas"],
    },
]


def score(answer: str, expected: list[str]) -> dict:
    lower = answer.lower()
    hits = [tok for tok in expected if tok.lower() in lower]
    return {"hits": hits, "hit_count": len(hits), "expected_count": len(expected), "ok": len(hits) >= max(1, len(expected)//2)}


def model_name() -> str:
    return os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "(default)"


async def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("experiment_logs/mind2web2_three_smoke_gpt55_budget") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(max_steps=40, time_limit_s=600, max_attempts_per_tool=6)
    rows = []
    for case in CASES:
        result = await orch.solve(TaskRequest(prompt=case["prompt"]))
        row = {
            "id": case["id"],
            "model": model_name(),
            "max_steps": 40,
            "time_limit_s": 600,
            "max_attempts_per_tool": 6,
            "prompt": case["prompt"],
            "answer": result.answer,
            "confidence": result.confidence,
            "flags": list(result.flags),
            "budget": str(result.budget),
            "capabilities": [step.capability for step in result.steps],
            "summaries": [step.summary for step in result.steps],
            "score_probe": score(result.answer, case["expect_any"]),
        }
        rows.append(row)
        (out_dir / f"{case['id']}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"SUMMARY_PATH={out_dir / 'summary.json'}")
    print(f"PASS_PROBE={sum(r['score_probe']['ok'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
