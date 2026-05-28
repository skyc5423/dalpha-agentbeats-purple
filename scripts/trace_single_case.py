#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from purple.llm import ChatMessage, LLMClient, OpenAICompatibleLLM
from purple.orchestrator import Orchestrator
from purple.schema import TaskRequest

CASES = {
    "llava_commit": "Identify the first commit on the main branch of the official Hugging Face transformers repository that added support for the LLaVA model. Please provide the short commit ID, date, contributors/authors, GitHub profiles and real names.",
    "yu_lineage": "Trace OSU Professor Yu Su's doctoral advisor lineage upward for five generations. Provide the lineage names in order and cite advisor-advisee evidence.",
    "lol_sylas": "Identify the Worlds finals Sylas/Rakan play and report the player's Sylas win rate during that S-series. Include the player/team and source-backed evidence.",
}

class RecordingLLM:
    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        tag: str = "",
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        text = await self.inner.complete(
            messages=messages,
            tag=tag,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.calls.append(
            {
                "tag": tag,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [m.to_dict() for m in messages],
                "response": text,
            }
        )
        return text


def build_llm() -> RecordingLLM:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY/LLM_API_KEY missing")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-5.5"
    return RecordingLLM(OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model, timeout_s=60.0))

async def main() -> int:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "lol_sylas"
    if case_id not in CASES:
        raise SystemExit(f"unknown case {case_id}; choose {sorted(CASES)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("experiment_logs/trace_single_case") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = build_llm()
    orch = Orchestrator(llm=llm, max_steps=int(os.getenv("TRACE_MAX_STEPS", "40")), time_limit_s=600, max_attempts_per_tool=6)
    result = await orch.solve(TaskRequest(prompt=CASES[case_id]))
    data = {
        "case_id": case_id,
        "prompt": CASES[case_id],
        "answer": result.answer,
        "confidence": result.confidence,
        "flags": list(result.flags),
        "budget": str(result.budget),
        "steps": [
            {"capability": s.capability, "summary": s.summary, "outputs": s.outputs}
            for s in result.steps
        ],
        "llm_calls": llm.calls,
    }
    path = out_dir / f"{case_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    print(f"TRACE_PATH={path}")
    print(f"ANSWER={result.answer}")
    print(f"STEPS={[s.capability for s in result.steps]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
