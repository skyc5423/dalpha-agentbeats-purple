#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from purple.llm import ChatMessage, llm_from_env

async def main():
    llm = llm_from_env()
    print('configured', bool(llm), 'model', os.getenv('OPENAI_MODEL') or os.getenv('LLM_MODEL'))
    if not llm:
        return
    try:
        text = await llm.complete(messages=[ChatMessage('user','Return exactly JSON: {"action":"final","answer":"ok"}')], tag='smoke', max_tokens=40)
        print('response_prefix', repr(text[:200]))
    except Exception as e:
        print('error', type(e).__name__, str(e)[:300])

asyncio.run(main())
