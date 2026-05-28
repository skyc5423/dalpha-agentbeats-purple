#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
from purple.schema import TaskRequest
from purple.runtime.tool import ToolContext
from purple.tools_api.research_answer import ResearchAnswerTool

async def main():
    q = "What is 100g당 5천원 소고기 200g plus 100g당 3천원 돼지고기 300g after 20% discount?"
    tool = ResearchAnswerTool()
    res = await tool.run({}, ToolContext(request=TaskRequest(prompt=q), notes={}, scratch={}, steps_remaining=5))
    print('ok', res.ok)
    print('summary', res.summary)
    print('obs', res.observation[:1000])

asyncio.run(main())
