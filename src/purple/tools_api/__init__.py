"""Primitive tools exposed to the controller.

Each module exports a single ``Tool`` class with a generic, benchmark-shape
free name (``search_docs``, ``extract_answer``, ``calculate``, ``web_search``,
``web_fetch``, ``shell_exec``, ``verify_claim``, ``finish``). The controller
never branches on dataset names; tools advertise themselves via ``description``
and ``arg_schema`` only.
"""

from __future__ import annotations

from typing import Callable

from ..llm import LLMClient
from ..registry import ToolRegistry
from ..tools import WebAnswerer, WebClient
from .analyze_requirements import AnalyzeRequirementsTool
from .calculate import CalculateTool
from .extract_answer import ExtractAnswerTool
from .finish import FinishTool
from .research_answer import ResearchAnswerTool
from .search_docs import SearchDocsTool
from .shell_exec import ShellExecTool
from .sufficiency_check import SufficiencyCheckTool
from .verify_claim import VerifyClaimTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool


def default_tools(
    *,
    llm: LLMClient | None = None,
    web_client: WebClient | None = None,
    web_answerer: WebAnswerer | None = None,
    shell_runner: Callable[[str], str] | None = None,
    use_env_web_answerer: bool = False,
) -> ToolRegistry:
    """Build the standard tool catalog the controller dispatches against."""

    registry = ToolRegistry()
    registry.register(AnalyzeRequirementsTool(llm=llm))
    registry.register(SearchDocsTool())
    registry.register(ResearchAnswerTool(web_answerer=web_answerer))
    registry.register(ExtractAnswerTool(llm=llm))
    registry.register(CalculateTool())
    registry.register(
        WebSearchTool(
            llm=llm,
            web_client=web_client,
            web_answerer=web_answerer,
            use_env_web_answerer=use_env_web_answerer,
        )
    )
    registry.register(WebFetchTool(web_client=web_client))
    registry.register(ShellExecTool(runner=shell_runner))
    registry.register(VerifyClaimTool(llm=llm))
    registry.register(SufficiencyCheckTool(llm=llm))
    registry.register(FinishTool())
    return registry


__all__ = [
    "CalculateTool",
    "AnalyzeRequirementsTool",
    "ExtractAnswerTool",
    "FinishTool",
    "ResearchAnswerTool",
    "SearchDocsTool",
    "ShellExecTool",
    "SufficiencyCheckTool",
    "VerifyClaimTool",
    "WebFetchTool",
    "WebSearchTool",
    "default_tools",
]
