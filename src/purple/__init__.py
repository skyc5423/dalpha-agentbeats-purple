"""Controller-loop purple-agent package.

Public entry points:

- :class:`Orchestrator` — runs the controller loop for a task request.
- :class:`TaskRequest` / :class:`TaskResult` — frozen wire-agnostic schemas.
- :class:`ToolRegistry` + :func:`default_registry` / :func:`default_tools` —
  the tool catalog the controller dispatches against.
- :class:`Tool`, :class:`ToolCall`, :class:`ToolResult` — primitive tool API.
- :class:`ControllerLoop`, :class:`LLMController`, :class:`RuleBasedController`,
  :class:`Finalizer`, :class:`PolicyGate`, :class:`Transcript` — runtime
  building blocks.
- :class:`LLMClient`, :class:`FakeLLM`, :class:`OpenAICompatibleLLM`,
  :func:`llm_from_env` — LLM client surface.
- :func:`a2a_message_to_request`, :func:`result_to_artifact_parts` —
  conversion helpers for the A2A adapter layer.

The package never branches on dataset names or peer-agent identifiers.
"""

from .io_adapter import (
    a2a_message_to_request,
    result_to_artifact_parts,
    result_to_status_message,
)
from .llm import (
    ChatMessage,
    FakeLLM,
    LLMClient,
    LLMError,
    OpenAICompatibleLLM,
    llm_from_env,
)
from .orchestrator import Orchestrator
from .profiler import CAPABILITIES, CapabilityProfiler
from .prompts import list_prompts, list_skills, load_prompt, load_skill
from .registry import ToolRegistry, default_registry
from .runtime import (
    ControllerLoop,
    FinalAnswer,
    FinalizationResult,
    Finalizer,
    LLMController,
    PolicyGate,
    RuleBasedController,
    Surrender,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
    Transcript,
)
from .schema import (
    Attachment,
    BudgetSnapshot,
    CapabilityProfile,
    StepRecord,
    TaskRequest,
    TaskResult,
)
from .tools_api import default_tools

__all__ = [
    "Attachment",
    "BudgetSnapshot",
    "CAPABILITIES",
    "CapabilityProfile",
    "CapabilityProfiler",
    "ChatMessage",
    "ControllerLoop",
    "FakeLLM",
    "FinalAnswer",
    "FinalizationResult",
    "Finalizer",
    "LLMClient",
    "LLMController",
    "LLMError",
    "OpenAICompatibleLLM",
    "Orchestrator",
    "PolicyGate",
    "RuleBasedController",
    "StepRecord",
    "Surrender",
    "TaskRequest",
    "TaskResult",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "Transcript",
    "a2a_message_to_request",
    "default_registry",
    "default_tools",
    "list_prompts",
    "list_skills",
    "llm_from_env",
    "load_prompt",
    "load_skill",
    "result_to_artifact_parts",
    "result_to_status_message",
]
