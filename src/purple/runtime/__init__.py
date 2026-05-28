"""Harness-style controller runtime for the purple orchestrator.

This package replaces the fixed specialist pipeline with a controller loop:
an LLM (or a deterministic rule-based fallback) picks a tool to call each
turn, observes the result on a shared transcript, and repeats until it emits
a final answer or the budget is exhausted. Safety gates (policy preflight
and the verifier-aware finalizer) wrap the loop so the controller cannot
bypass them.
"""

from .controller import Action, Controller, FinalAnswer, Surrender
from .finalizer import FinalizationResult, Finalizer
from .llm_controller import LLMController
from .loop import ControllerLoop, LoopOutcome
from .policy import PolicyGate, PolicyOutcome
from .rule_controller import RuleBasedController
from .tool import Tool, ToolCall, ToolContext, ToolResult
from .transcript import Transcript

__all__ = [
    "Action",
    "Controller",
    "ControllerLoop",
    "FinalAnswer",
    "FinalizationResult",
    "Finalizer",
    "LLMController",
    "LoopOutcome",
    "PolicyGate",
    "PolicyOutcome",
    "RuleBasedController",
    "Surrender",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolResult",
    "Transcript",
]
