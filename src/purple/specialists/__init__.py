"""Default specialist implementations for the purple orchestrator."""

from .base import Specialist
from .calculator import CalculatorSpecialist
from .composer import AnswerComposerSpecialist
from .doc_research import DocResearchSpecialist
from .fact_verifier import FactVerifierSpecialist
from .planner import PlannerSpecialist
from .policy import PolicyComplianceSpecialist
from .shell_code import ShellCodeSpecialist
from .web_research import WebResearchSpecialist

__all__ = [
    "Specialist",
    "CalculatorSpecialist",
    "AnswerComposerSpecialist",
    "DocResearchSpecialist",
    "FactVerifierSpecialist",
    "PlannerSpecialist",
    "PolicyComplianceSpecialist",
    "ShellCodeSpecialist",
    "WebResearchSpecialist",
]
