"""Shell/code specialist.

Default build: inert. The specialist never imports ``subprocess`` (or any
process-spawning module) at module top level. A deployer may inject a runner
callable to enable execution out-of-band; without one, the specialist simply
reports that execution is disabled.
"""

from __future__ import annotations

from typing import Callable

from ..schema import StepRecord
from ..state import WorkingState


class ShellCodeSpecialist:
    name = "shell_code"
    capability = "shell_code"

    def __init__(self, runner: Callable[[str], str] | None = None) -> None:
        self._runner = runner

    async def run(self, state: WorkingState) -> StepRecord:
        if self._runner is None:
            return StepRecord(
                capability=self.capability,
                summary="shell execution disabled in public build",
                outputs={"executed": False, "reason": "no runner injected"},
            )

        prompt = state.request.prompt
        try:
            output = self._runner(prompt)
        except Exception as exc:
            return StepRecord(
                capability=self.capability,
                summary=f"runner raised: {type(exc).__name__}",
                outputs={"executed": False, "error": str(exc)},
            )
        return StepRecord(
            capability=self.capability,
            summary="runner returned output",
            outputs={"executed": True, "output": output},
        )
