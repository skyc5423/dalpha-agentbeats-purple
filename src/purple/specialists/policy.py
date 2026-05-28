"""Policy / compliance specialist.

Inspects the prompt for obviously dangerous instruction patterns and adds
flags to ``state.notes['flags']``. Never rewrites or censors the prompt.
"""

from __future__ import annotations

from ..schema import StepRecord
from ..state import WorkingState

_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("destructive_disk_command", "rm -rf /"),
    ("destructive_disk_command", "rm -rf ~"),
    ("disk_overwrite", "dd if=/dev/zero of=/dev/"),
    ("fork_bomb", ":(){ :|:& };:"),
    ("credential_exfiltration", "/etc/shadow"),
    ("private_key_exfiltration", "id_rsa"),
)


class PolicyComplianceSpecialist:
    name = "policy"
    capability = "policy"

    async def run(self, state: WorkingState) -> StepRecord:
        haystack_parts = [state.request.prompt or ""]
        haystack_parts.extend(state.request.context)
        for att in state.request.attachments:
            if att.text:
                haystack_parts.append(att.text)
        haystack = "\n".join(haystack_parts).lower()

        flags: list[str] = []
        for label, pattern in _DENY_PATTERNS:
            if pattern.lower() in haystack:
                flags.append(f"policy-flag:{label}")

        existing = list(state.get_note("flags", ()))
        for flag in flags:
            if flag not in existing:
                existing.append(flag)
        state.set_note("flags", tuple(existing))

        summary = "no policy flags" if not flags else f"flagged {len(flags)} pattern(s)"
        return StepRecord(
            capability=self.capability,
            summary=summary,
            outputs={"flags": flags},
        )
