"""Policy gate — pre- and post-flight safety hooks the controller can't bypass."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import TaskRequest


_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("destructive_disk_command", "rm -rf /"),
    ("destructive_disk_command", "rm -rf ~"),
    ("disk_overwrite", "dd if=/dev/zero of=/dev/"),
    ("fork_bomb", ":(){ :|:& };:"),
    ("credential_exfiltration", "/etc/shadow"),
    ("private_key_exfiltration", "id_rsa"),
)


_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abpsr]-[A-Za-z0-9-]{10,}\b"),
)


@dataclass(frozen=True)
class PolicyOutcome:
    flags: tuple[str, ...] = ()
    redacted: bool = False
    reason: str = ""


class PolicyGate:
    """Pre-flight scans the prompt for dangerous patterns; post-flight scans
    the final answer for leaked-credential patterns and redacts on hit."""

    def preflight(self, request: TaskRequest) -> PolicyOutcome:
        haystack_parts: list[str] = [request.prompt or ""]
        haystack_parts.extend(request.context)
        for att in request.attachments:
            if att.text:
                haystack_parts.append(att.text)
        haystack = "\n".join(haystack_parts).lower()

        flags: list[str] = []
        for label, pattern in _DENY_PATTERNS:
            if pattern.lower() in haystack:
                tag = f"policy-flag:{label}"
                if tag not in flags:
                    flags.append(tag)
        return PolicyOutcome(flags=tuple(flags))

    def postflight(self, answer: str) -> PolicyOutcome:
        if not answer:
            return PolicyOutcome()
        for pat in _SECRET_PATTERNS:
            if pat.search(answer):
                return PolicyOutcome(
                    flags=("policy-flag:credential_in_answer",),
                    redacted=True,
                    reason="answer contained credential-like pattern",
                )
        return PolicyOutcome()

    @staticmethod
    def refusal_for(flags: tuple[str, ...]) -> str:
        if any("destructive_disk_command" in f for f in flags):
            return "No — avoid running destructive disk commands because they can permanently delete system data."
        if any("disk_overwrite" in f for f in flags):
            return "No — avoid disk-overwrite commands unless you have explicit recovery media and target verification."
        if any("fork_bomb" in f for f in flags):
            return "No — avoid fork bombs because they can make the system unusable."
        if any(("credential" in f or "private_key" in f) for f in flags):
            return "No — I can’t help expose or extract credentials or private keys."
        return "No — that request appears unsafe, so I can’t help perform it."


__all__ = ["PolicyGate", "PolicyOutcome"]
