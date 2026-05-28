"""Best-effort JSON extraction from LLM text output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\s*```$")
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def extract_json(text: str) -> Any:
    """Try to parse ``text`` as JSON, tolerating fences and surrounding prose.

    Returns ``None`` if no JSON object/array can be recovered.
    """
    if not text:
        return None
    stripped = text.strip()
    stripped = _FENCE_RE.sub("", stripped)
    stripped = _FENCE_END_RE.sub("", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Take the widest object substring, fall back to the widest array.
    for pattern in (_OBJ_RE, _ARR_RE):
        match = pattern.search(stripped)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None
