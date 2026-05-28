"""Load prompts and skills from disk.

Prompts and skills are real repo artifacts under ``prompts/`` and ``skills/``
so they can be reviewed and edited by humans without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
# src/purple/prompts.py → src/purple → src → repo root
_REPO_ROOT = _PKG_ROOT.parent.parent
PROMPTS_DIR = _REPO_ROOT / "prompts"
SKILLS_DIR = _REPO_ROOT / "skills"


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Return the text of ``prompts/<name>.md`` or ``""`` if missing."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=64)
def load_skill(name: str) -> str:
    """Return the text of ``skills/<name>.md`` or ``""`` if missing."""
    path = SKILLS_DIR / f"{name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def list_prompts() -> tuple[str, ...]:
    if not PROMPTS_DIR.exists():
        return ()
    return tuple(sorted(p.stem for p in PROMPTS_DIR.glob("*.md")))


def list_skills() -> tuple[str, ...]:
    if not SKILLS_DIR.exists():
        return ()
    return tuple(sorted(p.stem for p in SKILLS_DIR.glob("*.md")))


__all__ = [
    "PROMPTS_DIR",
    "SKILLS_DIR",
    "list_prompts",
    "list_skills",
    "load_prompt",
    "load_skill",
]
