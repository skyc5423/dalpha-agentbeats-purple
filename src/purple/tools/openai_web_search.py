"""OpenAI Responses API web_search_preview helper."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class WebAnswerer(Protocol):
    async def answer(self, *, prompt: str, system: str = "", max_tokens: int = 1200) -> str: ...


@dataclass(frozen=True)
class OpenAIWebSearchAnswerer:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 60.0

    async def answer(self, *, prompt: str, system: str = "", max_tokens: int = 1200) -> str:
        return await asyncio.to_thread(self._sync_answer, prompt, system, max_tokens)

    def _sync_answer(self, prompt: str, system: str, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system or "You are a careful web research agent."},
                {"role": "user", "content": prompt},
            ],
            "tools": [{"type": "web_search_preview"}],
            "max_output_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/responses",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""
        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        chunks: list[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    value = content.get("text") or content.get("content")
                    if isinstance(value, str):
                        chunks.append(value)
        return "\n".join(chunks).strip()


def openai_web_search_from_env() -> WebAnswerer | None:
    if os.getenv("OPENAI_WEB_SEARCH_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    # Keep the web-search Responses call on a known search-capable default.
    # Controller/chat model env vars (for example gpt-5.x mini variants) may not
    # support ``web_search_preview`` and previously caused silent no-search
    # fallbacks. Only explicit web-search env vars should override this default.
    model = (
        os.getenv("OPENAI_WEB_SEARCH_MODEL")
        or os.getenv("OPENAI_RESPONSES_MODEL")
        or "gpt-4o-mini"
    )
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
    return OpenAIWebSearchAnswerer(api_key=api_key, model=model, base_url=base_url)
