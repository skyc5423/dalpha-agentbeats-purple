"""Pluggable LLM client.

Two concrete implementations ship with the package:

- :class:`FakeLLM` — deterministic, scripted responder used in tests.
- :class:`OpenAICompatibleLLM` — speaks the OpenAI Chat Completions wire
  format over stdlib ``urllib``. Reads its config from environment variables.

If no API key is configured, :func:`llm_from_env` returns ``None`` and the
specialists fall back to deterministic behaviour. The orchestrator itself
never imports this module; it is wired in through the registry by callers
that have explicitly opted in.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMError(RuntimeError):
    """Raised when the LLM call fails for any reason."""


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        tag: str = "",
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str: ...


class FakeLLM:
    """Deterministic LLM stub used in tests.

    Either pass ``scripted={"<tag>": "<response>"}`` for static routing, or a
    ``responder`` callable for fully dynamic behaviour. Recorded calls are
    available on ``self.calls`` for assertions.
    """

    def __init__(
        self,
        scripted: dict[str, str] | None = None,
        *,
        responder: Callable[[list[ChatMessage], str], str] | None = None,
        default: str = "",
    ) -> None:
        self._scripted = dict(scripted or {})
        self._responder = responder
        self._default = default
        self.calls: list[dict] = []

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        tag: str = "",
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            {
                "tag": tag,
                "messages": [m.to_dict() for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._responder is not None:
            return self._responder(messages, tag)
        if tag in self._scripted:
            return self._scripted[tag]
        return self._default


class OpenAICompatibleLLM:
    """Minimal OpenAI Chat Completions client over stdlib ``urllib``.

    Synchronous HTTP is dispatched to a worker thread so ``complete`` is safe
    to await from an event loop without blocking it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    async def complete(
        self,
        *,
        messages: list[ChatMessage],
        tag: str = "",
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(
            self._sync_complete, messages, max_tokens, temperature
        )

    def _sync_complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
        }
        # Newer OpenAI reasoning models (including gpt-5.x) reject the legacy
        # chat-completions ``max_tokens`` field and only accept the default
        # temperature. Keep the stdlib client compatible with both families.
        if self._model.startswith("gpt-5"):
            payload["max_completion_tokens"] = max_tokens
            if temperature == 1:
                payload["temperature"] = temperature
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"LLM HTTP call failed: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON body: {exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""


def llm_from_env() -> LLMClient | None:
    """Return an :class:`OpenAICompatibleLLM` if env config is present, else
    ``None``. The orchestrator's default build does not call this — wiring is
    explicit so callers can choose.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)


__all__ = [
    "ChatMessage",
    "FakeLLM",
    "LLMClient",
    "LLMError",
    "OpenAICompatibleLLM",
    "llm_from_env",
]
