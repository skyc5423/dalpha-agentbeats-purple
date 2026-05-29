from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from purple.tools.openai_web_search import openai_web_search_from_env


def test_web_search_model_defaults_to_search_capable_model_when_controller_model_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.delenv("OPENAI_WEB_SEARCH_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_RESPONSES_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_WEB_SEARCH_DISABLED", raising=False)

    answerer = openai_web_search_from_env()

    assert answerer is not None
    assert getattr(answerer, "model") == "gpt-4o-mini"


def test_web_search_model_can_be_overridden_explicitly(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_WEB_SEARCH_MODEL", "gpt-4.1")
    monkeypatch.delenv("OPENAI_WEB_SEARCH_DISABLED", raising=False)

    answerer = openai_web_search_from_env()

    assert answerer is not None
    assert getattr(answerer, "model") == "gpt-4.1"
