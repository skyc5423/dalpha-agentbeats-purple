import os

import httpx
import pytest


# Strip LLM env vars at collection time so unit tests under
# ``Orchestrator()`` do not accidentally pick up a real API key from the dev
# shell. Tests that need an LLM inject one explicitly via ``llm=``.
for _var in (
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_BASE_URL",
    "OPENAI_MODEL",
    "LLM_MODEL",
):
    os.environ.pop(_var, None)
os.environ["OPENAI_WEB_SEARCH_DISABLED"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Agent must be running before tests start."""
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.exit(f"Agent at {url} returned status {response.status_code}", returncode=1)
    except Exception as e:
        pytest.exit(f"Could not connect to agent at {url}: {e}", returncode=1)

    return url
