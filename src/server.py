import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    skill = AgentSkill(
        id="purple-orchestrator",
        name="Capability-based purple agent",
        description=(
            "General-purpose A2A purple agent. A single orchestrator dispatches "
            "the task to capability specialists (planner, shell/code, "
            "document/research, policy, fact verifier, answer composer) based on "
            "a deterministic capability profile of the incoming task. The agent "
            "does no dataset-specific routing and stores no task-to-answer "
            "mappings."
        ),
        tags=[
            "agentbeats",
            "purple-agent",
            "a2a",
            "general-purpose",
            "capability-based",
            "orchestrator",
        ],
        examples=[
            "Read the attached document and answer the user's question.",
            "Summarize the provided context in two sentences.",
            "Given a short code snippet, explain what it does.",
        ],
    )

    agent_card = AgentCard(
        name="dalpha-agentbeats-purple",
        description=(
            "Capability-based A2A purple agent. A single orchestrator dispatches "
            "the task to capability specialists based on a deterministic, "
            "content-feature capability profile; routing never depends on "
            "dataset names or peer-agent identifiers."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
