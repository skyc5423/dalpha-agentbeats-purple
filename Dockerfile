FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser --disabled-password --gecos "" agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009