# syntax=docker/dockerfile:1.7

FROM python:3.13 AS builder


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app


COPY pyproject.toml uv.lock README.md ./


RUN uv sync --frozen --no-install-project --no-dev


COPY src/ ./src/
COPY LICENSE ./


RUN uv sync --frozen --no-dev


FROM python:3.13 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"


RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

RUN mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

WORKDIR /app


COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/pyproject.toml /app/

USER app


CMD ["multiagent-interviewer"]
