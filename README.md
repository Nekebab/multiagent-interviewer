# Multi-Agent Interviewer

Multi-agent AI interview system with hybrid RAG (BM25 + bi-encoder + cross-encoder).
Three LLM agents — Expert, Manager, Interviewer — coordinate via LangGraph to conduct
adaptive technical interviews and produce structured feedback.

> **Status:** Under active refactoring from a Jupyter notebook into a production-grade
> Python package. See [docs/architecture.md](docs/architecture.md).

## Quick start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Setup

```bash
# Clone the repo
git clone https://github.com/Nekebab/multiagent-interviewer.git
cd multiagent-interviewer

# Sync dependencies (creates .venv automatically)
uv sync

# Install the package itself in editable mode
uv pip install -e .

# Copy env template and add your API keys
cp .env.example .env
# edit .env in your editor

# Run tests to verify everything is wired up
uv run pytest
```

All tests should pass, confirming the setup is correct.

## Development workflow

```bash
uv run pytest                  # run tests
uv run ruff check              # lint
uv run ruff format             # auto-format
uv run mypy src tests          # type-check
uv run pre-commit run --all    # run all hooks across the codebase
```

## Project layout

```
src/multiagent_interviewer/
├── config.py            # Pydantic-based settings, loaded from .env
├── logging_setup.py     # Loguru configuration
├── rag/                 # Retrieval-Augmented Generation (BM25 + bi-encoder + cross-encoder)
├── tools/               # Agent tools: internet search, etc.
├── agents/              # Expert / Manager / Interviewer LLM nodes
├── graph/               # LangGraph state and graph builder
├── llm/                 # LLM client wrapper with retries
└── prompts/             # Jinja2 prompt templates
```

## License

MIT
