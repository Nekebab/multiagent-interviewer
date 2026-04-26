"""Prompt template loading and rendering via Jinja2."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATES_DIR = Path(__file__).parent


@lru_cache
def _get_env() -> Environment:
    """Return a configured Jinja2 environment, cached so we build it once."""
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


def render(template_name: str, **context: Any) -> str:
    """Render a Jinja2 template by name (e.g. 'expert.j2') with given context.

    Raises:
        TemplateNotFound: If the .j2 file doesn't exist.
        UndefinedError: If a variable is referenced but not provided.
    """
    template = _get_env().get_template(template_name)
    return template.render(**context)
