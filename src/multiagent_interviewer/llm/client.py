"""Wrapper around the Mistral SDK with retries and structured-output support."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from multiagent_interviewer.config import get_settings

if TYPE_CHECKING:
    from mistralai import Mistral


T = TypeVar("T", bound=BaseModel)


def _log_retry_attempt(retry_state: Any) -> None:
    """Log a retry attempt without letting curly braces in exception
    messages confuse loguru's str.format-style placeholders.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    sleep_for = retry_state.next_action.sleep if retry_state.next_action else 0
    attempt = retry_state.attempt_number
    logger.warning(
        "Retry #{} after {:.1f}s; reason: {}",
        attempt,
        sleep_for,
        repr(exception),
    )


def _should_retry(exception: BaseException) -> bool:
    """Decide whether an exception is worth retrying.

    Retries:
      - Network and timeout errors (transient)
      - Mistral API rate-limit errors (HTTP 429) — wait it out
      - Mistral API server errors (HTTP 5xx) — server hiccup

    Does NOT retry:
      - 4xx errors except 429 (auth, bad request — won't fix themselves)
      - JSON parse errors (won't fix themselves)
    """
    if isinstance(exception, ConnectionError | TimeoutError | OSError):
        return True

    # Mistral SDK error: check status code
    status_code: int | None = getattr(exception, "status_code", None)
    if status_code is None:
        message = str(exception)
        return (
            "429" in message
            or "Rate limit" in message
            or any(f"50{n}" in message for n in (0, 1, 2, 3, 4))
        )

    return status_code == 429 or 500 <= status_code < 600


def _example_from_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Build a minimal example dict for `schema` using field defaults and dummies."""
    from pydantic_core import PydanticUndefined

    example: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if field.default is not PydanticUndefined:
            example[name] = field.default
            continue
        if field.default_factory is not None:
            try:
                example[name] = field.default_factory()  # type: ignore[call-arg]
                continue
            except Exception:
                pass

        ann_str = str(field.annotation)
        ann_lower = ann_str.lower()
        if "list" in ann_lower:
            example[name] = []
        elif "bool" in ann_lower:
            example[name] = False
        elif "int" in ann_lower:
            example[name] = 0
        elif "float" in ann_lower:
            example[name] = 0.0
        else:
            example[name] = "..."
    return example


class LLMClient:
    """A thin wrapper around the Mistral chat-completion API.

    Adds three things over the raw SDK:
      1. Centralized config (model, temperature, retries) read from Settings
      2. Automatic retry with exponential backoff on transient failures
      3. Structured-output helper that validates LLM JSON against a Pydantic model
    """

    def __init__(self, client: Mistral | None = None) -> None:
        settings = get_settings()
        self._settings = settings
        if client is None:
            from mistralai import Mistral

            client = Mistral(api_key=settings.mistral_api_key)
        self._client = client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception(_should_retry),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def complete(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> str:
        """Run a single chat completion. Returns the model's text response."""
        messages: list[Any] = [{"role": "user", "content": prompt}]
        response = self._client.chat.complete(
            model=model or self._settings.llm_model,
            messages=messages,
            temperature=temperature if temperature is not None else self._settings.llm_temperature,
            response_format={"type": "json_object"},
        )
        return self._extract_content(response)

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float | None = None,
        model: str | None = None,
    ) -> T:
        """Run a completion that returns JSON matching `schema`."""
        json_schema = schema.model_json_schema()

        example = _example_from_schema(schema)

        full_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY a single JSON object with these fields. "
            f"Do NOT return the schema itself, do NOT wrap in extra keys, "
            f"do NOT add markdown code fences.\n\n"
            f"SCHEMA (for reference):\n"
            f"{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n\n"
            f"EXAMPLE (the format you must return — replace values with real ones):\n"
            f"{json.dumps(example, ensure_ascii=False, indent=2)}"
        )

        raw = self._complete_json(full_prompt, temperature, model)
        return schema.model_validate_json(raw)

    @staticmethod
    def _extract_content(response: object) -> str:
        """Extract message content from a Mistral chat-completion response."""
        if response is None:
            raise ValueError("Mistral API returned no response")

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Mistral API returned no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("Mistral API returned a choice with no message")

        content = getattr(message, "content", None)
        if content is None:
            raise ValueError("LLM returned empty response")
        return content if isinstance(content, str) else str(content)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception(_should_retry),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )
    def _complete_json(
        self,
        prompt: str,
        temperature: float | None,
        model: str | None,
    ) -> str:
        """completion with JSON response_format set."""
        messages: list[Any] = [{"role": "user", "content": prompt}]
        response = self._client.chat.complete(
            model=model or self._settings.llm_model,
            messages=messages,
            temperature=temperature if temperature is not None else self._settings.llm_temperature,
            response_format={"type": "json_object"},
        )
        return self._extract_content(response)
