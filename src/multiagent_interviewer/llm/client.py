"""Wrapper around the Mistral SDK with retries and structured-output support."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeVar

from loguru import logger
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multiagent_interviewer.config import get_settings

if TYPE_CHECKING:
    from mistralai import Mistral


T = TypeVar("T", bound=BaseModel)


_RETRYABLE = (
    ConnectionError,
    TimeoutError,
    OSError,
)


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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
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
        full_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY a JSON object matching this schema. No prose, "
            f"no markdown code fences:\n"
            f"{json.dumps(json_schema, ensure_ascii=False, indent=2)}"
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
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
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
