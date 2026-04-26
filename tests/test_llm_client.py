"""Tests for the LLM client wrapper.

These tests use mocks to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from multiagent_interviewer.llm.client import LLMClient


class FakeResponse:
    """Minimal stand-in for Mistral's chat.complete() response."""

    def __init__(self, content: str) -> None:
        self.choices = [MagicMock(message=MagicMock(content=content))]


class _OutputSchema(BaseModel):
    name: str
    age: int


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key-for-tests")


class TestComplete:
    def test_returns_text_content(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse("Hello there")

        llm = LLMClient(client=mock_client)
        result = llm.complete("Say hi")

        assert result == "Hello there"
        mock_client.chat.complete.assert_called_once()

    def test_uses_default_model_and_temperature(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse("ok")

        llm = LLMClient(client=mock_client)
        llm.complete("anything")

        call_kwargs = mock_client.chat.complete.call_args.kwargs
        assert call_kwargs["model"] == "mistral-large-latest"
        assert call_kwargs["temperature"] == 0.7

    def test_overrides_model_and_temperature(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse("ok")

        llm = LLMClient(client=mock_client)
        llm.complete("x", model="mistral-small", temperature=0.0)

        call_kwargs = mock_client.chat.complete.call_args.kwargs
        assert call_kwargs["model"] == "mistral-small"
        assert call_kwargs["temperature"] == 0.0

    def test_retries_on_connection_error(self) -> None:
        mock_client = MagicMock()
        # First two calls raise, third succeeds
        mock_client.chat.complete.side_effect = [
            ConnectionError("network glitch"),
            ConnectionError("still bad"),
            FakeResponse("recovered"),
        ]

        llm = LLMClient(client=mock_client)
        result = llm.complete("x")

        assert result == "recovered"
        assert mock_client.chat.complete.call_count == 3

    def test_does_not_retry_on_value_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.side_effect = ValueError("bad request")

        llm = LLMClient(client=mock_client)
        with pytest.raises(ValueError, match="bad request"):
            llm.complete("x")

        # ValueError isn't retryable — should be called only once
        assert mock_client.chat.complete.call_count == 1

    def test_raises_on_empty_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse(None)  # type: ignore[arg-type]

        llm = LLMClient(client=mock_client)
        with pytest.raises(ValueError, match="empty"):
            llm.complete("x")


class TestCompleteStructured:
    def test_parses_valid_json(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse('{"name": "Alice", "age": 30}')

        llm = LLMClient(client=mock_client)
        result = llm.complete_structured("Who?", _OutputSchema)

        assert isinstance(result, _OutputSchema)
        assert result.name == "Alice"
        assert result.age == 30

    def test_includes_schema_in_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse('{"name": "X", "age": 1}')

        llm = LLMClient(client=mock_client)
        llm.complete_structured("Original prompt", _OutputSchema)

        sent_prompt = mock_client.chat.complete.call_args.kwargs["messages"][0]["content"]
        assert "Original prompt" in sent_prompt
        assert "JSON" in sent_prompt
        assert '"name"' in sent_prompt  # the schema mentions the field

    def test_uses_json_response_format(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse('{"name": "X", "age": 1}')

        llm = LLMClient(client=mock_client)
        llm.complete_structured("x", _OutputSchema)

        kwargs = mock_client.chat.complete.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_raises_on_malformed_json(self) -> None:
        from pydantic import ValidationError

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse("not actually JSON")

        llm = LLMClient(client=mock_client)
        with pytest.raises(ValidationError):
            llm.complete_structured("x", _OutputSchema)

    def test_raises_on_schema_mismatch(self) -> None:
        from pydantic import ValidationError

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = FakeResponse(
            '{"name": "X"}'  # missing required `age`
        )

        llm = LLMClient(client=mock_client)
        with pytest.raises(ValidationError):
            llm.complete_structured("x", _OutputSchema)
