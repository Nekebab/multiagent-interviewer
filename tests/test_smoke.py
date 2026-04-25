"""Smoke tests: verify that imports work and basic config loads correctly."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def test_package_imports() -> None:
    import multiagent_interviewer

    assert multiagent_interviewer.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    import multiagent_interviewer.agents
    import multiagent_interviewer.graph
    import multiagent_interviewer.llm
    import multiagent_interviewer.rag
    import multiagent_interviewer.tools  # noqa: F401


def test_settings_load_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-123")

    from multiagent_interviewer.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    assert settings.mistral_api_key == "test-key-123"
    assert settings.llm_model == "mistral-large-latest"
    assert settings.chunk_size == 500


def test_settings_validation_rejects_bad_values(monkeypatch: MonkeyPatch) -> None:
    """Pydantic catches invalid values at load time."""
    import pytest
    from pydantic import ValidationError

    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "999")  # out of [0, 2] range

    from multiagent_interviewer.config import Settings

    with pytest.raises(ValidationError):
        Settings()  # type: ignore[call-arg]


def test_settings_paths(monkeypatch: MonkeyPatch) -> None:
    """Computed properties combine `data_dir` with CSV filenames."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    from multiagent_interviewer.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    assert str(settings.expert_csv_path) == "data/expert_knowledge.csv"
    assert str(settings.manager_csv_path) == "data/manager_knowledge.csv"
