"""Tests for final-feedback generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multiagent_interviewer.feedback import (
    generate_final_feedback,
    save_feedback_report,
)
from multiagent_interviewer.graph.state import (
    CandidateInfo,
    FinalFeedback,
    Grade,
    HiringRecommendation,
    InterviewState,
    Role,
)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")


@pytest.fixture
def sample_state() -> InterviewState:
    candidate = CandidateInfo(
        name="Alex Kirov",
        position="Backend Dev",
        grade=Grade.JUNIOR,
        experience="2 years",
    )
    state = InterviewState.initial(candidate)
    state.add_message(Role.ASSISTANT, "What's a decorator?")
    state.add_message(Role.USER, "A wrapper function")
    state.current_turn = 3
    return state


@pytest.fixture
def sample_feedback() -> FinalFeedback:
    return FinalFeedback(
        grade_assessment=Grade.JUNIOR,
        hiring_recommendation=HiringRecommendation.HIRE,
        confidence_score=72,
        confirmed_skills=["Python basics"],
        knowledge_gaps=["closures"],
        soft_skills_summary="Clear communicator",
        learning_roadmap=["Dive into closures"],
        suggested_resources=["Real Python article on decorators"],
    )


class TestGenerateFinalFeedback:
    def test_calls_llm_with_correct_schema(self, sample_state: InterviewState) -> None:
        llm = MagicMock()
        llm.complete_structured.return_value = FinalFeedback(
            grade_assessment=Grade.JUNIOR,
            hiring_recommendation=HiringRecommendation.HIRE,
            confidence_score=80,
            soft_skills_summary="ok",
        )

        result = generate_final_feedback(sample_state, llm)

        llm.complete_structured.assert_called_once()
        assert llm.complete_structured.call_args.args[1] is FinalFeedback
        assert isinstance(result, FinalFeedback)


class TestSaveFeedbackReport:
    def test_writes_json_file(
        self,
        sample_state: InterviewState,
        sample_feedback: FinalFeedback,
        tmp_path: Path,
    ) -> None:
        output_path = save_feedback_report(sample_state, sample_feedback, tmp_path)

        assert output_path.exists()
        assert output_path.suffix == ".json"

        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "candidate" in content
        assert "turns" in content
        assert "final_feedback" in content
        assert content["final_feedback"]["confidence_score"] == 72

    def test_sanitizes_filename(
        self,
        sample_state: InterviewState,
        sample_feedback: FinalFeedback,
        tmp_path: Path,
    ) -> None:
        output_path = save_feedback_report(sample_state, sample_feedback, tmp_path)

        assert " " not in output_path.name

    def test_creates_output_dir_if_missing(
        self,
        sample_state: InterviewState,
        sample_feedback: FinalFeedback,
        tmp_path: Path,
    ) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()

        output_path = save_feedback_report(sample_state, sample_feedback, nested)

        assert nested.exists()
        assert output_path.exists()
