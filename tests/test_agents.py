"""Tests for the three agent nodes (Expert, Manager, Interviewer).

These tests use a fake LLM to verify the agents wire up correctly,
without making real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from multiagent_interviewer.agents import (
    make_expert_node,
    make_interviewer_node,
    make_manager_node,
)
from multiagent_interviewer.graph.state import (
    CandidateInfo,
    ExpertAnalysis,
    Grade,
    InterviewState,
    ManagerDecision,
    Role,
)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key")


@pytest.fixture
def candidate() -> CandidateInfo:
    return CandidateInfo(
        name="Alex",
        position="Backend Developer",
        grade=Grade.JUNIOR,
        experience="2 years of Python",
    )


@pytest.fixture
def state_with_one_answer(candidate: CandidateInfo) -> InterviewState:
    """A state where the interviewer has asked and the candidate replied."""
    state = InterviewState.initial(candidate)
    state.add_message(Role.ASSISTANT, "What is a decorator in Python?")
    state.add_message(Role.USER, "A function that wraps another function")
    return state


# Expert


class TestExpertNode:
    def test_returns_empty_when_no_candidate_message(self, candidate: CandidateInfo) -> None:
        """On the very first turn (no answer yet), expert should noop."""
        state = InterviewState.initial(candidate)  # only system msg
        llm = MagicMock()

        node = make_expert_node(llm)
        result = node(state)

        assert result == {}
        llm.complete_structured.assert_not_called()

    def test_calls_llm_with_structured_output(self, state_with_one_answer: InterviewState) -> None:
        llm = MagicMock()
        llm.complete_structured.return_value = ExpertAnalysis(
            technical_correctness="ok",
            knowledge_gaps=[],
            recommended_follow_ups=["ask about closures"],
            difficulty_adjustment="same",
        )

        node = make_expert_node(llm)
        result = node(state_with_one_answer)

        assert "expert_analysis" in result
        analysis = result["expert_analysis"]
        assert isinstance(analysis, ExpertAnalysis)
        assert analysis.recommended_follow_ups == ["ask about closures"]

        # Verify the LLM was called with the right schema
        call_args = llm.complete_structured.call_args
        assert call_args.args[1] is ExpertAnalysis

    def test_includes_candidate_answer_in_prompt(
        self, state_with_one_answer: InterviewState
    ) -> None:
        llm = MagicMock()
        llm.complete_structured.return_value = ExpertAnalysis(
            technical_correctness="ok",
        )

        node = make_expert_node(llm)
        node(state_with_one_answer)

        prompt = llm.complete_structured.call_args.args[0]
        assert "function that wraps another function" in prompt


# Manager


class TestManagerNode:
    def test_returns_decision_and_active_flag(self, state_with_one_answer: InterviewState) -> None:
        llm = MagicMock()
        llm.complete_structured.return_value = ManagerDecision(
            progress_assessment="going well",
            soft_skills_score=7,
            direction="continue with the same topic",
            should_end_interview=False,
        )

        node = make_manager_node(llm)
        result = node(state_with_one_answer)

        assert "manager_decision" in result
        assert "is_active" in result
        assert result["is_active"] is True

    def test_does_not_end_before_minimum_turns(self, state_with_one_answer: InterviewState) -> None:
        """Even if LLM says 'end', policy enforces a minimum number of turns."""
        llm = MagicMock()
        llm.complete_structured.return_value = ManagerDecision(
            progress_assessment="enough",
            soft_skills_score=5,
            direction="end now",
            should_end_interview=True,  # LLM wants to end
            end_reason="enough signal",
        )

        # current_turn=1 < min_turns_before_end (8)
        node = make_manager_node(llm)
        result = node(state_with_one_answer)

        # still active
        assert result["is_active"] is True

    def test_force_ends_at_max_turns(self, state_with_one_answer: InterviewState) -> None:
        """Even if LLM wants to continue, hitting max_turns ends the interview."""
        llm = MagicMock()
        llm.complete_structured.return_value = ManagerDecision(
            progress_assessment="going strong",
            soft_skills_score=9,
            direction="keep going",
            should_end_interview=False,  # LLM wants to continue
        )

        state_with_one_answer.current_turn = 10

        node = make_manager_node(llm)
        result = node(state_with_one_answer)

        assert result["is_active"] is False


# Interviewer


class TestInterviewerNode:
    def test_appends_assistant_message(self, state_with_one_answer: InterviewState) -> None:
        llm = MagicMock()
        llm.complete.return_value = "What is a closure?"

        node = make_interviewer_node(llm)
        result = node(state_with_one_answer)

        # The new messages list should contain the previous + a new assistant msg
        new_messages = result["messages"]
        assert len(new_messages) == len(state_with_one_answer.messages) + 1
        assert new_messages[-1].role is Role.ASSISTANT
        assert new_messages[-1].content == "What is a closure?"

    def test_increments_turn(self, state_with_one_answer: InterviewState) -> None:
        llm = MagicMock()
        llm.complete.return_value = "next question?"

        node = make_interviewer_node(llm)
        result = node(state_with_one_answer)

        assert result["current_turn"] == state_with_one_answer.current_turn + 1

    def test_appends_log_entry(self, state_with_one_answer: InterviewState) -> None:
        llm = MagicMock()
        llm.complete.return_value = "next?"

        node = make_interviewer_node(llm)
        result = node(state_with_one_answer)

        new_log = result["log"]
        assert len(new_log) == len(state_with_one_answer.log) + 1
        latest = new_log[-1]
        assert latest.interviewer_message == "next?"
        assert latest.candidate_message == "A function that wraps another function"

    def test_uses_complete_not_structured(self, state_with_one_answer: InterviewState) -> None:
        """Interviewer asks freeform questions — no structured output needed."""
        llm = MagicMock()
        llm.complete.return_value = "ok"

        node = make_interviewer_node(llm)
        node(state_with_one_answer)

        llm.complete.assert_called_once()
        llm.complete_structured.assert_not_called()
