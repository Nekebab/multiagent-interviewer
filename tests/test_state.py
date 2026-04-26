"""Tests for the interview state domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from multiagent_interviewer.graph.state import (
    CandidateInfo,
    Grade,
    InterviewState,
    Message,
    Role,
    TurnLog,
)


class TestRole:
    def test_string_value(self) -> None:
        assert Role.USER.value == "user"
        assert Role.ASSISTANT == "assistant"  # StrEnum equals its string value

    def test_serializes_as_string(self) -> None:
        msg = Message(role=Role.USER, content="hi")
        assert msg.to_api_format() == {"role": "user", "content": "hi"}


class TestCandidateInfo:
    def test_construction(self) -> None:
        c = CandidateInfo(
            name="Alex",
            position="Backend Developer",
            grade=Grade.JUNIOR,
            experience="2 years of Python",
        )
        assert c.name == "Alex"
        assert c.grade is Grade.JUNIOR

    def test_grade_accepts_string(self) -> None:
        # Pydantic should coerce the enum from a string value
        c = CandidateInfo(
            name="Alex",
            position="Dev",
            grade="Middle",  # type: ignore[arg-type]
            experience="some",
        )
        assert c.grade is Grade.MIDDLE

    def test_rejects_invalid_grade(self) -> None:
        with pytest.raises(ValidationError):
            CandidateInfo(
                name="Alex",
                position="Dev",
                grade="Wizard",  # type: ignore[arg-type]
                experience="some",
            )

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError):
            CandidateInfo(
                name="",
                position="Dev",
                grade=Grade.JUNIOR,
                experience="some",
            )

    def test_frozen(self) -> None:
        c = CandidateInfo(
            name="Alex",
            position="Dev",
            grade=Grade.JUNIOR,
            experience="some",
        )
        with pytest.raises(ValidationError):
            c.name = "Bob"


class TestInterviewState:
    @pytest.fixture
    def candidate(self) -> CandidateInfo:
        return CandidateInfo(
            name="Alex",
            position="Backend Dev",
            grade=Grade.JUNIOR,
            experience="2y Python",
        )

    def test_initial_creates_system_message(self, candidate: CandidateInfo) -> None:
        state = InterviewState.initial(candidate)
        assert len(state.messages) == 1
        assert state.messages[0].role is Role.SYSTEM
        assert state.current_turn == 1
        assert state.is_active is True

    def test_last_candidate_message_when_no_user_yet(self, candidate: CandidateInfo) -> None:
        state = InterviewState.initial(candidate)
        assert state.last_candidate_message is None

    def test_last_candidate_message_finds_most_recent(self, candidate: CandidateInfo) -> None:
        state = InterviewState.initial(candidate)
        state.add_message(Role.USER, "first answer")
        state.add_message(Role.ASSISTANT, "next question")
        state.add_message(Role.USER, "second answer")

        assert state.last_candidate_message == "second answer"

    def test_recent_messages_returns_tail(self, candidate: CandidateInfo) -> None:
        state = InterviewState.initial(candidate)
        for i in range(5):
            state.add_message(Role.USER, f"msg{i}")

        recent = state.recent_messages(count=3)
        assert len(recent) == 3
        assert [m.content for m in recent] == ["msg2", "msg3", "msg4"]

    def test_recent_messages_handles_short_history(self, candidate: CandidateInfo) -> None:
        state = InterviewState.initial(candidate)
        # Only 1 system message exists
        recent = state.recent_messages(count=10)
        assert len(recent) == 1


class TestTurnLog:
    def test_minimal_construction(self) -> None:
        log = TurnLog(turn_id=1, interviewer_message="What is Python?")
        assert log.turn_id == 1
        assert log.candidate_message is None  # first turn

    def test_rejects_zero_turn_id(self) -> None:
        with pytest.raises(ValidationError):
            TurnLog(turn_id=0, interviewer_message="x")
