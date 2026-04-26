"""Tests for Jinja2 prompt rendering."""

from __future__ import annotations

import pytest
from jinja2.exceptions import UndefinedError

from multiagent_interviewer.graph.state import (
    CandidateInfo,
    Grade,
    Message,
    Role,
)
from multiagent_interviewer.prompts import render


@pytest.fixture
def candidate() -> CandidateInfo:
    return CandidateInfo(
        name="Alex",
        position="Backend Developer",
        grade=Grade.JUNIOR,
        experience="2y Python",
    )


@pytest.fixture
def messages() -> list[Message]:
    return [
        Message(role=Role.ASSISTANT, content="What's a decorator?"),
        Message(role=Role.USER, content="A function that takes a function..."),
    ]


class TestExpertPrompt:
    def test_renders_with_minimum_context(
        self, candidate: CandidateInfo, messages: list[Message]
    ) -> None:
        rendered = render(
            "expert.j2",
            candidate=candidate,
            recent_messages=messages,
            last_answer="A function that takes a function...",
            rag_context="",
            internet_context="",
        )
        assert "Alex" not in rendered  # name is not in expert prompt
        assert "Backend Developer" in rendered
        assert "Junior" in rendered
        assert "A function that takes a function" in rendered

    def test_includes_rag_context_when_provided(
        self, candidate: CandidateInfo, messages: list[Message]
    ) -> None:
        rendered = render(
            "expert.j2",
            candidate=candidate,
            recent_messages=messages,
            last_answer="answer",
            rag_context="DECORATORS ARE FUNCTIONS",
            internet_context="",
        )
        assert "DECORATORS ARE FUNCTIONS" in rendered

    def test_omits_rag_section_when_empty(
        self, candidate: CandidateInfo, messages: list[Message]
    ) -> None:
        rendered = render(
            "expert.j2",
            candidate=candidate,
            recent_messages=messages,
            last_answer="answer",
            rag_context="",
            internet_context="",
        )
        assert "KNOWLEDGE BASE" not in rendered  # the header is gone

    def test_strict_undefined_catches_typos(self, candidate: CandidateInfo) -> None:
        # If the template references `last_answer` and we forget to pass it,
        # we get a clean error instead of silent corruption.
        with pytest.raises(UndefinedError):
            render(
                "expert.j2",
                candidate=candidate,
                recent_messages=[],
                # forgot: last_answer
                rag_context="",
                internet_context="",
            )


class TestManagerPrompt:
    def test_renders(self, candidate: CandidateInfo, messages: list[Message]) -> None:
        rendered = render(
            "manager.j2",
            candidate=candidate,
            recent_messages=messages,
            current_turn=3,
            max_turns=10,
            min_turns_before_end=8,
            expert_analysis_text="The candidate knows decorators well.",
            rag_context="",
        )
        assert "Alex" in rendered
        assert "3 of 10" in rendered
        assert "knows decorators well" in rendered


class TestInterviewerPrompt:
    def test_renders_with_last_answer(
        self, candidate: CandidateInfo, messages: list[Message]
    ) -> None:
        rendered = render(
            "interviewer.j2",
            candidate=candidate,
            recent_messages=messages,
            last_answer="some answer",
            expert_recommendations="ask about generators",
            manager_direction="go deeper",
        )
        assert "some answer" in rendered
        assert "ask about generators" in rendered
        assert "START OF THE INTERVIEW" not in rendered

    def test_renders_at_interview_start(self, candidate: CandidateInfo) -> None:
        rendered = render(
            "interviewer.j2",
            candidate=candidate,
            recent_messages=[],
            last_answer=None,
            expert_recommendations="N/A",
            manager_direction="start with basics",
        )
        assert "START OF THE INTERVIEW" in rendered
