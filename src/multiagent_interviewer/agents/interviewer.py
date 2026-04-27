"""Interviewer agent: speaks to the candidate, asks questions."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from multiagent_interviewer.graph.state import (
    InterviewState,
    Message,
    Role,
    TurnLog,
)
from multiagent_interviewer.prompts import render

if TYPE_CHECKING:
    from multiagent_interviewer.llm.client import LLMClient


def _strip_json_wrapper(text: str) -> str:
    """If the LLM returned JSON despite our instructions, extract the message text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()

    if not (text.startswith("{") and text.endswith("}")):
        return text

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if not isinstance(parsed, dict):
        return text

    for key in ("question", "message", "content", "text", "response"):
        value = parsed.get(key)
        if isinstance(value, str) and value:
            return value

    return text


def make_interviewer_node(llm: LLMClient) -> Callable[[InterviewState], dict[str, Any]]:
    """Build the interviewer node function."""

    def interviewer_node(state: InterviewState) -> dict[str, Any]:
        last_answer = state.last_candidate_message

        expert_recommendations = (
            "\n".join(f"- {r}" for r in state.expert_analysis.recommended_follow_ups)
            if state.expert_analysis and state.expert_analysis.recommended_follow_ups
            else "Start with a foundational question matching the candidate level."
        )

        manager_direction = (
            state.manager_decision.direction
            if state.manager_decision is not None
            else "Begin the interview with a warm-up question."
        )

        prompt = render(
            "interviewer.j2",
            candidate=state.candidate,
            recent_messages=state.recent_messages(count=4),
            last_answer=last_answer,
            expert_recommendations=expert_recommendations,
            manager_direction=manager_direction,
        )

        raw_response = llm.complete(prompt).strip()
        question = _strip_json_wrapper(raw_response)
        logger.info("Interviewer turn {}: asks question", state.current_turn)

        assistant_message = Message(role=Role.ASSISTANT, content=question)
        turn_log = TurnLog(
            turn_id=state.current_turn,
            candidate_message=last_answer,
            interviewer_message=question,
            expert_analysis=state.expert_analysis,
            manager_decision=state.manager_decision,
        )

        return {
            "messages": [*state.messages, assistant_message],
            "log": [*state.log, turn_log],
            "current_turn": state.current_turn + 1,
        }

    return interviewer_node
