"""Final-feedback generation at the end of an interview."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from multiagent_interviewer.agents._helpers import format_messages_for_prompt
from multiagent_interviewer.graph.state import FinalFeedback, InterviewState

if TYPE_CHECKING:
    from multiagent_interviewer.llm.client import LLMClient


def generate_final_feedback(
    state: InterviewState,
    llm: LLMClient,
) -> FinalFeedback:
    """Synthesize a structured final feedback report for the candidate.

    Uses the entire interview log + agent thoughts to produce a verdict.
    """
    # Aggregate expert analyses across all turns
    expert_thoughts = [
        log.expert_analysis.model_dump_json(indent=2)
        for log in state.log
        if log.expert_analysis is not None
    ]
    manager_thoughts = [
        log.manager_decision.model_dump_json(indent=2)
        for log in state.log
        if log.manager_decision is not None
    ]

    transcript = format_messages_for_prompt(state.messages)

    prompt = f"""\
Generate structured final feedback for a technical interview candidate.

CANDIDATE INFORMATION:
- Name: {state.candidate.name}
- Position: {state.candidate.position}
- Level: {state.candidate.grade.value}
- Experience: {state.candidate.experience}

INTERVIEW TRANSCRIPT:
{transcript}

EXPERT ANALYSES (per-turn):
{chr(10).join(expert_thoughts) if expert_thoughts else "(none)"}

MANAGER DECISIONS (per-turn):
{chr(10).join(manager_thoughts) if manager_thoughts else "(none)"}

NUMBER OF QUESTIONS ASKED: {state.current_turn - 1}

Produce structured feedback with these sections:
- Verdict (grade assessment, hiring recommendation, confidence 0-100)
- Hard skills (confirmed skills, knowledge gaps)
- Soft skills (clarity, honesty, engagement)
- Roadmap (learning topics, suggested resources)

Write in Russian, be specific and constructive.
"""

    feedback = llm.complete_structured(prompt, FinalFeedback)
    logger.info(
        "Final feedback: grade={}, recommendation={}, confidence={}",
        feedback.grade_assessment.value,
        feedback.hiring_recommendation.value,
        feedback.confidence_score,
    )
    return feedback


def save_feedback_report(
    state: InterviewState,
    feedback: FinalFeedback,
    output_dir: Path,
) -> Path:
    """Save the full report (state + feedback) to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in state.candidate.name)
    output_path = output_dir / f"interview_{safe_name}_{timestamp}.json"

    report = {
        "candidate": state.candidate.model_dump(mode="json"),
        "turns": [log.model_dump(mode="json") for log in state.log],
        "final_feedback": feedback.model_dump(mode="json"),
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved interview report to {}", output_path)
    return output_path
