"""Final-feedback generation at the end of an interview."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from multiagent_interviewer.agents._helpers import format_messages_for_prompt
from multiagent_interviewer.graph.state import FinalFeedback, HiringRecommendation, InterviewState

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
    answered_turns = sum(1 for log in state.log if log.candidate_message)
    flagged_gaps = sum(
        len(log.expert_analysis.knowledge_gaps)
        for log in state.log
        if log.expert_analysis is not None
    )
    fact_errors = sum(
        1
        for log in state.log
        if log.expert_analysis is not None
        and (
            "ошибк" in log.expert_analysis.technical_correctness.lower()
            or "некорректн" in log.expert_analysis.technical_correctness.lower()
        )
    )

    prompt = f"""\
        You are a calibrated technical hiring assessor. Generate a structured final
        feedback report. Be honest, not encouraging — this is a high-stakes hiring
        decision, NOT a coaching conversation.

        CANDIDATE INFORMATION:
        - Name: {state.candidate.name}
        - Position: {state.candidate.position}
        - Stated level: {state.candidate.grade.value}
        - Experience: {state.candidate.experience}

        INTERVIEW STATISTICS:
        - Total turns conducted: {state.current_turn - 1}
        - Turns with candidate answers: {answered_turns}
        - Knowledge gaps flagged by expert (cumulative): {flagged_gaps}
        - Turns with fact errors flagged by expert: {fact_errors}

        INTERVIEW TRANSCRIPT:
        {transcript}

        EXPERT ANALYSES (per-turn — these are PROFESSIONAL assessments, do not soften):
        {chr(10).join(expert_thoughts) if expert_thoughts else "(none)"}

        MANAGER DECISIONS (per-turn):
        {chr(10).join(manager_thoughts) if manager_thoughts else "(none)"}

        ────────────────────────────────────────────────────────────────────
        CALIBRATION RULES — FOLLOW STRICTLY:

        confidence_score (0-100) — how SURE you are in your verdict:
        - 90-100: only when 8+ turns and answers are consistently strong/weak
        - 70-89: when 6+ turns and pattern is clear
        - 50-69: when 4-6 turns and signal is mixed or partial
        - 30-49: when 2-4 turns or contradictory evidence
        - 0-29: when fewer than 2 substantive answers — too little data

        hiring_recommendation:
        - Strong Hire: candidate exceeds the stated level with consistent depth
            AND no significant knowledge gaps in fundamentals
        - Hire: candidate matches the stated level, gaps are minor
        - Borderline: signals are mixed; gaps in fundamentals OR inconsistent depth
        - No Hire: candidate falls clearly below the stated level OR has critical
            gaps in fundamentals OR demonstrates dishonesty

        KEY HEURISTICS:
        - If the candidate said "I don't know" or "не помню" on basic questions for
        their stated level → cannot be Strong Hire, likely Borderline or below.
        - If the expert flagged factual errors in basics → recommendation cannot be
        above Borderline.
        - If interview was cut short ({state.current_turn - 1} turns), confidence
        should be lower; do not pretend you have full picture.
        - If candidate's experience claim seems inflated relative to demonstrated
        knowledge → flag in soft_skills_summary, lower confidence.
        - Don't pad confirmed_skills with generic items — only list skills the
        candidate actually demonstrated in answers (with concrete evidence).

        DISQUALIFYING SIGNALS (any single one of these → recommendation cannot be
        above No Hire, regardless of how strong other answers were):
        - Off-topic personal anecdotes inserted into a technical answer
        (e.g. "I baked bread last week" mid-explanation)
        - Avoiding a direct question with a deflection ("I missed it, repeat please"
        after going off on a tangent)
        - Confused fundamental concepts (mistaking variance for mean, etc.)
        after multiple chances to clarify
        - Dishonest or evasive behavior

        A candidate can have strong technical knowledge AND still be No Hire if
        they cannot maintain professional focus or honesty during the interview.
        The recommendation reflects HIRABILITY, not just technical knowledge.

        Write in Russian, be specific and constructive but realistic.
        """

    feedback = llm.complete_structured(prompt, FinalFeedback)
    answered_turns = sum(1 for log in state.log if log.candidate_message)
    if answered_turns < 3:
        feedback = feedback.model_copy(
            update={"confidence_score": min(feedback.confidence_score, 40)}
        )
    elif answered_turns < 5:
        feedback = feedback.model_copy(
            update={"confidence_score": min(feedback.confidence_score, 60)}
        )
    elif answered_turns < 7:
        feedback = feedback.model_copy(
            update={"confidence_score": min(feedback.confidence_score, 80)}
        )

    if answered_turns < 3 and feedback.hiring_recommendation == HiringRecommendation.STRONG_HIRE:
        feedback = feedback.model_copy(update={"hiring_recommendation": HiringRecommendation.HIRE})
        logger.warning(
            "Strong Hire downgraded to Hire — only {} answered turns",
            answered_turns,
        )
    if answered_turns < 2:
        feedback = feedback.model_copy(
            update={"hiring_recommendation": HiringRecommendation.BORDERLINE}
        )
        logger.warning(
            "Recommendation downgraded to Borderline — only {} answered turns",
            answered_turns,
        )

    logger.info(
        "Final feedback: grade={}, recommendation={}, confidence={} (after policy)",
        feedback.grade_assessment.value,
        feedback.hiring_recommendation.value,
        feedback.confidence_score,
    )
    if feedback.behavioral_red_flags and feedback.hiring_recommendation in (
        HiringRecommendation.STRONG_HIRE,
        HiringRecommendation.HIRE,
    ):
        logger.warning(
            "Behavioral red flags present; downgrading recommendation. Flags: {}",
            feedback.behavioral_red_flags,
        )
        feedback = feedback.model_copy(
            update={"hiring_recommendation": HiringRecommendation.BORDERLINE}
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
