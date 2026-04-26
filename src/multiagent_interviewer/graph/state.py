"""Domain models for the interview state graph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """A speaker role in the message log."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Grade(StrEnum):
    """Candidate seniority level."""

    JUNIOR = "Junior"
    MIDDLE = "Middle"
    SENIOR = "Senior"


class HiringRecommendation(StrEnum):
    """Final hiring verdict produced by the manager."""

    STRONG_HIRE = "Strong Hire"
    HIRE = "Hire"
    BORDERLINE = "Borderline"
    NO_HIRE = "No Hire"


class CandidateInfo(BaseModel):
    """Identifying information about the candidate being interviewed."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="Candidate's name")
    position: str = Field(..., min_length=1, description="Target position")
    grade: Grade = Field(..., description="Target seniority")
    experience: str = Field(..., min_length=1, description="Free-form experience description")


class Message(BaseModel):
    """A single message in the conversation history."""

    role: Role
    content: str

    def to_api_format(self) -> dict[str, str]:
        """Serialize to the dict shape expected by Mistral/OpenAI APIs."""
        return {"role": self.role.value, "content": self.content}


# Structured outputs — what each agent's LLM call returns.


class ExpertAnalysis(BaseModel):
    """The expert agent's structured analysis of a candidate's last answer."""

    technical_correctness: str = Field(..., description="Assessment of factual accuracy")
    knowledge_gaps: list[str] = Field(
        default_factory=list, description="Specific gaps or misconceptions"
    )
    recommended_follow_ups: list[str] = Field(
        default_factory=list,
        description="Suggested next questions for the interviewer",
    )
    difficulty_adjustment: Literal["easier", "same", "harder"] = Field(
        "same", description="Whether to adjust question difficulty"
    )


class ManagerDecision(BaseModel):
    """The manager agent's strategic decision after each turn."""

    progress_assessment: str = Field(..., description="Overall view of how the interview is going")
    soft_skills_score: int = Field(
        ..., ge=0, le=10, description="Communication, honesty, engagement (0-10)"
    )
    direction: str = Field(..., description="What topic/depth to pursue next")
    should_end_interview: bool = Field(False, description="Whether to terminate the interview now")
    end_reason: str | None = Field(
        None, description="If ending, why (e.g. 'enough signal to decide')"
    )


class FinalFeedback(BaseModel):
    """The final assessment generated at the end of the interview."""

    grade_assessment: Grade = Field(..., description="Verdict on actual level")
    hiring_recommendation: HiringRecommendation
    confidence_score: int = Field(..., ge=0, le=100)
    confirmed_skills: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)
    soft_skills_summary: str
    learning_roadmap: list[str] = Field(default_factory=list)
    suggested_resources: list[str] = Field(default_factory=list)


# a structured record of every turn, kept for the final report.


class TurnLog(BaseModel):
    """A complete record of one turn: what was said, what each agent thought."""

    turn_id: int = Field(..., ge=1)
    interviewer_message: str = Field(..., description="What the interviewer asked")
    timestamp: datetime = Field(default_factory=datetime.now)
    candidate_message: str | None = Field(
        default=None,
        description="What the candidate said",
    )
    expert_analysis: ExpertAnalysis | None = None
    manager_decision: ManagerDecision | None = None


class InterviewState(BaseModel):
    """The mutable state that flows through the LangGraph graph."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    candidate: CandidateInfo
    messages: list[Message] = Field(default_factory=list)
    log: list[TurnLog] = Field(default_factory=list)

    expert_analysis: ExpertAnalysis | None = None
    manager_decision: ManagerDecision | None = None

    current_turn: int = Field(1, ge=1)
    is_active: bool = True

    @property
    def last_candidate_message(self) -> str | None:
        """Return the most recent USER message text, or None if there isn't one."""
        for msg in reversed(self.messages):
            if msg.role is Role.USER:
                return msg.content
        return None

    def recent_messages(self, count: int = 3) -> list[Message]:
        """Return the last `count` messages (or fewer, if history is shorter)."""
        return self.messages[-count:]

    def add_message(self, role: Role, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    @classmethod
    def initial(cls, candidate: CandidateInfo) -> InterviewState:
        """Build the initial state at the start of an interview."""
        system_prompt = (
            f"You are conducting a technical interview.\n"
            f"Position: {candidate.position}\n"
            f"Level: {candidate.grade.value}\n"
            f"Experience: {candidate.experience}\n"
            f"Goal: assess the candidate fairly and produce useful feedback."
        )
        return cls(
            candidate=candidate,
            current_turn=1,
            messages=[Message(role=Role.SYSTEM, content=system_prompt)],
        )
