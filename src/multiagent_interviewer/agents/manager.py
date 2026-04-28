"""Manager agent: strategic decisions about interview flow."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from multiagent_interviewer.config import get_settings
from multiagent_interviewer.graph.state import InterviewState, ManagerDecision
from multiagent_interviewer.prompts import render

if TYPE_CHECKING:
    from multiagent_interviewer.llm.client import LLMClient
    from multiagent_interviewer.rag.system import RagSystem


def make_manager_node(
    llm: LLMClient,
    rag: RagSystem | None = None,
) -> Callable[[InterviewState], dict[str, Any]]:
    """Build the manager node function."""
    settings = get_settings()

    def manager_node(state: InterviewState) -> dict[str, Any]:
        # Pull RAG context for hiring/management knowledge if available
        rag_context = ""
        if rag is not None and rag.manager is not None:
            query = f"{state.candidate.position} {state.candidate.grade.value} hiring"
            results = rag.search_manager(query)
            if results:
                rag_context = "\n".join(f"- {r}" for r in results)
                logger.info(
                    "Manager: retrieved {} snippets from HR knowledge base",
                    len(results),
                )
                print(f" RAG: retrieved {len(results)} relevant snippets from manager KB")
                preview = results[0][:100].replace("\n", " ")
                print(f'     → "{preview}..."')

        # Format the expert's analysis as text for the prompt
        expert_text = (
            state.expert_analysis.model_dump_json(indent=2)
            if state.expert_analysis is not None
            else "(no analysis available yet — first turn)"
        )

        prompt = render(
            "manager.j2",
            candidate=state.candidate,
            recent_messages=state.recent_messages(count=6),
            current_turn=state.current_turn,
            max_turns=settings.max_turns,
            min_turns_before_end=settings.min_turns_before_end,
            expert_analysis_text=expert_text,
            rag_context=rag_context,
        )

        decision = llm.complete_structured(prompt, ManagerDecision)
        if state.current_turn == 1:
            decision = decision.model_copy(update={"soft_skills_score": 0})

        logger.info(
            "Manager turn {}: should_end={}, score={}",
            state.current_turn,
            decision.should_end_interview,
            decision.soft_skills_score,
        )

        should_end = decision.should_end_interview
        if state.current_turn < settings.min_turns_before_end:
            should_end = False
        if state.current_turn >= settings.max_turns:
            should_end = True

        return {
            "manager_decision": decision,
            "is_active": not should_end,
        }

    return manager_node
