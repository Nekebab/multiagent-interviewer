"""Expert agent: analyzes the candidate's last answer for technical correctness."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from loguru import logger

from multiagent_interviewer.graph.state import (
    ExpertAnalysis,
    InterviewState,
)
from multiagent_interviewer.prompts import render

if TYPE_CHECKING:
    from multiagent_interviewer.llm.client import LLMClient
    from multiagent_interviewer.rag.system import RagSystem


def make_expert_node(
    llm: LLMClient,
    rag: RagSystem | None = None,
) -> Callable[[InterviewState], dict[str, ExpertAnalysis]]:
    """Build the expert node function bound to a specific LLM and (optional) RAG."""

    def expert_node(state: InterviewState) -> dict[str, ExpertAnalysis]:
        last_answer = state.last_candidate_message

        if not last_answer:
            logger.debug("Expert: no candidate message yet, skipping")
            return {}

        # Pull RAG context (if available) using the candidate's answer as query.
        rag_context = ""
        rag_snippets_count = 0
        if rag is not None and rag.expert is not None:
            results = rag.search_expert(last_answer)
            if results:
                rag_context = "\n".join(f"- {r}" for r in results)
                rag_snippets_count = len(results)
                logger.info(
                    "Expert: retrieved {} snippets from knowledge base",
                    rag_snippets_count,
                )
                print(f"RAG: retrieved {rag_snippets_count} relevant snippets from expert KB")
                preview = results[0][:100].replace("\n", " ")
                print(f'     → "{preview}..."')

        prompt = render(
            "expert.j2",
            candidate=state.candidate,
            recent_messages=state.recent_messages(count=4),
            last_answer=last_answer,
            rag_context=rag_context,
            internet_context="",  # internet search not wired up yet
        )

        analysis = llm.complete_structured(prompt, ExpertAnalysis)
        logger.info("Expert produced analysis for turn {}", state.current_turn)
        return {"expert_analysis": analysis}

    return expert_node
