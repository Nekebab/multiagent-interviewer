"""LangGraph assembly: wire the three agents into a cyclic state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph
from loguru import logger

from multiagent_interviewer.agents import (
    make_expert_node,
    make_interviewer_node,
    make_manager_node,
)
from multiagent_interviewer.graph.state import InterviewState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from multiagent_interviewer.llm.client import LLMClient
    from multiagent_interviewer.rag.system import RagSystem


def build_interview_graph(
    llm: LLMClient,
    rag: RagSystem | None = None,
) -> CompiledStateGraph:
    """Assemble and compile the interview graph.

    Flow per turn:
        expert → manager → interviewer → END

    The graph runs once per candidate response.
    """
    graph = StateGraph(InterviewState)

    graph.add_node("expert", make_expert_node(llm, rag))
    graph.add_node("manager", make_manager_node(llm, rag))
    graph.add_node("interviewer", make_interviewer_node(llm))

    graph.set_entry_point("expert")
    graph.add_edge("expert", "manager")
    graph.add_edge("manager", "interviewer")
    graph.add_edge("interviewer", END)

    compiled = graph.compile()
    logger.info("Interview graph compiled: expert → manager → interviewer")
    return compiled
