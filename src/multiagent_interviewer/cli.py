"""CLI entry point: interactive command-line interview session."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from multiagent_interviewer.config import get_settings
from multiagent_interviewer.feedback import generate_final_feedback, save_feedback_report
from multiagent_interviewer.graph.builder import build_interview_graph
from multiagent_interviewer.graph.state import (
    CandidateInfo,
    Grade,
    InterviewState,
    Role,
)
from multiagent_interviewer.llm.client import LLMClient
from multiagent_interviewer.logging_setup import setup_logging
from multiagent_interviewer.rag.system import RagSystem

_END_COMMANDS = {"стоп", "stop", "завершить", "фидбэк", "feedback", "отчет"}


def _prompt_candidate_info() -> CandidateInfo:
    """Interactive collection of candidate info from stdin."""
    print("\n=== Candidate setup ===")
    name = input("Name: ").strip()
    position = input("Position (e.g. 'Backend Developer'): ").strip()
    grade_input = input("Grade [Junior/Middle/Senior]: ").strip()

    try:
        grade = Grade(grade_input.capitalize())
    except ValueError:
        print(f"Unknown grade '{grade_input}', defaulting to Junior")
        grade = Grade.JUNIOR

    experience = input("Experience and skills: ").strip()

    return CandidateInfo(
        name=name,
        position=position,
        grade=grade,
        experience=experience,
    )


def _initialize_rag() -> RagSystem | None:
    """Try to load CSV-backed RAG. Return None if files are missing."""
    settings = get_settings()
    expert_csv = settings.expert_csv_path
    manager_csv = settings.manager_csv_path

    expert_path = expert_csv if expert_csv.exists() else None
    manager_path = manager_csv if manager_csv.exists() else None

    if expert_path is None and manager_path is None:
        logger.warning(
            "No knowledge-base CSVs found at {} or {} — running without RAG",
            expert_csv,
            manager_csv,
        )
        print(" No RAG knowledge bases found — running in LLM-only mode")
        return None

    print("\n Initializing RAG knowledge bases...")
    if expert_path:
        print(f"  • Expert KB:  {expert_path}")
    if manager_path:
        print(f"  • Manager KB: {manager_path}")

    rag = RagSystem.from_csv(
        expert_csv=expert_path,
        manager_csv=manager_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    print("RAG ready — hybrid retrieval (BM25 + bi-encoder + cross-encoder)")
    return rag


def main() -> None:
    """Run an interactive interview session."""
    settings = get_settings()
    setup_logging(settings.log_level)

    print("=" * 60)
    print("  MULTI-AGENT INTERVIEW COACH")
    print("=" * 60)

    candidate = _prompt_candidate_info()
    state = InterviewState.initial(candidate)

    print("\nLoading models (first run downloads ~3 GB) ...")
    rag = _initialize_rag()
    llm = LLMClient()
    graph = build_interview_graph(llm, rag)

    print("\n" + "=" * 60)
    print(f"Starting interview with {candidate.name}")
    print("Type 'стоп' or 'feedback' to end and generate the report.")
    print("=" * 60)

    try:
        state_dict = graph.invoke(state)
    except Exception as e:
        print(f"\n Ошибка при обращении к LLM на старте: {e}")
        return
    state = InterviewState.model_validate(state_dict)
    _print_agent_thoughts(state)
    _print_last_interviewer_message(state)

    # Main loop
    try:
        while state.is_active and state.current_turn <= settings.max_turns:
            print(
                f"\n[Turn {state.current_turn - 1}] Your answer "
                f"(blank line to send, 'стоп' to end early):"
            )
            try:
                user_input = _read_multiline_input()
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted.")
                break

            if not user_input:
                continue
            if user_input.lower() in _END_COMMANDS:
                print("Ending interview at user request.")
                break

            state.add_message(Role.USER, user_input)
            try:
                state_dict = graph.invoke(state)
            except Exception as e:
                print(f"\n⚠ Ошибка при обращении к LLM: {e}")
                break
            state = InterviewState.model_validate(state_dict)

            _print_agent_thoughts(state)
            _print_last_interviewer_message(state)

            if not state.is_active:
                print("\nManager has decided to end the interview.")
                break
    except KeyboardInterrupt:
        print("\n\n Интервью прервано. Генерирую отчёт по тому, что есть...")

    print("\n" + "=" * 60)
    print("  Generating final feedback...")
    print("=" * 60)

    if state.current_turn <= 1 and not state.log:
        print("Недостаточно данных для отчёта.")
        return

    try:
        feedback = generate_final_feedback(state, llm)
    except Exception as e:
        print(f" Не удалось сгенерировать финальный отчёт: {e}")
        return

    output_path = save_feedback_report(state, feedback, Path("output"))

    print("\n" + "=" * 60)
    print("  ИТОГОВАЯ ОЦЕНКА")
    print("=" * 60)
    print(f"Уровень:        {feedback.grade_assessment.value}")
    print(f"Рекомендация:   {feedback.hiring_recommendation.value}")
    print(f"Уверенность:    {feedback.confidence_score}%")

    if feedback.confirmed_skills:
        print("\n— ПОДТВЕРЖДЁННЫЕ НАВЫКИ —")
        for s in feedback.confirmed_skills:
            print(f"  • {s}")

    if feedback.knowledge_gaps:
        print("\n— ПРОБЕЛЫ В ЗНАНИЯХ —")
        for g in feedback.knowledge_gaps:
            print(f"  • {g}")

    print("\n— SOFT SKILLS —")
    print(f"  {feedback.soft_skills_summary}")

    if feedback.learning_roadmap:
        print("\n— ПЛАН ОБУЧЕНИЯ —")
        for r in feedback.learning_roadmap:
            print(f"  • {r}")

    if feedback.suggested_resources:
        print("\n— РЕКОМЕНДУЕМЫЕ РЕСУРСЫ —")
        for r in feedback.suggested_resources:
            print(f"  • {r}")

    print(f"\nПолный отчёт: {output_path}")


def _print_last_interviewer_message(state: InterviewState) -> None:
    """Print the most recent assistant message (from the interviewer)."""
    for msg in reversed(state.messages):
        if msg.role is Role.ASSISTANT:
            print(f"\nInterviewer: {msg.content}")
            return


def _print_agent_thoughts(state: InterviewState) -> None:
    """Show what the Expert and Manager agents are thinking."""
    if state.expert_analysis is not None:
        print("\n" + "─" * 60)
        print("EXPERT AGENT")
        print("─" * 60)
        print("Корректность ответа:")
        print(f"  {state.expert_analysis.technical_correctness}")
        if state.expert_analysis.knowledge_gaps:
            print("\nПробелылы:")
            for gap in state.expert_analysis.knowledge_gaps[:3]:
                print(f"  • {gap}")
        if state.expert_analysis.recommended_follow_ups:
            print("\nРекомендованные follow-up:")
            for q in state.expert_analysis.recommended_follow_ups[:2]:
                print(f"  • {q}")
        print(f"\nСложность: {state.expert_analysis.difficulty_adjustment}")

    if state.manager_decision is not None:
        print("\n" + "─" * 60)
        print(" MANAGER AGENT")
        print("─" * 60)
        print(f"Прогресс: {state.manager_decision.progress_assessment}")
        print(f"Soft skills: {state.manager_decision.soft_skills_score}/10")
        print(f"Стратегия: {state.manager_decision.direction}")


def _read_multiline_input() -> str:
    """Read multi-line input from stdin until an empty line."""
    lines: list[str] = []
    while True:
        line = input()
        if not line.strip() and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


if __name__ == "__main__":
    main()
