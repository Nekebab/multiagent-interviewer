"""Shared utilities for agent nodes."""

from __future__ import annotations

from multiagent_interviewer.graph.state import Message, Role


def format_messages_for_prompt(messages: list[Message]) -> str:
    """Render messages as a Russian-labeled dialogue suitable for a prompt."""
    lines: list[str] = []
    for msg in messages:
        if msg.role is Role.SYSTEM:
            continue
        speaker = "Кандидат" if msg.role is Role.USER else "Интервьюер"
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines) if lines else "Диалог еще не начался"
