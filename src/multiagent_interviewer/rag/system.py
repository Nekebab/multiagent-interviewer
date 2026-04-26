"""High-level RAG system: two retrievers (expert + manager) over chunked CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from multiagent_interviewer.rag.retriever import RetrieverAgent, SearchParams

if TYPE_CHECKING:
    pass


# Chunking

_CHUNK_BOUNDARIES = (". ", "! ", "? ", "\n\n", "\n", " ")


def split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long text into overlapping chunks.

    Args:
        text: The full text to split.
        chunk_size: Target chunk length in characters.
        overlap: Characters to repeat at the start of each chunk
            (gives the next chunk some context from the previous one).

    Returns:
        List of chunk strings. Empty input → empty list.

    >>> split_with_overlap("", 100, 10)
    []
    >>> chunks = split_with_overlap("a" * 250, 100, 20)
    >>> [len(c) for c in chunks]
    [100, 100, 90]
    """
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must be in [0, chunk_size), got {overlap} (chunk_size={chunk_size})"
        )

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to find a natural boundary in the second half of the chunk.
        # If found, snap the chunk end to it. If not, cut at chunk_size.
        if end < text_len:
            for boundary in _CHUNK_BOUNDARIES:
                pos = text.rfind(boundary, start, end)
                if pos != -1 and pos > start + chunk_size // 2:
                    end = pos + len(boundary)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def load_csv_documents(csv_path: Path, text_column: str = "text") -> list[str]:
    """Load a column of text from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        text_column: Name of the column containing document text.

    Returns:
        List of document strings. Empty cells are skipped.

    Raises:
        FileNotFoundError: If `csv_path` doesn't exist.
        KeyError: If `text_column` isn't a column in the CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise KeyError(f"Column {text_column!r} not in CSV. Available: {list(df.columns)}")

    docs = [str(t) for t in df[text_column].dropna().tolist()]
    logger.info("Loaded {} documents from {}", len(docs), csv_path)
    return docs


@dataclass
class RagSystem:
    """Two-retriever RAG: separate knowledge bases for expert and manager agents."""

    expert: RetrieverAgent | None = None
    manager: RetrieverAgent | None = None

    @classmethod
    def from_csv(
        cls,
        expert_csv: Path | None = None,
        manager_csv: Path | None = None,
        *,
        text_column: str = "text",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> RagSystem:
        """Build a RagSystem by loading CSV(s), chunking, and indexing."""
        expert = _build_retriever_from_csv(
            expert_csv, text_column, chunk_size, chunk_overlap, role="expert"
        )
        manager = _build_retriever_from_csv(
            manager_csv, text_column, chunk_size, chunk_overlap, role="manager"
        )
        return cls(expert=expert, manager=manager)

    def search_expert(self, query: str, params: SearchParams | None = None) -> list[str]:
        """Search the expert knowledge base. Returns [] if no expert RAG."""
        if self.expert is None:
            logger.warning("Expert retriever not initialized — returning []")
            return []
        return self.expert.search(query, params)

    def search_manager(self, query: str, params: SearchParams | None = None) -> list[str]:
        """Search the manager knowledge base. Returns [] if no manager RAG."""
        if self.manager is None:
            logger.warning("Manager retriever not initialized — returning []")
            return []
        return self.manager.search(query, params)


def _build_retriever_from_csv(
    csv_path: Path | None,
    text_column: str,
    chunk_size: int,
    chunk_overlap: int,
    role: str,
) -> RetrieverAgent | None:
    """load → chunk → build a retriever, or return None on no-path."""
    if csv_path is None:
        logger.info("No CSV provided for {} — skipping retriever", role)
        return None

    docs = load_csv_documents(csv_path, text_column)
    if not docs:
        logger.warning("No docs found for {} — skipping retriever", role)
        return None

    chunks: list[str] = []
    for doc in docs:
        chunks.extend(split_with_overlap(doc, chunk_size, chunk_overlap))

    logger.info("Building {} retriever: {} docs → {} chunks", role, len(docs), len(chunks))
    return RetrieverAgent.build(docs=chunks)
