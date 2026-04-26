"""Tests for split_with_overlap, load_csv_documents, and RagSystem."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from multiagent_interviewer.rag.system import (
    RagSystem,
    load_csv_documents,
    split_with_overlap,
)


class TestSplitWithOverlap:
    def test_empty_returns_empty(self) -> None:
        assert split_with_overlap("", 100, 10) == []

    def test_text_shorter_than_chunk(self) -> None:
        result = split_with_overlap("short", 100, 10)
        assert result == ["short"]

    def test_creates_multiple_chunks(self) -> None:
        text = "a" * 250
        chunks = split_with_overlap(text, 100, 20)
        assert len(chunks) >= 3

    def test_chunks_have_overlap(self) -> None:
        text = "abcdefghij" * 30
        chunks = split_with_overlap(text, 100, 20)
        joined = "".join(chunks)
        assert "a" in joined and "j" in joined

    def test_breaks_at_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence." + "x" * 50
        chunks = split_with_overlap(text, 40, 5)
        assert chunks[0].endswith(".")

    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            split_with_overlap("text", 0, 5)

    def test_rejects_negative_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            split_with_overlap("text", -10, 5)

    def test_rejects_overlap_too_large(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            split_with_overlap("text", 100, 100)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            split_with_overlap("text", 100, -5)


class TestLoadCsvDocuments:
    def test_loads_simple_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nfirst doc\nsecond doc\n")

        docs = load_csv_documents(csv_file)

        assert docs == ["first doc", "second doc"]

    def test_uses_custom_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,content\n1,hello\n2,world\n")

        docs = load_csv_documents(csv_file, text_column="content")

        assert docs == ["hello", "world"]

    def test_skips_nan_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("text\nhello\n\nworld\n")

        docs = load_csv_documents(csv_file)

        assert "hello" in docs
        assert "world" in docs

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv_documents(tmp_path / "nope.csv")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("other_col\nvalue\n")

        with pytest.raises(KeyError, match="text"):
            load_csv_documents(csv_file)


class TestRagSystem:
    def test_default_construction_is_empty(self) -> None:
        system = RagSystem()
        assert system.expert is None
        assert system.manager is None

    def test_search_expert_returns_empty_when_uninitialized(self) -> None:
        system = RagSystem()
        assert system.search_expert("any query") == []

    def test_search_manager_returns_empty_when_uninitialized(self) -> None:
        system = RagSystem()
        assert system.search_manager("any query") == []

    def test_search_expert_delegates_to_retriever(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = ["doc1", "doc2"]

        system = RagSystem(expert=mock_retriever, manager=None)
        results = system.search_expert("query")

        assert results == ["doc1", "doc2"]
        mock_retriever.search.assert_called_once()

    def test_from_csv_with_no_paths(self) -> None:
        system = RagSystem.from_csv(expert_csv=None, manager_csv=None)
        assert system.expert is None
        assert system.manager is None
