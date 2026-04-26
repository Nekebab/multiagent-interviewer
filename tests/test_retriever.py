"""Tests for the hybrid retriever (BM25 + bi-encoder + cross-encoder)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from multiagent_interviewer.rag.retriever import (
    RetrieverAgent,
    SearchParams,
    lemmatize,
    tokenize,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# pure-function tests


class TestTokenize:
    """Tests for the `tokenize` function."""

    def test_basic_split(self) -> None:
        assert tokenize("hello world") == ["hello", "world"]

    def test_lowercases(self) -> None:
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_splits_camel_case(self) -> None:
        assert tokenize("MachineLearning") == ["machine", "learning"]

    def test_keeps_compound_camel(self) -> None:
        assert tokenize("MyMachineLearning") == ["my", "machine", "learning"]

    def test_strips_punctuation(self) -> None:
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_drops_pure_numbers(self) -> None:
        assert tokenize("hello 42 world") == ["hello", "world"]

    def test_keeps_alphanumeric(self) -> None:
        assert tokenize("python3 is great") == ["python3", "is", "great"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_only_punctuation(self) -> None:
        assert tokenize("!!! ??? ...") == []

    def test_russian_text(self) -> None:
        assert tokenize("Машинное обучение") == ["машинное", "обучение"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hello", ["hello"]),
            ("Hello", ["hello"]),
            ("HELLO", ["hello"]),
            ("hello-world", ["hello-world"]),
            ("hello_world", ["hello_world"]),
            ("a b c", ["a", "b", "c"]),
        ],
    )
    def test_examples(self, text: str, expected: list[str]) -> None:
        assert tokenize(text) == expected


class TestLemmatize:
    """Tests for the `lemmatize` function (Russian morphology)."""

    @pytest.fixture(scope="class")
    def morph(self) -> Any:
        import pymorphy3

        return pymorphy3.MorphAnalyzer()

    def test_basic_russian(self, morph: Any) -> None:
        result = lemmatize(["машинного", "обучения"], morph)
        assert result == ["машинный", "обучение"]

    def test_already_normal_form(self, morph: Any) -> None:
        result = lemmatize(["программа", "код"], morph)
        assert result == ["программа", "код"]

    def test_empty_list(self, morph: Any) -> None:
        assert lemmatize([], morph) == []

    def test_preserves_length(self, morph: Any) -> None:
        tokens = ["разработчики", "пишут", "код", "ежедневно"]
        result = lemmatize(tokens, morph)
        assert len(result) == len(tokens)


# SearchParams validation


class TestSearchParams:
    def test_defaults(self) -> None:
        params = SearchParams()
        assert params.k == 3
        assert params.rerank_k == 10
        assert params.bm25_weight == 0.25

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        params = SearchParams()
        with pytest.raises(FrozenInstanceError):
            params.k = 5  # type: ignore[misc]

    @pytest.mark.parametrize("bad_weight", [-0.1, 1.5, 2.0, -1.0])
    def test_rejects_bad_weight(self, bad_weight: float) -> None:
        with pytest.raises(ValueError, match="bm25_weight"):
            SearchParams(bm25_weight=bad_weight)

    @pytest.mark.parametrize(
        ("k", "rerank_k"),
        [(0, 10), (-1, 10), (5, 3), (10, 5)],
    )
    def test_rejects_inconsistent_k(self, k: int, rerank_k: int) -> None:
        with pytest.raises(ValueError, match="rerank_k"):
            SearchParams(k=k, rerank_k=rerank_k)


# testing RetrieverAgent.search() with fakes (no real ML)


class FakeEncoder:
    """A bi-encoder that returns deterministic vectors based on text length."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[Any] = []

    def encode(
        self,
        sentences: list[str] | str,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        self.calls.append(sentences)
        if isinstance(sentences, str):
            sentences = [sentences]
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            rng = np.random.default_rng(hash(s) % (2**32))
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
            # inner-product == cosine similarity (because of normalization)
            out[i] /= np.linalg.norm(out[i]) + 1e-9
        return out


class FakeReranker:
    """Cross-encoder that returns scores based on query-doc string similarity."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        scores = np.zeros(len(sentences), dtype=np.float32)
        for i, (query, doc) in enumerate(sentences):
            common = set(query.lower()) & set(doc.lower())
            scores[i] = len(common)
        return scores


@pytest.fixture
def small_corpus() -> list[str]:
    return [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Java is also a programming language",
        "Deep learning is a subset of machine learning",
        "Database management with SQL",
    ]


@pytest.fixture
def retriever(small_corpus: list[str]) -> RetrieverAgent:
    """A retriever built with real BM25 + FAISS, but FAKE encoder/reranker.

    testing FAISS index, BM25 setup,
    candidate merging, deduplication without loading 3 GB of ML models.
    """
    import faiss
    import pymorphy3
    from rank_bm25 import BM25Okapi

    encoder = FakeEncoder(dim=8)
    reranker = FakeReranker()
    morph = pymorphy3.MorphAnalyzer()

    # Build the FAISS
    embeddings = encoder.encode(small_corpus)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Build BM25
    tokenized = [lemmatize(tokenize(d), morph) for d in small_corpus]
    bm25 = BM25Okapi(tokenized)

    return RetrieverAgent(
        docs=small_corpus,
        encoder=encoder,
        reranker=reranker,
        morph=morph,
        bm25=bm25,
        index=index,
    )


class TestRetrieverSearch:
    def test_returns_requested_number_of_docs(self, retriever: RetrieverAgent) -> None:
        results = retriever.search("programming", SearchParams(k=2, rerank_k=5))
        assert len(results) == 2

    def test_returns_strings_from_corpus(
        self, retriever: RetrieverAgent, small_corpus: list[str]
    ) -> None:
        # Invariant: all returned docs must come from the indexed corpus.
        results = retriever.search("learning")
        for doc in results:
            assert doc in small_corpus

    def test_no_duplicates_in_results(self, retriever: RetrieverAgent) -> None:
        # Invariant: dedup logic should never return the same doc twice.
        results = retriever.search("programming language", SearchParams(k=5, rerank_k=10))
        assert len(results) == len(set(results))

    def test_uses_default_params_when_none_given(self, retriever: RetrieverAgent) -> None:
        # search() with no params should use SearchParams() defaults.
        results = retriever.search("python")
        assert len(results) == 3  # with default k=3

    def test_calls_encoder_for_each_query(self, retriever: RetrieverAgent) -> None:
        # encode() was called once at fixture setup (for docs).
        # Each search should call it once more for the query.
        encoder = retriever.encoder
        assert isinstance(encoder, FakeEncoder)
        calls_before = len(encoder.calls)

        retriever.search("test query")

        assert len(encoder.calls) == calls_before + 1

    def test_results_count_does_not_exceed_corpus_size(self, retriever: RetrieverAgent) -> None:
        # Edge case: ask for more than the corpus has.
        params = SearchParams(k=5, rerank_k=20)
        results = retriever.search("anything", params)
        assert len(results) <= 5


class TestRetrieverEdgeCases:
    def test_build_rejects_empty_docs(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RetrieverAgent.build(docs=[])

    def test_search_with_mock_objects(self) -> None:
        """RetrieverAgent doesn't care about real types, only Protocol."""
        encoder = MagicMock()
        encoder.encode.return_value = np.zeros((1, 4), dtype=np.float32)

        reranker = MagicMock()
        reranker.predict.return_value = np.array([0.5], dtype=np.float32)

        morph = MagicMock()
        morph.parse.return_value = [MagicMock(normal_form="word")]

        bm25 = MagicMock()
        bm25.get_scores.return_value = np.array([1.0])

        index = MagicMock()
        index.search.return_value = (None, np.array([[0]]))

        agent = RetrieverAgent(
            docs=["only one doc"],
            encoder=encoder,
            reranker=reranker,
            morph=morph,
            bm25=bm25,
            index=index,
        )

        results = agent.search("query")
        assert results == ["only one doc"]
