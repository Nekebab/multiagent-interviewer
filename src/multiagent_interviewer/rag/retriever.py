"""Hybrid retrieval: BM25 + bi-encoder + cross-encoder reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from pymorphy3 import MorphAnalyzer
    from rank_bm25 import BM25Okapi


class Encoder(Protocol):
    """A minimal bi-encoder interface (subset of SentenceTransformer)."""

    def encode(
        self,
        sentences: list[str] | str,
        **kwargs: Any,
    ) -> Any:
        """Encode sentences to vectors. Returns numpy array in our usage."""
        ...


class Reranker(Protocol):
    """A minimal cross-encoder interface (subset of CrossEncoder)."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        **kwargs: Any,
    ) -> Any:
        """Score (query, doc) pairs. Returns numpy array of scores."""
        ...


# Tokenization
_CAMEL_CASE_RE = re.compile(r"([a-z])([A-Z])")
_NON_WORD_RE = re.compile(r"[^\w\s-]")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    Handles CamelCase by splitting it into separate words.
    Drops tokens that contain no letters (pure numbers, punctuation).

    >>> tokenize("MachineLearning is fun!")
    ['machine', 'learning', 'is', 'fun']
    """
    text = _CAMEL_CASE_RE.sub(r"\1 \2", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    return [t for t in text.split() if any(c.isalpha() for c in t)]


def lemmatize(tokens: list[str], morph: MorphAnalyzer) -> list[str]:
    """Reduce each token to its dictionary form (Russian).

    >>> # "машинного обучения" → ["машинный", "обучение"]
    """
    return [morph.parse(t)[0].normal_form for t in tokens]


# Configuration object: passed to RetrieverAgent.search()


@dataclass(frozen=True)
class SearchParams:
    """Tunable parameters for hybrid search."""

    k: int = 3
    """Number of final documents to return after reranking."""

    rerank_k: int = 10
    """Number of candidates to gather before cross-encoder reranking."""

    bm25_weight: float = 0.25
    """Fraction of `rerank_k` candidates pulled from BM25"""

    def __post_init__(self) -> None:
        if not 0 <= self.bm25_weight <= 1:
            raise ValueError(f"bm25_weight must be in [0, 1], got {self.bm25_weight}")
        if self.k < 1 or self.rerank_k < self.k:
            raise ValueError(f"need rerank_k ({self.rerank_k}) >= k ({self.k}) >= 1")


@dataclass
class RetrieverAgent:
    """Hybrid retriever: BM25 (lexical) + bi-encoder (semantic) + cross-encoder (rerank).

    The retriever is built once over a fixed set of documents. Calling `search()`
    returns the top-k most relevant documents for a given query.
    """

    docs: list[str]
    encoder: Encoder
    reranker: Reranker
    morph: MorphAnalyzer
    bm25: BM25Okapi
    index: object

    @classmethod
    def build(
        cls,
        docs: list[str],
        *,
        embedding_model: str = "deepvk/USER-bge-m3",
        cross_encoder_model: str = "PitKoro/cross-encoder-ru-msmarco-passage",
    ) -> RetrieverAgent:
        """Construct a RetrieverAgent with real ML models.

        This method does the expensive work: loading models (~3 GB), building
        the FAISS index, tokenizing every document. Call it once at startup.
        """
        import faiss
        import pymorphy3
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if not docs:
            raise ValueError("docs must be non-empty")

        logger.info("Loading embedding model: {}", embedding_model)
        encoder = SentenceTransformer(embedding_model)

        logger.info("Loading cross-encoder: {}", cross_encoder_model)
        reranker = CrossEncoder(cross_encoder_model)

        logger.info("Encoding {} documents", len(docs))
        embeddings = encoder.encode(docs, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        logger.info("Tokenizing documents for BM25")
        morph = pymorphy3.MorphAnalyzer()
        tokenized = [lemmatize(tokenize(d), morph) for d in docs]
        bm25 = BM25Okapi(tokenized)

        logger.info("RetrieverAgent ready ({} docs)", len(docs))
        return cls(
            docs=docs,
            encoder=cast(Encoder, encoder),
            reranker=cast(Reranker, reranker),
            morph=morph,
            bm25=bm25,
            index=index,
        )

    def search(self, query: str, params: SearchParams | None = None) -> list[str]:
        """Return top-k documents most relevant to `query`."""
        params = params or SearchParams()

        # BM25 candidates
        tokenized_query = lemmatize(tokenize(query), self.morph)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        n_bm25 = int(params.rerank_k * params.bm25_weight)
        bm25_indices = np.argsort(bm25_scores)[::-1][:n_bm25]

        # bi-encoder candidates
        n_dense = params.rerank_k - n_bm25
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        _, dense_indices = self.index.search(query_emb, min(n_dense, len(self.docs)))  # type: ignore[attr-defined]
        dense_indices = dense_indices[0]

        # Merge candidate sets, deduplicate
        candidate_set = list(
            dict.fromkeys([int(i) for i in bm25_indices] + [int(i) for i in dense_indices])
        )[: params.rerank_k]

        if not candidate_set:
            logger.warning("No candidates found for query: {!r}", query)
            return []

        # Cross-encoder rerank
        candidate_docs = [self.docs[i] for i in candidate_set]
        pairs = [(query, doc) for doc in candidate_docs]
        scores = self.reranker.predict(pairs)

        # top-k by reranker score
        order = np.argsort(scores)[::-1]
        return [candidate_docs[i] for i in order[: params.k]]
