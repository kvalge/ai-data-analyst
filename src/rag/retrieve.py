# retrieve.py

"""Keyword/TF-IDF retrieve over in-memory context chunks."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from src.config import DEFAULT_RAG_TOP_K

_LOG = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


class ContextRetrieveError(ValueError):
    """A domain-context query could not be run."""


class DomainContextIndex:
    """In-memory TF-IDF index of context chunks. Does not read files."""

    def __init__(self, chunks: Sequence[Mapping[str, str]]) -> None:
        self._chunks = [
            {
                "source_file": str(chunk["source_file"]),
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"]),
            }
            for chunk in chunks
        ]
        self._idf = _idf_weights([_tokens(chunk["text"]) for chunk in self._chunks])
        self._vectors = [
            _tfidf(_tokens(chunk["text"]), self._idf) for chunk in self._chunks
        ]
        _LOG.info("index context chunks=%s", len(self._chunks))

    @property
    def chunks(self) -> list[dict[str, str]]:
        """Copied `{source_file, chunk_id, text}` rows. No scores."""
        return [dict(chunk) for chunk in self._chunks]

    def retrieve(self, query: str, *, top_k: int) -> dict[str, Any]:
        """Return `{chunks}` scored by TF-IDF cosine similarity."""
        if top_k < 1:
            raise ContextRetrieveError("top_k must be at least 1.")
        if not query.strip():
            raise ContextRetrieveError("Query must not be empty.")
        query_vec = _tfidf(_tokens(query), self._idf)
        ranked: list[tuple[float, int]] = []
        for position, vector in enumerate(self._vectors):
            score = _cosine(query_vec, vector)
            if score > 0.0:
                ranked.append((score, position))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        hits = ranked[:top_k]
        _LOG.info("retrieve context hits=%s top_k=%s", len(hits), top_k)
        return {
            "chunks": [
                {**self._chunks[position], "score": score}
                for score, position in hits
            ]
        }


def retrieve_domain_context(
    query: str,
    *,
    index: DomainContextIndex,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> dict[str, Any]:
    """Return `{chunks}` scored by TF-IDF cosine similarity.

    `index` is injected by the caller, not the LLM. Does not log the query
    or chunk text. Omits zero-score chunks.
    """
    return index.retrieve(query, top_k=top_k)


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens. Does not log `text`."""
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def _idf_weights(documents: Sequence[Sequence[str]]) -> dict[str, float]:
    """Smoothed inverse document frequency. Empty corpus yields `{}`."""
    n_docs = len(documents)
    if n_docs == 0:
        return {}
    df: dict[str, int] = {}
    for tokens in documents:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return {
        term: math.log((n_docs + 1) / (count + 1)) + 1.0
        for term, count in df.items()
    }


def _tfidf(tokens: Sequence[str], idf: Mapping[str, float]) -> dict[str, float]:
    """Term-frequency × IDF vector. Unknown query terms score 0."""
    if not tokens:
        return {}
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    length = len(tokens)
    return {
        term: (count / length) * idf.get(term, 0.0) for term, count in tf.items()
    }


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity of two sparse vectors."""
    if not left or not right:
        return 0.0
    dot = sum(value * right[key] for key, value in left.items() if key in right)
    norm_left = math.sqrt(sum(value * value for value in left.values()))
    norm_right = math.sqrt(sum(value * value for value in right.values()))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)
