# test_retrieve.py

"""Tests for keyword/TF-IDF retrieve. No dataset rows; text only."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_RAG_TOP_K
from src.rag.chunker import chunk_context
from src.rag.readers import read_context_file
from src.rag.retrieve import (
    ContextRetrieveError,
    DomainContextIndex,
    retrieve_domain_context,
)

_MAX = DEFAULT_MAX_UPLOAD_BYTES


def _mixed_index(sample_glossary_md: Path) -> DomainContextIndex:
    """Glossary fixture plus an unrelated shipping chunk."""
    return DomainContextIndex(
        [
            *chunk_context(read_context_file(sample_glossary_md, max_bytes=_MAX)),
            *chunk_context(
                {
                    "source_file": "shipping.md",
                    "text": "# Shipping\n\nLead time is five days.\n",
                }
            ),
        ]
    )


def test_retrieve_hits_glossary_fixture(sample_glossary_md: Path):
    """A revenue query returns the glossary chunk, not the distractor."""
    result = retrieve_domain_context(
        "What is revenue?",
        index=_mixed_index(sample_glossary_md),
        top_k=1,
    )
    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert chunk["source_file"] == "sample_glossary.md"
    assert chunk["chunk_id"] == "sample_glossary.md:0"
    assert "Revenue is net of returns." in chunk["text"]
    assert "Lead time" not in chunk["text"]
    assert "date,region,revenue" not in chunk["text"]
    assert chunk["score"] > 0


def test_retrieve_hits_matching_distractor(sample_glossary_md: Path):
    """A shipping query returns the shipping chunk, not the glossary."""
    result = retrieve_domain_context(
        "shipping lead time",
        index=_mixed_index(sample_glossary_md),
        top_k=1,
    )
    assert result["chunks"][0]["source_file"] == "shipping.md"
    assert "Lead time is five days." in result["chunks"][0]["text"]
    assert "Revenue is net of returns." not in result["chunks"][0]["text"]


def test_unmatched_query_returns_no_chunks(sample_glossary_md: Path):
    """A query with no overlapping terms does not invent a hit."""
    result = retrieve_domain_context(
        "zzzzz",
        index=_mixed_index(sample_glossary_md),
        top_k=DEFAULT_RAG_TOP_K,
    )
    assert result["chunks"] == []


def test_empty_query_raises(sample_glossary_md: Path):
    """A blank query fails closed."""
    with pytest.raises(ContextRetrieveError, match="empty"):
        retrieve_domain_context(
            "   ",
            index=_mixed_index(sample_glossary_md),
        )
