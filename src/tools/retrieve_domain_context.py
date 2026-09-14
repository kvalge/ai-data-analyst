# retrieve_domain_context.py

"""Tool: retrieve snippets from uploaded domain-context documents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import DEFAULT_RAG_TOP_K
from src.rag.index import reindex_context
from src.rag.retrieve import retrieve_domain_context as search_context_index
from src.tools.contracts import ToolContract

_LOG = logging.getLogger(__name__)

RETRIEVE_DOMAIN_CONTEXT = ToolContract(
    name="retrieve_domain_context",
    description=(
        "Retrieve snippets from uploaded domain-context documents "
        "(markdown, text, PDF) for glossary terms, KPI definitions, and "
        "business vocabulary. Provide query. Does not search analysis "
        "data files."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "default": DEFAULT_RAG_TOP_K,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "text": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": [
                        "source_file",
                        "chunk_id",
                        "text",
                        "score",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["chunks"],
        "additionalProperties": False,
    },
)


def retrieve_domain_context(
    query: str,
    *,
    context_dir: Path,
    cache_dir: Path,
    max_bytes: int,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> dict[str, Any]:
    """Return top-k context snippets. Never indexes data files.

    `context_dir`, `cache_dir`, and `max_bytes` are injected by the app,
    not the LLM.
    """
    _LOG.info("retrieve_domain_context top_k=%s", top_k)
    index = reindex_context(
        context_dir, cache_dir=cache_dir, max_bytes=max_bytes
    )
    return search_context_index(query, index=index, top_k=top_k)
