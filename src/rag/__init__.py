# __init__.py

"""Local domain-context RAG. Graph binding comes later."""

from src.rag.chunker import chunk_context
from src.rag.readers import ContextReadError, read_context_file
from src.rag.retrieve import (
    DEFAULT_TOP_K,
    ContextRetrieveError,
    DomainContextIndex,
    retrieve_domain_context,
)

__all__ = [
    "DEFAULT_TOP_K",
    "ContextReadError",
    "ContextRetrieveError",
    "DomainContextIndex",
    "chunk_context",
    "read_context_file",
    "retrieve_domain_context",
]
