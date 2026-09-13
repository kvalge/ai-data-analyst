# __init__.py

"""Local domain-context RAG. Retrieve comes later."""

from src.rag.chunker import chunk_context
from src.rag.readers import ContextReadError, read_context_file

__all__ = ["ContextReadError", "chunk_context", "read_context_file"]
