# __init__.py

"""Local domain-context RAG. Readers first; chunking and retrieve come later."""

from src.rag.readers import ContextReadError, read_context_file

__all__ = ["ContextReadError", "read_context_file"]
