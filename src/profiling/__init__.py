# __init__.py

"""Profiling functions (schema, later DQ/EDA). Not tools until 2.11."""

from src.profiling.schema import SCHEMA_RESULT_KEYS, detect_schema

__all__ = ["SCHEMA_RESULT_KEYS", "detect_schema"]
