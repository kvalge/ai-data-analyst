# __init__.py

"""Profiling functions (schema, later DQ/EDA). Not tools until 2.11."""

from src.profiling.cache import read_profile_cache, write_profile_cache
from src.profiling.dq import (
    DUPLICATES_RESULT_KEYS,
    FORMATTING_RESULT_KEYS,
    NULLS_RESULT_KEYS,
    OUTLIERS_RESULT_KEYS,
    TYPE_MISMATCHES_RESULT_KEYS,
    detect_duplicates,
    detect_inconsistent_formatting,
    detect_nulls,
    detect_outliers,
    detect_type_mismatches,
)
from src.profiling.eda import SUMMARY_STATS_RESULT_KEYS, summary_stats
from src.profiling.schema import SCHEMA_RESULT_KEYS, detect_schema

__all__ = [
    "DUPLICATES_RESULT_KEYS",
    "FORMATTING_RESULT_KEYS",
    "NULLS_RESULT_KEYS",
    "OUTLIERS_RESULT_KEYS",
    "SCHEMA_RESULT_KEYS",
    "SUMMARY_STATS_RESULT_KEYS",
    "TYPE_MISMATCHES_RESULT_KEYS",
    "detect_duplicates",
    "detect_inconsistent_formatting",
    "detect_nulls",
    "detect_outliers",
    "detect_schema",
    "detect_type_mismatches",
    "read_profile_cache",
    "summary_stats",
    "write_profile_cache",
]
