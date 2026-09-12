# profile_source.py

"""Tool: compact schema + DQ + EDA summary for a registered file source.

Goal: a *quick overview* of a bounded head of rows (`n_rows`, default
`SAMPLE_N_ROWS`), cached by content hash. This is not a dedicated
whole-file profiler.

When the file has fewer rows than `n_rows`, that overview happens to
cover every row. DQ/EDA still run only on the frame that was read —
there is no second pass over the rest of the file. An exact profile of
a large file is out of scope here (`load_full_file` is 2.12).

Exception: CSV `schema.file_row_count` is a cheap full-file line count.
Postgres is not profiled yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import DEFAULT_SAMPLE_N_ROWS, Settings
from src.db.postgres import postgres_configured
from src.profiling.cache import read_profile_cache, write_profile_cache
from src.profiling.dq import (
    detect_duplicates,
    detect_inconsistent_formatting,
    detect_nulls,
    detect_outliers,
    detect_type_mismatches,
)
from src.profiling.eda import correlations, distributions, summary_stats
from src.profiling.schema import detect_schema
from src.storage.registry import get_file_source
from src.storage.sources import DataSource, make_postgres_source
from src.tools.contracts import ToolContract
from src.tools.read_sample import _read_sample_frame
from src.validation.data_files import FileValidationError, validate_data_file

_LOG = logging.getLogger(__name__)


class ProfileError(Exception):
    """Profiling cannot run for this source."""

PROFILE_SOURCE = ToolContract(
    name="profile_source",
    description=(
        "Run schema detection, data-quality checks, and EDA on a registered "
        "file source. Returns a compact cached summary, never the dataset. "
        "Profiles a bounded head of rows (default SAMPLE_N_ROWS), not an "
        "exact whole-file profile unless the file is smaller than that cap. "
        "Postgres sources are not supported yet."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "n_rows": {
                "type": "integer",
                "minimum": 1,
                "default": DEFAULT_SAMPLE_N_ROWS,
            },
        },
        "required": ["source_id"],
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "cached": {"type": "boolean"},
            "sample_row_count": {"type": "integer"},
            "schema": {"type": "object"},
            "dq": {"type": "object"},
            "eda": {"type": "object"},
        },
        "required": [
            "source_id",
            "cached",
            "sample_row_count",
            "schema",
            "dq",
            "eda",
        ],
        "additionalProperties": False,
    },
)


def profile_source(
    *,
    upload_dir: Path,
    cache_dir: Path,
    n_rows: int,
    max_bytes: int,
    source_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return a cached compact profile of a bounded head of the file.

    This is a quick overview, not an exact whole-dataset profile. Schema
    DQ/EDA numbers (nulls, duplicates, outliers, stats, …) describe only
    the rows in that head. If the file is smaller than `n_rows`, the
    head is the whole file. CSV `schema.file_row_count` still counts all
    data lines. Full-file load is 2.12. Postgres is not profiled yet.

    `upload_dir`, `cache_dir`, `n_rows`, `max_bytes`, and `settings` are
    injected by the app, not the LLM.
    """
    if n_rows < 1:
        raise FileValidationError("n_rows must be at least 1.")
    source = _resolve_profile_source(upload_dir, source_id, settings)
    if source.kind == "postgres":
        raise ProfileError("Postgres profiling is not supported yet.")
    if source.stored_path is None:
        raise FileValidationError(f"Unknown source_id: {source_id}")
    path = source.stored_path

    cached = read_profile_cache(cache_dir, source)
    if cached is not None:
        _LOG.info("profile_source cache hit source_id=%s", source.source_id)
        return {**cached, "cached": True}

    validate_data_file(path, max_bytes=max_bytes)
    _LOG.info(
        "profile_source compute source_id=%s n_rows=%s path=%s",
        source.source_id,
        n_rows,
        path,
    )
    # Bounded head only (same reader as preview). DQ/EDA do not scan unread rows.
    frame = _read_sample_frame(path, n_rows)
    payload = {
        "source_id": source.source_id,
        "sample_row_count": int(len(frame)),
        "schema": detect_schema(frame, path=path),
        "dq": {
            "nulls": detect_nulls(frame),
            "duplicates": detect_duplicates(frame),
            "type_mismatches": detect_type_mismatches(frame),
            "formatting": detect_inconsistent_formatting(frame),
            "outliers": detect_outliers(frame),
        },
        "eda": {
            "summary": summary_stats(frame),
            "distributions": distributions(frame),
            "correlations": correlations(frame),
        },
    }
    write_profile_cache(cache_dir, source, payload)
    return {**payload, "cached": False}


def _resolve_profile_source(
    upload_dir: Path,
    source_id: str,
    settings: Settings | None,
) -> DataSource:
    source = get_file_source(upload_dir, source_id)
    if (
        source is None
        and settings is not None
        and postgres_configured(settings)
    ):
        candidate = make_postgres_source(settings)
        if candidate.source_id == source_id:
            source = candidate
    if source is None:
        raise FileValidationError(f"Unknown source_id: {source_id}")
    return source
