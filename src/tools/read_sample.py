# read_sample.py

"""Tool: peek at a data file's columns and a bounded head of rows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_SAMPLE_N_ROWS
from src.storage.registry import get_file_source
from src.tools.contracts import ToolContract
from src.validation.data_files import FileValidationError, validate_data_file

_LOG = logging.getLogger(__name__)

READ_FILE_SAMPLE = ToolContract(
    name="read_file_sample",
    description=(
        "Peek at an analysis data file: column names, dtypes, and a short "
        "head of rows. Provide exactly one of source_id or path. Never "
        "returns the full file."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "path": {"type": "string"},
            "n_rows": {
                "type": "integer",
                "minimum": 1,
                "default": DEFAULT_SAMPLE_N_ROWS,
            },
        },
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "columns": {"type": "array", "items": {"type": "string"}},
            "dtypes": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "row_count": {"type": "integer"},
            "rows": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["columns", "dtypes", "row_count", "rows"],
        "additionalProperties": False,
    },
)


def read_file_sample(
    *,
    upload_dir: Path,
    n_rows: int,
    max_bytes: int,
    source_id: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return dtypes, columns, and at most `n_rows` records. Never the full file.

    `upload_dir` and `max_bytes` are injected by the app, not the LLM.
    """
    if n_rows < 1:
        raise FileValidationError("n_rows must be at least 1.")
    resolved = _resolve_sample_path(
        upload_dir=upload_dir, source_id=source_id, path=path
    )
    validate_data_file(resolved, max_bytes=max_bytes)
    _LOG.info(
        "read_file_sample n_rows=%s source_id=%s path=%s",
        n_rows,
        source_id,
        resolved,
    )
    frame = _read_sample_frame(resolved, n_rows)
    payload = frame.to_json(orient="records")
    if payload is None:
        raise FileValidationError("Could not serialize the sample.")
    records = json.loads(payload)
    return {
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "row_count": len(records),
        "rows": records,
    }


def _resolve_sample_path(
    *,
    upload_dir: Path,
    source_id: str | None,
    path: str | Path | None,
) -> Path:
    has_id = source_id is not None and str(source_id) != ""
    has_path = path is not None and str(path) != ""
    if has_id == has_path:
        raise FileValidationError("Provide exactly one of source_id or path.")
    if has_id:
        source = get_file_source(upload_dir, str(source_id))
        if source is None or source.stored_path is None:
            raise FileValidationError(f"Unknown source_id: {source_id}")
        return source.stored_path

    incoming = Path(str(path)).expanduser()
    if not incoming.is_absolute():
        incoming = upload_dir / incoming
    resolved = incoming.resolve()
    upload_root = upload_dir.resolve()
    if not resolved.is_relative_to(upload_root):
        raise FileValidationError("Path is outside the upload directory.")
    return resolved


def _read_sample_frame(path: Path, n_rows: int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, nrows=n_rows)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path, nrows=n_rows)
        if suffix == ".json":
            # JSON arrays have no nrows; this reads the whole file (capped by
            # MAX_UPLOAD_BYTES). Line-delimited JSON could use lines=True + nrows.
            return pd.read_json(path).head(n_rows)
    except (OSError, ValueError, ImportError, UnicodeError) as exc:
        raise FileValidationError(f"Could not read a sample from {path.name}.") from exc
    raise FileValidationError(f"Unsupported sample suffix: {suffix}")
