# load_full_file.py

"""Tool: load a registered file only when it is under row and byte limits.

Over the limit this returns needs_approval (graph interrupt in 4.10).
It does not put row records in the result — large tables must not land
in the LLM context. Postgres is not a file load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings
from src.profiling.schema import cheap_file_row_count
from src.tools.contracts import ToolContract
from src.tools.profile_source import ProfileError, _resolve_profile_source
from src.validation.data_files import FileValidationError, validate_data_file

_LOG = logging.getLogger(__name__)

STATUS_LOADED = "loaded"
STATUS_NEEDS_APPROVAL = "needs_approval"
REASON_SIZE_BYTES = "size_bytes"
REASON_ROW_COUNT = "row_count"
REASON_ROW_COUNT_POST_LOAD = "row_count_post_load"


@dataclass(frozen=True)
class _LoadMemo:
    """Row count and column names from one pandas read. Not the frame."""

    fingerprint: tuple[str, int, int]
    row_count: int
    columns: tuple[str, ...]


_LOAD_MEMO: dict[tuple[str, str], _LoadMemo] = {}

LOAD_FULL_FILE = ToolContract(
    name="load_full_file",
    description=(
        "Load a registered data file when it is under MAX_FULL_LOAD_ROWS "
        "and the byte limit. Over the limit, return needs_approval without "
        "loading unless the graph injects an approved over-limit run. "
        "Never returns row records. Postgres is not supported."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
        },
        "required": ["source_id"],
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "source_id": {"type": "string"},
            "row_count": {"type": ["integer", "null"]},
            "size_bytes": {"type": "integer"},
            "max_full_load_rows": {"type": "integer"},
            "max_bytes": {"type": "integer"},
            "reason": {"type": "string"},
            "columns": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "status",
            "source_id",
            "row_count",
            "size_bytes",
            "max_full_load_rows",
            "max_bytes",
            "reason",
            "columns",
        ],
        "additionalProperties": False,
    },
)


def load_full_file(
    *,
    upload_dir: Path,
    source_id: str,
    max_full_load_rows: int,
    max_bytes: int,
    settings: Settings | None = None,
    allow_over_limit: bool = False,
) -> dict[str, Any]:
    """Load a file under limits, or return needs_approval without loading.

    `upload_dir`, limits, `settings`, and `allow_over_limit` are injected
    by the app, not the LLM. Success metadata has columns and row_count,
    never the dataset.
    `reason` is `row_count` when the cheap CSV count already exceeded the
    cap (no pandas load), `row_count_post_load` when pandas had to load
    first (xlsx/json), `size_bytes` when the file is too large, or a
    comma-joined pair of the pre-load tokens.
    """
    if max_full_load_rows < 1:
        raise FileValidationError("max_full_load_rows must be at least 1.")
    if max_bytes < 1:
        raise FileValidationError("max_bytes must be at least 1.")
    source = _resolve_profile_source(upload_dir, source_id, settings)
    if source.kind == "postgres":
        raise ProfileError("Postgres is not a file load.")
    if source.stored_path is None:
        raise FileValidationError(f"Unknown source_id: {source_id}")
    path = source.stored_path
    size_bytes = int(path.stat().st_size)
    estimated_rows = cheap_file_row_count(path)
    reasons: list[str] = []
    if size_bytes > max_bytes:
        reasons.append(REASON_SIZE_BYTES)
    if estimated_rows is not None and estimated_rows > max_full_load_rows:
        reasons.append(REASON_ROW_COUNT)
    if reasons and not allow_over_limit:
        _LOG.info(
            "load_full_file needs_approval source_id=%s reasons=%s "
            "row_count=%s size_bytes=%s",
            source.source_id,
            reasons,
            estimated_rows,
            size_bytes,
        )
        return _result(
            status=STATUS_NEEDS_APPROVAL,
            source_id=source.source_id,
            row_count=estimated_rows,
            size_bytes=size_bytes,
            max_full_load_rows=max_full_load_rows,
            max_bytes=max_bytes,
            reason=",".join(reasons),
            columns=[],
        )

    memo = _recall_load(source.source_id, path)
    if memo is not None:
        return _result_from_memo(
            source_id=source.source_id,
            memo=memo,
            size_bytes=size_bytes,
            max_full_load_rows=max_full_load_rows,
            max_bytes=max_bytes,
            allow_over_limit=allow_over_limit,
        )

    check_bytes = max(max_bytes, size_bytes) if allow_over_limit else max_bytes
    validate_data_file(path, max_bytes=check_bytes)
    _LOG.info(
        "load_full_file load source_id=%s size_bytes=%s path=%s",
        source.source_id,
        size_bytes,
        path,
    )
    frame = _read_full_frame(path)
    row_count = int(len(frame))
    columns = [str(column) for column in frame.columns]
    _remember_load(source.source_id, path, row_count, columns)
    if row_count > max_full_load_rows and not allow_over_limit:
        _LOG.info(
            "load_full_file needs_approval source_id=%s reasons=%s "
            "row_count=%s size_bytes=%s",
            source.source_id,
            [REASON_ROW_COUNT_POST_LOAD],
            row_count,
            size_bytes,
        )
        return _result(
            status=STATUS_NEEDS_APPROVAL,
            source_id=source.source_id,
            row_count=row_count,
            size_bytes=size_bytes,
            max_full_load_rows=max_full_load_rows,
            max_bytes=max_bytes,
            reason=REASON_ROW_COUNT_POST_LOAD,
            columns=[],
        )
    return _result(
        status=STATUS_LOADED,
        source_id=source.source_id,
        row_count=row_count,
        size_bytes=size_bytes,
        max_full_load_rows=max_full_load_rows,
        max_bytes=max_bytes,
        reason="",
        columns=columns,
    )


def _result(
    *,
    status: str,
    source_id: str,
    row_count: int | None,
    size_bytes: int,
    max_full_load_rows: int,
    max_bytes: int,
    reason: str,
    columns: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "source_id": source_id,
        "row_count": row_count,
        "size_bytes": size_bytes,
        "max_full_load_rows": max_full_load_rows,
        "max_bytes": max_bytes,
        "reason": reason,
        "columns": columns,
    }


def _memo_key(source_id: str, path: Path) -> tuple[str, str]:
    """Scope one pandas-read memo to a source_id and its resolved path."""
    return (source_id, str(path.resolve()))


def _fingerprint(path: Path) -> tuple[str, int, int]:
    """Staleness check for a memo: resolved path, size, and mtime."""
    resolved = path.resolve()
    stat = resolved.stat()
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def _remember_load(
    source_id: str, path: Path, row_count: int, columns: list[str]
) -> None:
    """Keep metadata from one pandas read. Never the frame or row values."""
    _LOAD_MEMO[_memo_key(source_id, path)] = _LoadMemo(
        fingerprint=_fingerprint(path),
        row_count=row_count,
        columns=tuple(columns),
    )


def _recall_load(source_id: str, path: Path) -> _LoadMemo | None:
    """Return a memo when this source_id's file has not changed."""
    key = _memo_key(source_id, path)
    memo = _LOAD_MEMO.get(key)
    if memo is None:
        return None
    if memo.fingerprint != _fingerprint(path):
        _LOAD_MEMO.pop(key, None)
        return None
    return memo


def _result_from_memo(
    *,
    source_id: str,
    memo: _LoadMemo,
    size_bytes: int,
    max_full_load_rows: int,
    max_bytes: int,
    allow_over_limit: bool,
) -> dict[str, Any]:
    """Reuse a prior read. Do not pandas-load again."""
    if memo.row_count > max_full_load_rows and not allow_over_limit:
        _LOG.info(
            "load_full_file needs_approval source_id=%s reasons=%s "
            "row_count=%s size_bytes=%s memo=1",
            source_id,
            [REASON_ROW_COUNT_POST_LOAD],
            memo.row_count,
            size_bytes,
        )
        return _result(
            status=STATUS_NEEDS_APPROVAL,
            source_id=source_id,
            row_count=memo.row_count,
            size_bytes=size_bytes,
            max_full_load_rows=max_full_load_rows,
            max_bytes=max_bytes,
            reason=REASON_ROW_COUNT_POST_LOAD,
            columns=[],
        )
    return _result(
        status=STATUS_LOADED,
        source_id=source_id,
        row_count=memo.row_count,
        size_bytes=size_bytes,
        max_full_load_rows=max_full_load_rows,
        max_bytes=max_bytes,
        reason="",
        columns=list(memo.columns),
    )


def _read_full_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".json":
            return pd.read_json(path)
    except (OSError, ValueError, ImportError, UnicodeError) as exc:
        raise FileValidationError(f"Could not load {path.name}.") from exc
    raise FileValidationError(f"Unsupported load suffix: {suffix}")
