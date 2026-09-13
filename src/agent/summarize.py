# summarize.py

"""Compact tool results for checkpointed state: summary + paths, not tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

DEFAULT_MAX_ARTIFACTS = 8

_RECORD_LIST_KEYS = frozenset({"rows"})
_SUMMARY_KEYS = frozenset(
    {
        "status",
        "row_count",
        "rows",
        "sample_row_count",
        "columns",
        "sources",
        "artifacts",
        "truncated",
        "stdout",
    }
)


def is_json_safe(value: Any) -> bool:
    """True when `json.dumps` can encode `value` without a default hook."""
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def artifact_paths_from_result(result: dict[str, Any]) -> list[str]:
    """Return artifact path strings from a tool result. Ignore anything else."""
    raw = result.get("artifacts")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item.strip()]


def _artifact_key(path: str) -> str:
    """Slash-normalize a path for dedup. Do not resolve or read the file."""
    return str(Path(path.strip()))


def merge_artifacts(
    prior: list[Any],
    incoming: list[Any],
    *,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
) -> list[str]:
    """Dedup paths and keep the last `max_artifacts`. Newest stays last."""
    if max_artifacts < 1:
        raise ValueError("max_artifacts must be at least 1.")
    merged: list[str] = []
    seen: set[str] = set()
    for raw in [*prior, *incoming]:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = raw.strip()
        key = _artifact_key(text)
        if key in seen:
            merged = [item for item in merged if _artifact_key(item) != key]
        seen.add(key)
        merged.append(text)
    dropped = max(0, len(merged) - max_artifacts)
    if dropped:
        _LOG.debug("artifacts dropped %s older path(s)", dropped)
    return merged[-max_artifacts:]


def summarize_tool_result(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool result, drop row lists, and add a short summary text."""
    if not isinstance(result, dict):
        raise TypeError("Tool result must be a dict.")
    stored = {
        key: value for key, value in result.items() if key not in _RECORD_LIST_KEYS
    }
    stored["name"] = name
    stored["summary"] = _summary_text(name, result)
    return stored


def checkpoint_tool_result(
    *,
    name: str,
    result: dict[str, Any],
    prior_artifacts: list[str],
) -> dict[str, Any]:
    """Build execute_tool updates: summarized result, artifacts, no row lists."""
    if not isinstance(result, dict):
        raise TypeError("Tool result must be a dict.")
    stored = summarize_tool_result(name, result)
    if not is_json_safe(stored):
        raise ValueError("Tool result is not JSON-safe.")
    updates: dict[str, Any] = {
        "last_tool_result": stored,
        "pending_tool": None,
        "error": None,
    }
    paths = artifact_paths_from_result(stored)
    if paths:
        updates["artifacts"] = merge_artifacts(prior_artifacts, paths)
    if name == "profile_source":
        if not is_json_safe(result):
            raise ValueError("Tool result is not JSON-safe.")
        updates["profile_summary"] = result
    return updates


def _summary_text(name: str, result: dict[str, Any]) -> str:
    """One-line description. Never includes row values."""
    parts: list[str] = []
    status = result.get("status")
    if isinstance(status, str) and status:
        parts.append(status)
    row_count = result.get("row_count")
    if not isinstance(row_count, int) and isinstance(result.get("rows"), list):
        row_count = len(result["rows"])
    if isinstance(row_count, int):
        parts.append(f"{row_count} rows")
    sample_n = result.get("sample_row_count")
    if isinstance(sample_n, int) and not isinstance(result.get("row_count"), int):
        parts.append(f"bounded head {sample_n} rows")
    columns = result.get("columns")
    if isinstance(columns, list) and columns:
        parts.append("columns " + ", ".join(str(column) for column in columns))
    sources = result.get("sources")
    if isinstance(sources, list):
        parts.append(f"{len(sources)} source(s)")
    n_artifacts = len(artifact_paths_from_result(result))
    if n_artifacts:
        parts.append(f"{n_artifacts} artifact(s)")
    stdout = result.get("stdout")
    if isinstance(stdout, str) and stdout:
        parts.append(f"stdout {len(stdout)} chars")
    if result.get("truncated") is True:
        parts.append("truncated")
    if not parts:
        if result and not (_SUMMARY_KEYS & result.keys()):
            _LOG.debug(
                "tool summary used no known keys name=%s keys=%s",
                name,
                sorted(result),
            )
        return name
    return f"{name}: {', '.join(parts)}"
