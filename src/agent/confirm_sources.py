# confirm_sources.py

"""Decide whether the graph must pause for source confirmation. No UI imports."""

from __future__ import annotations

from typing import Any

from src.config import Settings
from src.tools.list_sources import list_available_sources
from src.tools.read_sample import read_file_sample

KIND_CONFIRM_SOURCES = "confirm_sources"
REASON_NO_SOURCES = "no_sources"
REASON_MULTIPLE_SOURCES = "multiple_sources"
REASON_EMPTY_SCHEMA = "empty_schema"

ACTION_CONFIRM = "confirm"
ACTION_SELECT = "select"
ACTION_ABORT = "abort"

_COMPACT_KEYS = ("source_id", "kind", "original_name")


def compact_source(row: dict[str, Any]) -> dict[str, str]:
    """Source identity for an interrupt payload. No paths, hashes, or rows."""
    return {key: str(row.get(key, "")) for key in _COMPACT_KEYS}


def listed_source_ids(sources: list[dict[str, Any]]) -> set[str]:
    """Return the registered source_id set."""
    return {str(row["source_id"]) for row in sources if row.get("source_id")}


def valid_source_ids(
    source_ids: list[str], sources: list[dict[str, Any]]
) -> list[str]:
    """Keep only ids that are currently registered. Do not invent an id."""
    allowed = listed_source_ids(sources)
    return [item for item in source_ids if item in allowed]


def sample_is_empty(sample: dict[str, Any]) -> bool:
    """True when the bounded sample has no columns or no rows."""
    columns = sample.get("columns") or []
    row_count = sample.get("row_count")
    return len(columns) == 0 or (isinstance(row_count, int) and row_count < 1)


def decide_confirm_reason(
    sources: list[dict[str, Any]], source_ids: list[str]
) -> tuple[str | None, list[str]]:
    """Return (reason, selected ids). `reason` is None when the turn may proceed."""
    selected = valid_source_ids(source_ids, sources)
    if not sources:
        return REASON_NO_SOURCES, []
    if len(sources) > 1 and not selected:
        return REASON_MULTIPLE_SOURCES, []
    if not selected and len(sources) == 1:
        selected = [str(sources[0]["source_id"])]
    return None, selected


def build_interrupt_payload(
    reason: str,
    sources: list[dict[str, Any]],
    source_ids: list[str],
    *,
    column_count: int | None = None,
    row_count: int | None = None,
) -> dict[str, Any]:
    """JSON-safe interrupt value. Never includes dataset rows."""
    payload: dict[str, Any] = {
        "kind": KIND_CONFIRM_SOURCES,
        "reason": reason,
        "sources": [compact_source(row) for row in sources],
        "source_ids": list(source_ids),
    }
    if column_count is not None:
        payload["column_count"] = column_count
    if row_count is not None:
        payload["row_count"] = row_count
    return payload


def file_sample_emptiness(
    source_id: str,
    sources: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, int] | None:
    """Return column/row counts when a file sample is empty or unreadable."""
    match = next((row for row in sources if row.get("source_id") == source_id), None)
    if match is None or match.get("kind") != "file":
        return None
    try:
        sample = read_file_sample(
            upload_dir=settings.upload_dir,
            n_rows=settings.sample_n_rows,
            max_bytes=settings.max_upload_bytes,
            source_id=source_id,
        )
    except Exception:
        return {"column_count": 0, "row_count": 0}
    if not sample_is_empty(sample):
        return None
    columns = sample.get("columns") or []
    row_count = sample.get("row_count")
    return {
        "column_count": len(columns),
        "row_count": int(row_count) if isinstance(row_count, int) else 0,
    }


def with_cleared_empty_ids(existing: list[str], chosen: list[str]) -> list[str]:
    """Copy `existing` and append new ids. Does not mutate the caller's list."""
    merged = list(existing)
    for item in chosen:
        if item not in merged:
            merged.append(item)
    return merged


def apply_confirm_decision(
    decision: object,
    sources: list[dict[str, Any]],
    selected: list[str],
    *,
    cleared_empty_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Turn a resume value into state updates. Do not guess a source id."""
    if not isinstance(decision, dict):
        return {
            "error": "Source confirmation decision is invalid.",
            "source_ids": [],
            "pending_interrupt": None,
        }
    action = decision.get("action")
    if action == ACTION_ABORT:
        return {
            "error": "Source confirmation was aborted.",
            "source_ids": [],
            "pending_interrupt": None,
        }
    offered = decision.get("source_ids")
    offered_ids = offered if isinstance(offered, list) else []
    if action == ACTION_SELECT:
        chosen = valid_source_ids([str(item) for item in offered_ids], sources)
        if not chosen:
            return {
                "error": "Select a registered source.",
                "source_ids": [],
                "pending_interrupt": None,
            }
        return {
            "source_ids": chosen,
            "pending_interrupt": None,
            "error": None,
        }
    if action == ACTION_CONFIRM:
        chosen = valid_source_ids(
            [str(item) for item in offered_ids] or selected, sources
        )
        if not sources:
            return {
                "error": "No data sources available.",
                "source_ids": [],
                "pending_interrupt": None,
            }
        if not chosen and len(sources) == 1:
            chosen = [str(sources[0]["source_id"])]
        if not chosen:
            return {
                "error": "Select a registered source.",
                "source_ids": [],
                "pending_interrupt": None,
            }
        return {
            "source_ids": chosen,
            "cleared_empty_source_ids": with_cleared_empty_ids(
                list(cleared_empty_source_ids or []), chosen
            ),
            "pending_interrupt": None,
            "error": None,
        }
    return {
        "error": "Unknown source confirmation action.",
        "source_ids": [],
        "pending_interrupt": None,
    }


def list_confirm_sources(
    settings: Settings, *, include_postgres: bool
) -> list[dict[str, Any]]:
    """List sources the confirm node may offer. Password is never included."""
    listed = list_available_sources(
        settings.upload_dir,
        settings=settings,
        include_postgres=include_postgres,
    )
    return list(listed["sources"])
