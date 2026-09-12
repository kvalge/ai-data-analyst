# profile_steps.py

"""Guided schema / DQ / EDA pauses. No UI imports; no dataset rows."""

from __future__ import annotations

from typing import Any

from src.agent.state import HITL_MODE_GUIDED

KIND_PROFILE_STEP = "profile_step"

SECTION_SCHEMA = "schema"
SECTION_DQ = "dq"
SECTION_EDA = "eda"
PROFILE_SECTIONS = (SECTION_SCHEMA, SECTION_DQ, SECTION_EDA)

ACTION_CONTINUE = "continue"
ACTION_SKIP_REMAINING = "skip_remaining"
ACTION_ABORT = "abort"


def should_pause_profiling(hitl_mode: str) -> bool:
    """Guided pauses after schema, DQ, and EDA. Standard and Auto do not."""
    return hitl_mode == HITL_MODE_GUIDED


def visible_profile_sections(step: str) -> tuple[str, ...]:
    """Sections to show through this pause (earlier ones stay visible)."""
    if step not in PROFILE_SECTIONS:
        step = PROFILE_SECTIONS[0]
    return PROFILE_SECTIONS[: PROFILE_SECTIONS.index(step) + 1]


def with_cleared_profile_ids(existing: list[str], chosen: list[str]) -> list[str]:
    """Copy `existing` and append new ids. Does not mutate the caller's list."""
    merged = list(existing)
    for item in chosen:
        if item not in merged:
            merged.append(item)
    return merged


def build_profile_interrupt(step: str, summary: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe interrupt value. Compact section only; never dataset rows."""
    if step not in PROFILE_SECTIONS:
        raise ValueError(f"Unknown profile step: {step}.")
    section = summary.get(step)
    if not isinstance(section, dict):
        raise ValueError(f"Profile summary is missing {step}.")
    sample_row_count = summary.get("sample_row_count")
    return {
        "kind": KIND_PROFILE_STEP,
        "step": step,
        "source_id": str(summary.get("source_id") or ""),
        "cached": bool(summary.get("cached")),
        "sample_row_count": (
            int(sample_row_count) if isinstance(sample_row_count, int) else 0
        ),
        "section": section,
    }


def apply_profile_decision(
    decision: object,
    *,
    source_id: str,
    cleared_profile_source_ids: list[str],
) -> dict[str, Any]:
    """Turn a resume value into state updates. Do not invent a next step."""
    if not isinstance(decision, dict):
        return {
            "error": "Profiling decision is invalid.",
            "pending_interrupt": None,
        }
    action = decision.get("action")
    if action == ACTION_ABORT:
        return {
            "error": "Profiling was aborted.",
            "pending_interrupt": None,
        }
    if action == ACTION_SKIP_REMAINING:
        return {
            "cleared_profile_source_ids": with_cleared_profile_ids(
                list(cleared_profile_source_ids), [source_id]
            ),
            "pending_interrupt": None,
            "error": None,
        }
    if action == ACTION_CONTINUE:
        return {"pending_interrupt": None, "error": None}
    return {
        "error": "Unknown profiling action.",
        "pending_interrupt": None,
    }
