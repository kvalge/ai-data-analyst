# state.py

"""LangGraph-ready agent state. Summaries and path strings only; no dataframes."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from src.execution.paths import path_is_inside

# Match src/ui/hitl.py HitlMode values. Agent code must not import the UI.
HITL_MODE_GUIDED = "Guided"
HITL_MODE_STANDARD = "Standard"
HITL_MODE_AUTO = "Auto"
ALLOWED_HITL_MODES = frozenset(
    {HITL_MODE_GUIDED, HITL_MODE_STANDARD, HITL_MODE_AUTO}
)


class AgentMessage(TypedDict):
    """One conversation turn. Content is text, not a dataset."""

    role: str
    content: str


class AgentState(TypedDict):
    """Working graph state. Checkpoints may persist this; keep it JSON-safe."""

    messages: Annotated[list[AgentMessage], operator.add]
    hitl_mode: str
    source_ids: list[str]
    cleared_empty_source_ids: list[str]
    cleared_profile_source_ids: list[str]
    profile_summary: dict[str, Any] | None
    pending_interrupt: dict[str, Any] | None
    pending_tool: dict[str, Any] | None
    last_tool_result: dict[str, Any] | None
    artifacts: list[str]
    error: str | None


def empty_agent_state(*, hitl_mode: str = HITL_MODE_STANDARD) -> AgentState:
    """Return a new state dict. Lists are not shared across calls."""
    if hitl_mode not in ALLOWED_HITL_MODES:
        raise ValueError(f"Unknown hitl_mode: {hitl_mode}.")
    return {
        "messages": [],
        "hitl_mode": hitl_mode,
        "source_ids": [],
        "cleared_empty_source_ids": [],
        "cleared_profile_source_ids": [],
        "profile_summary": None,
        "pending_interrupt": None,
        "pending_tool": None,
        "last_tool_result": None,
        "artifacts": [],
        "error": None,
    }


def as_artifact_path(path: Path | str, *, artifact_dir: Path) -> str:
    """Store an artifact as a filesystem path string, never file contents.

    After resolve, the path must sit inside `artifact_dir` (ARTIFACT_DIR).
    Containment is checked before any existence probe of `path`.
    """
    if not isinstance(path, (Path, str)):
        raise TypeError("Artifacts must be filesystem paths.")
    text = str(path).strip()
    if not text:
        raise ValueError("Artifact path is empty.")
    root = Path(artifact_dir).resolve()
    if not root.is_dir():
        raise ValueError("artifact_dir is not a directory.")
    resolved = Path(text).resolve()
    if not path_is_inside(resolved, root):
        raise ValueError("Artifact path must resolve inside ARTIFACT_DIR.")
    return str(resolved)
