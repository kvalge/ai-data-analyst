# artifacts.py

"""Render sandbox table and chart artifacts in the UI. Paths only."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.state import as_artifact_path

_LOG = logging.getLogger(__name__)

TABLE_SUFFIXES = frozenset({".csv", ".parquet"})
CHART_SUFFIXES = frozenset({".png"})


class ArtifactRenderError(ValueError):
    """A table or chart artifact could not be shown."""


def is_table_artifact(path: str | Path) -> bool:
    """True when the path suffix is csv or parquet. Does not read the file."""
    return Path(path).suffix.lower() in TABLE_SUFFIXES


def is_chart_artifact(path: str | Path) -> bool:
    """True when the path suffix is png. Does not read the file."""
    return Path(path).suffix.lower() in CHART_SUFFIXES


def _existing_artifact(path: str | Path, *, artifact_dir: Path) -> Path:
    """Resolve `path` inside ARTIFACT_DIR and require the file to exist.

    Check the root exists before resolving `path`. Other TypeError/ValueError
    from `as_artifact_path` become ArtifactRenderError so a stale checkpoint
    path cannot crash a Streamlit rerun.
    """
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ArtifactRenderError("artifact_dir is not a directory.")
    try:
        resolved = Path(as_artifact_path(path, artifact_dir=root))
    except (TypeError, ValueError) as exc:
        raise ArtifactRenderError(str(exc)) from exc
    if not resolved.is_file():
        raise ArtifactRenderError(f"Artifact not found: {resolved.name}.")
    return resolved


def load_table_artifact(path: str | Path, *, artifact_dir: Path) -> pd.DataFrame:
    """Return a DataFrame for a csv/parquet path inside `artifact_dir`.

    Does not log row values. The path must resolve inside ARTIFACT_DIR.
    """
    resolved = _existing_artifact(path, artifact_dir=artifact_dir)
    if not is_table_artifact(resolved):
        raise ArtifactRenderError(
            f"Not a table artifact: {resolved.name}."
        )
    suffix = resolved.suffix.lower()
    _LOG.info("load table artifact name=%s suffix=%s", resolved.name, suffix)
    try:
        if suffix == ".csv":
            return pd.read_csv(resolved)
        return pd.read_parquet(resolved)
    except (OSError, ValueError, ImportError, UnicodeError) as exc:
        raise ArtifactRenderError(
            f"Could not read table artifact {resolved.name}."
        ) from exc


def render_table_artifacts(
    paths: Sequence[str], *, artifact_dir: Path
) -> None:
    """Show csv/parquet artifacts with `st.dataframe`. Skip other suffixes."""
    tables = [
        item
        for item in paths
        if isinstance(item, str) and is_table_artifact(item)
    ]
    if not tables:
        return
    st.subheader("Tables")
    for item in tables:
        try:
            frame = load_table_artifact(item, artifact_dir=artifact_dir)
        except ArtifactRenderError as exc:
            st.warning(str(exc))
            continue
        st.caption(Path(item).name)
        st.dataframe(frame, hide_index=True)


def resolve_chart_artifact(path: str | Path, *, artifact_dir: Path) -> Path:
    """Return a png path that resolves inside `artifact_dir` and exists."""
    resolved = _existing_artifact(path, artifact_dir=artifact_dir)
    if not is_chart_artifact(resolved):
        raise ArtifactRenderError(
            f"Not a chart artifact: {resolved.name}."
        )
    _LOG.info("load chart artifact name=%s", resolved.name)
    return resolved


def render_chart_artifacts(
    paths: Sequence[str], *, artifact_dir: Path
) -> None:
    """Show png artifacts with `st.image`. Skip other suffixes."""
    charts = [
        item
        for item in paths
        if isinstance(item, str) and is_chart_artifact(item)
    ]
    if not charts:
        return
    st.subheader("Charts")
    for item in charts:
        try:
            resolved = resolve_chart_artifact(item, artifact_dir=artifact_dir)
        except ArtifactRenderError as exc:
            st.warning(str(exc))
            continue
        st.caption(resolved.name)
        st.image(str(resolved))
