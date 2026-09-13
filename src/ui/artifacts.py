# artifacts.py

"""Render sandbox table artifacts in the UI. Paths only; no LLM prompt."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.state import as_artifact_path

_LOG = logging.getLogger(__name__)

TABLE_SUFFIXES = frozenset({".csv", ".parquet"})


class ArtifactRenderError(ValueError):
    """A table artifact could not be loaded for display."""


def is_table_artifact(path: str | Path) -> bool:
    """True when the path suffix is csv or parquet. Does not read the file."""
    return Path(path).suffix.lower() in TABLE_SUFFIXES


def load_table_artifact(path: str | Path, *, artifact_dir: Path) -> pd.DataFrame:
    """Return a DataFrame for a csv/parquet path inside `artifact_dir`.

    Does not log row values. The path must resolve inside ARTIFACT_DIR.
    """
    resolved = Path(as_artifact_path(path, artifact_dir=artifact_dir))
    if not resolved.is_file():
        raise ArtifactRenderError(f"Artifact not found: {resolved.name}.")
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
        except (ArtifactRenderError, TypeError, ValueError) as exc:
            st.warning(str(exc))
            continue
        st.caption(Path(item).name)
        st.dataframe(frame, hide_index=True)
