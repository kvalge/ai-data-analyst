# test_artifacts.py

"""Tests for loading table artifacts from ARTIFACT_DIR."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.artifacts import (
    ArtifactRenderError,
    is_table_artifact,
    load_table_artifact,
)


def test_is_table_artifact_by_suffix():
    """csv and parquet are tables; png is not."""
    assert is_table_artifact("summary.csv")
    assert is_table_artifact("out.parquet")
    assert not is_table_artifact("chart.png")


def test_load_table_artifact_reads_known_csv(
    tmp_path: Path, sample_sales_csv: Path
):
    """A csv under ARTIFACT_DIR loads as a table, not as a raw dump."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dest = artifact_dir / "summary.csv"
    dest.write_bytes(sample_sales_csv.read_bytes())
    frame = load_table_artifact(dest, artifact_dir=artifact_dir)
    assert list(frame.columns) == ["date", "region", "revenue"]
    assert len(frame) == 5


def test_load_table_artifact_rejects_path_outside_dir(tmp_path: Path):
    """A path outside ARTIFACT_DIR is not read."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside = tmp_path / "secret.csv"
    outside.write_text("date,region,revenue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        load_table_artifact(outside, artifact_dir=artifact_dir)


def test_load_table_artifact_rejects_missing_file(tmp_path: Path):
    """A contained path that is not on disk fails closed."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    missing = artifact_dir / "summary.csv"
    with pytest.raises(ArtifactRenderError, match="not found"):
        load_table_artifact(missing, artifact_dir=artifact_dir)


def test_load_table_artifact_rejects_png(tmp_path: Path):
    """A chart path is not treated as a table. PNG display is 7.2."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dest = artifact_dir / "chart.png"
    dest.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(ArtifactRenderError, match="Not a table"):
        load_table_artifact(dest, artifact_dir=artifact_dir)
