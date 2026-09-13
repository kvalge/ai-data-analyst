# test_artifacts.py

"""Tests for loading table and chart artifacts from ARTIFACT_DIR."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.artifacts import (
    ArtifactRenderError,
    is_chart_artifact,
    is_table_artifact,
    load_table_artifact,
    resolve_chart_artifact,
)


def test_is_table_artifact_by_suffix():
    """csv and parquet are tables; png is not."""
    assert is_table_artifact("summary.csv")
    assert is_table_artifact("out.parquet")
    assert not is_table_artifact("chart.png")


def test_is_chart_artifact_by_suffix():
    """png is a chart; csv is not."""
    assert is_chart_artifact("chart.png")
    assert not is_chart_artifact("summary.csv")


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
    with pytest.raises(ArtifactRenderError, match="inside ARTIFACT_DIR"):
        load_table_artifact(outside, artifact_dir=artifact_dir)


def test_load_table_artifact_rejects_missing_artifact_dir(tmp_path: Path):
    """A moved or unset ARTIFACT_DIR fails as ArtifactRenderError, not a crash."""
    missing_dir = tmp_path / "gone"
    dest = missing_dir / "summary.csv"
    with pytest.raises(ArtifactRenderError, match="not a directory"):
        load_table_artifact(dest, artifact_dir=missing_dir)


def test_load_table_artifact_rejects_missing_file(tmp_path: Path):
    """A contained path that is not on disk fails closed."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    missing = artifact_dir / "summary.csv"
    with pytest.raises(ArtifactRenderError, match="not found"):
        load_table_artifact(missing, artifact_dir=artifact_dir)


def test_load_table_artifact_rejects_png(tmp_path: Path):
    """A chart path is not treated as a table."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dest = artifact_dir / "chart.png"
    dest.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(ArtifactRenderError, match="Not a table"):
        load_table_artifact(dest, artifact_dir=artifact_dir)


def test_resolve_chart_artifact_accepts_png(tmp_path: Path):
    """A png under ARTIFACT_DIR is accepted for display."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dest = artifact_dir / "chart.png"
    dest.write_bytes(b"\x89PNG\r\n")
    resolved = resolve_chart_artifact(dest, artifact_dir=artifact_dir)
    assert resolved == dest.resolve()


def test_resolve_chart_artifact_rejects_path_outside_dir(tmp_path: Path):
    """A path outside ARTIFACT_DIR is not shown as a chart."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"\x89PNG\r\n")
    with pytest.raises(ArtifactRenderError, match="inside ARTIFACT_DIR"):
        resolve_chart_artifact(outside, artifact_dir=artifact_dir)


def test_resolve_chart_artifact_rejects_missing_file(tmp_path: Path):
    """A contained chart path that is not on disk fails closed."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    missing = artifact_dir / "chart.png"
    with pytest.raises(ArtifactRenderError, match="not found"):
        resolve_chart_artifact(missing, artifact_dir=artifact_dir)


def test_resolve_chart_artifact_rejects_csv(tmp_path: Path):
    """A table path is not treated as a chart."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    dest = artifact_dir / "summary.csv"
    dest.write_text("date,region,revenue\n", encoding="utf-8")
    with pytest.raises(ArtifactRenderError, match="Not a chart"):
        resolve_chart_artifact(dest, artifact_dir=artifact_dir)
