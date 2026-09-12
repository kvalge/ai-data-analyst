# test_workspace.py

"""Tests for per-run work dirs and artifact copy into ARTIFACT_DIR."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.agent.state import as_artifact_path
from src.execution.sandbox import run_python_file
from src.execution.workspace import copy_artifacts, sandbox_work_dir


def test_sandbox_work_dir_is_removed_after_context(tmp_path: Path):
    """The per-run work dir does not remain after the context exits."""
    with sandbox_work_dir(parent=tmp_path) as work:
        (work / "marker.txt").write_text("ok", encoding="utf-8")
        assert work.is_dir()
        kept = work
    assert not kept.exists()


def test_file_inside_work_dir_is_copied_to_artifact_dir(tmp_path: Path):
    """A file written in the work dir is copied under ARTIFACT_DIR."""
    artifact_dir = tmp_path / "artifacts"
    with sandbox_work_dir(parent=tmp_path) as work:
        (work / "ok.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        copied = copy_artifacts(work, artifact_dir)
    assert len(copied) == 1
    dest = copied[0]
    assert dest.read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert as_artifact_path(dest, artifact_dir=artifact_dir) == str(dest.resolve())


def test_write_outside_work_dir_is_not_copied(tmp_path: Path):
    """A child write outside the work dir is not an artifact."""
    artifact_dir = tmp_path / "artifacts"
    with sandbox_work_dir(parent=tmp_path) as work:
        script = work / "write.py"
        script.write_text(
            "from pathlib import Path\n"
            "Path('inside.csv').write_text('ok\\n', encoding='utf-8')\n"
            "Path('..').joinpath('outside.csv').write_text("
            "'nope\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        result = run_python_file(script, work, timeout_s=10)
        assert result.exit_code == 0
        copied = copy_artifacts(work, artifact_dir, exclude=(script,))
        leaked = work.parent / "outside.csv"
        assert leaked.is_file()
        assert leaked.read_text(encoding="utf-8") == "nope\n"
        assert [path.name for path in copied] == ["inside.csv"]
        assert not any(path.name == "outside.csv" for path in artifact_dir.rglob("*"))


def test_skipped_symlink_is_logged_at_debug(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """A symlink is not copied; debug explains the skip."""
    artifact_dir = tmp_path / "artifacts"
    real_is_symlink = Path.is_symlink

    def fake_is_symlink(self: Path) -> bool:
        if self.name == "link.csv":
            return True
        return real_is_symlink(self)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    with sandbox_work_dir(parent=tmp_path) as work:
        (work / "real.csv").write_text("ok\n", encoding="utf-8")
        (work / "link.csv").write_text("nope\n", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger="src.execution.workspace"):
            copied = copy_artifacts(work, artifact_dir)
        assert [path.name for path in copied] == ["real.csv"]
        assert "skipped symlink/out-of-bounds artifact:" in caplog.text
        assert "link.csv" in caplog.text


def test_skipped_out_of_bounds_is_logged_at_debug(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """A path that resolves outside the work dir is skipped and logged."""
    artifact_dir = tmp_path / "artifacts"
    real_resolve = Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self.name == "escape.csv":
            return real_resolve(tmp_path / "outside-of-work.csv")
        return real_resolve(self, strict)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with sandbox_work_dir(parent=tmp_path) as work:
        (work / "real.csv").write_text("ok\n", encoding="utf-8")
        (work / "escape.csv").write_text("nope\n", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger="src.execution.workspace"):
            copied = copy_artifacts(work, artifact_dir)
        assert [path.name for path in copied] == ["real.csv"]
        assert "skipped symlink/out-of-bounds artifact:" in caplog.text
        assert "escape.csv" in caplog.text
