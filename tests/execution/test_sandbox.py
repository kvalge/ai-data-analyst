# test_sandbox.py

"""Tests for the subprocess Python runner. No exec/eval in the app process."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.execution.sandbox import SandboxError, SandboxTimeout, run_python_file


def test_child_prints_hello(tmp_path: Path):
    """A child that prints hello returns that stdout and exit 0."""
    script = tmp_path / "hello.py"
    script.write_text("print('hello')\n", encoding="utf-8")
    result = run_python_file(script, tmp_path, timeout_s=10)
    assert result.stdout.strip() == "hello"
    assert result.stderr == ""
    assert result.exit_code == 0


def test_timeout_kills_a_sleeper(tmp_path: Path):
    """A child that sleeps past the timeout is killed instead of finishing."""
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    started = time.monotonic()
    with pytest.raises(SandboxTimeout, match="killed"):
        run_python_file(script, tmp_path, timeout_s=1)
    assert time.monotonic() - started < 10


def test_child_matplotlib_backend_is_headless(tmp_path: Path):
    """Charts save headless, so a generated plt.show() cannot block the run."""
    script = tmp_path / "backend.py"
    script.write_text(
        "import os\nprint(os.environ['MPLBACKEND'])\n",
        encoding="utf-8",
    )
    result = run_python_file(script, tmp_path, timeout_s=10)
    assert result.stdout.strip() == "Agg"
    assert result.exit_code == 0


def test_child_cwd_is_work_dir(tmp_path: Path):
    """The child process current directory is the sandbox work dir."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    script = work_dir / "cwd.py"
    script.write_text(
        "from pathlib import Path\nprint(Path.cwd())\n",
        encoding="utf-8",
    )
    result = run_python_file(script, work_dir, timeout_s=10)
    assert Path(result.stdout.strip()) == work_dir.resolve()
    assert result.exit_code == 0


def test_script_outside_work_dir_is_rejected(tmp_path: Path):
    """A path that resolves outside work_dir is not executed."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('hello')\n", encoding="utf-8")
    sneaky = work_dir / ".." / "outside.py"
    with pytest.raises(SandboxError, match="inside"):
        run_python_file(sneaky, work_dir, timeout_s=10)
    missing = work_dir / ".." / "no-such.py"
    with pytest.raises(SandboxError, match="inside"):
        run_python_file(missing, work_dir, timeout_s=10)
