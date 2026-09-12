# sandbox.py

"""Run a Python file in a subprocess. Never exec/eval in this process."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.execution.paths import path_is_inside

_LOG = logging.getLogger(__name__)


class SandboxError(Exception):
    """The sandbox runner could not start or finish the child."""


class SandboxTimeout(SandboxError):
    """The child exceeded the timeout and was killed."""


@dataclass(frozen=True)
class SandboxResult:
    """Captured child output. Non-zero exit is still a completed run."""

    stdout: str
    stderr: str
    exit_code: int


def run_python_file(
    script_path: Path,
    work_dir: Path,
    *,
    timeout_s: int,
) -> SandboxResult:
    """Run `script_path` with cwd=`work_dir`. Kill the child if it exceeds `timeout_s`.

    `timeout_s` is `SANDBOX_TIMEOUT_S` from settings at the call site.
    The file is executed by a child interpreter, not by exec or eval here.
    After resolve, the file must sit inside `work_dir` (symlinks and `..`
    included). Containment is checked before `is_file`, so a disallowed
    path is rejected without an existence probe. cwd alone does not
    decide which file is executed.
    """
    if timeout_s <= 0:
        raise SandboxError("timeout_s must be positive.")
    cwd = Path(work_dir).resolve()
    if not cwd.is_dir():
        raise SandboxError(f"Sandbox work dir is not a directory: {cwd}")
    script = Path(script_path).resolve()
    if not path_is_inside(script, cwd):
        raise SandboxError(
            "Python file must resolve inside the sandbox work dir."
        )
    if not script.is_file():
        raise SandboxError(f"Python file does not exist: {script}")

    _LOG.info("sandbox run script=%s timeout_s=%s", script.name, timeout_s)
    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _LOG.warning("sandbox timeout script=%s timeout_s=%s", script.name, timeout_s)
        raise SandboxTimeout(
            f"Sandbox exceeded {timeout_s}s and was killed."
        ) from exc

    return SandboxResult(
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        exit_code=int(completed.returncode),
    )
