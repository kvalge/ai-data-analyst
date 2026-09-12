# workspace.py

"""Per-run sandbox work dirs and copy of artifacts into ARTIFACT_DIR."""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

from src.execution.paths import path_is_inside
from src.execution.sandbox import SandboxError

_LOG = logging.getLogger(__name__)


@contextmanager
def sandbox_work_dir(*, parent: Path | None = None) -> Generator[Path]:
    """Yield a unique work dir, then delete it.

    `parent` is for tests. Production leaves it unset (system temp).
    """
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sandbox-", dir=parent) as raw:
        yield Path(raw).resolve()


def copy_artifacts(
    work_dir: Path,
    artifact_dir: Path,
    *,
    exclude: Iterable[Path] = (),
) -> list[Path]:
    """Copy regular files from `work_dir` into a new folder under `artifact_dir`.

    Only files that resolve inside `work_dir` are copied. Writes that landed
    outside the work dir are not treated as artifacts. Symlinks are skipped.
    Destination paths resolve inside `artifact_dir`.
    """
    work = Path(work_dir).resolve()
    dest_parent = Path(artifact_dir).resolve()
    if not work.is_dir():
        raise SandboxError(f"Sandbox work dir is not a directory: {work}")
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest_root = (dest_parent / uuid.uuid4().hex).resolve()
    if not path_is_inside(dest_root, dest_parent):
        raise SandboxError("Artifact destination escaped ARTIFACT_DIR.")
    dest_root.mkdir(parents=True, exist_ok=False)

    excluded = {Path(item).resolve() for item in exclude}
    copied: list[Path] = []
    for child in work.rglob("*"):
        if child.is_symlink():
            _log_skipped(child)
            continue
        resolved = child.resolve()
        if not path_is_inside(resolved, work):
            _log_skipped(child)
            continue
        if resolved in excluded:
            continue
        if not resolved.is_file():
            continue
        relative = resolved.relative_to(work)
        dest = (dest_root / relative).resolve()
        if not path_is_inside(dest, dest_parent):
            _log_skipped(child)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)
        copied.append(dest)
    _LOG.info("sandbox copied %s artifact(s)", len(copied))
    return copied


def _log_skipped(child: Path) -> None:
    """Record a skipped candidate. Does not change copy behavior."""
    _LOG.debug("skipped symlink/out-of-bounds artifact: %s", child)
