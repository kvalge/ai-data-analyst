"""Create local runtime directories used by the app."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings


def ensure_runtime_dirs(
    settings: Settings, *, log_dir: Path | None = None
) -> list[Path]:
    """Create upload, context, cache, artifact, checkpoint, and log dirs.

    Checkpoint is a file path in settings, so its parent directory is created.
    `log_dir` defaults to a `logs` sibling of `upload_dir` (with defaults:
    `data/logs`). Safe to call more than once.
    """
    logs = log_dir if log_dir is not None else settings.upload_dir.parent / "logs"
    directories = list(
        dict.fromkeys(
            [
                settings.upload_dir,
                settings.context_dir,
                settings.cache_dir,
                settings.artifact_dir,
                settings.checkpoint_path.parent,
                logs,
            ]
        )
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories
