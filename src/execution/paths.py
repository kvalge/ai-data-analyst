# paths.py

"""Path containment for the sandbox work dir and ARTIFACT_DIR."""

from __future__ import annotations

from pathlib import Path


def path_is_inside(path: Path, root: Path) -> bool:
    """True when resolved `path` is resolved `root` or a descendant."""
    return path.resolve().is_relative_to(root.resolve())
