# context.py

"""Store business-context documents in CONTEXT_DIR. Retrieve is a separate tool."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from src.validation.context_files import CONTEXT_FILE_SUFFIXES, validate_context_file
from src.validation.uploads import FileValidationError

_LOG = logging.getLogger(__name__)

_SAVE_FAILED = (
    "Could not save the file. Check disk space and that the upload folder is writable."
)


def list_context_files(context_dir: Path) -> list[Path]:
    """Return allowlisted context files in `context_dir` (flat, names only)."""
    if not context_dir.is_dir():
        return []
    return sorted(
        path
        for path in context_dir.iterdir()
        if path.is_file() and path.suffix.lower() in CONTEXT_FILE_SUFFIXES
    )


def ingest_context_upload(
    payload: bytes,
    original_name: str,
    context_dir: Path,
    *,
    max_bytes: int,
) -> Path:
    """Write `payload` to a temp file, validate, then copy into `context_dir`.

    A later upload with the same basename overwrites the stored file.
    """
    stored_name = Path(original_name).name
    if stored_name in {"", ".", ".."}:
        raise FileValidationError("Context file name is missing.")
    suffix = Path(stored_name).suffix.lower() or ".upload"
    handle = None
    tmp_path: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(handle.name)
        handle.write(payload)
        handle.close()
        validate_context_file(tmp_path, max_bytes=max_bytes)
        context_dir.mkdir(parents=True, exist_ok=True)
        stored_path = context_dir / stored_name
        # Same basename replaces the previous document (plan: overwrite, not rename/reject).
        if stored_path.exists():
            _LOG.info("Replacing context file %s", stored_name)
        shutil.copyfile(tmp_path, stored_path)
        return stored_path
    except FileValidationError:
        raise
    except OSError as exc:
        _LOG.exception("Failed to persist uploaded context file")
        raise FileValidationError(_SAVE_FAILED) from exc
    finally:
        if handle is not None:
            handle.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
