# ingest.py

"""Validate an uploaded data file and persist it into UPLOAD_DIR."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.storage.registry import RegistryError, save_file_source
from src.storage.sources import DataSource
from src.validation.data_files import FileValidationError, validate_data_file

_LOG = logging.getLogger(__name__)

_SAVE_FAILED = (
    "Could not save the file. Check disk space and that the upload folder is writable."
)


def ingest_data_upload(
    payload: bytes,
    original_name: str,
    upload_dir: Path,
    *,
    max_bytes: int,
) -> DataSource:
    """Write `payload` to a temp file, validate, then save into `upload_dir`."""
    suffix = Path(original_name).suffix.lower() or ".upload"
    handle = None
    tmp_path: Path | None = None
    try:
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(handle.name)
        handle.write(payload)
        handle.close()
        validate_data_file(tmp_path, max_bytes=max_bytes)
        return save_file_source(tmp_path, upload_dir, original_name=original_name)
    except (FileValidationError, RegistryError):
        raise
    except OSError as exc:
        _LOG.exception("Failed to persist uploaded data file")
        raise FileValidationError(_SAVE_FAILED) from exc
    finally:
        if handle is not None:
            handle.close()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
