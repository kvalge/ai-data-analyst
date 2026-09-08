# logging_setup.py

"""Configure application logging from a level name (typically LOG_LEVEL)."""

from __future__ import annotations

import logging
import sys

from src.config import DEFAULT_LOG_LEVEL, SettingsError

_HANDLER_NAME = "ai_data_analyst"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> int:
    """Set the root logger to `level` (default INFO). Return the numeric level.

    Other console StreamHandlers (including subclasses, e.g. Streamlit) are
    removed so lines are not printed twice. FileHandlers and pytest caplog
    stay.
    """
    numeric = parse_log_level(level or DEFAULT_LOG_LEVEL)
    root = logging.getLogger()
    root.setLevel(numeric)
    _remove_other_stream_handlers(root)

    handler = _ensure_stream_handler(root)
    handler.setLevel(numeric)
    return numeric


def parse_log_level(level: str) -> int:
    """Parse a logging level name such as INFO. Raise if it is not valid."""
    name = level.strip().upper()
    numeric = logging.getLevelNamesMapping().get(name)
    if numeric is None:
        raise SettingsError(
            f"Invalid log level {level!r}. Use a name like DEBUG, INFO, or WARNING."
        )
    return numeric


def _is_foreign_console_handler(handler: logging.Handler) -> bool:
    """True for untagged handlers that write to stderr/stdout.

    Matches StreamHandler subclasses (Streamlit), but not FileHandler or
    pytest caplog (StringIO). A strict ``type is StreamHandler`` check
    would miss those subclasses and leave duplicate console output.
    """
    if getattr(handler, "name", "") == _HANDLER_NAME:
        return False
    if not isinstance(handler, logging.StreamHandler):
        return False
    if isinstance(handler, logging.FileHandler):
        return False
    stream = getattr(handler, "stream", None)
    return stream in _current_console_streams()


def _current_console_streams() -> tuple[object, ...]:
    """stderr/stdout at call time, not import time (pytest/Streamlit reassign them)."""
    return (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__)


def _remove_other_stream_handlers(root: logging.Logger) -> None:
    """Drop untagged console StreamHandlers that would duplicate output."""
    for handler in list(root.handlers):
        if _is_foreign_console_handler(handler):
            root.removeHandler(handler)


def _ensure_stream_handler(root: logging.Logger) -> logging.Handler:
    """Reuse our stream handler if present so repeated setup stays idempotent."""
    for handler in root.handlers:
        if getattr(handler, "name", "") == _HANDLER_NAME:
            return handler

    handler = logging.StreamHandler()
    handler.name = _HANDLER_NAME
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    return handler
