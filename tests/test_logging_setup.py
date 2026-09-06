"""Tests for root logging configuration."""

import io
import logging
import sys

import pytest

from src.config import SettingsError
from src.logging_setup import configure_logging


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """Keep root level and handlers from leaking into other tests."""
    root = logging.getLogger()
    previous_level = root.level
    previous_handlers = root.handlers[:]
    yield
    root.setLevel(previous_level)
    root.handlers[:] = previous_handlers


def test_logger_emits_at_configured_level(caplog):
    """INFO is recorded when the level is INFO; DEBUG is not."""
    configure_logging("INFO")
    logger = logging.getLogger("src.logging_setup")

    logger.info("visible-info")
    logger.debug("hidden-debug")

    assert "visible-info" in caplog.text
    assert "hidden-debug" not in caplog.text


def test_debug_emitted_when_level_is_debug(caplog):
    """DEBUG is recorded after the level is lowered."""
    configure_logging("DEBUG")
    logger = logging.getLogger("src.logging_setup")

    logger.debug("visible-debug")

    assert "visible-debug" in caplog.text


def test_invalid_log_level_names_the_value():
    """A bad level name appears in the error; the function is not LOG_LEVEL-specific."""
    with pytest.raises(SettingsError, match="Invalid log level") as exc_info:
        configure_logging("not-a-level")

    assert "not-a-level" in str(exc_info.value)
    assert "LOG_LEVEL" not in str(exc_info.value)


class _StreamlitLikeHandler(logging.StreamHandler):
    """Stand-in for a library that attaches a StreamHandler subclass."""


def test_replaces_other_root_stream_handlers(tmp_path):
    """Plain and subclass console handlers go; FileHandler and others stay."""
    root = logging.getLogger()
    extra = logging.StreamHandler()
    extra.name = "streamlit-plain"
    subclassed = _StreamlitLikeHandler()
    subclassed.name = "streamlit-subclass"
    dummy = logging.Handler()
    file_handler = logging.FileHandler(tmp_path / "app.log", encoding="utf-8")
    root.addHandler(extra)
    root.addHandler(subclassed)
    root.addHandler(dummy)
    root.addHandler(file_handler)
    try:
        configure_logging("INFO")
        tagged = [
            handler
            for handler in root.handlers
            if getattr(handler, "name", "") == "ai_data_analyst"
        ]
        assert extra not in root.handlers
        assert subclassed not in root.handlers
        assert dummy in root.handlers
        assert file_handler in root.handlers
        assert len(tagged) == 1
    finally:
        root.removeHandler(extra)
        root.removeHandler(subclassed)
        root.removeHandler(dummy)
        root.removeHandler(file_handler)
        file_handler.close()


def test_removes_handler_bound_to_reassigned_stdout():
    """A handler on a post-import sys.stdout replacement is still removed."""
    root = logging.getLogger()
    original_stdout = sys.stdout
    extra: logging.Handler | None = None
    try:
        sys.stdout = io.StringIO()
        extra = logging.StreamHandler(sys.stdout)
        extra.name = "reassigned-stdout"
        root.addHandler(extra)
        configure_logging("INFO")
        assert extra not in root.handlers
    finally:
        sys.stdout = original_stdout
        if extra is not None:
            root.removeHandler(extra)
