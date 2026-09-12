# test_streamlit_logging.py

"""Check Streamlit import vs our root logging setup."""

import logging
import sys

import pytest

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


def _console_handlers() -> list[logging.Handler]:
    console = (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__)
    found: list[logging.Handler] = []
    for handler in logging.getLogger().handlers:
        if not isinstance(handler, logging.StreamHandler):
            continue
        if isinstance(handler, logging.FileHandler):
            continue
        if getattr(handler, "stream", None) in console:
            found.append(handler)
    return found


def test_streamlit_import_then_configure_logging_one_tagged_console():
    """After Streamlit import, configure_logging leaves one tagged console handler."""
    import streamlit  # noqa: F401

    configure_logging("INFO")
    console = _console_handlers()
    tagged = [h for h in console if getattr(h, "name", "") == "ai_data_analyst"]
    foreign = [h for h in console if getattr(h, "name", "") != "ai_data_analyst"]
    assert len(tagged) == 1
    assert foreign == []
