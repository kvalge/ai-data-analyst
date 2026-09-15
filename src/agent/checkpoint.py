# checkpoint.py

"""Sqlite checkpointer and the current thread id beside CHECKPOINT_PATH."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS

_LOG = logging.getLogger(__name__)


def thread_id_path(checkpoint_path: Path) -> Path:
    """Sidecar file that remembers the current thread across process restarts."""
    return checkpoint_path.with_suffix(".thread")


def load_persisted_thread_id(checkpoint_path: Path) -> str | None:
    """Return the stored thread id, or None if it is missing or blank."""
    path = thread_id_path(checkpoint_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def persist_thread_id(checkpoint_path: Path, thread_id: str) -> None:
    """Write the current thread id next to the sqlite file."""
    path = thread_id_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(thread_id.strip() + "\n", encoding="utf-8")
    _LOG.info("checkpoint thread persisted")


def sqlite_checkpointer(checkpoint_path: Path) -> SqliteSaver:
    """Open a process-lived SqliteSaver. Do not close the connection.

    ``SqliteSaver.from_conn_string`` closes the file when the context exits.
    Streamlit keeps the compiled graph in session_state, so the connection
    must stay open. ``check_same_thread=False`` is required because a
    Streamlit rerun is a new thread, and because chat invoke runs on a
    worker thread while ``get_state`` polls from the script thread.

    Concurrent use of one ``sqlite3.Connection`` is still unsafe on its
    own. ``SqliteSaver.cursor()`` holds ``SqliteSaver.lock`` around every
    read and write, which serializes those two threads. After setup, call
    saver methods only — do not use ``saver.conn`` from more than one
    thread. WAL and ``busy_timeout`` are for *other connections* (a
    second browser tab), not for sharing this connection object.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        f"PRAGMA busy_timeout={DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS}"
    )
    saver = SqliteSaver(conn)
    saver.setup()
    _LOG.info("checkpoint sqlite path=%s", checkpoint_path)
    return saver
