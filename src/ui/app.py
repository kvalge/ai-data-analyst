# app.py
# streamlit run src/ui/app.py

"""Streamlit shell: data-file upload, source list, no agent yet."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# src/ui/app.py → repository root. Streamlit puts this file's dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.config import load_settings
from src.logging_setup import configure_logging
from src.storage.ingest import ingest_data_upload
from src.storage.paths import ensure_runtime_dirs
from src.storage.registry import RegistryError
from src.tools.list_sources import list_available_sources
from src.validation.data_files import DATA_FILE_SUFFIXES, FileValidationError

settings = load_settings(require_models=False)
configure_logging(settings.log_level)
ensure_runtime_dirs(settings)
logging.getLogger(__name__).info("UI started")

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")

uploader_types = sorted(suffix.lstrip(".") for suffix in DATA_FILE_SUFFIXES)

with st.sidebar:
    st.header("Session")
    uploaded = st.file_uploader("Data file", type=uploader_types)
    if uploaded is not None:
        payload = uploaded.getvalue()
        token = f"{uploaded.name}:{len(payload)}"
        if st.session_state.get("ingested_data_token") != token:
            try:
                ingest_data_upload(
                    payload,
                    uploaded.name,
                    settings.upload_dir,
                    max_bytes=settings.max_upload_bytes,
                )
                st.session_state.ingested_data_token = token
                st.session_state.data_upload_error = None
            except (FileValidationError, RegistryError) as exc:
                st.session_state.data_upload_error = str(exc)
        if st.session_state.get("data_upload_error"):
            st.error(st.session_state.data_upload_error)

    st.subheader("Data sources")
    listed = list_available_sources(settings.upload_dir)
    if not listed["sources"]:
        st.caption("No data files yet.")
    for row in listed["sources"]:
        st.write(f"{row['original_name']} (`{row['source_id'][:20]}…`)")

st.info("Ask questions here after you add a data source.")
