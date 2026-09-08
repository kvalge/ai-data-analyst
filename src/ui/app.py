# app.py
# streamlit run src/ui/app.py

"""Thin Streamlit shell. No uploads or agent logic yet."""

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

settings = load_settings(require_models=False)
configure_logging(settings.log_level)
logging.getLogger(__name__).info("UI started")

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("AI Data Analyst")

with st.sidebar:
    st.header("Session")
    st.caption("HITL mode, uploads, and sources will appear here.")

st.info("Add a data source in a later step, then ask questions here.")
