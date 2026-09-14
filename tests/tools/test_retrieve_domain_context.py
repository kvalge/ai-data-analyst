# test_retrieve_domain_context.py

"""Tests for the retrieve_domain_context tool. Data files stay out of RAG."""

from __future__ import annotations

from pathlib import Path

from src.config import DEFAULT_MAX_UPLOAD_BYTES, DEFAULT_RAG_TOP_K, load_settings
from src.storage.registry import save_file_source
from src.tools.retrieve_domain_context import (
    RETRIEVE_DOMAIN_CONTEXT,
    retrieve_domain_context,
)
from src.agent.execute import run_allowlisted_tool, validate_tool_call
from tests.tool_schema import assert_keys_match_required

_MAX = DEFAULT_MAX_UPLOAD_BYTES
_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}


def _write_glossary(context_dir: Path, sample_glossary_md: Path) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    (context_dir / "sample_glossary.md").write_bytes(sample_glossary_md.read_bytes())


def test_retrieve_domain_context_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    contract = RETRIEVE_DOMAIN_CONTEXT
    assert contract.name == "retrieve_domain_context"
    assert contract.description
    assert contract.input_schema["required"] == ["query"]
    assert contract.result_schema["required"] == ["chunks"]


def test_retrieve_hits_glossary_and_skips_data_files(
    tmp_path: Path, sample_glossary_md: Path, sample_sales_csv: Path
):
    """A revenue query hits the glossary. CSV rows are not in the corpus."""
    context_dir = tmp_path / "context"
    upload_dir = tmp_path / "uploads"
    _write_glossary(context_dir, sample_glossary_md)
    (context_dir / "sales.csv").write_bytes(sample_sales_csv.read_bytes())
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, upload_dir, original_name="sales.csv")

    result = retrieve_domain_context(
        "What is revenue?",
        context_dir=context_dir,
        cache_dir=tmp_path / "cache",
        max_bytes=_MAX,
        top_k=DEFAULT_RAG_TOP_K,
    )

    assert result["chunks"]
    assert all(chunk["source_file"] != "sales.csv" for chunk in result["chunks"])
    assert "Revenue is net of returns." in result["chunks"][0]["text"]
    assert "date,region,revenue" not in str(result)
    assert_keys_match_required(
        result["chunks"][0],
        RETRIEVE_DOMAIN_CONTEXT.result_schema["properties"]["chunks"]["items"],
    )


def test_run_allowlisted_retrieve_uses_context_dir(
    tmp_path: Path, sample_glossary_md: Path, sample_sales_csv: Path
):
    """The graph injects CONTEXT_DIR; upload CSVs are not searched."""
    context_dir = tmp_path / "context"
    upload_dir = tmp_path / "uploads"
    _write_glossary(context_dir, sample_glossary_md)
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    save_file_source(incoming, upload_dir, original_name="sales.csv")
    settings = load_settings(
        environ={
            "OLLAMA_HOST": "http://ollama.test:11434",
            "UPLOAD_DIR": str(upload_dir),
            "CONTEXT_DIR": str(context_dir),
            **_PLACEHOLDER_MODELS,
        },
        load_dotenv_file=False,
        project_root=tmp_path,
    )

    result = run_allowlisted_tool(
        "retrieve_domain_context",
        {"query": "What is revenue?"},
        settings,
    )

    assert "Revenue is net of returns." in result["chunks"][0]["text"]
    assert "date,region,revenue" not in str(result)


def test_validate_tool_call_strips_context_dir():
    """context_dir from the model is dropped before the handler runs."""
    checked = validate_tool_call(
        "retrieve_domain_context",
        {"query": "What is revenue?", "context_dir": "/evil"},
    )
    assert checked["arguments"] == {"query": "What is revenue?"}
