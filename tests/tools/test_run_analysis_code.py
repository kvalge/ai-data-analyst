# test_run_analysis_code.py

"""Tests for run_analysis_code: AST check, then sandbox, then artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_settings
from src.execution.ast_check import PythonAstError
from src.storage.registry import save_file_source
from src.tools.run_analysis_code import (
    RUN_ANALYSIS_CODE,
    AnalysisCodeError,
    run_analysis_code,
)
from src.validation.data_files import FileValidationError
from tests.tool_schema import assert_keys_match_required

_SUM_CODE = """
import pandas as pd
df = pd.read_csv("sales.csv")
print(int(df["revenue"].sum()))
df.to_csv("summary.csv", index=False)
"""


def _settings(tmp_path: Path):
    return load_settings(
        environ={},
        load_dotenv_file=False,
        project_root=tmp_path,
        require_models=False,
    )


def _saved_sales(tmp_path: Path, sample_sales_csv: Path):
    settings = _settings(tmp_path)
    incoming = tmp_path / "sales.csv"
    incoming.write_bytes(sample_sales_csv.read_bytes())
    saved = save_file_source(
        incoming, settings.upload_dir, original_name="sales.csv"
    )
    return settings, saved


def _run(
    settings,
    source_id: str,
    code: str,
    *,
    max_prompt_chars: int | None = None,
):
    return run_analysis_code(
        settings=settings,
        source_id=source_id,
        code=code,
        upload_dir=settings.upload_dir,
        artifact_dir=settings.artifact_dir,
        timeout_s=settings.sandbox_timeout_s,
        max_prompt_chars=(
            settings.max_prompt_chars if max_prompt_chars is None else max_prompt_chars
        ),
    )


def test_fixture_csv_in_work_dir_writes_summary(
    tmp_path: Path, sample_sales_csv: Path
):
    """A pandas snippet reads the copied fixture and writes a csv artifact."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    result = _run(settings, saved.source_id, _SUM_CODE)
    assert "575" in result["stdout"]
    assert result["source_id"] == saved.source_id
    assert len(result["artifacts"]) == 1
    artifact = Path(result["artifacts"][0])
    assert artifact.is_file()
    assert artifact.suffix == ".csv"
    assert artifact.resolve().is_relative_to(settings.artifact_dir.resolve())
    text = artifact.read_text(encoding="utf-8")
    assert "North" in text
    assert artifact.name == "summary.csv"
    assert_keys_match_required(result, RUN_ANALYSIS_CODE.result_schema)


def test_denied_import_is_rejected_before_spawn(
    tmp_path: Path, sample_sales_csv: Path
):
    """A subprocess import fails the AST check; no artifacts are written."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    with pytest.raises(PythonAstError, match="subprocess"):
        _run(settings, saved.source_id, "import subprocess\n")
    artifact_dir = settings.artifact_dir
    if artifact_dir.is_dir():
        assert list(artifact_dir.rglob("*")) == []


def test_nonzero_exit_includes_capped_stderr(
    tmp_path: Path, sample_sales_csv: Path
):
    """A non-zero exit is an error and includes a bounded stderr snippet."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    with pytest.raises(AnalysisCodeError, match="no-such-column") as exc_info:
        _run(settings, saved.source_id, 'raise ValueError("no-such-column")\n')
    assert "exited" in str(exc_info.value)


def test_nonzero_stderr_is_capped(tmp_path: Path, sample_sales_csv: Path):
    """Stderr on AnalysisCodeError is cut at max_prompt_chars."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    with pytest.raises(AnalysisCodeError) as exc_info:
        _run(
            settings,
            saved.source_id,
            "import sys\nsys.stderr.write('Z' * 80)\nraise SystemExit(1)\n",
            max_prompt_chars=20,
        )
    text = str(exc_info.value)
    assert "exited" in text
    assert text.count("Z") == 20


def test_success_stdout_is_capped(tmp_path: Path, sample_sales_csv: Path):
    """Stdout on a successful run is cut at max_prompt_chars."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    result = _run(
        settings,
        saved.source_id,
        "import sys\nsys.stdout.write('Y' * 80)\n",
        max_prompt_chars=20,
    )
    assert result["stdout"] == "Y" * 20
    assert_keys_match_required(result, RUN_ANALYSIS_CODE.result_schema)


def test_unknown_source_id_is_rejected(tmp_path: Path):
    """A missing file source is not copied into the sandbox."""
    settings = _settings(tmp_path)
    with pytest.raises(FileValidationError, match="Unknown source_id"):
        _run(settings, "file-missing", _SUM_CODE)


def test_empty_code_is_rejected(tmp_path: Path, sample_sales_csv: Path):
    """Blank code is rejected before spawn."""
    settings, saved = _saved_sales(tmp_path, sample_sales_csv)
    with pytest.raises(AnalysisCodeError, match="empty"):
        _run(settings, saved.source_id, "   ")


def test_run_analysis_code_contract_is_complete():
    """The MCP-shaped contract has a name, description, and JSON schemas."""
    assert RUN_ANALYSIS_CODE.name == "run_analysis_code"
    assert RUN_ANALYSIS_CODE.description
    assert RUN_ANALYSIS_CODE.input_schema["required"] == ["source_id", "code"]
    assert RUN_ANALYSIS_CODE.result_schema["required"] == [
        "source_id",
        "stdout",
        "artifacts",
    ]
