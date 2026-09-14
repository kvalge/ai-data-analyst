# run_analysis_code.py

"""Tool: AST-check generated Python, then run it in the sandbox."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.config import Settings, bound_text
from src.execution.ast_check import check_python_ast
from src.execution.sandbox import run_python_file
from src.execution.workspace import copy_artifacts, sandbox_work_dir
from src.storage.registry import get_file_source
from src.tools.contracts import ToolContract
from src.validation.data_files import FileValidationError

_LOG = logging.getLogger(__name__)

_SCRIPT_NAME = "_sandbox_run.py"
_ARTIFACT_SUFFIXES = frozenset({".csv", ".png"})


class AnalysisCodeError(ValueError):
    """Sandbox analysis failed after the AST check, or exited non-zero."""


RUN_ANALYSIS_CODE = ToolContract(
    name="run_analysis_code",
    description=(
        "Run checked Python in a sandbox against one registered file source. "
        "Provide source_id and code. The file is copied into the work dir "
        "under its original basename. Save csv/png there; do not print "
        "large tables. Returns stdout and artifact paths, never row records."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "code": {"type": "string"},
        },
        "required": ["source_id", "code"],
        "additionalProperties": False,
    },
    result_schema={
        "type": "object",
        "properties": {
            "source_id": {"type": "string"},
            "stdout": {"type": "string"},
            "artifacts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["source_id", "stdout", "artifacts"],
        "additionalProperties": False,
    },
)


def run_analysis_code(
    *,
    settings: Settings,
    source_id: str,
    code: str,
    upload_dir: Path,
    artifact_dir: Path,
    timeout_s: int,
    max_prompt_chars: int,
) -> dict[str, Any]:
    """Check `code`, run it in a subprocess, return stdout and artifact paths.

    Paths and limits come from settings. Does not exec or eval in this process.
    """
    # Imported here to avoid tools ↔ agent import cycle at module load.
    from src.agent.json_output import strip_markdown_fences
    from src.agent.state import as_artifact_path

    stripped = strip_markdown_fences(code)
    if not stripped.strip():
        raise AnalysisCodeError("code is empty.")
    source = get_file_source(upload_dir, source_id)
    if source is None or source.stored_path is None:
        raise FileValidationError(f"Unknown source_id: {source_id}")
    input_name = _input_basename(source.original_name)
    check_python_ast(stripped)
    _LOG.info("run_analysis_code source_id=%s timeout_s=%s", source_id, timeout_s)

    with sandbox_work_dir() as work:
        data_path = work / input_name
        shutil.copy2(source.stored_path, data_path)
        script = work / _SCRIPT_NAME
        script.write_text(stripped, encoding="utf-8")
        result = run_python_file(script, work, timeout_s=timeout_s)
        if result.exit_code != 0:
            snippet = bound_text(result.stderr, max_prompt_chars)
            detail = f"Sandbox exited with code {result.exit_code}."
            if snippet:
                detail = f"{detail}\n{snippet}"
            raise AnalysisCodeError(detail)
        copied = copy_artifacts(
            work, artifact_dir, exclude=(script, data_path)
        )

    artifacts = [
        as_artifact_path(path, artifact_dir=artifact_dir)
        for path in copied
        if path.suffix.lower() in _ARTIFACT_SUFFIXES
    ]
    return {
        "source_id": source_id,
        "stdout": bound_text(result.stdout, max_prompt_chars),
        "artifacts": artifacts,
    }


def _input_basename(original_name: str) -> str:
    """Work-dir name for the copied source. Never the runner script name."""
    name = Path(original_name).name
    if not name or name in {".", ".."} or name == _SCRIPT_NAME:
        raise FileValidationError("Source file name is not usable in the sandbox.")
    return name
