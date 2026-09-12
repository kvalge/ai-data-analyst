# test_codegen.py

"""Tests for plan→code generation: role selection, fence strip, AST/SQL check."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.agent.codegen import (
    KIND_PYTHON,
    KIND_SQL,
    CodegenError,
    generate_checked_code,
)
from src.agent.llm import ROLE_CODING, ROLE_PRIMARY
from src.config import Settings, load_settings

_PLACEHOLDER_MODELS = {
    "OLLAMA_MODEL_PRIMARY": "placeholder-primary:tag",
    "OLLAMA_MODEL_FALLBACK_FAST": "placeholder-fast:tag",
    "OLLAMA_MODEL_AGENTIC": "placeholder-agentic:tag",
    "OLLAMA_MODEL_CODING": "placeholder-coding:tag",
}

_PYTHON = "print(1)\n"
_SELECT = "SELECT date, region, revenue FROM sales"


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Frozen settings with placeholder model names. No live Ollama."""
    return load_settings(
        environ={"OLLAMA_HOST": "http://ollama.test:11434", **_PLACEHOLDER_MODELS},
        load_dotenv_file=False,
        project_root=tmp_path,
    )


class _Scripted:
    """Return queued plan then code replies. Records role and structured."""

    def __init__(self, plans: list[str], codes: list[str]) -> None:
        self._plans = list(plans)
        self._codes = list(codes)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        prompt: str,
        *,
        settings: Settings,
        role: str = ROLE_PRIMARY,
        structured: bool = True,
    ) -> str:
        self.calls.append(
            {"role": role, "structured": structured, "prompt": prompt}
        )
        if role == ROLE_PRIMARY:
            return self._plans.pop(0)
        if role == ROLE_CODING:
            return self._codes.pop(0)
        raise AssertionError(f"unexpected role {role}")


def _plan(
    kind: str = KIND_PYTHON,
    task: str = "print one",
    rationale: str = "unit",
    source_id: str | None = None,
) -> str:
    payload: dict[str, str] = {
        "kind": kind,
        "task": task,
        "rationale": rationale,
    }
    if source_id is not None:
        payload["source_id"] = source_id
    return json.dumps(payload)


def _generate(settings: Settings, scripted: _Scripted, request: str = "sum revenue"):
    return generate_checked_code(
        request, settings=settings, complete_fn=scripted
    )


def test_plan_uses_primary_and_code_uses_coding(settings: Settings):
    """Plan JSON is primary/structured; raw code is coding/unstructured."""
    scripted = _Scripted([_plan(task="print one")], [_PYTHON])
    result = _generate(settings, scripted)
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_CODING,
    ]
    assert scripted.calls[0]["structured"] is True
    assert scripted.calls[1]["structured"] is False
    assert result["kind"] == KIND_PYTHON
    assert result["code"] == _PYTHON.strip()
    assert "print one" in scripted.calls[1]["prompt"]


def test_unknown_kind_does_not_call_coding(settings: Settings):
    """A non-code plan retries once on primary and never reaches coding."""
    bad = _plan(kind="prose", task="explain")
    scripted = _Scripted([bad, bad], [])
    with pytest.raises(CodegenError, match="Unknown plan kind"):
        _generate(settings, scripted)
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_PRIMARY,
    ]


def test_unknown_kind_retries_on_primary(settings: Settings):
    """A schema-valid but unknown kind gets the same one-shot retry as bad JSON."""
    scripted = _Scripted([_plan(kind="prose", task="explain"), _plan()], [_PYTHON])
    result = _generate(settings, scripted)
    assert result["code"] == _PYTHON.strip()
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_PRIMARY,
        ROLE_CODING,
    ]


def test_fenced_python_is_stripped_then_checked(settings: Settings):
    """A ```python wrapper is removed before the AST check."""
    scripted = _Scripted([_plan()], ["```python\nprint(1)\n```"])
    result = _generate(settings, scripted)
    assert result["code"] == "print(1)"


def test_denied_python_is_rejected(settings: Settings):
    """A subprocess import fails the AST check on both the first try and retry."""
    denied = "import subprocess\n"
    scripted = _Scripted([_plan()], [denied, denied])
    with pytest.raises(CodegenError, match="subprocess"):
        _generate(settings, scripted)
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_CODING,
        ROLE_CODING,
    ]


def test_invalid_code_retries_once(settings: Settings):
    """A failed AST check retries the coding model once."""
    scripted = _Scripted([_plan()], ["import subprocess\n", _PYTHON])
    result = _generate(settings, scripted)
    assert result["code"] == _PYTHON.strip()
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_CODING,
        ROLE_CODING,
    ]


def test_select_sql_is_accepted(settings: Settings):
    """A single SELECT plan uses the SQL checker, not the AST checker."""
    scripted = _Scripted([_plan(kind=KIND_SQL, task="revenue by region")], [_SELECT])
    result = _generate(settings, scripted)
    assert result["kind"] == KIND_SQL
    assert result["code"] == _SELECT


def test_insert_sql_is_rejected(settings: Settings):
    """INSERT is rejected after fence strip; the coding model is retried once."""
    insert = "INSERT INTO sales (revenue) VALUES (1)"
    scripted = _Scripted([_plan(kind=KIND_SQL)], [insert, insert])
    with pytest.raises(CodegenError, match="INSERT"):
        _generate(settings, scripted)


def test_malformed_plan_retries_on_primary(settings: Settings):
    """A bad plan JSON retries the primary model, not the coding model."""
    scripted = _Scripted(["not-json", _plan()], [_PYTHON])
    result = _generate(settings, scripted)
    assert result["code"] == _PYTHON.strip()
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_PRIMARY,
        ROLE_CODING,
    ]
    assert all(call["structured"] is True for call in scripted.calls[:2])


def test_failed_plan_does_not_call_coding(settings: Settings):
    """Two bad plan replies fail visibly without a coding call."""
    scripted = _Scripted(["not-json", "still-not-json"], [])
    with pytest.raises(CodegenError, match="Could not parse JSON"):
        _generate(settings, scripted)
    assert [call["role"] for call in scripted.calls] == [
        ROLE_PRIMARY,
        ROLE_PRIMARY,
    ]


def test_empty_request_does_not_call_llm(settings: Settings):
    """Blank input is rejected before either model is called."""
    scripted = _Scripted([], [])
    with pytest.raises(CodegenError, match="empty"):
        _generate(settings, scripted, request="   ")
    assert scripted.calls == []
