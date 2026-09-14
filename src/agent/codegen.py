# codegen.py

"""Plan on the primary model; raw code on the coding model; then AST/SQL check."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from src.agent.artifact_policy import ARTIFACT_POLICY_TEXT
from src.agent.json_output import (
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    JsonOutputError,
    JsonSchemaError,
    decide_after_parse_failure,
    parse_json_output,
    retry_prompt_after_validation,
    strip_markdown_fences,
)
from src.agent.llm import ROLE_CODING, ROLE_PRIMARY, complete
from src.config import Settings, bound_text
from src.execution.ast_check import PythonAstError, check_python_ast
from src.execution.sql_check import SqlCheckError, check_sql

_LOG = logging.getLogger(__name__)

KIND_PYTHON = "python"
KIND_SQL = "sql"
_CODE_KINDS = frozenset({KIND_PYTHON, KIND_SQL})

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "task": {"type": "string"},
        "rationale": {"type": "string"},
        "source_id": {"type": "string"},
    },
    "required": ["kind", "task"],
    "additionalProperties": False,
}


class CodegenError(ValueError):
    """Plan or generated code failed validation. Do not guess a replacement."""


class CodegenCompleteFn(Protocol):
    """LLM callable codegen may inject. Matches `complete` role and structured."""

    def __call__(
        self,
        prompt: str,
        *,
        settings: Settings,
        role: str = ...,
        structured: bool = ...,
    ) -> str: ...


def generate_checked_code(
    user_request: str,
    *,
    settings: Settings,
    source_id: str | None = None,
    complete_fn: CodegenCompleteFn | None = None,
) -> dict[str, Any]:
    """Return checked Python or SQL. Does not execute the generated text.

    Primary produces the plan JSON. The coding model is called only for
    kind python or sql. Fences are stripped before AST/SQL validation.
    """
    request = user_request.strip()
    if not request:
        raise CodegenError("user_request is empty.")
    completer = complete_fn or complete
    plan = _complete_plan(
        completer,
        settings,
        _plan_prompt(request, source_id, settings.max_prompt_chars),
    )
    kind = str(plan["kind"])
    task = str(plan.get("task", "")).strip()
    if not task:
        raise CodegenError("Plan task is empty.")
    planned_source = plan.get("source_id")
    resolved_source = (
        planned_source.strip()
        if isinstance(planned_source, str) and planned_source.strip()
        else source_id
    )
    _LOG.info("codegen plan kind=%s", kind)
    code = _complete_code(
        completer,
        settings,
        kind,
        _code_prompt(kind, task, settings.max_prompt_chars),
    )
    return {
        "kind": kind,
        "code": code,
        "task": task,
        "rationale": str(plan.get("rationale") or ""),
        "source_id": resolved_source,
    }


def _complete_plan(
    completer: CodegenCompleteFn,
    settings: Settings,
    prompt: str,
) -> dict[str, Any]:
    """Ask the primary model for plan JSON. One stricter retry, then fail."""
    reply = completer(
        prompt, settings=settings, role=ROLE_PRIMARY, structured=True
    )
    try:
        return _parsed_plan(reply)
    except JsonOutputError as exc:
        # Same first-failure hook as graph agent_node (3.2 / 3.9).
        if decide_after_parse_failure(FIRST_PARSE_FAILURE) != RETRY_STRICT:
            raise CodegenError(str(exc)) from exc
        _LOG.info("codegen plan parse retry")
        retry_prompt = retry_prompt_after_validation(
            prompt, str(exc), limit=settings.max_prompt_chars
        )
        reply = completer(
            retry_prompt, settings=settings, role=ROLE_PRIMARY, structured=True
        )
        try:
            return _parsed_plan(reply)
        except JsonOutputError as retry_exc:
            raise CodegenError(str(retry_exc)) from retry_exc


def _parsed_plan(reply: str) -> dict[str, Any]:
    """Parse plan JSON and require kind python or sql. Does not invent a kind."""
    plan = parse_json_output(reply, PLAN_SCHEMA)
    kind = str(plan.get("kind", "")).strip().lower()
    if kind not in _CODE_KINDS:
        raise JsonSchemaError(f"Unknown plan kind: {kind or '(empty)'}.")
    plan["kind"] = kind
    return plan


def _complete_code(
    completer: CodegenCompleteFn,
    settings: Settings,
    kind: str,
    prompt: str,
) -> str:
    """Ask the coding model for raw code. One check retry, then fail."""
    reply = completer(
        prompt, settings=settings, role=ROLE_CODING, structured=False
    )
    try:
        return _checked_code(kind, reply)
    except CodegenError as exc:
        _LOG.info("codegen code check retry kind=%s", kind)
        retry_prompt = retry_prompt_after_validation(
            prompt,
            str(exc),
            limit=settings.max_prompt_chars,
            instruction="",
        )
        reply = completer(
            retry_prompt, settings=settings, role=ROLE_CODING, structured=False
        )
        return _checked_code(kind, reply)


def _checked_code(kind: str, text: str) -> str:
    """Strip fences, then run the matching static checker. Never exec."""
    stripped = strip_markdown_fences(text)
    if not stripped.strip():
        raise CodegenError("Generated code is empty.")
    try:
        if kind == KIND_PYTHON:
            check_python_ast(stripped)
        else:
            check_sql(stripped)
    except (PythonAstError, SqlCheckError) as exc:
        raise CodegenError(str(exc)) from exc
    return stripped


def _plan_prompt(request: str, source_id: str | None, limit: int) -> str:
    """Structured-plan prompt. No dataset rows."""
    parts = [
        "Plan one local analysis step. Reply with a single JSON object only:",
        '{"kind": "python" or "sql", "task": "<instruction for the code model>", '
        '"rationale": "<short why>", "source_id": "<id if known>"}',
        "No markdown fences and no commentary. Do not include dataset rows.",
        f"User request:\n{bound_text(request, limit)}",
    ]
    if source_id:
        parts.append(f"Source id: {bound_text(source_id, limit)}")
    return "\n\n".join(parts)


def _code_prompt(kind: str, task: str, limit: int) -> str:
    """Raw-code prompt for the coding model. No dataset rows."""
    if kind == KIND_PYTHON:
        header = (
            "Write Python for a local sandbox. Reply with Python only, "
            "no markdown fences, no commentary. The registered file is "
            "already in the work dir under its original basename. "
            f"{ARTIFACT_POLICY_TEXT} "
            "Do not import "
            "subprocess, socket, requests, http, or urllib. Do not use "
            "eval, exec, or __import__."
        )
    else:
        header = (
            "Write one parameterized PostgreSQL SELECT or WITH. Reply with "
            "SQL only, no markdown fences. No INSERT, UPDATE, DELETE, DDL, "
            "or multiple statements."
        )
    return f"{header}\n\nTask:\n{bound_text(task, limit)}"
