# graph.py

"""LangGraph: confirm sources, profile, then agent ⇄ execute_tool."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agent.audit import (
    DECISION_AUTO,
    OUTCOME_ERROR,
    OUTCOME_REJECTED,
    append_tool_use,
    code_text_for_audit,
    default_audit_dir,
    source_id_for_audit,
)
from src.agent.code_approval import (
    ACTION_REJECT,
    CODE_TOOLS,
    apply_code_decision,
    build_code_interrupt,
    generated_code_text,
    should_pause_generated_code,
    source_id_for_code_tool,
)
from src.agent.confirm_sources import (
    ACTION_SELECT,
    REASON_EMPTY_SCHEMA,
    apply_confirm_decision,
    build_interrupt_payload,
    decide_confirm_reason,
    file_sample_emptiness,
    list_confirm_sources,
)
from src.agent.execute import (
    ToolValidationError,
    interpret_model_reply,
    run_allowlisted_tool,
)
from src.agent.json_output import (
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    JsonOutputError,
    decide_after_parse_failure,
    retry_prompt_after_validation,
)
from src.agent.load_approval import (
    apply_load_decision,
    build_load_interrupt,
    is_over_limit_load,
)
from src.agent.llm import LlmError, complete
from src.agent.profile_steps import (
    SECTION_DQ,
    SECTION_EDA,
    SECTION_SCHEMA,
    apply_profile_decision,
    build_profile_interrupt,
    should_pause_profiling,
    with_cleared_profile_ids,
)
from src.agent.history import recent_messages
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.agent.summarize import checkpoint_tool_result, is_json_safe, merge_artifacts
from src.config import DEFAULT_MAX_ARTIFACTS, Settings
from src.db.postgres import postgres_configured
from src.storage.registry import get_file_source
from src.storage.sources import make_postgres_source
from src.tools.profile_source import profile_source
from src.tools.registry import TOOL_REGISTRY

_LOG = logging.getLogger(__name__)

RUN_STOPPED_MESSAGE = "Run stopped."


class CompleteFn(Protocol):
    """LLM callable the graph may inject. Matches `complete` keyword args."""

    def __call__(
        self,
        prompt: str,
        *,
        settings: Settings,
        structured: bool = ...,
    ) -> str: ...


def build_prompt(
    state: AgentState,
    *,
    max_turns: int,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
) -> str:
    """System prompt, last N turns, latest summary, and artifact paths."""
    parts = [build_system_prompt(hitl_mode=state["hitl_mode"])]
    for message in recent_messages(state["messages"], max_turns=max_turns):
        content = message.get("content", "").strip()
        if not content:
            continue
        parts.append(f"{message.get('role', '')}: {content}")
    result = state.get("last_tool_result")
    if result is not None:
        parts.append(
            "Last tool result (do not invent extra rows):\n"
            + json.dumps(result)
        )
    raw_artifacts = state.get("artifacts") or []
    paths = merge_artifacts(
        raw_artifacts if isinstance(raw_artifacts, list) else [],
        [],
        max_artifacts=max_artifacts,
    )
    if paths:
        parts.append(
            "Available artifacts (paths only; reuse these, do not invent):\n"
            + "\n".join(paths)
        )
    summary = state.get("profile_summary")
    if summary is not None:
        parts.append(
            "Profile summary (bounded head, not the dataset):\n"
            + json.dumps(summary)
        )
    return "\n\n".join(parts)


def route_after_agent(state: AgentState) -> str:
    """Send a parsed tool call to execute_tool. Otherwise end the turn."""
    if state.get("error"):
        return END
    pending = state.get("pending_tool")
    if isinstance(pending, dict) and pending.get("name"):
        return "execute_tool"
    return END


def route_after_execute(state: AgentState) -> str:
    """After a failed tool, end. After success, let the agent read the result."""
    if state.get("error"):
        return END
    return "agent"


def route_after_confirm(state: AgentState) -> str:
    """After abort or a confirm error, end. Otherwise run schema detection."""
    if state.get("error"):
        return END
    return "detect_schema"


def route_after_schema(state: AgentState) -> str:
    """After a schema error or abort, end. Otherwise run data-quality checks."""
    if state.get("error"):
        return END
    return "run_dq"


def route_after_dq(state: AgentState) -> str:
    """After a DQ error or abort, end. Otherwise run EDA."""
    if state.get("error"):
        return END
    return "run_eda"


def route_after_eda(state: AgentState) -> str:
    """After an EDA error or abort, end. Otherwise continue to the agent."""
    if state.get("error"):
        return END
    return "agent"


def build_graph(
    *,
    settings: Settings,
    complete_fn: CompleteFn | None = None,
    include_postgres: bool = False,
    connect: Callable[..., Any] | None = None,
    checkpointer: Any | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> Any:
    """Compile START → confirm_sources → profile nodes → agent ⇄ execute_tool.

    `connect` is a test seam for query_database. The LLM cannot supply it.
    Tests omit `checkpointer` and get MemorySaver. The app passes SqliteSaver.
    `stop_requested` is cooperative: in-flight work may finish; the next
    tool is not started.
    """
    completer = complete_fn or complete

    def _halt_if_stopped() -> dict[str, Any] | None:
        if stop_requested is None or not stop_requested():
            return None
        _LOG.info("graph run stopped")
        return {"error": RUN_STOPPED_MESSAGE, "pending_tool": None}

    def _selected_source_id(state: AgentState) -> str | None:
        ids = [
            item
            for item in (state.get("source_ids") or [])
            if isinstance(item, str) and item
        ]
        return ids[0] if ids else None

    def _cleared_profile_ids(state: AgentState) -> list[str]:
        return [
            item
            for item in (state.get("cleared_profile_source_ids") or [])
            if isinstance(item, str)
        ]

    def _is_postgres_source(source_id: str) -> bool:
        if get_file_source(settings.upload_dir, source_id) is not None:
            return False
        if not postgres_configured(settings):
            return False
        return make_postgres_source(settings).source_id == source_id

    def _load_profile(state: AgentState, source_id: str) -> dict[str, Any]:
        existing = state.get("profile_summary")
        if isinstance(existing, dict) and existing.get("source_id") == source_id:
            return existing
        return profile_source(
            upload_dir=settings.upload_dir,
            cache_dir=settings.cache_dir,
            n_rows=settings.sample_n_rows,
            max_bytes=settings.max_upload_bytes,
            source_id=source_id,
            settings=settings,
        )

    def _maybe_pause_profile(
        step: str,
        summary: dict[str, Any],
        state: AgentState,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        source_id = str(summary.get("source_id") or "")
        cleared = _cleared_profile_ids(state)
        hitl_mode = state.get("hitl_mode") or ""
        if not should_pause_profiling(str(hitl_mode)) or source_id in cleared:
            return updates
        _LOG.info("graph %s interrupt source_id=%s", step, source_id)
        decision = interrupt(build_profile_interrupt(step, summary))
        applied = apply_profile_decision(
            decision,
            source_id=source_id,
            cleared_profile_source_ids=cleared,
        )
        return {**updates, **applied}

    def confirm_sources_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        sources = list_confirm_sources(
            settings, include_postgres=include_postgres
        )
        incoming = [
            item for item in (state.get("source_ids") or []) if isinstance(item, str)
        ]
        cleared = [
            item
            for item in (state.get("cleared_empty_source_ids") or [])
            if isinstance(item, str)
        ]
        reason, selected = decide_confirm_reason(sources, incoming)
        extra: dict[str, int] = {}
        if reason is None and selected:
            empty = file_sample_emptiness(selected[0], sources, settings)
            if empty is not None and selected[0] not in cleared:
                reason = REASON_EMPTY_SCHEMA
                extra = empty
        if reason is None:
            _LOG.info("graph confirm_sources ok")
            return {
                "source_ids": selected,
                "pending_interrupt": None,
                "error": None,
            }
        payload = build_interrupt_payload(
            reason, sources, selected, **extra
        )
        _LOG.info("graph confirm_sources interrupt reason=%s", reason)
        decision = interrupt(payload)
        updates = apply_confirm_decision(
            decision, sources, selected, cleared_empty_source_ids=cleared
        )
        if updates.get("error"):
            return updates
        if isinstance(decision, dict) and decision.get("action") == ACTION_SELECT:
            chosen = updates.get("source_ids") or []
            if chosen:
                empty = file_sample_emptiness(chosen[0], sources, settings)
                if empty is not None and chosen[0] not in cleared:
                    second = build_interrupt_payload(
                        REASON_EMPTY_SCHEMA, sources, chosen, **empty
                    )
                    return apply_confirm_decision(
                        interrupt(second),
                        sources,
                        chosen,
                        cleared_empty_source_ids=cleared,
                    )
        return updates

    def detect_schema_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        source_id = _selected_source_id(state)
        if not source_id:
            return {"error": "No data source selected."}
        if _is_postgres_source(source_id):
            _LOG.info("graph detect_schema skip postgres")
            return {"error": None}
        try:
            summary = _load_profile(state, source_id)
        except Exception as exc:
            _LOG.info("graph detect_schema failed")
            return {"error": str(exc)}
        if not is_json_safe(summary):
            _LOG.info("graph detect_schema not json-safe")
            return {"error": "Profile summary is not JSON-safe."}
        updates: dict[str, Any] = {
            "profile_summary": summary,
            "pending_interrupt": None,
            "error": None,
        }
        return _maybe_pause_profile(SECTION_SCHEMA, summary, state, updates)

    def run_dq_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        source_id = _selected_source_id(state)
        if not source_id:
            return {"error": "No data source selected."}
        if _is_postgres_source(source_id):
            _LOG.info("graph run_dq skip postgres")
            return {"error": None}
        summary = state.get("profile_summary")
        if not isinstance(summary, dict):
            return {"error": "Profile summary is missing."}
        return _maybe_pause_profile(
            SECTION_DQ, summary, state, {"pending_interrupt": None, "error": None}
        )

    def run_eda_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        source_id = _selected_source_id(state)
        if not source_id:
            return {"error": "No data source selected."}
        if _is_postgres_source(source_id):
            _LOG.info("graph run_eda skip postgres")
            return {"error": None}
        summary = state.get("profile_summary")
        if not isinstance(summary, dict):
            return {"error": "Profile summary is missing."}
        updates = _maybe_pause_profile(
            SECTION_EDA, summary, state, {"pending_interrupt": None, "error": None}
        )
        if updates.get("error"):
            return updates
        cleared = updates.get("cleared_profile_source_ids")
        if not isinstance(cleared, list):
            cleared = _cleared_profile_ids(state)
        updates["cleared_profile_source_ids"] = with_cleared_profile_ids(
            list(cleared), [source_id]
        )
        return updates

    def _llm_reply(prompt: str) -> str:
        reply = completer(prompt, settings=settings, structured=True)
        if not isinstance(reply, str) or not reply.strip():
            raise LlmError("Ollama response was missing.")
        return reply

    def agent_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        if not any(
            message.get("role") == "user" and message.get("content", "").strip()
            for message in state["messages"]
        ):
            return {"error": "No user message to reply to."}
        prompt = build_prompt(
            state,
            max_turns=settings.max_prompt_turns,
            max_artifacts=settings.max_artifacts,
        )
        try:
            reply = _llm_reply(prompt)
            call = interpret_model_reply(reply)
        except LlmError as exc:
            _LOG.info("graph agent llm error")
            return {"error": str(exc), "pending_tool": None}
        except (JsonOutputError, ToolValidationError) as exc:
            if decide_after_parse_failure(FIRST_PARSE_FAILURE) != RETRY_STRICT:
                return {"error": str(exc), "pending_tool": None}
            _LOG.info("graph tool validation retry")
            try:
                reply = _llm_reply(
                    retry_prompt_after_validation(
                        prompt,
                        str(exc),
                        limit=settings.max_prompt_chars,
                    )
                )
                call = interpret_model_reply(reply)
            except LlmError as retry_exc:
                _LOG.info("graph agent llm error")
                return {"error": str(retry_exc), "pending_tool": None}
            except (JsonOutputError, ToolValidationError) as retry_exc:
                _LOG.info("graph tool validation failed after retry")
                return {"error": str(retry_exc), "pending_tool": None}
        if call is None:
            _LOG.info("graph agent replied")
            return {
                "messages": [{"role": "assistant", "content": reply}],
                "pending_tool": None,
                "error": None,
            }
        _LOG.info("graph agent requested tool=%s", call["name"])
        return {"pending_tool": call, "error": None}

    def _append_code_audit(
        name: str,
        arguments: dict[str, Any],
        *,
        decision: str,
        outcome: str,
    ) -> None:
        """Log generated SQL/Python, the HITL decision, and sandbox outcome."""
        append_tool_use(
            default_audit_dir(settings.upload_dir),
            tool=name,
            source_id=source_id_for_audit(arguments),
            code=code_text_for_audit(
                name, arguments, settings.max_prompt_chars
            ),
            decision=decision,
            outcome=outcome,
        )

    def execute_tool_node(state: AgentState) -> dict[str, Any]:
        halted = _halt_if_stopped()
        if halted:
            return halted
        pending = state.get("pending_tool")
        if not isinstance(pending, dict):
            return {"error": "Unknown tool.", "pending_tool": None}
        name = pending.get("name")
        if not isinstance(name, str) or name not in TOOL_REGISTRY:
            return {"error": "Unknown tool.", "pending_tool": None}
        arguments = pending.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        decision = DECISION_AUTO
        if name in CODE_TOOLS and should_pause_generated_code(
            str(state.get("hitl_mode") or "")
        ):
            text = generated_code_text(name, arguments)
            if not text:
                return {"error": "Generated code is empty.", "pending_tool": None}
            _LOG.info("graph approve_code interrupt tool=%s", name)
            applied = apply_code_decision(
                interrupt(
                    build_code_interrupt(
                        tool=name,
                        code=text,
                        source_id=source_id_for_code_tool(name, arguments),
                    )
                ),
                name=name,
                arguments=arguments,
            )
            raw_decision = applied.get("decision")
            decision = (
                raw_decision
                if isinstance(raw_decision, str) and raw_decision
                else ACTION_REJECT
            )
            if applied.get("error"):
                _append_code_audit(
                    name,
                    arguments,
                    decision=decision,
                    outcome=(
                        OUTCOME_REJECTED
                        if raw_decision == ACTION_REJECT
                        else OUTCOME_ERROR
                    ),
                )
                return {"error": applied["error"], "pending_tool": None}
            next_args = applied.get("arguments")
            arguments = next_args if isinstance(next_args, dict) else {}
        halted = _halt_if_stopped()
        if halted:
            return halted
        try:
            result = run_allowlisted_tool(
                name,
                arguments,
                settings,
                include_postgres=include_postgres,
                connect=connect,
                decision=decision if name in CODE_TOOLS else None,
            )
        except Exception as exc:
            _LOG.info("graph tool failed name=%s", name)
            if name in CODE_TOOLS:
                _append_code_audit(
                    name,
                    arguments,
                    decision=decision,
                    outcome=OUTCOME_ERROR,
                )
            return {"error": str(exc), "pending_tool": None}
        if name == "load_full_file" and is_over_limit_load(result):
            _LOG.info(
                "graph approve_load interrupt source_id=%s",
                result.get("source_id"),
            )
            applied = apply_load_decision(interrupt(build_load_interrupt(result)))
            if applied.get("error"):
                return {"error": applied["error"], "pending_tool": None}
            halted = _halt_if_stopped()
            if halted:
                return halted
            try:
                result = run_allowlisted_tool(
                    name,
                    arguments,
                    settings,
                    include_postgres=include_postgres,
                    connect=connect,
                    allow_over_limit=True,
                )
            except Exception as exc:
                _LOG.info("graph tool failed name=%s", name)
                return {"error": str(exc), "pending_tool": None}
        prior = [
            item
            for item in (state.get("artifacts") or [])
            if isinstance(item, str)
        ]
        try:
            return checkpoint_tool_result(
                name=name,
                result=result,
                prior_artifacts=prior,
                max_artifacts=settings.max_artifacts,
            )
        except (TypeError, ValueError) as exc:
            _LOG.info("graph tool result rejected name=%s", name)
            return {"error": str(exc), "pending_tool": None}

    graph = StateGraph(AgentState)
    graph.add_node("confirm_sources", confirm_sources_node)
    graph.add_node("detect_schema", detect_schema_node)
    graph.add_node("run_dq", run_dq_node)
    graph.add_node("run_eda", run_eda_node)
    graph.add_node("agent", agent_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_edge(START, "confirm_sources")
    graph.add_conditional_edges(
        "confirm_sources",
        route_after_confirm,
        {"detect_schema": "detect_schema", END: END},
    )
    graph.add_conditional_edges(
        "detect_schema",
        route_after_schema,
        {"run_dq": "run_dq", END: END},
    )
    graph.add_conditional_edges(
        "run_dq",
        route_after_dq,
        {"run_eda": "run_eda", END: END},
    )
    graph.add_conditional_edges(
        "run_eda",
        route_after_eda,
        {"agent": "agent", END: END},
    )
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"execute_tool": "execute_tool", END: END},
    )
    graph.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {"agent": "agent", END: END},
    )
    return graph.compile(checkpointer=checkpointer or MemorySaver())
