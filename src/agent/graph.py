# graph.py

"""LangGraph: agent decides; execute_tool runs allowlisted Phase 3 tools."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.execute import (
    ToolValidationError,
    interpret_model_reply,
    retry_prompt_after_validation,
    run_allowlisted_tool,
)
from src.agent.json_output import (
    FIRST_PARSE_FAILURE,
    RETRY_STRICT,
    JsonOutputError,
    decide_after_parse_failure,
)
from src.agent.llm import LlmError, complete
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.config import Settings
from src.tools.registry import TOOL_REGISTRY

_LOG = logging.getLogger(__name__)


class CompleteFn(Protocol):
    """LLM callable the graph may inject. Matches `complete` keyword args."""

    def __call__(
        self,
        prompt: str,
        *,
        settings: Settings,
        structured: bool = ...,
    ) -> str: ...


def build_prompt(state: AgentState) -> str:
    """System prompt plus conversation text. Never includes a full file."""
    parts = [build_system_prompt(hitl_mode=state["hitl_mode"])]
    for message in state["messages"]:
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


def build_graph(
    *,
    settings: Settings,
    complete_fn: CompleteFn | None = None,
    include_postgres: bool = False,
) -> Any:
    """Compile START → agent ⇄ execute_tool → END. Checkpointer is MemorySaver."""
    completer = complete_fn or complete

    def _llm_reply(prompt: str) -> str:
        reply = completer(prompt, settings=settings, structured=True)
        if not isinstance(reply, str) or not reply.strip():
            raise LlmError("Ollama response was missing.")
        return reply

    def agent_node(state: AgentState) -> dict[str, Any]:
        if not any(
            message.get("role") == "user" and message.get("content", "").strip()
            for message in state["messages"]
        ):
            return {"error": "No user message to reply to."}
        prompt = build_prompt(state)
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
                reply = _llm_reply(retry_prompt_after_validation(prompt, str(exc)))
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

    def execute_tool_node(state: AgentState) -> dict[str, Any]:
        pending = state.get("pending_tool")
        if not isinstance(pending, dict):
            return {"error": "Unknown tool.", "pending_tool": None}
        name = pending.get("name")
        if not isinstance(name, str) or name not in TOOL_REGISTRY:
            return {"error": "Unknown tool.", "pending_tool": None}
        arguments = pending.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            result = run_allowlisted_tool(
                name,
                arguments,
                settings,
                include_postgres=include_postgres,
            )
        except Exception as exc:
            _LOG.info("graph tool failed name=%s", name)
            return {"error": str(exc), "pending_tool": None}
        updates: dict[str, Any] = {
            "last_tool_result": result,
            "pending_tool": None,
            "error": None,
        }
        if name == "profile_source":
            updates["profile_summary"] = result
        return updates

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_edge(START, "agent")
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
    return graph.compile(checkpointer=MemorySaver())
