# graph.py

"""LangGraph stub: one agent node, tools off, MemorySaver checkpointer."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.llm import LlmError, complete
from src.agent.prompts import build_system_prompt
from src.agent.state import AgentState
from src.config import Settings

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
    """System prompt plus conversation text. Does not include dataset rows."""
    parts = [build_system_prompt(hitl_mode=state["hitl_mode"])]
    for message in state["messages"]:
        content = message.get("content", "").strip()
        if not content:
            continue
        parts.append(f"{message.get('role', '')}: {content}")
    return "\n\n".join(parts)


def build_graph(
    *,
    settings: Settings,
    complete_fn: CompleteFn | None = None,
) -> Any:
    """Compile START → agent → END. Tools are not bound. Checkpointer is MemorySaver."""
    completer = complete_fn or complete

    def agent_node(state: AgentState) -> dict[str, Any]:
        if not any(
            message.get("role") == "user" and message.get("content", "").strip()
            for message in state["messages"]
        ):
            return {"error": "No user message to reply to."}
        prompt = build_prompt(state)
        try:
            reply = completer(prompt, settings=settings, structured=False)
        except LlmError as exc:
            _LOG.info("graph agent llm error")
            return {"error": str(exc)}
        if not isinstance(reply, str) or not reply.strip():
            return {"error": "Ollama response was missing."}
        _LOG.info("graph agent replied")
        return {
            "messages": [{"role": "assistant", "content": reply}],
            "error": None,
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=MemorySaver())
