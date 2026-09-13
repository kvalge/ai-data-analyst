# history.py

"""Last-N user turns for the LLM prompt. Checkpointed history stays complete."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

_LOG = logging.getLogger(__name__)


def recent_messages(
    messages: Sequence[Mapping[str, Any]], *, max_turns: int
) -> list[Mapping[str, Any]]:
    """Return messages from the last `max_turns` user turns onward.

    Blank user texts do not count as turns. Does not log message content.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1.")
    items = list(messages)
    # Each user-role message is exactly one turn (no multi-part user bursts).
    user_indexes = [
        index
        for index, message in enumerate(items)
        if message.get("role") == "user"
        and str(message.get("content", "")).strip()
    ]
    if len(user_indexes) <= max_turns:
        return items
    dropped = len(user_indexes) - max_turns
    _LOG.debug("prompt dropped %s older turn(s)", dropped)
    return items[user_indexes[-max_turns] :]
