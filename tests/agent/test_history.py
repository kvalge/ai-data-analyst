# test_history.py

"""Tests for last-N prompt window. Does not inspect message payloads as data."""

from __future__ import annotations

import logging

import pytest

from src.agent.history import recent_messages


def test_recent_messages_keeps_all_when_under_limit():
    """A short thread is unchanged."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
    ]
    assert recent_messages(messages, max_turns=2) == messages


def test_recent_messages_starts_at_nth_last_user():
    """Older user turns drop; the kept window includes later assistant text."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "three"},
    ]
    assert recent_messages(messages, max_turns=2) == messages[2:]


def test_window_ending_on_assistant_keeps_that_reply():
    """A completed turn (user then assistant) is kept as one window."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "a2"},
    ]
    assert recent_messages(messages, max_turns=1) == messages[2:]


def test_blank_user_does_not_count_as_a_turn():
    """Whitespace-only user text is not a turn."""
    messages = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "two"},
    ]
    assert recent_messages(messages, max_turns=1) == [messages[2]]


def test_max_turns_below_one_raises():
    """A non-positive window is not silently treated as keep-all."""
    with pytest.raises(ValueError, match="at least 1"):
        recent_messages([{"role": "user", "content": "one"}], max_turns=0)


def test_dropped_turns_log_count_only(caplog: pytest.LogCaptureFixture):
    """Debug notes how many turns dropped, not what they said."""
    caplog.set_level(logging.DEBUG, logger="src.agent.history")
    recent_messages(
        [
            {"role": "user", "content": "secret-one"},
            {"role": "user", "content": "two"},
        ],
        max_turns=1,
    )
    assert "dropped 1 older turn(s)" in caplog.text
    assert "secret-one" not in caplog.text
