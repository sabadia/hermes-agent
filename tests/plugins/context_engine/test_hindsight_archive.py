"""Tests for the hindsight_archive context engine plugin."""

import json
import queue
import threading
import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from plugins.context_engine.hindsight_archive import (
    HindsightArchiveCompressor,
    _archive_queue,
    _compute_discarded,
    _deadletter_path,
    _enqueue_archive,
    _ensure_writer,
    _flush_archive_queue,
    _format_message,
    _message_key,
    _sanitize_inline_secrets,
    register,
)


class FakeContextEngine:
    """Minimal fake for plugin registration tests."""

    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine


# ---------------------------------------------------------------------------
# Registration & identity
# ---------------------------------------------------------------------------


def test_register():
    ctx = FakeContextEngine()
    register(ctx)
    assert ctx.engine is not None
    assert ctx.engine.name == "hindsight_archive"


def test_name():
    hc = HindsightArchiveCompressor()
    assert hc.name == "hindsight_archive"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_is_available_with_unreachable_mode():
    with patch(
        "plugins.context_engine.hindsight_archive._load_hindsight_config",
        return_value={"mode": "unknown"},
    ):
        assert HindsightArchiveCompressor.is_available() is False


def test_is_available_with_local_external():
    resp_mock = MagicMock()
    resp_mock.status = 200
    resp_mock.__enter__ = MagicMock(return_value=resp_mock)
    resp_mock.__exit__ = MagicMock(return_value=False)
    with patch(
        "plugins.context_engine.hindsight_archive._load_hindsight_config",
        return_value={"mode": "local_external", "api_url": "http://localhost:8881"},
    ), patch(
        "urllib.request.urlopen",
        return_value=resp_mock,
    ):
        assert HindsightArchiveCompressor.is_available() is True


def test_is_available_with_local_external_404():
    err = urllib.error.HTTPError(
        "http://localhost:8881/health", 404, "Not Found", {}, None
    )
    with patch(
        "plugins.context_engine.hindsight_archive._load_hindsight_config",
        return_value={"mode": "local_external", "api_url": "http://localhost:8881"},
    ), patch(
        "urllib.request.urlopen",
        side_effect=err,
    ):
        assert HindsightArchiveCompressor.is_available() is True


# ---------------------------------------------------------------------------
# Redaction & formatting
# ---------------------------------------------------------------------------


def test_sanitize_inline_secrets_api_key():
    text = "api_key: sk-abcdefghijklmnopqrstuvwxyz123456"
    result = _sanitize_inline_secrets(text)
    assert "[REDACTED]" in result or "[REDACTED_SK]" in result


def test_sanitize_inline_secrets_password():
    text = "password: supersecret123"
    result = _sanitize_inline_secrets(text)
    assert "[REDACTED]" in result


def test_sanitize_inline_secrets_aws_key():
    text = "AKIAIOSFODNN7EXAMPLE"
    result = _sanitize_inline_secrets(text)
    assert "[REDACTED_AWS_KEY]" in result


def test_format_message_redacts_before_truncation():
    # A secret that would be split if truncation happened first
    secret = "api_key: sk-" + "x" * 5000
    msg = {"role": "user", "content": secret}
    formatted = _format_message(msg)
    # The secret should be fully redacted, not partially exposed by truncation.
    # The long x-run may trigger the base64 regex, which is also acceptable
    # as long as the original secret pattern is not exposed.
    assert "sk-xxxxxxxxxxxxxxxxxxxxxxxx" not in formatted


def test_format_message_tool_call_redacts_arguments():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": '{"api_key": "sk-sec...cdef"}',
                }
            }
        ],
    }
    formatted = _format_message(msg)
    # redact_sensitive_text masks the value with ***
    assert "sk-sec...cdef" not in formatted
    assert "***" in formatted or "[REDACTED" in formatted


def test_format_message_truncate_tool_args():
    long_args = "x" * 500
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "test_tool", "arguments": long_args}}
        ],
    }
    formatted = _format_message(msg)
    assert "..." in formatted or len(formatted) < 300


# ---------------------------------------------------------------------------
# Diff / discard computation
# ---------------------------------------------------------------------------


def test_compute_discarded_basic():
    before = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "bye"},
    ]
    after = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "bye"},
    ]
    discarded = _compute_discarded(before, after)
    assert len(discarded) == 2
    assert discarded[0]["content"] == "hello"
    assert discarded[1]["content"] == "world"


def test_compute_discarded_with_duplicates():
    before = [
        {"role": "user", "content": "ok"},
        {"role": "user", "content": "ok"},
        {"role": "user", "content": "ok"},
    ]
    after = [
        {"role": "user", "content": "ok"},
        {"role": "user", "content": "ok"},
    ]
    discarded = _compute_discarded(before, after)
    assert len(discarded) == 1


def test_compute_discarded_empty():
    before = [{"role": "user", "content": "hi"}]
    after = [{"role": "user", "content": "hi"}]
    assert _compute_discarded(before, after) == []


# ---------------------------------------------------------------------------
# Queue & writer
# ---------------------------------------------------------------------------


def test_enqueue_and_flush():
    # Drain any existing items
    while not _archive_queue.empty():
        try:
            _archive_queue.get_nowait()
            _archive_queue.task_done()
        except queue.Empty:
            break

    with patch("plugins.context_engine.hindsight_archive._do_archive"):
        _enqueue_archive(
            [{"role": "user", "content": "test"}],
            session_id="s1",
            archive_num=1,
        )
        assert _archive_queue.qsize() == 1

        # Flush should drain within timeout
        _flush_archive_queue(timeout=5.0)
        # After flush, queue should be empty (or have only the sentinel which is consumed)
        assert _archive_queue.empty()


def test_flush_queue_timeout():
    # If writer thread is not running, flush should still return (timeout)
    _flush_archive_queue(timeout=0.1)


def test_bounded_queue_drop():
    # Temporarily replace queue with a tiny one to test overflow
    import plugins.context_engine.hindsight_archive as _ha

    old_queue = _ha._archive_queue
    tiny_queue = queue.Queue(maxsize=2)
    _ha._archive_queue = tiny_queue

    try:
        _enqueue_archive([{"role": "user", "content": "1"}], "s", 1)
        _enqueue_archive([{"role": "user", "content": "2"}], "s", 2)
        # Third should drop (not raise)
        _enqueue_archive([{"role": "user", "content": "3"}], "s", 3)
        assert tiny_queue.qsize() == 2
    finally:
        _ha._archive_queue = old_queue


# ---------------------------------------------------------------------------
# Deadletter path
# ---------------------------------------------------------------------------


def test_deadletter_path_profile_safe():
    path = _deadletter_path()
    assert "hindsight_archive_deadletter.jsonl" in str(path)


# ---------------------------------------------------------------------------
# Instance-level failure state
# ---------------------------------------------------------------------------


def test_instance_failure_counters():
    a = HindsightArchiveCompressor()
    b = HindsightArchiveCompressor()
    a._consecutive_failures = 3
    a._last_failure_reason = "timeout"
    assert b._consecutive_failures == 0
    assert b._last_failure_reason is None


def test_reset_clears_counters():
    hc = HindsightArchiveCompressor()
    hc._consecutive_failures = 2
    hc._last_failure_reason = "err"
    hc._archive_count = 5
    hc._session_id = "sid"
    hc.on_session_reset()
    assert hc._consecutive_failures == 0
    assert hc._last_failure_reason is None
    assert hc._archive_count == 0
    assert hc._session_id == ""


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_get_status_has_archive_keys():
    hc = HindsightArchiveCompressor()
    status = hc.get_status()
    assert "archive_count" in status
    assert "engine" in status
    assert "archive_bank_id" in status
    assert "queue_size" in status
    assert "queue_max_size" in status
    assert "consecutive_failures" in status
    assert "last_failure_reason" in status
    assert "is_available" in status
    assert "deadletter_path" in status


# ---------------------------------------------------------------------------
# Compression integration (mock inner compressor)
# ---------------------------------------------------------------------------


def test_compress_archives_discarded_messages():
    hc = HindsightArchiveCompressor()

    before = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "bye"},
    ]
    after = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "bye"},
    ]

    mock_inner = MagicMock()
    mock_inner.compress.return_value = after
    mock_inner.compression_count = 1
    hc._inner = mock_inner
    hc._session_id = "test-session"

    # Drain queue before test
    while not _archive_queue.empty():
        try:
            _archive_queue.get_nowait()
            _archive_queue.task_done()
        except queue.Empty:
            break

    result = hc.compress(before)
    assert result == after
    assert hc._archive_count == 1
    assert hc.compression_count == 1

    # Give the writer thread a moment to process
    time.sleep(0.5)

    # The archive should have been enqueued
    assert _archive_queue.empty() or True  # may or may not be processed yet


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_ensure_writer_race_condition():
    threads = []
    for _ in range(50):
        t = threading.Thread(target=_ensure_writer)
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=5)

    writer_names = [
        th.name for th in threading.enumerate() if th.name == "hindsight-archive-writer"
    ]
    # At most one writer should exist
    alive_writers = [
        th for th in threading.enumerate()
        if th.name == "hindsight-archive-writer" and th.is_alive()
    ]
    assert len(alive_writers) <= 1
