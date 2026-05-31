"""Hindsight Context Archive Engine.

Wraps the built-in ContextCompressor with Hindsight archival.
Before compression discards messages, they are retained to Hindsight
so the full conversation remains recoverable.

Activate with:
    context:
      engine: hindsight_archive
"""

import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from agent.context_engine import ContextEngine
from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_MESSAGE_LEN = 4096          # per-message truncation
_MAX_TRANSCRIPT_LEN = 65536      # total transcript truncation
_MAX_BASE64_INLINE = 100           # truncate inline base64 blobs
_MAX_TOOL_ARG_LEN = 200            # truncate tool call arguments
_ARCHIVE_RETRY_ATTEMPTS = 3
_ARCHIVE_RETRY_BASE_DELAY = 1.0    # seconds
_ARCHIVE_QUEUE_FLUSH_TIMEOUT = 30  # seconds on session end
_ARCHIVE_QUEUE_MAX_SIZE = 1000     # bounded to prevent unbounded growth

# ---------------------------------------------------------------------------
# Hindsight config helpers (mirrors plugins/memory/hindsight/__init__.py)
# ---------------------------------------------------------------------------


def _load_hindsight_config() -> dict:
    """Load Hindsight config from profile-scoped path, legacy path, or env vars."""
    profile_path = get_hermes_home() / "hindsight" / "config.json"
    if profile_path.exists():
        try:
            return json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    legacy_path = Path.home() / ".hindsight" / "config.json"
    if legacy_path.exists():
        try:
            return json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {}


def _resolve_archive_bank_id(cfg: dict) -> str:
    """Return the dedicated archive bank ID, with optional profile template."""
    template = cfg.get("archive_bank_id_template", "session-archive-{profile}")
    profile = os.environ.get("HERMES_PROFILE", "default")
    return template.replace("{profile}", profile)


def _get_hindsight_credentials() -> tuple[str, str, str]:
    """Return (api_url, api_key, bank_id) from config/env."""
    cfg = _load_hindsight_config()
    api_url = (cfg.get("api_url") or "http://localhost:8888").rstrip("/")
    api_key = cfg.get("apiKey") or cfg.get("api_key", "")
    bank_id = _resolve_archive_bank_id(cfg)
    return api_url, api_key, bank_id


def _deadletter_path() -> Path:
    """Return the deadletter file path, computed at runtime for profile safety."""
    return get_hermes_home() / "logs" / "hindsight_archive_deadletter.jsonl"


# ---------------------------------------------------------------------------
# Background archival queue (single writer thread, fire-and-forget)
# ---------------------------------------------------------------------------

_ArchiveJob = Dict[str, Any]
_archive_queue: queue.Queue = queue.Queue(maxsize=_ARCHIVE_QUEUE_MAX_SIZE)
_archive_writer_thread: Optional[threading.Thread] = None
_writer_spawn_lock = threading.Lock()


def _archive_writer_loop() -> None:
    """Daemon thread that drains the archival queue."""
    while True:
        job = _archive_queue.get()
        if job is None:  # global sentinel for shutdown
            _archive_queue.task_done()
            break
        # Per-job sentinel for flush
        if job.get("__sentinel__"):
            event = job.get("__event__")
            if event:
                event.set()
            _archive_queue.task_done()
            continue
        try:
            _do_archive(
                job["messages"],
                job["session_id"],
                job["archive_num"],
                job.get("consecutive_failures_ref"),
                job.get("last_failure_reason_ref"),
            )
        except Exception as exc:
            logger.debug("hindsight_archive: writer failed: %s", exc)
        finally:
            _archive_queue.task_done()


def _ensure_writer() -> None:
    """Start the background writer thread if not already running."""
    global _archive_writer_thread
    with _writer_spawn_lock:
        if _archive_writer_thread is not None and _archive_writer_thread.is_alive():
            return
        _archive_writer_thread = threading.Thread(
            target=_archive_writer_loop,
            daemon=True,
            name="hindsight-archive-writer",
        )
        _archive_writer_thread.start()


def _enqueue_archive(
    messages: List[Dict[str, Any]],
    session_id: str,
    archive_num: int,
    failures_ref: Optional[List] = None,
    reason_ref: Optional[List] = None,
) -> None:
    """Queue messages for background archival."""
    _ensure_writer()
    job = {
        "messages": messages,
        "session_id": session_id,
        "archive_num": archive_num,
        "consecutive_failures_ref": failures_ref,
        "last_failure_reason_ref": reason_ref,
    }
    try:
        _archive_queue.put_nowait(job)
    except queue.Full:
        logger.warning(
            "hindsight_archive: queue full (max %d), dropping archive #%d",
            _ARCHIVE_QUEUE_MAX_SIZE, archive_num,
        )


def _flush_archive_queue(timeout: float = _ARCHIVE_QUEUE_FLUSH_TIMEOUT) -> None:
    """Block until the archive queue is drained or timeout elapses."""
    done_event = threading.Event()
    try:
        _archive_queue.put_nowait({"__sentinel__": True, "__event__": done_event})
    except queue.Full:
        logger.warning("hindsight_archive: queue full during flush, cannot send sentinel")
        return
    done_event.wait(timeout=timeout)
    if not done_event.is_set():
        logger.warning("hindsight_archive: queue flush timed out after %.1fs", timeout)


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------


def _truncate_text(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding a truncation indicator."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _sanitize_inline_secrets(text: str) -> str:
    """Pre-redaction regex pass for common credential patterns."""
    # Base64 blobs
    text = re.sub(r"[A-Za-z0-9+/]{100,}={0,2}", "[base64:truncated]", text)
    # Common key patterns
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[A-Za-z0-9_\-]{16,}['\"]?", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(password\s*[:=]\s*)['\"]?[^\s'\"]{4,}['\"]?", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(token\s*[:=]\s*)['\"]?[A-Za-z0-9_\-]{8,}['\"]?", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(secret\s*[:=]\s*)['\"]?[A-Za-z0-9_\-]{8,}['\"]?", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)['\"]?[A-Za-z0-9_\-\.]{8,}['\"]?", r"\1[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9]{20,}", "[REDACTED_SK]", text)
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]", text)
    return text


# ---------------------------------------------------------------------------
# Core archival logic
# ---------------------------------------------------------------------------


def _format_message(msg: Dict[str, Any]) -> str:
    """Best-effort formatting of a conversation message for archival."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    if content is None:
        content = ""
    elif isinstance(content, list):
        # Multimodal / tool-result lists → compact JSON
        parts = []
        for p in content:
            if isinstance(p, dict):
                ptype = p.get("type", "")
                if ptype in {"image_url", "input_image", "image"}:
                    parts.append("[image]")
                else:
                    text = p.get("text", "")
                    if text:
                        parts.append(_truncate_text(text, _MAX_MESSAGE_LEN))
            elif isinstance(p, str):
                parts.append(_truncate_text(p, _MAX_MESSAGE_LEN))
        content = " ".join(parts)
    elif not isinstance(content, str):
        content = str(content)

    # Tool calls — sanitize arguments BEFORE truncation
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        tc_summary = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if args:
                    args = _sanitize_inline_secrets(str(args))
                    args = _truncate_text(args, _MAX_TOOL_ARG_LEN)
                tc_summary.append(f"tool_call:{name}({args})")
            else:
                tc_summary.append("[tool_call]")
        if tc_summary:
            content = f"{content}\n{' '.join(tc_summary)}"

    # Tool results
    tool_call_id = msg.get("tool_call_id", "")
    if tool_call_id and role == "tool":
        content = f"result for {tool_call_id}: {content}"

    # Redact secrets BEFORE truncation so split secrets don't leak
    content = _sanitize_inline_secrets(content)
    content = redact_sensitive_text(content, force=True)
    content = _truncate_text(content, _MAX_MESSAGE_LEN)
    return f"[{role}] {content.strip()}"


def _do_archive(
    messages: List[Dict[str, Any]],
    session_id: str,
    archive_num: int,
    failures_ref: Optional[List] = None,
    reason_ref: Optional[List] = None,
) -> None:
    """Synchronous archival to Hindsight via REST with retry and dead-letter."""
    api_url, api_key, bank_id = _get_hindsight_credentials()

    # Build transcript
    lines = [_format_message(m) for m in messages]
    transcript = "\n".join(lines)
    transcript = _truncate_text(transcript, _MAX_TRANSCRIPT_LEN)

    if not transcript.strip():
        logger.debug("hindsight_archive: transcript empty after redaction, skipping")
        return

    # Construct payload matching Hindsight RetainRequest schema
    payload = {
        "items": [
            {
                "content": transcript,
                "context": "Archived conversation context before compression",
                "tags": ["archive", "context-compression", f"archive:{archive_num}"],
                "metadata": {
                    "session_id": session_id or "",
                    "archive_number": str(archive_num),
                    "message_count": str(len(messages)),
                },
            }
        ],
        "async": True,
    }

    url = f"{api_url}/v1/default/banks/{bank_id}/memories"
    body = json.dumps(payload).encode("utf-8")

    last_exc: Optional[Exception] = None
    for attempt in range(_ARCHIVE_RETRY_ATTEMPTS):
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                logger.debug(
                    "hindsight_archive: archived %d messages (archive #%d, session=%s), status=%d",
                    len(messages), archive_num, session_id, resp.status,
                )
                if failures_ref is not None:
                    failures_ref[0] = 0
                if reason_ref is not None:
                    reason_ref[0] = None
                return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # 4xx → don't retry; 5xx → retry
            if 400 <= exc.code < 500:
                break
            time.sleep(_ARCHIVE_RETRY_BASE_DELAY * (2 ** attempt))
        except Exception as exc:
            last_exc = exc
            time.sleep(_ARCHIVE_RETRY_BASE_DELAY * (2 ** attempt))

    # All retries exhausted → dead-letter
    reason = f"HTTP {last_exc.code} {last_exc.reason}" if isinstance(last_exc, urllib.error.HTTPError) else str(last_exc)

    if failures_ref is not None:
        failures_ref[0] = failures_ref[0] + 1
    if reason_ref is not None:
        reason_ref[0] = reason

    consecutive = failures_ref[0] if failures_ref else 1
    log_fn = logger.warning if consecutive >= 3 else logger.debug
    log_fn(
        "hindsight_archive: archive #%d failed after %d attempts (%s)",
        archive_num, _ARCHIVE_RETRY_ATTEMPTS, reason,
    )

    try:
        dl_path = _deadletter_path()
        dl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dl_path, "a", encoding="utf-8") as f:
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "session_id": session_id or "",
                "archive_num": archive_num,
                "bank_id": bank_id,
                "reason": reason,
                "message_count": len(messages),
                "truncated_transcript": transcript[:2000],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as dl_exc:
        logger.debug("hindsight_archive: dead-letter write failed: %s", dl_exc)


# ---------------------------------------------------------------------------
# Diff helper — compute exactly what the compressor discarded
# ---------------------------------------------------------------------------

def _message_key(msg: Dict[str, Any]) -> tuple:
    """Return a hashable key for multiset diffing."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    if isinstance(content, list):
        content = json.dumps(content, sort_keys=True, ensure_ascii=False)
    else:
        content = str(content)
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        content += "|tc:" + json.dumps(tool_calls, sort_keys=True, ensure_ascii=False)
    tool_call_id = msg.get("tool_call_id", "")
    if tool_call_id:
        content += "|tid:" + str(tool_call_id)
    return (role, content)


def _compute_discarded(
    before: List[Dict[str, Any]], after: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return messages from before that don't appear in after (including tool prunes)."""
    after_counts = Counter(_message_key(m) for m in after)
    discarded = []
    for m in before:
        key = _message_key(m)
        if after_counts[key] > 0:
            after_counts[key] -= 1
        else:
            discarded.append(m)
    return discarded


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------

class HindsightArchiveCompressor(ContextEngine):
    """Context engine that archives to Hindsight before compressing."""

    def __init__(self):
        super().__init__()
        self._inner: Optional[ContextCompressor] = None
        self._archive_count = 0
        self._session_id = ""
        # Instance-level failure tracking (prevents cross-session pollution)
        self._consecutive_failures = 0
        self._last_failure_reason: Optional[str] = None

    def _ensure_inner(self) -> ContextCompressor:
        if self._inner is None:
            # Lazy creation with a placeholder model; update_model() will
            # replace with the real model parameters immediately after init.
            self._inner = ContextCompressor(model="gpt-4o-mini")
        return self._inner

    @property
    def name(self) -> str:
        return "hindsight_archive"

    @staticmethod
    def is_available() -> bool:
        """Return True when Hindsight is configured and reachable."""
        cfg = _load_hindsight_config()
        mode = cfg.get("mode", "")
        if mode not in {"local_external", "local_embedded", "cloud"}:
            return False

        api_url = (cfg.get("api_url") or "http://localhost:8888").rstrip("/")
        try:
            req = urllib.request.Request(f"{api_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status in {200, 204}
        except urllib.error.HTTPError as exc:
            # Some Hindsight versions return 404 on /health but are alive
            return exc.code in {200, 204, 404}
        except Exception:
            # Final fallback: if mode is local and we can't reach it, still
            # return True because embedded daemon auto-starts on first use.
            return mode in {"local_external", "local_embedded"}

    # -- Token tracking ----------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        inner = self._ensure_inner()
        inner.update_from_response(usage)
        self.last_prompt_tokens = inner.last_prompt_tokens
        self.last_completion_tokens = inner.last_completion_tokens
        self.last_total_tokens = inner.last_total_tokens
        self.compression_count = inner.compression_count

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return self._ensure_inner().should_compress(prompt_tokens)

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        return self._ensure_inner().should_compress_preflight(messages)

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        return self._ensure_inner().should_defer_preflight_to_real_usage(rough_tokens)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return self._ensure_inner().has_content_to_compress(messages)

    # -- Compression with archival -------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        # Compress first, then diff to find exactly what was discarded.
        # This is more accurate than guessing boundaries because it captures
        # tool-result pruning and any other mutations the compressor makes.
        result = self._ensure_inner().compress(
            messages, current_tokens=current_tokens, focus_topic=focus_topic
        )
        self.compression_count = self._ensure_inner().compression_count

        discarded = _compute_discarded(messages, result)
        if discarded:
            self._archive_count += 1
            _enqueue_archive(
                discarded,
                self._session_id,
                self._archive_count,
                failures_ref=[self._consecutive_failures],
                reason_ref=[self._last_failure_reason],
            )

        return result

    # -- Session lifecycle -------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = str(session_id) if session_id else ""
        self._ensure_inner().on_session_start(session_id, **kwargs)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        # Attempt to flush any pending archives before shutting down
        _flush_archive_queue(timeout=_ARCHIVE_QUEUE_FLUSH_TIMEOUT)
        self._ensure_inner().on_session_end(session_id, messages)

    def on_session_reset(self) -> None:
        if self._inner is not None:
            self._inner.on_session_reset()
        super().on_session_reset()
        self._archive_count = 0
        self._session_id = ""
        self._consecutive_failures = 0
        self._last_failure_reason = None

    # -- Model updates -----------------------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        # Prefer creating the inner compressor with real params if we haven't yet.
        if self._inner is None:
            self._inner = ContextCompressor(
                model=model,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                api_mode=api_mode,
                config_context_length=context_length,
            )
        else:
            self._inner.update_model(
                model, context_length, base_url, api_key, provider, api_mode
            )
        self.context_length = self._inner.context_length
        self.threshold_tokens = self._inner.threshold_tokens
        # Synchronize tail protection so archive block aligns with compressor
        self.protect_last_n = self._inner.protect_last_n

    # -- Status / tools ----------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        status = self._ensure_inner().get_status()
        _, _, bank_id = _get_hindsight_credentials()
        status.update({
            "archive_count": self._archive_count,
            "engine": self.name,
            "archive_bank_id": bank_id,
            "queue_size": _archive_queue.qsize(),
            "queue_max_size": _ARCHIVE_QUEUE_MAX_SIZE,
            "consecutive_failures": self._consecutive_failures,
            "last_failure_reason": self._last_failure_reason,
            "is_available": self.is_available(),
            "deadletter_path": str(_deadletter_path()),
        })
        return status


# ---------------------------------------------------------------------------
# Plugin registration (preferred path; class fallback remains for compatibility)
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register this context engine with the plugin system."""
    ctx.register_context_engine(HindsightArchiveCompressor())
