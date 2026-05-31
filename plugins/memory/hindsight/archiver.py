"""Lightweight compression archiver for Hindsight memory provider.

Replaces the heavy JSON-per-compression anti-pattern with a small,
keyword-routed text summary retained to the most relevant bank(s).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Minimum number of compressible turns to bother archiving
_MIN_ARCHIVE_TURNS = 4
# Cooldown between archives (seconds)
_ARCHIVE_COOLDOWN_SECONDS = 300

# Bank routing keywords (must match keys in Hindsight config)
_BANK_ROUTE_KEYWORDS: dict[str, list[str]] = {
    "projects-active": [
        "code", "pr", "commit", "build", "deploy", "docker", "nginx",
        "fastapi", "dashboard", "project", "api", "frontend", "backend",
        "database", "postgres", "redis", "minio", "git", "github",
    ],
    "boss-profile": [
        "user", "boss", "preference", "schedule", "like", "dislike",
        "mahamudul", "habit", "routine", "favorite", "prefer", "want",
    ],
    "failures-lessons": [
        "bug", "error", "mistake", "lesson", "fix", "broken", "crash",
        "incident", "fail", "failure", "problem", "issue", "trouble",
        "rm -rf", "timeout", "slow", "degraded",
    ],
    "relationship": [
        "relationship", "us", "we", "history", "milestone", "intimate",
        "together", "experience", "shared", "moment", "memory",
    ],
    "lena-persona": [
        "lena", "secretary", "persona", "roleplay", "character", "dominant",
        "submissive", "breeding", "creampie", "milf", "aesthetic",
    ],
    "hermes": [
        "hermes", "agent", "config", "setup", "install", "plugin", "skill",
    ],
}


def _extract_summary(messages: list[dict[str, Any]]) -> str:
    """Extract a lightweight text summary from compressible messages."""
    texts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        # Skip system/tool/empty messages
        if role in ("system", "tool") or not content:
            continue
        # Truncate very long individual messages
        text = content.strip()
        if len(text) > 400:
            text = text[:400] + " …"
        texts.append(f"{role.capitalize()}: {text}")
    return "\n".join(texts)


def _route_banks(summary: str) -> list[str]:
    """Return bank IDs that match keywords in the summary."""
    summary_lower = summary.lower()
    matched: list[str] = []
    for bank_id, keywords in _BANK_ROUTE_KEYWORDS.items():
        if any(kw in summary_lower for kw in keywords):
            matched.append(bank_id)
    # Always fall back to hermes if nothing matched
    if not matched:
        matched.append("hermes")
    return matched


class CompressArchiver:
    """Stateful helper that archives compression summaries to Hindsight."""

    def __init__(self) -> None:
        self._last_archive_time: float = 0.0

    def should_archive(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if this compression block is worth archiving."""
        if len(messages) < _MIN_ARCHIVE_TURNS:
            return False
        if time.time() - self._last_archive_time < _ARCHIVE_COOLDOWN_SECONDS:
            return False
        return True

    def build_archive_item(
        self, messages: list[dict[str, Any]], session_id: str = ""
    ) -> tuple[str, list[str]] | None:
        """Build (summary_text, target_bank_ids) or None if skipped."""
        if not self.should_archive(messages):
            return None

        summary = _extract_summary(messages)
        if not summary:
            return None

        banks = _route_banks(summary)

        tags = ["context-compression", "session-archive"]
        if session_id:
            tags.append(f"session:{session_id}")

        # Prefix with tags so Hindsight can index them
        tagged_summary = f"Tags: {', '.join(tags)}\n\n{summary}"

        self._last_archive_time = time.time()
        logger.debug(
            "CompressArchiver: %d messages -> %d chars -> banks=%s",
            len(messages), len(tagged_summary), banks,
        )
        return tagged_summary, banks
