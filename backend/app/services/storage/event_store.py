"""Append-only file storage for timeline events.

Each session's events are stored as one JSON object per line (JSONL format)
in a separate file, enabling simple crash recovery and debugging.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.models.domain import TimelineEvent


class EventLogStore:
    """Persists timeline events to disk as JSONL files (one file per session)."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def append(self, event: TimelineEvent) -> None:
        path = self._base_dir / f"{event.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json")) + "\n")

    def delete(self, session_id: str) -> None:
        path = self._base_dir / f"{session_id}.jsonl"
        if path.exists():
            path.unlink()
