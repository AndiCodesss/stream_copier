"""JSON file storage for session snapshots.

Each session is saved as a single JSON file. On startup, all session files
are loaded back into memory so the application can resume where it left off.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.models.domain import StreamSession

_LOGGER = logging.getLogger(__name__)


class SessionStore:
    """Persists full session state to disk as JSON files (one file per session)."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[StreamSession]:
        """Read all saved sessions from disk. Skips corrupted files with a warning."""
        sessions: list[StreamSession] = []
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                sessions.append(StreamSession.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                _LOGGER.warning("Failed to load session from %s", path, exc_info=True)
                continue
        return sessions

    def save(self, session: StreamSession) -> None:
        path = self._base_dir / f"{session.id}.json"
        path.write_text(session.model_dump_json(), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        path = self._base_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
