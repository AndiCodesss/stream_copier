from __future__ import annotations

from pathlib import Path

from app.models.domain import StreamSession


class SessionStore:
    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[StreamSession]:
        sessions: list[StreamSession] = []
        for path in sorted(self._base_dir.glob("*.json")):
            try:
                sessions.append(StreamSession.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return sessions

    def save(self, session: StreamSession) -> None:
        path = self._base_dir / f"{session.id}.json"
        path.write_text(session.model_dump_json(), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        path = self._base_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
