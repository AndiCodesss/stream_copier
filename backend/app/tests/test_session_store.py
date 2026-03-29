from __future__ import annotations

from pathlib import Path

from app.models.domain import SessionConfig, StreamSession
from app.services.storage.session_store import SessionStore


def test_session_store_round_trip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = StreamSession(config=SessionConfig(source_name="Test Source"))

    store.save(session)
    restored = store.load_all()

    assert len(restored) == 1
    assert restored[0].id == session.id
    assert restored[0].config.source_name == "Test Source"


def test_session_store_delete_removes_snapshot(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = StreamSession(config=SessionConfig(source_name="Delete Me"))

    store.save(session)
    store.delete(session.id)

    assert store.load_all() == []
