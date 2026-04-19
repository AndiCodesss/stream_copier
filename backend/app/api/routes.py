"""REST and WebSocket endpoint definitions.

Provides CRUD routes for sessions and two WebSocket endpoints:
one for streaming real-time session events to the UI, and one for
ingesting raw audio from the browser microphone.
"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from app.models.domain import (
    CreateSessionRequest,
    ManualTradeRequest,
    TextSegmentRequest,
    UpdateSessionConfigRequest,
)
from app.services.session_manager import SessionManager

_LOGGER = logging.getLogger(__name__)


def create_router(manager: SessionManager) -> APIRouter:
    """Build the REST API router with all session CRUD and trade endpoints."""
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "sessions": len(manager.list_sessions()),
            "transcription_backend": manager.settings.transcription_backend,
        }

    @router.get("/sessions")
    async def list_sessions() -> list[dict]:
        return [session.model_dump(mode="json") for session in manager.list_sessions()]

    @router.post("/sessions")
    async def create_session(request: CreateSessionRequest) -> dict:
        session = await manager.create_session(request)
        return session.model_dump(mode="json")

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        try:
            session = manager.get_session(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error
        return session.model_dump(mode="json")

    @router.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        try:
            await manager.delete_session(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error

    @router.patch("/sessions/{session_id}/config")
    async def update_session_config(session_id: str, request: UpdateSessionConfigRequest) -> dict:
        try:
            session = await manager.update_session_config(session_id, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error
        return session.model_dump(mode="json")

    @router.post("/sessions/{session_id}/segments")
    async def ingest_segment(session_id: str, request: TextSegmentRequest) -> dict:
        try:
            session = await manager.ingest_segment(session_id, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error
        return session.model_dump(mode="json")

    @router.post("/sessions/{session_id}/manual-trade")
    async def manual_trade(session_id: str, request: ManualTradeRequest) -> dict:
        try:
            session = await manager.manual_trade(session_id, request)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error
        return session.model_dump(mode="json")

    @router.get("/sessions/{session_id}/broker-state")
    async def get_broker_state(session_id: str, account: str | None = None, symbol: str | None = None) -> dict:
        try:
            return await manager.get_broker_state(session_id, account=account, symbol=symbol)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error

    return router


def attach_websockets(app: FastAPI, manager: SessionManager) -> None:
    """Register WebSocket endpoints directly on the app (outside the REST router)."""

    @app.websocket("/ws/sessions/{session_id}/events")
    async def session_events(websocket: WebSocket, session_id: str) -> None:
        """Push real-time session updates to the UI via pub/sub."""
        # Validate existence before accepting the connection.
        try:
            manager.get_session(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return

        await websocket.accept()

        # Subscribe BEFORE reading session state so the snapshot is guaranteed to be
        # at least as fresh as the first queued delta event. Any event published
        # between get_session and subscribe would be missed otherwise.
        queue = await manager.event_hub.subscribe(session_id)
        try:
            # Read session state after subscribing to close the race window.
            session = manager.get_session(session_id)
            await websocket.send_json({
                "type": "snapshot",
                "session": session.model_dump(mode="json"),
            })
            while True:
                message = await queue.get()
                await websocket.send_json(message)
        except WebSocketDisconnect:
            return
        finally:
            await manager.event_hub.unsubscribe(session_id, queue)

    @app.websocket("/ws/sessions/{session_id}/audio")
    async def audio_ingest(websocket: WebSocket, session_id: str) -> None:
        """Receive raw PCM audio chunks from the browser and feed them to the transcriber."""
        try:
            manager.get_session(session_id)
        except KeyError:
            await websocket.close(code=4404, reason="Capture session not found.")
            return

        await websocket.accept()
        sample_rate = 48_000
        try:
            while True:
                payload = await websocket.receive()

                if payload.get("type") == "websocket.disconnect":
                    return

                if payload.get("bytes") is not None:
                    await manager.push_audio(session_id, payload["bytes"], sample_rate)
                    continue

                text = payload.get("text")
                if text is None:
                    continue

                message = json.loads(text)
                # "audio_config" is sent once at start to negotiate sample rate.
                if message.get("type") == "audio_config":
                    sample_rate = int(message.get("sample_rate", 48_000))
                    await manager.ensure_transcriber(session_id)
                    continue

                if message.get("type") != "audio_chunk":
                    continue

                # Audio data arrives base64-encoded inside a JSON text frame.
                sample_rate = int(message.get("sample_rate", sample_rate))
                data = base64.b64decode(message["pcm_base64"])
                await manager.push_audio(session_id, data, sample_rate)
        except WebSocketDisconnect:
            return
        except Exception:
            _LOGGER.exception("Audio ingest error for session %s", session_id)
            await websocket.close(code=1011, reason="Audio ingest error.")
            return
