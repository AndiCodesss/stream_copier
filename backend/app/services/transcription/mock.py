from __future__ import annotations

from app.services.transcription.base import BaseTranscriber


class NoopTranscriber(BaseTranscriber):
    async def start(self) -> None:
        return None

    async def push_audio(self, data: bytes, sample_rate: int) -> None:
        return None

    async def close(self) -> None:
        return None

