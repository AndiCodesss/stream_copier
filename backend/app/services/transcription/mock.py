"""No-op transcriber used when audio capture is disabled.

When the application runs without a microphone or with transcription turned
off, this stub satisfies the BaseTranscriber interface without doing any
real work.  All methods return immediately.
"""

from __future__ import annotations

from app.services.transcription.base import BaseTranscriber


class NoopTranscriber(BaseTranscriber):
    """Silently discards all audio -- no model is loaded, nothing is transcribed."""

    async def start(self) -> None:
        return None

    async def push_audio(self, data: bytes, sample_rate: int) -> None:
        return None

    async def close(self) -> None:
        return None

