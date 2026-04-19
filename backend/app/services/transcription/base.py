"""Abstract base interface that every transcriber implementation must follow.

All transcribers (local Whisper, mock, etc.) inherit from BaseTranscriber
so the rest of the application can work with any backend without knowing
which one is active.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.models.domain import TranscriptSegment

# Callback type: called whenever a transcriber produces a new segment.
SegmentHandler = Callable[[TranscriptSegment], Awaitable[None]]


class BaseTranscriber:
    """Defines the lifecycle every transcriber must support:
    start -> push_audio (repeatedly) -> close.
    """

    async def start(self) -> None:  # pragma: no cover - interface only
        """Initialize resources (models, background workers, etc.)."""
        raise NotImplementedError

    async def wait_until_ready(self) -> dict[str, Any] | None:
        """Block until the transcriber is fully loaded; return runtime info."""
        return None

    async def push_audio(self, data: bytes, sample_rate: int) -> None:  # pragma: no cover - interface only
        """Feed raw audio bytes into the transcriber for processing."""
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface only
        """Flush remaining audio and release all resources."""
        raise NotImplementedError

    def runtime_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model, device, etc."""
        return {}
