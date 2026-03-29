from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.models.domain import TranscriptSegment

SegmentHandler = Callable[[TranscriptSegment], Awaitable[None]]


class BaseTranscriber:
    async def start(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    async def wait_until_ready(self) -> dict[str, Any] | None:
        return None

    async def push_audio(self, data: bytes, sample_rate: int) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def runtime_info(self) -> dict[str, Any]:
        return {}
