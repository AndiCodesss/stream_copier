"""In-memory publish/subscribe hub for pushing real-time session updates
to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventHub:
    """In-process pub/sub for WebSocket session events.

    Publishes pre-serialized JSON-compatible dicts so the WebSocket route
    handler can call ``send_json`` directly without any further processing.
    """

    def __init__(self) -> None:
        # Each session_id maps to a set of queues, one per connected WebSocket.
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Create a new queue for a WebSocket client and register it."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._subscribers[session_id].add(queue)
        return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers[session_id].discard(queue)
            if not self._subscribers[session_id]:
                self._subscribers.pop(session_id, None)

    async def publish(self, session_id: str, message: dict[str, Any]) -> None:
        """Send a message to every WebSocket client subscribed to this session."""
        # Snapshot the queue set under the lock, then write outside it
        # to avoid holding the lock while awaiting queue.put().
        async with self._lock:
            queues = list(self._subscribers.get(session_id, set()))
        for queue in queues:
            await queue.put(message)

