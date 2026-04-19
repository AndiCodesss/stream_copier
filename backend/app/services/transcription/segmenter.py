"""Speech boundary detection -- decides where one utterance ends and the next begins.

The transcriber needs discrete audio chunks (utterances) to feed to Whisper.
This module listens to a continuous audio stream and splits it into
utterances by detecting when the speaker starts and stops talking.

Two backends are provided:
- EnergyBasedSpeechSegmenter: simple RMS volume threshold (fast, no dependencies).
- WebRtcVadSpeechSegmenter: Google's WebRTC Voice Activity Detection (more accurate).

Both inherit from _BufferedSpeechSegmenter, which handles the common
buffering, finalization, and snapshot logic.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class ReadyAudioSegment:
    """A completed (or in-progress snapshot of an) utterance ready for transcription."""

    utterance_id: str
    pcm16: bytes
    duration_ms: int
    voice_duration_ms: int
    started_monotonic: float  # wall-clock time when the utterance started
    ready_monotonic: float    # wall-clock time when it was finalized/snapshotted


class _BufferedSpeechSegmenter:
    """Base class with shared buffering logic for all segmenter backends.

    Subclasses implement push() to classify each audio chunk as voice or
    silence, then call _start_utterance / _append_voice / _append_silence.
    This base class handles buffer management, finalization, and snapshots.
    """

    backend_name = "buffered"

    def __init__(
        self,
        *,
        sample_rate: int,
        min_duration_ms: int,
        silence_duration_ms: int,
        max_duration_ms: int,
    ) -> None:
        self._sample_rate = sample_rate
        self._min_duration_ms = min_duration_ms
        self._silence_duration_ms = silence_duration_ms
        self._max_duration_ms = max_duration_ms

        self._utterance_id: str | None = None
        self._utterance_started_monotonic: float | None = None
        self._buffer = bytearray()
        self._speaking = False
        self._voice_ms = 0.0
        self._silence_ms = 0.0

    def flush(self) -> list[ReadyAudioSegment]:
        """Finalize any in-progress utterance (called at session close)."""
        self._flush_pending_tail()
        segment = self._finalize()
        return [segment] if segment is not None else []

    def snapshot(self) -> ReadyAudioSegment | None:
        """Return a read-only copy of the current in-progress utterance for preview."""
        pcm16 = self._current_pcm16()
        if not self._speaking or self._utterance_id is None or not pcm16:
            return None
        started_monotonic = self._utterance_started_monotonic or time.monotonic()
        return ReadyAudioSegment(
            utterance_id=self._utterance_id,
            pcm16=pcm16,
            duration_ms=self._buffer_duration_ms(pcm16),
            voice_duration_ms=int(self._voice_ms),
            started_monotonic=started_monotonic,
            ready_monotonic=time.monotonic(),
        )

    def _start_utterance(self, *, initial_pcm16: bytes = b"", initial_voice_ms: float = 0.0) -> None:
        self._reset_state()
        self._utterance_id = uuid4().hex
        self._utterance_started_monotonic = time.monotonic()
        self._speaking = True
        if initial_pcm16:
            self._buffer.extend(initial_pcm16)
        self._voice_ms = initial_voice_ms
        self._silence_ms = 0.0

    def _append_voice(self, pcm16: bytes, *, duration_ms: float) -> None:
        self._buffer.extend(pcm16)
        self._voice_ms += duration_ms
        self._silence_ms = 0.0

    def _append_silence(self, pcm16: bytes, *, duration_ms: float) -> None:
        self._buffer.extend(pcm16)
        self._silence_ms += duration_ms

    def _should_finalize(self) -> bool:
        """End the utterance if silence exceeds the threshold or max duration is reached."""
        if not self._speaking:
            return False
        return self._buffer_duration_ms() >= self._max_duration_ms or self._silence_ms >= self._silence_duration_ms

    def _finalize(self) -> ReadyAudioSegment | None:
        self._flush_pending_tail()
        pcm16 = self._current_pcm16()
        try:
            if self._utterance_id is None or self._voice_ms < self._min_duration_ms or not pcm16:
                return None
            started_monotonic = self._utterance_started_monotonic or time.monotonic()
            return ReadyAudioSegment(
                utterance_id=self._utterance_id,
                pcm16=pcm16,
                duration_ms=self._buffer_duration_ms(pcm16),
                voice_duration_ms=int(self._voice_ms),
                started_monotonic=started_monotonic,
                ready_monotonic=time.monotonic(),
            )
        finally:
            self._reset_state()

    def _reset_state(self) -> None:
        self._buffer = bytearray()
        self._speaking = False
        self._voice_ms = 0.0
        self._silence_ms = 0.0
        self._utterance_id = None
        self._utterance_started_monotonic = None

    def _duration_ms(self, pcm16: bytes) -> float:
        samples = len(pcm16) / 2
        return (samples / self._sample_rate) * 1000.0

    def _buffer_duration_ms(self, pcm16: bytes | None = None) -> int:
        return int(self._duration_ms(pcm16 if pcm16 is not None else self._current_pcm16()))

    def _current_pcm16(self) -> bytes:
        return bytes(self._buffer)

    def _flush_pending_tail(self) -> None:
        return None


class EnergyBasedSpeechSegmenter(_BufferedSpeechSegmenter):
    """Detects speech by comparing RMS energy against a fixed threshold.

    Simple and fast -- no external dependencies.  Works well in quiet
    environments but struggles with background noise.
    """

    backend_name = "energy"

    def __init__(
        self,
        *,
        sample_rate: int,
        energy_threshold: float,
        min_duration_ms: int,
        silence_duration_ms: int,
        max_duration_ms: int,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            min_duration_ms=min_duration_ms,
            silence_duration_ms=silence_duration_ms,
            max_duration_ms=max_duration_ms,
        )
        self._energy_threshold = energy_threshold

    def push(self, data: bytes) -> list[ReadyAudioSegment]:
        """Feed raw PCM16 audio. Returns finalized segments (if any)."""
        if not data:
            return []

        duration_ms = self._duration_ms(data)
        # Classify chunk as voice or silence based on RMS energy.
        is_voice = self._energy(data) >= self._energy_threshold
        ready: list[ReadyAudioSegment] = []

        if is_voice:
            if not self._speaking:
                self._start_utterance()
            self._append_voice(data, duration_ms=duration_ms)
        elif self._speaking:
            self._append_silence(data, duration_ms=duration_ms)

        if self._should_finalize():
            segment = self._finalize()
            if segment is not None:
                ready.append(segment)

        return ready

    def _energy(self, data: bytes) -> float:
        """Compute normalized RMS energy (0.0 = silence, 1.0 = max amplitude)."""
        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return rms / 32768.0  # normalize to [0, 1]


class WebRtcVadSpeechSegmenter(_BufferedSpeechSegmenter):
    """Uses Google's WebRTC VAD for more robust voice/silence classification.

    Processes audio in fixed-size frames (10/20/30 ms).  A "start trigger"
    requires several voiced frames within a sliding window before an
    utterance begins, which prevents brief noise from being misclassified
    as speech.  A configurable pre-roll keeps audio from just before the
    trigger so the beginning of the utterance is not clipped.
    """

    backend_name = "webrtc"

    def __init__(
        self,
        *,
        sample_rate: int,
        min_duration_ms: int,
        silence_duration_ms: int,
        max_duration_ms: int,
        aggressiveness: int,
        frame_ms: int,
        preroll_ms: int,
        start_ms: int,
        start_window_ms: int,
        vad: object | None = None,
    ) -> None:
        super().__init__(
            sample_rate=sample_rate,
            min_duration_ms=min_duration_ms,
            silence_duration_ms=silence_duration_ms,
            max_duration_ms=max_duration_ms,
        )
        if frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD frame_ms must be 10, 20, or 30")

        frame_samples = (sample_rate * frame_ms) / 1000
        if int(frame_samples) != frame_samples:
            raise ValueError("WebRTC VAD frame size must align with the configured sample rate")

        self._frame_ms = float(frame_ms)
        self._frame_bytes = int(frame_samples) * 2  # 2 bytes per 16-bit sample
        self._pending_pcm16 = bytearray()
        # Pre-roll: how many past frames to include when an utterance starts.
        self._pre_roll_frames = max(0, math.ceil(preroll_ms / frame_ms))
        # Sliding window size for the start trigger.
        self._start_window_frames = max(1, math.ceil(start_window_ms / frame_ms))
        # How many voiced frames within the window are needed to trigger start.
        self._start_trigger_frames = max(1, math.ceil(start_ms / frame_ms))
        # Ring buffer of recent frames used for start detection and pre-roll.
        self._recent_frames: deque[tuple[bytes, bool]] = deque(
            maxlen=max(self._pre_roll_frames, self._start_window_frames, self._start_trigger_frames)
        )
        self._vad = vad or _build_webrtc_vad(aggressiveness)

    def push(self, data: bytes) -> list[ReadyAudioSegment]:
        """Feed raw PCM16 audio. Splits into fixed-size frames for the VAD."""
        if not data:
            return []

        self._pending_pcm16.extend(data)
        ready: list[ReadyAudioSegment] = []

        # Process as many complete frames as possible.
        while len(self._pending_pcm16) >= self._frame_bytes:
            frame = bytes(self._pending_pcm16[: self._frame_bytes])
            del self._pending_pcm16[: self._frame_bytes]
            segment = self._consume_frame(frame)
            if segment is not None:
                ready.append(segment)

        return ready

    def _consume_frame(self, frame: bytes) -> ReadyAudioSegment | None:
        is_voice = bool(self._vad.is_speech(frame, self._sample_rate))

        if not self._speaking:
            self._recent_frames.append((frame, is_voice))
            if not self._should_start_utterance():
                return None

            initial_frames = list(self._recent_frames)
            if self._pre_roll_frames > 0:
                initial_frames = initial_frames[-self._pre_roll_frames :]
            initial_pcm16 = b"".join(chunk for chunk, _ in initial_frames)
            initial_voice_ms = self._frame_ms * sum(1 for _, voiced in initial_frames if voiced)
            self._start_utterance(initial_pcm16=initial_pcm16, initial_voice_ms=initial_voice_ms)
            self._recent_frames.clear()
            return None

        if is_voice:
            self._append_voice(frame, duration_ms=self._frame_ms)
        else:
            self._append_silence(frame, duration_ms=self._frame_ms)

        if not self._should_finalize():
            return None
        return self._finalize()

    def _should_start_utterance(self) -> bool:
        """True when enough voiced frames appear in the recent sliding window.

        This prevents isolated noise spikes from triggering a new utterance.
        """
        window = list(self._recent_frames)[-self._start_window_frames :]
        if len(window) < self._start_trigger_frames:
            return False
        voiced_frames = sum(1 for _, is_voice in window if is_voice)
        return voiced_frames >= self._start_trigger_frames

    def _current_pcm16(self) -> bytes:
        if not self._speaking or not self._pending_pcm16:
            return bytes(self._buffer)
        return bytes(self._buffer) + bytes(self._pending_pcm16)

    def _flush_pending_tail(self) -> None:
        if self._pending_pcm16 and self._speaking:
            self._buffer.extend(self._pending_pcm16)
        self._pending_pcm16 = bytearray()

    def _reset_state(self) -> None:
        super()._reset_state()
        self._recent_frames.clear()


def build_speech_segmenter(
    *,
    sample_rate: int,
    energy_threshold: float,
    min_duration_ms: int,
    silence_duration_ms: int,
    max_duration_ms: int,
    vad_backend: str,
    vad_aggressiveness: int,
    vad_frame_ms: int,
    vad_preroll_ms: int,
    vad_start_ms: int,
    vad_start_window_ms: int,
) -> EnergyBasedSpeechSegmenter | WebRtcVadSpeechSegmenter:
    """Factory: create the right segmenter backend based on config.

    Tries WebRTC first if requested; falls back to energy-based if it fails.
    """
    backend = vad_backend.strip().lower()
    if backend == "webrtc":
        try:
            return WebRtcVadSpeechSegmenter(
                sample_rate=sample_rate,
                min_duration_ms=min_duration_ms,
                silence_duration_ms=silence_duration_ms,
                max_duration_ms=max_duration_ms,
                aggressiveness=vad_aggressiveness,
                frame_ms=vad_frame_ms,
                preroll_ms=vad_preroll_ms,
                start_ms=vad_start_ms,
                start_window_ms=vad_start_window_ms,
            )
        except RuntimeError as error:
            _LOGGER.warning("Falling back to energy VAD: %s", error)
            backend = "energy"
    if backend == "energy":
        return EnergyBasedSpeechSegmenter(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            min_duration_ms=min_duration_ms,
            silence_duration_ms=silence_duration_ms,
            max_duration_ms=max_duration_ms,
        )
    raise ValueError(f"Unsupported speech VAD backend: {vad_backend}")


def _build_webrtc_vad(aggressiveness: int) -> object:
    try:
        import webrtcvad
    except ModuleNotFoundError as error:  # pragma: no cover - dependency wiring
        raise RuntimeError(
            "WebRTC VAD is configured but the 'webrtcvad' module is not installed"
        ) from error

    return webrtcvad.Vad(int(aggressiveness))
