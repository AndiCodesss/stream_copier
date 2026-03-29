from __future__ import annotations

import math
from array import array

from app.services.transcription.segmenter import (
    EnergyBasedSpeechSegmenter,
    WebRtcVadSpeechSegmenter,
)


def test_energy_segmenter_emits_after_voice_followed_by_silence() -> None:
    segmenter = EnergyBasedSpeechSegmenter(
        sample_rate=16_000,
        energy_threshold=0.01,
        min_duration_ms=200,
        silence_duration_ms=200,
        max_duration_ms=2_000,
    )

    voice = _pcm_sine_wave(duration_ms=400, amplitude=0.3)
    silence = _pcm_silence(duration_ms=250)

    ready = []
    ready.extend(segmenter.push(voice))
    ready.extend(segmenter.push(silence))

    assert len(ready) == 1
    assert ready[0].duration_ms >= 400


def test_webrtc_segmenter_emits_after_voiced_frames_followed_by_silence() -> None:
    segmenter = WebRtcVadSpeechSegmenter(
        sample_rate=16_000,
        min_duration_ms=200,
        silence_duration_ms=200,
        max_duration_ms=2_000,
        aggressiveness=2,
        frame_ms=20,
        preroll_ms=60,
        start_ms=60,
        start_window_ms=80,
        vad=_FakeVad(),
    )

    audio = (
        _pcm_constant(duration_ms=20, amplitude=0)
        + _pcm_constant(duration_ms=80, amplitude=0.35)
        + _pcm_constant(duration_ms=320, amplitude=0.25)
        + _pcm_constant(duration_ms=240, amplitude=0)
    )

    ready = segmenter.push(audio)

    assert len(ready) == 1
    assert ready[0].voice_duration_ms >= 320
    assert ready[0].duration_ms >= 380


def test_webrtc_segmenter_snapshot_includes_partial_frame_tail() -> None:
    segmenter = WebRtcVadSpeechSegmenter(
        sample_rate=16_000,
        min_duration_ms=200,
        silence_duration_ms=200,
        max_duration_ms=2_000,
        aggressiveness=2,
        frame_ms=20,
        preroll_ms=40,
        start_ms=40,
        start_window_ms=60,
        vad=_FakeVad(),
    )

    audio = _pcm_constant(duration_ms=50, amplitude=0.3)
    segmenter.push(audio)
    snapshot = segmenter.snapshot()

    assert snapshot is not None
    assert snapshot.duration_ms >= 40
    assert snapshot.voice_duration_ms >= 40


class _FakeVad:
    def is_speech(self, frame: bytes, _sample_rate: int) -> bool:
        samples = array("h")
        samples.frombytes(frame)
        return any(abs(sample) > 0 for sample in samples)


def _pcm_sine_wave(*, duration_ms: int, amplitude: float) -> bytes:
    total_samples = int(16_000 * duration_ms / 1000)
    values = array("h")
    for index in range(total_samples):
        sample = math.sin(2.0 * math.pi * 440 * (index / 16_000))
        values.append(int(sample * amplitude * 32767))
    return values.tobytes()


def _pcm_constant(*, duration_ms: int, amplitude: float) -> bytes:
    total_samples = int(16_000 * duration_ms / 1000)
    value = int(amplitude * 32767)
    return array("h", [value] * total_samples).tobytes()


def _pcm_silence(*, duration_ms: int) -> bytes:
    total_samples = int(16_000 * duration_ms / 1000)
    return array("h", [0] * total_samples).tobytes()
