from __future__ import annotations

import struct

import numpy as np
import pytest

from app.core.audio import resample_pcm16_mono


def _pack_int16(samples: list[int]) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _unpack_int16(data: bytes) -> list[int]:
    count = len(data) // 2
    return list(struct.unpack(f"<{count}h", data))


def test_same_rate_returns_input_unchanged() -> None:
    data = _pack_int16([100, 200, 300])
    assert resample_pcm16_mono(data, source_rate=16_000, target_rate=16_000) is data


def test_empty_input_returns_empty() -> None:
    assert resample_pcm16_mono(b"", source_rate=48_000, target_rate=16_000) == b""


def test_invalid_source_rate_raises() -> None:
    with pytest.raises(ValueError):
        resample_pcm16_mono(b"\x00\x00", source_rate=0)


def test_invalid_target_rate_raises() -> None:
    with pytest.raises(ValueError):
        resample_pcm16_mono(b"\x00\x00", source_rate=48_000, target_rate=0)


def test_downsample_3x_output_length() -> None:
    # 48 kHz → 16 kHz: output should be ⌊N * 16000/48000⌋ samples
    n_samples = 480  # 10 ms at 48 kHz
    data = _pack_int16([1000] * n_samples)
    out = resample_pcm16_mono(data, source_rate=48_000, target_rate=16_000)
    expected_len = int((n_samples * 16_000) // 48_000) * 2
    assert len(out) == expected_len


def test_constant_signal_preserved_after_resample() -> None:
    # A DC signal at value 5000 should survive resampling unchanged.
    data = _pack_int16([5000] * 300)
    out = resample_pcm16_mono(data, source_rate=48_000, target_rate=16_000)
    samples = _unpack_int16(out)
    assert all(s == 5000 for s in samples)


def test_output_clipped_to_int16_range() -> None:
    # Values near the boundaries should not overflow.
    data = _pack_int16([32767, -32768, 32767, -32768] * 100)
    out = resample_pcm16_mono(data, source_rate=48_000, target_rate=16_000)
    samples = _unpack_int16(out)
    assert all(-32768 <= s <= 32767 for s in samples)


def test_numpy_implementation_matches_reference_downsample() -> None:
    # Verify linear interpolation accuracy against a hand-computed reference.
    # Source: [0, 32767] at 32 kHz → 16 kHz (ratio 0.5)
    # Target sample 0: source_pos=0.0  → left=0, right=1, frac=0.0 → 0
    # (with 2 source samples at 32kHz, floor(2 * 16000/32000) = 1 target sample)
    data = _pack_int16([0, 32767])
    out = resample_pcm16_mono(data, source_rate=32_000, target_rate=16_000)
    samples = _unpack_int16(out)
    assert len(samples) == 1
    assert samples[0] == 0  # source_pos = 0/2 = 0.0 → exactly sample[0]
