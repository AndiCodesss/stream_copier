"""Audio resampling utility for converting microphone input to the sample rate
expected by the speech-to-text model (16 kHz mono PCM-16).
"""

from __future__ import annotations

import math

import numpy as np


# Matches Settings.speech_target_sample_rate default (16 000 Hz).
# The old constant was 24 000 which was inconsistent with the settings default;
# this corrects it. All production callers pass target_rate explicitly from
# Settings, so the default is only a fallback and the correction is safe.
TARGET_SAMPLE_RATE = 16_000


def resample_pcm16_mono(data: bytes, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Resample mono PCM-16 audio from source_rate to target_rate using linear interpolation.

    Returns the original bytes object unchanged when rates are equal (no copy).
    """
    if source_rate <= 0:
        raise ValueError("source_rate must be positive")
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if source_rate == target_rate:
        return data

    source = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if len(source) == 0:
        return b""

    ratio = target_rate / source_rate
    target_length = max(1, int(math.floor(len(source) * ratio)))

    # Build fractional source positions for all target samples at once.
    positions = np.arange(target_length, dtype=np.float64) / ratio
    left = positions.astype(np.int64)
    right = np.minimum(left + 1, len(source) - 1)
    frac = (positions - left).astype(np.float32)

    resampled = source[left] * (1.0 - frac) + source[right] * frac
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16).tobytes()
