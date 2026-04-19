"""Post-processing corrections for automatic speech recognition (ASR) errors.

Live speech-to-text engines frequently mishear trading jargon (e.g. "VWAP"
becomes "view up", "piece" becomes "peace"). This module applies safe,
pattern-based text replacements to fix these known errors before the
transcript reaches the interpretation pipeline.
"""

from __future__ import annotations

import re

# Each tuple maps a compiled regex of a known ASR mis-transcription to its
# correct trading term. Only high-confidence, unambiguous corrections are
# included to avoid introducing new errors.
_LOW_RISK_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bv\s*w\s*a\s*p\b"), "vwap"),
    (re.compile(r"\b(?:view|vw)\s+up\b"), "vwap"),
    (re.compile(r"\bbreak\s+even\b"), "breakeven"),
    (re.compile(r"\bbroke\s+even\b"), "breakeven"),
    (re.compile(r"\bgot my ad on\b"), "got my add on"),
    (re.compile(r"\bm\s+n\s+q\b"), "mnq"),
    (re.compile(r"\bn\s+q\b"), "nq"),
    (re.compile(r"\bm\s+e\s+s\b"), "mes"),
    (re.compile(r"\be\s+s\b"), "es"),
    (re.compile(r"\bpaying myself(?:\s+a)?\s+peace\b"), "paying myself piece"),
    (re.compile(r"\bpaying ourselves(?:\s+a)?\s+peace\b"), "paying ourselves piece"),
    (re.compile(r"\bpaid myself(?:\s+a)?\s+peace\b"), "paid myself piece"),
    (re.compile(r"\bpaid ourselves(?:\s+a)?\s+peace\b"), "paid ourselves piece"),
    (re.compile(r"\bpeace(?=\s+(?:on|here|there|at|versus|long|short)\b)"), "piece"),
)


def apply_trading_asr_corrections(text: str) -> str:
    corrected = text.lower()
    corrected = corrected.replace("’", "'")
    corrected = re.sub(r"(?<=\d),(?=\d)", "", corrected)
    for pattern, replacement in _LOW_RISK_REPLACEMENTS:
        corrected = pattern.sub(replacement, corrected)
    return corrected
