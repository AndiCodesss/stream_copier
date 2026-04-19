"""Preview text stabilization for streaming transcription.

Whisper re-transcribes the growing audio buffer on every preview cycle, so
the raw output can flicker (words appear, disappear, change).  This module
smooths the preview text by tracking which words have been stable long
enough to "commit" and only displaying new words beyond that stable prefix.

Key idea: once a word has appeared in the same position across consecutive
previews, it is considered committed and will not be removed even if a
later preview omits it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Strips leading/trailing punctuation so "hello," and "hello" match.
_EDGE_PUNCTUATION_RE = re.compile(r"(^[^\w$]+|[^\w$]+$)")


@dataclass
class _StreamingPreviewState:
    """Tracks committed and displayed words for one utterance."""

    utterance_id: str
    committed_words: list[str] = field(default_factory=list)
    last_display_words: list[str] = field(default_factory=list)


class StreamingPreviewAssembler:
    """Reduces preview flicker by incrementally committing stable words.

    Works at the word level: each new preview is merged with previously
    committed words.  Words that survive across consecutive previews
    get committed and become permanent.
    """

    def __init__(self, *, context_words: int, stability_margin_words: int) -> None:
        self._context_words = max(0, context_words)
        # How many words at the end stay uncommitted to absorb Whisper corrections.
        self._stability_margin_words = max(0, stability_margin_words)
        self._state: _StreamingPreviewState | None = None

    def reset(self) -> None:
        self._state = None

    def build_prompt(self, *, utterance_id: str, base_prompt: str) -> str:
        """Build a Whisper prompt that includes recently committed words.

        Feeding committed words back as prompt context helps Whisper
        stay consistent with what it already transcribed.
        """
        state = self._ensure_state(utterance_id)
        if self._context_words == 0 or not state.committed_words:
            return base_prompt

        context = self._join_words(state.committed_words[-self._context_words :])
        if not base_prompt:
            return context
        return f"{base_prompt}\nRecent confirmed transcript: {context}"

    def stabilize(self, *, utterance_id: str, tail_text: str) -> str:
        """Merge new preview text with committed words and return the display string.

        Steps:
        1. Merge the new tail onto committed words (finding any overlap).
        2. Compare with the previous display to find a shared prefix.
        3. Commit words that have been stable (shared prefix minus margin).
        """
        state = self._ensure_state(utterance_id)
        tail_words = self._split_words(tail_text)
        if not tail_words:
            state.last_display_words = list(state.committed_words)
            return self._join_words(state.last_display_words)

        candidate_words = self._merge_overlap(state.committed_words, tail_words)
        if state.last_display_words:
            common_prefix_len = self._common_prefix_len(state.last_display_words, candidate_words)
            # Only commit words that survived across previews, minus a safety margin.
            commit_target = max(len(state.committed_words), common_prefix_len - self._stability_margin_words)
            if commit_target > len(state.committed_words):
                state.committed_words = list(candidate_words[:commit_target])
                candidate_words = self._merge_overlap(state.committed_words, tail_words)

        state.last_display_words = list(candidate_words)
        return self._join_words(state.last_display_words)

    def committed_text(self, *, utterance_id: str) -> str:
        """Return only the words that have been permanently committed."""
        state = self._ensure_state(utterance_id)
        return self._join_words(state.committed_words)

    def _ensure_state(self, utterance_id: str) -> _StreamingPreviewState:
        state = self._state
        if state is None or state.utterance_id != utterance_id:
            state = _StreamingPreviewState(utterance_id=utterance_id)
            self._state = state
        return state

    @staticmethod
    def _split_words(text: str) -> list[str]:
        return [word for word in text.split() if word]

    @classmethod
    def _join_words(cls, words: list[str]) -> str:
        return " ".join(words).strip()

    @classmethod
    def _merge_overlap(cls, prefix_words: list[str], incoming_words: list[str]) -> list[str]:
        """Concatenate two word lists, deduplicating any overlapping region.

        E.g. prefix=["the","quick"] + incoming=["quick","fox"] -> ["the","quick","fox"]
        """
        if not prefix_words:
            return list(incoming_words)
        if not incoming_words:
            return list(prefix_words)

        # Find the longest suffix of prefix that matches a prefix of incoming.
        max_overlap = min(len(prefix_words), len(incoming_words))
        overlap = 0
        for size in range(max_overlap, 0, -1):
            if cls._normalized_words(prefix_words[-size:]) == cls._normalized_words(incoming_words[:size]):
                overlap = size
                break
        return list(prefix_words) + list(incoming_words[overlap:])

    @classmethod
    def _common_prefix_len(cls, left: list[str], right: list[str]) -> int:
        shared = 0
        for left_word, right_word in zip(left, right, strict=False):
            if cls._normalize_word(left_word) != cls._normalize_word(right_word):
                break
            shared += 1
        return shared

    @classmethod
    def _normalized_words(cls, words: list[str]) -> list[str]:
        return [cls._normalize_word(word) for word in words]

    @staticmethod
    def _normalize_word(word: str) -> str:
        lowered = word.strip().lower()
        normalized = _EDGE_PUNCTUATION_RE.sub("", lowered)
        return normalized or lowered
