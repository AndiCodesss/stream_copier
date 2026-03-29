from __future__ import annotations

import re
from dataclasses import dataclass, field

_EDGE_PUNCTUATION_RE = re.compile(r"(^[^\w$]+|[^\w$]+$)")


@dataclass
class _StreamingPreviewState:
    utterance_id: str
    committed_words: list[str] = field(default_factory=list)
    last_display_words: list[str] = field(default_factory=list)


class StreamingPreviewAssembler:
    def __init__(self, *, context_words: int, stability_margin_words: int) -> None:
        self._context_words = max(0, context_words)
        self._stability_margin_words = max(0, stability_margin_words)
        self._state: _StreamingPreviewState | None = None

    def reset(self) -> None:
        self._state = None

    def build_prompt(self, *, utterance_id: str, base_prompt: str) -> str:
        state = self._ensure_state(utterance_id)
        if self._context_words == 0 or not state.committed_words:
            return base_prompt

        context = self._join_words(state.committed_words[-self._context_words :])
        if not base_prompt:
            return context
        return f"{base_prompt}\nRecent confirmed transcript: {context}"

    def stabilize(self, *, utterance_id: str, tail_text: str) -> str:
        state = self._ensure_state(utterance_id)
        tail_words = self._split_words(tail_text)
        if not tail_words:
            state.last_display_words = list(state.committed_words)
            return self._join_words(state.last_display_words)

        candidate_words = self._merge_overlap(state.committed_words, tail_words)
        if state.last_display_words:
            common_prefix_len = self._common_prefix_len(state.last_display_words, candidate_words)
            commit_target = max(len(state.committed_words), common_prefix_len - self._stability_margin_words)
            if commit_target > len(state.committed_words):
                state.committed_words = list(candidate_words[:commit_target])
                candidate_words = self._merge_overlap(state.committed_words, tail_words)

        state.last_display_words = list(candidate_words)
        return self._join_words(state.last_display_words)

    def committed_text(self, *, utterance_id: str) -> str:
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
        if not prefix_words:
            return list(incoming_words)
        if not incoming_words:
            return list(prefix_words)

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
