from __future__ import annotations

from typing import Any

import numpy as np

_CANONICAL_PHRASES: tuple[str, ...] = (
    "i am going long now",
    "entering a long trade here",
    "getting into a long position",
    "buying here",
    "he enters long",
    "taking a long position",
    "i am going short now",
    "entering a short trade here",
    "selling here",
    "getting into a short position",
    "he enters short",
    "entering a trade now",
    "getting into position",
    "he enters now",
    "taking some profit now",
    "trimming my position",
    "paying myself here",
    "you can pay yourself here",
    "you can take profit",
    "reducing position size",
    "taking a little off",
    "i am out of the trade",
    "exiting the position now",
    "getting flat",
    "closing the trade",
    "all out",
    "stopped out",
    "adding to my position",
    "scaling into the trade",
    "adding here",
    "buying more",
    "moving my stop to breakeven",
    "adjusting the stop loss",
    "stop is now at",
    "moving stop up",
)


class EmbeddingGate:
    """Semantic gate for trade-relevant transcript segments."""

    def __init__(self, *, model_name: str = "BAAI/bge-small-en-v1.5", threshold: float = 0.40) -> None:
        self._model_name = model_name
        self._threshold = max(0.0, min(1.0, float(threshold)))
        self._model: Any | None = None
        self._canonical_embeddings: np.ndarray | None = None
        self._canonical_embeddings_normalized: np.ndarray | None = None

    def _create_model(self) -> Any:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self._model_name)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        model = self._create_model()
        canonical_vectors = np.array(list(model.embed(_CANONICAL_PHRASES)), dtype=np.float32)
        if canonical_vectors.ndim != 2 or canonical_vectors.shape[0] == 0:
            raise RuntimeError("Embedding gate failed to initialize canonical vectors.")

        self._model = model
        self._canonical_embeddings = canonical_vectors
        self._canonical_embeddings_normalized = _normalize_rows(canonical_vectors)

    def best_score(self, text: str) -> float:
        candidate = text.strip()
        if not candidate:
            return 0.0

        self._ensure_loaded()
        assert self._model is not None
        assert self._canonical_embeddings_normalized is not None

        vector = np.array(list(self._model.embed([candidate]))[0], dtype=np.float32)
        vector_normalized = _normalize_vector(vector)
        scores = self._canonical_embeddings_normalized @ vector_normalized
        best = float(scores.max())
        return max(0.0, min(1.0, best))

    def is_trade_relevant(self, text: str) -> bool:
        return self.best_score(text) >= self._threshold


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-9)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-9)
