from __future__ import annotations

import numpy as np

from app.services.interpretation.embedding_gate import EmbeddingGate


class _FakeModel:
    def embed(self, texts):  # noqa: ANN001
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "long" in normalized:
                vectors.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            elif "profit" in normalized or "pay yourself" in normalized:
                vectors.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            elif "consolidating" in normalized or "watching" in normalized:
                vectors.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            else:
                vectors.append(np.array([0.2, 0.2, 0.2], dtype=np.float32))
        return vectors


def test_embedding_gate_trade_relevant_when_similarity_is_high(monkeypatch) -> None:
    monkeypatch.setattr(EmbeddingGate, "_create_model", lambda self: _FakeModel())
    gate = EmbeddingGate(threshold=0.50)

    assert gate.is_trade_relevant("you can go long here") is True
    assert gate.is_trade_relevant("you can take some profit here") is True


def test_embedding_gate_rejects_non_trade_commentary(monkeypatch) -> None:
    monkeypatch.setattr(EmbeddingGate, "_create_model", lambda self: _FakeModel())
    gate = EmbeddingGate(threshold=0.95)

    assert gate.is_trade_relevant("the market is consolidating and we are watching") is False


def test_embedding_gate_best_score_in_range(monkeypatch) -> None:
    monkeypatch.setattr(EmbeddingGate, "_create_model", lambda self: _FakeModel())
    gate = EmbeddingGate(threshold=0.30)

    score = gate.best_score("pay yourself a piece here")
    assert 0.0 <= score <= 1.0
