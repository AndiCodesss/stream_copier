"""Ablation: TF-IDF vocabulary size sensitivity for LogReg.

Tests max_features at 5k, 10k, 20k, 50k via 5-fold CV.

Usage:
    python -m app.services.interpretation.ablate_tfidf_vocab
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.services.interpretation.benchmark_models import (
    cross_validate,
    _DEFAULT_DATASET,
)


class TfidfLogRegCustom:
    """LogReg with configurable max_features."""

    def __init__(self, max_features: int = 20_000) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        X = self._vectorizer.fit_transform(texts)
        self._clf.fit(X, labels)

    def predict(self, texts: list[str]) -> list[int]:
        X = self._vectorizer.transform(texts)
        return self._clf.predict(X).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    vocab_sizes = [5_000, 10_000, 20_000, 50_000]

    print(f"{'max_features':>14} {'Accuracy':>14} {'Macro F1':>14} {'Action F1':>14}")
    print("-" * 60)

    for max_feat in vocab_sizes:
        result = cross_validate(
            model_factory=lambda mf=max_feat: TfidfLogRegCustom(max_features=mf),
            dataset=args.dataset,
            k=args.k,
        )
        s = result["summary"]
        acc = f"{s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}"
        mf1 = f"{s['macro_f1']['mean']:.4f}+/-{s['macro_f1']['std']:.4f}"
        af1 = f"{s['action_f1']['mean']:.4f}+/-{s['action_f1']['std']:.4f}"
        print(f"{max_feat:>14,} {acc:>14} {mf1:>14} {af1:>14}")


if __name__ == "__main__":
    main()
