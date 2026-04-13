"""Thesis benchmark: train and evaluate 5 model families on trade-intent classification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from app.models.domain import ActionTag
from app.services.interpretation.train_local_classifier import (
    _load_examples,
    _split_examples_by_transcript,
    _rebalance_training_examples,
    _classification_metrics,
    _example_file_key,
)

LABELS: tuple[ActionTag, ...] = (
    ActionTag.no_action,
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
)

_LABEL_TO_ID = {label.value: i for i, label in enumerate(LABELS)}

_DEFAULT_DATASET = Path("data/training_data_clean.jsonl")


def load_and_split(
    dataset: Path = _DEFAULT_DATASET,
    validation_ratio: float = 0.2,
    test_transcripts: int = 5,
    split_mode: str = "stable_hash",
    no_action_ratio: float = 2.5,
    max_no_action: int = 6_000,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    """Load JSONL dataset, split by transcript, return (texts, labels) per split."""
    examples, _ = _load_examples(dataset)
    train_ex, val_ex, test_ex, _ = _split_examples_by_transcript(
        examples,
        validation_ratio=validation_ratio,
        test_transcripts=test_transcripts,
        split_mode=split_mode,
    )
    train_ex, _ = _rebalance_training_examples(
        train_ex, no_action_ratio=no_action_ratio, max_no_action_examples=max_no_action,
    )

    def _extract(exs: list[dict]) -> tuple[list[str], list[int]]:
        texts = [ex["prompt"] for ex in exs]
        labels = [_LABEL_TO_ID[ex["label"]] for ex in exs]
        return texts, labels

    return _extract(train_ex), _extract(val_ex), _extract(test_ex)


def load_examples_grouped(
    dataset: Path = _DEFAULT_DATASET,
    no_action_ratio: float = 2.5,
    max_no_action: int = 6_000,
) -> tuple[list[dict], list[str]]:
    """Load and dedupe examples, return (examples, sorted_file_keys)."""
    examples, _ = _load_examples(dataset)
    by_file: dict[str, list[dict]] = {}
    for ex in examples:
        by_file.setdefault(_example_file_key(ex), []).append(ex)
    file_keys = sorted(by_file.keys(), key=lambda f: hashlib.sha256(f.encode()).hexdigest())
    return examples, file_keys


def cross_validate(
    model_factory,
    *,
    dataset: Path = _DEFAULT_DATASET,
    k: int = 5,
    no_action_ratio: float = 2.5,
    max_no_action: int = 6_000,
) -> dict:
    """K-fold cross-validation with transcript-level splits.

    Transcripts are grouped into k folds. Each fold is used as the test set once
    while the remaining k-1 folds are used for training. Returns mean and std
    of metrics across folds, plus aggregated per-class metrics.
    """
    examples, file_keys = load_examples_grouped(dataset)
    by_file: dict[str, list[dict]] = {}
    for ex in examples:
        by_file.setdefault(_example_file_key(ex), []).append(ex)

    # Assign transcripts to folds
    fold_size = math.ceil(len(file_keys) / k)
    folds: list[list[str]] = []
    for i in range(k):
        folds.append(file_keys[i * fold_size : (i + 1) * fold_size])

    all_true: list[int] = []
    all_pred: list[int] = []
    fold_metrics: list[dict] = []

    for fold_idx in range(k):
        test_files = set(folds[fold_idx])
        train_ex = [ex for f in file_keys if f not in test_files for ex in by_file[f]]
        test_ex = [ex for f in folds[fold_idx] for ex in by_file[f]]

        if not train_ex or not test_ex:
            continue

        train_ex, _ = _rebalance_training_examples(
            train_ex, no_action_ratio=no_action_ratio, max_no_action_examples=max_no_action,
        )

        train_texts = [ex["prompt"] for ex in train_ex]
        train_labels = [_LABEL_TO_ID[ex["label"]] for ex in train_ex]
        test_texts = [ex["prompt"] for ex in test_ex]
        test_labels = [_LABEL_TO_ID[ex["label"]] for ex in test_ex]

        model = model_factory()
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)

        metrics = _classification_metrics(
            true_labels=test_labels, predicted_labels=predictions, labels=LABELS,
        )
        fold_metrics.append(metrics)
        all_true.extend(test_labels)
        all_pred.extend(predictions)

        print(f"  Fold {fold_idx + 1}/{k}: "
              f"train={len(train_texts)} test={len(test_texts)} "
              f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    # Aggregate: pooled predictions across all folds
    aggregated = _classification_metrics(true_labels=all_true, predicted_labels=all_pred, labels=LABELS)
    cm = _confusion_matrix(all_true, all_pred, len(LABELS))

    # Mean/std across folds
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        m = _mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))

    summary = {}
    for key in ("accuracy", "macro_f1", "action_f1", "action_precision", "action_recall"):
        vals = [fm[key] for fm in fold_metrics]
        summary[key] = {"mean": round(_mean(vals), 4), "std": round(_std(vals), 4)}

    return {
        "k": k,
        "total_examples": len(all_true),
        "folds": len(fold_metrics),
        "summary": summary,
        "aggregated_metrics": aggregated,
        "confusion_matrix": cm,
        "per_fold": fold_metrics,
    }


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLogReg:
    """TF-IDF + Logistic Regression baseline."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

    def fit(self, texts: list[str], labels: list[int]) -> None:
        X = self._vectorizer.fit_transform(texts)
        self._clf.fit(X, labels)

    def predict(self, texts: list[str]) -> list[int]:
        X = self._vectorizer.transform(texts)
        return self._clf.predict(X).tolist()


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class TfidfSVM:
    """TF-IDF + Linear SVM baseline."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
        self._base_svc = LinearSVC(max_iter=2000, class_weight="balanced", C=1.0)
        self._clf: CalibratedClassifierCV | LinearSVC = self._base_svc

    def fit(self, texts: list[str], labels: list[int]) -> None:
        from collections import Counter
        X = self._vectorizer.fit_transform(texts)
        min_class_count = min(Counter(labels).values())
        cv = min(5, min_class_count)
        if cv >= 2:
            self._clf = CalibratedClassifierCV(
                LinearSVC(max_iter=2000, class_weight="balanced", C=1.0), cv=cv
            )
        else:
            self._clf = LinearSVC(max_iter=2000, class_weight="balanced", C=1.0)
        self._clf.fit(X, labels)

    def predict(self, texts: list[str]) -> list[int]:
        X = self._vectorizer.transform(texts)
        return self._clf.predict(X).tolist()


from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


class TfidfMLP:
    """TF-IDF + 2-layer MLP baseline with balanced class weighting."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
        self._clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

    def fit(self, texts: list[str], labels: list[int]) -> None:
        X = self._vectorizer.fit_transform(texts)
        n_samples = X.shape[0]
        use_early_stopping = n_samples >= 10
        self._clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            early_stopping=use_early_stopping,
            validation_fraction=0.15 if use_early_stopping else 0.0,
            random_state=42,
        )
        # MLPClassifier lacks class_weight; replicate via sample_weight
        sample_weights = compute_sample_weight("balanced", labels)
        # Repeat samples proportionally to their weight to approximate sample_weight
        # (sklearn MLP.fit does not accept sample_weight, so we oversample)
        import numpy as np
        rng = np.random.RandomState(42)
        weights_normalized = sample_weights / sample_weights.min()
        indices = []
        for i, w in enumerate(weights_normalized):
            count = int(np.round(w))
            indices.extend([i] * max(1, count))
        indices = np.array(indices)
        rng.shuffle(indices)
        from scipy.sparse import issparse
        X_resampled = X[indices] if issparse(X) else X[indices]
        labels_resampled = [labels[i] for i in indices]
        self._clf.fit(X_resampled, labels_resampled)

    def predict(self, texts: list[str]) -> list[int]:
        X = self._vectorizer.transform(texts)
        return self._clf.predict(X).tolist()


import torch
from app.services.interpretation.train_local_classifier import (
    _embed_prompts,
    _resolve_device,
    _build_head,
    _class_weights,
)


class TransformerFrozenHead:
    """Frozen transformer encoder + trained linear classification head."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        epochs: int = 20,
        lr: float = 3e-3,
        batch_size: int = 24,
        max_length: int = 256,
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._max_length = max_length
        self._configured_device = device
        self._head = None
        self._encoder = None
        self._tokenizer = None
        self._device = None

    def _load_encoder(self):
        if self._encoder is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        self._device = _resolve_device(torch, self._configured_device)
        model_path = snapshot_download(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self._encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
        self._encoder.to(self._device)
        self._encoder.eval()

    def _embed(self, texts: list[str]):
        return _embed_prompts(
            texts=texts,
            tokenizer=self._tokenizer,
            model=self._encoder,
            torch_module=torch,
            device=self._device,
            max_length=self._max_length,
            batch_size=self._batch_size,
        )

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self._load_encoder()
        embeddings = self._embed(texts)
        labels_t = torch.tensor(labels, dtype=torch.long)
        hidden_size = int(embeddings.shape[1])
        self._head = _build_head(torch, hidden_size=hidden_size, num_labels=len(LABELS)).to(self._device)
        optimizer = torch.optim.AdamW(self._head.parameters(), lr=self._lr, weight_decay=0.01)
        weights = _class_weights(torch, labels_t).to(self._device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        train_x = embeddings.to(self._device)
        train_y = labels_t.to(self._device)

        for _ in range(self._epochs):
            self._head.train()
            perm = torch.randperm(train_x.shape[0], device=self._device)
            for start in range(0, train_x.shape[0], self._batch_size):
                batch = perm[start : start + self._batch_size]
                loss = criterion(self._head(train_x[batch]), train_y[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def predict(self, texts: list[str]) -> list[int]:
        self._load_encoder()
        embeddings = self._embed(texts).to(self._device)
        self._head.eval()
        with torch.inference_mode():
            logits = self._head(embeddings)
            return logits.argmax(dim=-1).cpu().tolist()


from collections import Counter


def evaluate_model(
    model,
    *,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict:
    """Fit model on train, predict on test, return metrics dict."""
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    return _classification_metrics(true_labels=test_labels, predicted_labels=predictions, labels=LABELS)


def _confusion_matrix(true_labels: list[int], predicted_labels: list[int], num_classes: int) -> list[list[int]]:
    """Row = true, col = predicted."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(true_labels, predicted_labels):
        matrix[t][p] += 1
    return matrix


def _parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark 5 model families on trade-intent classification.")
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_results.json"))
    parser.add_argument("--split-mode", default="stable_hash", choices=("stable_hash", "temporal_recent"))
    parser.add_argument("--test-transcripts", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--cv", type=int, default=0, help="K-fold cross-validation (0 = single split, 5 = 5-fold CV)")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["logreg", "svm", "mlp", "distilbert", "modernbert"],
        help="Which models to run. Options: logreg, svm, mlp, distilbert, modernbert",
    )
    return parser.parse_args()


_MODEL_REGISTRY: dict[str, type | tuple] = {
    "logreg": TfidfLogReg,
    "svm": TfidfSVM,
    "mlp": TfidfMLP,
    "distilbert": (TransformerFrozenHead, {"model_name": "distilbert-base-uncased"}),
    "modernbert": (TransformerFrozenHead, {"model_name": "answerdotai/ModernBERT-base"}),
}


def _make_model(model_key: str, device: str = "auto"):
    """Instantiate a model from the registry."""
    entry = _MODEL_REGISTRY.get(model_key)
    if entry is None:
        return None
    if isinstance(entry, tuple):
        cls, kwargs = entry
        return cls(**{**kwargs, "device": device})
    return entry()


def _main() -> int:
    args = _parse_benchmark_args()

    if args.cv >= 2:
        return _main_cv(args)
    return _main_single_split(args)


def _main_cv(args: argparse.Namespace) -> int:
    """K-fold cross-validation mode."""
    examples, file_keys = load_examples_grouped(dataset=args.dataset)
    print(f"{args.cv}-fold cross-validation over {len(examples)} examples from {len(file_keys)} transcripts\n")

    results: dict[str, dict] = {}
    for model_key in args.models:
        if model_key not in _MODEL_REGISTRY:
            print(f"Unknown model: {model_key}, skipping")
            continue

        print(f"\n{'='*60}\n{model_key} ({args.cv}-fold CV)\n{'='*60}")

        cv_result = cross_validate(
            model_factory=lambda mk=model_key: _make_model(mk, args.device),
            dataset=args.dataset,
            k=args.cv,
        )
        results[model_key] = cv_result
        s = cv_result["summary"]
        print(f"  => acc={s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}  "
              f"macro_f1={s['macro_f1']['mean']:.4f}+/-{s['macro_f1']['std']:.4f}  "
              f"action_f1={s['action_f1']['mean']:.4f}+/-{s['action_f1']['std']:.4f}")

    # Print comparison table
    print(f"\n{'='*60}\nRESULTS ({args.cv}-fold CV, mean +/- std)\n{'='*60}")
    print(f"{'Model':<14} {'Accuracy':>14} {'Macro F1':>14} {'Action F1':>14}")
    print("-" * 60)
    for model_key, data in results.items():
        s = data["summary"]
        acc = f"{s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}"
        mf1 = f"{s['macro_f1']['mean']:.4f}+/-{s['macro_f1']['std']:.4f}"
        af1 = f"{s['action_f1']['mean']:.4f}+/-{s['action_f1']['std']:.4f}"
        print(f"{model_key:<14} {acc:>14} {mf1:>14} {af1:>14}")

    # Also print aggregated (pooled) results
    print(f"\n{'='*60}\nAGGREGATED (pooled across all folds)\n{'='*60}")
    print(f"{'Model':<14} {'Accuracy':>8} {'Macro F1':>8} {'Action F1':>9}")
    print("-" * 43)
    for model_key, data in results.items():
        m = data["aggregated_metrics"]
        print(f"{model_key:<14} {m['accuracy']:>8.4f} {m['macro_f1']:>8.4f} {m['action_f1']:>9.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "mode": f"{args.cv}-fold-cv",
        "total_examples": len(examples),
        "total_transcripts": len(file_keys),
        "k": args.cv,
        "labels": [label.value for label in LABELS],
        "models": results,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")
    return 0


def _main_single_split(args: argparse.Namespace) -> int:
    """Original single train/val/test split mode."""
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_split(
        dataset=args.dataset,
        split_mode=args.split_mode,
        test_transcripts=args.test_transcripts,
    )
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")
    print(f"Train label distribution: {dict(sorted(Counter(train_labels).items()))}")
    print(f"Test label distribution:  {dict(sorted(Counter(test_labels).items()))}")

    results: dict[str, dict] = {}
    for model_key in args.models:
        model = _make_model(model_key, args.device)
        if model is None:
            print(f"Unknown model: {model_key}, skipping")
            continue

        print(f"\n{'='*60}\nTraining: {model_key}\n{'='*60}")
        model.fit(train_texts, train_labels)

        predictions = model.predict(test_texts)
        metrics = _classification_metrics(true_labels=test_labels, predicted_labels=predictions, labels=LABELS)
        cm = _confusion_matrix(test_labels, predictions, len(LABELS))

        results[model_key] = {
            "metrics": metrics,
            "confusion_matrix": cm,
        }
        print(f"  accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
              f"action_f1={metrics['action_f1']:.4f}")

    # Print comparison table
    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"{'Model':<14} {'Accuracy':>8} {'Macro F1':>8} {'Action F1':>9}")
    print("-" * 43)
    for model_key, data in results.items():
        m = data["metrics"]
        print(f"{model_key:<14} {m['accuracy']:>8.4f} {m['macro_f1']:>8.4f} {m['action_f1']:>9.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "mode": "single_split",
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "labels": [label.value for label in LABELS],
        "models": results,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
