"""Dump misclassified examples from 5-fold CV using LogReg.

Collects all test-set predictions across folds, filters for errors,
and prints representative examples for thesis error-analysis section.

Usage:
    python -m app.services.interpretation.dump_misclassified [--output data/misclassified.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.services.interpretation.benchmark_models import (
    LABELS,
    TfidfLogReg,
    _LABEL_TO_ID,
    _DEFAULT_DATASET,
    load_examples_grouped,
)
from app.services.interpretation.train_local_classifier import (
    _rebalance_training_examples,
    _example_file_key,
)


_LABEL_NAMES = {i: label.value for i, label in enumerate(LABELS)}


def collect_misclassified(
    dataset: Path = _DEFAULT_DATASET,
    k: int = 5,
    no_action_ratio: float = 2.5,
    max_no_action: int = 6_000,
) -> list[dict]:
    """Run k-fold CV and return all misclassified examples with metadata."""
    examples, file_keys = load_examples_grouped(dataset)
    by_file: dict[str, list[dict]] = {}
    for ex in examples:
        by_file.setdefault(_example_file_key(ex), []).append(ex)

    fold_size = math.ceil(len(file_keys) / k)
    folds = [file_keys[i * fold_size : (i + 1) * fold_size] for i in range(k)]

    misclassified: list[dict] = []

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

        model = TfidfLogReg()
        model.fit(train_texts, train_labels)
        predictions = model.predict(test_texts)

        for ex, true_id, pred_id in zip(test_ex, test_labels, predictions):
            if true_id != pred_id:
                misclassified.append({
                    "fold": fold_idx,
                    "true_label": _LABEL_NAMES[true_id],
                    "predicted_label": _LABEL_NAMES[pred_id],
                    "prompt": ex["prompt"],
                    "evidence_text": ex.get("evidence_text", ""),
                    "current_text": ex.get("current_text", ""),
                    "file": ex.get("file", ""),
                    "timecode": ex.get("timecode", ""),
                })

    return misclassified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("data/misclassified.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=10, help="Examples to print per error pattern")
    args = parser.parse_args()

    errors = collect_misclassified(dataset=args.dataset, k=args.k)
    print(f"\nTotal misclassified: {len(errors)}")

    # Group by error pattern
    pattern_counts: Counter = Counter()
    pattern_examples: dict[str, list[dict]] = defaultdict(list)
    for err in errors:
        key = f"{err['true_label']} -> {err['predicted_label']}"
        pattern_counts[key] += 1
        pattern_examples[key].append(err)

    print(f"\n{'Error pattern':<40} {'Count':>6}")
    print("-" * 48)
    for pattern, count in pattern_counts.most_common(20):
        print(f"{pattern:<40} {count:>6}")

    # Print representative examples for top patterns
    print(f"\n{'='*70}\nREPRESENTATIVE MISCLASSIFIED EXAMPLES\n{'='*70}")
    for pattern, _ in pattern_counts.most_common(5):
        examples = pattern_examples[pattern][:args.top_n]
        print(f"\n--- {pattern} ({pattern_counts[pattern]} total) ---")
        for i, ex in enumerate(examples[:3]):
            print(f"\n  Example {i+1}:")
            print(f"    Evidence: {ex['evidence_text'][:120]}")
            print(f"    True: {ex['true_label']}  Predicted: {ex['predicted_label']}")

    # Save full dump
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull dump saved to {args.output}")


if __name__ == "__main__":
    main()
