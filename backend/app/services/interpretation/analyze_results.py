"""Post-benchmark analysis: feature importance and statistical significance tests.

Produces two outputs for the thesis:
1. Top TF-IDF features per class for the LogReg model (explains WHY classicals win)
2. Paired statistical tests between model pairs (are differences significant?)

Usage:
    cd backend
    python -m app.services.interpretation.analyze_results
    python -m app.services.interpretation.analyze_results --top-k 15
    python -m app.services.interpretation.analyze_results --output data/analysis_results.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from app.models.domain import ActionTag
from app.services.interpretation.benchmark_models import (
    LABELS,
    TfidfLogReg,
    TfidfSVM,
    TfidfMLP,
    TransformerFrozenHead,
    _make_model,
    load_examples_grouped,
    _MODEL_REGISTRY,
)
from app.services.interpretation.train_local_classifier import (
    _load_examples,
    _rebalance_training_examples,
    _example_file_key,
    _classification_metrics,
)

_LABEL_TO_ID = {label.value: i for i, label in enumerate(LABELS)}
_DEFAULT_DATASET = Path("data/training_data_clean.jsonl")


# ---------------------------------------------------------------------------
# Feature importance: extract top TF-IDF features per class from LogReg
# ---------------------------------------------------------------------------

def extract_feature_importance(
    dataset: Path = _DEFAULT_DATASET,
    top_k: int = 15,
    no_action_ratio: float = 2.5,
) -> dict[str, list[dict[str, float]]]:
    """Train LogReg on full dataset and extract top-k features per class.

    Returns a dict mapping each label name to a list of
    {"feature": str, "weight": float} entries sorted by weight descending.
    """
    examples, _ = _load_examples(dataset)
    examples, _ = _rebalance_training_examples(
        examples, no_action_ratio=no_action_ratio, max_no_action_examples=6_000,
    )
    texts = [ex["prompt"] for ex in examples]
    labels = [_LABEL_TO_ID[ex["label"]] for ex in examples]

    # Train a LogReg model to get coefficients
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X, labels)

    feature_names = vectorizer.get_feature_names_out()
    # clf.coef_ shape: (n_classes, n_features)
    importance: dict[str, list[dict[str, float]]] = {}

    for class_idx, label in enumerate(LABELS):
        weights = clf.coef_[class_idx]
        # Get indices of top-k highest weights (most predictive for this class)
        top_indices = np.argsort(weights)[-top_k:][::-1]
        importance[label.value] = [
            {"feature": feature_names[i], "weight": round(float(weights[i]), 4)}
            for i in top_indices
        ]

    return importance


# ---------------------------------------------------------------------------
# Statistical significance: paired t-test on per-fold metrics
# ---------------------------------------------------------------------------

def paired_significance_test(
    fold_metrics_a: list[float],
    fold_metrics_b: list[float],
) -> dict[str, float]:
    """Paired t-test between two models' per-fold metric values.

    Returns t-statistic, p-value, mean difference, and 95% confidence interval.
    Appropriate when comparing k-fold CV results from the same folds.
    """
    a = np.array(fold_metrics_a)
    b = np.array(fold_metrics_b)
    differences = a - b

    n = len(differences)
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))

    if std_diff == 0 or n < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_difference": mean_diff,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "significant_at_005": False,
            "n_folds": n,
        }

    t_stat, p_value = stats.ttest_rel(a, b)
    # 95% confidence interval for the mean difference
    se = std_diff / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return {
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "mean_difference": round(mean_diff, 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "significant_at_005": float(p_value) < 0.05,
        "n_folds": n,
    }


def run_significance_tests(
    benchmark_results_path: Path = Path("data/benchmark_results.json"),
    metric: str = "macro_f1",
) -> dict[str, dict]:
    """Load benchmark JSON and run pairwise significance tests.

    Requires benchmark_results.json from a --cv run (contains per_fold metrics).
    """
    if not benchmark_results_path.is_file():
        return {"error": f"File not found: {benchmark_results_path}. Run benchmark with --cv first."}

    data = json.loads(benchmark_results_path.read_text(encoding="utf-8"))
    models = data.get("models", {})

    # Extract per-fold metric values for each model
    per_fold: dict[str, list[float]] = {}
    for model_key, model_data in models.items():
        folds = model_data.get("per_fold", [])
        if folds:
            per_fold[model_key] = [fold.get(metric, 0.0) for fold in folds]

    # Run pairwise tests
    model_keys = sorted(per_fold.keys())
    results: dict[str, dict] = {}
    for i, model_a in enumerate(model_keys):
        for model_b in model_keys[i + 1:]:
            pair_key = f"{model_a}_vs_{model_b}"
            results[pair_key] = paired_significance_test(
                per_fold[model_a], per_fold[model_b],
            )
            results[pair_key]["model_a"] = model_a
            results[pair_key]["model_b"] = model_b
            results[pair_key]["metric"] = metric

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-benchmark analysis: feature importance and statistical tests."
    )
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--benchmark-results", type=Path, default=Path("data/benchmark_results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis_results.json"))
    parser.add_argument("--top-k", type=int, default=15, help="Top-k features per class to extract.")
    parser.add_argument("--metric", default="macro_f1", help="Metric for significance tests.")
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()

    print("=" * 60)
    print("FEATURE IMPORTANCE (LogReg, top TF-IDF features per class)")
    print("=" * 60)

    importance = extract_feature_importance(dataset=args.dataset, top_k=args.top_k)

    for label, features in importance.items():
        print(f"\n  {label}:")
        for entry in features:
            print(f"    {entry['weight']:+.4f}  {entry['feature']}")

    print(f"\n{'=' * 60}")
    print(f"STATISTICAL SIGNIFICANCE (paired t-test on {args.metric})")
    print("=" * 60)

    sig_results = run_significance_tests(
        benchmark_results_path=args.benchmark_results,
        metric=args.metric,
    )

    if "error" in sig_results:
        print(f"\n  {sig_results['error']}")
    else:
        print(f"\n  {'Comparison':<25} {'Mean Diff':>10} {'p-value':>8} {'Significant':>12}")
        print(f"  {'-' * 58}")
        for pair_key, result in sig_results.items():
            sig = "YES" if result["significant_at_005"] else "no"
            print(f"  {pair_key:<25} {result['mean_difference']:>+10.4f} {result['p_value']:>8.4f} {sig:>12}")

    # Save combined results
    output = {
        "feature_importance": importance,
        "statistical_tests": sig_results,
        "config": {
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "metric": args.metric,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
