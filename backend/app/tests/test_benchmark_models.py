from app.services.interpretation.benchmark_models import load_and_split, LABELS


def test_load_and_split_returns_three_non_empty_sets():
    """Smoke test: dataset loads and splits into train/val/test with text+label tuples."""
    train, val, test = load_and_split()
    assert len(train[0]) > 0, "train texts empty"
    assert len(val[0]) > 0, "val texts empty"
    assert len(test[0]) > 0, "test texts empty"
    assert len(train[0]) == len(train[1]), "train texts/labels length mismatch"
    # Labels should be ints in range [0, len(LABELS))
    assert all(0 <= label < len(LABELS) for label in train[1])


def test_labels_tuple_matches_expected():
    from app.models.domain import ActionTag
    assert LABELS[0] == ActionTag.no_action
    assert len(LABELS) == 7


from app.services.interpretation.benchmark_models import TfidfLogReg


def test_tfidf_logreg_fit_predict():
    texts = ["i m long here", "just watching the market", "flatten everything", "small piece on"]
    labels = [1, 0, 4, 1]
    model = TfidfLogReg()
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert len(preds) == 4
    assert all(isinstance(p, int) for p in preds)


from app.services.interpretation.benchmark_models import TfidfSVM


def test_tfidf_svm_fit_predict():
    texts = ["i m long here", "just watching the market", "flatten everything", "small piece on"]
    labels = [1, 0, 4, 1]
    model = TfidfSVM()
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert len(preds) == 4
    assert all(isinstance(p, int) for p in preds)


from app.services.interpretation.benchmark_models import TfidfMLP


def test_tfidf_mlp_fit_predict():
    texts = ["i m long here", "just watching the market", "flatten everything", "small piece on"]
    labels = [1, 0, 4, 1]
    model = TfidfMLP()
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert len(preds) == 4
    assert all(isinstance(p, int) for p in preds)


import pytest
from app.services.interpretation.benchmark_models import TransformerFrozenHead


@pytest.mark.slow
def test_transformer_frozen_head_fit_predict():
    """Smoke test with tiny input. Marked slow because it loads a transformer."""
    texts = [
        "i m long here", "just watching the market", "flatten everything",
        "small piece on", "taking some off", "stopped out", "no action here",
        "move stop to breakeven", "i m short from 200", "commentary only",
    ]
    labels = [1, 0, 4, 1, 3, 4, 0, 5, 2, 0]
    model = TransformerFrozenHead(model_name="distilbert-base-uncased", epochs=2, batch_size=4)
    model.fit(texts, labels)
    preds = model.predict(texts)
    assert len(preds) == 10
    assert all(0 <= p < 7 for p in preds)


from app.services.interpretation.benchmark_models import evaluate_model, cross_validate, TfidfLogReg, LABELS


def test_evaluate_model_returns_metrics():
    """evaluate_model runs fit+predict and returns a metrics dict."""
    model = TfidfLogReg()
    texts = ["long here", "no trade", "flatten", "piece on", "trim half", "stopped out", "watching"]
    labels = [1, 0, 4, 1, 3, 4, 0]
    metrics = evaluate_model(model, train_texts=texts, train_labels=labels, test_texts=texts, test_labels=labels)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_label" in metrics


def test_cross_validate_returns_summary():
    """cross_validate runs k-fold CV and returns mean/std metrics."""
    result = cross_validate(model_factory=TfidfLogReg, k=3)
    assert result["k"] == 3
    assert result["total_examples"] > 0
    assert result["folds"] == 3
    assert "summary" in result
    assert "accuracy" in result["summary"]
    assert "mean" in result["summary"]["accuracy"]
    assert "std" in result["summary"]["accuracy"]
    assert "aggregated_metrics" in result
    assert "confusion_matrix" in result
