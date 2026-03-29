from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import ActionTag

_LABELS: tuple[ActionTag, ...] = (
    ActionTag.no_action,
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
)

_SOURCE_PRIORITY = {
    "ai_review_corrected": 7,
    "ai_review_accept": 7,
    "ai_review_hard_negative": 7,
}

# Per-sample quality weights for the loss function.  Applied only to positive
# (action) examples — NO_ACTION always gets weight 1.0.  Reviewed examples get
# a boost so their higher-quality labels have more influence on the decision
# boundary than lower-signal reviewed examples.
#
# NOTE: Effective only when the reviewed corpus is large enough (500+ examples).
# With < 100 reviewed positives in training, boosting overfits to few samples
# and hurts validation metrics.  Scale these up as the reviewed corpus grows.
_SOURCE_QUALITY_WEIGHT: dict[str, float] = {
    "ai_review_corrected": 1.0,
    "ai_review_accept": 1.0,
    "ai_review_hard_negative": 1.0,
}

_FILE_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_ALT_FILE_DATE_PATTERN = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local ModernBERT intent classifier head from reviewed execution labels.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training_data.jsonl"),
        help="Reviewed execution JSONL dataset path.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for metadata and classifier head artifacts.",
    )
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base", help="Backbone model name.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Encoder/training device.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument("--batch-size", type=int, default=24, help="Embedding batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Linear head training epochs.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Linear head learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Stable transcript-hash validation split ratio applied after reserving the test transcripts.",
    )
    parser.add_argument(
        "--test-transcripts",
        type=int,
        default=5,
        help="Number of whole transcripts to reserve as an untouched test set.",
    )
    parser.add_argument(
        "--split-mode",
        default="stable_hash",
        choices=("stable_hash", "temporal_recent"),
        help="Transcript split strategy. 'temporal_recent' reserves the newest reviewed transcripts for validation/test.",
    )
    parser.add_argument(
        "--no-action-ratio",
        type=float,
        default=2.5,
        help="Maximum NO_ACTION-to-positive ratio retained in the training split.",
    )
    parser.add_argument(
        "--max-no-action-examples",
        type=int,
        default=6_000,
        help="Hard cap for NO_ACTION examples retained in the training split.",
    )
    return parser.parse_args()


def _load_examples(path: Path) -> tuple[list[dict], int]:
    examples: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            examples.append(json.loads(raw_line))
    deduped = _dedupe_examples(examples)
    return deduped, max(0, len(examples) - len(deduped))


def _example_identity(example: dict) -> tuple[str, ...]:
    return (
        str(example.get("label", "")),
        str(example.get("prompt", "")),
        str(example.get("current_text", "")),
        str(example.get("position_side", "")),
        str(example.get("last_side", "")),
    )


def _dedupe_examples(examples: list[dict]) -> list[dict]:
    unique: dict[tuple[str, ...], dict] = {}
    for example in examples:
        key = _example_identity(example)
        existing = unique.get(key)
        if existing is None:
            unique[key] = example
            continue
        current_priority = _SOURCE_PRIORITY.get(str(example.get("source", "")), 0)
        existing_priority = _SOURCE_PRIORITY.get(str(existing.get("source", "")), 0)
        if current_priority > existing_priority:
            unique[key] = example
    return sorted(unique.values(), key=_stable_order_key)


def _example_file_key(example: dict) -> str:
    return str(Path(str(example.get("file", ""))).expanduser())


def _stable_file_order(file_keys: list[str]) -> list[str]:
    return sorted(
        dict.fromkeys(file_keys),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )


def _date_from_file_name(file_key: str) -> datetime | None:
    name = Path(file_key).name
    match = _FILE_DATE_PATTERN.search(name)
    if match is not None:
        year, month, day = (int(part) for part in match.groups())
        return datetime(year, month, day, tzinfo=UTC)
    alt_match = _ALT_FILE_DATE_PATTERN.search(name)
    if alt_match is not None:
        day, month, short_year = (int(part) for part in alt_match.groups())
        return datetime(2000 + short_year, month, day, tzinfo=UTC)
    return None


def _file_timestamp_order(file_examples: list[dict]) -> tuple[datetime, str]:
    timestamps: list[datetime] = []
    file_name = Path(_example_file_key(file_examples[0])).name if file_examples else ""
    file_date = _date_from_file_name(file_name)
    if file_date is not None:
        return file_date, file_name
    for example in file_examples:
        raw_timestamp = str(example.get("timestamp", "")).strip()
        if not raw_timestamp:
            continue
        try:
            timestamps.append(datetime.fromisoformat(raw_timestamp))
        except ValueError:
            continue
    if timestamps:
        return max(timestamps), file_name
    return datetime.min.replace(tzinfo=UTC), file_name


def _temporal_file_order(by_file: dict[str, list[dict]]) -> list[str]:
    return sorted(
        dict.fromkeys(by_file),
        key=lambda file_key: _file_timestamp_order(by_file[file_key]),
    )


def _file_has_action(examples: list[dict]) -> bool:
    return any(example.get("label") != ActionTag.no_action.value for example in examples)


def _split_examples_by_transcript(
    examples: list[dict],
    *,
    validation_ratio: float,
    test_transcripts: int,
    split_mode: str = "stable_hash",
) -> tuple[list[dict], list[dict], list[dict], dict[str, object]]:
    by_file: dict[str, list[dict]] = {}
    for example in examples:
        by_file.setdefault(_example_file_key(example), []).append(example)

    if split_mode == "temporal_recent":
        ordered_files = _temporal_file_order(by_file)
    else:
        ordered_files = _stable_file_order(list(by_file))
    if not ordered_files:
        return [], [], [], {
            "train_files": [],
            "validation_files": [],
            "test_files": [],
            "split_mode": split_mode,
        }

    max_test_files = max(0, len(ordered_files) - 2)
    requested_test_files = max(0, test_transcripts)
    test_file_count = min(requested_test_files, max_test_files)

    action_files = [file_key for file_key in ordered_files if _file_has_action(by_file[file_key])]
    test_candidates = action_files or ordered_files
    if split_mode == "temporal_recent":
        test_files = test_candidates[-test_file_count:] if test_file_count else []
    else:
        test_files = test_candidates[:test_file_count]

    remaining_files = [file_key for file_key in ordered_files if file_key not in test_files]
    bounded_ratio = max(0.05, min(0.45, validation_ratio))
    if len(remaining_files) >= 2:
        validation_file_count = int(round(len(remaining_files) * bounded_ratio))
        validation_file_count = max(1, min(validation_file_count, len(remaining_files) - 1))
    else:
        validation_file_count = 0

    remaining_action_files = [file_key for file_key in remaining_files if _file_has_action(by_file[file_key])]
    validation_candidates = [
        *remaining_action_files,
        *(file_key for file_key in remaining_files if file_key not in remaining_action_files),
    ]
    if split_mode == "temporal_recent":
        validation_files = validation_candidates[-validation_file_count:] if validation_file_count else []
    else:
        validation_files = validation_candidates[:validation_file_count]
    train_files = [file_key for file_key in remaining_files if file_key not in validation_files]

    train = [example for file_key in train_files for example in sorted(by_file[file_key], key=_stable_order_key)]
    validation = [example for file_key in validation_files for example in sorted(by_file[file_key], key=_stable_order_key)]
    test = [example for file_key in test_files for example in sorted(by_file[file_key], key=_stable_order_key)]
    split_summary = {
        "split_mode": split_mode,
        "train_files": [Path(file_key).name for file_key in train_files],
        "validation_files": [Path(file_key).name for file_key in validation_files],
        "test_files": [Path(file_key).name for file_key in test_files],
        "requested_test_transcripts": requested_test_files,
        "actual_test_transcripts": len(test_files),
    }
    return train, validation, test, split_summary


def _stable_order_key(example: dict) -> str:
    parts = (
        str(example.get("file", "")),
        str(example.get("line", "")),
        str(example.get("timestamp", "")),
        str(example.get("label", "")),
        str(example.get("source", "")),
        str(example.get("current_text", "")),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _take_stable(examples: list[dict], limit: int) -> list[dict]:
    if limit <= 0:
        return []
    ordered = sorted(examples, key=_stable_order_key)
    return ordered[: min(limit, len(ordered))]


def _rebalance_training_examples(
    examples: list[dict],
    *,
    no_action_ratio: float,
    max_no_action_examples: int,
) -> tuple[list[dict], dict[str, int | float]]:
    positives = [example for example in examples if example["label"] != ActionTag.no_action.value]
    no_actions = [example for example in examples if example["label"] == ActionTag.no_action.value]
    if not positives or not no_actions:
        return examples, {
            "positive_examples": len(positives),
            "raw_no_action_examples": len(no_actions),
            "retained_no_action_examples": len(no_actions),
        }

    bounded_ratio = max(0.5, min(6.0, no_action_ratio))
    no_action_budget = min(
        len(no_actions),
        max(1, min(max_no_action_examples, int(round(len(positives) * bounded_ratio)))),
    )
    selected_no_actions = _take_stable(no_actions, no_action_budget)

    rebalanced = sorted([*positives, *selected_no_actions], key=_stable_order_key)
    return rebalanced, {
        "positive_examples": len(positives),
        "raw_no_action_examples": len(no_actions),
        "retained_no_action_examples": len(selected_no_actions),
    }


def _resolve_device(torch_module: object, configured: str):
    if configured == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device=cuda requested but torch.cuda.is_available() is false")
        return torch_module.device("cuda")
    if configured == "auto" and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _mean_pool(torch_module: object, hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _embed_prompts(*, texts: list[str], tokenizer, model, torch_module: object, device, max_length: int, batch_size: int):
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch_module.inference_mode():
            outputs = model(**encoded)
            pooled = _mean_pool(torch_module, outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch_module.nn.functional.normalize(pooled, p=2.0, dim=-1)
        chunks.append(pooled.detach().cpu())
    return torch_module.cat(chunks, dim=0)


def _classification_metrics(*, true_labels: list[int], predicted_labels: list[int], labels: tuple[ActionTag, ...]) -> dict:
    totals = {}
    macro_f1 = 0.0
    for label_index, label in enumerate(labels):
        tp = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == label_index and pred == label_index)
        fp = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth != label_index and pred == label_index)
        fn = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == label_index and pred != label_index)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        totals[label.value] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for truth in true_labels if truth == label_index),
        }
        macro_f1 += f1
    accuracy = sum(1 for truth, pred in zip(true_labels, predicted_labels, strict=False) if truth == pred) / max(1, len(true_labels))
    macro_f1 /= max(1, len(labels))
    action_label_ids = {index for index, label in enumerate(labels) if label != ActionTag.no_action}
    action_tp = sum(
        1
        for truth, pred in zip(true_labels, predicted_labels, strict=False)
        if truth in action_label_ids and pred in action_label_ids
    )
    action_fp = sum(
        1
        for truth, pred in zip(true_labels, predicted_labels, strict=False)
        if truth not in action_label_ids and pred in action_label_ids
    )
    action_fn = sum(
        1
        for truth, pred in zip(true_labels, predicted_labels, strict=False)
        if truth in action_label_ids and pred not in action_label_ids
    )
    action_precision = action_tp / max(1, action_tp + action_fp)
    action_recall = action_tp / max(1, action_tp + action_fn)
    action_f1 = (
        0.0
        if action_precision + action_recall == 0
        else (2 * action_precision * action_recall) / (action_precision + action_recall)
    )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "action_precision": round(action_precision, 4),
        "action_recall": round(action_recall, 4),
        "action_f1": round(action_f1, 4),
        "per_label": totals,
    }


def _calibrate_thresholds(*, probabilities, true_labels: list[int], labels: tuple[ActionTag, ...]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for label_index, label in enumerate(labels):
        best_threshold = 0.55
        best_f1 = -1.0
        for bucket in range(30, 91, 2):
            threshold = bucket / 100.0
            tp = fp = fn = 0
            for row_index, truth in enumerate(true_labels):
                predicted_positive = float(probabilities[row_index][label_index]) >= threshold
                actual_positive = truth == label_index
                if predicted_positive and actual_positive:
                    tp += 1
                elif predicted_positive and not actual_positive:
                    fp += 1
                elif actual_positive and not predicted_positive:
                    fn += 1
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds[label.value] = round(best_threshold, 3)
    return thresholds


def _build_head(torch_module: object, *, hidden_size: int, num_labels: int):
    nn = torch_module.nn
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, num_labels),
    )


def _tensorize_labels(torch_module: object, examples: list[dict], label_to_id: dict[str, int]):
    return torch_module.tensor([label_to_id[example["label"]] for example in examples], dtype=torch_module.long)


def _class_weights(torch_module: object, labels_tensor):
    counts = torch_module.bincount(labels_tensor, minlength=len(_LABELS)).float()
    counts = torch_module.where(counts == 0, torch_module.ones_like(counts), counts)
    weights = counts.sum() / (len(_LABELS) * counts)
    return weights


def _sample_quality_weights(torch_module: object, examples: list[dict]) -> tuple:
    """Build a per-sample weight tensor from source quality and return summary stats.

    Quality weights only boost *positive* (action) examples — high-quality reviewed
    labels get more influence over the decision boundary.  NO_ACTION examples always
    get weight 1.0 regardless of source, because down-weighting the dominant negative
    class destroys precision on the most important label.
    """
    raw_weights: list[float] = []
    for ex in examples:
        if ex.get("label") == ActionTag.no_action.value:
            raw_weights.append(1.0)
        else:
            raw_weights.append(_SOURCE_QUALITY_WEIGHT.get(str(ex.get("source", "")), 1.0))

    tensor = torch_module.tensor(raw_weights, dtype=torch_module.float32)
    mean_weight = float(tensor.mean())
    normalized = tensor / mean_weight  # mean == 1.0 so total gradient magnitude is unchanged

    # Compute effective contribution per source for logging
    source_effective: dict[str, float] = {}
    total_effective = float(normalized.sum())
    for example, weight in zip(examples, normalized.tolist(), strict=False):
        source = str(example.get("source", ""))
        source_effective[source] = source_effective.get(source, 0.0) + weight
    source_share = {
        source: round(effective / total_effective, 4)
        for source, effective in sorted(source_effective.items())
    }
    return normalized, {"source_effective_share": source_share}


def _print_counts(prefix: str, examples: list[dict]) -> None:
    counts = Counter(example["label"] for example in examples)
    print(prefix, flush=True)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}", flush=True)


def _main() -> int:
    args = _parse_args()
    examples, deduped_examples = _load_examples(args.dataset)
    if not examples:
        print(f"No examples found in {args.dataset}", flush=True)
        return 1
    if deduped_examples:
        print(f"Removed {deduped_examples} duplicate examples before splitting.", flush=True)

    train_examples, validation_examples, test_examples, split_summary = _split_examples_by_transcript(
        examples,
        validation_ratio=args.validation_ratio,
        test_transcripts=args.test_transcripts,
        split_mode=args.split_mode,
    )
    if not train_examples or not validation_examples:
        print("Dataset is too small for transcript-level train/validation splitting.", flush=True)
        return 1
    raw_train_examples = list(train_examples)
    train_examples, rebalance_summary = _rebalance_training_examples(
        train_examples,
        no_action_ratio=args.no_action_ratio,
        max_no_action_examples=args.max_no_action_examples,
    )
    _print_counts("TRAIN (REBALANCED)", train_examples)
    _print_counts("VALIDATION", validation_examples)
    _print_counts("TEST", test_examples)
    print("SPLIT", flush=True)
    print(json.dumps(split_summary, indent=2), flush=True)
    print("REBALANCE", flush=True)
    print(json.dumps(rebalance_summary, indent=2), flush=True)

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    device = _resolve_device(torch, args.device)
    model_path = snapshot_download(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True)
    encoder.to(device)
    encoder.eval()

    train_embeddings = _embed_prompts(
        texts=[example["prompt"] for example in train_examples],
        tokenizer=tokenizer,
        model=encoder,
        torch_module=torch,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    validation_embeddings = _embed_prompts(
        texts=[example["prompt"] for example in validation_examples],
        tokenizer=tokenizer,
        model=encoder,
        torch_module=torch,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    test_embeddings = _embed_prompts(
        texts=[example["prompt"] for example in test_examples],
        tokenizer=tokenizer,
        model=encoder,
        torch_module=torch,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    ) if test_examples else None

    label_to_id = {label.value: index for index, label in enumerate(_LABELS)}
    train_labels = _tensorize_labels(torch, train_examples, label_to_id)
    validation_labels = _tensorize_labels(torch, validation_examples, label_to_id)
    test_labels = _tensorize_labels(torch, test_examples, label_to_id) if test_examples else None

    sample_weights, sample_weight_summary = _sample_quality_weights(torch, train_examples)
    print("SAMPLE QUALITY WEIGHTS", flush=True)
    print(json.dumps(sample_weight_summary, indent=2), flush=True)

    head = _build_head(torch, hidden_size=int(train_embeddings.shape[1]), num_labels=len(_LABELS)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _class_weights(torch, train_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    best_state = None
    best_metrics = None
    best_probabilities = None
    best_macro_f1 = -1.0
    patience = 3
    patience_remaining = patience

    train_x = train_embeddings.to(device)
    train_y = train_labels.to(device)
    train_sample_w = sample_weights.to(device)
    validation_x = validation_embeddings.to(device)
    test_x = test_embeddings.to(device) if test_embeddings is not None else None

    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_ids = permutation[start : start + args.batch_size]
            logits = head(train_x[batch_ids])
            per_sample_loss = criterion(logits, train_y[batch_ids])
            loss = (per_sample_loss * train_sample_w[batch_ids]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
            validation_probabilities = torch.softmax(validation_logits, dim=-1).detach().cpu()
            validation_predictions = validation_probabilities.argmax(dim=-1).tolist()
        metrics = _classification_metrics(
            true_labels=validation_labels.tolist(),
            predicted_labels=validation_predictions,
            labels=_LABELS,
        )
        macro_f1 = float(metrics["macro_f1"])
        print(
            f"epoch={epoch} loss={total_loss:.4f} "
            f"val_accuracy={metrics['accuracy']:.4f} val_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            best_metrics = metrics
            best_probabilities = validation_probabilities.numpy()
            patience_remaining = patience
        else:
            patience_remaining -= 1
            if patience_remaining <= 0:
                break

    if best_state is None or best_metrics is None or best_probabilities is None:
        raise RuntimeError("training produced no best checkpoint")

    head.load_state_dict(best_state)
    thresholds = _calibrate_thresholds(
        probabilities=best_probabilities,
        true_labels=validation_labels.tolist(),
        labels=_LABELS,
    )
    test_metrics = None
    if test_x is not None and test_labels is not None:
        head.eval()
        with torch.inference_mode():
            test_logits = head(test_x)
            test_probabilities = torch.softmax(test_logits, dim=-1)
            test_predictions = test_probabilities.argmax(dim=-1).tolist()
        test_metrics = _classification_metrics(
            true_labels=test_labels.tolist(),
            predicted_labels=test_predictions,
            labels=_LABELS,
        )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    save_file(best_state, str(args.artifact_dir / "classifier_head.safetensors"))
    metadata = {
        "artifact_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_name": args.model_name,
        "model_path": model_path,
        "max_length": args.max_length,
        "hidden_size": int(train_embeddings.shape[1]),
        "deduplicated_examples": deduped_examples,
        "labels": [label.value for label in _LABELS],
        "thresholds": thresholds,
        "split": split_summary,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "train_counts": dict(sorted(Counter(example["label"] for example in train_examples).items())),
        "raw_train_counts": dict(sorted(Counter(example["label"] for example in raw_train_examples).items())),
        "validation_counts": dict(sorted(Counter(example["label"] for example in validation_examples).items())),
        "test_counts": dict(sorted(Counter(example["label"] for example in test_examples).items())),
        "rebalance": rebalance_summary,
        "sample_quality_weights": sample_weight_summary,
        "metrics": best_metrics,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    (args.artifact_dir / "classifier_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote classifier artifacts to {args.artifact_dir}", flush=True)
    print("VALIDATION METRICS", flush=True)
    print(json.dumps(best_metrics, indent=2), flush=True)
    if test_metrics is not None:
        print("TEST METRICS", flush=True)
        print(json.dumps(test_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
