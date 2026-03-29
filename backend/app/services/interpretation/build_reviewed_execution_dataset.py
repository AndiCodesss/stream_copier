from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import ActionTag
from app.services.interpretation.path_utils import canonicalize_existing_path

_OUTPUT_LABELS = {
    ActionTag.no_action,
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
}
_SETUP_LABELS = {ActionTag.setup_long, ActionTag.setup_short}
_SIDE_TO_ENTRY_LABEL = {
    "LONG": ActionTag.enter_long,
    "SHORT": ActionTag.enter_short,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed corpora into a classifier-ready execution dataset.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Reviewed corpus directories or reviewed_examples.jsonl files. Later inputs override earlier ones on the same transcript line.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("data/training_data.jsonl"),
        help="Output JSONL path for the merged execution dataset.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/training_data_summary.json"),
        help="Summary JSON path.",
    )
    return parser.parse_args()


def _resolve_reviewed_jsonl_inputs(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidate = path / "reviewed_examples.jsonl"
            if candidate.is_file():
                resolved.append(candidate)
            continue
        if path.is_file() and path.suffix == ".jsonl":
            resolved.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def _normalize_path(raw: str, *, dataset_path: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(canonicalize_existing_path(path))
    candidates = (
        Path.cwd() / path,
        Path.cwd() / "backend" / path,
        dataset_path.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(canonicalize_existing_path(candidate.resolve()))
    return str(canonicalize_existing_path((Path.cwd() / path).resolve()))


def _normalize_side_token(value: object) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if token in {"LONG", "SHORT"}:
        return token
    return None


def _side_for_add(example: dict) -> str | None:
    for key in ("original_side", "reviewed_side", "position_side", "last_side"):
        token = _normalize_side_token(example.get(key))
        if token is not None:
            return token
    return None


def _normalize_example(*, example: dict, dataset_path: Path) -> tuple[dict | None, str | None]:
    try:
        label = ActionTag(str(example["label"]))
    except Exception:
        return None, "unknown_label"

    normalized = dict(example)
    normalized["file"] = _normalize_path(str(example.get("file", "")), dataset_path=dataset_path)
    note_parts: list[str] = []

    if label in _SETUP_LABELS:
        normalized["label"] = ActionTag.no_action.value
        note_parts.append(f"{label.value.lower()}_to_no_action")
    elif label == ActionTag.add:
        side = _side_for_add(example)
        if side is None:
            return None, "unmapped_add"
        normalized["label"] = _SIDE_TO_ENTRY_LABEL[side].value
        note_parts.append(f"add_to_{normalized['label'].lower()}")
    elif label not in _OUTPUT_LABELS:
        return None, "unsupported_label"

    if note_parts:
        existing_note = str(normalized.get("classifier_label_note") or "").strip()
        notes = [existing_note] if existing_note else []
        notes.extend(note_parts)
        normalized["classifier_label_note"] = "; ".join(notes)
    return normalized, None


def _line_key(example: dict) -> tuple[str, int]:
    return (str(example["file"]), int(example["line"]))


def _stable_key(example: dict) -> str:
    parts = (
        str(example.get("file", "")),
        str(example.get("line", "")),
        str(example.get("timestamp", "")),
        str(example.get("label", "")),
        str(example.get("source", "")),
        str(example.get("current_text", "")),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _sort_key(example: dict) -> tuple[str, int, str, str]:
    return (
        str(example.get("file", "")),
        int(example.get("line", 0)),
        str(example.get("timestamp", "")),
        _stable_key(example),
    )


def build_execution_dataset(*, dataset_paths: list[Path]) -> tuple[list[dict], dict]:
    merged_by_line: dict[tuple[str, int], dict] = {}
    input_counts: dict[str, int] = {}
    raw_label_counts = Counter()
    output_label_counts = Counter()
    dropped_counts = Counter()
    replaced = 0

    for dataset_path in dataset_paths:
        dataset_key = str(dataset_path.resolve())
        input_counts[dataset_key] = 0
        with dataset_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                example = json.loads(raw_line)
                input_counts[dataset_key] += 1
                raw_label_counts.update([str(example.get("label", ""))])
                normalized, drop_reason = _normalize_example(example=example, dataset_path=dataset_path)
                if normalized is None:
                    dropped_counts.update([str(drop_reason or "dropped")])
                    continue
                key = _line_key(normalized)
                if key in merged_by_line:
                    replaced += 1
                merged_by_line[key] = normalized

    merged = sorted(merged_by_line.values(), key=_sort_key)
    output_label_counts.update(example["label"] for example in merged)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_datasets": input_counts,
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "output_label_counts": dict(sorted(output_label_counts.items())),
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "replaced_same_line_examples": replaced,
        "exported_examples": len(merged),
    }
    return merged, summary


def _main() -> int:
    args = _parse_args()
    dataset_paths = _resolve_reviewed_jsonl_inputs(args.inputs)
    if not dataset_paths:
        print("No reviewed_examples.jsonl inputs found.", flush=True)
        return 1

    merged, summary = build_execution_dataset(dataset_paths=dataset_paths)
    if not merged:
        print("No execution examples were produced from the reviewed corpora.", flush=True)
        return 1

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_out.open("w", encoding="utf-8") as handle:
        for example in merged:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(merged)} execution examples to {args.jsonl_out}", flush=True)
    print(f"Wrote dataset summary to {args.summary_out}", flush=True)
    for label, count in sorted(summary["output_label_counts"].items()):
        print(f"{label}\t{count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
