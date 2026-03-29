from __future__ import annotations

import json
from pathlib import Path

from app.services.interpretation.build_reviewed_execution_dataset import _normalize_path, build_execution_dataset


def _write_dataset(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _example(*, file: str, line: int, label: str, source: str, original_side: str | None = None) -> dict:
    return {
        "file": file,
        "line": line,
        "timestamp": f"2026-03-12T09:00:{line:02d}+00:00",
        "label": label,
        "source": source,
        "current_text": f"text {line}",
        "analysis_text": f"text {line}",
        "entry_text": f"text {line}",
        "prompt": f"prompt {line}",
        "symbol": "MNQ 03-26",
        "position_side": "LONG" if original_side == "LONG" else "FLAT",
        "last_side": original_side or "NONE",
        "original_side": original_side,
    }


def test_build_execution_dataset_maps_setups_and_adds(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_examples.jsonl"
    transcript = tmp_path / "sample.txt"
    transcript.write_text("", encoding="utf-8")
    _write_dataset(
        dataset,
        [
            _example(file=str(transcript), line=1, label="SETUP_SHORT", source="ai_review_accept", original_side="SHORT"),
            _example(file=str(transcript), line=2, label="ADD", source="ai_review_corrected", original_side="LONG"),
            _example(file=str(transcript), line=3, label="ENTER_SHORT", source="ai_review_accept", original_side="SHORT"),
        ],
    )

    merged, summary = build_execution_dataset(dataset_paths=[dataset])

    assert [row["label"] for row in merged] == ["NO_ACTION", "ENTER_LONG", "ENTER_SHORT"]
    assert merged[0]["classifier_label_note"] == "setup_short_to_no_action"
    assert merged[1]["classifier_label_note"] == "add_to_enter_long"
    assert summary["output_label_counts"]["NO_ACTION"] == 1
    assert summary["output_label_counts"]["ENTER_LONG"] == 1


def test_build_execution_dataset_prefers_later_inputs_on_same_line(tmp_path: Path) -> None:
    transcript = tmp_path / "sample.txt"
    transcript.write_text("", encoding="utf-8")
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    _write_dataset(
        older,
        [_example(file=str(transcript), line=10, label="MOVE_STOP", source="ai_review_accept", original_side="LONG")],
    )
    _write_dataset(
        newer,
        [_example(file=str(transcript), line=10, label="NO_ACTION", source="ai_review_hard_negative", original_side=None)],
    )

    merged, summary = build_execution_dataset(dataset_paths=[older, newer])

    assert len(merged) == 1
    assert merged[0]["label"] == "NO_ACTION"
    assert summary["replaced_same_line_examples"] == 1


def test_build_execution_dataset_drops_add_without_side(tmp_path: Path) -> None:
    dataset = tmp_path / "reviewed_examples.jsonl"
    transcript = tmp_path / "sample.txt"
    transcript.write_text("", encoding="utf-8")
    _write_dataset(
        dataset,
        [_example(file=str(transcript), line=5, label="ADD", source="ai_review_accept", original_side=None)],
    )

    merged, summary = build_execution_dataset(dataset_paths=[dataset])

    assert merged == []
    assert summary["dropped_counts"]["unmapped_add"] == 1


def test_normalize_path_canonicalizes_existing_case(tmp_path: Path) -> None:
    parent = tmp_path / "MixedCase"
    parent.mkdir()
    transcript = parent / "Sample.txt"
    transcript.write_text("", encoding="utf-8")

    normalized = _normalize_path(str((tmp_path / "mixedcase" / "sample.txt").resolve()), dataset_path=tmp_path / "rows.jsonl")

    assert normalized == str(transcript.resolve())
