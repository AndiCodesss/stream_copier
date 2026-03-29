from __future__ import annotations

import json
from pathlib import Path

from app.models.domain import ActionTag
from app.services.interpretation.ai_annotation_review import (
    _collapse_reviewed_decisions,
    _auto_review_state,
    build_review_training_examples,
    load_or_initialize_review,
)
from app.services.interpretation.ai_transcript_annotator import TranscriptRow


def _row(line: int, timecode: str, text: str) -> TranscriptRow:
    from datetime import UTC, datetime

    hh, mm, ss = (int(part) for part in timecode.split(":"))
    return TranscriptRow(
        line=line,
        timecode=timecode,
        text=text,
        received_at=datetime(2026, 3, 1, hh, mm, ss, tzinfo=UTC),
    )


def test_load_or_initialize_review_resolves_files_and_seeds_pending_candidates(tmp_path: Path) -> None:
    transcript = tmp_path / "sample.txt"
    transcript.write_text("[00:00:10] We are looking to sell.\n", encoding="utf-8")
    report = tmp_path / "annotations.json"
    report.write_text(
        json.dumps(
            {
                "model": "gemini-test",
                "symbol": "MNQ 03-26",
                "market_price": 24600.0,
                "files": [
                    {
                        "file": str(transcript),
                        "annotations": [
                            {
                                "line": 1,
                                "timecode": "00:00:10",
                                "label": "SETUP_SHORT",
                                "side": "SHORT",
                                "confidence": 0.8,
                                "reason": "watching for downside",
                                "evidence_text": "We are looking to sell.",
                                "current_text": "We are looking to sell.",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = load_or_initialize_review(report_path=report, review_path=tmp_path / "review.json")

    assert state["symbol"] == "MNQ 03-26"
    assert len(state["candidates"]) == 1
    candidate = state["candidates"][0]
    assert candidate["review_status"] == "pending"
    assert Path(candidate["file"]) == transcript.resolve()
    assert candidate["original_label"] == "SETUP_SHORT"


def test_collapse_reviewed_decisions_prefers_positive_over_rejected_same_line() -> None:
    candidates = [
        {
            "id": "accept-1",
            "file": "/tmp/sample.txt",
            "line": 10,
            "timecode": "00:00:10",
            "original_label": "ENTER_SHORT",
            "original_side": "SHORT",
            "ai_confidence": 0.9,
            "ai_reason": "direct entry",
            "evidence_text": "I'm in short here.",
            "current_text": "I'm in short here.",
            "review_status": "accepted",
            "reviewed_label": "ENTER_SHORT",
            "reviewed_side": "SHORT",
            "review_note": None,
        },
        {
            "id": "reject-1",
            "file": "/tmp/sample.txt",
            "line": 10,
            "timecode": "00:00:10",
            "original_label": "TRIM",
            "original_side": "SHORT",
            "ai_confidence": 0.7,
            "ai_reason": "wrong second label",
            "evidence_text": "I'm in short here.",
            "current_text": "I'm in short here.",
            "review_status": "rejected",
            "reviewed_label": "NO_ACTION",
            "reviewed_side": None,
            "review_note": None,
        },
    ]

    selected, dropped = _collapse_reviewed_decisions(candidates)

    assert dropped == 1
    decision = selected["/tmp/sample.txt"][10]
    assert decision.label == ActionTag.enter_short
    assert decision.source == "ai_review_accept"


def test_build_review_training_examples_exports_hard_negative_for_rejected_false_positive() -> None:
    rows_by_file = {
        "/tmp/sample.txt": [
            _row(1, "00:00:01", "We're looking to sell here."),
            _row(2, "00:00:03", "Still waiting."),
        ]
    }
    selected_by_file = {
        "/tmp/sample.txt": {
            1: _collapse_reviewed_decisions(
                [
                    {
                        "id": "reject-1",
                        "file": "/tmp/sample.txt",
                        "line": 1,
                        "timecode": "00:00:01",
                        "original_label": "ENTER_SHORT",
                        "original_side": "SHORT",
                        "ai_confidence": 0.82,
                        "ai_reason": "bad false positive",
                        "evidence_text": "We're looking to sell here.",
                        "current_text": "We're looking to sell here.",
                        "review_status": "rejected",
                        "reviewed_label": "NO_ACTION",
                        "reviewed_side": None,
                        "review_note": "setup only",
                    }
                ]
            )[0]["/tmp/sample.txt"][1]
        }
    }

    examples = build_review_training_examples(
        rows_by_file=rows_by_file,
        selected_by_file=selected_by_file,
        symbol="MNQ 03-26",
        market_price=24600.0,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example["label"] == "NO_ACTION"
    assert example["source"] == "ai_review_hard_negative"
    assert example["original_label"] == "ENTER_SHORT"
    assert example["review_note"] == "setup only"


def test_build_review_training_examples_preserves_reviewed_setup_label() -> None:
    rows_by_file = {
        "/tmp/sample.txt": [
            _row(1, "00:00:01", "We're looking to sell here."),
            _row(2, "00:00:03", "Small piece on here."),
        ]
    }
    selected, _ = _collapse_reviewed_decisions(
        [
            {
                "id": "setup-1",
                "file": "/tmp/sample.txt",
                "line": 1,
                "timecode": "00:00:01",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.8,
                "ai_reason": "fixed to setup",
                "evidence_text": "We're looking to sell here.",
                "current_text": "We're looking to sell here.",
                "review_status": "corrected",
                "reviewed_label": "SETUP_SHORT",
                "reviewed_side": "SHORT",
                "review_note": "watchlist only",
            },
            {
                "id": "entry-1",
                "file": "/tmp/sample.txt",
                "line": 2,
                "timecode": "00:00:03",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.95,
                "ai_reason": "direct entry",
                "evidence_text": "Small piece on here.",
                "current_text": "Small piece on here.",
                "review_status": "accepted",
                "reviewed_label": "ENTER_SHORT",
                "reviewed_side": "SHORT",
                "review_note": None,
            },
        ]
    )

    examples = build_review_training_examples(
        rows_by_file=rows_by_file,
        selected_by_file=selected,
        symbol="MNQ 03-26",
        market_price=24600.0,
    )

    assert [example["label"] for example in examples] == ["SETUP_SHORT", "ENTER_SHORT"]
    assert examples[0]["source"] == "ai_review_corrected"
    assert examples[1]["source"] == "ai_review_accept"


def test_auto_review_state_corrects_setup_like_false_positive() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.8,
                "ai_reason": "bad candidate",
                "evidence_text": "We're looking to sell.",
                "current_text": "We're looking to sell.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            }
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    candidate = state["candidates"][0]
    assert candidate["review_status"] == "corrected"
    assert candidate["reviewed_label"] == "SETUP_SHORT"
    assert candidate["reviewed_side"] == "SHORT"
    assert counts["pending"] == 0


def test_auto_review_state_rejects_future_add_and_second_person_trim() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "ADD",
                "original_side": "SHORT",
                "ai_confidence": 0.8,
                "ai_reason": "bad add",
                "evidence_text": "I'm going to add on a pop now.",
                "current_text": "I'm going to add on a pop now.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            },
            {
                "id": "2",
                "file": "/tmp/sample.txt",
                "line": 12,
                "timecode": "00:00:12",
                "original_label": "TRIM",
                "original_side": "SHORT",
                "ai_confidence": 0.8,
                "ai_reason": "bad trim",
                "evidence_text": "so you can pay yourself some here",
                "current_text": "so you can pay yourself some here",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            },
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    assert state["candidates"][0]["review_status"] == "rejected"
    assert state["candidates"][0]["reviewed_label"] == "NO_ACTION"
    assert state["candidates"][1]["review_status"] == "rejected"
    assert state["candidates"][1]["reviewed_label"] == "NO_ACTION"
    assert counts["rejected"] == 2


def test_auto_review_state_accepts_explicit_entry() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.95,
                "ai_reason": "good entry",
                "evidence_text": "Small piece on here.",
                "current_text": "Small piece on here.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            }
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    assert state["candidates"][0]["review_status"] == "accepted"
    assert state["candidates"][0]["reviewed_label"] == "ENTER_SHORT"
    assert counts["accepted"] == 1


def test_auto_review_state_rejects_other_instrument_candidate() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "TRIM",
                "original_side": "LONG",
                "ai_confidence": 0.9,
                "ai_reason": "other instrument",
                "evidence_text": "UNH, I trimmed a bit.",
                "current_text": "UNH, I trimmed a bit.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            }
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    assert state["candidates"][0]["review_status"] == "rejected"
    assert state["candidates"][0]["reviewed_label"] == "NO_ACTION"
    assert counts["rejected"] == 1


def test_auto_review_state_corrects_entry_like_add_to_add() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.9,
                "ai_reason": "wrong tag",
                "evidence_text": "Added back in on this.",
                "current_text": "Added back in on this.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            }
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    assert state["candidates"][0]["review_status"] == "corrected"
    assert state["candidates"][0]["reviewed_label"] == "ADD"
    assert counts["corrected"] == 1


def test_auto_review_state_accepts_back_on_again_entry_and_done_exit() -> None:
    state = {
        "candidates": [
            {
                "id": "1",
                "file": "/tmp/sample.txt",
                "line": 10,
                "timecode": "00:00:10",
                "original_label": "ENTER_SHORT",
                "original_side": "SHORT",
                "ai_confidence": 0.9,
                "ai_reason": "re-entry",
                "evidence_text": "So, now I stick this back on here again.",
                "current_text": "So, now I stick this back on here again.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            },
            {
                "id": "2",
                "file": "/tmp/sample.txt",
                "line": 12,
                "timecode": "00:00:12",
                "original_label": "EXIT_ALL",
                "original_side": "SHORT",
                "ai_confidence": 0.92,
                "ai_reason": "finished",
                "evidence_text": "Out. Done. Finished.",
                "current_text": "Out. Done. Finished.",
                "review_status": "pending",
                "reviewed_label": None,
                "reviewed_side": None,
                "review_note": None,
            },
        ]
    }

    counts = _auto_review_state(state, overwrite_reviewed=False)

    assert state["candidates"][0]["review_status"] == "accepted"
    assert state["candidates"][0]["reviewed_label"] == "ENTER_SHORT"
    assert state["candidates"][1]["review_status"] == "accepted"
    assert state["candidates"][1]["reviewed_label"] == "EXIT_ALL"
    assert counts["accepted"] == 2
