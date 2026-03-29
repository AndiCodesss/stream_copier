from __future__ import annotations

from app.services.interpretation.train_local_classifier import (
    _dedupe_examples,
    _rebalance_training_examples,
    _split_examples_by_transcript,
)


def _example(*, label: str, source: str, line: int, current_text: str = "text", prompt: str = "prompt") -> dict:
    return {
        "file": "sample.txt",
        "line": line,
        "timestamp": f"2026-03-12T09:00:{line:02d}+00:00",
        "label": label,
        "source": source,
        "current_text": current_text,
        "analysis_text": current_text,
        "entry_text": current_text,
        "prompt": prompt,
        "symbol": "MNQ 03-26",
        "position_side": "FLAT",
        "last_side": "NONE",
    }


def test_dedupe_examples_keeps_higher_priority_source() -> None:
    examples = [
        _example(label="NO_ACTION", source="unknown", line=1, current_text="same", prompt="same"),
        _example(label="NO_ACTION", source="ai_review_hard_negative", line=2, current_text="same", prompt="same"),
    ]

    deduped = _dedupe_examples(examples)

    assert len(deduped) == 1
    assert deduped[0]["source"] == "ai_review_hard_negative"


def test_rebalance_training_examples_caps_no_action_budget() -> None:
    examples = [
        _example(label="ENTER_LONG", source="rule", line=1, current_text="enter long", prompt="enter long"),
        _example(label="EXIT_ALL", source="rule", line=2, current_text="we're out", prompt="we're out"),
    ]
    examples.extend(
        _example(
            label="NO_ACTION",
            source="ai_review_hard_negative",
            line=10 + index,
            current_text=f"negative {index}",
            prompt=f"negative {index}",
        )
        for index in range(10)
    )

    rebalanced, summary = _rebalance_training_examples(
        examples,
        no_action_ratio=2.0,
        max_no_action_examples=4,
    )

    no_action_count = sum(1 for example in rebalanced if example["label"] == "NO_ACTION")
    assert no_action_count == 4
    assert summary["positive_examples"] == 2
    assert summary["retained_no_action_examples"] == 4


def test_dedupe_examples_prefers_reviewed_labels_over_unknown_sources() -> None:
    examples = [
        _example(label="ENTER_SHORT", source="unknown", line=1, current_text="same", prompt="same"),
        _example(label="ENTER_SHORT", source="ai_review_accept", line=2, current_text="same", prompt="same"),
    ]

    deduped = _dedupe_examples(examples)

    assert len(deduped) == 1
    assert deduped[0]["source"] == "ai_review_accept"


def test_split_examples_by_transcript_reserves_five_holdout_files() -> None:
    examples = []
    for index in range(8):
        file_name = f"session_{index}.txt"
        examples.append(
            {
                **_example(label="ENTER_LONG", source="ai_review_accept", line=index + 1),
                "file": file_name,
                "prompt": f"prompt {index}",
                "current_text": f"current {index}",
                "analysis_text": f"analysis {index}",
                "entry_text": f"entry {index}",
            }
        )

    train, validation, test, summary = _split_examples_by_transcript(
        examples,
        validation_ratio=0.25,
        test_transcripts=5,
    )

    assert len({example["file"] for example in test}) == 5
    assert len({example["file"] for example in validation}) == 1
    assert len({example["file"] for example in train}) == 2
    assert summary["actual_test_transcripts"] == 5


def test_split_examples_by_transcript_prefers_action_files_for_test_holdout() -> None:
    examples = [
        {
            **_example(label="NO_ACTION", source="ai_review_hard_negative", line=1),
            "file": "no_action_only.txt",
            "prompt": "negative",
            "current_text": "negative",
            "analysis_text": "negative",
            "entry_text": "negative",
        },
        {
            **_example(label="ENTER_SHORT", source="ai_review_accept", line=2),
            "file": "action_a.txt",
            "prompt": "action a",
            "current_text": "action a",
            "analysis_text": "action a",
            "entry_text": "action a",
        },
        {
            **_example(label="TRIM", source="ai_review_accept", line=3),
            "file": "action_b.txt",
            "prompt": "action b",
            "current_text": "action b",
            "analysis_text": "action b",
            "entry_text": "action b",
        },
        {
            **_example(label="EXIT_ALL", source="ai_review_accept", line=4),
            "file": "action_c.txt",
            "prompt": "action c",
            "current_text": "action c",
            "analysis_text": "action c",
            "entry_text": "action c",
        },
    ]

    _, _, test, summary = _split_examples_by_transcript(
        examples,
        validation_ratio=0.25,
        test_transcripts=2,
    )

    test_files = {example["file"] for example in test}
    assert "no_action_only.txt" not in test_files
    assert len(test_files) == 2
    assert set(summary["test_files"]) == test_files


def test_split_examples_by_transcript_temporal_recent_holds_out_newest_files() -> None:
    examples = []
    for index, file_name in enumerate(
        [
            "2026-03-01_a.txt",
            "2026-03-02_b.txt",
            "2026-03-03_c.txt",
            "2026-03-04_d.txt",
            "2026-03-05_e.txt",
            "2026-03-06_f.txt",
        ],
        start=1,
    ):
        examples.append(
            {
                **_example(label="ENTER_LONG", source="ai_review_accept", line=index),
                "file": file_name,
                "timestamp": f"2026-03-{index:02d}T09:00:00+00:00",
                "prompt": f"prompt {index}",
                "current_text": f"current {index}",
                "analysis_text": f"analysis {index}",
                "entry_text": f"entry {index}",
            }
        )

    train, validation, test, summary = _split_examples_by_transcript(
        examples,
        validation_ratio=0.25,
        test_transcripts=2,
        split_mode="temporal_recent",
    )

    assert summary["split_mode"] == "temporal_recent"
    assert {example["file"] for example in test} == {"2026-03-05_e.txt", "2026-03-06_f.txt"}
    assert {example["file"] for example in validation} == {"2026-03-04_d.txt"}
    assert {example["file"] for example in train} == {
        "2026-03-01_a.txt",
        "2026-03-02_b.txt",
        "2026-03-03_c.txt",
    }


def test_split_examples_by_transcript_temporal_recent_prefers_filename_date_over_row_timestamp() -> None:
    examples = [
        {
            **_example(label="ENTER_LONG", source="ai_review_accept", line=1),
            "file": "2025-11-12_old.txt",
            "timestamp": "2026-03-12T09:00:00+00:00",
            "prompt": "old",
            "current_text": "old",
            "analysis_text": "old",
            "entry_text": "old",
        },
        {
            **_example(label="ENTER_SHORT", source="ai_review_accept", line=2),
            "file": "2026-03-10_new.txt",
            "timestamp": "2026-03-10T09:00:00+00:00",
            "prompt": "new",
            "current_text": "new",
            "analysis_text": "new",
            "entry_text": "new",
        },
        {
            **_example(label="TRIM", source="ai_review_accept", line=3),
            "file": "transcript_09.03.26.txt",
            "timestamp": "2026-03-09T09:00:00+00:00",
            "prompt": "alt",
            "current_text": "alt",
            "analysis_text": "alt",
            "entry_text": "alt",
        },
    ]

    _, validation, test, summary = _split_examples_by_transcript(
        examples,
        validation_ratio=0.34,
        test_transcripts=1,
        split_mode="temporal_recent",
    )

    assert summary["test_files"] == ["2026-03-10_new.txt"]
    assert {example["file"] for example in validation} == {"transcript_09.03.26.txt"}
