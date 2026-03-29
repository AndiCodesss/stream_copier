from __future__ import annotations

import json
from pathlib import Path

from app.services.interpretation.build_hard_negative_corpus import build_hard_negative_corpus


def _write_reviewed(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_hard_negative_corpus_mines_retrospective_commentary(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript = transcripts_dir / "2026-03-11_sample.txt"
    transcript.write_text(
        "\n".join(
            [
                "[00:01:00] I thought they were going to go without me.",
                "[00:01:02] I put a small piece on and took a loss.",
                "[00:01:05] So now I need to wait for the proper level.",
            ]
        ),
        encoding="utf-8",
    )
    reviewed = tmp_path / "reviewed_examples.jsonl"
    _write_reviewed(reviewed, [])

    examples, summary = build_hard_negative_corpus(
        transcripts_dir=transcripts_dir,
        reviewed_paths=[reviewed],
        excluded_file_names=set(),
        max_per_pattern=10,
    )

    assert len(examples) == 1
    assert examples[0]["label"] == "NO_ACTION"
    assert examples[0]["candidate_family"] == "retrospective_entry_loss"
    assert "I put a small piece on and took a loss." in examples[0]["current_text"]
    assert summary["candidate_family_counts"]["retrospective_entry_loss"] == 1


def test_build_hard_negative_corpus_skips_reviewed_and_guarded_positive_lines(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript = transcripts_dir / "2026-03-12_sample.txt"
    transcript.write_text(
        "\n".join(
            [
                "[00:02:00] We are long from down here.",
                "[00:02:02] I don't want to be entering long here.",
                "[00:02:04] If you were long you could pay yourself.",
            ]
        ),
        encoding="utf-8",
    )
    reviewed = tmp_path / "reviewed_examples.jsonl"
    _write_reviewed(
        reviewed,
        [
            {
                "file": str(transcript),
                "line": 2,
                "timestamp": "2026-03-12T00:02:02+00:00",
                "label": "ENTER_LONG",
                "source": "ai_review_accept",
                "current_text": "I don't want to be entering long here.",
                "analysis_text": "i don't want to be entering long here.",
                "entry_text": "i don't want to be entering long here.",
                "prompt": "prompt",
            }
        ],
    )

    examples, summary = build_hard_negative_corpus(
        transcripts_dir=transcripts_dir,
        reviewed_paths=[reviewed],
        excluded_file_names=set(),
        max_per_pattern=10,
    )

    assert examples == []
    assert summary["exported_examples"] == 0


def test_build_hard_negative_corpus_respects_excluded_files(tmp_path: Path) -> None:
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    transcript = transcripts_dir / "2026-03-13_sample.txt"
    transcript.write_text("[00:03:00] If you were short this looked good.", encoding="utf-8")
    reviewed = tmp_path / "reviewed_examples.jsonl"
    _write_reviewed(reviewed, [])

    examples, summary = build_hard_negative_corpus(
        transcripts_dir=transcripts_dir,
        reviewed_paths=[reviewed],
        excluded_file_names={transcript.name},
        max_per_pattern=10,
    )

    assert examples == []
    assert summary["excluded_files"] == [transcript.name]
