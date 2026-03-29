from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from app.core.config import Settings
from app.services.interpretation.ai_transcript_annotator import GeminiCliAnnotator, GeminiTranscriptAnnotator
from app.services.interpretation.build_reviewed_ai_corpus import (
    _build_annotator,
    _load_reviewed_coverage,
)


def test_build_annotator_defaults_to_gemini_cli_backend() -> None:
    args = Namespace(
        backend="gemini_cli",
        model=None,
        min_request_interval_ms=5_500,
    )

    annotator = _build_annotator(args=args, settings=Settings(_env_file=None))

    assert isinstance(annotator, GeminiCliAnnotator)
    assert annotator.model_name == "gemini-cli-gemini-2.5-pro"


def test_build_annotator_preserves_gemini_api_backend() -> None:
    args = Namespace(
        backend="gemini",
        model="gemini-2.5-flash",
        min_request_interval_ms=500,
    )

    annotator = _build_annotator(args=args, settings=Settings(_env_file=None))

    assert isinstance(annotator, GeminiTranscriptAnnotator)
    assert annotator.model_name == "gemini-2.5-flash"


def test_load_reviewed_coverage_prefers_reviewed_examples_jsonl(tmp_path: Path) -> None:
    transcript = (tmp_path / "sample.txt").resolve()
    transcript.write_text("[00:00:01] hello\n", encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "reviewed_examples.jsonl").write_text(
        json.dumps({"file": str(transcript), "line": 1, "label": "ENTER_LONG"}) + "\n",
        encoding="utf-8",
    )
    (corpus_dir / "combined_review.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "file": str((tmp_path / "other.txt").resolve()),
                        "review_status": "accepted",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    covered = _load_reviewed_coverage([str(corpus_dir)])

    assert covered == {str(transcript)}


def test_load_reviewed_coverage_uses_accepted_combined_review_when_jsonl_missing(tmp_path: Path) -> None:
    accepted = (tmp_path / "accepted.txt").resolve()
    accepted.write_text("[00:00:01] accepted\n", encoding="utf-8")
    rejected = (tmp_path / "rejected.txt").resolve()
    rejected.write_text("[00:00:02] rejected\n", encoding="utf-8")
    combined_review = tmp_path / "combined_review.json"
    combined_review.write_text(
        json.dumps(
            {
                "candidates": [
                    {"file": str(accepted), "review_status": "accepted"},
                    {"file": str(accepted), "review_status": "corrected"},
                    {"file": str(rejected), "review_status": "rejected"},
                    {"file": str(rejected), "review_status": "pending"},
                ]
            }
        ),
        encoding="utf-8",
    )

    covered = _load_reviewed_coverage([str(combined_review)])

    assert covered == {str(accepted)}
