from __future__ import annotations

import json
from datetime import UTC, datetime

from app.models.domain import ActionTag, TradeSide
from app.services.interpretation.ai_transcript_annotator import (
    AiAnnotation,
    GeminiCliAnnotator,
    TranscriptRow,
    _system_prompt,
    _coerce_chunk_annotations,
    _extract_gemini_cli_response_text,
    TranscriptChunk,
    build_candidate_chunks,
    build_training_examples,
    build_transcript_chunks,
    merge_chunk_annotations,
)


def _row(line: int, timecode: str, text: str) -> TranscriptRow:
    hh, mm, ss = (int(part) for part in timecode.split(":"))
    return TranscriptRow(
        line=line,
        timecode=timecode,
        text=text,
        received_at=datetime(2026, 3, 1, hh, mm, ss, tzinfo=UTC),
    )


def test_build_transcript_chunks_uses_overlap() -> None:
    rows = [
        _row(1, "00:00:01", "one"),
        _row(2, "00:00:02", "two"),
        _row(3, "00:00:03", "three"),
        _row(4, "00:00:04", "four"),
        _row(5, "00:00:05", "five"),
        _row(6, "00:00:06", "six"),
    ]

    chunks = build_transcript_chunks(
        file="sample.txt",
        rows=rows,
        chunk_lines=3,
        overlap_lines=1,
        max_chars=10_000,
    )

    assert len(chunks) == 3
    assert [chunk.start_line for chunk in chunks] == [1, 3, 5]
    assert [chunk.end_line for chunk in chunks] == [3, 5, 6]


def test_build_candidate_chunks_focuses_on_trade_like_windows() -> None:
    rows = [
        _row(1, "00:00:01", "Good morning everyone."),
        _row(2, "00:00:02", "Watching the open here."),
        _row(3, "00:00:03", "Looking for a short into 500s."),
        _row(4, "00:00:04", "If we get it, fine."),
        _row(5, "00:00:05", "Gonna have to put something on here in case."),
        _row(6, "00:00:06", "Very tight stop."),
        _row(7, "00:00:07", "Back to market structure."),
    ]

    chunks = build_candidate_chunks(
        file="sample.txt",
        rows=rows,
        chunk_lines=10,
        max_chars=10_000,
        context_before=1,
        context_after=1,
        merge_gap_lines=1,
    )

    assert len(chunks) == 1
    assert chunks[0].start_line == 2
    assert chunks[0].end_line == 6


def test_merge_chunk_annotations_prefers_higher_confidence_duplicate() -> None:
    low = AiAnnotation(
        file="sample.txt",
        line=10,
        timecode="00:00:10",
        label=ActionTag.enter_short,
        side=TradeSide.short,
        confidence=0.61,
        evidence_text="piece on short",
        reason="low",
        chunk_index=0,
        chunk_start_line=1,
        chunk_end_line=20,
        current_text="piece on short",
    )
    high = AiAnnotation(
        file="sample.txt",
        line=10,
        timecode="00:00:10",
        label=ActionTag.enter_short,
        side=TradeSide.short,
        confidence=0.92,
        evidence_text="piece on short",
        reason="high",
        chunk_index=1,
        chunk_start_line=5,
        chunk_end_line=25,
        current_text="piece on short",
    )

    merged = merge_chunk_annotations(
        [
            type("Result", (), {"file": "sample.txt", "chunk_index": 0, "annotations": (low,), "error": None})(),
            type("Result", (), {"file": "sample.txt", "chunk_index": 1, "annotations": (high,), "error": None})(),
        ]
    )

    assert len(merged["sample.txt"]) == 1
    assert merged["sample.txt"][0].confidence == 0.92
    assert merged["sample.txt"][0].reason == "high"


def test_build_training_examples_renders_prompt_from_ai_annotation() -> None:
    rows = [
        _row(1, "00:00:01", "Looking for a short here."),
        _row(2, "00:00:03", "Gonna have to put something on here in case."),
    ]
    annotations = [
        AiAnnotation(
            file="sample.txt",
            line=2,
            timecode="00:00:03",
            label=ActionTag.enter_short,
            side=TradeSide.short,
            confidence=0.9,
            evidence_text="Gonna have to put something on here in case.",
            reason="direct entry seed",
            chunk_index=0,
            chunk_start_line=1,
            chunk_end_line=2,
            current_text="Gonna have to put something on here in case.",
        )
    ]

    examples = build_training_examples(
        rows=rows,
        annotations=annotations,
        symbol="MNQ 03-26",
        market_price=25045.0,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.label == "ENTER_SHORT"
    assert example.source == "ai_gemini"
    assert example.current_text == "Gonna have to put something on here in case."
    assert "current=Gonna have to put something on here in case." in example.prompt
    assert example.last_side == "NONE"


def test_system_prompt_calls_out_setup_vs_entry_failure_modes() -> None:
    prompt = _system_prompt()

    assert "Prefer false negatives over false positives." in prompt
    assert "We're looking to sell. => SETUP_SHORT" in prompt
    assert "We can look for this short here. => SETUP_SHORT, not ENTER_SHORT" in prompt
    assert "Looking for one more push up to get some more size on. => omit" in prompt
    assert "I will try and add on pops where available. => omit" in prompt
    assert "I'm in short here versus VWAP. => ENTER_SHORT" in prompt
    assert "UNH, I trimmed a bit because it didn't do the move I wanted. => omit" in prompt


def test_coerce_chunk_annotations_preserves_setup_labels() -> None:
    chunk = TranscriptChunk(
        file="sample.txt",
        chunk_index=0,
        rows=(
            _row(10, "00:00:10", "We're looking to sell here."),
            _row(11, "00:00:11", "Still watching this level."),
        ),
    )

    annotations = _coerce_chunk_annotations(
        chunk=chunk,
        payload={
            "events": [
                {
                    "line": 10,
                    "timestamp": "00:00:10",
                    "label": "SETUP_SHORT",
                    "side": "SHORT",
                    "confidence": 0.88,
                    "evidence_text": "We're looking to sell here.",
                    "reason": "watching for a short setup",
                }
            ]
        },
    )

    assert len(annotations) == 1
    assert annotations[0].label == ActionTag.setup_short
    assert annotations[0].side == TradeSide.short


def test_coerce_chunk_annotations_realigns_misaligned_entry_to_nearby_explicit_line() -> None:
    chunk = TranscriptChunk(
        file="sample.txt",
        chunk_index=0,
        rows=(
            _row(10, "00:00:10", "Still holding 60s. Got to be careful."),
            _row(11, "00:00:11", "Test of 70s. Might cut this."),
            _row(12, "00:00:12", "Don't want to get caught in a squeeze."),
            _row(13, "00:00:13", "I'm in short."),
            _row(14, "00:00:14", "Trimming my size a bit here."),
        ),
    )

    annotations = _coerce_chunk_annotations(
        chunk=chunk,
        payload={
            "events": [
                {
                    "line": 12,
                    "timestamp": "00:00:12",
                    "label": "ENTER_SHORT",
                    "side": "SHORT",
                    "confidence": 0.93,
                    "evidence_text": "I'm in short.",
                    "reason": "entry is live in nearby context",
                }
            ]
        },
    )

    assert len(annotations) == 1
    assert annotations[0].line == 13
    assert annotations[0].timecode == "00:00:13"
    assert annotations[0].current_text == "I'm in short."


def test_extract_gemini_cli_response_text_reads_wrapped_response_field() -> None:
    output = json.dumps(
        {
            "response": "{\"events\":[]}",
            "stats": {"models": {"gemini-2.5-pro": {"api": {"totalRequests": 1}}}},
        }
    )

    extracted = _extract_gemini_cli_response_text(output)

    assert extracted == "{\"events\":[]}"


def test_extract_gemini_cli_response_text_ignores_prefixed_tool_logs() -> None:
    output = "\n".join(
        [
            "Loaded cached credentials.",
            "Error executing tool read_file: File not found.",
            json.dumps({"response": "{\"events\":[]}"}),
        ]
    )

    extracted = _extract_gemini_cli_response_text(output)

    assert extracted == "{\"events\":[]}"


def test_extract_gemini_cli_response_text_reads_stream_json_assistant_deltas() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "init", "model": "gemini-2.5-flash"}),
            json.dumps({"type": "message", "role": "user", "content": "prompt"}),
            json.dumps({"type": "message", "role": "assistant", "content": "```json\n{\"events\":", "delta": True}),
            json.dumps({"type": "message", "role": "assistant", "content": "[]}\n```", "delta": True}),
            json.dumps({"type": "result", "status": "success"}),
        ]
    )

    extracted = _extract_gemini_cli_response_text(output)

    assert extracted == "```json\n{\"events\":[]}\n```"


def test_extract_gemini_cli_response_text_ignores_stream_error_without_success() -> None:
    output = "\n".join(
        [
            json.dumps({"type": "init", "model": "gemini-2.5-pro"}),
            json.dumps({"type": "result", "status": "error", "error": {"message": "quota exhausted"}}),
        ]
    )

    extracted = _extract_gemini_cli_response_text(output)

    assert extracted is None


def test_gemini_cli_annotator_parses_wrapped_json_response() -> None:
    annotator = GeminiCliAnnotator(model="gemini-2.5-pro")
    stdout = json.dumps(
        {
            "response": (
                "{\"events\":[{\"line\":10,\"timestamp\":\"00:00:10\",\"label\":\"SETUP_SHORT\","
                "\"side\":\"SHORT\",\"confidence\":0.88,\"evidence_text\":\"Looking for a short here.\","
                "\"reason\":\"watching for downside\"}]}"
            )
        }
    )

    parsed = annotator._parse_stdout(stdout)

    assert parsed == {
        "events": [
            {
                "line": 10,
                "timestamp": "00:00:10",
                "label": "SETUP_SHORT",
                "side": "SHORT",
                "confidence": 0.88,
                "evidence_text": "Looking for a short here.",
                "reason": "watching for downside",
            }
        ]
    }


def test_gemini_cli_annotator_parses_stream_json_response() -> None:
    annotator = GeminiCliAnnotator(model="gemini-2.5-flash")
    stdout = "\n".join(
        [
            json.dumps({"type": "init", "model": "gemini-2.5-flash"}),
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "{\"events\":[{\"line\":10,\"timestamp\":\"00:00:10\",",
                    "delta": True,
                }
            ),
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "\"label\":\"ENTER_SHORT\",\"side\":\"SHORT\",\"confidence\":0.9,\"evidence_text\":\"I'm in short.\",\"reason\":\"live entry\"}]}",
                    "delta": True,
                }
            ),
            json.dumps({"type": "result", "status": "success"}),
        ]
    )

    parsed = annotator._parse_stdout(stdout)

    assert parsed == {
        "events": [
            {
                "line": 10,
                "timestamp": "00:00:10",
                "label": "ENTER_SHORT",
                "side": "SHORT",
                "confidence": 0.9,
                "evidence_text": "I'm in short.",
                "reason": "live entry",
            }
        ]
    }
