from __future__ import annotations

from pathlib import Path

from app.services.transcription.youtube_captions import (
    _build_output_path,
    _looks_like_upcoming_live_error,
    _parse_json3,
    _parse_vtt,
    _render_transcript,
)


def test_parse_vtt_collapses_rolling_caption_updates() -> None:
    raw = """WEBVTT

00:00:07.000 --> 00:00:08.000
Let's do it

00:00:07.500 --> 00:00:09.000
Let's do it team.

00:00:09.000 --> 00:00:10.000
Groundhog Day
"""

    lines = _parse_vtt(raw)

    assert _render_transcript(lines) == (
        "[00:00:07] Let's do it team.\n"
        "[00:00:09] Groundhog Day\n"
    )


def test_parse_json3_renders_plain_timestamped_text() -> None:
    raw = """
    {
      "events": [
        {"tStartMs": 431000, "segs": [{"utf8": "absolute "}, {"utf8": "banger"}]},
        {"tStartMs": 433000, "segs": [{"utf8": "of a trade"}]}
      ]
    }
    """

    lines = _parse_json3(raw)

    assert _render_transcript(lines) == (
        "[00:07:11] absolute banger\n"
        "[00:07:13] of a trade\n"
    )


def test_build_output_path_uses_date_title_and_video_id() -> None:
    path = _build_output_path(
        out_dir=Path("transcripts"),
        info={
            "id": "abc123",
            "title": "NFP breakdown / Friday recap",
            "upload_date": "20260306",
        },
    )

    assert str(path) == "transcripts/2026-03-06__NFP-breakdown-Friday-recap__abc123.txt"


def test_upcoming_live_error_detection_matches_youtube_message() -> None:
    assert _looks_like_upcoming_live_error("ERROR: [youtube] abc: This live event will begin in 2 hours.")
