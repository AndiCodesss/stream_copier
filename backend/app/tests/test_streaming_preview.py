from __future__ import annotations

from app.core.config import Settings
from app.services.transcription.local_whisper import LocalWhisperTranscriber
from app.services.transcription.segmenter import ReadyAudioSegment
from app.services.transcription.streaming_preview import StreamingPreviewAssembler


def test_streaming_preview_assembler_builds_prompt_from_committed_context() -> None:
    assembler = StreamingPreviewAssembler(context_words=3, stability_margin_words=1)
    utterance_id = "abc"

    assert assembler.stabilize(utterance_id=utterance_id, tail_text="we are testing the") == "we are testing the"
    assert assembler.stabilize(utterance_id=utterance_id, tail_text="we are testing the stream") == "we are testing the stream"

    prompt = assembler.build_prompt(utterance_id=utterance_id, base_prompt="Trading livestream.")

    assert "Trading livestream." in prompt
    assert "Recent confirmed transcript: we are testing" in prompt


def test_streaming_preview_assembler_keeps_committed_prefix_when_tail_rolls_forward() -> None:
    assembler = StreamingPreviewAssembler(context_words=8, stability_margin_words=1)
    utterance_id = "abc"

    assembler.stabilize(utterance_id=utterance_id, tail_text="spy into that level we")
    assembler.stabilize(utterance_id=utterance_id, tail_text="spy into that level we just")
    stabilized = assembler.stabilize(utterance_id=utterance_id, tail_text="we just talked about right here")

    assert stabilized.startswith("spy into that level")
    assert "talked about right here" in stabilized


def test_streaming_transcriber_preview_uses_audio_tail_and_committed_context(monkeypatch) -> None:
    settings = Settings(
        transcription_engine="streaming",
        transcription_preview_tail_ms=1_000,
        transcription_preview_context_words=6,
        transcription_preview_stability_margin_words=1,
        speech_vad_backend="energy",
    )
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="Trading livestream.",
        on_segment=_noop,
    )
    snapshot = ReadyAudioSegment(
        utterance_id="abc",
        pcm16=b"\x00\x00" * 24_000,
        duration_ms=1_500,
        voice_duration_ms=1_500,
        started_monotonic=0.0,
        ready_monotonic=1.5,
    )
    calls: list[tuple[int, str | None]] = []
    responses = iter(
        (
            ("spy into that level we", 0.81),
            ("spy into that level we just", 0.82),
            ("we just talked about", 0.83),
        )
    )

    def fake_transcribe_with_profile(
        _model,
        pcm16: bytes,
        *,
        beam_size: int,
        initial_prompt: str | None = None,
        decode_profile: str = "preview",
    ):
        calls.append((len(pcm16), initial_prompt))
        assert decode_profile == "preview"
        return next(responses)

    monkeypatch.setattr(transcriber, "_transcribe_with_profile", fake_transcribe_with_profile)

    first_text, _ = transcriber._transcribe_preview_snapshot(object(), snapshot)
    second_text, _ = transcriber._transcribe_preview_snapshot(object(), snapshot)
    third_text, _ = transcriber._transcribe_preview_snapshot(object(), snapshot)

    assert first_text == "spy into that level we"
    assert second_text == "spy into that level we just"
    assert third_text.startswith("spy into that level")
    assert "talked about" in third_text
    assert all(length == 32_000 for length, _ in calls)
    assert calls[0][1] == "Trading livestream."
    assert calls[2][1] is not None
    assert "Recent confirmed transcript: spy into that level" in calls[2][1]


async def _noop(*_args, **_kwargs) -> None:
    return None
