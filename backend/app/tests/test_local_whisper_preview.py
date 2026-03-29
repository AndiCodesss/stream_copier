from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.transcription.local_whisper import LocalWhisperTranscriber
from app.services.transcription.segmenter import ReadyAudioSegment


async def test_preview_scheduler_marks_preview_window(monkeypatch) -> None:
    settings = Settings(
        speech_preview_min_duration_ms=1_000,
        speech_preview_interval_ms=1_000,
        transcription_device="cuda",
    )
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    transcriber._segmenter._speaking = True
    transcriber._segmenter._utterance_id = "abc"
    transcriber._segmenter._buffer.extend(b"\x00\x00" * 20_000)
    transcriber._segmenter._voice_ms = 1_500
    transcriber._segmenter._silence_ms = 0
    preview_calls: list[str] = []

    async def fake_run_preview(snapshot, _generation: int) -> None:
        preview_calls.append(snapshot.utterance_id)

    monkeypatch.setattr(transcriber, "_run_preview", fake_run_preview)

    await transcriber._maybe_schedule_preview()

    assert transcriber._preview_task is not None
    await transcriber._preview_task
    assert preview_calls == ["abc"]


async def test_preview_scheduler_skips_short_cpu_preview(monkeypatch) -> None:
    settings = Settings(
        speech_preview_min_duration_ms=1_000,
        speech_preview_interval_ms=1_000,
        transcription_device="cpu",
    )
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    transcriber._segmenter._speaking = True
    transcriber._segmenter._utterance_id = "abc"
    transcriber._segmenter._buffer.extend(b"\x00\x00" * 15_200)
    transcriber._segmenter._voice_ms = 950
    transcriber._segmenter._silence_ms = 0

    await transcriber._maybe_schedule_preview()

    assert transcriber._preview_task is None


async def test_cpu_preview_scheduler_can_emit_multiple_updates_for_same_utterance(monkeypatch) -> None:
    settings = Settings(
        speech_preview_min_duration_ms=700,
        speech_preview_interval_ms=400,
        transcription_device="cpu",
    )
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    transcriber._segmenter._speaking = True
    transcriber._segmenter._utterance_id = "abc"
    transcriber._segmenter._buffer.extend(b"\x00\x00" * 16_800)
    transcriber._segmenter._voice_ms = 1_050
    transcriber._segmenter._silence_ms = 0
    preview_calls: list[int] = []

    async def fake_run_preview(snapshot, _generation: int) -> None:
        preview_calls.append(snapshot.duration_ms)

    monkeypatch.setattr(transcriber, "_run_preview", fake_run_preview)

    await transcriber._maybe_schedule_preview()
    assert transcriber._preview_task is not None
    await transcriber._preview_task

    transcriber._segmenter._voice_ms = 1_550
    transcriber._segmenter._buffer.extend(b"\x00\x00" * 8_000)

    await transcriber._maybe_schedule_preview()
    assert transcriber._preview_task is not None
    await transcriber._preview_task

    assert len(preview_calls) == 2
    assert preview_calls[1] > preview_calls[0]


async def test_gpu_preview_scheduler_can_run_while_final_is_processing(monkeypatch) -> None:
    settings = Settings(
        speech_preview_min_duration_ms=700,
        speech_preview_interval_ms=400,
        transcription_device="cuda",
    )
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    transcriber._segmenter._speaking = True
    transcriber._segmenter._utterance_id = "abc"
    transcriber._segmenter._buffer.extend(b"\x00\x00" * 20_000)
    transcriber._segmenter._voice_ms = 1_200
    transcriber._segmenter._silence_ms = 0
    transcriber._active_finals = 1
    preview_calls: list[str] = []

    async def fake_run_preview(snapshot, _generation: int) -> None:
        preview_calls.append(snapshot.utterance_id)

    monkeypatch.setattr(transcriber, "_run_preview", fake_run_preview)

    await transcriber._maybe_schedule_preview()

    assert transcriber._preview_task is not None
    await transcriber._preview_task
    assert preview_calls == ["abc"]


async def test_preview_model_load_survives_waiter_cancellation() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    ready = asyncio.Event()
    release = asyncio.Event()
    marker = object()

    async def slow_preview_load() -> object:
        ready.set()
        await release.wait()
        return marker

    transcriber._preview_model_task = asyncio.create_task(slow_preview_load())

    waiter = asyncio.create_task(transcriber._ensure_preview_model())
    await ready.wait()
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter

    assert transcriber._preview_model_task is not None
    assert transcriber._preview_model_task.cancelled() is False

    release.set()
    resolved = await transcriber._ensure_preview_model()
    assert resolved is marker


async def test_start_preloads_preview_model_task() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )

    async def blocking_worker() -> None:
        await asyncio.sleep(9_999)

    def fake_to_thread(callback, *args, **kwargs):
        if callback == transcriber._load_model:
            return blocking_worker()
        if callback == transcriber._load_preview_model:
            return blocking_worker()
        raise AssertionError("Unexpected background task")

    original_to_thread = asyncio.to_thread
    asyncio.to_thread = fake_to_thread  # type: ignore[assignment]
    try:
        await transcriber.start()
        assert transcriber._model_task is not None
        assert transcriber._preview_model_task is not None
    finally:
        asyncio.to_thread = original_to_thread  # type: ignore[assignment]
        if transcriber._model_task is not None:
            transcriber._model_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await transcriber._model_task
        if transcriber._preview_model_task is not None:
            transcriber._preview_model_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await transcriber._preview_model_task
        if transcriber._worker_task is not None:
            transcriber._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await transcriber._worker_task


def test_cuda_library_error_triggers_cpu_fallback_detection() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )

    assert transcriber._should_fallback_to_cpu(RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"))


def test_runtime_info_prefers_loaded_device_over_resolved_device() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(transcription_device="auto"),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    transcriber._resolved_device = "cuda"
    transcriber._resolved_compute_type = "int8_float16"
    transcriber._final_loaded_device = "cpu"
    transcriber._final_loaded_compute_type = "int8"
    transcriber._preview_loaded_device = "cpu"
    transcriber._preview_loaded_compute_type = "int8"

    runtime = transcriber.runtime_info()

    assert runtime["device"] == "cpu"
    assert runtime["compute_type"] == "int8"
    assert runtime["resolved_device"] == "cuda"
    assert runtime["resolved_compute_type"] == "int8_float16"


async def test_wait_until_ready_awaits_final_and_preview_models(monkeypatch) -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(speech_vad_backend="energy"),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    calls: list[str] = []

    async def fake_ensure_model() -> object:
        calls.append("final")
        return object()

    async def fake_ensure_preview_model() -> object:
        calls.append("preview")
        return object()

    monkeypatch.setattr(transcriber, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(transcriber, "_ensure_preview_model", fake_ensure_preview_model)

    runtime = await transcriber.wait_until_ready()

    assert runtime is not None
    assert sorted(calls) == ["final", "preview"]


def test_build_model_for_name_raises_when_cuda_required(monkeypatch) -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(
            transcription_device="cuda",
            transcription_require_cuda=True,
            speech_vad_backend="energy",
        ),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )

    class _FakeWhisperModel:
        def __init__(self, _model_name: str, *, device: str, **_kwargs) -> None:
            if device == "cuda":
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_FakeWhisperModel))

    with pytest.raises(RuntimeError, match="CUDA is required"):
        transcriber._build_model_for_name("small.en", is_preview=False)


async def test_push_audio_cancels_inflight_preview_when_final_segment_ready(monkeypatch) -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    preview_task = asyncio.create_task(asyncio.sleep(9_999))
    transcriber._preview_task = preview_task

    def fake_segmenter_push(_data: bytes) -> list[ReadyAudioSegment]:
        return [
            ReadyAudioSegment(
                utterance_id="abc",
                pcm16=b"",
                duration_ms=1_900,
                voice_duration_ms=1_640,
                started_monotonic=10.0,
                ready_monotonic=11.7,
            )
        ]

    async def noop_preview() -> None:
        return None

    async def blocking_worker() -> None:
        await asyncio.sleep(9_999)

    monkeypatch.setattr(transcriber._segmenter, "push", fake_segmenter_push)
    monkeypatch.setattr(transcriber, "_maybe_schedule_preview", noop_preview)
    transcriber._worker_task = asyncio.create_task(blocking_worker())

    await transcriber.push_audio(b"\x00\x00" * 160, transcriber._settings.speech_target_sample_rate)
    await asyncio.sleep(0)

    assert preview_task.cancelled() is True

    transcriber._worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await transcriber._worker_task


def test_build_metrics_breaks_out_total_and_processing_latency() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )

    metrics = transcriber._build_metrics(
        ReadyAudioSegment(
            utterance_id="abc",
            pcm16=b"",
            duration_ms=1_900,
            voice_duration_ms=1_640,
            started_monotonic=10.0,
            ready_monotonic=11.4,
        ),
        emitted_monotonic=11.68,
    )

    assert metrics.total_latency_ms == 1679
    assert metrics.speech_capture_ms == 1400
    assert metrics.processing_ms == 279
    assert metrics.audio_duration_ms == 1900
    assert metrics.voice_duration_ms == 1640


def test_load_model_runs_warmup_when_enabled(monkeypatch) -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(transcription_warmup_enabled=True),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    built_model = object()
    calls: list[object] = []

    def fake_build_model() -> object:
        calls.append("build")
        return built_model

    def fake_warm_model(model: object, *, beam_size: int = 1) -> None:
        calls.append((model, beam_size))

    monkeypatch.setattr(transcriber, "_build_model", fake_build_model)
    monkeypatch.setattr(transcriber, "_warm_model", fake_warm_model)

    model = transcriber._load_model()

    assert model is built_model
    assert calls == ["build", (built_model, 2)]


def test_load_preview_model_runs_preview_warmup_when_enabled(monkeypatch) -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(transcription_warmup_enabled=True, transcription_preview_beam_size=1),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    built_model = object()
    calls: list[object] = []

    def fake_build_preview_model() -> object:
        calls.append("build-preview")
        return built_model

    def fake_warm_model(model: object, *, beam_size: int = 1) -> None:
        calls.append((model, beam_size))

    monkeypatch.setattr(transcriber, "_build_preview_model", fake_build_preview_model)
    monkeypatch.setattr(transcriber, "_warm_model", fake_warm_model)

    model = transcriber._load_preview_model()

    assert model is built_model
    assert calls == ["build-preview", (built_model, 1)]


def test_transcription_profiles_use_different_beam_sizes() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(transcription_preview_beam_size=1, transcription_final_beam_size=3),
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )
    beam_sizes: list[int] = []

    class _FakeSegment:
        text = "long here"
        avg_logprob = -0.2

    class _FakeModel:
        def transcribe(self, _audio, **kwargs):
            beam_sizes.append(int(kwargs["beam_size"]))
            return [_FakeSegment()], None

    model = _FakeModel()

    preview_text, _ = transcriber._transcribe_preview_segment(model, b"\x00\x00" * 20)
    final_text, _ = transcriber._transcribe_final_segment(model, b"\x00\x00" * 20)

    assert preview_text == "long here"
    assert final_text == "long here"
    assert beam_sizes == [1, 3]


def test_transcribe_with_profile_uses_guarded_decoder_options() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="Trading livestream.",
        on_segment=_noop,
    )
    calls: list[dict[str, object]] = []

    class _FakeModel:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(text="holding vwap here", avg_logprob=-0.2)], None

    text, confidence = transcriber._transcribe_final_segment(_FakeModel(), b"\x00\x00" * 160)

    assert text == "holding vwap here"
    assert confidence > 0.0
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["repetition_penalty"] == 1.05
    assert calls[0]["no_repeat_ngram_size"] == 3
    assert calls[0]["compression_ratio_threshold"] == 2.2
    assert calls[0]["log_prob_threshold"] == -1.0
    assert calls[0]["no_speech_threshold"] == 0.6
    assert calls[0]["without_timestamps"] is True


def test_preview_transcription_uses_preview_profile_without_final_guardrails() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="Trading livestream.",
        on_segment=_noop,
    )
    calls: list[dict[str, object]] = []

    class _FakeModel:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(text="watching vwap here", avg_logprob=-0.2)], None

    text, confidence = transcriber._transcribe_preview_segment(_FakeModel(), b"\x00\x00" * 160)

    assert text == "watching vwap here"
    assert confidence > 0.0
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["repetition_penalty"] == 1.0
    assert calls[0]["no_repeat_ngram_size"] == 0
    assert "compression_ratio_threshold" not in calls[0]
    assert "log_prob_threshold" not in calls[0]
    assert "no_speech_threshold" not in calls[0]


def test_transcribe_with_profile_retries_pathological_repetition() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="Trading livestream.",
        on_segment=_noop,
    )
    calls: list[dict[str, object]] = []
    responses = iter(
        (
            [SimpleNamespace(text="we, we, we, we, we, we, we, we, we, we", avg_logprob=-0.1)],
            [SimpleNamespace(text="we are holding vwap here", avg_logprob=-0.2)],
        )
    )

    class _FakeModel:
        def transcribe(self, _audio, **kwargs):
            calls.append(kwargs)
            return next(responses), None

    text, confidence = transcriber._transcribe_final_segment(_FakeModel(), b"\x00\x00" * 160)

    assert text == "we are holding vwap here"
    assert confidence > 0.0
    assert len(calls) == 2
    assert calls[0]["repetition_penalty"] == 1.05
    assert calls[1]["repetition_penalty"] == 1.12
    assert calls[0]["no_repeat_ngram_size"] == 3
    assert calls[1]["no_repeat_ngram_size"] == 2
    assert calls[1]["temperature"] == 0.0


def test_transcribe_with_profile_drops_persistent_repetition_loops() -> None:
    transcriber = LocalWhisperTranscriber(
        settings=Settings(),
        session_id="session",
        model_name="small.en",
        prompt="Trading livestream.",
        on_segment=_noop,
    )
    responses = iter(
        (
            [SimpleNamespace(text="because they can track because they can track because they can track", avg_logprob=-0.2)],
            [SimpleNamespace(text="all right all right all right all right all right all right", avg_logprob=-0.2)],
        )
    )

    class _FakeModel:
        def transcribe(self, _audio, **_kwargs):
            return next(responses), None

    text, confidence = transcriber._transcribe_final_segment(_FakeModel(), b"\x00\x00" * 160)

    assert text == ""
    assert confidence == 0.0


async def test_push_audio_evicts_oldest_segment_when_queue_full(monkeypatch) -> None:
    """push_audio drops the oldest queued segment to admit the newest when queue is full."""
    settings = Settings(transcription_device="cpu", transcription_warmup_enabled=False)
    transcriber = LocalWhisperTranscriber(
        settings=settings,
        session_id="session",
        model_name="small.en",
        prompt="",
        on_segment=_noop,
    )

    # Stub the segmenter to emit exactly one named segment per push call.
    push_index = 0

    def fake_segmenter_push(data: bytes) -> list[ReadyAudioSegment]:
        nonlocal push_index
        push_index += 1
        return [ReadyAudioSegment(
            utterance_id=f"seg_{push_index}",
            pcm16=b"",
            duration_ms=1_000,
            voice_duration_ms=1_000,
            started_monotonic=0.0,
            ready_monotonic=1.0,
        )]

    monkeypatch.setattr(transcriber._segmenter, "push", fake_segmenter_push)

    # Stub preview scheduling to avoid side effects.
    async def noop_preview() -> None:
        pass

    monkeypatch.setattr(transcriber, "_maybe_schedule_preview", noop_preview)

    # Install a blocking stub worker so push_audio won't try to load the model,
    # and so the queue is never drained during the test.
    async def blocking_worker() -> None:
        await asyncio.sleep(9_999)

    transcriber._worker_task = asyncio.create_task(blocking_worker())

    maxsize = transcriber._queue.maxsize
    assert maxsize > 0, "Queue must have a finite maxsize"

    # Push maxsize + 1 frames through push_audio — the (maxsize+1)-th call must evict seg_1.
    dummy_pcm = b"\x00\x00" * 160  # minimal valid PCM at target sample rate
    for _ in range(maxsize + 1):
        await transcriber.push_audio(dummy_pcm, settings.speech_target_sample_rate)

    assert transcriber._queue.qsize() == maxsize, "Queue must not grow beyond maxsize"

    segments = []
    while not transcriber._queue.empty():
        segments.append(transcriber._queue.get_nowait())

    ids = [s.utterance_id for s in segments]
    assert "seg_1" not in ids, "Oldest segment should have been evicted"
    assert f"seg_{maxsize + 1}" in ids, "Newest segment must be present"

    transcriber._worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await transcriber._worker_task


async def _noop(_: object) -> None:
    return None
