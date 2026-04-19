"""Streaming Whisper-based transcription with dual models (final + preview).

This is the main transcription engine.  It uses the faster-whisper library
to run OpenAI's Whisper model locally.  The pipeline works as follows:

1. Raw audio arrives via push_audio() and is fed into a speech segmenter.
2. When the segmenter detects a completed utterance, it goes into a queue.
3. A background worker picks up queued segments and runs "final" transcription
   with high accuracy settings (larger beam, repetition penalty, etc.).
4. While the user is still speaking, a separate "preview" path periodically
   snapshots the in-progress audio and transcribes it with a faster, smaller
   model to show live partial results.

Why two models?  The final model prioritizes accuracy (larger beam search,
n-gram blocking).  The preview model prioritizes speed (greedy decoding)
so the user sees text appearing in near real-time.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import site
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.core.audio import resample_pcm16_mono
from app.core.config import Settings
from app.models.domain import SegmentStatus, TranscriptSegment, TranscriptionMetrics
from app.services.interpretation.transcript_normalizer import apply_trading_asr_corrections
from app.services.transcription.base import BaseTranscriber, SegmentHandler
from app.services.transcription.segmenter import ReadyAudioSegment, build_speech_segmenter
from app.services.transcription.streaming_preview import StreamingPreviewAssembler

_QUEUE_MAX_SEGMENTS = 20

# -- Decode parameters for preview vs. final vs. retry profiles --
_PREVIEW_TEMPERATURE = 0.0
_FINAL_TEMPERATURE = 0.0
_PREVIEW_REPETITION_PENALTY = 1.0          # no penalty -- speed over quality
_FINAL_REPETITION_PENALTY = 1.05           # mild penalty to reduce loops
_FINAL_RETRY_REPETITION_PENALTY = 1.12     # stronger penalty on retry
_PREVIEW_NO_REPEAT_NGRAM_SIZE = 0          # disabled for preview speed
_FINAL_NO_REPEAT_NGRAM_SIZE = 3            # block repeated 3-grams
_FINAL_RETRY_NO_REPEAT_NGRAM_SIZE = 2      # more aggressive on retry
_COMPRESSION_RATIO_THRESHOLD = 2.2
_LOG_PROB_THRESHOLD = -1.0
_NO_SPEECH_THRESHOLD = 0.6
# Degenerate detection: catch Whisper hallucination loops.
_DEGENERATE_MIN_TOKENS = 10
_TRANSCRIPT_TOKEN_RE = re.compile(r"[a-z0-9']+")


class LocalWhisperTranscriber(BaseTranscriber):
    """Runs Whisper locally with a dual-model architecture (final + preview).

    Audio flows:  push_audio -> segmenter -> queue -> _worker (final)
                                          |-> _run_preview (preview, concurrent)

    The final path produces high-accuracy transcripts after the speaker
    stops.  The preview path gives fast partial results while the speaker
    is still talking.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session_id: str,
        model_name: str,
        prompt: str,
        on_segment: SegmentHandler,
    ) -> None:
        self._settings = settings
        self._session_id = session_id
        self._model_name = model_name
        self._preview_model_name = (settings.transcription_preview_model or "").strip() or model_name
        self._engine = settings.transcription_engine
        self._prompt = prompt
        self._on_segment = on_segment
        # The segmenter splits continuous audio into discrete utterances.
        self._segmenter = build_speech_segmenter(
            sample_rate=settings.speech_target_sample_rate,
            vad_backend=settings.speech_vad_backend,
            vad_aggressiveness=settings.speech_vad_aggressiveness,
            vad_frame_ms=settings.speech_vad_frame_ms,
            vad_preroll_ms=settings.speech_vad_preroll_ms,
            vad_start_ms=settings.speech_vad_start_ms,
            vad_start_window_ms=settings.speech_vad_start_window_ms,
            energy_threshold=settings.speech_energy_threshold,
            min_duration_ms=settings.speech_min_duration_ms,
            silence_duration_ms=settings.speech_silence_duration_ms,
            max_duration_ms=settings.speech_max_duration_ms,
        )
        self._segmenter_backend = getattr(self._segmenter, "backend_name", settings.speech_vad_backend)
        # Preview stabilizer reduces flicker in the live partial text.
        self._streaming_preview = (
            StreamingPreviewAssembler(
                context_words=settings.transcription_preview_context_words,
                stability_margin_words=settings.transcription_preview_stability_margin_words,
            )
            if self._engine == "streaming"
            else None
        )
        # Queue of completed utterances waiting for final transcription.
        # None sentinel signals the worker to shut down.
        self._queue: asyncio.Queue[ReadyAudioSegment | None] = asyncio.Queue(maxsize=_QUEUE_MAX_SEGMENTS)
        self._worker_task: asyncio.Task[None] | None = None
        self._model_task: asyncio.Task[Any] | None = None
        self._preview_model_task: asyncio.Task[Any] | None = None
        self._preview_task: asyncio.Task[None] | None = None
        self._model: Any = None
        self._preview_model: Any = None
        self._last_error: str | None = None
        self._last_preview_utterance_id: str | None = None
        self._last_preview_duration_ms: int = 0
        # Locks prevent the final and preview models from running concurrently
        # on CPU, where parallel inference would thrash the cache.
        self._final_transcribe_lock = asyncio.Lock()
        self._preview_transcribe_lock = asyncio.Lock()
        self._resolved_device: str | None = None
        self._resolved_compute_type: str | None = None
        self._final_loaded_device: str | None = None
        self._final_loaded_compute_type: str | None = None
        self._preview_loaded_device: str | None = None
        self._preview_loaded_compute_type: str | None = None
        self._active_finals = 0        # how many final transcriptions are in progress
        self._preview_generation = 0   # incremented to invalidate stale previews

    async def start(self) -> None:
        """Launch the background worker and begin loading both models in parallel."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker())
        # Load final and preview models concurrently on background threads.
        self._model_task = asyncio.create_task(asyncio.to_thread(self._load_model))
        self._preview_model_task = asyncio.create_task(asyncio.to_thread(self._load_preview_model))

    async def wait_until_ready(self) -> dict[str, Any] | None:
        await asyncio.gather(self._ensure_model(), self._ensure_preview_model())
        return self.runtime_info()

    async def push_audio(self, data: bytes, sample_rate: int) -> None:
        """Resample incoming audio, feed it to the segmenter, and manage previews."""
        if self._last_error is not None:
            raise RuntimeError(self._last_error)
        if self._worker_task is None:
            await self.start()
        pcm16 = resample_pcm16_mono(data, source_rate=sample_rate, target_rate=self._settings.speech_target_sample_rate)
        ready_segments = self._segmenter.push(pcm16)
        # A completed utterance means the preview for it is now obsolete.
        if ready_segments:
            self._cancel_preview()
        for segment in ready_segments:
            self._reset_preview_state()
            self._last_preview_utterance_id = None
            self._last_preview_duration_ms = 0
            # Drop oldest segment if queue is full to prevent backpressure.
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(segment)
        await self._maybe_schedule_preview()

    async def close(self) -> None:
        self._cancel_preview()
        self._reset_preview_state()
        for segment in self._segmenter.flush():
            await self._queue.put(segment)
        if self._worker_task is not None:
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None
        if self._preview_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._preview_task
            self._preview_task = None

    async def _worker(self) -> None:
        """Background loop: pull completed utterances from the queue and transcribe them."""
        while True:
            item = await self._queue.get()
            if item is None:  # shutdown sentinel
                return

            # Cancel any in-flight preview -- we are about to produce the final text.
            self._cancel_preview()
            self._active_finals += 1
            try:
                model = await self._ensure_model()
                async with self._final_transcribe_lock:
                    text, confidence = await asyncio.to_thread(self._transcribe_final_segment, model, item.pcm16)
                if not text:
                    continue

                await self._on_segment(
                    TranscriptSegment(
                        session_id=self._session_id,
                        text=text,
                        status=SegmentStatus.final,
                        source="local_whisper",
                        item_id=item.utterance_id,
                        confidence=confidence,
                        metrics=self._build_metrics(item, emitted_monotonic=time.monotonic()),
                    )
                )
            except Exception as error:
                self._last_error = str(error)
                return
            finally:
                self._active_finals = max(0, self._active_finals - 1)

    async def _maybe_schedule_preview(self) -> None:
        if self._preview_requires_quiet_backend() and (self._queue.qsize() > 0 or self._active_finals > 0):
            return
        if self._preview_task is not None and not self._preview_task.done():
            return

        snapshot = self._segmenter.snapshot()
        if snapshot is None:
            return

        if self._should_skip_preview(snapshot):
            return

        if snapshot.duration_ms < self._settings.speech_preview_min_duration_ms:
            return

        if snapshot.utterance_id != self._last_preview_utterance_id:
            self._last_preview_utterance_id = snapshot.utterance_id
            self._last_preview_duration_ms = 0

        if snapshot.duration_ms - self._last_preview_duration_ms < self._settings.speech_preview_interval_ms:
            return

        self._last_preview_duration_ms = snapshot.duration_ms
        generation = self._preview_generation
        self._preview_task = asyncio.create_task(self._run_preview(snapshot, generation))

    async def _run_preview(self, snapshot: ReadyAudioSegment, generation: int) -> None:
        """Transcribe a snapshot of the in-progress utterance for live display.

        Checks _should_abort_preview at multiple points so it can bail out
        early if the utterance was finalized or a newer preview was scheduled.
        """
        try:
            if self._should_abort_preview(snapshot, generation):
                return
            model = await self._ensure_preview_model()
            if self._should_abort_preview(snapshot, generation):
                return
            async with self._preview_transcribe_lock:
                if self._should_abort_preview(snapshot, generation):
                    return
                text, confidence = await asyncio.to_thread(self._transcribe_preview_snapshot, model, snapshot)
            if self._should_abort_preview(snapshot, generation) or not text:
                return
            await self._on_segment(
                TranscriptSegment(
                    session_id=self._session_id,
                    text=text,
                    status=SegmentStatus.partial,
                    source="local_whisper_preview",
                    item_id=snapshot.utterance_id,
                    confidence=confidence,
                    metrics=self._build_metrics(snapshot, emitted_monotonic=time.monotonic()),
                )
            )
        except asyncio.CancelledError:
            return
        except Exception as error:
            self._last_error = str(error)
        finally:
            if self._preview_task is asyncio.current_task():
                self._preview_task = None

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        task = self._model_task
        if task is None or task.cancelled() or (task.done() and task.exception() is not None):
            task = asyncio.create_task(asyncio.to_thread(self._load_model))
            self._model_task = task
        self._model = await asyncio.shield(task)
        return self._model

    async def _ensure_preview_model(self) -> Any:
        if self._preview_model is not None:
            return self._preview_model
        task = self._preview_model_task
        if task is None or task.cancelled() or (task.done() and task.exception() is not None):
            task = asyncio.create_task(asyncio.to_thread(self._load_preview_model))
            self._preview_model_task = task
        self._preview_model = await asyncio.shield(task)
        return self._preview_model

    def _load_model(self) -> Any:
        model = self._build_model()
        if self._settings.transcription_warmup_enabled:
            self._warm_model(model, beam_size=max(1, int(self._settings.transcription_final_beam_size)))
        return model

    def _load_preview_model(self) -> Any:
        model = self._build_preview_model()
        if self._settings.transcription_warmup_enabled:
            self._warm_model(model, beam_size=max(1, int(self._settings.transcription_preview_beam_size)))
        return model

    def _build_model(self) -> Any:
        model = self._build_model_for_name(self._model_name, is_preview=False)
        return model

    def _build_preview_model(self) -> Any:
        model = self._build_model_for_name(self._preview_model_name, is_preview=True)
        return model

    def _build_model_for_name(self, model_name: str, *, is_preview: bool) -> Any:
        from faster_whisper import WhisperModel

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        profile = "preview" if is_preview else "final"
        try:
            model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=self._settings.transcription_cpu_threads,
            )
            self._record_loaded_runtime(device=device, compute_type=compute_type, is_preview=is_preview)
            return model
        except Exception as error:
            if device != "cuda" or not self._should_fallback_to_cpu(error):
                raise
            if self._settings.transcription_require_cuda:
                raise RuntimeError(
                    f"CUDA is required for the {profile} transcription model '{model_name}', "
                    f"but loading on CUDA failed: {error}"
                ) from error

            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=self._settings.transcription_cpu_threads,
            )
            self._record_loaded_runtime(device="cpu", compute_type="int8", is_preview=is_preview)
            return model

    def _warm_model(self, model: Any, *, beam_size: int = 1) -> None:
        """Run a short silent audio through the model to pre-warm caches/JIT."""
        import numpy as np

        warmup_audio = np.zeros(int(self._settings.speech_target_sample_rate * 0.35), dtype=np.float32)
        self._decode_segments(
            model,
            warmup_audio,
            beam_size=max(1, int(beam_size)),
            initial_prompt=self._prompt,
            decode_profile="warmup",
        )

    def _resolve_device(self) -> str:
        if self._resolved_device is not None:
            return self._resolved_device
        if self._settings.transcription_device != "auto":
            self._resolved_device = self._settings.transcription_device
            return self._resolved_device
        if not shutil.which("nvidia-smi"):
            self._resolved_device = "cpu"
            return self._resolved_device
        self._resolved_device = "cuda" if self._cuda_runtime_looks_available() else "cpu"
        return self._resolved_device

    def _resolve_compute_type(self, device: str) -> str:
        if self._resolved_compute_type is not None and self._resolved_device == device:
            return self._resolved_compute_type
        if self._settings.transcription_compute_type != "auto":
            self._resolved_compute_type = self._settings.transcription_compute_type
            return self._resolved_compute_type
        self._resolved_compute_type = "int8_float16" if device == "cuda" else "int8"
        return self._resolved_compute_type

    def _cuda_runtime_looks_available(self) -> bool:
        found_cublas = False
        found_cudnn = False

        for directory in self._candidate_cuda_library_dirs():
            if not directory.exists():
                continue
            if not found_cublas and any(directory.glob("libcublas.so.12*")):
                found_cublas = True
            if not found_cudnn and any(directory.glob("libcudnn.so*")):
                found_cudnn = True
            if found_cublas and found_cudnn:
                return True

        return found_cublas and found_cudnn

    def runtime_info(self) -> dict[str, str | bool | int]:
        resolved_device = self._resolve_device()
        return {
            "backend": "local_whisper",
            "engine": self._engine,
            "configured_device": self._settings.transcription_device,
            "device": self._final_loaded_device or resolved_device,
            "compute_type": self._final_loaded_compute_type or self._resolve_compute_type(resolved_device),
            "resolved_device": resolved_device,
            "resolved_compute_type": self._resolve_compute_type(resolved_device),
            "preview_device": self._preview_loaded_device or resolved_device,
            "preview_compute_type": self._preview_loaded_compute_type or self._resolve_compute_type(resolved_device),
            "model": self._model_name,
            "preview_model": self._preview_model_name,
            "segmenter_backend": self._segmenter_backend,
            "segmenter_requested_backend": self._settings.speech_vad_backend,
            "preview_model_distinct": self._preview_model_name != self._model_name,
            "preview_tail_ms": self._settings.transcription_preview_tail_ms,
            "final_beam_size": max(1, int(self._settings.transcription_final_beam_size)),
            "preview_beam_size": max(1, int(self._settings.transcription_preview_beam_size)),
            "cuda_runtime_available": self._cuda_runtime_looks_available(),
        }

    def _record_loaded_runtime(self, *, device: str, compute_type: str, is_preview: bool) -> None:
        if is_preview:
            self._preview_loaded_device = device
            self._preview_loaded_compute_type = compute_type
            return

        self._final_loaded_device = device
        self._final_loaded_compute_type = compute_type

    def _candidate_cuda_library_dirs(self) -> list[Path]:
        candidates: list[Path] = [
            Path("/usr/lib/wsl/lib"),
            Path("/usr/local/cuda/lib64"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]

        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        for entry in ld_library_path.split(":"):
            if entry:
                candidates.append(Path(entry))

        try:
            for base in site.getsitepackages():
                candidates.append(Path(base) / "nvidia" / "cublas" / "lib")
                candidates.append(Path(base) / "nvidia" / "cudnn" / "lib")
        except Exception:
            pass

        try:
            user_site = site.getusersitepackages()
            if user_site:
                candidates.append(Path(user_site) / "nvidia" / "cublas" / "lib")
                candidates.append(Path(user_site) / "nvidia" / "cudnn" / "lib")
        except Exception:
            pass

        deduped: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped

    def _should_fallback_to_cpu(self, error: Exception) -> bool:
        message = str(error).lower()
        return any(
            marker in message
            for marker in (
                "libcublas",
                "libcuda",
                "cuda",
                "cudnn",
                "cannot be loaded",
                "not found",
            )
        )

    def _cancel_preview(self) -> None:
        """Invalidate in-flight previews by bumping the generation counter."""
        self._preview_generation += 1
        preview_task = self._preview_task
        if preview_task is not None and not preview_task.done():
            preview_task.cancel()

    def _reset_preview_state(self) -> None:
        if self._streaming_preview is not None:
            self._streaming_preview.reset()

    def _should_abort_preview(self, snapshot: ReadyAudioSegment, generation: int) -> bool:
        """True if this preview is stale (utterance ended or a new preview was scheduled)."""
        if generation != self._preview_generation:
            return True
        # On CPU, don't preview while final transcription is running.
        if self._preview_requires_quiet_backend() and (self._queue.qsize() > 0 or self._active_finals > 0):
            return True
        current_snapshot = self._segmenter.snapshot()
        if current_snapshot is None:
            return True
        return current_snapshot.utterance_id != snapshot.utterance_id

    def _should_skip_preview(self, snapshot: ReadyAudioSegment) -> bool:
        """On CPU, skip short previews to avoid slowing down final transcription."""
        if self._resolve_device() != "cpu":
            return False

        if snapshot.duration_ms < max(self._settings.speech_preview_min_duration_ms, 1_000):
            return True

        return False

    def _preview_requires_quiet_backend(self) -> bool:
        """CPU cannot run preview and final in parallel without thrashing."""
        return self._resolve_device() == "cpu"

    def _build_metrics(self, item: ReadyAudioSegment, *, emitted_monotonic: float) -> TranscriptionMetrics:
        speech_capture_ms = max(0, int((item.ready_monotonic - item.started_monotonic) * 1000))
        processing_ms = max(0, int((emitted_monotonic - item.ready_monotonic) * 1000))
        total_latency_ms = max(0, int((emitted_monotonic - item.started_monotonic) * 1000))
        return TranscriptionMetrics(
            total_latency_ms=total_latency_ms,
            speech_capture_ms=speech_capture_ms,
            processing_ms=processing_ms,
            audio_duration_ms=max(0, item.duration_ms),
            voice_duration_ms=max(0, item.voice_duration_ms),
        )

    def _transcribe_final_segment(self, model: Any, pcm16: bytes) -> tuple[str, float]:
        """Run final (high-accuracy) transcription on a completed utterance."""
        return self._transcribe_with_profile(
            model,
            pcm16,
            beam_size=max(1, int(self._settings.transcription_final_beam_size)),
            decode_profile="final",
        )

    def _transcribe_preview_snapshot(self, model: Any, snapshot: ReadyAudioSegment) -> tuple[str, float]:
        """Run fast preview transcription and stabilize the output text.

        If streaming preview is enabled, only the tail of the audio is
        transcribed (since the beginning was already committed), and the
        result is merged with committed words to reduce flicker.
        """
        if self._streaming_preview is None:
            return self._transcribe_with_profile(
                model,
                snapshot.pcm16,
                beam_size=max(1, int(self._settings.transcription_preview_beam_size)),
                decode_profile="preview",
            )

        preview_pcm16 = self._slice_preview_audio(snapshot.pcm16)
        preview_prompt = self._streaming_preview.build_prompt(
            utterance_id=snapshot.utterance_id,
            base_prompt=self._prompt,
        )
        preview_text, confidence = self._transcribe_with_profile(
            model,
            preview_pcm16,
            beam_size=max(1, int(self._settings.transcription_preview_beam_size)),
            initial_prompt=preview_prompt,
            decode_profile="preview",
        )
        stabilized = self._streaming_preview.stabilize(
            utterance_id=snapshot.utterance_id,
            tail_text=preview_text,
        )
        if stabilized:
            return stabilized, confidence
        return self._streaming_preview.committed_text(utterance_id=snapshot.utterance_id), confidence

    def _transcribe_preview_segment(self, model: Any, pcm16: bytes) -> tuple[str, float]:
        return self._transcribe_with_profile(
            model,
            pcm16,
            beam_size=max(1, int(self._settings.transcription_preview_beam_size)),
            decode_profile="preview",
        )

    def _slice_preview_audio(self, pcm16: bytes) -> bytes:
        """Keep only the last N ms of audio for preview transcription.

        Transcribing just the tail is faster and sufficient because earlier
        audio has already been committed by the preview stabilizer.
        """
        tail_ms = max(0, int(self._settings.transcription_preview_tail_ms))
        if tail_ms == 0:
            return pcm16

        tail_samples = int((self._settings.speech_target_sample_rate * tail_ms) / 1000)
        tail_bytes = tail_samples * 2  # 2 bytes per 16-bit sample
        if len(pcm16) <= tail_bytes:
            return pcm16
        return pcm16[-tail_bytes:]

    def _transcribe_with_profile(
        self,
        model: Any,
        pcm16: bytes,
        *,
        beam_size: int,
        initial_prompt: str | None = None,
        decode_profile: str = "final",
    ) -> tuple[str, float]:
        """Core transcription: decode audio, collect text, and handle retries.

        For final transcriptions, if the result looks degenerate (Whisper
        hallucination loop), it retries with stronger repetition penalties.
        If the retry is also degenerate, the segment is discarded.
        """
        import numpy as np

        # Convert 16-bit PCM to float32 in [-1, 1] as Whisper expects.
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        prompt = self._prompt if initial_prompt is None else initial_prompt
        segments = self._decode_segments(
            model,
            audio,
            beam_size=beam_size,
            initial_prompt=prompt,
            decode_profile=decode_profile,
        )
        text, confidence = self._collect_transcription_result(segments)
        if decode_profile != "final":
            return text, confidence

        if not text:
            return "", 0.0
        if not self._is_degenerate_transcript(text):
            return text, confidence

        # Retry with stronger repetition penalty to break the hallucination loop.
        retry_segments = self._decode_segments(
            model,
            audio,
            beam_size=beam_size,
            initial_prompt=prompt,
            decode_profile="final_retry",
        )
        retry_text, retry_confidence = self._collect_transcription_result(retry_segments)
        if retry_text and not self._is_degenerate_transcript(retry_text):
            return retry_text, retry_confidence
        return "", 0.0

    def _decode_segments(
        self,
        model: Any,
        audio: Any,
        *,
        beam_size: int,
        initial_prompt: str,
        decode_profile: str,
    ) -> list[Any]:
        """Call faster-whisper's transcribe() with profile-specific parameters.

        Profiles: "preview" (fast, lenient), "final" (accurate),
        "final_retry" (aggressive anti-repetition), "warmup" (minimal).
        """
        kwargs: dict[str, Any] = {
            "language": self._settings.transcription_language,
            "beam_size": beam_size,
            "best_of": 1,
            "condition_on_previous_text": False,
            "initial_prompt": initial_prompt,
            "without_timestamps": True,
            "word_timestamps": False,
        }

        if decode_profile == "preview":
            kwargs["temperature"] = _PREVIEW_TEMPERATURE
            kwargs["repetition_penalty"] = _PREVIEW_REPETITION_PENALTY
            kwargs["no_repeat_ngram_size"] = _PREVIEW_NO_REPEAT_NGRAM_SIZE
        elif decode_profile == "final":
            kwargs["temperature"] = _FINAL_TEMPERATURE
            kwargs["repetition_penalty"] = _FINAL_REPETITION_PENALTY
            kwargs["no_repeat_ngram_size"] = _FINAL_NO_REPEAT_NGRAM_SIZE
            kwargs["compression_ratio_threshold"] = _COMPRESSION_RATIO_THRESHOLD
            kwargs["log_prob_threshold"] = _LOG_PROB_THRESHOLD
            kwargs["no_speech_threshold"] = _NO_SPEECH_THRESHOLD
        elif decode_profile == "final_retry":
            kwargs["temperature"] = _FINAL_TEMPERATURE
            kwargs["repetition_penalty"] = _FINAL_RETRY_REPETITION_PENALTY
            kwargs["no_repeat_ngram_size"] = _FINAL_RETRY_NO_REPEAT_NGRAM_SIZE
            kwargs["compression_ratio_threshold"] = _COMPRESSION_RATIO_THRESHOLD
            kwargs["log_prob_threshold"] = _LOG_PROB_THRESHOLD
            kwargs["no_speech_threshold"] = _NO_SPEECH_THRESHOLD
        else:
            kwargs["temperature"] = _FINAL_TEMPERATURE

        segments, _ = model.transcribe(audio, **kwargs)
        return list(segments)

    def _collect_transcription_result(self, segments: list[Any]) -> tuple[str, float]:
        """Join Whisper segments into a single string and compute a confidence score.

        Confidence is derived from Whisper's average log-probability,
        mapped to a [0.3, 0.99] range.  Domain-specific ASR corrections
        (e.g., trading terminology) are applied before returning.
        """
        text_parts: list[str] = []
        logprobs: list[float] = []
        for segment in segments:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)
            avg_logprob = getattr(segment, "avg_logprob", None)
            if isinstance(avg_logprob, float):
                logprobs.append(avg_logprob)

        text = " ".join(text_parts).strip()
        if not text:
            return "", 0.0

        text = apply_trading_asr_corrections(text)

        if not logprobs:
            return text, 0.85  # default confidence when logprobs unavailable

        # Map average log-prob to [0.3, 0.99] confidence range.
        average = sum(logprobs) / len(logprobs)
        confidence = max(0.3, min(0.99, 1.0 + (average / 5.0)))
        return text, round(confidence, 3)

    def _is_degenerate_transcript(self, text: str) -> bool:
        """Detect Whisper hallucination loops (e.g., "the the the the...").

        Two heuristics:
        1. Any single word makes up >= 55% of all tokens.
        2. Consecutive n-gram repetitions exceed thresholds
           (e.g., a single word repeated 5+ times in a row).
        Short transcripts (< 10 tokens) are never flagged.
        """
        tokens = _TRANSCRIPT_TOKEN_RE.findall(text.lower())
        if len(tokens) < _DEGENERATE_MIN_TOKENS:
            return False

        # Heuristic 1: dominant single word.
        most_common_count = Counter(tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) >= 0.55:
            return True

        # Heuristic 2: consecutive repeated n-grams of various sizes.
        repeat_thresholds = (
            (1, 5),   # single word repeated 5+ times
            (2, 4),   # bigram repeated 4+ times
            (3, 3),   # trigram repeated 3+ times
            (4, 3),   # 4-gram repeated 3+ times
        )
        return any(
            self._has_consecutive_repeated_ngram(tokens, ngram_size=ngram_size, min_repeats=min_repeats)
            for ngram_size, min_repeats in repeat_thresholds
        )

    def _has_consecutive_repeated_ngram(
        self,
        tokens: list[str],
        *,
        ngram_size: int,
        min_repeats: int,
    ) -> bool:
        max_start = len(tokens) - (ngram_size * min_repeats) + 1
        if max_start <= 0:
            return False

        for start in range(max_start):
            unit = tokens[start : start + ngram_size]
            repeats = 1
            cursor = start + ngram_size
            while cursor + ngram_size <= len(tokens) and tokens[cursor : cursor + ngram_size] == unit:
                repeats += 1
                if repeats >= min_repeats:
                    return True
                cursor += ngram_size
        return False
