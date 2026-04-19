from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "stream-copier"
    api_prefix: str = "/api"
    cors_origin: str = "http://localhost:4300"
    cors_origin_regex: str = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
    data_dir: Path = Path("data")
    max_transcript_segments: int = 250
    max_events: int = 500
    default_symbol: str = "NQ"

    transcription_backend: str = "local_whisper"
    transcription_model: str = "distil-large-v3"
    transcription_preview_model: str = "distil-small.en"
    transcription_engine: str = "streaming"
    transcription_device: str = "auto"
    transcription_compute_type: str = "auto"
    transcription_require_cuda: bool = True
    transcription_warn_on_cpu_fallback: bool = True
    transcription_cpu_threads: int = 4
    transcription_language: str = "en"
    transcription_warmup_enabled: bool = True
    transcription_preview_beam_size: int = 1
    transcription_final_beam_size: int = 2
    transcription_preview_tail_ms: int = 2_200
    transcription_preview_context_words: int = 28
    transcription_preview_stability_margin_words: int = 1

    speech_target_sample_rate: int = 16_000
    speech_vad_backend: str = "webrtc"
    speech_vad_aggressiveness: int = Field(default=2, ge=0, le=3)
    speech_vad_frame_ms: int = 20
    speech_vad_start_ms: int = 60
    speech_vad_start_window_ms: int = 80
    speech_vad_preroll_ms: int = 120
    speech_energy_threshold: float = 0.012
    speech_min_duration_ms: int = 300
    speech_silence_duration_ms: int = 300
    speech_max_duration_ms: int = 4_000
    speech_preview_min_duration_ms: int = 450
    speech_preview_interval_ms: int = 400

    interpreter_mode: str = "rule_first"
    enable_local_intent_classifier: bool = True
    local_intent_classifier_model: str = "answerdotai/ModernBERT-base"
    local_intent_classifier_artifact_dir: Path | None = None
    local_intent_classifier_device: str = "cpu"
    local_intent_classifier_max_length: int = 256
    local_intent_classifier_min_probability: float = 0.62
    local_intent_classifier_block_probability: float = 0.84
    local_intent_classifier_recovery_probability: float = 0.9
    candidate_window_ms: int = 6_000
    candidate_preroll_ms: int = 2_400
    candidate_max_fragments: int = 5
    candidate_open_probability: float = 0.43
    candidate_keep_probability: float = 0.28
    enable_gemini_fallback: bool = False
    enable_embedding_gate: bool = False
    embedding_gate_model: str = "BAAI/bge-small-en-v1.5"
    embedding_gate_threshold: float = 0.40
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.0-flash"
    gemini_confirmation_timeout_ms: int = 1_500

    min_confidence: float = 0.74
    max_entry_distance_points: float = 12.0
    max_entry_signal_age_ms: int = 20_000
    preview_entry_confirmation_window_ms: int = 12_000
    max_entry_chase_points: float = 8.0
    entry_context_window_ms: int = 15_000
    entry_guard_window_ms: int = 20_000
    force_wide_brackets: bool = True
    wide_stop_points: float = 120.0
    wide_target_points: float = 240.0
    stale_intent_ms: int = 5_000
    default_contract_size: int = 1
    max_contract_size: int = 4
    ninjatrader_bridge_url: str = "http://127.0.0.1:18080"
    ninjatrader_bridge_token: str | None = None
    ninjatrader_bridge_timeout_ms: int = 2_500
    ninjatrader_account: str | None = "Sim101"
    ninjatrader_symbol: str | None = None
    ninjatrader_time_in_force: str = "Day"

    audio_prompt: str = ""

    @model_validator(mode="after")
    def normalize_legacy_transcription_config(self) -> "Settings":
        legacy_openai_models = {
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "whisper-1",
            "gpt-realtime",
        }
        self.transcription_engine = self.transcription_engine.strip().lower()
        self.transcription_device = self.transcription_device.strip().lower()
        self.transcription_compute_type = self.transcription_compute_type.strip().lower()
        self.speech_vad_backend = self.speech_vad_backend.strip().lower()
        self.local_intent_classifier_device = self.local_intent_classifier_device.strip().lower()
        if self.transcription_backend == "local_whisper" and self.transcription_model in legacy_openai_models:
            self.transcription_model = "distil-small.en"
        if self.transcription_backend == "local_whisper" and self.transcription_preview_model in legacy_openai_models:
            self.transcription_preview_model = "distil-small.en"
        if self.transcription_engine not in {"segment", "streaming"}:
            raise ValueError("TRANSCRIPTION_ENGINE must be 'segment' or 'streaming'")
        if self.speech_vad_backend not in {"webrtc", "energy"}:
            raise ValueError("SPEECH_VAD_BACKEND must be 'webrtc' or 'energy'")
        if self.local_intent_classifier_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("LOCAL_INTENT_CLASSIFIER_DEVICE must be 'auto', 'cpu', or 'cuda'")
        if self.speech_vad_frame_ms not in {10, 20, 30}:
            raise ValueError("SPEECH_VAD_FRAME_MS must be one of 10, 20, or 30")
        if self.speech_vad_start_window_ms < self.speech_vad_start_ms:
            self.speech_vad_start_window_ms = self.speech_vad_start_ms
        self.transcription_preview_tail_ms = max(self.transcription_preview_tail_ms, self.speech_preview_min_duration_ms)
        self.transcription_preview_context_words = max(0, self.transcription_preview_context_words)
        self.transcription_preview_stability_margin_words = max(0, self.transcription_preview_stability_margin_words)
        self.local_intent_classifier_max_length = max(64, self.local_intent_classifier_max_length)
        self.local_intent_classifier_min_probability = max(0.0, min(1.0, self.local_intent_classifier_min_probability))
        self.local_intent_classifier_block_probability = max(
            0.0,
            min(1.0, self.local_intent_classifier_block_probability),
        )
        self.local_intent_classifier_recovery_probability = max(
            0.0,
            min(1.0, self.local_intent_classifier_recovery_probability),
        )
        self.candidate_window_ms = max(1_000, self.candidate_window_ms)
        self.candidate_preroll_ms = max(0, min(self.candidate_preroll_ms, self.candidate_window_ms))
        self.candidate_max_fragments = max(2, self.candidate_max_fragments)
        self.candidate_open_probability = max(0.0, min(1.0, self.candidate_open_probability))
        self.candidate_keep_probability = max(0.0, min(1.0, self.candidate_keep_probability))
        if self.local_intent_classifier_artifact_dir is None:
            self.local_intent_classifier_artifact_dir = self.data_dir
        return self

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def events_dir(self) -> Path:
        return self.data_dir / "events"

    @property
    def local_intent_classifier_dir(self) -> Path:
        if self.local_intent_classifier_artifact_dir is None:
            raise ValueError("local_intent_classifier_artifact_dir is not configured")
        return self.local_intent_classifier_artifact_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
