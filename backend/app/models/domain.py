from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class ActionTag(str, Enum):
    no_action = "NO_ACTION"
    setup_long = "SETUP_LONG"
    setup_short = "SETUP_SHORT"
    enter_long = "ENTER_LONG"
    enter_short = "ENTER_SHORT"
    add = "ADD"
    trim = "TRIM"
    exit_all = "EXIT_ALL"
    move_stop = "MOVE_STOP"
    move_to_breakeven = "MOVE_TO_BREAKEVEN"
    target = "TARGET"
    cancel_setup = "CANCEL_SETUP"
    commentary = "COMMENTARY"


class TradeSide(str, Enum):
    long = "LONG"
    short = "SHORT"


class EventType(str, Enum):
    info = "INFO"
    warning = "WARNING"
    transcript = "TRANSCRIPT"
    intent = "INTENT"
    risk = "RISK"
    execution = "EXECUTION"
    market = "MARKET"
    system = "SYSTEM"


class ExecutionMode(str, Enum):
    auto = "auto"
    review = "review"


class ManualTradeAction(str, Enum):
    buy = "BUY"
    sell = "SELL"
    close = "CLOSE"


class SegmentStatus(str, Enum):
    partial = "partial"
    final = "final"


class ConfidenceMixin(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TranscriptionMetrics(BaseModel):
    total_latency_ms: int = Field(default=0, ge=0)
    speech_capture_ms: int = Field(default=0, ge=0)
    processing_ms: int = Field(default=0, ge=0)
    audio_duration_ms: int = Field(default=0, ge=0)
    voice_duration_ms: int = Field(default=0, ge=0)


class TranscriptSegment(ConfidenceMixin):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    text: str
    status: SegmentStatus = SegmentStatus.final
    source: str = "manual"
    item_id: str | None = None
    metrics: TranscriptionMetrics | None = None
    received_at: datetime = Field(default_factory=utc_now)


class MarketSnapshot(BaseModel):
    symbol: str = "NQ"
    last_price: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    received_at: datetime | None = None


class PositionState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    side: TradeSide
    quantity: int = Field(ge=1)
    average_price: float
    stop_price: float | None = None
    target_price: float | None = None
    opened_at: datetime = Field(default_factory=utc_now)
    realized_pnl: float = 0.0


class SessionConfig(BaseModel):
    source_name: str = "Flow Zone Trader"
    symbol: str = "NQ"
    execution_mode: ExecutionMode = ExecutionMode.auto
    enable_audio_capture: bool = True
    enable_ai_fallback: bool = False
    enable_partial_intent_detection: bool = True
    enable_early_preview_entries: bool = False
    auto_execute: bool = True
    default_contract_size: int = Field(default=1, ge=1, le=10)
    transcription_model: str = "distil-small.en"
    broker_account_override: str | None = None
    broker_symbol_override: str | None = None

    @field_validator("execution_mode", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        if value == "paper_auto":
            return ExecutionMode.auto
        return value


class TradeIntent(ConfidenceMixin):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    tag: ActionTag
    symbol: str = "NQ"
    side: TradeSide | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    quantity_hint: str | None = None
    evidence_text: str
    source_segment_id: str | None = None
    source_received_at: datetime | None = None
    source_latency_ms: int = Field(default=0, ge=0)
    guard_reason: str | None = None
    stale_after_ms: int = 5_000
    created_at: datetime = Field(default_factory=utc_now)


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    intent: TradeIntent


class ExecutionResult(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    session_id: str
    action: ActionTag
    approved: bool
    message: str
    market_price: float | None = None
    position: PositionState | None = None
    realized_pnl_change: float = 0.0
    executed_at: datetime = Field(default_factory=utc_now)


class TimelineEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    type: EventType
    title: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class StreamSession(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=utc_now)
    config: SessionConfig = Field(default_factory=SessionConfig)
    market: MarketSnapshot = Field(default_factory=MarketSnapshot)
    position: PositionState | None = None
    realized_pnl: float = 0.0
    transcripts: list[TranscriptSegment] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    latest_candidate_intent: TradeIntent | None = None
    last_intent: TradeIntent | None = None
    latest_partial_text: str = ""
    latest_partial_metrics: TranscriptionMetrics | None = None
    latest_final_metrics: TranscriptionMetrics | None = None


class SessionPatch(BaseModel):
    """Delta of session fields changed by a single event.

    Only fields explicitly passed to the constructor are serialized
    (use ``model_dump(exclude_unset=True)``). Absent fields mean "no change";
    a field present as ``null`` means "set this field to null".
    String fields (e.g. ``latest_partial_text``) are cleared by sending ``""``
    rather than ``null`` to match the non-nullable ``str`` type on ``StreamSession``.
    """

    model_config = ConfigDict(use_enum_values=True)

    latest_partial_text: str | None = Field(default=None)
    latest_partial_metrics: TranscriptionMetrics | None = Field(default=None)
    latest_final_metrics: TranscriptionMetrics | None = Field(default=None)
    latest_candidate_intent: TradeIntent | None = Field(default=None)
    last_intent: TradeIntent | None = Field(default=None)
    market: MarketSnapshot | None = Field(default=None)
    position: PositionState | None = Field(default=None)
    realized_pnl: float | None = Field(default=None)
    new_transcript: TranscriptSegment | None = Field(default=None)


class CreateSessionRequest(BaseModel):
    config: SessionConfig = Field(default_factory=SessionConfig)


class UpdateSessionConfigRequest(BaseModel):
    enable_partial_intent_detection: bool | None = None
    enable_ai_fallback: bool | None = None
    enable_early_preview_entries: bool | None = None
    transcription_model: str | None = None
    broker_account_override: str | None = None
    broker_symbol_override: str | None = None


class ManualTradeRequest(BaseModel):
    action: ManualTradeAction
    contract_size: int = Field(default=3, ge=1, le=10)
    account: str | None = None
    symbol: str | None = None


class TextSegmentRequest(ConfidenceMixin):
    text: str
    status: SegmentStatus = SegmentStatus.final
    source: str = "manual"
    item_id: str | None = None
