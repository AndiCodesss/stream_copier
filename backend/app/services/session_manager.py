from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import Settings
from app.models.domain import (
    ActionTag,
    CreateSessionRequest,
    EventType,
    ExecutionMode,
    ExecutionResult,
    MarketSnapshot,
    ManualTradeAction,
    ManualTradeRequest,
    PositionState,
    SegmentStatus,
    SessionPatch,
    StreamSession,
    TextSegmentRequest,
    TimelineEvent,
    TradeIntent,
    TradeSide,
    TranscriptSegment,
    UpdateSessionConfigRequest,
    utc_now,
)
from app.services.event_hub import EventHub
from app.services.execution.ninjatrader import NinjaTraderBridgeClient, NinjaTraderExecutor
from app.services.execution.risk import RiskEngine
from app.services.interpretation.embedding_gate import EmbeddingGate
from app.services.interpretation.gemini_fallback import GeminiFallbackInterpreter
from app.services.interpretation.local_classifier import ModernBertIntentClassifier
from app.services.interpretation.rule_engine import RuleBasedTradeInterpreter
from app.services.storage.event_store import EventLogStore
from app.services.storage.session_store import SessionStore
from app.services.transcription.base import BaseTranscriber
from app.services.transcription.mock import NoopTranscriber
from app.services.transcription.local_whisper import LocalWhisperTranscriber

_SESSION_SAVE_DEBOUNCE_S = 1.0
_BROKER_STATE_CACHE_TTL_S = 0.5
_ENTRY_ACTIONS = {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}
_LOGGER = logging.getLogger(__name__)


@dataclass
class PendingPreviewExecution:
    intent: TradeIntent
    executed_at: datetime


@dataclass
class CachedBrokerState:
    state: dict[str, Any]
    fetched_monotonic: float


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, StreamSession] = {}
        self._event_hub = EventHub()
        self._store = EventLogStore(settings.events_dir)
        self._session_store = SessionStore(settings.sessions_dir)
        self._sessions = {session.id: session for session in self._session_store.load_all()}
        fallback = GeminiFallbackInterpreter(settings) if settings.interpreter_mode != "rule_only" else None
        local_classifier = ModernBertIntentClassifier(settings)
        if local_classifier.is_available():
            self._classifier_notice = (
                EventType.system,
                "Local intent classifier ready",
                self._format_classifier_ready_message(local_classifier.runtime_info()),
                local_classifier.runtime_info(),
            )
        else:
            classifier_runtime = local_classifier.runtime_info()
            self._classifier_notice = (
                EventType.warning,
                "Local intent classifier unavailable",
                str(classifier_runtime.get("error", "local intent classifier unavailable")),
                classifier_runtime,
            )
        embedding_gate = (
            EmbeddingGate(
                model_name=settings.embedding_gate_model,
                threshold=settings.embedding_gate_threshold,
            )
            if settings.enable_embedding_gate
            else None
        )
        self._interpreter = RuleBasedTradeInterpreter(
            fallback=fallback,
            embedding_gate=embedding_gate,
            local_classifier=local_classifier,
            classifier_min_probability=settings.local_intent_classifier_min_probability,
            classifier_block_probability=settings.local_intent_classifier_block_probability,
            classifier_recovery_probability=settings.local_intent_classifier_recovery_probability,
            candidate_window_ms=settings.candidate_window_ms,
            candidate_preroll_ms=settings.candidate_preroll_ms,
            candidate_max_fragments=settings.candidate_max_fragments,
            candidate_open_probability=settings.candidate_open_probability,
            candidate_keep_probability=settings.candidate_keep_probability,
            entry_context_window_ms=settings.entry_context_window_ms,
            entry_guard_window_ms=settings.entry_guard_window_ms,
            fallback_confirmation_timeout_ms=settings.gemini_confirmation_timeout_ms,
        )
        self._risk_engine = RiskEngine(settings)
        self._bridge_client = NinjaTraderBridgeClient(settings)
        self._executor = NinjaTraderExecutor(settings, bridge_client=self._bridge_client)
        self._transcribers: dict[str, BaseTranscriber] = {}
        self._pending_preview_executions: dict[str, PendingPreviewExecution] = {}
        self._preview_confirmation_window = timedelta(
            milliseconds=max(1, settings.preview_entry_confirmation_window_ms)
        )
        # Debounce: at most one pending save task per session.
        self._pending_save_tasks: dict[str, asyncio.Task[None]] = {}
        self._event_write_tasks: set[asyncio.Task[None]] = set()
        self._event_write_locks: dict[str, asyncio.Lock] = {}
        self._pending_event_writes: dict[str, int] = {}
        self._pending_event_waiters: dict[str, asyncio.Event] = {}
        self._transcriber_ready_tasks: dict[str, asyncio.Task[None]] = {}
        self._broker_state_cache: dict[tuple[str, str | None, str | None], CachedBrokerState] = {}
        self._broker_state_requests: dict[
            tuple[str, str | None, str | None],
            asyncio.Task[dict[str, Any]],
        ] = {}

    @property
    def event_hub(self) -> EventHub:
        return self._event_hub

    @property
    def settings(self) -> Settings:
        return self._settings

    async def get_broker_state(
        self,
        session_id: str,
        *,
        account: str | None = None,
        symbol: str | None = None,
    ) -> dict:
        session = self.get_session(session_id)
        resolved_account, resolved_symbol = self._resolve_broker_query(
            session,
            account=account,
            symbol=symbol,
        )

        try:
            return await self._fetch_broker_state(
                session.id,
                account=resolved_account,
                symbol=resolved_symbol,
            )
        except Exception as error:
            return {
                "ok": False,
                "code": "bridge_unavailable",
                "message": str(error),
                "timestamp_utc": utc_now().isoformat(),
                "account": resolved_account,
                "symbol": resolved_symbol,
            }

    def list_sessions(self) -> list[StreamSession]:
        sessions = sorted(self._sessions.values(), key=lambda session: session.created_at, reverse=True)
        return [session.model_copy(deep=True) for session in sessions]

    def get_session(self, session_id: str) -> StreamSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    async def delete_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        self._interpreter.clear_session(session_id)
        self._pending_preview_executions.pop(session_id, None)
        self._clear_broker_state_cache(session_id)
        await self._wait_for_pending_event_writes(session_id)
        await self._cancel_transcriber_ready_task(session_id)

        # Cancel any pending debounced save for this session.
        task = self._pending_save_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

        transcriber = self._transcribers.pop(session_id, None)
        if transcriber is not None:
            await transcriber.close()

        # Remove from _sessions last so callbacks during cleanup can still
        # look up the session (e.g. transcriber flush → on_segment).
        self._sessions.pop(session_id, None)
        self._session_store.delete(session_id)
        self._store.delete(session_id)

    async def create_session(self, request: CreateSessionRequest) -> StreamSession:
        config = request.config.model_copy(deep=True)
        requested_symbol = _clean_optional(config.symbol)
        default_symbol = _clean_optional(self._settings.default_symbol)
        # If caller kept the legacy default symbol, respect configured backend default.
        if default_symbol is not None and (requested_symbol is None or requested_symbol.upper() == "NQ"):
            config.symbol = default_symbol
        if config.default_contract_size == 1 and self._settings.default_contract_size != 1:
            config.default_contract_size = max(1, min(10, self._settings.default_contract_size))

        session = StreamSession(config=config)
        session.market = MarketSnapshot(symbol=config.symbol)
        self._sessions[session.id] = session
        await self._emit(
            session.id,
            EventType.system,
            "Session created",
            f"Session started for {session.config.source_name}.",
            {"config": session.config.model_dump(mode="json")},
            persist_session=False,  # saved explicitly below
        )
        classifier_notice = getattr(self, "_classifier_notice", None)
        if classifier_notice is not None:
            await self._emit(
                session.id,
                classifier_notice[0],
                classifier_notice[1],
                classifier_notice[2],
                classifier_notice[3],
                persist_session=False,
            )
        # Save immediately on creation (not debounced) to persist the new session.
        self._session_store.save(session)
        return session.model_copy(deep=True)

    async def manual_trade(self, session_id: str, request: ManualTradeRequest) -> StreamSession:
        session = self.get_session(session_id)
        self._apply_broker_overrides_from_request(session, request)
        await self._sync_session_from_broker(session)

        action = request.action.value if hasattr(request.action, "value") else str(request.action)
        if action == ManualTradeAction.close.value:
            intents = [
                self._build_manual_intent(
                    session=session,
                    tag=ActionTag.exit_all,
                    side=session.position.side if session.position is not None else None,
                )
            ]
        elif action == ManualTradeAction.buy.value:
            if session.position is None:
                intents = [self._build_manual_intent(session=session, tag=ActionTag.enter_long, side=TradeSide.long)]
            elif session.position.side == TradeSide.long:
                intents = [self._build_manual_intent(session=session, tag=ActionTag.add, side=TradeSide.long)]
            else:
                intents = [
                    self._build_manual_intent(session=session, tag=ActionTag.exit_all, side=session.position.side),
                    self._build_manual_intent(session=session, tag=ActionTag.enter_long, side=TradeSide.long),
                ]
        else:  # SELL
            if session.position is None:
                intents = [self._build_manual_intent(session=session, tag=ActionTag.enter_short, side=TradeSide.short)]
            elif session.position.side == TradeSide.short:
                intents = [self._build_manual_intent(session=session, tag=ActionTag.add, side=TradeSide.short)]
            else:
                intents = [
                    self._build_manual_intent(session=session, tag=ActionTag.exit_all, side=session.position.side),
                    self._build_manual_intent(session=session, tag=ActionTag.enter_short, side=TradeSide.short),
                ]

        for intent in intents:
            self._apply_wide_brackets(intent, session)
            session.last_intent = intent
            await self._emit(
                session.id,
                EventType.intent,
                "Manual trade",
                intent.tag.value if hasattr(intent.tag, "value") else str(intent.tag),
                intent.model_dump(mode="json"),
                patch=SessionPatch(last_intent=intent),
            )

            result = await self._execute_with_contract_size(session, intent, request.contract_size)
            risk_message = "Manual trade requested."
            if result.approved:
                risk_message = "Manual trade sent to executor."
            await self._emit(
                session.id,
                EventType.risk,
                "Risk check",
                risk_message,
                {"approved": result.approved, "intent": intent.model_dump(mode="json"), "manual": True},
            )
            if result.approved:
                await self._sync_session_from_broker(session, force_refresh=True)
                self._refresh_execution_result(session, result)
            await self._emit_execution(session.id, result)
            if not result.approved and len(intents) > 1:
                break

        return session.model_copy(deep=True)

    async def update_session_config(self, session_id: str, request: UpdateSessionConfigRequest) -> StreamSession:
        session = self.get_session(session_id)
        changes: list[str] = []
        restart_transcriber = False

        if "enable_partial_intent_detection" in request.model_fields_set and (
            request.enable_partial_intent_detection != session.config.enable_partial_intent_detection
        ):
            session.config.enable_partial_intent_detection = request.enable_partial_intent_detection
            if not request.enable_partial_intent_detection:
                session.latest_candidate_intent = None
            state = "enabled" if request.enable_partial_intent_detection else "disabled"
            changes.append(f"partial intent {state}")

        if "enable_ai_fallback" in request.model_fields_set and request.enable_ai_fallback != session.config.enable_ai_fallback:
            session.config.enable_ai_fallback = bool(request.enable_ai_fallback)
            state = "enabled" if session.config.enable_ai_fallback else "disabled"
            changes.append(f"gemini entry confirm {state}")

        if (
            "enable_early_preview_entries" in request.model_fields_set
            and request.enable_early_preview_entries != session.config.enable_early_preview_entries
        ):
            session.config.enable_early_preview_entries = bool(request.enable_early_preview_entries)
            if not session.config.enable_early_preview_entries:
                self._pending_preview_executions.pop(session_id, None)
            state = "enabled" if session.config.enable_early_preview_entries else "disabled"
            changes.append(f"early preview entry {state}")

        if (
            "transcription_model" in request.model_fields_set
            and request.transcription_model
            and request.transcription_model != session.config.transcription_model
        ):
            session.config.transcription_model = request.transcription_model
            session.latest_partial_text = ""
            session.latest_partial_metrics = None
            session.latest_candidate_intent = None
            restart_transcriber = True
            changes.append(f"model {request.transcription_model}")

        if "broker_account_override" in request.model_fields_set:
            broker_account_override = _clean_optional(request.broker_account_override)
            if broker_account_override != session.config.broker_account_override:
                session.config.broker_account_override = broker_account_override
                self._clear_broker_state_cache(session_id)
                changes.append(
                    f"broker account {broker_account_override}" if broker_account_override else "broker account cleared"
                )

        if "broker_symbol_override" in request.model_fields_set:
            broker_symbol_override = _clean_optional(request.broker_symbol_override)
            if broker_symbol_override != session.config.broker_symbol_override:
                session.config.broker_symbol_override = broker_symbol_override
                self._clear_broker_state_cache(session_id)
                changes.append(
                    f"broker symbol {broker_symbol_override}" if broker_symbol_override else "broker symbol cleared"
                )

        if restart_transcriber:
            await self._cancel_transcriber_ready_task(session_id)
            transcriber = self._transcribers.pop(session_id, None)
            if transcriber is not None:
                await transcriber.close()

        if changes:
            await self._emit(
                session.id,
                EventType.system,
                "Session config updated",
                ", ".join(changes),
                {"config": session.config.model_dump(mode="json")},
            )

        return session.model_copy(deep=True)

    async def ingest_segment(self, session_id: str, request: TextSegmentRequest) -> StreamSession:
        session = self.get_session(session_id)
        segment = TranscriptSegment(
            session_id=session_id,
            text=request.text,
            status=request.status,
            source=request.source,
            item_id=request.item_id,
            confidence=request.confidence,
        )
        await self._process_segment(session, segment)
        return session.model_copy(deep=True)

    async def handle_live_segment(self, segment: TranscriptSegment) -> None:
        session = self.get_session(segment.session_id)
        await self._process_segment(session, segment)

    async def push_audio(self, session_id: str, data: bytes, sample_rate: int) -> None:
        transcriber = await self.ensure_transcriber(session_id)
        try:
            await transcriber.push_audio(data, sample_rate)
        except Exception as error:
            await self._emit(
                session_id,
                EventType.warning,
                "Audio ingest error",
                str(error),
                {"stage": "stream"},
            )
            raise

    async def ensure_transcriber(self, session_id: str) -> BaseTranscriber:
        session = self.get_session(session_id)
        transcriber = self._transcribers.get(session_id)
        if transcriber is None:
            try:
                transcriber = await self._build_transcriber(session)
            except Exception as error:
                await self._emit(
                    session_id,
                    EventType.warning,
                    "Transcriber error",
                    str(error),
                    {"stage": "startup"},
                )
                raise
            self._transcribers[session_id] = transcriber
        return transcriber

    async def close(self) -> None:
        transcriber_ready_tasks = list(self._transcriber_ready_tasks.values())
        for task in transcriber_ready_tasks:
            task.cancel()
        if transcriber_ready_tasks:
            await asyncio.gather(*transcriber_ready_tasks, return_exceptions=True)
        self._transcriber_ready_tasks.clear()
        for transcriber in self._transcribers.values():
            await transcriber.close()
        await self._interpreter.close()
        # Cancel all debounced save tasks and do a final synchronous flush.
        for task in self._pending_save_tasks.values():
            task.cancel()
        self._pending_save_tasks.clear()
        for session in self._sessions.values():
            self._session_store.save(session)
        if self._event_write_tasks:
            await asyncio.gather(*list(self._event_write_tasks), return_exceptions=True)
        await self._bridge_client.close()

    async def _process_segment(self, session: StreamSession, segment: TranscriptSegment) -> None:
        if segment.status == SegmentStatus.partial:
            session.latest_partial_text = segment.text
            session.latest_partial_metrics = segment.metrics
            session.latest_candidate_intent = None
            if session.config.enable_partial_intent_detection and segment.text.strip():
                session.latest_candidate_intent = self._interpreter.interpret_partial(session, segment)
            await self._emit(
                session.id,
                EventType.transcript,
                "Transcript",
                segment.text,
                segment.model_dump(mode="json"),
                patch=SessionPatch(
                    latest_partial_text=session.latest_partial_text,
                    latest_partial_metrics=session.latest_partial_metrics,
                    latest_candidate_intent=session.latest_candidate_intent,
                ),
                persist_session=False,
                persist_event=False,
                append_to_session=False,
            )
            if session.config.enable_early_preview_entries and segment.text.strip():
                await self._maybe_execute_preview_entry(session, segment)
            return

        # Final segment.
        session.latest_partial_text = ""
        session.latest_partial_metrics = None
        session.latest_candidate_intent = None
        session.latest_final_metrics = segment.metrics
        session.transcripts.append(segment)
        del session.transcripts[: -self._settings.max_transcript_segments]

        await self._emit(
            session.id,
            EventType.transcript,
            "Transcript",
            segment.text,
            segment.model_dump(mode="json"),
            patch=SessionPatch(
                latest_partial_text="",
                latest_partial_metrics=None,
                latest_candidate_intent=None,
                latest_final_metrics=segment.metrics,
                new_transcript=segment,
            ),
        )

        if not segment.text.strip():
            return

        pending_preview = self._get_pending_preview_execution(session.id)
        expired_preview = self._pop_expired_preview_execution(session.id)
        if pending_preview is not None:
            if self._interpreter.confirm_preview_entry(session, segment, pending_intent=pending_preview.intent):
                self._pending_preview_executions.pop(session.id, None)
                await self._emit(
                    session.id,
                    EventType.system,
                    "Preview entry confirmed",
                    "Final transcript confirmed the early preview entry.",
                    {
                        "intent": pending_preview.intent.model_dump(mode="json"),
                        "segment_id": segment.id,
                    },
                )
                return

            self._pending_preview_executions.pop(session.id, None)
            await self._emit(
                session.id,
                EventType.warning,
                "Preview entry rejected",
                "Final transcript did not confirm the early preview entry. Flattening position.",
                {
                    "intent": pending_preview.intent.model_dump(mode="json"),
                    "segment_id": segment.id,
                },
            )
            await self._flatten_preview_entry(session, pending_preview)
            # Fall through to normal interpretation so the final segment's
            # own trading signal (if any) is not silently lost.

        if expired_preview is not None:
            await self._emit(
                session.id,
                EventType.warning,
                "Preview entry expired",
                "Preview confirmation window expired. Flattening position.",
                {"intent": expired_preview.intent.model_dump(mode="json")},
            )
            await self._flatten_preview_entry(session, expired_preview)

        await self._sync_session_from_broker(session)

        intent = await self._interpreter.interpret(session, segment)
        diagnostic = self._interpreter.consume_diagnostic(session.id)
        if diagnostic is not None:
            await self._emit(
                session.id,
                diagnostic.event_type,
                diagnostic.title,
                diagnostic.message,
                diagnostic.data,
            )
        if intent is None:
            return

        self._apply_wide_brackets(intent, session)

        session.last_intent = intent
        intent_tag = intent.tag.value if hasattr(intent.tag, "value") else str(intent.tag)
        await self._emit(
            session.id,
            EventType.intent,
            "Intent detected",
            intent_tag,
            intent.model_dump(mode="json"),
            patch=SessionPatch(last_intent=intent),
        )

        if not session.config.auto_execute or session.config.execution_mode == ExecutionMode.review:
            return

        decision = self._risk_engine.evaluate(session, intent)
        await self._emit(
            session.id,
            EventType.risk,
            "Risk check",
            decision.reason,
            {"approved": decision.approved, "intent": intent.model_dump(mode="json")},
        )

        if not decision.approved:
            return

        result = await self._executor.execute(session, intent)
        if result.approved:
            await self._sync_session_from_broker(session, force_refresh=True)
            self._refresh_execution_result(session, result)
        await self._emit_execution(session.id, result)

    def _build_manual_intent(
        self,
        *,
        session: StreamSession,
        tag: ActionTag,
        side: TradeSide | None,
    ) -> TradeIntent:
        configured_symbol = _clean_optional(session.config.broker_symbol_override) or _clean_optional(
            self._settings.ninjatrader_symbol
        )
        entry_reference = session.market.last_price
        if entry_reference is None and session.position is not None:
            entry_reference = session.position.average_price
        return TradeIntent(
            session_id=session.id,
            tag=tag,
            symbol=configured_symbol or session.market.symbol,
            side=side,
            entry_price=entry_reference,
            evidence_text=f"manual_{tag.value.lower()}",
            confidence=1.0,
            source_latency_ms=0,
            stale_after_ms=max(self._settings.stale_intent_ms, 60_000),
            created_at=utc_now(),
        )

    async def _execute_with_contract_size(
        self,
        session: StreamSession,
        intent: TradeIntent,
        contract_size: int,
    ) -> ExecutionResult:
        previous_size = session.config.default_contract_size
        session.config.default_contract_size = max(1, min(10, int(contract_size)))
        try:
            return await self._executor.execute(session, intent)
        finally:
            session.config.default_contract_size = previous_size

    def _get_pending_preview_execution(self, session_id: str) -> PendingPreviewExecution | None:
        pending = self._pending_preview_executions.get(session_id)
        if pending is None:
            return None
        if datetime.now(UTC) - pending.executed_at <= self._preview_confirmation_window:
            return pending
        return None

    def _pop_expired_preview_execution(self, session_id: str) -> PendingPreviewExecution | None:
        """Remove and return an expired preview execution, or None if not expired."""
        pending = self._pending_preview_executions.get(session_id)
        if pending is None:
            return None
        if datetime.now(UTC) - pending.executed_at <= self._preview_confirmation_window:
            return None
        return self._pending_preview_executions.pop(session_id, None)

    async def _maybe_execute_preview_entry(self, session: StreamSession, segment: TranscriptSegment) -> None:
        if not session.config.auto_execute or session.config.execution_mode == ExecutionMode.review:
            return
        if self._get_pending_preview_execution(session.id) is not None:
            return

        await self._sync_session_from_broker(session)
        preview_intent = self._interpreter.interpret_preview_entry(session, segment)
        if preview_intent is None:
            return

        confirmation_intent = preview_intent.model_copy(deep=True)
        self._apply_wide_brackets(preview_intent, session)
        session.last_intent = preview_intent
        await self._emit(
            session.id,
            EventType.intent,
            "Early preview entry",
            preview_intent.tag.value if hasattr(preview_intent.tag, "value") else str(preview_intent.tag),
            {"preview_entry": True, "intent": preview_intent.model_dump(mode="json")},
            patch=SessionPatch(last_intent=preview_intent),
        )

        decision = self._risk_engine.evaluate(session, preview_intent)
        await self._emit(
            session.id,
            EventType.risk,
            "Risk check",
            decision.reason,
            {"approved": decision.approved, "intent": preview_intent.model_dump(mode="json"), "preview_entry": True},
        )
        if not decision.approved:
            return

        result = await self._executor.execute(session, preview_intent)
        if result.approved:
            await self._sync_session_from_broker(session, force_refresh=True)
            self._refresh_execution_result(session, result)
        await self._emit_execution(session.id, result)
        if not result.approved:
            return

        self._pending_preview_executions[session.id] = PendingPreviewExecution(
            intent=confirmation_intent,
            executed_at=datetime.now(UTC),
        )
        await self._emit(
            session.id,
            EventType.system,
            "Preview entry awaiting final",
            "Early preview entry executed. Waiting for the final transcript to confirm it.",
            {"intent": confirmation_intent.model_dump(mode="json")},
        )

    async def _flatten_preview_entry(self, session: StreamSession, pending: PendingPreviewExecution) -> None:
        exit_intent = TradeIntent(
            session_id=session.id,
            tag=ActionTag.exit_all,
            symbol=pending.intent.symbol,
            side=pending.intent.side,
            confidence=1.0,
            evidence_text=f"preview_flatten_{pending.intent.id}",
            source_latency_ms=0,
            stale_after_ms=max(self._settings.stale_intent_ms, 60_000),
            created_at=utc_now(),
        )
        session.last_intent = exit_intent
        await self._emit(
            session.id,
            EventType.intent,
            "Preview flatten",
            exit_intent.tag.value if hasattr(exit_intent.tag, "value") else str(exit_intent.tag),
            {"preview_flatten": True, "intent": exit_intent.model_dump(mode="json")},
            patch=SessionPatch(last_intent=exit_intent),
        )
        result = await self._executor.execute(session, exit_intent)
        if result.approved:
            await self._sync_session_from_broker(session, force_refresh=True)
            self._refresh_execution_result(session, result)
        await self._emit(
            session.id,
            EventType.risk,
            "Risk check",
            "Preview confirmation failed. Flattening the early entry.",
            {"approved": result.approved, "intent": exit_intent.model_dump(mode="json"), "preview_flatten": True},
        )
        await self._emit_execution(session.id, result)

    def _apply_wide_brackets(self, intent: TradeIntent, session: StreamSession) -> None:
        if not self._settings.force_wide_brackets:
            return
        if intent.tag not in _ENTRY_ACTIONS:
            return

        side = intent.side
        if side is None:
            if intent.tag == ActionTag.enter_long:
                side = TradeSide.long
            elif intent.tag == ActionTag.enter_short:
                side = TradeSide.short
            elif intent.tag == ActionTag.add and session.position is not None:
                side = session.position.side
        if side is None:
            return

        reference_price = intent.entry_price if intent.entry_price is not None else session.market.last_price
        if reference_price is None and session.position is not None:
            reference_price = session.position.average_price
        if reference_price is None:
            return

        stop_offset = max(1.0, float(self._settings.wide_stop_points))
        target_offset = max(stop_offset, float(self._settings.wide_target_points))
        if side == TradeSide.long:
            intent.stop_price = reference_price - stop_offset
            intent.target_price = reference_price + target_offset
        else:
            intent.stop_price = reference_price + stop_offset
            intent.target_price = reference_price - target_offset

    async def _sync_session_from_broker(self, session: StreamSession, *, force_refresh: bool = False) -> None:
        try:
            configured_account, resolved_symbol = self._resolve_broker_query(session)
            state = await self._fetch_broker_state(
                session.id,
                account=configured_account,
                symbol=resolved_symbol,
                force_refresh=force_refresh,
            )
        except Exception:
            return

        if not isinstance(state, dict) or not state.get("ok"):
            return

        symbol_from_state = _clean_optional(
            state["symbol"] if isinstance(state.get("symbol"), str) else None
        )
        last = _as_optional_float(state.get("last_price"))
        bid = _as_optional_float(state.get("bid_price"))
        ask = _as_optional_float(state.get("ask_price"))

        if symbol_from_state or resolved_symbol or last is not None or bid is not None or ask is not None:
            session.market = MarketSnapshot(
                symbol=symbol_from_state or resolved_symbol,
                last_price=last if last is not None else session.market.last_price,
                bid_price=bid if bid is not None else session.market.bid_price,
                ask_price=ask if ask is not None else session.market.ask_price,
                received_at=utc_now(),
            )

        market_position = str(state.get("market_position", "")).upper()
        quantity = int(_as_optional_float(state.get("quantity")) or 0)
        average_price = _as_optional_float(state.get("average_price"))
        stop_price = _as_optional_float(state.get("stop_price"))
        target_price = _as_optional_float(state.get("target_price"))

        if market_position in {"LONG", "SHORT"} and quantity > 0 and average_price is not None:
            side = TradeSide.long if market_position == "LONG" else TradeSide.short
            opened_at = session.position.opened_at if session.position else utc_now()
            session.position = PositionState(
                side=side,
                quantity=quantity,
                average_price=average_price,
                stop_price=stop_price,
                target_price=target_price,
                opened_at=opened_at,
                realized_pnl=session.position.realized_pnl if session.position else 0.0,
            )
        elif market_position == "FLAT":
            session.position = None

        realized = _as_optional_float(state.get("account_realized_pnl"))
        if realized is not None:
            session.realized_pnl = realized

    async def _fetch_broker_state(
        self,
        session_id: str,
        *,
        account: str | None,
        symbol: str | None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        cache_key = (session_id, account, symbol)
        if not force_refresh:
            in_flight = self._broker_state_requests.get(cache_key)
            if in_flight is not None:
                return await asyncio.shield(in_flight)
            cached = self._broker_state_cache.get(cache_key)
            if cached is not None and (time.monotonic() - cached.fetched_monotonic) <= _BROKER_STATE_CACHE_TTL_S:
                return cached.state

        task = asyncio.create_task(self._bridge_client.fetch_state(account=account, symbol=symbol))
        self._broker_state_requests[cache_key] = task
        try:
            state = await asyncio.shield(task)
        finally:
            current = self._broker_state_requests.get(cache_key)
            if current is task:
                self._broker_state_requests.pop(cache_key, None)

        if isinstance(state, dict):
            self._broker_state_cache[cache_key] = CachedBrokerState(
                state=state,
                fetched_monotonic=time.monotonic(),
            )
        return state

    def _clear_broker_state_cache(self, session_id: str) -> None:
        for cache_key in [key for key in self._broker_state_cache if key[0] == session_id]:
            self._broker_state_cache.pop(cache_key, None)
        for request_key in [key for key in self._broker_state_requests if key[0] == session_id]:
            self._broker_state_requests.pop(request_key, None)

    def _apply_broker_overrides_from_request(self, session: StreamSession, request: ManualTradeRequest) -> None:
        if "account" in request.model_fields_set:
            account = _clean_optional(request.account)
            if account != session.config.broker_account_override:
                session.config.broker_account_override = account
                self._clear_broker_state_cache(session.id)
        if "symbol" in request.model_fields_set:
            symbol = _clean_optional(request.symbol)
            if symbol != session.config.broker_symbol_override:
                session.config.broker_symbol_override = symbol
                self._clear_broker_state_cache(session.id)

    def _resolve_broker_query(
        self,
        session: StreamSession,
        *,
        account: str | None = None,
        symbol: str | None = None,
    ) -> tuple[str | None, str | None]:
        session_account = _clean_optional(session.config.broker_account_override)
        session_symbol_override = _clean_optional(session.config.broker_symbol_override)
        resolved_account = _clean_optional(account) or session_account or _clean_optional(self._settings.ninjatrader_account)
        resolved_symbol = _clean_optional(symbol)
        configured_symbol = _clean_optional(self._settings.ninjatrader_symbol)

        if resolved_symbol is None:
            session_symbol = _clean_optional(session.market.symbol)
            if session_symbol_override is not None:
                resolved_symbol = session_symbol_override
            elif _looks_like_ninjatrader_contract(session_symbol):
                resolved_symbol = session_symbol
            else:
                resolved_symbol = configured_symbol

        return resolved_account, resolved_symbol

    async def _build_transcriber(self, session: StreamSession) -> BaseTranscriber:
        if not session.config.enable_audio_capture:
            return NoopTranscriber()

        if self._settings.transcription_backend != "local_whisper":
            return NoopTranscriber()

        transcriber = LocalWhisperTranscriber(
            settings=self._settings,
            session_id=session.id,
            model_name=session.config.transcription_model,
            prompt=self._settings.audio_prompt,
            on_segment=self.handle_live_segment,
        )
        await transcriber.start()
        runtime = transcriber.runtime_info()
        await self._emit(
            session.id,
            EventType.system,
            "Transcriber starting",
            f"{runtime['model']} {runtime.get('engine', 'segment')} preview pipeline booting on {runtime['device']}",
            runtime,
        )
        if self._settings.transcription_require_cuda:
            ready_runtime = await transcriber.wait_until_ready()
            if ready_runtime:
                await self._emit_transcriber_runtime(session.id, ready_runtime)
        else:
            self._schedule_transcriber_ready_task(session.id, transcriber)
        return transcriber

    def _schedule_transcriber_ready_task(self, session_id: str, transcriber: BaseTranscriber) -> None:
        existing = self._transcriber_ready_tasks.get(session_id)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(self._observe_transcriber_runtime(session_id, transcriber))
        self._transcriber_ready_tasks[session_id] = task

        def _cleanup(done_task: asyncio.Task[None]) -> None:
            current = self._transcriber_ready_tasks.get(session_id)
            if current is done_task:
                self._transcriber_ready_tasks.pop(session_id, None)

        task.add_done_callback(_cleanup)

    async def _cancel_transcriber_ready_task(self, session_id: str) -> None:
        task = self._transcriber_ready_tasks.pop(session_id, None)
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _observe_transcriber_runtime(self, session_id: str, transcriber: BaseTranscriber) -> None:
        try:
            runtime = await transcriber.wait_until_ready()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if session_id in self._sessions:
                await self._emit(
                    session_id,
                    EventType.warning,
                    "Transcriber error",
                    str(error),
                    {"stage": "runtime"},
                )
            return

        if runtime is None or session_id not in self._sessions:
            return
        await self._emit_transcriber_runtime(session_id, runtime)

    async def _emit_transcriber_runtime(self, session_id: str, runtime: dict[str, object]) -> None:
        if session_id not in self._sessions:
            return

        await self._emit(
            session_id,
            EventType.system,
            "Transcriber ready",
            self._format_transcriber_ready_message(runtime),
            runtime,
        )

        warning_message = self._transcriber_runtime_warning(runtime)
        if warning_message is None:
            return
        await self._emit(
            session_id,
            EventType.warning,
            "Transcriber degraded",
            warning_message,
            runtime,
        )

    def _format_transcriber_ready_message(self, runtime: dict[str, object]) -> str:
        model = str(runtime.get("model", "transcriber"))
        preview_model = str(runtime.get("preview_model", model))
        device = str(runtime.get("device", "unknown"))
        preview_device = str(runtime.get("preview_device", device))
        segmenter_backend = str(runtime.get("segmenter_backend", "unknown"))
        engine = str(runtime.get("engine", "segment"))

        if preview_model == model and preview_device == device:
            return f"{model} ready on {device} with {segmenter_backend} VAD and {engine} preview"
        return (
            f"{model} ready on {device}; preview {preview_model} ready on {preview_device} "
            f"with {segmenter_backend} VAD and {engine} preview"
        )

    def _transcriber_runtime_warning(self, runtime: dict[str, object]) -> str | None:
        if not self._settings.transcription_warn_on_cpu_fallback:
            return None

        resolved_device = str(runtime.get("resolved_device", ""))
        configured_device = str(runtime.get("configured_device", ""))
        if resolved_device != "cuda" or configured_device == "cpu":
            return None

        degraded_paths: list[str] = []
        final_device = str(runtime.get("device", ""))
        preview_device = str(runtime.get("preview_device", ""))
        if final_device != "cuda":
            degraded_paths.append(f"final model loaded on {final_device}")
        if preview_device != "cuda":
            degraded_paths.append(f"preview model loaded on {preview_device}")
        if not degraded_paths:
            return None
        return "CUDA was expected, but " + " and ".join(degraded_paths)

    def _format_classifier_ready_message(self, runtime: dict[str, object]) -> str:
        model_name = str(runtime.get("model_name", "classifier"))
        device = str(runtime.get("device", "unknown"))
        label_count = len(runtime.get("labels", [])) if isinstance(runtime.get("labels"), list) else 0
        return f"{model_name} ready on {device} with {label_count} intent labels"

    def _refresh_execution_result(self, session: StreamSession, result: ExecutionResult) -> None:
        result.market_price = session.market.last_price
        result.position = session.position.model_copy(deep=True) if session.position else None

    async def _emit_execution(self, session_id: str, result: ExecutionResult) -> None:
        session = self.get_session(session_id)
        await self._emit(
            session_id,
            EventType.execution,
            "Execution",
            result.message,
            result.model_dump(mode="json"),
            patch=SessionPatch(
                market=session.market,
                position=session.position,
                realized_pnl=session.realized_pnl,
            ),
        )

    async def _emit(
        self,
        session_id: str,
        event_type: EventType,
        title: str,
        message: str,
        data: dict,
        *,
        patch: SessionPatch | None = None,
        persist_session: bool = True,
        persist_event: bool = True,
        append_to_session: bool = True,
    ) -> None:
        session = self.get_session(session_id)
        event = TimelineEvent(session_id=session_id, type=event_type, title=title, message=message, data=data)

        if append_to_session:
            session.events.append(event)
            del session.events[: -self._settings.max_events]

        if persist_session:
            self._schedule_session_save(session_id)

        ws_message: dict = {
            "type": "event",
            "event": event.model_dump(mode="json"),
            "patch": (patch or SessionPatch()).model_dump(mode="json", exclude_unset=True),
            "append_event": append_to_session,
        }
        await self._event_hub.publish(session_id, ws_message)

        if persist_event:
            self._schedule_event_write(event)

    def _schedule_session_save(self, session_id: str) -> None:
        """Schedule a debounced session save — at most one write per debounce window."""
        task = self._pending_save_tasks.get(session_id)
        if task is not None and not task.done():
            return
        self._pending_save_tasks[session_id] = asyncio.create_task(
            self._debounced_save(session_id)
        )

    async def _debounced_save(self, session_id: str) -> None:
        await asyncio.sleep(_SESSION_SAVE_DEBOUNCE_S)
        session = self._sessions.get(session_id)
        if session is None:
            return
        await asyncio.to_thread(self._session_store.save, session)
        self._pending_save_tasks.pop(session_id, None)

    def _schedule_event_write(self, event: TimelineEvent) -> None:
        session_id = event.session_id
        self._pending_event_writes[session_id] = self._pending_event_writes.get(session_id, 0) + 1
        waiter = self._pending_event_waiters.get(session_id)
        if waiter is not None:
            waiter.clear()

        task = asyncio.create_task(self._persist_event(event))
        self._event_write_tasks.add(task)
        task.add_done_callback(self._handle_event_write_done)

    async def _persist_event(self, event: TimelineEvent) -> None:
        session_id = event.session_id
        lock = self._event_write_locks.setdefault(session_id, asyncio.Lock())
        try:
            async with lock:
                await asyncio.to_thread(self._store.append, event)
        except Exception:
            _LOGGER.exception("Failed to persist event for session %s", session_id)
        finally:
            remaining = max(0, self._pending_event_writes.get(session_id, 0) - 1)
            if remaining == 0:
                self._pending_event_writes.pop(session_id, None)
                waiter = self._pending_event_waiters.get(session_id)
                if waiter is not None:
                    waiter.set()
            else:
                self._pending_event_writes[session_id] = remaining

    def _handle_event_write_done(self, task: asyncio.Task[None]) -> None:
        self._event_write_tasks.discard(task)
        try:
            task.result()
        except BaseException:
            pass

    async def _wait_for_pending_event_writes(self, session_id: str) -> None:
        if self._pending_event_writes.get(session_id, 0) == 0:
            return

        waiter = self._pending_event_waiters.get(session_id)
        if waiter is None:
            waiter = asyncio.Event()
            self._pending_event_waiters[session_id] = waiter

        try:
            await waiter.wait()
        finally:
            if self._pending_event_writes.get(session_id, 0) == 0:
                self._pending_event_waiters.pop(session_id, None)
                self._event_write_locks.pop(session_id, None)
def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _looks_like_ninjatrader_contract(symbol: str | None) -> bool:
    if not symbol:
        return False
    return re.search(r"\s\d{2}-\d{2}$", symbol) is not None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            cleaned = value.strip()
            if not cleaned:
                return None
            return float(cleaned)
        except ValueError:
            return None
    return None
