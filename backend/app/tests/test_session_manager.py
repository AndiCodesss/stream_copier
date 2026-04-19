from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.domain import (
    ActionTag,
    CreateSessionRequest,
    EventType,
    ManualTradeAction,
    ManualTradeRequest,
    SegmentStatus,
    SessionConfig,
    TextSegmentRequest,
    TradeSide,
    TranscriptSegment,
    UpdateSessionConfigRequest,
)
from app.services.interpretation.rule_engine import InterpreterDiagnostic
from app.services.session_manager import SessionManager


class _DummyTranscriber:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _BridgeSimulator:
    def __init__(self, *, symbol: str, last_price: float | None) -> None:
        self.symbol = symbol
        self.last_price = last_price
        self.position_side: TradeSide | None = None
        self.position_quantity = 0
        self.average_price: float | None = None
        self.stop_price: float | None = None
        self.target_price: float | None = None
        self.realized_pnl = 0.0

    async def post_command(self, payload: dict[str, object]) -> httpx.Response:
        action = str(payload.get("action", ""))
        quantity_hint = payload.get("quantity_hint")
        fill_price = _as_optional_float(payload.get("entry_price"))
        if fill_price is None:
            fill_price = _as_optional_float(payload.get("market_price"))
        if fill_price is None:
            fill_price = self.last_price if self.last_price is not None else self.average_price
        self.last_price = _as_optional_float(payload.get("market_price")) or self.last_price
        self.symbol = str(payload.get("symbol") or self.symbol)

        if action in {ActionTag.enter_long.value, ActionTag.enter_short.value, ActionTag.add.value}:
            quantity = _resolve_quantity(quantity_hint, int(payload.get("default_contract_size") or 1))
            side = TradeSide.long if str(payload.get("side")) == TradeSide.long.value else TradeSide.short
            assert fill_price is not None
            if self.position_side is None or self.position_quantity <= 0:
                self.position_side = side
                self.position_quantity = quantity
                self.average_price = fill_price
            else:
                new_quantity = self.position_quantity + quantity
                assert self.average_price is not None
                self.average_price = ((self.average_price * self.position_quantity) + (fill_price * quantity)) / new_quantity
                self.position_quantity = new_quantity
                self.position_side = side
            self.stop_price = _as_optional_float(payload.get("stop_price"))
            self.target_price = _as_optional_float(payload.get("target_price"))
        elif action == ActionTag.move_stop.value and self.position_side is not None:
            self.stop_price = _as_optional_float(payload.get("stop_price"))
        elif action == ActionTag.move_to_breakeven.value and self.position_side is not None:
            self.stop_price = self.average_price
        elif action == ActionTag.trim.value and self.position_side is not None and self.average_price is not None:
            quantity = _resolve_trim_quantity(quantity_hint, self.position_quantity)
            exit_price = _as_optional_float(payload.get("target_price")) or fill_price or self.average_price
            self.realized_pnl += _realized_pnl(self.position_side, self.average_price, exit_price, quantity)
            self.position_quantity -= quantity
            if self.position_quantity <= 0:
                self.position_side = None
                self.position_quantity = 0
                self.average_price = None
                self.stop_price = None
                self.target_price = None
            else:
                self.stop_price = self.average_price
        elif action == ActionTag.exit_all.value and self.position_side is not None and self.average_price is not None:
            exit_price = fill_price or self.last_price or self.average_price
            self.realized_pnl += _realized_pnl(
                self.position_side,
                self.average_price,
                exit_price,
                self.position_quantity,
            )
            self.position_side = None
            self.position_quantity = 0
            self.average_price = None
            self.stop_price = None
            self.target_price = None

        request = httpx.Request("POST", "http://127.0.0.1:18080/api/stream-copier/commands")
        return httpx.Response(status_code=200, json={"ok": True, "message": "accepted"}, request=request)

    async def fetch_state(self, *, account: str | None = None, symbol: str | None = None) -> dict[str, object]:
        resolved_symbol = symbol or self.symbol
        market_position = "FLAT"
        if self.position_side == TradeSide.long:
            market_position = "LONG"
        elif self.position_side == TradeSide.short:
            market_position = "SHORT"
        return {
            "ok": True,
            "code": "state",
            "account": account,
            "symbol": resolved_symbol,
            "market_position": market_position,
            "quantity": self.position_quantity,
            "average_price": self.average_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "last_price": self.last_price,
            "account_realized_pnl": self.realized_pnl,
        }


def _make_broker_backed_manager(
    tmp_path: Path,
    *,
    symbol: str = "MNQ 03-26",
    last_price: float | None = 21243.75,
    **settings_overrides: object,
) -> tuple[SessionManager, _BridgeSimulator]:
    manager = SessionManager(
        Settings(
            data_dir=tmp_path,
            default_symbol=symbol,
            ninjatrader_symbol=symbol,
            **settings_overrides,
        )
    )
    bridge = _BridgeSimulator(symbol=symbol, last_price=last_price)
    manager._bridge_client.post_command = bridge.post_command  # type: ignore[method-assign]
    manager._bridge_client.fetch_state = bridge.fetch_state  # type: ignore[method-assign]
    return manager, bridge


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return float(stripped)
    return None


def _resolve_quantity(quantity_hint: object, default_contract_size: int) -> int:
    if quantity_hint == "half":
        return max(1, default_contract_size // 2)
    return default_contract_size


def _resolve_trim_quantity(quantity_hint: object, current_quantity: int) -> int:
    if quantity_hint == "all":
        return current_quantity
    if quantity_hint == "most":
        return max(1, current_quantity - 1)
    if quantity_hint == "half":
        return max(1, current_quantity // 2)
    return 1 if current_quantity > 1 else current_quantity


def _realized_pnl(side: TradeSide, average_price: float, fill_price: float, quantity: int) -> float:
    points = fill_price - average_price if side == TradeSide.long else average_price - fill_price
    return round(points * 20.0 * quantity, 2)


async def test_partial_segments_update_candidate_intent(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    await manager.handle_live_segment(
        TranscriptSegment(
            session_id=session.id,
            text="I'm long here at 46, stop under 36",
            status=SegmentStatus.partial,
            source="test",
            confidence=0.92,
        )
    )

    stored = manager.get_session(session.id)
    assert stored.latest_candidate_intent is not None
    assert stored.latest_candidate_intent.tag == ActionTag.enter_long
    assert stored.transcripts == []


async def test_updating_model_restarts_existing_transcriber(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    transcriber = _DummyTranscriber()
    manager._transcribers[session.id] = transcriber

    updated = await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(transcription_model="small.en"),
    )

    assert updated.config.transcription_model == "small.en"
    assert transcriber.closed is True
    assert session.id not in manager._transcribers


async def test_updating_model_cancels_pending_transcriber_ready_task(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    transcriber = _DummyTranscriber()
    manager._transcribers[session.id] = transcriber
    ready_task = asyncio.create_task(asyncio.sleep(9_999))
    manager._transcriber_ready_tasks[session.id] = ready_task

    await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(transcription_model="small.en"),
    )

    assert ready_task.cancelled() is True


async def test_disabling_partial_intent_detection_clears_candidate(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    await manager.handle_live_segment(
        TranscriptSegment(
            session_id=session.id,
            text="I'm long here at 46, stop under 36",
            status=SegmentStatus.partial,
            source="test",
            confidence=0.92,
        )
    )

    updated = await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(enable_partial_intent_detection=False),
    )

    assert updated.config.enable_partial_intent_detection is False
    assert updated.latest_candidate_intent is None


async def test_partial_segment_can_execute_early_preview_entry_and_wait_for_final(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=25045.5)
    session = await manager.create_session(
        CreateSessionRequest(config=SessionConfig(enable_early_preview_entries=True, default_contract_size=1))
    )

    await manager.handle_live_segment(
        TranscriptSegment(
            session_id=session.id,
            text="So I'm long versus 35 right now.",
            status=SegmentStatus.partial,
            source="test",
            confidence=0.92,
        )
    )

    stored = manager.get_session(session.id)
    assert stored.last_intent is not None
    assert stored.last_intent.tag == ActionTag.enter_long
    assert stored.position is not None
    assert stored.position.side == TradeSide.long
    assert any(event.title == "Preview entry awaiting final" for event in stored.events)


async def test_final_segment_confirms_early_preview_without_adding_position(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=25045.5, default_contract_size=1)
    session = await manager.create_session(
        CreateSessionRequest(config=SessionConfig(enable_early_preview_entries=True, default_contract_size=1))
    )

    await manager.handle_live_segment(
        TranscriptSegment(
            session_id=session.id,
            text="So I'm long versus 35 right now.",
            status=SegmentStatus.partial,
            source="test",
            confidence=0.92,
        )
    )
    updated = await manager.ingest_segment(
        session.id,
        TextSegmentRequest(text="But this is the dip back in. So I'm long versus 35 right now."),
    )

    assert updated.position is not None
    assert updated.position.quantity == 1
    assert any(event.title == "Preview entry confirmed" for event in updated.events)


async def test_final_segment_rejects_early_preview_and_flattens_position(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=25045.5)
    session = await manager.create_session(
        CreateSessionRequest(config=SessionConfig(enable_early_preview_entries=True, default_contract_size=1))
    )

    await manager.handle_live_segment(
        TranscriptSegment(
            session_id=session.id,
            text="So I'm long versus 35 right now.",
            status=SegmentStatus.partial,
            source="test",
            confidence=0.92,
        )
    )
    updated = await manager.ingest_segment(
        session.id,
        TextSegmentRequest(text="Let's say you're buying here and it fills."),
    )

    assert updated.position is None
    assert any(event.title == "Preview entry rejected" for event in updated.events)
    assert any(event.title == "Preview flatten" for event in updated.events)


async def test_updating_gemini_confirmation_flag_persists(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(enable_ai_fallback=False)))

    updated = await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(enable_ai_fallback=True),
    )

    assert updated.config.enable_ai_fallback is True


async def test_updating_early_preview_entry_flag_persists(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(enable_early_preview_entries=False)))

    updated = await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(enable_early_preview_entries=True),
    )

    assert updated.config.enable_early_preview_entries is True


async def test_updating_broker_routing_overrides_persists_clean_values(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    updated = await manager.update_session_config(
        session.id,
        UpdateSessionConfigRequest(
            broker_account_override=" Sim101 ",
            broker_symbol_override=" MNQ 03-26 ",
        ),
    )

    assert updated.config.broker_account_override == "Sim101"
    assert updated.config.broker_symbol_override == "MNQ 03-26"


from app.models.domain import SessionPatch


def test_session_patch_excludes_unset_fields() -> None:
    """Fields not passed to the constructor must be absent from the serialized dict."""
    patch = SessionPatch()
    dumped = patch.model_dump(mode="json", exclude_unset=True)
    assert dumped == {}


def test_session_patch_includes_explicitly_set_none() -> None:
    """A field explicitly set to None must appear in the serialized dict as null."""
    patch = SessionPatch(latest_partial_text=None, latest_partial_metrics=None)
    dumped = patch.model_dump(mode="json", exclude_unset=True)
    assert "latest_partial_text" in dumped
    assert dumped["latest_partial_text"] is None
    assert "latest_partial_metrics" in dumped
    assert dumped["latest_partial_metrics"] is None
    assert "last_intent" not in dumped  # not set → absent


def test_session_patch_serializes_set_values() -> None:
    patch = SessionPatch(latest_partial_text="hello", realized_pnl=42.5)
    dumped = patch.model_dump(mode="json", exclude_unset=True)
    assert dumped["latest_partial_text"] == "hello"
    assert dumped["realized_pnl"] == 42.5
    assert "last_intent" not in dumped


def test_session_manager_uses_ninjatrader_executor_when_configured(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    assert manager._executor.__class__.__name__ == "NinjaTraderExecutor"


async def test_create_session_uses_settings_default_symbol_when_legacy_default_is_requested(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path, default_symbol="MNQ 03-26"))

    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="NQ")))

    assert session.config.symbol == "MNQ 03-26"
    assert session.market.symbol == "MNQ 03-26"


async def test_create_session_keeps_explicit_symbol_even_with_default_override(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path, default_symbol="MNQ 03-26"))

    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="MES 03-26")))

    assert session.config.symbol == "MES 03-26"
    assert session.market.symbol == "MES 03-26"


async def test_final_segments_execute_entry_flow(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    await manager.ingest_segment(session.id, TextSegmentRequest(text="In short there for a small piece."))
    updated = await manager.ingest_segment(session.id, TextSegmentRequest(text="Short versus 60s."))

    assert updated.last_intent is not None
    assert updated.last_intent.tag == ActionTag.move_stop
    assert updated.position is not None
    assert updated.position.side == TradeSide.short
    assert updated.position.stop_price == 21260.0
    assert updated.position.target_price == 21003.75


async def test_final_segment_enters_without_spoken_stop_and_applies_wide_brackets(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    updated = await manager.ingest_segment(session.id, TextSegmentRequest(text="In short there for a small piece."))

    assert updated.last_intent is not None
    assert updated.last_intent.tag == ActionTag.enter_short
    assert updated.last_intent.stop_price == 21363.75
    assert updated.last_intent.target_price == 21003.75
    assert updated.position is not None
    assert updated.position.side == TradeSide.short
    assert updated.position.stop_price == 21363.75
    assert updated.position.target_price == 21003.75


async def test_final_segment_emits_interpreter_diagnostic_when_entry_is_blocked(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    async def fake_interpret(_session, _segment):
        return None

    def fake_consume_diagnostic(_session_id):
        return InterpreterDiagnostic(
            event_type=EventType.warning,
            title="Entry confirmation blocked",
            message="gemini confirmation timed out",
            data={"candidate_intent": {"tag": "ENTER_LONG"}},
        )

    manager._interpreter.interpret = fake_interpret  # type: ignore[method-assign]
    manager._interpreter.consume_diagnostic = fake_consume_diagnostic  # type: ignore[method-assign]

    updated = await manager.ingest_segment(session.id, TextSegmentRequest(text="I'm long versus 35 right now."))

    assert updated.last_intent is None
    assert updated.events[-1].type == EventType.warning
    assert updated.events[-1].title == "Entry confirmation blocked"
    assert updated.events[-1].message == "gemini confirmation timed out"


async def test_manual_buy_uses_contract_size_and_forced_wide_brackets(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75, default_contract_size=1)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    updated = await manager.manual_trade(
        session.id,
        ManualTradeRequest(action=ManualTradeAction.buy, contract_size=3),
    )

    assert updated.last_intent is not None
    assert updated.last_intent.tag == ActionTag.enter_long
    assert updated.position is not None
    assert updated.position.side == TradeSide.long
    assert updated.position.quantity == 3
    assert updated.position.stop_price == 21123.75
    assert updated.position.target_price == 21483.75
    # Contract size override is per-request only.
    assert updated.config.default_contract_size == 1


async def test_manual_trade_applies_broker_overrides_before_execution(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))

    captured: dict[str, str | None] = {}
    original_execute = manager._executor.execute

    async def fake_execute(current_session, intent):
        captured["account"] = current_session.config.broker_account_override
        captured["symbol"] = current_session.config.broker_symbol_override
        return await original_execute(current_session, intent)

    manager._executor.execute = fake_execute  # type: ignore[method-assign]
    try:
        updated = await manager.manual_trade(
            session.id,
            ManualTradeRequest(
                action=ManualTradeAction.buy,
                contract_size=3,
                account="Sim101",
                symbol="MNQ 03-26",
            ),
        )
    finally:
        manager._executor.execute = original_execute  # type: ignore[method-assign]

    assert captured["account"] == "Sim101"
    assert captured["symbol"] == "MNQ 03-26"
    assert updated.config.broker_account_override == "Sim101"
    assert updated.config.broker_symbol_override == "MNQ 03-26"


async def test_manual_close_flattens_position(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    await manager.manual_trade(session.id, ManualTradeRequest(action=ManualTradeAction.buy, contract_size=3))

    closed = await manager.manual_trade(
        session.id,
        ManualTradeRequest(action=ManualTradeAction.close, contract_size=3),
    )

    assert closed.last_intent is not None
    assert closed.last_intent.tag == ActionTag.exit_all
    assert closed.position is None


async def test_manual_sell_reverses_existing_long_position(tmp_path: Path) -> None:
    manager, _bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    await manager.manual_trade(session.id, ManualTradeRequest(action=ManualTradeAction.buy, contract_size=3))

    reversed_session = await manager.manual_trade(
        session.id,
        ManualTradeRequest(action=ManualTradeAction.sell, contract_size=2),
    )

    assert reversed_session.last_intent is not None
    assert reversed_session.last_intent.tag == ActionTag.enter_short
    assert reversed_session.position is not None
    assert reversed_session.position.side == TradeSide.short
    assert reversed_session.position.quantity == 2
    assert reversed_session.position.stop_price == 21363.75
    assert reversed_session.position.target_price == 21003.75


async def test_manual_add_uses_position_average_when_market_price_missing(tmp_path: Path) -> None:
    manager, bridge = _make_broker_backed_manager(tmp_path, last_price=21243.75)
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    await manager.manual_trade(session.id, ManualTradeRequest(action=ManualTradeAction.buy, contract_size=3))

    # Simulate temporary missing live-last while still having an open position.
    stored = manager.get_session(session.id)
    stored.market.last_price = None
    bridge.last_price = None

    updated = await manager.manual_trade(
        session.id,
        ManualTradeRequest(action=ManualTradeAction.buy, contract_size=2),
    )

    assert updated.position is not None
    assert updated.position.side == TradeSide.long
    assert updated.position.quantity == 5
    # Uses existing position average as the wide-bracket reference.
    assert updated.position.stop_price == 21123.75
    assert updated.position.target_price == 21483.75


async def test_ingest_segment_syncs_market_and_position_from_ninjatrader_state(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(
        Settings(
            data_dir=tmp_path,
            ninjatrader_account="Playback101",
            ninjatrader_symbol="MNQ 03-26",
        )
    )
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="NQ")))

    async def fake_interpret(_session, _segment):
        return None

    manager._interpreter.interpret = fake_interpret  # type: ignore[method-assign]

    async def fake_fetch(*, account, symbol):
        assert account == "Playback101"
        assert symbol == "MNQ 03-26"
        return {
            "ok": True,
            "symbol": "MNQ 03-26",
            "last_price": 24686.0,
            "bid_price": 24685.0,
            "ask_price": 24685.5,
            "market_position": "LONG",
            "quantity": 2,
            "average_price": 24680.0,
            "stop_price": 24670.0,
            "target_price": 24710.0,
            "account_realized_pnl": 125.5,
        }

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    await manager.ingest_segment(session.id, TextSegmentRequest(text="status check"))

    stored = manager.get_session(session.id)
    assert stored.market.symbol == "MNQ 03-26"
    assert stored.market.last_price == 24686.0
    assert stored.position is not None
    assert stored.position.side == TradeSide.long
    assert stored.position.quantity == 2
    assert stored.position.average_price == 24680.0
    assert stored.position.stop_price == 24670.0
    assert stored.position.target_price == 24710.0
    assert stored.realized_pnl == 125.5


async def test_get_broker_state_uses_configured_account_and_session_symbol(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path, ninjatrader_account="Playback101"))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="MNQ 03-26")))

    capture: dict[str, object] = {}

    async def fake_fetch(*, account, symbol):
        capture["account"] = account
        capture["symbol"] = symbol
        return {"ok": True, "code": "state", "account": account, "symbol": symbol}

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    state = await manager.get_broker_state(session.id)

    assert state["ok"] is True
    assert capture["account"] == "Playback101"
    assert capture["symbol"] == "MNQ 03-26"


async def test_get_broker_state_prefers_session_broker_overrides(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path, ninjatrader_account="Playback101"))
    session = await manager.create_session(
        CreateSessionRequest(
            config=SessionConfig(
                symbol="NQ",
                broker_account_override="Sim101",
                broker_symbol_override="MNQ 03-26",
            )
        )
    )

    capture: dict[str, object] = {}

    async def fake_fetch(*, account, symbol):
        capture["account"] = account
        capture["symbol"] = symbol
        return {"ok": True, "code": "state", "account": account, "symbol": symbol}

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    state = await manager.get_broker_state(session.id)

    assert state["ok"] is True
    assert capture["account"] == "Sim101"
    assert capture["symbol"] == "MNQ 03-26"


async def test_get_broker_state_reuses_recent_bridge_response(tmp_path: Path) -> None:
    manager = SessionManager(
        Settings(
            data_dir=tmp_path,
            ninjatrader_account="Playback101",
            ninjatrader_symbol="MNQ 03-26",
        )
    )
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="NQ")))

    calls = 0

    async def fake_fetch(*, account, symbol):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "code": "state",
            "account": account,
            "symbol": symbol,
            "timestamp_utc": f"call-{calls}",
        }

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    first = await manager.get_broker_state(session.id)
    second = await manager.get_broker_state(session.id)

    assert calls == 1
    assert second == first


async def test_get_broker_state_returns_bridge_unavailable_payload(monkeypatch, tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path, default_symbol="NQ", ninjatrader_symbol=""))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="NQ")))

    async def fake_fetch(**_kwargs):
        raise RuntimeError("bridge offline")

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    state = await manager.get_broker_state(session.id)

    assert state["ok"] is False
    assert state["code"] == "bridge_unavailable"
    assert "bridge offline" in state["message"]
    assert state["symbol"] is None


async def test_get_broker_state_uses_configured_ninjatrader_symbol_when_session_symbol_not_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        Settings(
            data_dir=tmp_path,
            ninjatrader_account="Playback101",
            ninjatrader_symbol="MNQ 03-26",
        )
    )
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig(symbol="NQ")))

    capture: dict[str, object] = {}

    async def fake_fetch(*, account, symbol):
        capture["account"] = account
        capture["symbol"] = symbol
        return {"ok": True, "code": "state", "account": account, "symbol": symbol}

    manager._bridge_client.fetch_state = fake_fetch  # type: ignore[method-assign]

    state = await manager.get_broker_state(session.id)

    assert state["ok"] is True
    assert capture["account"] == "Playback101"
    assert capture["symbol"] == "MNQ 03-26"


async def test_emit_publishes_before_event_log_write_completes(tmp_path: Path) -> None:
    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    await manager._wait_for_pending_event_writes(session.id)
    queue = await manager.event_hub.subscribe(session.id)
    append_started = threading.Event()
    allow_append = threading.Event()
    persisted_messages: list[str] = []
    original_append = manager._store.append

    def slow_append(event) -> None:
        if event.message == "hot path":
            append_started.set()
            allow_append.wait(timeout=5)
        original_append(event)
        persisted_messages.append(event.message)

    manager._store.append = slow_append  # type: ignore[assignment]

    emit_task = asyncio.create_task(
        manager._emit(
            session.id,
            EventType.system,
            "Hot Path",
            "hot path",
            {},
            persist_session=False,
            persist_event=True,
        )
    )

    message = await asyncio.wait_for(queue.get(), timeout=1)
    assert message["event"]["message"] == "hot path"

    await asyncio.wait_for(emit_task, timeout=1)
    assert await asyncio.to_thread(append_started.wait, 1) is True
    assert "hot path" not in persisted_messages

    allow_append.set()
    await manager.close()
    assert "hot path" in persisted_messages


async def test_ensure_transcriber_emits_runtime_ready_event(tmp_path: Path, monkeypatch) -> None:
    ready = asyncio.Event()

    class _FakeLocalWhisperTranscriber:
        def __init__(self, **_kwargs) -> None:
            self.closed = False

        async def start(self) -> None:
            return None

        async def wait_until_ready(self) -> dict[str, object]:
            await ready.wait()
            return {
                "configured_device": "auto",
                "resolved_device": "cuda",
                "device": "cuda",
                "preview_device": "cuda",
                "model": "distil-large-v3",
                "preview_model": "tiny.en",
                "segmenter_backend": "webrtc",
            }

        def runtime_info(self) -> dict[str, object]:
            return {
                "configured_device": "auto",
                "resolved_device": "cuda",
                "device": "cuda",
                "preview_device": "cuda",
                "model": "distil-large-v3",
                "preview_model": "tiny.en",
                "segmenter_backend": "webrtc",
            }

        async def push_audio(self, data: bytes, sample_rate: int) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("app.services.session_manager.LocalWhisperTranscriber", _FakeLocalWhisperTranscriber)

    manager = SessionManager(Settings(data_dir=tmp_path))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    queue = await manager.event_hub.subscribe(session.id)

    ensure_task = asyncio.create_task(manager.ensure_transcriber(session.id))
    starting = await asyncio.wait_for(queue.get(), timeout=1)
    assert starting["event"]["title"] == "Transcriber starting"

    ready.set()
    await asyncio.wait_for(ensure_task, timeout=1)
    ready_event = await asyncio.wait_for(queue.get(), timeout=1)
    assert ready_event["event"]["title"] == "Transcriber ready"
    assert "webrtc VAD" in ready_event["event"]["message"]


async def test_ensure_transcriber_emits_warning_when_cuda_falls_back_to_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class _FakeLocalWhisperTranscriber:
        def __init__(self, **_kwargs) -> None:
            self.closed = False

        async def start(self) -> None:
            return None

        async def wait_until_ready(self) -> dict[str, object]:
            return {
                "configured_device": "auto",
                "resolved_device": "cuda",
                "device": "cpu",
                "preview_device": "cpu",
                "model": "distil-large-v3",
                "preview_model": "tiny.en",
                "segmenter_backend": "webrtc",
            }

        def runtime_info(self) -> dict[str, object]:
            return {
                "configured_device": "auto",
                "resolved_device": "cuda",
                "device": "cuda",
                "preview_device": "cuda",
                "model": "distil-large-v3",
                "preview_model": "tiny.en",
                "segmenter_backend": "webrtc",
            }

        async def push_audio(self, data: bytes, sample_rate: int) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("app.services.session_manager.LocalWhisperTranscriber", _FakeLocalWhisperTranscriber)

    manager = SessionManager(Settings(data_dir=tmp_path, transcription_warn_on_cpu_fallback=True))
    session = await manager.create_session(CreateSessionRequest(config=SessionConfig()))
    queue = await manager.event_hub.subscribe(session.id)

    await manager.ensure_transcriber(session.id)

    starting = await asyncio.wait_for(queue.get(), timeout=1)
    ready_event = await asyncio.wait_for(queue.get(), timeout=1)
    warning_event = await asyncio.wait_for(queue.get(), timeout=1)

    assert starting["event"]["title"] == "Transcriber starting"
    assert ready_event["event"]["title"] == "Transcriber ready"
    assert warning_event["event"]["title"] == "Transcriber degraded"
    assert "final model loaded on cpu" in warning_event["event"]["message"]
