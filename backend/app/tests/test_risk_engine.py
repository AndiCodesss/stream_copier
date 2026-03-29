from datetime import UTC, datetime

from app.core.config import Settings
from app.models.domain import ActionTag, MarketSnapshot, PositionState, SessionConfig, StreamSession, TradeIntent, TradeSide
from app.services.execution.risk import RiskEngine


def test_risk_engine_requires_stop_for_entry() -> None:
    engine = RiskEngine(Settings())
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21240))
    intent = TradeIntent(
        session_id=session.id,
        tag=ActionTag.enter_long,
        side=TradeSide.long,
        entry_price=21241,
        evidence_text="I'm long here",
        confidence=0.95,
        created_at=datetime.now(UTC),
    )

    decision = engine.evaluate(session, intent)

    assert decision.approved is False
    assert decision.reason == "Stop price is required for entries."


def test_risk_engine_rejects_old_entry_signal_from_stream_latency() -> None:
    engine = RiskEngine(Settings(max_entry_signal_age_ms=4_000))
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21240))
    intent = TradeIntent(
        session_id=session.id,
        tag=ActionTag.enter_short,
        side=TradeSide.short,
        entry_price=21239,
        stop_price=21250,
        source_latency_ms=9_000,
        evidence_text="Putting a little piece on short versus 50s",
        confidence=0.95,
        created_at=datetime.now(UTC),
    )

    decision = engine.evaluate(session, intent)

    assert decision.approved is False
    assert decision.reason == "Entry signal too old (9000 ms)."


def test_risk_engine_rejects_entry_when_context_guard_is_hit() -> None:
    engine = RiskEngine(Settings())
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21240))
    intent = TradeIntent(
        session_id=session.id,
        tag=ActionTag.enter_short,
        side=TradeSide.short,
        entry_price=21239,
        stop_price=21250,
        guard_reason="recent management cue detected",
        source_latency_ms=500,
        evidence_text="Putting a little piece on short versus 50s",
        confidence=0.95,
        created_at=datetime.now(UTC),
    )

    decision = engine.evaluate(session, intent)

    assert decision.approved is False
    assert decision.reason == "Entry blocked by context guard: recent management cue detected."


def test_risk_engine_allows_add_with_existing_position_stop() -> None:
    engine = RiskEngine(Settings())
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="NQ", last_price=21240),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=21220, stop_price=21210),
    )
    intent = TradeIntent(
        session_id=session.id,
        tag=ActionTag.add,
        side=TradeSide.long,
        entry_price=21240,
        evidence_text="Got my add on there",
        confidence=0.95,
        created_at=datetime.now(UTC),
    )

    decision = engine.evaluate(session, intent)

    assert decision.approved is True
