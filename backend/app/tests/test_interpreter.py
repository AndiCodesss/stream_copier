import asyncio
from unittest.mock import AsyncMock

from app.models.domain import (
    ActionTag,
    ExecutionMode,
    MarketSnapshot,
    PositionState,
    SessionConfig,
    StreamSession,
    TradeIntent,
    TradeSide,
    TranscriptSegment,
)
from app.services.interpretation.rule_engine import RuleBasedTradeInterpreter, _compact_repeated_trade_text, _resolve_price


class _FakeLocalClassifier:
    def __init__(self, prediction) -> None:  # noqa: ANN001
        self._prediction = prediction

    def is_available(self) -> bool:
        return True

    def classify(self, _envelope):  # noqa: ANN001
        return self._prediction

    def close(self) -> None:
        return None


async def test_rule_interpreter_extracts_long_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    segment = TranscriptSegment(session_id=session.id, text="I'm long here at 46, stop under 36, first trim at 58")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.entry_price == 21246.0
    assert intent.stop_price == 21236.0
    assert intent.target_price == 21258.0


def test_resolve_price_supports_spoken_tens() -> None:
    assert _resolve_price("forty six", 21243.75) == 21246.0


def test_resolve_price_supports_grouped_spoken_shorthand() -> None:
    assert _resolve_price("five fifty", 548.0) == 550.0


def test_resolve_price_supports_thousand_phrases() -> None:
    assert _resolve_price("twenty five thousand twenty", 24962.5) == 25020.0


def test_rule_interpreter_partial_candidate_stays_conservative() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    segment = TranscriptSegment(session_id=session.id, text="I'm long here at 46, stop under 36")

    intent = interpreter.interpret_partial(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.entry_price == 21246.0
    assert intent.stop_price == 21236.0


def test_rule_interpreter_partial_ignores_non_trade_preview() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))
    segment = TranscriptSegment(session_id=session.id, text="We're just waiting for the next candle here")

    intent = interpreter.interpret_partial(session, segment)

    assert intent is None


def test_rule_interpreter_preview_entry_requires_strong_first_person_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5))

    intent = interpreter.interpret_preview_entry(
        session,
        TranscriptSegment(session_id=session.id, text="So I'm long versus 35 right now."),
    )

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.stop_price == 25035.0


def test_rule_interpreter_preview_entry_rejects_plain_short_here_without_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5))

    intent = interpreter.interpret_preview_entry(
        session,
        TranscriptSegment(session_id=session.id, text="I'm short here."),
    )

    assert intent is None


def test_rule_interpreter_confirm_preview_entry_requires_matching_final_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5))
    pending = interpreter.interpret_preview_entry(
        session,
        TranscriptSegment(session_id=session.id, text="So I'm long versus 35 right now."),
    )

    assert pending is not None
    assert interpreter.confirm_preview_entry(
        session,
        TranscriptSegment(session_id=session.id, text="But this is the dip back in. So I'm long versus 35 right now."),
        pending_intent=pending,
    )
    assert not interpreter.confirm_preview_entry(
        session,
        TranscriptSegment(session_id=session.id, text="Let's say you're buying here."),
        pending_intent=pending,
    )


async def test_rule_interpreter_enters_on_piece_without_spoken_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))

    first = TranscriptSegment(session_id=session.id, text="In short there for a small peace...")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is not None
    assert first_intent.tag == ActionTag.enter_short
    assert first_intent.side == TradeSide.short
    assert first_intent.stop_price is None
    assert first_intent.entry_price == 21243.75

    session.market.last_price = 21247.0
    second = TranscriptSegment(session_id=session.id, text="Short versus the 60s")
    second_intent = await interpreter.interpret(session, second)

    # Interpreter-only tests do not execute orders, so no open position exists yet.
    # The follow-up "versus" line becomes management only after an actual fill.
    assert second_intent is None


async def test_rule_interpreter_infers_long_entry_from_piece_seed_and_upside_target() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25030.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="So we've got a little piece on for a squeeze. Let's see if we can get up to 70s now.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.side == TradeSide.long
    assert intent.entry_price == 25030.0
    assert intent.target_price == 25070.0


async def test_rule_interpreter_infers_short_entry_from_piece_seed_and_downside_target() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25080.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Small piece on here. Let's see if they can walk this down to 50s.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.side == TradeSide.short
    assert intent.entry_price == 25080.0
    assert intent.target_price == 25050.0


async def test_rule_interpreter_candidate_window_recovers_split_piece_seed_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25044.0))

    first = TranscriptSegment(session_id=session.id, text="Putting a little piece on here.")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="Short versus 44s.")
    second_intent = await interpreter.interpret(session, second)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert second_intent is not None
    assert second_intent.tag == ActionTag.enter_short
    assert second_intent.side == TradeSide.short
    assert second_intent.stop_price == 25044.0
    assert diagnostic is not None
    assert diagnostic.title == "Candidate window recovered intent"


async def test_rule_interpreter_candidate_window_does_not_force_setup_into_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25030.0))

    first = TranscriptSegment(session_id=session.id, text="Looking for the long on a reclaim.")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="If they hold 30s here.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_candidate_window_ignores_retrospective_entry_story() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25044.0))

    first = TranscriptSegment(
        session_id=session.id,
        text="At some point today I thought they were going to go without me, so I put a small piece",
    )
    second = TranscriptSegment(
        session_id=session.id,
        text="on and I took a loss because I didn't stick to my rules.",
    )

    first_intent = await interpreter.interpret(session, first)
    second_intent = await interpreter.interpret(session, second)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert first_intent is None
    assert second_intent is None
    assert diagnostic is None


async def test_rule_interpreter_infers_entry_side_from_recent_setup_window() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.0))

    first = TranscriptSegment(session_id=session.id, text="Looking for a short here.")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(
        session_id=session.id,
        text="Gonna have to put something on here in case. Very tight stop.",
    )
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is not None
    assert second_intent.tag == ActionTag.enter_short
    assert second_intent.side == TradeSide.short
    assert second_intent.entry_price == 25045.0
    assert second_intent.stop_price is None


async def test_rule_interpreter_extracts_no_reclaim_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21250.25))
    segment = TranscriptSegment(session_id=session.id, text="I'm short on this, no reclaim of 40s.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.stop_price == 21240.0


async def test_rule_interpreter_detects_contextual_in_this_short_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=24620.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="So in this short, smaller size, because unfortunately, it wouldn't give me ideal entry, but I wanted to get something on.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.side == TradeSide.short


async def test_rule_interpreter_detects_bare_leadin_in_this_short_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=24620.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Let's see then folks in this short.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.side == TradeSide.short


async def test_rule_interpreter_detects_bare_leadin_in_this_long_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=24620.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Let's see then folks in this long.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.side == TradeSide.long


async def test_rule_interpreter_ignores_if_you_re_in_this_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=24620.0))
    segment = TranscriptSegment(session_id=session.id, text="If you're in this short, you look good.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_some_of_you_might_still_be_in_this_long_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=24620.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Some of you might still be in this long which is fantastic.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_detects_my_stops_gone_in_move_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25020.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25000.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="So my stop's gone in basically 25,000.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_stop
    assert intent.stop_price == 25000.0


async def test_rule_interpreter_detects_stops_moved_to_move_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25080.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="All right, stops moved to 55 now.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_stop
    assert intent.stop_price == 25055.0


async def test_rule_interpreter_detects_ill_be_out_on_reclaim_as_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25150.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25120.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="I'll be out on a reclaim of 180s.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_stop
    assert intent.stop_price == 25180.0


async def test_rule_interpreter_ignores_retrospective_stopped_out_on_runner_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25150.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25120.0),
    )
    segment = TranscriptSegment(
        session_id=session.id,
        text="I'm a bit salty that got stopped out on the runner, but it's all right.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_split_retrospective_stopped_out_on_runner_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25150.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25120.0),
    )

    first = TranscriptSegment(
        session_id=session.id,
        text="they are adding uh levels above at 190 and 200s. I'm a bit salty that got",
    )
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(
        session_id=session.id,
        text="stopped out on the runner, but there we go. It's all right.",
    )
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_detects_runner_break_even_as_breakeven() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25080.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="Runner break even.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_to_breakeven


async def test_rule_interpreter_ignores_conditional_im_out_without_explicit_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="If this doesn't move lower now, I'm out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_turns_stay_heavy_or_im_out_into_move_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=24560.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=24595.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="We'd need to stay heavy at 180s or I'm out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_stop
    assert intent.stop_price == 24180.0


async def test_rule_interpreter_turns_if_we_hold_here_im_out_into_move_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=24550.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=24570.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="If we hold 30s here I'm out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.move_stop
    assert intent.stop_price == 24530.0


async def test_rule_interpreter_turns_split_if_we_hold_here_im_out_into_move_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=24550.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=24570.0),
    )

    first = TranscriptSegment(session_id=session.id, text="Like if we hold")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="30s here I'm out.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is not None
    assert second_intent.tag == ActionTag.move_stop
    assert second_intent.stop_price == 24530.0


async def test_rule_interpreter_marks_entry_with_recent_management_guard() -> None:
    interpreter = RuleBasedTradeInterpreter(entry_guard_window_ms=20_000)
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21243.75))

    management = TranscriptSegment(session_id=session.id, text="Paying myself a piece here.")
    management_intent = await interpreter.interpret(session, management)

    assert management_intent is not None
    assert management_intent.tag == ActionTag.trim

    entry = TranscriptSegment(session_id=session.id, text="Putting a little piece on short versus 44s.")
    entry_intent = await interpreter.interpret(session, entry)

    assert entry_intent is not None
    assert entry_intent.tag == ActionTag.enter_short
    assert entry_intent.guard_reason == "recent management cue detected"


async def test_rule_interpreter_treats_got_my_ad_on_as_add() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="NQ", last_price=21248.0),
        position=PositionState(
            side=TradeSide.short,
            quantity=1,
            average_price=21255.0,
            stop_price=21264.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="Got my ad on there.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.add
    assert intent.side == TradeSide.short
    assert intent.stop_price == 21264.0


async def test_rule_interpreter_detects_imperative_pay_yourself_trim_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25065.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=2,
            average_price=25030.0,
            stop_price=25010.0,
        ),
    )
    segment = TranscriptSegment(
        session_id=session.id,
        text="Come on, up you go. There's value area load. Pay yourself some here at 70s.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.trim
    assert intent.side == TradeSide.long
    assert intent.target_price == 25070.0


async def test_rule_interpreter_detects_you_can_pay_yourself() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=2,
            average_price=21220.0,
            stop_price=21190.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="You can pay yourself here if you want.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.trim


async def test_rule_interpreter_detects_you_can_take_profit() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=2,
            average_price=21220.0,
            stop_price=21190.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="You can take some profit here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.trim


async def test_rule_interpreter_detects_you_can_go_long_without_spoken_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="You can go long here if you like.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.stop_price is None


async def test_rule_interpreter_detects_you_can_go_short_without_spoken_stop() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="You can go short here, risk to the highs.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short


async def test_rule_interpreter_ignores_advisory_trim_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=2,
            average_price=21220.0,
            stop_price=21190.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="You can pay yourself here if you want.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_advisory_entry_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="NQ", last_price=21250.0))
    segment = TranscriptSegment(session_id=session.id, text="You can go short here, risk to the highs.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_detects_paid_myself_with_target_context() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=24962.5),
        position=PositionState(
            side=TradeSide.long,
            quantity=3,
            average_price=24940.0,
            stop_price=24840.0,
        ),
    )
    segment = TranscriptSegment(
        session_id=session.id,
        text=(
            "Yes, I'm in long, I already paid myself a bit into 60s, "
            "I'm looking for 80s, and then I'm looking for acceptance over 25,000 for high of day."
        ),
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.trim
    assert intent.side == TradeSide.long
    assert intent.target_price == 24980.0


async def test_rule_interpreter_detects_taking_some_off() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=2,
            average_price=21220.0,
            stop_price=21190.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="Taking some off here into the open.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.trim


async def test_rule_interpreter_detects_longing_this_as_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25081.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm now longing this.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long


async def test_rule_interpreter_detects_long_again_as_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25080.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm long again. Seeing if we can squeeze through 80s now.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long


async def test_rule_interpreter_detects_flattening_as_exit() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25020.0),
        position=PositionState(
            side=TradeSide.long,
            quantity=1,
            average_price=25000.0,
            stop_price=24980.0,
        ),
    )
    segment = TranscriptSegment(session_id=session.id, text="Again, I'm flattening this, folks.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.exit_all


def test_rule_interpreter_ignores_long_bias_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm long bias in this range, right?")

    intent = interpreter.interpret_partial(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_auctioned_flat_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="auctioned down. We auctioned flat. This week we've auctioned lower.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_stream_outro_out_of_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="All right, folks. I'm out of here. Big love.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_hypothetical_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="can't just be like, I'm short because it's gone up")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_short_question() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Was it this short here?")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_planning_short_question() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="How am I going to get short here now?")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_find_a_way_to_get_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Got to find a way to get short here now.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_negated_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Don't want to short here because you're in a support system.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_detects_im_in_short_as_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm in short.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short


async def test_rule_interpreter_detects_clean_bare_short_here_as_entry() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Short here at 40s, stop over 25050.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.entry_price == 25040.0
    assert intent.stop_price == 25050.0


async def test_rule_interpreter_stitches_split_entry_utterance_before_firing() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    first = TranscriptSegment(session_id=session.id, text="All right, long")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="here at 40s, stop under 30s.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is not None
    assert second_intent.tag == ActionTag.enter_long
    assert second_intent.entry_price == 25040.0
    assert second_intent.stop_price == 25030.0


async def test_rule_interpreter_ignores_fragmentary_short_here_artifact() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="got were short here. Evan, if you're")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_setup_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="We can look for this short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_split_setup_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Can look for this short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_planning_take_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'll think about taking a short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_huge_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="Again, we've had a huge short here, folks.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_i_was_short_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="So, I think I was short here, covered, short here, covered.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_advisory_adding_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="If you're adding short here you better have a big call position up here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_example_signal_phrase_split_across_segments() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    first = TranscriptSegment(session_id=session.id, text="You just want me to say I'm buying here,")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="I'm selling here.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_ignores_example_if_you_took_the_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If you took the short up here into 70, then you're still comfortable.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_would_i_consider_long_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Would I consider a long here? You could potentially play a long.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_i_said_you_could_be_long_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="We were short here and here and then I said you could be long here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_split_historical_i_was_short_from_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    first = TranscriptSegment(session_id=session.id, text="be long here. I'm I'm done with it. So,")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="but I was short from here.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_ignores_you_can_think_about_getting_short_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If you start to see lower highs put in now, you can think about getting short.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_might_be_able_to_put_a_piece_on_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Any pop that fail 75 and 90s, you might be able to put a piece on.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_hows_me_being_able_to_pay_myself_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="How's me being able to pay myself got anything to do with contracts?")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_not_paying_myself_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="Still not paying myself. Only in small size there.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_havent_stopped_out_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If they haven't stopped out, they're starting to be in the money.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_one_break_even_summary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="We had like one break even or small loss down there.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_i_took_the_short_here_to_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I took the short here to here, right? And made money.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_i_took_the_long_from_here_to_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I took the long from here to here. Right. And I stopped out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_post_fill_took_the_long_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I took the long out the gate on the squeeze.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_they_are_piling_in_short_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="They are piling in short here trying to look for this move.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_if_they_are_short_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If they are short here or here, they will be rushing to hit buy to get out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_if_you_are_short_here_example() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If you're short here, you should stop out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_if_i_was_you_trim_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=24980.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="I'd be trimming my positions if I was you.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_lets_say_youre_long_here_example() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If you, let's say you're long here looking for this, your stop needs to be underneath this swing low.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_split_lets_say_youre_selling_here_example() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    first = TranscriptSegment(session_id=session.id, text="Let's say you're selling")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is None

    second = TranscriptSegment(session_id=session.id, text="here. All right? And it might fill your")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_ignores_would_i_take_a_long_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Would I take a long here? Probably not.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_looking_for_the_short_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Looking for the short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_could_think_about_getting_long_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="If the next pullback holds 25s, you could think about getting long.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_think_about_buying_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm going to think about buying here or selling here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_could_argue_there_was_a_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="You could argue there was a short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_interested_in_the_short_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm interested in the short.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_tempted_to_put_piece_on_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Tempted to put a little piece on here just in case we start to go.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_could_try_a_cheap_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="You could try a cheap short here, but I don't love it.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_looking_to_get_piece_on_in_auto_mode() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'm looking to see if I can get a piece on here at the right time.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_almost_stopped_out_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="I almost stopped out here. I was a bit nervous there to be honest.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_stopped_out_on_what_i_was_looking_for() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )
    first = TranscriptSegment(session_id=session.id, text="I was in a short earlier. Paid myself some.")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is not None

    second = TranscriptSegment(session_id=session.id, text="Stopped out on what I was looking for the 400 point run at break even.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_rule_interpreter_ignores_if_you_had_another_go_after_stop_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Stopped out. But if you had another go, you would have got it.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_can_get_out_of_this_now_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="To get out of this now, you can. I still think it goes lower.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_chase_back_in_get_stopped_out_coaching() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="You are more likely to take profit straight away, chase back in, get stopped out, and panic.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_name_saying_im_out_quote() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text='Daniel saying, "I\'m out. Don\'t want to assume it would break the high."',
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_im_out_with_drawdown_summary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Wave Gaming, I'm out. I was down 1K. I'm now back green.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_im_flat_now_with_trade_recap() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="I'm flat now. So, I was short from this break and then took the long.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_all_out_now_named_chat_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="All out now Peter. Nice work if you held that long.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_oracle_holding_context_until_session_symbol_returns() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    foreign_context = TranscriptSegment(session_id=session.id, text="I'm still holding this long on Oracle.")
    foreign_intent = await interpreter.interpret(session, foreign_context)

    assert foreign_intent is None

    carryover = TranscriptSegment(session_id=session.id, text="You can see where I'm long from. Still holding this long.")
    carryover_intent = await interpreter.interpret(session, carryover)

    assert carryover_intent is None

    reset = TranscriptSegment(session_id=session.id, text="NQ. Let's get back into the market.")
    reset_intent = await interpreter.interpret(session, reset)

    assert reset_intent is None

    valid_entry = TranscriptSegment(session_id=session.id, text="I'm short here.")
    valid_intent = await interpreter.interpret(session, valid_entry)

    assert valid_intent is not None
    assert valid_intent.tag == ActionTag.enter_short


async def test_rule_interpreter_ignores_i_was_positioned_long_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="So I was positioned long here. Took some off and then moved on.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_i_was_long_again_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="And then I was long again.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_would_i_be_long_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Would I be long here? I said the level earlier.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_if_youre_longing_this_example() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="So, if you're longing this, you'd be risking this low.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_could_argue_theres_a_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Short here. You could argue there's a short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_if_youre_buying_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="You are trying to squeeze 200 points out of this market if you're buying here.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_could_think_about_a_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="You could think about a short here, but I don't really like what I'm seeing.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_historical_i_was_buying_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Uh I was buying here once we did this right for that move.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_i_was_short_then_i_took_the_long_history() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="I was short and then I took the long here and then stopped out.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_was_short_then_i_took_the_long_history_without_subject() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Was short and then I took the long here and then stopped out.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_you_saw_me_get_short_here_history() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Like you saw me get short here. So I was long here for the first move up.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_think_about_whos_playing_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Think about who's playing short here. People guessing tops all day long.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_dont_want_to_be_entering_long_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="You don't want to be entering long here. Remember, we're long from down here.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_i_wouldnt_be_adding_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="I wouldn't be adding short here, for example, but if you're still holding you should be good.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_dont_want_to_be_adding_short_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="You do not want to be adding short here. If you're going to do that, you should already be short.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_stopped_out_if_you_think_about_this_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Stopped out. All right, if you think about this, his stop was here.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_stopped_out_in_profit_but_i_was_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Stopped out in profit, but I was looking for the bigger rotation back down.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_actually_got_stopped_out_so_i_was_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="I actually got stopped out. So I was trying to hold for the bigger move.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_tempted_to_try_another_long_here() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="Tempted to try another long here to get up into those highs.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_runner_break_even_i_was_looking_for_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Cut the runner break even. I was looking for the same thing after that first attempt.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_ignored_short_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="I'm ignoring the short, right? Just like I ignored the short here. I'm waiting for the pullback.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_anyone_who_got_short_here_commentary() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(
        session_id=session.id,
        text="Anyone who got short here has become the pain to move this market higher.",
    )

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_suppresses_foreign_instrument_context_until_session_symbol_returns() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    foreign_context = TranscriptSegment(session_id=session.id, text="UNH. All right. I'm still in this long waiting to see if this is ready to go.")
    foreign_intent = await interpreter.interpret(session, foreign_context)

    assert foreign_intent is None

    carryover = TranscriptSegment(session_id=session.id, text="So, I'm positioned long here looking for this move here.")
    carryover_intent = await interpreter.interpret(session, carryover)

    assert carryover_intent is None

    reset = TranscriptSegment(session_id=session.id, text="The NQ in my head.")
    reset_intent = await interpreter.interpret(session, reset)

    assert reset_intent is None

    valid_entry = TranscriptSegment(session_id=session.id, text="I'm long here.")
    valid_intent = await interpreter.interpret(session, valid_entry)

    assert valid_intent is not None
    assert valid_intent.tag == ActionTag.enter_long


async def test_rule_interpreter_suppresses_named_stock_context_until_session_symbol_returns() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    foreign_context = TranscriptSegment(session_id=session.id, text="I've been long from down here on Netflix.")
    foreign_intent = await interpreter.interpret(session, foreign_context)

    assert foreign_intent is None

    carryover = TranscriptSegment(session_id=session.id, text="Position for a multi-year long here.")
    carryover_intent = await interpreter.interpret(session, carryover)

    assert carryover_intent is None

    reset = TranscriptSegment(session_id=session.id, text="NQ. Let's get back into the market that we're trading actively today.")
    reset_intent = await interpreter.interpret(session, reset)

    assert reset_intent is None

    valid_entry = TranscriptSegment(session_id=session.id, text="I'm short here.")
    valid_intent = await interpreter.interpret(session, valid_entry)

    assert valid_intent is not None
    assert valid_intent.tag == ActionTag.enter_short


async def test_rule_interpreter_suppresses_inline_foreign_instrument_context_until_session_symbol_returns() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    foreign_context = TranscriptSegment(session_id=session.id, text="Started getting long in PayPal last month.")
    foreign_intent = await interpreter.interpret(session, foreign_context)

    assert foreign_intent is None

    carryover = TranscriptSegment(session_id=session.id, text="I'm long here and just holding this for now.")
    carryover_intent = await interpreter.interpret(session, carryover)

    assert carryover_intent is None

    reset = TranscriptSegment(session_id=session.id, text="NQ. Let's get back into the market.")
    reset_intent = await interpreter.interpret(session, reset)

    assert reset_intent is None

    valid_entry = TranscriptSegment(session_id=session.id, text="I'm short here.")
    valid_intent = await interpreter.interpret(session, valid_entry)

    assert valid_intent is not None
    assert valid_intent.tag == ActionTag.enter_short


async def test_rule_interpreter_suppresses_swing_context_until_session_symbol_returns() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    foreign_context = TranscriptSegment(session_id=session.id, text="I'm long on my swings, Matt. Yeah.")
    foreign_intent = await interpreter.interpret(session, foreign_context)

    assert foreign_intent is None

    carryover = TranscriptSegment(session_id=session.id, text="I'm long in all the software companies we've been talking about.")
    carryover_intent = await interpreter.interpret(session, carryover)

    assert carryover_intent is None

    reset = TranscriptSegment(session_id=session.id, text="NQ. Let's get back into the market that we're trading actively today.")
    reset_intent = await interpreter.interpret(session, reset)

    assert reset_intent is None

    valid_entry = TranscriptSegment(session_id=session.id, text="I'm short here.")
    valid_intent = await interpreter.interpret(session, valid_entry)

    assert valid_intent is not None
    assert valid_intent.tag == ActionTag.enter_short


async def test_rule_interpreter_ignores_stream_outro_out_there() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'll read the last few questions I'm out there.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_ignores_stream_outro_split_out_there() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))
    segment = TranscriptSegment(session_id=session.id, text="I'll read the last few questions I'm out.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None


async def test_rule_interpreter_dedupes_entry_seed_followup_as_duplicate() -> None:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0))

    first = TranscriptSegment(session_id=session.id, text="Feeler on short here.")
    first_intent = await interpreter.interpret(session, first)

    assert first_intent is not None
    assert first_intent.tag == ActionTag.enter_short

    session.position = PositionState(side=TradeSide.short, quantity=1, average_price=25000.0)
    second = TranscriptSegment(session_id=session.id, text="Just a feeler on.")
    second_intent = await interpreter.interpret(session, second)

    assert second_intent is None


async def test_embedding_gate_routes_novel_phrase_to_fallback() -> None:
    class _AlwaysTradeGate:
        def is_trade_relevant(self, text: str) -> bool:
            return "he enters now" in text

    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()
            self.confirm_intent = AsyncMock()

        def is_available(self) -> bool:
            return True

    mock_fallback = _Fallback()
    mock_fallback.interpret.return_value = TradeIntent(
        session_id="test",
        tag=ActionTag.enter_long,
        symbol="NQ",
        side=TradeSide.long,
        confidence=0.82,
        evidence_text="okay he enters now",
        source_latency_ms=0,
    )
    interpreter = RuleBasedTradeInterpreter(fallback=mock_fallback, embedding_gate=_AlwaysTradeGate())
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="NQ", last_price=21250.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="okay he enters now")

    intent = await interpreter.interpret(session, segment)

    mock_fallback.interpret.assert_called_once()
    assert intent is not None
    assert intent.tag == ActionTag.enter_long


async def test_rule_interpreter_uses_gemini_confirmation_to_veto_auto_entry() -> None:
    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()
            self.confirm_intent = AsyncMock()

        def is_available(self) -> bool:
            return True

    mock_fallback = _Fallback()
    mock_fallback.confirm_intent.return_value.confirmed = False
    interpreter = RuleBasedTradeInterpreter(fallback=mock_fallback)
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.auto),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="I'm short here.")

    intent = await interpreter.interpret(session, segment)

    assert intent is None
    mock_fallback.confirm_intent.assert_called_once()
    mock_fallback.interpret.assert_not_called()


def test_compact_repeated_trade_text_collapses_stt_loops() -> None:
    compacted = _compact_repeated_trade_text(
        "So I'm long versus 35 right now. We lose 35. We lose 35. We lose 35. We need to hold. We need to hold."
    )

    assert compacted.count("We lose 35.") == 1
    assert compacted.count("We need to hold.") == 1


async def test_rule_interpreter_allows_strong_entry_when_gemini_confirmation_times_out() -> None:
    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()

        def is_available(self) -> bool:
            return True

        async def confirm_intent(self, **_kwargs):
            await asyncio.sleep(0.2)

    interpreter = RuleBasedTradeInterpreter(fallback=_Fallback(), fallback_confirmation_timeout_ms=1)
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.auto),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5),
    )
    segment = TranscriptSegment(
        session_id=session.id,
        text="So I'm long versus 35 right now. We lose 35. We lose 35. We lose 35.",
    )

    intent = await interpreter.interpret(session, segment)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert intent is not None
    assert intent.tag == ActionTag.enter_long
    assert intent.stop_price == 25035.0
    assert diagnostic is not None
    assert diagnostic.message == "gemini confirmation timed out; executed on strong rule signal."


async def test_rule_interpreter_blocks_non_strong_entry_when_gemini_confirmation_times_out() -> None:
    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()

        def is_available(self) -> bool:
            return True

        async def confirm_intent(self, **_kwargs):
            await asyncio.sleep(0.2)

    interpreter = RuleBasedTradeInterpreter(fallback=_Fallback(), fallback_confirmation_timeout_ms=1)
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.auto),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5),
    )
    segment = TranscriptSegment(session_id=session.id, text="I'm short here.")

    intent = await interpreter.interpret(session, segment)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert intent is None
    assert diagnostic is not None
    assert diagnostic.title == "Entry confirmation blocked"


async def test_rule_interpreter_keeps_review_mode_extractive_fallback_behavior() -> None:
    class _AlwaysTradeGate:
        def is_trade_relevant(self, text: str) -> bool:
            return True

    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()
            self.confirm_intent = AsyncMock()

        def is_available(self) -> bool:
            return True

    mock_fallback = _Fallback()
    mock_fallback.interpret.return_value = TradeIntent(
        session_id="test",
        tag=ActionTag.enter_short,
        symbol="MNQ 03-26",
        side=TradeSide.short,
        confidence=0.84,
        evidence_text="okay he enters now",
        source_latency_ms=0,
    )
    interpreter = RuleBasedTradeInterpreter(fallback=mock_fallback, embedding_gate=_AlwaysTradeGate())
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.review),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="okay he enters now")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    mock_fallback.interpret.assert_called_once()


async def test_rule_interpreter_does_not_use_extractive_fallback_in_auto_mode_without_rule_intent() -> None:
    class _AlwaysTradeGate:
        def is_trade_relevant(self, text: str) -> bool:
            return True

    class _Fallback:
        def __init__(self) -> None:
            self.interpret = AsyncMock()
            self.confirm_intent = AsyncMock()

        def is_available(self) -> bool:
            return True

    mock_fallback = _Fallback()
    mock_fallback.interpret.return_value = TradeIntent(
        session_id="test",
        tag=ActionTag.enter_long,
        symbol="MNQ 03-26",
        side=TradeSide.long,
        confidence=0.84,
        evidence_text="okay he enters now",
        source_latency_ms=0,
    )
    interpreter = RuleBasedTradeInterpreter(fallback=mock_fallback, embedding_gate=_AlwaysTradeGate())
    session = StreamSession(
        config=SessionConfig(enable_ai_fallback=True, execution_mode=ExecutionMode.auto),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="okay he enters now")

    intent = await interpreter.interpret(session, segment)

    assert intent is None
    mock_fallback.interpret.assert_not_called()


async def test_rule_interpreter_local_classifier_blocks_weak_bare_entry() -> None:
    from app.services.interpretation.local_classifier import IntentClassifierPrediction

    prediction = IntentClassifierPrediction(
        tag=ActionTag.no_action,
        confidence=0.91,
        probabilities={
            ActionTag.no_action: 0.91,
            ActionTag.enter_long: 0.07,
            ActionTag.enter_short: 0.01,
            ActionTag.add: 0.01,
        },
        thresholds={ActionTag.enter_long: 0.55},
        model_name="test-local-classifier",
    )
    interpreter = RuleBasedTradeInterpreter(local_classifier=_FakeLocalClassifier(prediction))
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5))
    segment = TranscriptSegment(session_id=session.id, text="Long here.")

    intent = await interpreter.interpret(session, segment)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert intent is None
    assert diagnostic is not None
    assert diagnostic.title == "Local classifier blocked intent"


async def test_rule_interpreter_local_classifier_recovers_collective_short_entry() -> None:
    from app.services.interpretation.local_classifier import IntentClassifierPrediction

    prediction = IntentClassifierPrediction(
        tag=ActionTag.enter_short,
        confidence=0.92,
        probabilities={
            ActionTag.enter_short: 0.92,
            ActionTag.no_action: 0.03,
            ActionTag.add: 0.03,
            ActionTag.setup_short: 0.02,
        },
        thresholds={ActionTag.enter_short: 0.58},
        model_name="test-local-classifier",
    )
    interpreter = RuleBasedTradeInterpreter(local_classifier=_FakeLocalClassifier(prediction))
    session = StreamSession(config=SessionConfig(), market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5))
    segment = TranscriptSegment(session_id=session.id, text="We're short versus 90s right here.")

    intent = await interpreter.interpret(session, segment)
    diagnostic = interpreter.consume_diagnostic(session.id)

    assert intent is not None
    assert intent.tag == ActionTag.enter_short
    assert intent.side == TradeSide.short
    assert intent.stop_price == 25090.0
    assert diagnostic is not None
    assert diagnostic.title == "Local classifier promoted intent"


async def test_rule_interpreter_local_classifier_recovers_collective_exit() -> None:
    from app.services.interpretation.local_classifier import IntentClassifierPrediction

    prediction = IntentClassifierPrediction(
        tag=ActionTag.exit_all,
        confidence=0.89,
        probabilities={
            ActionTag.exit_all: 0.89,
            ActionTag.no_action: 0.08,
            ActionTag.trim: 0.03,
        },
        thresholds={ActionTag.exit_all: 0.55},
        model_name="test-local-classifier",
    )
    interpreter = RuleBasedTradeInterpreter(local_classifier=_FakeLocalClassifier(prediction))
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25045.5),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25020.0),
    )
    segment = TranscriptSegment(session_id=session.id, text="We're out of this now.")

    intent = await interpreter.interpret(session, segment)

    assert intent is not None
    assert intent.tag == ActionTag.exit_all


# ---------------------------------------------------------------------------
# Cross-segment stitching tests — all action types
# ---------------------------------------------------------------------------


async def test_cross_segment_exit_i_m_then_out() -> None:
    """'i m' at end of segment 1, 'out' at start of segment 2 → exit_all."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=24980.0),
    )

    first = TranscriptSegment(session_id=session.id, text="all right so I'm")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="out of this now")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.exit_all


async def test_cross_segment_exit_all_then_out() -> None:
    """'all' at end of segment 1, 'out' at start of segment 2 → exit_all."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.short, quantity=1, average_price=25020.0),
    )

    first = TranscriptSegment(session_id=session.id, text="yeah so we are all")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="out of this one")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.exit_all


async def test_cross_segment_exit_i_m_then_flat() -> None:
    """'i m' at end of segment 1, 'flat' at start of segment 2 → exit_all."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=24980.0),
    )

    first = TranscriptSegment(session_id=session.id, text="that s it I'm")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="flat on this trade")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.exit_all


async def test_cross_segment_trim_paying_then_myself() -> None:
    """'paying' at end of segment 1, 'myself' at start of segment 2 → trim."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25050.0),
        position=PositionState(side=TradeSide.long, quantity=2, average_price=25000.0),
    )

    first = TranscriptSegment(session_id=session.id, text="all right so I'm paying")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="myself some here")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.trim


async def test_cross_segment_trim_taking_then_some_off() -> None:
    """'taking' at end of segment 1, 'some off' at start of segment 2 → trim."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25050.0),
        position=PositionState(side=TradeSide.short, quantity=2, average_price=25080.0),
    )

    first = TranscriptSegment(session_id=session.id, text="right so I'm taking")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="some off into this move")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.trim


async def test_cross_segment_breakeven_stop_then_breakeven() -> None:
    """'stop' at end of segment 1, 'breakeven' at start of segment 2 → move_to_breakeven."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25050.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25000.0),
    )

    first = TranscriptSegment(session_id=session.id, text="moving my stop")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="to breakeven now")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.move_to_breakeven


async def test_cross_segment_breakeven_break_then_even() -> None:
    """'break' at end of segment 1, 'even' at start of segment 2 → move_to_breakeven."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25050.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25000.0),
    )

    first = TranscriptSegment(session_id=session.id, text="all right stop to break")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="even on this one")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.move_to_breakeven


async def test_cross_segment_move_stop_moving_then_stop() -> None:
    """'moving' at end of segment 1, 'my stop' + price in segment 2 → move_stop."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25050.0),
        position=PositionState(side=TradeSide.long, quantity=1, average_price=25000.0),
    )

    first = TranscriptSegment(session_id=session.id, text="all right I'm moving")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="my stop to 25 now")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.move_stop


async def test_cross_segment_entry_in_then_this_short() -> None:
    """'in' at end of segment 1, 'this short' at start of segment 2 → enter_short."""
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(),
        market=MarketSnapshot(symbol="MNQ 03-26", last_price=25000.0),
    )

    first = TranscriptSegment(session_id=session.id, text="all right I'm in")
    assert await interpreter.interpret(session, first) is None

    second = TranscriptSegment(session_id=session.id, text="this short versus 10s")
    intent = await interpreter.interpret(session, second)
    assert intent is not None
    assert intent.tag == ActionTag.enter_short
