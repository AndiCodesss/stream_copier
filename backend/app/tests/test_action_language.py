from __future__ import annotations

from app.models.domain import ActionTag, TradeSide
from app.services.interpretation.action_language import detect_present_trade_signal, detect_setup_signal


# --- Collective short entry ---

def test_action_language_detects_collective_short_entry() -> None:
    signal = detect_present_trade_signal("we re short versus 90s right here", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short
    assert signal.side == TradeSide.short


def test_action_language_rejects_historical_recap() -> None:
    signal = detect_present_trade_signal("remember we re short from 80s", position_side=None)

    assert signal is None


def test_action_language_detects_i_m_in_this_short() -> None:
    signal = detect_present_trade_signal("i m in this short now", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short
    assert signal.side == TradeSide.short


def test_action_language_detects_bare_leadin_in_this_short() -> None:
    signal = detect_present_trade_signal("let s see then folks in this short", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short
    assert signal.side == TradeSide.short


# --- Setup signals ---

def test_action_language_detects_setup_short() -> None:
    signal = detect_setup_signal("looking for the short here if we fail")

    assert signal is not None
    assert signal.tag == ActionTag.setup_short


def test_setup_watching_shorts() -> None:
    signal = detect_setup_signal("watching shorts now")

    assert signal is not None
    assert signal.tag == ActionTag.setup_short


def test_setup_focused_on_shorts() -> None:
    signal = detect_setup_signal("i m focused on the shorts until")

    assert signal is not None
    assert signal.tag == ActionTag.setup_short


def test_setup_looking_for_downside() -> None:
    signal = detect_setup_signal("looking for downside pressure")

    assert signal is not None
    assert signal.tag == ActionTag.setup_short


def test_setup_looking_for_longs() -> None:
    signal = detect_setup_signal("looking for longs just to fill")

    assert signal is not None
    assert signal.tag == ActionTag.setup_long


# --- Self long patterns ---

def test_long_small_long_now() -> None:
    signal = detect_present_trade_signal("small long now here we go", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long
    assert signal.side == TradeSide.long


def test_long_in_long_here() -> None:
    signal = detect_present_trade_signal("in long here just for a scalp", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_long_from_here() -> None:
    signal = detect_present_trade_signal("we re long from here some of you were really good", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_long_i_m_long() -> None:
    signal = detect_present_trade_signal("i m long from 20s", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_long_at_the_moment() -> None:
    signal = detect_present_trade_signal("so long at the moment now", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_small_sizing_this_long() -> None:
    signal = detect_present_trade_signal("i am small sizing this long", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


# --- Self short patterns ---

def test_short_from_here() -> None:
    signal = detect_present_trade_signal("so short from here on this move", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short


def test_short_i_m_in_a_short() -> None:
    signal = detect_present_trade_signal("i m in a short now small size", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short


def test_short_i_m_short_on() -> None:
    signal = detect_present_trade_signal("i m short on this for a small piece", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short


def test_short_i_m_short_versus() -> None:
    signal = detect_present_trade_signal("i m short versus vwap", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_short


# --- Side-neutral entry patterns ---

def test_neutral_piece_here() -> None:
    signal = detect_present_trade_signal("through the 40s in a small piece here", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long
    assert signal.source == "neutral_entry"


def test_neutral_feathering_in() -> None:
    signal = detect_present_trade_signal("all right feathering in now", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_got_a_piece_on() -> None:
    signal = detect_present_trade_signal("got a little piece on this let s see", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_with_position_returns_add() -> None:
    signal = detect_present_trade_signal("got a piece on here", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.add
    assert signal.side == TradeSide.long


def test_neutral_i_m_in_this_now() -> None:
    signal = detect_present_trade_signal("i m in this now small size", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_i_m_in() -> None:
    signal = detect_present_trade_signal("going we re going i m in", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_starter_on_here() -> None:
    signal = detect_present_trade_signal("there s the break a starter on here", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_i_ve_entered() -> None:
    signal = detect_present_trade_signal("i ve re entered versus flow zones now", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_i_did_enter() -> None:
    signal = detect_present_trade_signal("i did enter short versus new york here", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


def test_neutral_forcing_me_to_join() -> None:
    signal = detect_present_trade_signal("forcing me to join in there once it broke", position_side=None)

    assert signal is not None
    assert signal.tag == ActionTag.enter_long


# --- EXIT_ALL patterns ---

def test_exit_i_m_out() -> None:
    signal = detect_present_trade_signal("i m out now waiting for a reload", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_taking_this_off() -> None:
    signal = detect_present_trade_signal("taking this off we can reload if it offers something", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_fully_out() -> None:
    signal = detect_present_trade_signal("fully out of this now", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_stopped_out() -> None:
    signal = detect_present_trade_signal("stopped out on this", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_knocked_out() -> None:
    signal = detect_present_trade_signal("okay knocked out be", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_i_m_done() -> None:
    signal = detect_present_trade_signal("right i m done take this off", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_done_with_this() -> None:
    signal = detect_present_trade_signal("i m done with this long now", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_cut_this() -> None:
    signal = detect_present_trade_signal("all right cut this i can always reassess", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_flat_now() -> None:
    signal = detect_present_trade_signal("i m flat now and we ll just wait", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_job_done() -> None:
    signal = detect_present_trade_signal("here s our target i m out now job done", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


def test_exit_i_ve_cut() -> None:
    signal = detect_present_trade_signal("i do not like that i ve cut", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.exit_all


# --- TRIM patterns ---

def test_trim_taking_partial() -> None:
    signal = detect_present_trade_signal("taking partial here folks hold runners", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_tp_here() -> None:
    signal = detect_present_trade_signal("all right tp here into this vw up", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_peeling_some_off() -> None:
    signal = detect_present_trade_signal("peeling some off here you can hear me on my hot keys", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_covering_some() -> None:
    signal = detect_present_trade_signal("covering some into these deep pullbacks", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_paid_myself() -> None:
    signal = detect_present_trade_signal("paid myself a tiny bit i m up 300 bucks", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_locking_some_in() -> None:
    signal = detect_present_trade_signal("already locking in some money there", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_scalping_some_out() -> None:
    signal = detect_present_trade_signal("scalping some out into flow zones here", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_little_piece_off() -> None:
    signal = detect_present_trade_signal("i ll take a little piece off here", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_partial_here() -> None:
    signal = detect_present_trade_signal("partial here folks hold onto your runners", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_taking_some_off() -> None:
    signal = detect_present_trade_signal("taking some off straight away i don t like it", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_trimming_some() -> None:
    signal = detect_present_trade_signal("trimming some there", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.trim


def test_trim_selling_some() -> None:
    signal = detect_present_trade_signal("selling some into the vwap", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.trim


# --- MOVE_STOP patterns ---

def test_move_stop_moving_my_stop() -> None:
    signal = detect_present_trade_signal("moving my stop up now", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.move_stop


def test_move_stop_trailing() -> None:
    signal = detect_present_trade_signal("starting to trail this now", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.move_stop


def test_move_stop_stops_moving_down() -> None:
    signal = detect_present_trade_signal("stops moving down i m going to trail more aggressively", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.move_stop


# --- BREAKEVEN patterns ---

def test_breakeven_stop_break_even() -> None:
    signal = detect_present_trade_signal("and move my stop into break even now", position_side=TradeSide.long)

    assert signal is not None
    assert signal.tag == ActionTag.move_to_breakeven


def test_breakeven_stop_in_the_money() -> None:
    signal = detect_present_trade_signal("stop in the money now holding runners", position_side=TradeSide.short)

    assert signal is not None
    assert signal.tag == ActionTag.move_to_breakeven


# --- Hypothetical rejection ---

def test_hypothetical_looking_for_rejected() -> None:
    signal = detect_present_trade_signal("looking for a trigger long if they re going to let this start", position_side=None)

    assert signal is None


def test_hypothetical_want_to_rejected() -> None:
    signal = detect_present_trade_signal("want to see a flush first", position_side=None)

    assert signal is None
