"""Regex pattern library for detecting trade signals in spoken language.

This module is a curated dictionary of regular expressions organized by
action type: entries (long/short), exits, trims, stop moves, breakeven,
and setups. Each pattern list captures the natural phrases a futures trader
uses when announcing live actions (e.g. "i m long", "stopped out", "paying
myself"). The module also filters out historical and hypothetical speech
so that past trades and "what if" scenarios are not mistaken for live actions.

The detection pipeline works in a fixed priority order:
  historical filter -> setup detection -> hypothetical filter ->
  exit -> trim -> breakeven -> stop move -> long entry -> short entry ->
  side-neutral entry
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.domain import ActionTag, TradeSide

_ACTIONABLE_LABELS = {
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.add,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
}

# Phrases that indicate the speaker is describing a past trade, not a live one.
_HISTORICAL_PATTERNS = [
    r"\bremember\b",
    r"\byesterday\b",
    r"\bthis morning\b",
    r"\bearlier\b",
    r"\blast (?:week|trade|session)\b",
    r"\bon stream\b",
    r"\bi was\b",
    r"\bwe were\b",
    r"\bi took\b",
    r"\bwe took\b",
    r"\bi had the\b",
    r"\bwe had the\b",
]

# Phrases that indicate speculation or advice, not a committed action.
_HYPOTHETICAL_PATTERNS = [
    r"\bif you\b",
    r"\bif we\b",
    r"\bif i\b",
    r"\blet s say\b",
    r"\bfor example\b",
    r"\bwould\b",
    r"\bcould\b",
    r"\bshould\b",
    r"\blooking for\b",
    r"\bwatching for\b",
    r"\binterested in\b",
    r"\btempted\b",
    r"\bwant to\b",
    r"\bnot going to\b",
    r"\bnot entering\b",
    r"\bnot interested\b",
]

# Setup patterns -- the trader is watching for an opportunity, not yet in.
_SETUP_LONG_PATTERNS = [
    r"\blooking for (?:the |a )?long\b",
    r"\bwatching for (?:the |a )?long\b",
    r"\binterested in (?:the |a )?long\b",
    r"\btempted to (?:try another |take a )?long\b",
    r"\blooking for longs\b",
    r"\bfocused on the longs\b",
]

_SETUP_SHORT_PATTERNS = [
    r"\blooking for (?:the |a )?short\b",
    r"\bwatching for (?:the |a )?short\b",
    r"\binterested in (?:the |a )?short\b",
    r"\btempted to (?:try another |take a )?short\b",
    r"\blooking for shorts\b",
    r"\bwatching shorts\b",
    r"\bfocused on the shorts\b",
    r"\binterested in shorts\b",
    r"\bwaiting to see if a short sets up\b",
    r"\blooking for (?:a )?short entry\b",
    r"\blooking for downside\b",
]

# First-person long entry phrases (e.g. "we re long", "i m in this long").
_SELF_LONG_PATTERNS = [
    r"\bwe re long\b(?!\s+bias)",
    r"\bwe are long\b(?!\s+bias)",
    r"\bwe re in this long\b",
    r"\bwe are in this long\b",
    r"\bwe re buying\b",
    r"\bwe are buying\b",
    r"\bi m in this long\b",
    r"\bi am in this long\b",
    r"\bi m in long\b",
    r"\bi am in long\b",
    r"\bi ve got\b[^.]{0,80}\blong side\b",
    r"\bi have got\b[^.]{0,80}\blong side\b",
    r"\bpiece long on now\b",
    r"\blittle piece long here\b",
    r"\bsmall long (?:now|here|in this)\b",
    r"\blong from here\b",
    r"\blong from \d+s?\b",
    r"\bin long (?:here|now|from)\b",
    r"\blong(?:ed)? out\b",
    r"\bsmall long in this\b",
    r"\bsmall siz(?:e|ing) (?:this |in )?long\b",
    r"\blong at the moment\b",
    r"\bin this long\b",
]

# First-person short entry phrases (e.g. "we re short", "i m in this short").
_SELF_SHORT_PATTERNS = [
    r"\bwe re short\b(?!\s+bias)",
    r"\bwe are short\b(?!\s+bias)",
    r"\bwe re in this short\b",
    r"\bwe are in this short\b",
    r"\bwe re selling\b",
    r"\bwe are selling\b",
    r"\bi m in this short\b",
    r"\bi am in this short\b",
    r"\bi m in a short\b",
    r"\bi am in a short\b",
    r"\bi m short (?:from|in|on|versus) ",
    r"\bi am short (?:from|in|on|versus) ",
    r"\bi ve got\b[^.]{0,80}\bshort side\b",
    r"\bi have got\b[^.]{0,80}\bshort side\b",
    r"\bpiece short side on now\b",
    r"\bsmall piece short side on now\b",
    r"\bshort from here\b",
    r"\bshort from \d+s?\b",
    r"\bshort now\b(?!.*\blooking\b)",
    r"\bin (?:the |this )?short now\b",
    r"\bsmall (?:piece )?short here\b",
    r"\bi ve sold\b",
    r"\bi have sold\b",
    r"\bin this short\b(?!\s+(?:term|period|time))",
]

_EXIT_PATTERNS = [
    r"\bwe re out\b",
    r"\bwe are out\b",
    r"\bwe re flat\b",
    r"\bwe are flat\b",
    r"\bwe re out of this now\b",
    r"\bwe are out of this now\b",
    r"\bi m out\b",
    r"\bi am out\b",
    r"\bi m flat\b",
    r"\bi am flat\b",
    r"\bi m done\b",
    r"\bi am done\b",
    r"\bfully out\b",
    r"\ball out of this\b",
    r"\bout of this (?:now|position)\b",
    r"\bout of that position\b",
    r"\bout here\b(?!.*\blooking\b)",
    r"\bout on (?:that|the|this)\b",
    r"\btaking this off\b",
    r"\btake this off\b",
    r"\bcut this\b",
    r"\bi ve cut\b",
    r"\bi have cut\b",
    r"\bcut (?:my |the )?(?:position|rest)\b",
    r"\bknocked out\b",
    r"\bstopped out\b",
    r"\bstopping out\b",
    r"\bjob done\b",
    r"\bdone with (?:this|that|it)\b",
    r"\bflat (?:now|right now)\b",
]

_TRIM_PATTERNS = [
    r"\bwe re trimming\b",
    r"\bwe are trimming\b",
    r"\bwe re paying ourselves\b",
    r"\bwe are paying ourselves\b",
    r"\bcovering (?:a |some |one )?(?:little |small )?(?:piece|bit)\b",
    r"\bcovering (?:some|one|more)\b",
    r"\bcover (?:some|one|a little)\b",
    r"\btrimming (?:some|more)\b",
    r"\btaking (?:some|a little|another|more) (?:off|partials?|here|partial)\b",
    r"\btaking (?:a )?partial(?:s)?\b",
    r"\bpartial (?:here|into|now)\b",
    r"\btp(?:1|2|s)? here\b",
    r"\btp(?:1|2|s)? hit\b",
    r"\bpeeling (?:some|off)\b",
    r"\bscalping (?:some|a bit) out\b",
    r"\blocking (?:some|more|in)\b",
    r"\bpaid myself\b",
    r"\bpaying (?:a little|myself)\b",
    r"\btake (?:a )?(?:little )?piece off\b",
    r"\blittle piece off\b",
    r"\bselling (?:some|into)\b",
    r"\balready (?:covering|peeling|locking|selling)\b",
    r"\bdown to tiny size\b",
]

_BREAKEVEN_PATTERNS = [
    r"\bstop now breakeven\b",
    r"\bstop breakeven\b",
    r"\bmove(?:d|ing)? stop to breakeven\b",
    r"\bmove(?:d|ing)? my stop to breakeven\b",
    r"\bstop(?:s)? (?:into |to )?break ?even\b",
    r"\bbreak ?even stop\b",
    r"\bstop(?:s)? (?:now )?in(?:to)? the money\b",
]

_MOVE_STOP_PATTERNS = [
    r"\b(?:i|we)\s+(?:ll|will)\s+cut\b",
    r"\bstop(?: s| is)?\s+(?:at|to|under|over|below|above)\b",
    r"\bstops? moved to\b",
    r"\bmove(?:d|ing)? (?:my )?stop\b",
    r"\bmoving (?:my )?stop\b",
    r"\bstop(?:s)? (?:is )?(?:now )?(?:going to |at )\d+\b",
    r"\btrail(?:ing)? (?:this|more)\b",
    r"\bstop(?:s)? moving (?:down|up)\b",
    r"\bput (?:it |my stop )at \d+\b",
]

# Side-neutral entry patterns -- side is inferred from position or classifier.
_SELF_ENTRY_PATTERNS = [
    r"\b(?:small |a )?piece (?:in |on )?here\b",
    r"\bsmall size (?:in )?here\b",
    r"\bfeather(?:ing|ed) in\b",
    r"\bgot (?:a |my )?(?:small |little )?piece on\b",
    r"\bstarting in\b",
    r"\bi m buying here\b",
    r"\bi am buying here\b",
    r"\bi m in for\b",
    r"\bi am in for\b",
    r"\bi m in this now\b",
    r"\bi am in this now\b",
    r"\bi m in this\b",
    r"\bi am in this\b",
    r"\bi m in\b(?!\s+(?:runners?|the))",
    r"\bi am in\b(?!\s+(?:runners?|the))",
    r"\bi m playing this\b",
    r"\bi am playing this\b",
    r"\bhere we go\b[^.]{0,40}\b(?:small|piece|in)\b",
    r"\bforcing me to (?:join|get in)\b",
    r"\bforced me to join\b",
    r"\brejoining this\b",
    r"\bi ve feathered in\b",
    r"\bi have feathered in\b",
    r"\bsmall position\b",
    r"\bi ve (?:re ?)?entered\b",
    r"\bi have (?:re ?)?entered\b",
    r"\bi did enter\b",
    r"\bsmall piece on here\b",
    r"\blittle piece on this\b",
    r"\ba starter on here\b",
    r"\bstarter on\b",
]


@dataclass(frozen=True)
class PhraseSignal:
    """A detected trade phrase with its action type and inferred side."""

    tag: ActionTag
    side: TradeSide | None = None
    source: str = "pattern"

    @property
    def actionable(self) -> bool:
        return self.tag in _ACTIONABLE_LABELS


def is_historical_trade_context(text: str) -> bool:
    return _matches_any(_HISTORICAL_PATTERNS, text)


def is_hypothetical_trade_context(text: str) -> bool:
    return _matches_any(_HYPOTHETICAL_PATTERNS, text)


def detect_setup_signal(text: str) -> PhraseSignal | None:
    if _matches_any(_SETUP_LONG_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.setup_long, side=TradeSide.long, source="setup_long")
    if _matches_any(_SETUP_SHORT_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.setup_short, side=TradeSide.short, source="setup_short")
    return None


def detect_present_trade_signal(text: str, *, position_side: TradeSide | None) -> PhraseSignal | None:
    """Scan text for a live trade action phrase, filtering out non-actionable speech.

    Returns the first matching PhraseSignal in priority order (exit before
    trim before entry), or None if no actionable language is found.
    """
    if not text.strip():
        return None
    if is_historical_trade_context(text):
        return None

    setup_signal = detect_setup_signal(text)
    if setup_signal is not None:
        return setup_signal
    if is_hypothetical_trade_context(text):
        return None

    if _matches_any(_EXIT_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.exit_all, side=position_side, source="collective_exit")
    if _matches_any(_TRIM_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.trim, side=position_side, source="collective_trim")
    if _matches_any(_BREAKEVEN_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.move_to_breakeven, side=position_side, source="breakeven")
    if _matches_any(_MOVE_STOP_PATTERNS, text):
        return PhraseSignal(tag=ActionTag.move_stop, side=position_side, source="move_stop")
    if _matches_any(_SELF_LONG_PATTERNS, text):
        if position_side == TradeSide.long:
            return PhraseSignal(tag=ActionTag.add, side=TradeSide.long, source="collective_long_add")
        return PhraseSignal(tag=ActionTag.enter_long, side=TradeSide.long, source="collective_long_entry")
    if _matches_any(_SELF_SHORT_PATTERNS, text):
        if position_side == TradeSide.short:
            return PhraseSignal(tag=ActionTag.add, side=TradeSide.short, source="collective_short_add")
        return PhraseSignal(tag=ActionTag.enter_short, side=TradeSide.short, source="collective_short_entry")
    if _matches_any(_SELF_ENTRY_PATTERNS, text):
        if position_side is not None:
            return PhraseSignal(tag=ActionTag.add, side=position_side, source="neutral_entry_add")
        return PhraseSignal(tag=ActionTag.enter_long, side=None, source="neutral_entry")
    return None


def looks_explicit_trade_language(text: str) -> bool:
    if detect_setup_signal(text) is not None:
        return True
    return detect_present_trade_signal(text, position_side=None) is not None


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
