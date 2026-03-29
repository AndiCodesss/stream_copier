from __future__ import annotations

from app.models.domain import ActionTag, PositionState, TradeSide
from app.services.interpretation.gemini_fallback import (
    _apply_safety_overrides,
    _coerce_action_tag,
    _coerce_optional_str,
    _coerce_trade_side,
    _parse_json_payload,
)


def test_parse_json_payload_accepts_fenced_json() -> None:
    payload = _parse_json_payload("```json\n{\"tag\":\"ENTER_SHORT\",\"side\":\"SHORT\"}\n```")
    assert payload is not None
    assert payload["tag"] == "ENTER_SHORT"


def test_coerce_action_tag_maps_common_short_forms() -> None:
    assert _coerce_action_tag("SHORT", None) == ActionTag.enter_short
    assert _coerce_action_tag("SELL", None) == ActionTag.enter_short
    assert _coerce_action_tag("LONG", None) == ActionTag.enter_long
    assert _coerce_action_tag("BUY", None) == ActionTag.enter_long


def test_coerce_action_tag_uses_side_when_tag_unknown() -> None:
    assert _coerce_action_tag("OPEN", "SELL") == ActionTag.enter_short
    assert _coerce_action_tag("OPEN", "BUY") == ActionTag.enter_long


def test_coerce_trade_side_infers_from_tag() -> None:
    assert _coerce_trade_side(None, ActionTag.enter_short) == TradeSide.short
    assert _coerce_trade_side(None, ActionTag.enter_long) == TradeSide.long


def test_safety_override_converts_exit_all_to_trim_on_trim_language() -> None:
    result = _apply_safety_overrides(
        transcript_text="Paying myself some here.",
        session_position=PositionState(side=TradeSide.long, quantity=2, average_price=21230.0),
        tag=ActionTag.exit_all,
        side=None,
    )
    assert result is not None
    tag, side = result
    assert tag == ActionTag.trim
    assert side == TradeSide.long


def test_safety_override_rejects_trim_without_position() -> None:
    result = _apply_safety_overrides(
        transcript_text="Trim some here.",
        session_position=None,
        tag=ActionTag.trim,
        side=None,
    )
    assert result is None


def test_safety_override_converts_add_to_entry_without_position() -> None:
    result = _apply_safety_overrides(
        transcript_text="Adding here short.",
        session_position=None,
        tag=ActionTag.add,
        side=TradeSide.short,
    )
    assert result == (ActionTag.enter_short, TradeSide.short)


def test_coerce_optional_str_accepts_non_string_values() -> None:
    assert _coerce_optional_str(3) == "3"
    assert _coerce_optional_str("  some  ") == "some"
    assert _coerce_optional_str("") is None
