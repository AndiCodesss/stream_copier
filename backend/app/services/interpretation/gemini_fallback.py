"""Gemini API fallback for trade intent extraction and confirmation.

When the rule engine and local classifier cannot confidently determine a
trade intent, this module calls the Gemini LLM via its REST API as a
fallback. It serves two roles:

1. Extractive fallback -- ask Gemini to extract a structured trade action
   from the transcript (used in review mode).
2. Confirmation gate -- ask Gemini to confirm or reject a proposed entry
   before it is executed (used in auto mode to prevent false positives).

Safety overrides ensure that Gemini's raw output is sanitized: trim-like
language is not misread as a full exit, side is inferred from context when
missing, and management actions are blocked when no position is open.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import Settings
from app.models.domain import ActionTag, PositionState, StreamSession, TradeIntent, TradeSide, TranscriptSegment

# Regex patterns used by safety overrides to distinguish trims from exits.
_TRIM_HINT_PATTERNS = (
    r"\bpay(?:ing)? yourself\b",
    r"\bpay(?:ing)? myself\b",
    r"\btrim(?:ming)?\b",
    r"\bpeel(?:ing)?\b",
    r"\btake (?:some )?profit\b",
    r"\brunner(?:s)? on deck\b",
)

_HARD_EXIT_HINT_PATTERNS = (
    r"\bi(?:'| )?m out\b",
    r"\bout of this\b",
    r"\bcut that\b",
    r"\bstopped out\b",
    r"\bflatten\b",
    r"\bflat\b",
    r"\bexit all\b",
    r"\bclose all\b",
)

_SHORT_HINT_PATTERNS = (r"\bshort\b", r"\bsell(?:ing)?\b")
_LONG_HINT_PATTERNS = (r"\blong\b", r"\bbuy(?:ing)?\b")


@dataclass
class GeminiConfirmation:
    """Result of asking Gemini to confirm or reject a proposed trade action.

    system_failure=True means the API call itself failed (timeout, network
    error), as opposed to Gemini deliberately rejecting the intent.
    """

    confirmed: bool
    confidence: float = 0.0
    reason: str | None = None
    evidence_text: str | None = None
    system_failure: bool = False


class GeminiFallbackInterpreter:
    """Async client that calls Gemini for trade extraction or confirmation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    def is_available(self) -> bool:
        return bool(self._settings.enable_gemini_fallback and self._settings.gemini_api_key)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def interpret(self, session: StreamSession, segment: TranscriptSegment) -> TradeIntent | None:
        """Ask Gemini to extract a trade action from the transcript segment."""
        if not self.is_available():
            return None

        parsed = await self._generate_json(
            system_text=(
                "Extract only actionable futures-trading instructions. "
                "Return strict JSON with keys: tag, side, entry_price, stop_price, "
                "target_price, quantity_hint, confidence, evidence_text. "
                "Allowed tag values: NO_ACTION, ENTER_LONG, ENTER_SHORT, ADD, TRIM, "
                "EXIT_ALL, MOVE_STOP, MOVE_TO_BREAKEVEN. "
                "Allowed side values: LONG, SHORT, or null. "
                "If nothing actionable is present, return {\"tag\":\"NO_ACTION\",\"confidence\":0.0}."
            ),
            user_text=self._build_prompt(session=session, segment=segment),
        )
        if parsed is None:
            return None

        tag = _coerce_action_tag(parsed.get("tag"), parsed.get("side"))
        if tag is None or tag == ActionTag.no_action:
            return None

        trade_side = _coerce_trade_side(parsed.get("side"), tag)
        override = _apply_safety_overrides(
            transcript_text=segment.text,
            session_position=session.position,
            tag=tag,
            side=trade_side,
        )
        if override is None:
            return None
        tag, trade_side = override

        return TradeIntent(
            session_id=session.id,
            tag=tag,
            symbol=session.market.symbol,
            side=trade_side,
            entry_price=_coerce_float(parsed.get("entry_price")),
            stop_price=_coerce_float(parsed.get("stop_price")),
            target_price=_coerce_float(parsed.get("target_price")),
            quantity_hint=_coerce_optional_str(parsed.get("quantity_hint")),
            confidence=_coerce_confidence(parsed.get("confidence")),
            evidence_text=parsed.get("evidence_text") or segment.text,
        )

    async def confirm_intent(
        self,
        *,
        session: StreamSession,
        segment: TranscriptSegment,
        proposed_intent: TradeIntent,
        context_text: str | None = None,
    ) -> GeminiConfirmation:
        """Ask Gemini whether a proposed entry should actually execute."""
        if not self.is_available():
            return GeminiConfirmation(confirmed=False, reason="gemini unavailable", system_failure=True)

        parsed = await self._generate_json(
            system_text=(
                "You are validating whether a proposed futures-trading action should execute right now. "
                "Confirm only when the speaker is clearly describing their own live first-person trading action "
                "for the current session instrument. Reject historical recap, teaching examples, hypothetical "
                "scenarios, second-person advice, commentary about other traders, copy-trading explanations, "
                "questions, and other instruments. Return strict JSON with keys: confirmed, confidence, "
                "reason, evidence_text."
            ),
            user_text=self._build_confirmation_prompt(
                session=session,
                segment=segment,
                proposed_intent=proposed_intent,
                context_text=context_text,
            ),
        )
        if parsed is None:
            return GeminiConfirmation(confirmed=False, reason="invalid gemini response", system_failure=True)

        return GeminiConfirmation(
            confirmed=_coerce_confirmation(parsed.get("confirmed")),
            confidence=_coerce_confidence(parsed.get("confidence")),
            reason=_coerce_optional_str(parsed.get("reason")),
            evidence_text=_coerce_optional_str(parsed.get("evidence_text")),
        )

    def _build_prompt(self, *, session: StreamSession, segment: TranscriptSegment) -> str:
        position = session.position.model_dump(mode="json") if session.position else None
        return json.dumps(
            {
                "symbol": session.market.symbol,
                "market_price": session.market.last_price,
                "position": position,
                "latest_partial_text": session.latest_partial_text,
                "segment": segment.model_dump(mode="json"),
            }
        )

    def _build_confirmation_prompt(
        self,
        *,
        session: StreamSession,
        segment: TranscriptSegment,
        proposed_intent: TradeIntent,
        context_text: str | None,
    ) -> str:
        position = session.position.model_dump(mode="json") if session.position else None
        return json.dumps(
            {
                "symbol": session.market.symbol,
                "market_price": session.market.last_price,
                "position": position,
                "latest_partial_text": session.latest_partial_text,
                "stitched_context_text": context_text,
                "segment": segment.model_dump(mode="json"),
                "proposed_intent": proposed_intent.model_dump(mode="json"),
            }
        )

    async def _generate_json(self, *, system_text: str, user_text: str) -> dict[str, Any] | None:
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            client = self._client
            if client is None:
                client = httpx.AsyncClient(
                    base_url=self._settings.gemini_base_url,
                    timeout=10.0,
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
                )
                self._client = client
            response = await client.post(
                f"/models/{self._settings.gemini_model}:generateContent",
                params={"key": self._settings.gemini_api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return None

        text = _extract_candidate_text(data)
        if text is None:
            return None

        return _parse_json_payload(text)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _coerce_confirmation(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    return token in {"true", "1", "yes", "confirm", "confirmed"}


def _extract_candidate_text(data: Any) -> str | None:
    try:
        return str(data["candidates"][0]["content"]["parts"][0]["text"])
    except Exception:
        return None


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _coerce_action_tag(tag_value: Any, side_value: Any) -> ActionTag | None:
    """Map Gemini's free-form tag string to a canonical ActionTag enum."""
    token = _norm_token(tag_value)
    side_token = _norm_token(side_value)

    if token in {"NO_ACTION", ""}:
        return ActionTag.no_action
    if token in {"ENTER_LONG", "LONG_ENTRY", "GO_LONG"}:
        return ActionTag.enter_long
    if token in {"ENTER_SHORT", "SHORT_ENTRY", "GO_SHORT"}:
        return ActionTag.enter_short
    if token in {"ADD", "SCALE_IN"}:
        return ActionTag.add
    if token in {"TRIM", "TAKE_PROFIT", "PARTIAL_EXIT", "PAY_YOURSELF"}:
        return ActionTag.trim
    if token in {"EXIT_ALL", "EXIT", "FLAT", "CLOSE_ALL", "CLOSE"}:
        return ActionTag.exit_all
    if token in {"MOVE_STOP", "ADJUST_STOP", "STOP_MOVE"}:
        return ActionTag.move_stop
    if token in {"MOVE_TO_BREAKEVEN", "BREAKEVEN"}:
        return ActionTag.move_to_breakeven

    # Common Gemini variants:
    if token in {"LONG", "BUY"}:
        return ActionTag.enter_long
    if token in {"SHORT", "SELL"}:
        return ActionTag.enter_short

    # Side-only fallback for implicit entry.
    if side_token in {"LONG", "BUY"}:
        return ActionTag.enter_long
    if side_token in {"SHORT", "SELL"}:
        return ActionTag.enter_short
    return None


def _coerce_trade_side(side_value: Any, tag: ActionTag) -> TradeSide | None:
    side_token = _norm_token(side_value)
    if side_token in {"LONG", "BUY"}:
        return TradeSide.long
    if side_token in {"SHORT", "SELL"}:
        return TradeSide.short
    if tag == ActionTag.enter_long:
        return TradeSide.long
    if tag == ActionTag.enter_short:
        return TradeSide.short
    return None


def _norm_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().upper()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^A-Z0-9_]", "", token)
    return token


def _apply_safety_overrides(
    *,
    transcript_text: str,
    session_position: PositionState | None,
    tag: ActionTag,
    side: TradeSide | None,
) -> tuple[ActionTag, TradeSide | None] | None:
    """Correct or reject unsafe Gemini outputs before they reach execution.

    Prevents exit_all when the text only describes trimming, blocks
    management actions when no position is open, and infers missing side.
    Returns None to suppress the action entirely.
    """
    normalized = _normalize_text(transcript_text)
    has_trim_hint = _contains_any(_TRIM_HINT_PATTERNS, normalized)
    has_hard_exit_hint = _contains_any(_HARD_EXIT_HINT_PATTERNS, normalized)

    # Prevent accidental full-flatten interpretation on clear scale-out language.
    if tag == ActionTag.exit_all and has_trim_hint and not has_hard_exit_hint:
        tag = ActionTag.trim

    if tag == ActionTag.add and session_position is None:
        inferred = side or _infer_side_from_text(normalized)
        if inferred == TradeSide.long:
            return ActionTag.enter_long, inferred
        if inferred == TradeSide.short:
            return ActionTag.enter_short, inferred
        return None

    if tag in {ActionTag.trim, ActionTag.move_stop, ActionTag.move_to_breakeven, ActionTag.add}:
        if session_position is None:
            return None
        if side is None:
            side = session_position.side

    if tag in {ActionTag.enter_long, ActionTag.enter_short} and side is None:
        side = TradeSide.long if tag == ActionTag.enter_long else TradeSide.short

    return tag, side


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^\w\s']", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def _infer_side_from_text(text: str) -> TradeSide | None:
    has_short = _contains_any(_SHORT_HINT_PATTERNS, text)
    has_long = _contains_any(_LONG_HINT_PATTERNS, text)
    if has_short and has_long:
        return None
    if has_short:
        return TradeSide.short
    if has_long:
        return TradeSide.long
    return None
