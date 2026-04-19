"""Context envelope that bundles all inputs the classifier needs.

Before the ModernBERT classifier can predict an intent, it needs the current
transcript text plus surrounding context (recent speech, position state,
market symbol). This module packages those inputs into a single object and
renders them into the text format the classifier was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import TradeSide


@dataclass(frozen=True)
class IntentContextEnvelope:
    """Immutable snapshot of everything the classifier needs for one prediction.

    Fields capture the current segment, recent context, position state, and
    market info. The render() method serializes them into the key=value text
    format expected by the fine-tuned ModernBERT tokenizer.
    """
    symbol: str
    current_text: str
    current_normalized: str
    recent_text: str | None
    analysis_text: str
    entry_text: str
    position_side: TradeSide | None
    last_side: TradeSide | None
    market_price: float | None

    def render(self) -> str:
        position = (
            self.position_side.value if isinstance(self.position_side, TradeSide) else str(self.position_side or "FLAT")
        )
        last_side = self.last_side.value if isinstance(self.last_side, TradeSide) else str(self.last_side or "NONE")
        lines = [
            f"symbol={self.symbol}",
            f"position={position}",
            f"last_side={last_side}",
        ]
        if self.recent_text:
            lines.append(f"recent={_clip(self.recent_text)}")
        if self.analysis_text and self.analysis_text != self.current_normalized:
            lines.append(f"analysis={_clip(self.analysis_text)}")
        if self.entry_text and self.entry_text not in {self.current_normalized, self.analysis_text}:
            lines.append(f"entry={_clip(self.entry_text)}")
        lines.append(f"current={_clip(self.current_text)}")
        return "\n".join(lines)


def _clip(text: str, *, max_words: int = 48) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()
