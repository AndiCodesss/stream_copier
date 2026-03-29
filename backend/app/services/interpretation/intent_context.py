from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import TradeSide


@dataclass(frozen=True)
class IntentContextEnvelope:
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
        market = f"{self.market_price:.2f}" if self.market_price is not None else "UNKNOWN"
        lines = [
            f"symbol={self.symbol}",
            f"position={position}",
            f"last_side={last_side}",
            f"market_price={market}",
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
