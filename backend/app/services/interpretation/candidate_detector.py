from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.domain import ActionTag, TradeSide
from app.services.interpretation.action_language import PhraseSignal
from app.services.interpretation.local_classifier import IntentClassifierPrediction

_ENTRY_FAMILY = {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}
_MANAGEMENT_FAMILY = {
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
}
_SIDE_HINTS = {
    ActionTag.enter_long: TradeSide.long,
    ActionTag.enter_short: TradeSide.short,
}
_SEED_PATTERNS = (
    re.compile(r"\b(?:small|smaller)\s+size\b"),
    re.compile(r"\b(?:piece|peace)\s+on\b"),
    re.compile(r"\b(?:putting|put|got|get)\s+(?:something|a\s+(?:small|little)\s+(?:piece|peace))\s+on\b"),
    re.compile(r"\bget something on\b"),
    re.compile(r"\b(?:one|two|three)\s+lots?\b(?:\s+(?:filled|on|at|here|there|versus))"),
    re.compile(r"\b(?:one|two|three)\s+lot\b(?:\s+(?:filled|on|at|here|there|versus))"),
    re.compile(r"\bversus\b"),
    re.compile(r"\bstay heavy\b"),
    re.compile(r"\b(?:pay(?:ing|ed)?\s+(?:myself|ourselves)|breakeven|flatten)\b"),
)
_CONTINUATION_PATTERNS = (
    re.compile(r"^(?:and|but|so|then|now|because)\b"),
    re.compile(r"^(?:versus|under|over|through|at|from|into)\b"),
    re.compile(r"^(?:short|long)\s+versus\b"),
    re.compile(r"^(?:risk|stop|stops?)\b"),
)


@dataclass(frozen=True)
class CandidateAssessment:
    probability: float
    source: str
    tag_hint: ActionTag | None = None
    side_hint: TradeSide | None = None

    def should_track(self, *, open_threshold: float, keep_threshold: float, window_active: bool) -> bool:
        threshold = keep_threshold if window_active else open_threshold
        return self.probability >= threshold


def looks_candidate_seed(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SEED_PATTERNS)


def looks_candidate_continuation(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CONTINUATION_PATTERNS)


def assess_trade_candidate(
    *,
    text: str,
    analysis_text: str,
    entry_text: str,
    classifier_prediction: IntentClassifierPrediction | None,
    explicit_signal: PhraseSignal | None,
    setup_signal: PhraseSignal | None,
) -> CandidateAssessment | None:
    candidates: list[CandidateAssessment] = []

    if classifier_prediction is not None:
        if classifier_prediction.entry_probability > 0.0:
            candidates.append(
                CandidateAssessment(
                    probability=classifier_prediction.entry_probability,
                    source="classifier_entry",
                    tag_hint=classifier_prediction.tag if classifier_prediction.tag in _ENTRY_FAMILY else None,
                    side_hint=_SIDE_HINTS.get(classifier_prediction.tag),
                )
            )
        if classifier_prediction.management_probability > 0.0:
            candidates.append(
                CandidateAssessment(
                    probability=classifier_prediction.management_probability,
                    source="classifier_management",
                    tag_hint=classifier_prediction.tag if classifier_prediction.tag in _MANAGEMENT_FAMILY else None,
                    side_hint=_SIDE_HINTS.get(classifier_prediction.tag),
                )
            )
    if explicit_signal is not None:
        candidates.append(
            CandidateAssessment(
                probability=0.96 if explicit_signal.actionable else 0.78,
                source="explicit_signal",
                tag_hint=explicit_signal.tag,
                side_hint=explicit_signal.side,
            )
        )

    if setup_signal is not None:
        candidates.append(
            CandidateAssessment(
                probability=0.74,
                source="setup_signal",
                tag_hint=setup_signal.tag,
                side_hint=setup_signal.side,
            )
        )

    if looks_candidate_seed(entry_text) or looks_candidate_seed(text):
        candidates.append(
            CandidateAssessment(
                probability=0.58,
                source="seed_phrase",
                tag_hint=None,
                side_hint=_side_hint_from_text(entry_text or analysis_text or text),
            )
        )

    if looks_candidate_continuation(text) and (
        looks_candidate_seed(analysis_text)
        or looks_candidate_seed(entry_text)
        or (classifier_prediction is not None and classifier_prediction.action_probability >= 0.22)
    ):
        candidates.append(
            CandidateAssessment(
                probability=0.48,
                source="continuation",
                tag_hint=None,
                side_hint=_side_hint_from_text(entry_text or analysis_text or text),
            )
        )

    if not candidates:
        return None

    strongest = max(candidates, key=lambda candidate: candidate.probability)
    return CandidateAssessment(
        probability=round(strongest.probability, 6),
        source=strongest.source,
        tag_hint=strongest.tag_hint,
        side_hint=strongest.side_hint,
    )


def _side_hint_from_text(text: str) -> TradeSide | None:
    if re.search(r"\blong\b", text):
        return TradeSide.long
    if re.search(r"\bshort\b", text):
        return TradeSide.short
    return None
