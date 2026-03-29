from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import re

from app.models.domain import ActionTag, MarketSnapshot, SessionConfig, StreamSession, TradeSide
from app.services.interpretation.ai_transcript_annotator import (
    AiAnnotation,
    _annotation_sort_key,
    _apply_annotation_state,
    _normalize_label_token,
    _side_value,
    _timestamped_lines,
)
from app.services.interpretation.intent_context import IntentContextEnvelope
from app.services.interpretation.rule_engine import RuleBasedTradeInterpreter, _normalize

_REVIEW_PENDING = "pending"
_REVIEW_ACCEPTED = "accepted"
_REVIEW_CORRECTED = "corrected"
_REVIEW_REJECTED = "rejected"
_REVIEW_STATUSES = {_REVIEW_PENDING, _REVIEW_ACCEPTED, _REVIEW_CORRECTED, _REVIEW_REJECTED}
_REVIEWED_STATUSES = {_REVIEW_ACCEPTED, _REVIEW_CORRECTED, _REVIEW_REJECTED}
_EXPORT_SOURCE_BY_STATUS = {
    _REVIEW_ACCEPTED: "ai_review_accept",
    _REVIEW_CORRECTED: "ai_review_corrected",
    _REVIEW_REJECTED: "ai_review_hard_negative",
}
_STATUS_PRIORITY = {
    _REVIEW_CORRECTED: 3,
    _REVIEW_ACCEPTED: 2,
    _REVIEW_REJECTED: 1,
    _REVIEW_PENDING: 0,
}
_FIRST_PERSON_ENTRY_PATTERNS = (
    r"\blet'?s put one of these on\b",
    r"\bsmall piece on\b",
    r"\bsmall (?:long|short) now\b",
    r"\bput(?:ting)? (?:a |one of these |something |a little )?piece on\b",
    r"\bput something on\b",
    r"\bput a little piece on here just in case\b",
    r"\bfeeler on (?:short|long)\b",
    r"\bback on again\b",
    r"\b(?:i'?m|im) in (?:a |this )?(?:short|long)\b",
    r"\b(?:i'?m|im) (?:short|long)(?: again)? now\b",
    r"\bstick this back on here again\b",
    r"\bback in this now (?:short|long) side\b",
    r"\b(?:short|long) versus this\b",
    r"\bputting a little piece on\b",
)
_SETUP_SHORT_PATTERNS = (
    r"\bwe(?:'re| are)? looking to sell\b",
    r"\blooking to sell\b",
    r"\blooking for (?:a |this )?short\b",
    r"\bwatching\b.*\bsell into\b",
    r"\bwatching\b",
    r"\bcan look for this short\b",
    r"\bfade this market\b",
    r"\bsell pops\b",
    r"\bpop to sell\b",
    r"\blook for the same trade again\b",
)
_SETUP_LONG_PATTERNS = (
    r"\bwe(?:'re| are)? looking to buy\b",
    r"\blooking to buy\b",
    r"\blooking for (?:a |this )?long\b",
    r"\bwatching\b.*\bbuy into\b",
    r"\bbuy dips\b",
    r"\bpullback to buy\b",
)
_FUTURE_ADD_PATTERNS = (
    r"\b(?:i'?ll|i will)\b.*\badd\b",
    r"\b(?:i'?m|im) going to add\b",
    r"\blooking for .*add\b",
    r"\blook for an ad\b",
    r"\btry and add on pops\b",
    r"\bif .* add\b",
)
_EXPLICIT_ADD_PATTERNS = (
    r"\bgot my ad on\b",
    r"\bpopped one back on\b",
    r"\badded back into this\b",
    r"\badded back in on this\b",
    r"\badd back on here\b",
)
_FIRST_PERSON_TRIM_PATTERNS = (
    r"\bpay yourself\b",
    r"\bpay(?:ing)? myself\b",
    r"\btrim(?:ming)?\b",
    r"\blittle piece off\b",
    r"\btake a partial\b",
    r"\btake partial\b",
    r"\bcover(?:ing)? some\b",
    r"\bpeel(?:ing)?\b",
)
_SECOND_PERSON_TRIM_REJECT_PATTERNS = (
    r"\bso you can pay yourself\b",
    r"\byou should be paying yourself\b",
    r"\byou can pay yourself\b",
)
_EXPLICIT_EXIT_PATTERNS = (
    r"\bi(?:'| )?m out\b",
    r"\bout of that\b",
    r"\bout of this now\b",
    r"\bout here\b",
    r"\bout completely\b",
    r"\bflat now\b",
    r"\bout\.\s*done\.\s*finished\b",
    r"\bout\s+done\s+finished\b",
)
_EXIT_REJECT_PATTERNS = (
    r"\bi(?:'| )?ll cut the rest\b",
    r"\bcut the rest into\b",
)
_STOP_PATTERNS = (
    r"\bmove my stop\b",
    r"\bstop tighter\b",
    r"\bstops? break even\b",
    r"\bbreak even stop\b",
    r"\brisk(?:ing)? up to\b",
)
_OTHER_INSTRUMENT_PATTERNS = (
    r"\bunh\b",
    r"\bnvda\b",
    r"\bnvidia\b",
)


@dataclass(frozen=True)
class ReviewCandidate:
    id: str
    file: str
    line: int
    timecode: str
    original_label: str
    original_side: str | None
    ai_confidence: float
    ai_reason: str | None
    evidence_text: str
    current_text: str
    review_status: str = _REVIEW_PENDING
    reviewed_label: str | None = None
    reviewed_side: str | None = None
    review_note: str | None = None


@dataclass(frozen=True)
class ReviewedLineDecision:
    candidate_id: str
    file: str
    line: int
    timecode: str
    label: ActionTag
    side: TradeSide | None
    source: str
    original_label: str
    original_side: str | None
    ai_confidence: float
    ai_reason: str | None
    evidence_text: str
    current_text: str
    review_status: str
    review_note: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Gemini transcript annotations and export clean training labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review", help="Review AI-generated candidates interactively.")
    review_parser.add_argument("report", type=Path, help="Path to ai_transcript_annotations JSON report.")
    review_parser.add_argument(
        "--review-out",
        type=Path,
        default=Path("data/interpretation/ai_annotation_review.json"),
        help="Persistent review-state JSON path.",
    )
    review_parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Number of transcript lines to show before and after the candidate line.",
    )
    review_parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="Iterate through already-reviewed items as well.",
    )

    auto_parser = subparsers.add_parser("auto", help="Auto-review obvious accept/reject/correct cases.")
    auto_parser.add_argument("report", type=Path, help="Path to ai_transcript_annotations JSON report.")
    auto_parser.add_argument(
        "--review-out",
        type=Path,
        default=Path("data/interpretation/ai_annotation_review.json"),
        help="Persistent review-state JSON path.",
    )
    auto_parser.add_argument(
        "--overwrite-reviewed",
        action="store_true",
        help="Re-apply auto review even to items that were already reviewed.",
    )

    export_parser = subparsers.add_parser("export", help="Export reviewed labels into classifier-ready JSONL.")
    export_parser.add_argument("review", type=Path, help="Path to review-state JSON.")
    export_parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("data/interpretation/ai_reviewed_intent_examples.jsonl"),
        help="Output JSONL path for reviewed examples.",
    )
    export_parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/interpretation/ai_reviewed_intent_examples.summary.json"),
        help="Output JSON summary path.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_id(*, file: str, annotation: dict[str, Any]) -> str:
    parts = (
        file,
        str(annotation.get("line", "")),
        str(annotation.get("timecode", "")),
        str(annotation.get("label", "")),
        str(annotation.get("side", "")),
        str(annotation.get("evidence_text", "")),
        str(annotation.get("current_text", "")),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _resolve_candidate_file(raw: str, *, cwd: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    resolved = (cwd / path).resolve()
    return str(resolved)


def _build_review_candidates(*, report: dict[str, Any], cwd: Path) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for file_report in report.get("files", []):
        raw_file = str(file_report.get("file", ""))
        resolved_file = _resolve_candidate_file(raw_file, cwd=cwd)
        for annotation in file_report.get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            candidates.append(
                ReviewCandidate(
                    id=_candidate_id(file=resolved_file, annotation=annotation),
                    file=resolved_file,
                    line=int(annotation.get("line", 0)),
                    timecode=str(annotation.get("timecode", "")),
                    original_label=str(annotation.get("label", "")),
                    original_side=str(annotation.get("side")) if annotation.get("side") is not None else None,
                    ai_confidence=float(annotation.get("confidence", 0.0) or 0.0),
                    ai_reason=str(annotation.get("reason")) if annotation.get("reason") is not None else None,
                    evidence_text=str(annotation.get("evidence_text", "")),
                    current_text=str(annotation.get("current_text", "")),
                )
            )
    return sorted(candidates, key=lambda item: (Path(item.file).name, item.line, item.timecode, item.original_label, item.id))


def _merge_review_candidates(
    *,
    fresh_candidates: list[ReviewCandidate],
    existing_candidates: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    existing_by_id = {
        str(candidate.get("id", "")): candidate
        for candidate in (existing_candidates or [])
        if isinstance(candidate, dict)
    }
    merged: list[dict[str, Any]] = []
    for candidate in fresh_candidates:
        payload = asdict(candidate)
        existing = existing_by_id.get(candidate.id)
        if existing is not None:
            payload["review_status"] = str(existing.get("review_status", _REVIEW_PENDING))
            payload["reviewed_label"] = existing.get("reviewed_label")
            payload["reviewed_side"] = existing.get("reviewed_side")
            payload["review_note"] = existing.get("review_note")
        merged.append(payload)
    return merged


def load_or_initialize_review(*, report_path: Path, review_path: Path) -> dict[str, Any]:
    report = _load_json(report_path)
    existing = _load_json(review_path) if review_path.exists() else {}
    fresh_candidates = _build_review_candidates(report=report, cwd=Path.cwd())
    state = {
        "created_at": existing.get("created_at") or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "report_path": str(report_path.resolve()),
        "model": report.get("model"),
        "symbol": report.get("symbol"),
        "market_price": report.get("market_price"),
        "candidates": _merge_review_candidates(
            fresh_candidates=fresh_candidates,
            existing_candidates=existing.get("candidates"),
        ),
    }
    return state


def _save_review_state(review_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(UTC).isoformat()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("vwop", "vwap")
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _side_from_label(label: ActionTag) -> TradeSide | None:
    if label in {ActionTag.enter_long, ActionTag.setup_long}:
        return TradeSide.long
    if label in {ActionTag.enter_short, ActionTag.setup_short}:
        return TradeSide.short
    return None


def _auto_review_candidate(candidate: dict[str, Any]) -> bool:
    if candidate.get("review_status") in _REVIEWED_STATUSES:
        return False

    text = _normalize_text(f"{candidate.get('current_text', '')} {candidate.get('evidence_text', '')}")
    label = _parse_action_tag(str(candidate.get("original_label") or ""))
    side = _to_trade_side(candidate.get("original_side"))
    if label is None:
        return False

    if _contains_any(text, _OTHER_INSTRUMENT_PATTERNS):
        _apply_reject(candidate, note="other instrument")
        return True

    if label in {ActionTag.setup_long, ActionTag.setup_short}:
        _apply_accept(candidate, note="auto setup")
        return True

    if label in {ActionTag.enter_short, ActionTag.enter_long, ActionTag.add}:
        if _contains_any(text, _SETUP_SHORT_PATTERNS):
            _apply_correction(candidate, label=ActionTag.setup_short, side=TradeSide.short.value, note="auto setup short")
            return True
        if _contains_any(text, _SETUP_LONG_PATTERNS):
            _apply_correction(candidate, label=ActionTag.setup_long, side=TradeSide.long.value, note="auto setup long")
            return True
        if label in {ActionTag.enter_short, ActionTag.enter_long} and _contains_any(text, _EXPLICIT_ADD_PATTERNS):
            _apply_correction(candidate, label=ActionTag.add, side=(side.value if side is not None else None), note="explicit add")
            return True
        if label == ActionTag.add:
            if _contains_any(text, _FUTURE_ADD_PATTERNS):
                _apply_reject(candidate, note="future or conditional add")
                return True
            if _contains_any(text, _EXPLICIT_ADD_PATTERNS):
                _apply_accept(candidate, note="explicit add")
                return True
        if label in {ActionTag.enter_short, ActionTag.enter_long}:
            if _contains_any(text, _FIRST_PERSON_ENTRY_PATTERNS):
                _apply_accept(candidate, note="explicit entry")
                return True
            if "reselling here" in text:
                _apply_reject(candidate, note="ambiguous market action")
                return True
        return False

    if label == ActionTag.trim:
        if _contains_any(text, _SECOND_PERSON_TRIM_REJECT_PATTERNS):
            _apply_reject(candidate, note="second person advice")
            return True
        if _contains_any(text, _FIRST_PERSON_TRIM_PATTERNS):
            _apply_accept(candidate, note="explicit trim")
            return True
        return False

    if label == ActionTag.exit_all:
        if _contains_any(text, _EXIT_REJECT_PATTERNS):
            _apply_reject(candidate, note="future exit target")
            return True
        if _contains_any(text, _EXPLICIT_EXIT_PATTERNS):
            _apply_accept(candidate, note="explicit exit")
            return True
        return False

    if label in {ActionTag.move_stop, ActionTag.move_to_breakeven}:
        if _contains_any(text, _STOP_PATTERNS):
            _apply_accept(candidate, note="explicit stop move")
            return True
        return False

    if label == ActionTag.no_action:
        _apply_accept(candidate, note="already no action")
        return True

    if side is not None and _contains_any(text, _FIRST_PERSON_ENTRY_PATTERNS):
        _apply_accept(candidate, note="explicit side action")
        return True
    return False


def _auto_review_state(state: dict[str, Any], *, overwrite_reviewed: bool) -> dict[str, int]:
    changed = 0
    auto_reviewed = 0
    for candidate in state.get("candidates", []):
        if not overwrite_reviewed and candidate.get("review_status") in _REVIEWED_STATUSES:
            continue
        before = (
            candidate.get("review_status"),
            candidate.get("reviewed_label"),
            candidate.get("reviewed_side"),
            candidate.get("review_note"),
        )
        if _auto_review_candidate(candidate):
            auto_reviewed += 1
        after = (
            candidate.get("review_status"),
            candidate.get("reviewed_label"),
            candidate.get("reviewed_side"),
            candidate.get("review_note"),
        )
        if after != before:
            changed += 1
    counts = _review_counts(state.get("candidates", []))
    counts["changed"] = changed
    counts["auto_reviewed"] = auto_reviewed
    return counts


def _review_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(candidate.get("review_status", _REVIEW_PENDING)) for candidate in candidates)
    return {status: counts.get(status, 0) for status in sorted(_REVIEW_STATUSES)}


def _load_transcript_cache(path: str, cache: dict[str, list[str]]) -> list[str]:
    cached = cache.get(path)
    if cached is not None:
        return cached
    rows = Path(path).read_text(encoding="utf-8").splitlines()
    cache[path] = rows
    return rows


def _print_candidate(
    *,
    candidate: dict[str, Any],
    index: int,
    total: int,
    context_lines: int,
    transcript_cache: dict[str, list[str]],
) -> None:
    print("")
    print(f"[{index}/{total}] {Path(str(candidate['file'])).name}:{candidate['line']} {candidate['timecode']}")
    print(
        "original="
        f"{candidate['original_label']}"
        f"{' ' + candidate['original_side'] if candidate.get('original_side') else ''} "
        f"confidence={float(candidate.get('ai_confidence', 0.0)):.2f}"
    )
    if candidate.get("review_status") != _REVIEW_PENDING:
        print(
            "review="
            f"{candidate.get('review_status')} "
            f"label={candidate.get('reviewed_label')} "
            f"side={candidate.get('reviewed_side') or 'NONE'}"
        )
    print(f"evidence={candidate.get('evidence_text')}")
    if candidate.get("ai_reason"):
        print(f"reason={candidate['ai_reason']}")
    rows = _load_transcript_cache(str(candidate["file"]), transcript_cache)
    start = max(1, int(candidate["line"]) - context_lines)
    end = min(len(rows), int(candidate["line"]) + context_lines)
    print("context:")
    for line_number in range(start, end + 1):
        marker = ">" if line_number == int(candidate["line"]) else " "
        print(f"{marker}{line_number}: {rows[line_number - 1]}")


def _parse_action_tag(value: str) -> ActionTag | None:
    token = _normalize_label_token(value)
    for tag in ActionTag:
        if tag.value == token:
            return tag
    return None


def _normalize_side_for_label(label: ActionTag, side_value: str | None) -> str | None:
    if label == ActionTag.no_action:
        return None
    if label in {ActionTag.enter_long, ActionTag.setup_long}:
        return TradeSide.long.value
    if label in {ActionTag.enter_short, ActionTag.setup_short}:
        return TradeSide.short.value
    if side_value is None:
        return None
    token = _normalize_label_token(side_value)
    if token in {"LONG", "BUY"}:
        return TradeSide.long.value
    if token in {"SHORT", "SELL"}:
        return TradeSide.short.value
    return None


def _apply_accept(candidate: dict[str, Any], note: str | None = None) -> None:
    candidate["review_status"] = _REVIEW_ACCEPTED
    candidate["reviewed_label"] = candidate.get("original_label")
    candidate["reviewed_side"] = candidate.get("original_side")
    candidate["review_note"] = note


def _apply_reject(candidate: dict[str, Any], note: str | None = None) -> None:
    candidate["review_status"] = _REVIEW_REJECTED
    candidate["reviewed_label"] = ActionTag.no_action.value
    candidate["reviewed_side"] = None
    candidate["review_note"] = note


def _apply_correction(candidate: dict[str, Any], *, label: ActionTag, side: str | None, note: str | None = None) -> None:
    if label == ActionTag.no_action:
        _apply_reject(candidate, note=note)
        return
    candidate["review_status"] = _REVIEW_CORRECTED
    candidate["reviewed_label"] = label.value
    candidate["reviewed_side"] = _normalize_side_for_label(label, side or candidate.get("original_side"))
    candidate["review_note"] = note


def _print_review_help() -> None:
    print("commands:")
    print("  a [note]                 accept as-is")
    print("  r [note]                 reject and export as NO_ACTION")
    print("  c LABEL [SIDE] [note]    correct label and optional side")
    print("  s                        skip for now")
    print("  q                        save and quit")
    print("  ?                        show help")


def _review_command(candidate: dict[str, Any], command: str) -> bool:
    tokens = command.strip().split()
    if not tokens:
        return False
    action = tokens[0].lower()
    if action == "a":
        note = " ".join(tokens[1:]) or None
        _apply_accept(candidate, note=note)
        return True
    if action == "r":
        note = " ".join(tokens[1:]) or None
        _apply_reject(candidate, note=note)
        return True
    if action == "c":
        if len(tokens) < 2:
            print("missing label for correction")
            return False
        label = _parse_action_tag(tokens[1])
        if label is None:
            print(f"unknown label: {tokens[1]}")
            return False
        side: str | None = None
        note_start = 2
        if len(tokens) >= 3:
            maybe_side = _normalize_side_for_label(label, tokens[2])
            if maybe_side is not None and tokens[2].upper() in {"LONG", "SHORT", "BUY", "SELL"}:
                side = maybe_side
                note_start = 3
        note = " ".join(tokens[note_start:]) or None
        _apply_correction(candidate, label=label, side=side, note=note)
        return True
    if action == "?":
        _print_review_help()
        return False
    if action == "s":
        return True
    if action == "q":
        raise EOFError
    print(f"unknown command: {action}")
    return False


def _run_review(args: argparse.Namespace) -> int:
    state = load_or_initialize_review(report_path=args.report, review_path=args.review_out)
    _save_review_state(args.review_out, state)
    transcript_cache: dict[str, list[str]] = {}
    candidates = state["candidates"]
    reviewable = [
        candidate
        for candidate in candidates
        if args.show_reviewed or candidate.get("review_status", _REVIEW_PENDING) == _REVIEW_PENDING
    ]
    total = len(reviewable)
    if total == 0:
        counts = _review_counts(candidates)
        print(f"No candidates to review. pending={counts[_REVIEW_PENDING]}")
        return 0

    print(f"Loaded {len(candidates)} candidates from {args.report}")
    print(f"Review state: {args.review_out}")
    _print_review_help()

    for index, candidate in enumerate(reviewable, start=1):
        _print_candidate(
            candidate=candidate,
            index=index,
            total=total,
            context_lines=max(1, int(args.context_lines)),
            transcript_cache=transcript_cache,
        )
        while True:
            try:
                command = input("review> ").strip()
            except EOFError:
                command = "q"
            try:
                handled = _review_command(candidate, command)
            except EOFError:
                _save_review_state(args.review_out, state)
                counts = _review_counts(candidates)
                print(f"Saved review state to {args.review_out}")
                print(f"pending={counts[_REVIEW_PENDING]}")
                return 0
            if handled:
                _save_review_state(args.review_out, state)
                break

    counts = _review_counts(candidates)
    print(f"Saved review state to {args.review_out}")
    print(f"accepted={counts[_REVIEW_ACCEPTED]} corrected={counts[_REVIEW_CORRECTED]} rejected={counts[_REVIEW_REJECTED]} pending={counts[_REVIEW_PENDING]}")
    return 0


def _run_auto(args: argparse.Namespace) -> int:
    state = load_or_initialize_review(report_path=args.report, review_path=args.review_out)
    counts = _auto_review_state(state, overwrite_reviewed=bool(args.overwrite_reviewed))
    _save_review_state(args.review_out, state)
    print(f"Saved auto-reviewed state to {args.review_out}")
    print(
        "accepted="
        f"{counts[_REVIEW_ACCEPTED]} corrected={counts[_REVIEW_CORRECTED]} "
        f"rejected={counts[_REVIEW_REJECTED]} pending={counts[_REVIEW_PENDING]} "
        f"auto_reviewed={counts['auto_reviewed']} changed={counts['changed']}"
    )
    return 0


def _to_trade_side(value: str | None) -> TradeSide | None:
    token = _normalize_label_token(value)
    if token == "LONG":
        return TradeSide.long
    if token == "SHORT":
        return TradeSide.short
    return None


def _decision_from_candidate(candidate: dict[str, Any]) -> ReviewedLineDecision | None:
    status = str(candidate.get("review_status", _REVIEW_PENDING))
    if status not in _REVIEWED_STATUSES:
        return None
    label = _parse_action_tag(str(candidate.get("reviewed_label") or ""))
    if label is None:
        return None
    return ReviewedLineDecision(
        candidate_id=str(candidate["id"]),
        file=str(candidate["file"]),
        line=int(candidate["line"]),
        timecode=str(candidate["timecode"]),
        label=label,
        side=_to_trade_side(candidate.get("reviewed_side")),
        source=_EXPORT_SOURCE_BY_STATUS[status],
        original_label=str(candidate.get("original_label", "")),
        original_side=str(candidate.get("original_side")) if candidate.get("original_side") is not None else None,
        ai_confidence=float(candidate.get("ai_confidence", 0.0) or 0.0),
        ai_reason=str(candidate.get("ai_reason")) if candidate.get("ai_reason") is not None else None,
        evidence_text=str(candidate.get("evidence_text", "")),
        current_text=str(candidate.get("current_text", "")),
        review_status=status,
        review_note=str(candidate.get("review_note")) if candidate.get("review_note") is not None else None,
    )


def _decision_sort_key(decision: ReviewedLineDecision) -> tuple[int, float, tuple[float, int, int]]:
    if decision.label == ActionTag.no_action:
        annotation_key = (0.0, 0, 0)
    else:
        annotation = AiAnnotation(
            file=decision.file,
            line=decision.line,
            timecode=decision.timecode,
            label=decision.label,
            side=decision.side,
            confidence=decision.ai_confidence,
            evidence_text=decision.evidence_text,
            reason=decision.ai_reason,
            chunk_index=0,
            chunk_start_line=decision.line,
            chunk_end_line=decision.line,
            current_text=decision.current_text,
        )
        annotation_key = _annotation_sort_key(annotation)
    return (_STATUS_PRIORITY[decision.review_status], decision.ai_confidence, annotation_key)


def _collapse_reviewed_decisions(candidates: list[dict[str, Any]]) -> tuple[dict[str, dict[int, ReviewedLineDecision]], int]:
    grouped: dict[str, dict[int, list[ReviewedLineDecision]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        decision = _decision_from_candidate(candidate)
        if decision is None:
            continue
        grouped[decision.file][decision.line].append(decision)

    selected: dict[str, dict[int, ReviewedLineDecision]] = defaultdict(dict)
    dropped = 0
    for file, by_line in grouped.items():
        for line, decisions in by_line.items():
            positives = [decision for decision in decisions if decision.label != ActionTag.no_action]
            chosen_pool = positives or decisions
            chosen = max(chosen_pool, key=_decision_sort_key)
            selected[file][line] = chosen
            dropped += max(0, len(decisions) - 1)
    return selected, dropped


def build_review_training_examples(
    *,
    rows_by_file: dict[str, list[Any]],
    selected_by_file: dict[str, dict[int, ReviewedLineDecision]],
    symbol: str,
    market_price: float,
) -> list[dict[str, Any]]:
    interpreter = RuleBasedTradeInterpreter()
    examples: list[dict[str, Any]] = []

    for file in sorted(rows_by_file):
        session = StreamSession(
            config=SessionConfig(symbol=symbol, enable_ai_fallback=False),
            market=MarketSnapshot(symbol=symbol, last_price=market_price),
        )
        selected_by_line = selected_by_file.get(file, {})
        for row in rows_by_file[file]:
            normalized = _normalize(row.text)
            state_before = interpreter._get_state(session.id, mutate_state=False)
            analysis_text = interpreter._analysis_text(state_before, text=normalized, received_at=row.received_at)
            entry_text = interpreter._entry_text(state_before, text=normalized, received_at=row.received_at)
            decision = selected_by_line.get(row.line)
            if decision is not None:
                envelope = IntentContextEnvelope(
                    symbol=symbol,
                    current_text=row.text,
                    current_normalized=normalized,
                    recent_text=state_before.recent_text,
                    analysis_text=analysis_text,
                    entry_text=entry_text,
                    position_side=session.position.side if session.position is not None else None,
                    last_side=state_before.last_side,
                    market_price=market_price,
                )
                examples.append(
                    {
                        "file": decision.file,
                        "line": decision.line,
                        "timecode": decision.timecode,
                        "timestamp": row.received_at.isoformat(),
                        "label": decision.label.value,
                        "source": decision.source,
                        "current_text": row.text,
                        "analysis_text": analysis_text,
                        "entry_text": entry_text,
                        "prompt": envelope.render(),
                        "symbol": symbol,
                        "position_side": _side_value(session.position.side if session.position is not None else None, default="FLAT"),
                        "last_side": _side_value(state_before.last_side, default="NONE"),
                        "original_label": decision.original_label,
                        "original_side": decision.original_side,
                        "review_status": decision.review_status,
                        "review_note": decision.review_note,
                        "ai_confidence": decision.ai_confidence,
                        "ai_reason": decision.ai_reason,
                        "evidence_text": decision.evidence_text,
                    }
                )

            state = interpreter._get_state(session.id, mutate_state=True)
            state.recent_text = normalized
            state.recent_text_at = row.received_at
            if decision is not None and decision.label != ActionTag.no_action:
                _apply_annotation_state(
                    session=session,
                    state=state,
                    annotation=AiAnnotation(
                        file=decision.file,
                        line=decision.line,
                        timecode=decision.timecode,
                        label=decision.label,
                        side=decision.side,
                        confidence=decision.ai_confidence,
                        evidence_text=decision.evidence_text,
                        reason=decision.ai_reason,
                        chunk_index=0,
                        chunk_start_line=decision.line,
                        chunk_end_line=decision.line,
                        current_text=decision.current_text,
                    ),
                )
    return examples


def _run_export(args: argparse.Namespace) -> int:
    state = _load_json(args.review)
    candidates = state.get("candidates", [])
    selected_by_file, dropped_conflicts = _collapse_reviewed_decisions(candidates)
    if not selected_by_file:
        print("No reviewed candidates found. Review candidates before exporting.")
        return 1

    rows_by_file = {
        file: _timestamped_lines(Path(file))
        for file in sorted(selected_by_file)
    }
    examples = build_review_training_examples(
        rows_by_file=rows_by_file,
        selected_by_file=selected_by_file,
        symbol=str(state.get("symbol") or "MNQ 03-26"),
        market_price=float(state.get("market_price") or 0.0),
    )
    counts = Counter(example["label"] for example in examples)
    source_counts = Counter(example["source"] for example in examples)
    pending = sum(1 for candidate in candidates if candidate.get("review_status", _REVIEW_PENDING) == _REVIEW_PENDING)
    reviewed = len(candidates) - pending
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "review_path": str(args.review.resolve()),
        "symbol": state.get("symbol"),
        "market_price": state.get("market_price"),
        "reviewed_candidates": reviewed,
        "pending_candidates": pending,
        "exported_examples": len(examples),
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "dropped_same_line_conflicts": dropped_conflicts,
    }
    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} reviewed examples to {args.jsonl_out}")
    print(f"Wrote review summary to {args.summary_out}")
    for label, count in sorted(counts.items()):
        print(f"{label}\t{count}")
    return 0


def _main() -> int:
    args = _parse_args()
    if args.command == "auto":
        return _run_auto(args)
    if args.command == "review":
        return _run_review(args)
    if args.command == "export":
        return _run_export(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(_main())
