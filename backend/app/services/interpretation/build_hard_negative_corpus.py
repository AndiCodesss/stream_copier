from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import ActionTag
from app.services.interpretation.intent_context import IntentContextEnvelope
from app.services.interpretation.path_utils import canonicalize_existing_path
from app.services.interpretation.transcript_normalizer import apply_trading_asr_corrections

_TIME_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
_DATE_IN_NAME_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_ALT_DATE_IN_NAME_PATTERN = re.compile(r"(\d{2})\.(\d{2})\.(\d{2})")
_LONG_HINT_PATTERN = re.compile(r"\b(?:long|buyer|buying|buy)\b")
_SHORT_HINT_PATTERN = re.compile(r"\b(?:short|seller|selling|sell)\b")

_PATTERN_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "retrospective_entry_loss",
        (
            r"\bi thought\b.*\b(?:put|got)\s+(?:a\s+)?(?:small|little)?\s*(?:piece|peace)\s+on\b",
            r"\b(?:put|got)\s+(?:a\s+)?(?:small|little)?\s*(?:piece|peace)\s+on\b.*\b(?:took|take)\s+a\s+loss\b",
            r"\bdidn t stick to my rules\b",
        ),
    ),
    (
        "retrospective_position_story",
        (
            r"\bremember\b.*\b(?:i was|we were|we re|we are)\s+(?:long|short)\b",
            r"\bi was (?:long|short)\s+on\b",
            r"\bwe(?: re| are)\s+(?:long|short)\s+from\b",
            r"\blong from down here\b",
            r"\bshort from up here\b",
            r"\bgot knocked out\b",
        ),
    ),
    (
        "advisory_or_hypothetical",
        (
            r"\bdon t want to be (?:a buyer|a seller|buying|selling|entering (?:long|short))\b",
            r"\bthe only time we can be a buyer\b",
            r"\bi ll look for a reload\b",
            r"\byou could (?:be|get|go) (?:long|short)\b",
            r"\bif you(?: re| are| were)\s+(?:long|short)\b",
        ),
    ),
)


@dataclass(frozen=True)
class _TranscriptRow:
    line_number: int
    timecode: str
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reviewed hard-negative corpus from transcript commentary patterns.")
    parser.add_argument("transcripts_dir", type=Path, help="Directory containing transcript .txt files.")
    parser.add_argument(
        "reviewed_inputs",
        nargs="+",
        help="Reviewed corpus directories or reviewed_examples.jsonl files used to exclude already-reviewed lines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interpretation/hard_negative_mine_round1"),
        help="Directory where reviewed_examples.jsonl and reviewed_examples.summary.json are written.",
    )
    parser.add_argument(
        "--exclude-file",
        action="append",
        default=[],
        help="Transcript basename to exclude. Repeat the flag to exclude multiple files.",
    )
    parser.add_argument(
        "--max-per-pattern",
        type=int,
        default=40,
        help="Maximum accepted rows to keep per hard-negative pattern family.",
    )
    return parser.parse_args()


def _resolve_reviewed_jsonl_inputs(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            candidate = path / "reviewed_examples.jsonl"
            if candidate.is_file():
                resolved.append(candidate.resolve())
            continue
        if path.is_file() and path.suffix == ".jsonl":
            resolved.append(path.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in resolved:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _normalize_path(raw: str, *, dataset_path: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        return str(canonicalize_existing_path(path))
    candidates = (
        Path.cwd() / path,
        Path.cwd() / "backend" / path,
        dataset_path.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(canonicalize_existing_path(candidate.resolve()))
    return str(canonicalize_existing_path((Path.cwd() / path).resolve()))


def _load_reviewed_line_exclusions(reviewed_paths: list[Path]) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    reviewed_lines: set[tuple[str, int]] = set()
    positive_guard_lines: set[tuple[str, int]] = set()
    for reviewed_path in reviewed_paths:
        with reviewed_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                example = json.loads(raw_line)
                file_path = _normalize_path(str(example.get("file", "")), dataset_path=reviewed_path)
                line_number = int(example.get("line", 0))
                key = (file_path, line_number)
                reviewed_lines.add(key)
                if str(example.get("label")) != ActionTag.no_action.value:
                    for candidate_line in range(line_number - 1, line_number + 2):
                        if candidate_line > 0:
                            positive_guard_lines.add((file_path, candidate_line))
    return reviewed_lines, positive_guard_lines


def _load_transcript_rows(path: Path) -> list[_TranscriptRow]:
    rows: list[_TranscriptRow] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = _TIME_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        hh, mm, ss, text = match.groups()
        rows.append(_TranscriptRow(line_number=line_number, timecode=f"{hh}:{mm}:{ss}", text=text))
    return rows


def _base_date_for_path(path: Path) -> datetime:
    match = _DATE_IN_NAME_PATTERN.search(path.name)
    if match is not None:
        year, month, day = (int(part) for part in match.groups())
        return datetime(year, month, day, tzinfo=UTC)
    alt_match = _ALT_DATE_IN_NAME_PATTERN.search(path.name)
    if alt_match is not None:
        day, month, short_year = (int(part) for part in alt_match.groups())
        return datetime(2000 + short_year, month, day, tzinfo=UTC)
    return datetime(2026, 1, 1, tzinfo=UTC)


def _context_window(rows: list[_TranscriptRow], index: int, *, before: int, after: int) -> list[str]:
    start = max(0, index - before)
    stop = min(len(rows), index + after + 1)
    return [rows[position].text for position in range(start, stop)]


def _infer_side(text: str) -> tuple[str, str]:
    normalized = apply_trading_asr_corrections(text)
    has_long = _LONG_HINT_PATTERN.search(normalized) is not None
    has_short = _SHORT_HINT_PATTERN.search(normalized) is not None
    if has_long and not has_short:
        return "LONG", "LONG"
    if has_short and not has_long:
        return "SHORT", "SHORT"
    return "FLAT", "NONE"


def _pattern_family(text: str) -> str | None:
    normalized = apply_trading_asr_corrections(text)
    for family, patterns in _PATTERN_GROUPS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return family
    return None


def _build_example(*, path: Path, row_index: int, rows: list[_TranscriptRow], pattern_family: str) -> dict:
    row = rows[row_index]
    recent_lines = _context_window(rows, row_index, before=2, after=0)
    analysis_lines = _context_window(rows, row_index, before=2, after=1)
    entry_lines = _context_window(rows, row_index, before=1, after=0)
    recent_text = apply_trading_asr_corrections(" ".join(recent_lines[:-1])) if len(recent_lines) > 1 else None
    analysis_text = apply_trading_asr_corrections(" ".join(analysis_lines))
    entry_text = apply_trading_asr_corrections(" ".join(entry_lines))
    current_normalized = apply_trading_asr_corrections(row.text)
    position_side, last_side = _infer_side(" ".join(analysis_lines))
    base_date = _base_date_for_path(path)
    hh, mm, ss = (int(part) for part in row.timecode.split(":"))
    timestamp = base_date.replace(hour=hh, minute=mm, second=ss).isoformat()
    prompt = IntentContextEnvelope(
        symbol="MNQ 03-26",
        current_text=row.text,
        current_normalized=current_normalized,
        recent_text=recent_text,
        analysis_text=analysis_text,
        entry_text=entry_text,
        position_side=position_side,
        last_side=last_side,
        market_price=24600.0,
    ).render()
    return {
        "file": str(canonicalize_existing_path(path.resolve())),
        "line": row.line_number,
        "timecode": row.timecode,
        "timestamp": timestamp,
        "label": ActionTag.no_action.value,
        "source": "ai_review_hard_negative",
        "current_text": row.text,
        "analysis_text": analysis_text,
        "entry_text": entry_text,
        "prompt": prompt,
        "symbol": "MNQ 03-26",
        "position_side": position_side,
        "last_side": last_side,
        "review_status": "accepted",
        "review_note": f"manual hard negative: {pattern_family}",
        "ai_confidence": 1.0,
        "evidence_text": row.text,
        "candidate_family": pattern_family,
        "classifier_label_note": f"hard_negative:{pattern_family}",
    }


def build_hard_negative_corpus(
    *,
    transcripts_dir: Path,
    reviewed_paths: list[Path],
    excluded_file_names: set[str],
    max_per_pattern: int,
) -> tuple[list[dict], dict]:
    reviewed_lines, positive_guard_lines = _load_reviewed_line_exclusions(reviewed_paths)
    examples: list[dict] = []
    family_counts = Counter()
    for path in sorted(transcripts_dir.glob("*.txt")):
        canonical_path = canonicalize_existing_path(path.resolve())
        if path.name in excluded_file_names:
            continue
        rows = _load_transcript_rows(path)
        for row_index, row in enumerate(rows):
            pattern_family = _pattern_family(row.text)
            if pattern_family is None or family_counts[pattern_family] >= max_per_pattern:
                continue
            key = (str(canonical_path), row.line_number)
            if key in reviewed_lines or key in positive_guard_lines:
                continue
            examples.append(_build_example(path=path, row_index=row_index, rows=rows, pattern_family=pattern_family))
            family_counts.update([pattern_family])
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "reviewed_input_count": len(reviewed_paths),
        "excluded_files": sorted(excluded_file_names),
        "max_per_pattern": max_per_pattern,
        "exported_examples": len(examples),
        "candidate_family_counts": dict(sorted(family_counts.items())),
    }
    return examples, summary


def _main() -> int:
    args = _parse_args()
    reviewed_paths = _resolve_reviewed_jsonl_inputs(args.reviewed_inputs)
    if not reviewed_paths:
        print("No reviewed_examples.jsonl inputs found.", flush=True)
        return 1
    examples, summary = build_hard_negative_corpus(
        transcripts_dir=args.transcripts_dir.resolve(),
        reviewed_paths=reviewed_paths,
        excluded_file_names=set(args.exclude_file),
        max_per_pattern=max(1, args.max_per_pattern),
    )
    if not examples:
        print("No hard-negative examples were produced.", flush=True)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_path = args.output_dir / "reviewed_examples.jsonl"
    with reviewed_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    (args.output_dir / "reviewed_examples.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(examples)} hard-negative examples to {reviewed_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
