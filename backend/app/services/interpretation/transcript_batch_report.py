from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.models.domain import ActionTag, MarketSnapshot, PositionState, SessionConfig, StreamSession, TranscriptSegment
from app.services.interpretation.rule_engine import RuleBasedTradeInterpreter

TIMESTAMPED_LINE_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
SUSPICIOUS_INTENT_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\blooking for (?:the |a )?(?:long|short)\b",
        r"\binterested in (?:the |a )?(?:long|short)\b",
        r"\btempted to put\b",
        r"\bthink about\b",
        r"\bif you\b",
        r"\byou can\b",
        r"\byou could\b",
        r"\bwould i\b",
        r"\bthey are\b",
        r"\bthey re\b",
        r"\bhaven t\b",
        r"\bnot paying myself\b",
        r"\bone break ?even\b",
        r"\bi took the\b",
        r"\bi was\b",
        r"\bwe ve had\b",
        r"\bmy swings?\b",
        r"\bsoftware companies\b",
    ]
]


@dataclass
class IntentHit:
    line: int
    tag: str
    text: str


@dataclass
class FileReport:
    file: str
    total_intents: int
    counts: dict[str, int]
    suspicious_hits: list[IntentHit]
    first_hits: list[IntentHit]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay transcript files through the rule interpreter.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["../transcripts"],
        help="Transcript files or directories. Defaults to ../transcripts from backend/.",
    )
    parser.add_argument("--symbol", default="MNQ 03-26", help="Session symbol used for replay.")
    parser.add_argument("--market-price", type=float, default=24600.0, help="Anchor market price for shorthand resolution.")
    parser.add_argument("--first-hit-limit", type=int, default=12, help="How many first intent hits to include per file.")
    parser.add_argument("--suspicious-limit", type=int, default=8, help="How many suspicious hits to include per file.")
    parser.add_argument("--json-out", type=Path, help="Optional path to write the full report as JSON.")
    return parser.parse_args()


def _iter_transcript_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(item for item in path.iterdir() if item.is_file() and item.suffix == ".txt"))
            continue
        if path.is_file() and path.suffix == ".txt":
            files.append(path)
    return sorted(dict.fromkeys(files))


def _suspicious_intent(text: str) -> bool:
    normalized = text.lower()
    return any(pattern.search(normalized) for pattern in SUSPICIOUS_INTENT_PATTERNS)


def _tag_value(tag: ActionTag | str) -> str:
    return tag.value if isinstance(tag, ActionTag) else str(tag)


async def _replay_file(
    path: Path,
    *,
    symbol: str,
    market_price: float,
    first_hit_limit: int,
    suspicious_limit: int,
) -> FileReport:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(symbol=symbol),
        market=MarketSnapshot(symbol=symbol, last_price=market_price),
    )
    counts: Counter[str] = Counter()
    first_hits: list[IntentHit] = []
    suspicious_hits: list[IntentHit] = []

    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)
    if date_match is not None:
        base_date = datetime(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
            tzinfo=UTC,
        )
    else:
        base_date = datetime(2026, 3, 1, tzinfo=UTC)

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TIMESTAMPED_LINE_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        hh, mm, ss, text = match.groups()
        segment = TranscriptSegment(
            session_id=session.id,
            text=text,
            received_at=base_date.replace(hour=int(hh), minute=int(mm), second=int(ss)),
        )
        intent = await interpreter.interpret(session, segment)
        if intent is None:
            continue

        tag = _tag_value(intent.tag)
        counts[tag] += 1
        hit = IntentHit(line=line_number, tag=tag, text=text)
        if len(first_hits) < first_hit_limit:
            first_hits.append(hit)
        if _suspicious_intent(text) and len(suspicious_hits) < suspicious_limit:
            suspicious_hits.append(hit)

        if tag in {ActionTag.enter_long.value, ActionTag.enter_short.value} and intent.side is not None:
            session.position = PositionState(
                side=intent.side,
                quantity=1,
                average_price=intent.entry_price or session.market.last_price or 0.0,
                stop_price=intent.stop_price,
                target_price=intent.target_price,
            )
        elif tag == ActionTag.add.value and session.position is None and intent.side is not None:
            session.position = PositionState(
                side=intent.side,
                quantity=1,
                average_price=intent.entry_price or session.market.last_price or 0.0,
                stop_price=intent.stop_price,
                target_price=intent.target_price,
            )
        elif tag == ActionTag.exit_all.value:
            session.position = None
        elif tag in {ActionTag.move_stop.value, ActionTag.move_to_breakeven.value} and session.position is not None:
            if tag == ActionTag.move_stop.value and intent.stop_price is not None:
                session.position.stop_price = intent.stop_price
            elif tag == ActionTag.move_to_breakeven.value:
                session.position.stop_price = session.position.average_price

    return FileReport(
        file=str(path),
        total_intents=sum(counts.values()),
        counts=dict(sorted(counts.items())),
        suspicious_hits=suspicious_hits,
        first_hits=first_hits,
    )


async def _main() -> int:
    args = _parse_args()
    files = _iter_transcript_files(args.inputs)
    if not files:
        print("No transcript files found.")
        return 1

    reports = [
        await _replay_file(
            path,
            symbol=args.symbol,
            market_price=args.market_price,
            first_hit_limit=args.first_hit_limit,
            suspicious_limit=args.suspicious_limit,
        )
        for path in files
    ]

    aggregate_counts: Counter[str] = Counter()
    for report in reports:
        aggregate_counts.update(report.counts)

    sorted_reports = sorted(
        reports,
        key=lambda report: (len(report.suspicious_hits), report.total_intents),
        reverse=True,
    )

    print("FILES")
    for report in sorted_reports:
        suspicious_count = len(report.suspicious_hits)
        count_summary = ", ".join(f"{tag}={count}" for tag, count in report.counts.items()) or "no intents"
        print(f"{Path(report.file).name}\tintents={report.total_intents}\tsuspicious={suspicious_count}\t{count_summary}")
        for hit in report.suspicious_hits:
            print(f"  line {hit.line}: {hit.tag} | {hit.text}")

    print("AGGREGATE")
    for tag, count in sorted(aggregate_counts.items()):
        print(f"{tag}\t{count}")
    print(f"TOTAL\t{sum(aggregate_counts.values())}")

    if args.json_out is not None:
        payload = {
            "reports": [
                {
                    "file": report.file,
                    "total_intents": report.total_intents,
                    "counts": report.counts,
                    "suspicious_hits": [asdict(hit) for hit in report.suspicious_hits],
                    "first_hits": [asdict(hit) for hit in report.first_hits],
                }
                for report in reports
            ],
            "aggregate_counts": dict(sorted(aggregate_counts.items())),
            "total_intents": sum(aggregate_counts.values()),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
