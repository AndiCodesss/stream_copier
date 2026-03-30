"""Clean up training_data.jsonl: fix position state, remove placeholder prices,
deduplicate, and flag suspicious examples."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_DEFAULT_INPUT = Path("data/training_data.jsonl")

# Labels that require an open position
_POSITION_REQUIRED_LABELS = {"TRIM", "EXIT_ALL", "MOVE_STOP", "MOVE_TO_BREAKEVEN"}

# Second-person advice patterns (case-insensitive)
_ADVICE_PATTERNS = [
    re.compile(r"\byou can\b", re.IGNORECASE),
    re.compile(r"\byou could\b", re.IGNORECASE),
    re.compile(r"\byou should\b", re.IGNORECASE),
    re.compile(r"\byour stop\b", re.IGNORECASE),
    re.compile(r"\byou must\b", re.IGNORECASE),
    re.compile(r"\bmove your\b", re.IGNORECASE),
    re.compile(r"\byou(?:'|')d\b", re.IGNORECASE),
]

# Action labels (not NO_ACTION) — these are where advice patterns matter
_ACTION_LABELS = {"ENTER_LONG", "ENTER_SHORT", "TRIM", "EXIT_ALL", "MOVE_STOP", "MOVE_TO_BREAKEVEN"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean up training data JSONL.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output-clean", type=Path, default=Path("data/training_data_clean.jsonl"))
    parser.add_argument("--output-flagged", type=Path, default=Path("data/training_data_flagged.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/cleanup_report.json"))
    return parser.parse_args()


def _load_examples(path: Path) -> list[dict]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _rebuild_prompt(example: dict) -> str:
    """Rebuild the prompt field with corrected position_side and no market_price."""
    parts = [
        f"symbol={example.get('symbol', 'MNQ 03-26')}",
        f"position={example.get('position_side', 'FLAT')}",
        f"last_side={example.get('last_side', 'NONE')}",
    ]

    # Parse existing prompt to extract text fields (recent, analysis, current)
    prompt = example.get("prompt", "")
    for field in ("recent", "analysis", "current"):
        match = re.search(rf"^{field}=(.*)$", prompt, re.MULTILINE)
        if match:
            parts.append(f"{field}={match.group(1)}")

    return "\n".join(parts)


def _repair_position_state(examples: list[dict]) -> tuple[list[dict], int]:
    """Trust original position_side from the full corpus. For impossible combos
    (management label + FLAT), try to infer from original_side/last_side.
    Returns (fixed_examples, repair_count)."""

    repaired = 0
    result = []

    for ex in examples:
        ex = dict(ex)  # copy
        label = ex["label"]

        # If a management label has position_side=FLAT, try to repair it
        if label in _POSITION_REQUIRED_LABELS and ex.get("position_side") == "FLAT":
            # Try original_side first (set during AI labeling)
            side = ex.get("original_side")
            if side in ("LONG", "SHORT"):
                ex["position_side"] = side
                ex["position_repaired"] = True
                repaired += 1
            else:
                # Try last_side as fallback
                last = ex.get("last_side")
                if last in ("LONG", "SHORT"):
                    ex["position_side"] = last
                    ex["position_repaired"] = True
                    repaired += 1

        # Rebuild prompt with corrected fields (removes market_price)
        ex["prompt"] = _rebuild_prompt(ex)
        result.append(ex)

    return result, repaired


def _find_duplicates(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove duplicate entries for the same trade action within a short time window."""
    kept: list[dict] = []
    dupes: list[dict] = []

    for i, ex in enumerate(examples):
        if i == 0:
            kept.append(ex)
            continue

        prev = kept[-1] if kept else None
        if (
            prev is not None
            and ex["file"] == prev["file"]
            and ex["label"] == prev["label"]
            and ex["label"] in _ACTION_LABELS
            and abs(ex["line"] - prev["line"]) <= 3
        ):
            ex["flag_reason"] = "duplicate_same_action_within_3_lines"
            dupes.append(ex)
        else:
            kept.append(ex)

    return kept, dupes


def _flag_impossible_state(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flag examples where the label requires a position but position_side is FLAT."""
    clean = []
    flagged = []

    for ex in examples:
        if ex["label"] in _POSITION_REQUIRED_LABELS and ex["position_side"] == "FLAT":
            ex = dict(ex)
            ex["flag_reason"] = f"{ex['label']}_while_FLAT"
            flagged.append(ex)
        else:
            clean.append(ex)

    return clean, flagged


def _flag_advice_patterns(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Flag action-labeled examples that contain second-person advice language."""
    clean = []
    flagged = []

    for ex in examples:
        if ex["label"] not in _ACTION_LABELS:
            clean.append(ex)
            continue

        current = ex.get("current_text", "")
        matched_patterns = [p.pattern for p in _ADVICE_PATTERNS if p.search(current)]

        if matched_patterns:
            ex = dict(ex)
            ex["flag_reason"] = f"advice_pattern: {', '.join(matched_patterns)}"
            flagged.append(ex)
        else:
            clean.append(ex)

    return clean, flagged


def _write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")


def cleanup(
    input_path: Path,
    output_clean: Path,
    output_flagged: Path,
    report_path: Path,
) -> dict:
    """Run the full cleanup pipeline. Returns the report dict."""
    raw = _load_examples(input_path)
    stats: dict = {"input_examples": len(raw)}

    # Step 1: Repair position state and rebuild prompts (removes market_price)
    fixed, repaired = _repair_position_state(raw)

    prompt_changes = sum(
        1 for orig, new in zip(raw, fixed)
        if orig.get("prompt") != new.get("prompt")
    )
    stats["position_side_repaired"] = repaired
    stats["prompts_rebuilt"] = prompt_changes

    all_flagged: list[dict] = []

    # Step 2: Deduplicate
    fixed, dupes = _find_duplicates(fixed)
    all_flagged.extend(dupes)
    stats["duplicates_removed"] = len(dupes)

    # Step 3: Flag impossible state+label combos
    fixed, impossible = _flag_impossible_state(fixed)
    all_flagged.extend(impossible)
    stats["impossible_state_flagged"] = len(impossible)

    # Step 4: Flag advice patterns
    fixed, advice = _flag_advice_patterns(fixed)
    all_flagged.extend(advice)
    stats["advice_pattern_flagged"] = len(advice)

    stats["output_clean"] = len(fixed)
    stats["output_flagged"] = len(all_flagged)

    # Label distributions
    stats["clean_label_distribution"] = dict(sorted(Counter(ex["label"] for ex in fixed).items()))
    stats["flagged_label_distribution"] = dict(sorted(Counter(ex["label"] for ex in all_flagged).items()))
    stats["flag_reasons"] = dict(sorted(Counter(ex.get("flag_reason", "unknown") for ex in all_flagged).items()))

    # Write outputs
    _write_jsonl(output_clean, fixed)
    _write_jsonl(output_flagged, all_flagged)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return stats


def _main() -> int:
    args = _parse_args()
    stats = cleanup(
        input_path=args.input,
        output_clean=args.output_clean,
        output_flagged=args.output_flagged,
        report_path=args.report,
    )

    print(f"\n{'='*60}")
    print("TRAINING DATA CLEANUP REPORT")
    print(f"{'='*60}")
    print(f"Input examples:          {stats['input_examples']}")
    print(f"Position side repaired:  {stats['position_side_repaired']}")
    print(f"Prompts rebuilt:         {stats['prompts_rebuilt']}")
    print(f"Duplicates removed:      {stats['duplicates_removed']}")
    print(f"Impossible state flagged:{stats['impossible_state_flagged']}")
    print(f"Advice pattern flagged:  {stats['advice_pattern_flagged']}")
    print(f"{'='*60}")
    print(f"Clean output:            {stats['output_clean']}")
    print(f"Flagged output:          {stats['output_flagged']}")
    print(f"\nClean label distribution:")
    for label, count in stats["clean_label_distribution"].items():
        print(f"  {label:<22} {count}")
    print(f"\nFlag reasons:")
    for reason, count in stats["flag_reasons"].items():
        print(f"  {reason:<45} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
