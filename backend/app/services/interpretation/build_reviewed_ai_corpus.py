from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from app.core.config import get_settings
from app.services.interpretation.ai_annotation_review import (
    _auto_review_state,
    _collapse_reviewed_decisions,
    _save_review_state,
    build_review_training_examples,
    load_or_initialize_review,
)
from app.services.interpretation.ai_transcript_annotator import (
    ClaudeCliAnnotator,
    GeminiCliAnnotator,
    GeminiTranscriptAnnotator,
    TranscriptChunkAnnotator,
    _annotate_chunks,
    _build_file_reports,
    _iter_transcript_files,
    _timestamped_lines,
    build_candidate_chunks,
    build_transcript_chunks,
    merge_chunk_annotations,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reviewed AI transcript-training corpus across many transcript files.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["../transcripts"],
        help="Transcript files or directories. Defaults to ../transcripts from backend/.",
    )
    parser.add_argument("--symbol", default="MNQ 03-26")
    parser.add_argument("--market-price", type=float, default=24600.0)
    parser.add_argument(
        "--backend",
        choices=["gemini_cli", "gemini", "claude"],
        default="gemini_cli",
        help=(
            "LLM backend: 'gemini_cli' uses the Gemini CLI with Google login, "
            "'gemini' uses the Gemini API key, and 'claude' uses the Claude CLI."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model override. Recommended defaults: gemini_cli=gemini-2.5-pro, "
            "claude=sonnet, gemini=<your configured Gemini API model>."
        ),
    )
    parser.add_argument("--chunk-lines", type=int, default=400)
    parser.add_argument("--overlap-lines", type=int, default=24)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Only send trade-like candidate windows instead of processing the full transcript coverage.",
    )
    parser.add_argument("--candidate-context-before", type=int, default=6)
    parser.add_argument("--candidate-context-after", type=int, default=6)
    parser.add_argument("--candidate-merge-gap", type=int, default=18)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--max-transcript-concurrency",
        type=int,
        default=1,
        help="Number of transcripts to process in parallel. Each transcript still uses its own chunk concurrency.",
    )
    parser.add_argument(
        "--min-request-interval-ms",
        type=int,
        default=5500,
        help="Pacing interval between annotator requests to avoid rate limits.",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interpretation/full_ai_corpus"),
        help="Directory for per-file reports, combined review state, and exports.",
    )
    parser.add_argument(
        "--overwrite-reports",
        action="store_true",
        help="Re-run annotation even if a per-file report already exists.",
    )
    parser.add_argument(
        "--exclude-reviewed-corpora",
        nargs="*",
        default=[],
        help=(
            "Reviewed corpus directories or files to exclude from this run. "
            "Directories resolve to reviewed_examples.jsonl first, then combined_review.json."
        ),
    )
    return parser.parse_args()


def _safe_report_name(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}__{digest}.json"


def _report_payload(*, model: str, symbol: str, market_price: float, report) -> dict:
    counts = Counter(annotation.label.value for annotation in report.annotations)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "symbol": symbol,
        "market_price": market_price,
        "files": [
            {
                "file": report.file,
                "total_rows": report.total_rows,
                "chunk_count": report.chunk_count,
                "errors": list(report.errors),
                "annotations": [annotation.model_dump() if hasattr(annotation, "model_dump") else _annotation_dict(annotation) for annotation in report.annotations],
            }
        ],
        "counts": dict(sorted(counts.items())),
        "example_count": sum(counts.values()),
    }


def _annotation_dict(annotation) -> dict:
    return {
        "file": annotation.file,
        "line": annotation.line,
        "timecode": annotation.timecode,
        "label": annotation.label.value,
        "side": annotation.side.value if annotation.side is not None else None,
        "confidence": annotation.confidence,
        "evidence_text": annotation.evidence_text,
        "reason": annotation.reason,
        "chunk_index": annotation.chunk_index,
        "chunk_start_line": annotation.chunk_start_line,
        "chunk_end_line": annotation.chunk_end_line,
        "current_text": annotation.current_text,
    }


async def _annotate_file(
    *,
    annotator: TranscriptChunkAnnotator,
    path: Path,
    rows: list,
    chunks: list,
    symbol: str,
    market_price: float,
    max_concurrency: int,
) -> dict:
    rows_by_file = {str(path): rows}
    chunks_by_file = {str(path): chunks}
    results = await _annotate_chunks(
        annotator=annotator,
        chunks=chunks_by_file[str(path)],
        symbol=symbol,
        market_price=market_price,
        max_concurrency=max_concurrency,
    )
    merged_annotations = merge_chunk_annotations(results)
    report = _build_file_reports(
        files=[path],
        rows_by_file=rows_by_file,
        chunks_by_file=chunks_by_file,
        merged_annotations=merged_annotations,
        results=results,
    )[0]
    return _report_payload(
        model=annotator.model_name,
        symbol=symbol,
        market_price=market_price,
        report=report,
    )


async def _annotate_file_to_report(
    *,
    annotator: TranscriptChunkAnnotator,
    path: Path,
    symbol: str,
    market_price: float,
    max_concurrency: int,
    report_path: Path,
    args: argparse.Namespace,
) -> dict:
    rows, chunks = _build_chunks_for_path(path=path, args=args)
    print(
        f"start {path.name} rows={len(rows)} chunks={len(chunks)}",
        flush=True,
    )
    payload = await _annotate_file(
        annotator=annotator,
        path=path,
        rows=rows,
        chunks=chunks,
        symbol=symbol,
        market_price=market_price,
        max_concurrency=max_concurrency,
    )
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_chunks_for_path(*, path: Path, args: argparse.Namespace) -> tuple[list, list]:
    rows = _timestamped_lines(path)
    if args.candidate_only:
        chunks = build_candidate_chunks(
            file=str(path),
            rows=rows,
            chunk_lines=args.chunk_lines,
            max_chars=args.max_chars,
            context_before=args.candidate_context_before,
            context_after=args.candidate_context_after,
            merge_gap_lines=args.candidate_merge_gap,
        )
    else:
        chunks = build_transcript_chunks(
            file=str(path),
            rows=rows,
            chunk_lines=args.chunk_lines,
            overlap_lines=args.overlap_lines,
            max_chars=args.max_chars,
        )
    return rows, chunks


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_file_key(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _iter_review_sources(inputs: Iterable[str]) -> list[Path]:
    sources: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            reviewed_jsonl = path / "reviewed_examples.jsonl"
            combined_review = path / "combined_review.json"
            if reviewed_jsonl.exists():
                sources.append(reviewed_jsonl)
                continue
            if combined_review.exists():
                sources.append(combined_review)
            continue
        if path.is_file():
            sources.append(path)
    return sorted(dict.fromkeys(sources))


def _load_reviewed_coverage(inputs: Iterable[str]) -> set[str]:
    covered: set[str] = set()
    for source in _iter_review_sources(inputs):
        if source.suffix == ".jsonl":
            with source.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    payload = json.loads(raw_line)
                    file_value = payload.get("file")
                    if isinstance(file_value, str) and file_value.strip():
                        covered.add(_normalize_file_key(file_value))
            continue

        payload = json.loads(source.read_text(encoding="utf-8"))
        for candidate in payload.get("candidates", []):
            if candidate.get("review_status") not in {"accepted", "corrected"}:
                continue
            file_value = candidate.get("file")
            if isinstance(file_value, str) and file_value.strip():
                covered.add(_normalize_file_key(file_value))
    return covered


def _aggregate_reports(*, report_paths: list[Path], combined_report_path: Path) -> dict:
    files_payload = []
    counts = Counter()
    model = None
    symbol = None
    market_price = None
    for report_path in report_paths:
        payload = _load_report(report_path)
        model = model or payload.get("model")
        symbol = symbol or payload.get("symbol")
        market_price = market_price if market_price is not None else payload.get("market_price")
        for file_report in payload.get("files", []):
            files_payload.append(file_report)
        counts.update(payload.get("counts", {}))
    combined = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "symbol": symbol,
        "market_price": market_price,
        "files": files_payload,
        "counts": dict(sorted(counts.items())),
        "example_count": sum(counts.values()),
    }
    combined_report_path.parent.mkdir(parents=True, exist_ok=True)
    combined_report_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def _export_combined_review(*, review_path: Path, jsonl_out: Path, summary_out: Path) -> dict:
    state = json.loads(review_path.read_text(encoding="utf-8"))
    candidates = state.get("candidates", [])
    selected_by_file, dropped_conflicts = _collapse_reviewed_decisions(candidates)
    rows_by_file = {file: _timestamped_lines(Path(file)) for file in sorted(selected_by_file)}
    examples = build_review_training_examples(
        rows_by_file=rows_by_file,
        selected_by_file=selected_by_file,
        symbol=str(state.get("symbol") or "MNQ 03-26"),
        market_price=float(state.get("market_price") or 0.0),
    )
    counts = Counter(example["label"] for example in examples)
    source_counts = Counter(example["source"] for example in examples)
    pending = sum(1 for candidate in candidates if candidate.get("review_status") == "pending")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "review_path": str(review_path.resolve()),
        "reviewed_candidates": len(candidates) - pending,
        "pending_candidates": pending,
        "exported_examples": len(examples),
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "dropped_same_line_conflicts": dropped_conflicts,
    }
    jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


async def _main_async(args: argparse.Namespace) -> int:
    settings = get_settings()

    annotator = _build_annotator(args=args, settings=settings)
    if not annotator.is_available():
        print(f"Annotator backend '{args.backend}' is not available.")
        return 1

    files = _iter_transcript_files(args.inputs)
    excluded = _load_reviewed_coverage(args.exclude_reviewed_corpora)
    if excluded:
        original_count = len(files)
        files = [path for path in files if _normalize_file_key(path) not in excluded]
        print(
            f"Excluded {original_count - len(files)} transcripts already covered by reviewed corpora.",
            flush=True,
        )
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print("No transcript files found.")
        return 1

    output_dir = args.output_dir
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = [reports_dir / _safe_report_name(path) for path in files]
    completed = 0
    completed_lock = asyncio.Lock()
    print(
        f"Starting reviewed corpus build for {len(files)} transcripts "
        f"model={annotator.model_name} chunk_lines={args.chunk_lines} max_chars={args.max_chars} "
        f"candidate_only={bool(args.candidate_only)} min_interval_ms={args.min_request_interval_ms} "
        f"max_concurrency={args.max_concurrency} "
        f"max_transcript_concurrency={args.max_transcript_concurrency}",
        flush=True,
    )
    try:
        transcript_semaphore = asyncio.Semaphore(max(1, args.max_transcript_concurrency))

        async def _process_file(index: int, path: Path, report_path: Path) -> None:
            nonlocal completed
            if report_path.exists() and not args.overwrite_reports:
                async with completed_lock:
                    completed += 1
                    print(f"[{completed}/{len(files)}] reuse {path.name}", flush=True)
                return

            async with transcript_semaphore:
                print(f"[{index}/{len(files)}] queue {path.name}", flush=True)
                payload = await _annotate_file_to_report(
                    annotator=annotator,
                    path=path,
                    symbol=args.symbol,
                    market_price=args.market_price,
                    max_concurrency=args.max_concurrency,
                    report_path=report_path,
                    args=args,
                )
            file_report = payload["files"][0]
            async with completed_lock:
                completed += 1
                print(
                    f"[{completed}/{len(files)}] annotated {path.name} "
                    f"annotations={len(file_report['annotations'])} errors={len(file_report['errors'])}",
                    flush=True,
                )

        tasks = [
            asyncio.create_task(_process_file(index, path, report_path))
            for index, (path, report_path) in enumerate(zip(files, report_paths, strict=False), start=1)
        ]
        for task in asyncio.as_completed(tasks):
            await task
    finally:
        await annotator.close()

    combined_report_path = output_dir / "combined_annotations.json"
    combined = _aggregate_reports(report_paths=report_paths, combined_report_path=combined_report_path)
    review_path = output_dir / "combined_review.json"
    state = load_or_initialize_review(report_path=combined_report_path, review_path=review_path)
    counts = _auto_review_state(state, overwrite_reviewed=False)
    _save_review_state(review_path, state)

    reviewed_jsonl = output_dir / "reviewed_examples.jsonl"
    reviewed_summary = output_dir / "reviewed_examples.summary.json"
    summary = _export_combined_review(
        review_path=review_path,
        jsonl_out=reviewed_jsonl,
        summary_out=reviewed_summary,
    )

    print(f"Wrote combined annotation report to {combined_report_path}", flush=True)
    print(f"Wrote combined review state to {review_path}", flush=True)
    print(
        "auto-review "
        f"accepted={counts['accepted']} corrected={counts['corrected']} "
        f"rejected={counts['rejected']} pending={counts['pending']}",
        flush=True,
    )
    print(
        f"Wrote {summary['exported_examples']} reviewed examples to {reviewed_jsonl} "
        f"(pending={summary['pending_candidates']})",
        flush=True,
    )
    return 0


def _build_annotator(*, args: argparse.Namespace, settings) -> TranscriptChunkAnnotator:
    if args.backend == "claude":
        return ClaudeCliAnnotator(
            model=args.model or "sonnet",
            min_request_interval_seconds=max(0.0, args.min_request_interval_ms / 1000.0),
        )
    if args.backend == "gemini_cli":
        return GeminiCliAnnotator(
            model=args.model or "gemini-2.5-pro",
            min_request_interval_seconds=max(0.0, args.min_request_interval_ms / 1000.0),
        )
    return GeminiTranscriptAnnotator(
            settings,
            model_name=args.model,
            min_request_interval_seconds=max(0.0, args.min_request_interval_ms / 1000.0),
        )


def _main() -> int:
    args = _parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(_main())
