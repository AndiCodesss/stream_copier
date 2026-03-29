from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from app.core.config import Settings, get_settings
from app.models.domain import ActionTag, MarketSnapshot, PositionState, SessionConfig, StreamSession, TradeSide
from app.services.interpretation.gemini_fallback import (
    _coerce_action_tag,
    _coerce_trade_side,
    _extract_candidate_text,
    _parse_json_payload,
)
from app.services.interpretation.action_language import (
    detect_present_trade_signal,
    detect_setup_signal,
    looks_explicit_trade_language,
)
from app.services.interpretation.candidate_detector import looks_candidate_seed
from app.services.interpretation.intent_context import IntentContextEnvelope
from app.services.interpretation.rule_engine import RuleBasedTradeInterpreter, TRADE_KEYWORDS, _normalize

TIMESTAMPED_LINE_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
_ACTIONABLE_LABELS = {
    ActionTag.setup_long,
    ActionTag.setup_short,
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.add,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
}
_ENTRY_LABELS = {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}
_LABEL_PRIORITY = {
    ActionTag.enter_long: 9,
    ActionTag.enter_short: 9,
    ActionTag.add: 8,
    ActionTag.trim: 7,
    ActionTag.exit_all: 7,
    ActionTag.move_stop: 6,
    ActionTag.move_to_breakeven: 6,
    ActionTag.setup_long: 3,
    ActionTag.setup_short: 3,
}
_LOW_SIGNAL_TRADE_KEYWORDS = {"buy", "sell", "profit", "yourself", "reclaim", "flat"}
_CANDIDATE_KEYWORDS = tuple(keyword for keyword in TRADE_KEYWORDS if keyword not in _LOW_SIGNAL_TRADE_KEYWORDS)
_ALIGNMENT_WINDOW = 10


@dataclass(frozen=True)
class TranscriptRow:
    line: int
    timecode: str
    text: str
    received_at: datetime


@dataclass(frozen=True)
class TranscriptChunk:
    file: str
    chunk_index: int
    rows: tuple[TranscriptRow, ...]

    @property
    def start_line(self) -> int:
        return self.rows[0].line

    @property
    def end_line(self) -> int:
        return self.rows[-1].line


@dataclass(frozen=True)
class AiAnnotation:
    file: str
    line: int
    timecode: str
    label: ActionTag
    side: TradeSide | None
    confidence: float
    evidence_text: str
    reason: str | None
    chunk_index: int
    chunk_start_line: int
    chunk_end_line: int
    current_text: str


@dataclass(frozen=True)
class ChunkAnnotationResult:
    file: str
    chunk_index: int
    annotations: tuple[AiAnnotation, ...]
    error: str | None = None


@dataclass(frozen=True)
class FileAnnotationReport:
    file: str
    total_rows: int
    chunk_count: int
    annotations: tuple[AiAnnotation, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotationTrainingExample:
    file: str
    line: int
    timecode: str
    timestamp: str
    label: str
    source: str
    current_text: str
    analysis_text: str
    entry_text: str
    prompt: str
    symbol: str
    position_side: str
    last_side: str
    ai_confidence: float
    ai_reason: str | None
    evidence_text: str


class TranscriptChunkAnnotator(Protocol):
    @property
    def model_name(self) -> str: ...

    def is_available(self) -> bool: ...

    async def close(self) -> None: ...

    async def annotate_chunk(
        self,
        *,
        chunk: TranscriptChunk,
        symbol: str,
        market_price: float,
    ) -> ChunkAnnotationResult: ...


class _RateLimitedAnnotator:
    def __init__(self, *, min_request_interval_seconds: float = 0.0) -> None:
        self._min_request_interval_seconds = max(0.0, min_request_interval_seconds)
        self._next_request_time = 0.0
        self._request_lock = asyncio.Lock()

    async def _wait_for_request_slot(self) -> None:
        if self._min_request_interval_seconds <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._request_lock:
            now = loop.time()
            if now < self._next_request_time:
                await asyncio.sleep(self._next_request_time - now)
                now = loop.time()
            self._next_request_time = now + self._min_request_interval_seconds


class GeminiTranscriptAnnotator(_RateLimitedAnnotator):
    def __init__(
        self,
        settings: Settings,
        *,
        model_name: str | None = None,
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        super().__init__(min_request_interval_seconds=min_request_interval_seconds)
        self._settings = settings
        self._model_name = (model_name or settings.gemini_model).strip() or settings.gemini_model
        self._client: httpx.AsyncClient | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return bool(self._settings.gemini_api_key)

    async def close(self) -> None:
        client = self._client
        if client is not None:
            await client.aclose()
            self._client = None

    async def annotate_chunk(
        self,
        *,
        chunk: TranscriptChunk,
        symbol: str,
        market_price: float,
    ) -> ChunkAnnotationResult:
        if not self.is_available():
            return ChunkAnnotationResult(
                file=chunk.file,
                chunk_index=chunk.chunk_index,
                annotations=(),
                error="GEMINI_API_KEY is not configured",
            )

        payload = await self._generate_json(
            system_text=_system_prompt(),
            user_text=_chunk_prompt(chunk=chunk, symbol=symbol, market_price=market_price),
        )
        if payload is None:
            return ChunkAnnotationResult(
                file=chunk.file,
                chunk_index=chunk.chunk_index,
                annotations=(),
                error="invalid gemini response",
            )

        annotations = _coerce_chunk_annotations(chunk=chunk, payload=payload)
        return ChunkAnnotationResult(
            file=chunk.file,
            chunk_index=chunk.chunk_index,
            annotations=tuple(annotations),
        )

    async def _generate_json(self, *, system_text: str, user_text: str) -> dict[str, Any] | None:
        payload = {
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseJsonSchema": _annotation_response_schema(),
            },
        }
        try:
            client = self._client
            if client is None:
                client = httpx.AsyncClient(
                    base_url=self._settings.gemini_base_url,
                    timeout=25.0,
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=5),
                )
                self._client = client

            attempts = 0
            while attempts < 5:
                attempts += 1
                await self._wait_for_request_slot()
                response = await client.post(
                    f"/models/{self._model_name}:generateContent",
                    params={"key": self._settings.gemini_api_key},
                    json=payload,
                )
                if response.status_code == 429:
                    if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                        print(
                            f"gemini 429 model={self._model_name} delay={_retry_delay_seconds(response):.1f}s "
                            f"detail={_retry_reason(response)}",
                            flush=True,
                        )
                    await asyncio.sleep(_retry_delay_seconds(response))
                    continue
                response.raise_for_status()
                data = response.json()
                text = _extract_candidate_text(data)
                if text is None:
                    if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                        print(f"gemini invalid-text model={self._model_name} attempt={attempts}", flush=True)
                    await asyncio.sleep(min(4.0, 1.0 + attempts))
                    continue
                parsed = _parse_json_payload(text)
                if parsed is not None:
                    return parsed
                if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                    print(f"gemini invalid-json model={self._model_name} attempt={attempts}", flush=True)
                await asyncio.sleep(min(4.0, 1.0 + attempts))
            return None
        except Exception:
            return None
        return None


class _PromptCliAnnotator(_RateLimitedAnnotator):
    def __init__(
        self,
        *,
        model: str,
        model_name: str,
        executable: str,
        unavailable_error: str,
        invalid_response_error: str,
        debug_label: str,
        min_request_interval_seconds: float = 0.0,
        max_attempts: int = 3,
        timeout_seconds: float = 180.0,
    ) -> None:
        super().__init__(min_request_interval_seconds=min_request_interval_seconds)
        self._model = model
        self._model_name = model_name
        self._executable = executable
        self._unavailable_error = unavailable_error
        self._invalid_response_error = invalid_response_error
        self._debug_label = debug_label
        self._max_attempts = max(1, int(max_attempts))
        self._timeout_seconds = max(5.0, float(timeout_seconds))

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return shutil.which(self._executable) is not None

    async def close(self) -> None:
        pass

    async def annotate_chunk(
        self,
        *,
        chunk: TranscriptChunk,
        symbol: str,
        market_price: float,
    ) -> ChunkAnnotationResult:
        if not self.is_available():
            return ChunkAnnotationResult(
                file=chunk.file,
                chunk_index=chunk.chunk_index,
                annotations=(),
                error=self._unavailable_error,
            )

        payload = await self._generate_json(
            system_text=_system_prompt(),
            user_text=_chunk_prompt(chunk=chunk, symbol=symbol, market_price=market_price),
        )
        if payload is None:
            return ChunkAnnotationResult(
                file=chunk.file,
                chunk_index=chunk.chunk_index,
                annotations=(),
                error=self._invalid_response_error,
            )

        annotations = _coerce_chunk_annotations(chunk=chunk, payload=payload)
        return ChunkAnnotationResult(
            file=chunk.file,
            chunk_index=chunk.chunk_index,
            annotations=tuple(annotations),
        )

    async def _generate_json(self, *, system_text: str, user_text: str) -> dict[str, Any] | None:
        combined_prompt = _build_cli_annotation_prompt(system_text=system_text, user_text=user_text)
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            await self._wait_for_request_slot()
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._command(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._environment(),
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=combined_prompt.encode("utf-8")),
                    timeout=self._timeout_seconds,
                )
                if proc.returncode != 0:
                    if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                        print(
                            f"{self._debug_label} error model={self.model_name} attempt={attempts} "
                            f"rc={proc.returncode} stderr={stderr.decode(errors='replace')[:200]}",
                            flush=True,
                        )
                    await asyncio.sleep(min(5.0, 1.0 + attempts))
                    continue
                parsed = self._parse_stdout(stdout.decode("utf-8"))
                if parsed is not None:
                    return parsed
                if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                    print(f"{self._debug_label} invalid-json model={self.model_name} attempt={attempts}", flush=True)
                await asyncio.sleep(min(4.0, 1.0 + attempts))
            except asyncio.TimeoutError:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    try:
                        await proc.communicate()
                    except Exception:
                        pass
                if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                    print(f"{self._debug_label} timeout model={self.model_name} attempt={attempts}", flush=True)
            except Exception as exc:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    try:
                        await proc.communicate()
                    except Exception:
                        pass
                if os.getenv("GEMINI_DEBUG_RETRIES") == "1":
                    print(f"{self._debug_label} exception model={self.model_name} attempt={attempts} {exc!r}", flush=True)
                return None
        return None

    def _environment(self) -> dict[str, str]:
        return {**os.environ}

    def _command(self) -> tuple[str, ...]:
        raise NotImplementedError

    def _parse_stdout(self, stdout_text: str) -> dict[str, Any] | None:
        return _parse_json_payload(_strip_code_fences(stdout_text))


class ClaudeCliAnnotator(_PromptCliAnnotator):
    """Annotator backend that uses ``claude -p`` (Claude Code CLI) instead of Gemini API."""

    def __init__(
        self,
        *,
        model: str = "sonnet",
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            model_name=f"claude-{model}",
            executable="claude",
            unavailable_error="claude CLI not found in PATH",
            invalid_response_error="invalid claude response",
            debug_label="claude",
            min_request_interval_seconds=min_request_interval_seconds,
            max_attempts=3,
            timeout_seconds=180.0,
        )

    def _environment(self) -> dict[str, str]:
        env = {**os.environ}
        env.pop("CLAUDECODE", None)  # allow nested invocation
        return env

    def _command(self) -> tuple[str, ...]:
        return ("claude", "-p", "--model", self._model)


class GeminiCliAnnotator(_PromptCliAnnotator):
    """Annotator backend that uses the Gemini CLI with Google login instead of the Gemini API."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-pro",
        min_request_interval_seconds: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            model_name=f"gemini-cli-{model}",
            executable="gemini",
            unavailable_error="gemini CLI not found in PATH",
            invalid_response_error="invalid gemini CLI response",
            debug_label="gemini-cli",
            min_request_interval_seconds=min_request_interval_seconds,
            max_attempts=2,
            timeout_seconds=60.0,
        )

    def _command(self) -> tuple[str, ...]:
        return (
            "gemini",
            "--prompt",
            "",
            "--model",
            self._model,
            "--approval-mode",
            "plan",
            "--output-format",
            "json",
        )

    def _parse_stdout(self, stdout_text: str) -> dict[str, Any] | None:
        wrapped_response = _extract_gemini_cli_response_text(stdout_text)
        if wrapped_response is None:
            return None
        return _parse_json_payload(_strip_code_fences(wrapped_response))


def _build_cli_annotation_prompt(*, system_text: str, user_text: str) -> str:
    schema = json.dumps(_annotation_response_schema(), indent=2)
    return (
        f"{system_text}\n\n"
        f"Respond with ONLY valid JSON matching this schema:\n{schema}\n\n"
        f"---\n{user_text}"
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
    return re.sub(r"\n?```\s*$", "", stripped).strip()


def _extract_gemini_cli_response_text(stdout_text: str) -> str | None:
    stripped = stdout_text.strip()
    if not stripped:
        return None
    streamed = _extract_gemini_cli_stream_response_text(stripped)
    if streamed is not None:
        return streamed
    normalized = _extract_json_fragment(stripped) or stripped
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return normalized

    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, str):
            return response.strip()
        for key in ("text", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            response = item.get("response")
            if isinstance(response, str):
                return response.strip()
    return None


def _extract_gemini_cli_stream_response_text(stdout_text: str) -> str | None:
    chunks: list[str] = []
    saw_success = False
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "message" and payload.get("role") == "assistant":
            content = payload.get("content")
            if isinstance(content, str):
                chunks.append(content)
            continue
        if payload.get("type") == "result" and payload.get("status") == "success":
            saw_success = True
    if not chunks:
        return None
    if not saw_success:
        return None
    return "".join(chunks).strip()


def _extract_json_fragment(text: str) -> str | None:
    decoder = json.JSONDecoder()
    best_match: str | None = None
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        candidate = text[index : index + end].strip()
        if candidate and (best_match is None or len(candidate) > len(best_match)):
            best_match = candidate
    return best_match


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Gemini to label actionable transcript lines for classifier training.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["../transcripts"],
        help="Transcript files or directories. Defaults to ../transcripts from backend/.",
    )
    parser.add_argument("--symbol", default="MNQ 03-26", help="Session symbol used for transcript review.")
    parser.add_argument("--market-price", type=float, default=24600.0, help="Reference market price for context rendering.")
    parser.add_argument("--chunk-lines", type=int, default=140, help="Maximum transcript lines per AI request.")
    parser.add_argument("--overlap-lines", type=int, default=24, help="Overlap between adjacent chunks.")
    parser.add_argument("--max-chars", type=int, default=12_000, help="Maximum serialized transcript chars per chunk.")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Concurrent Gemini requests.")
    parser.add_argument(
        "--min-request-interval-ms",
        type=int,
        default=0,
        help="Optional pacing interval between Gemini requests to avoid rate limits.",
    )
    parser.add_argument("--model", default="", help="Optional Gemini model override.")
    parser.add_argument(
        "--candidate-only",
        action="store_true",
        help="Only send trade-like candidate windows to Gemini instead of every transcript chunk.",
    )
    parser.add_argument("--candidate-context-before", type=int, default=6, help="Context lines before each candidate line.")
    parser.add_argument("--candidate-context-after", type=int, default=6, help="Context lines after each candidate line.")
    parser.add_argument("--candidate-merge-gap", type=int, default=18, help="Merge nearby candidate windows separated by this many lines or less.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/interpretation/ai_transcript_annotations.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("data/interpretation/ai_intent_examples.jsonl"),
        help="Output JSONL training-example path.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Optional file limit for sampling runs.")
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


def _base_date_for_file(path: Path) -> datetime:
    date_match = re.search(r"(\d{4})-(\d{2})-(\d{2})", path.name)
    if date_match is not None:
        return datetime(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
            tzinfo=UTC,
        )
    return datetime(2026, 3, 1, tzinfo=UTC)


def _timestamped_lines(path: Path) -> list[TranscriptRow]:
    base_date = _base_date_for_file(path)
    rows: list[TranscriptRow] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TIMESTAMPED_LINE_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        hh, mm, ss, text = match.groups()
        rows.append(
            TranscriptRow(
                line=line_number,
                timecode=f"{hh}:{mm}:{ss}",
                text=text.strip(),
                received_at=base_date.replace(hour=int(hh), minute=int(mm), second=int(ss)),
            )
        )
    return rows


def build_transcript_chunks(
    *,
    file: str,
    rows: list[TranscriptRow],
    chunk_lines: int,
    overlap_lines: int,
    max_chars: int,
) -> list[TranscriptChunk]:
    if not rows:
        return []
    bounded_chunk_lines = max(1, chunk_lines)
    bounded_overlap = max(0, min(overlap_lines, bounded_chunk_lines - 1))
    start = 0
    chunk_index = 0
    chunks: list[TranscriptChunk] = []
    while start < len(rows):
        end = min(len(rows), start + bounded_chunk_lines)
        candidate = rows[start:end]
        serialized = _serialize_chunk_lines(candidate)
        while len(candidate) > 8 and len(serialized) > max_chars:
            candidate = candidate[:-1]
            serialized = _serialize_chunk_lines(candidate)
        chunks.append(
            TranscriptChunk(
                file=file,
                chunk_index=chunk_index,
                rows=tuple(candidate),
            )
        )
        if end >= len(rows):
            break
        next_start = max(start + 1, end - bounded_overlap)
        start = next_start
        chunk_index += 1
    return chunks


def build_candidate_chunks(
    *,
    file: str,
    rows: list[TranscriptRow],
    chunk_lines: int,
    max_chars: int,
    context_before: int,
    context_after: int,
    merge_gap_lines: int,
) -> list[TranscriptChunk]:
    candidate_indexes = [index for index, row in enumerate(rows) if _looks_trade_candidate(_normalize(row.text))]
    if not candidate_indexes:
        return []

    intervals: list[tuple[int, int]] = []
    for index in candidate_indexes:
        start = max(0, index - max(0, context_before))
        end = min(len(rows) - 1, index + max(0, context_after))
        if intervals and start <= intervals[-1][1] + max(0, merge_gap_lines):
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))

    chunks: list[TranscriptChunk] = []
    chunk_index = 0
    bounded_chunk_lines = max(1, chunk_lines)
    for start, end in intervals:
        cursor = start
        while cursor <= end:
            chunk_end = min(end + 1, cursor + bounded_chunk_lines)
            candidate = rows[cursor:chunk_end]
            serialized = _serialize_chunk_lines(candidate)
            while len(candidate) > 8 and len(serialized) > max_chars:
                candidate = candidate[:-1]
                serialized = _serialize_chunk_lines(candidate)
            if not candidate:
                break
            chunks.append(
                TranscriptChunk(
                    file=file,
                    chunk_index=chunk_index,
                    rows=tuple(candidate),
                )
            )
            chunk_index += 1
            cursor += len(candidate)
    return chunks


def _looks_trade_candidate(normalized: str) -> bool:
    if not normalized.strip():
        return False
    if looks_explicit_trade_language(normalized):
        return True
    if looks_candidate_seed(normalized):
        return True
    if detect_setup_signal(normalized) is not None:
        return True
    if detect_present_trade_signal(normalized, position_side=None) is not None:
        return True
    return any(keyword in normalized for keyword in _CANDIDATE_KEYWORDS)


def merge_chunk_annotations(results: list[ChunkAnnotationResult]) -> dict[str, list[AiAnnotation]]:
    merged: dict[str, dict[tuple[int, str, str | None], AiAnnotation]] = defaultdict(dict)
    for result in results:
        for annotation in result.annotations:
            key = (
                annotation.line,
                annotation.label.value,
                annotation.side.value if isinstance(annotation.side, TradeSide) else None,
            )
            current = merged[annotation.file].get(key)
            if current is None or _annotation_sort_key(annotation) > _annotation_sort_key(current):
                merged[annotation.file][key] = annotation
    return {
        file: sorted(file_annotations.values(), key=lambda item: (item.line, item.label.value, item.confidence))
        for file, file_annotations in merged.items()
    }


def build_training_examples(
    *,
    rows: list[TranscriptRow],
    annotations: list[AiAnnotation],
    symbol: str,
    market_price: float,
) -> list[AnnotationTrainingExample]:
    interpreter = RuleBasedTradeInterpreter()
    session = StreamSession(
        config=SessionConfig(symbol=symbol, enable_ai_fallback=False),
        market=MarketSnapshot(symbol=symbol, last_price=market_price),
    )
    selected_annotations = _select_training_annotations(annotations)
    annotation_by_line = {annotation.line: annotation for annotation in selected_annotations}
    examples: list[AnnotationTrainingExample] = []

    for row in rows:
        normalized = _normalize(row.text)
        state_before = interpreter._get_state(session.id, mutate_state=False)
        analysis_text = interpreter._analysis_text(state_before, text=normalized, received_at=row.received_at)
        entry_text = interpreter._entry_text(state_before, text=normalized, received_at=row.received_at)

        annotation = annotation_by_line.get(row.line)
        if annotation is not None:
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
                AnnotationTrainingExample(
                    file=annotation.file,
                    line=annotation.line,
                    timecode=annotation.timecode,
                    timestamp=row.received_at.isoformat(),
                    label=annotation.label.value,
                    source="ai_gemini",
                    current_text=row.text,
                    analysis_text=analysis_text,
                    entry_text=entry_text,
                    prompt=envelope.render(),
                    symbol=symbol,
                    position_side=_side_value(session.position.side if session.position is not None else None, default="FLAT"),
                    last_side=_side_value(state_before.last_side, default="NONE"),
                    ai_confidence=annotation.confidence,
                    ai_reason=annotation.reason,
                    evidence_text=annotation.evidence_text,
                )
            )

        state = interpreter._get_state(session.id, mutate_state=True)
        state.recent_text = normalized
        state.recent_text_at = row.received_at
        if annotation is not None:
            _apply_annotation_state(session=session, state=state, annotation=annotation)

    return examples


def _select_training_annotations(annotations: list[AiAnnotation]) -> list[AiAnnotation]:
    by_line: dict[int, AiAnnotation] = {}
    for annotation in annotations:
        current = by_line.get(annotation.line)
        if current is None or _annotation_sort_key(annotation) > _annotation_sort_key(current):
            by_line[annotation.line] = annotation
    return sorted(by_line.values(), key=lambda item: item.line)


def _annotation_sort_key(annotation: AiAnnotation) -> tuple[float, int, int]:
    return (
        round(annotation.confidence, 6),
        _LABEL_PRIORITY.get(annotation.label, 0),
        -annotation.chunk_index,
    )


def _serialize_chunk_lines(rows: list[TranscriptRow] | tuple[TranscriptRow, ...]) -> str:
    return "\n".join(f"L{row.line} [{row.timecode}] {row.text}" for row in rows)


def _annotation_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line": {"type": "integer"},
                        "timestamp": {"type": "string"},
                        "label": {"type": "string"},
                        "side": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "evidence_text": {"type": "string"},
                        "reason": {"type": ["string", "null"]},
                    },
                    "required": [
                        "line",
                        "timestamp",
                        "label",
                        "side",
                        "confidence",
                        "evidence_text",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["events"],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    allowed_tags = ", ".join(tag.value for tag in sorted(_ACTIONABLE_LABELS, key=lambda item: item.value))
    return (
        "You are labeling transcript lines from a live futures day-trading livestream.\n"
        "Goal: build a HIGH-PRECISION training set. Prefer false negatives over false positives.\n"
        "Only emit an event when the speaker is clearly taking, managing, or closing a live position NOW "
        "for the current session instrument.\n"
        "Use nearby lines in the same chunk for local context, but the chosen line must be the FIRST line "
        "where the action becomes live.\n"
        "If a line is planning, watching, conditional, hypothetical, educational, historical, or ambiguous, "
        "do not upgrade it into an executed action.\n"
        "Allowed labels: "
        f"{allowed_tags}.\n"
        "Decision checklist for every event:\n"
        "1. Is the action happening now, not just being watched or planned?\n"
        "2. Is the chosen line the earliest live line, not a later explanation or follow-up?\n"
        "3. Is the statement about the speaker's own live trade or live stream command for this instrument?\n"
        "4. If any answer is uncertain, omit the event.\n"
        "Definitions:\n"
        "- SETUP_LONG/SETUP_SHORT: directional bias or watchlist only. The speaker is NOT in yet. "
        "Typical cues: looking for, watching, interested in, want, would like, can look for, maybe, "
        "trying to see, pop to sell, pullback to buy.\n"
        "- ENTER_LONG/ENTER_SHORT: entering NOW or explicitly being in NOW. Requires live execution/state "
        "language such as I'm in, I'm long here, I'm short here, small piece on, put something on, "
        "in this long, in this short, long versus this, short versus this.\n"
        "- ADD: adding NOW to an existing position. Requires live add language now such as add here, "
        "adding here, getting more size on here, add on this pop now. Future or conditional add talk is NOT ADD.\n"
        "- TRIM: reducing NOW or taking partial profit NOW. Includes paying myself, pay yourself here, "
        "little piece off, trimming size, take some off. Past-tense recap or generic coaching is NOT TRIM.\n"
        "- EXIT_ALL: flattening NOW. Examples: I'm out, out of that, cut the rest, flat now. "
        "Future stop conditions like if they reclaim I'm out are NOT EXIT_ALL.\n"
        "- MOVE_STOP/MOVE_TO_BREAKEVEN: explicit stop adjustment NOW.\n"
        "Hard negative rules:\n"
        "- Do NOT label setup/watch language as ENTER or ADD. Examples: We're looking to sell. "
        "Looking for a short. Watching 500s to sell into. Pop to sell. I'd like it to fail. "
        "If they get through this we're going 160s.\n"
        "- Do NOT label future or conditional add language as ADD. Examples: I'll add on pops. "
        "Looking for one more push to get more size on. If it pops I'll add. I'll look for the same trade again.\n"
        "- Do NOT label second-person coaching as action unless it is clearly a live management command for an "
        "already-open trade in the immediate context.\n"
        "- Do NOT label past-tense recap as current action. Example: glad I paid myself.\n"
        "- Do NOT label other instruments or off-book examples. Ignore lines about UNH, NVDA, ES, SPY, "
        "or any ticker that is not the current session instrument unless the line is clearly referring back "
        "to the session instrument itself.\n"
        "Positive examples:\n"
        "- We're looking to sell. => SETUP_SHORT\n"
        "- Watching sort of 500s, 550s, and any retest up towards 600 and 630s to sell into. => SETUP_SHORT\n"
        "- I'm looking for two leg or a fail two leg to sell into. => SETUP_SHORT\n"
        "- I'm in short there for a small piece versus VWAP. => ENTER_SHORT\n"
        "- I'm in short here versus VWAP. => ENTER_SHORT\n"
        "- Paying myself some there. => TRIM\n"
        "- No resale. They've gone reclaim. I'm out. => EXIT_ALL\n"
        "- I'm going to move my stop tighter. => MOVE_STOP\n"
        "Negative or omit examples:\n"
        "- Looking for one more push up to get some more size on. => omit\n"
        "- I will try and add on pops where available. => omit\n"
        "- I'll look for the same trade again. => omit\n"
        "- We can look for this short here. => SETUP_SHORT, not ENTER_SHORT\n"
        "- UNH, I trimmed a bit because it didn't do the move I wanted. => omit\n"
        "Return strict JSON with this shape only:\n"
        "{\"events\":[{\"line\":123,\"timestamp\":\"01:23:45\",\"label\":\"ENTER_SHORT\",\"side\":\"SHORT\","
        "\"confidence\":0.87,\"evidence_text\":\"exact transcript text from the chosen line\",\"reason\":\"short justification\"}]}"
    )


def _chunk_prompt(*, chunk: TranscriptChunk, symbol: str, market_price: float) -> str:
    lines_text = _serialize_chunk_lines(chunk.rows)
    return (
        f"file={Path(chunk.file).name}\n"
        f"symbol={symbol}\n"
        f"market_price={market_price:.2f}\n"
        f"chunk_index={chunk.chunk_index}\n"
        f"line_range={chunk.start_line}-{chunk.end_line}\n"
        "Transcript lines:\n"
        f"{lines_text}\n"
        "Choose only lines from this chunk. If nothing actionable is present, return {\"events\":[]}."
    )


def _coerce_chunk_annotations(*, chunk: TranscriptChunk, payload: dict[str, Any] | None) -> list[AiAnnotation]:
    if payload is None:
        return []
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    line_lookup = {row.line: row for row in chunk.rows}
    annotations: list[AiAnnotation] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        line_number = _coerce_int(item.get("line"))
        if line_number is None:
            continue
        row = line_lookup.get(line_number)
        if row is None:
            continue
        label = _coerce_annotation_action_tag(item.get("label"), item.get("side"))
        if label is None or label not in _ACTIONABLE_LABELS:
            continue
        side = _coerce_annotation_trade_side(item.get("side"), label)
        row = _align_annotation_row(chunk=chunk, row=row, label=label, side=side)
        confidence = _coerce_confidence(item.get("confidence"))
        evidence_text = _coerce_optional_str(item.get("evidence_text")) or row.text
        reason = _coerce_optional_str(item.get("reason"))
        annotations.append(
            AiAnnotation(
                file=chunk.file,
                line=row.line,
                timecode=row.timecode,
                label=label,
                side=side,
                confidence=confidence,
                evidence_text=evidence_text,
                reason=reason,
                chunk_index=chunk.chunk_index,
                chunk_start_line=chunk.start_line,
                chunk_end_line=chunk.end_line,
                current_text=row.text,
            )
        )
    return annotations


def _align_annotation_row(
    *,
    chunk: TranscriptChunk,
    row: TranscriptRow,
    label: ActionTag,
    side: TradeSide | None,
) -> TranscriptRow:
    line_to_index = {candidate.line: index for index, candidate in enumerate(chunk.rows)}
    row_index = line_to_index.get(row.line)
    if row_index is None:
        return row

    best_row = row
    best_score = _row_alignment_score(row=row, label=label, side=side)
    start = max(0, row_index - _ALIGNMENT_WINDOW)
    end = min(len(chunk.rows), row_index + _ALIGNMENT_WINDOW + 1)
    for candidate in chunk.rows[start:end]:
        score = _row_alignment_score(row=candidate, label=label, side=side)
        if score <= 0:
            continue
        if score > best_score:
            best_row = candidate
            best_score = score
            continue
        if score == best_score and best_row is not row:
            if candidate.line < best_row.line and abs(candidate.line - row.line) <= abs(best_row.line - row.line):
                best_row = candidate
    return best_row


def _row_alignment_score(*, row: TranscriptRow, label: ActionTag, side: TradeSide | None) -> int:
    normalized = _normalize(row.text)
    explicit_signal = detect_present_trade_signal(normalized, position_side=side)
    setup_signal = detect_setup_signal(normalized)
    side_matches = side is None or side == TradeSide.long and "long" in normalized or side == TradeSide.short and "short" in normalized

    if label == ActionTag.setup_long:
        if setup_signal is not None and setup_signal.tag == ActionTag.setup_long:
            return 8
        return 0
    if label == ActionTag.setup_short:
        if setup_signal is not None and setup_signal.tag == ActionTag.setup_short:
            return 8
        return 0
    if label in {ActionTag.enter_long, ActionTag.enter_short}:
        if explicit_signal is not None and explicit_signal.tag == label:
            return 10
        if _looks_direct_entry_text(normalized=normalized, label=label):
            return 9
        if looks_candidate_seed(normalized) and side_matches:
            return 6
        return 0
    if label == ActionTag.add:
        if explicit_signal is not None and explicit_signal.tag == ActionTag.add:
            return 10
        if looks_candidate_seed(normalized) and "again" in normalized:
            return 6
        return 0
    if label == ActionTag.trim:
        if explicit_signal is not None and explicit_signal.tag == ActionTag.trim:
            return 10
        return 0
    if label == ActionTag.exit_all:
        if explicit_signal is not None and explicit_signal.tag == ActionTag.exit_all:
            return 10
        return 0
    if label == ActionTag.move_stop:
        if explicit_signal is not None and explicit_signal.tag in {ActionTag.move_stop, ActionTag.move_to_breakeven}:
            return 10
        return 0
    if label == ActionTag.move_to_breakeven:
        if explicit_signal is not None and explicit_signal.tag == ActionTag.move_to_breakeven:
            return 10
        if explicit_signal is not None and explicit_signal.tag == ActionTag.move_stop:
            return 7
        return 0
    return 0


def _looks_direct_entry_text(*, normalized: str, label: ActionTag) -> bool:
    if label == ActionTag.enter_long:
        return bool(
            re.search(r"\b(?:i'?m|im|i m)\s+in\s+(?:a\s+|this\s+)?long\b", normalized)
            or re.search(r"\b(?:i'?m|im|i m)\s+long(?:\s+again)?\b", normalized)
            or re.search(r"\bin\s+this\s+long\b", normalized)
        )
    if label == ActionTag.enter_short:
        return bool(
            re.search(r"\b(?:i'?m|im|i m)\s+in\s+(?:a\s+|this\s+)?short\b", normalized)
            or re.search(r"\b(?:i'?m|im|i m)\s+short(?:\s+again)?\b", normalized)
            or re.search(r"\bin\s+this\s+short\b", normalized)
        )
    return False


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_annotation_action_tag(tag_value: Any, side_value: Any) -> ActionTag | None:
    token = _normalize_label_token(tag_value)
    if token in {"SETUP_LONG", "LONG_SETUP"}:
        return ActionTag.setup_long
    if token in {"SETUP_SHORT", "SHORT_SETUP"}:
        return ActionTag.setup_short
    return _coerce_action_tag(tag_value, side_value)


def _coerce_annotation_trade_side(side_value: Any, tag: ActionTag) -> TradeSide | None:
    side = _coerce_trade_side(side_value, tag)
    if side is not None:
        return side
    if tag == ActionTag.setup_long:
        return TradeSide.long
    if tag == ActionTag.setup_short:
        return TradeSide.short
    return None


def _normalize_label_token(value: Any) -> str:
    if value is None:
        return ""
    token = str(value).strip().upper()
    token = token.replace("-", "_").replace(" ", "_")
    return re.sub(r"[^A-Z0-9_]", "", token)


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _retry_delay_seconds(response: httpx.Response) -> float:
    try:
        data = response.json()
        message = str(data.get("error", {}).get("message", ""))
    except Exception:
        message = response.text
    match = re.search(r"retry in ([0-9.]+)s", message, flags=re.IGNORECASE)
    if match is not None:
        try:
            return max(1.0, float(match.group(1)) + 0.5)
        except ValueError:
            pass
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return 5.0


def _retry_reason(response: httpx.Response) -> str:
    try:
        data = response.json()
        message = str(data.get("error", {}).get("message", ""))
    except Exception:
        message = response.text
    return re.sub(r"\s+", " ", message).strip()[:240]


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _side_value(side: TradeSide | str | None, *, default: str) -> str:
    if isinstance(side, TradeSide):
        return side.value
    return str(side or default)


def _apply_annotation_state(*, session: StreamSession, state: Any, annotation: AiAnnotation) -> None:
    if annotation.side is not None:
        state.last_side = annotation.side
    if annotation.label == ActionTag.setup_long:
        state.last_side = TradeSide.long
        return
    if annotation.label == ActionTag.setup_short:
        state.last_side = TradeSide.short
        return
    if annotation.label in _ENTRY_LABELS:
        side = annotation.side
        if side is None and annotation.label == ActionTag.enter_long:
            side = TradeSide.long
        elif side is None and annotation.label == ActionTag.enter_short:
            side = TradeSide.short
        if side is None:
            return
        state.last_side = side
        session.position = PositionState(
            side=side,
            quantity=1,
            average_price=session.market.last_price or 0.0,
            stop_price=None,
            target_price=None,
        )
        return
    if annotation.label == ActionTag.exit_all:
        session.position = None
        return
    if annotation.label == ActionTag.move_to_breakeven and session.position is not None:
        session.position.stop_price = session.position.average_price


async def _annotate_chunks(
    *,
    annotator: TranscriptChunkAnnotator,
    chunks: list[TranscriptChunk],
    symbol: str,
    market_price: float,
    max_concurrency: int,
) -> list[ChunkAnnotationResult]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run(chunk: TranscriptChunk) -> ChunkAnnotationResult:
        async with semaphore:
            return await annotator.annotate_chunk(chunk=chunk, symbol=symbol, market_price=market_price)

    return list(await asyncio.gather(*(run(chunk) for chunk in chunks)))


def _build_file_reports(
    *,
    files: list[Path],
    rows_by_file: dict[str, list[TranscriptRow]],
    chunks_by_file: dict[str, list[TranscriptChunk]],
    merged_annotations: dict[str, list[AiAnnotation]],
    results: list[ChunkAnnotationResult],
) -> list[FileAnnotationReport]:
    errors_by_file: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.error:
            errors_by_file[result.file].append(f"chunk {result.chunk_index}: {result.error}")
    reports: list[FileAnnotationReport] = []
    for path in files:
        file_key = str(path)
        reports.append(
            FileAnnotationReport(
                file=file_key,
                total_rows=len(rows_by_file.get(file_key, [])),
                chunk_count=len(chunks_by_file.get(file_key, [])),
                annotations=tuple(merged_annotations.get(file_key, [])),
                errors=tuple(errors_by_file.get(file_key, [])),
            )
        )
    return reports


async def _main() -> int:
    args = _parse_args()
    settings = get_settings()
    annotator = GeminiTranscriptAnnotator(
        settings,
        model_name=args.model or None,
        min_request_interval_seconds=max(0.0, args.min_request_interval_ms / 1000.0),
    )
    files = _iter_transcript_files(args.inputs)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print("No transcript files found.")
        return 1
    if not annotator.is_available():
        print("Gemini transcript annotation requires GEMINI_API_KEY.")
        return 1

    rows_by_file = {str(path): _timestamped_lines(path) for path in files}
    if args.candidate_only:
        chunks_by_file = {
            str(path): build_candidate_chunks(
                file=str(path),
                rows=rows_by_file[str(path)],
                chunk_lines=args.chunk_lines,
                max_chars=args.max_chars,
                context_before=args.candidate_context_before,
                context_after=args.candidate_context_after,
                merge_gap_lines=args.candidate_merge_gap,
            )
            for path in files
        }
    else:
        chunks_by_file = {
            str(path): build_transcript_chunks(
                file=str(path),
                rows=rows_by_file[str(path)],
                chunk_lines=args.chunk_lines,
                overlap_lines=args.overlap_lines,
                max_chars=args.max_chars,
            )
            for path in files
        }
    all_chunks = [chunk for chunks in chunks_by_file.values() for chunk in chunks]
    try:
        results = await _annotate_chunks(
            annotator=annotator,
            chunks=all_chunks,
            symbol=args.symbol,
            market_price=args.market_price,
            max_concurrency=args.max_concurrency,
        )
    finally:
        await annotator.close()

    merged_annotations = merge_chunk_annotations(results)
    reports = _build_file_reports(
        files=files,
        rows_by_file=rows_by_file,
        chunks_by_file=chunks_by_file,
        merged_annotations=merged_annotations,
        results=results,
    )

    examples: list[AnnotationTrainingExample] = []
    for path in files:
        file_key = str(path)
        examples.extend(
            build_training_examples(
                rows=rows_by_file[file_key],
                annotations=merged_annotations.get(file_key, []),
                symbol=args.symbol,
                market_price=args.market_price,
            )
        )

    counts = Counter(example.label for example in examples)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model or settings.gemini_model,
        "symbol": args.symbol,
        "market_price": args.market_price,
        "files": [
            {
                "file": report.file,
                "total_rows": report.total_rows,
                "chunk_count": report.chunk_count,
                "errors": list(report.errors),
                "annotations": [asdict(annotation) for annotation in report.annotations],
            }
            for report in reports
        ],
        "counts": dict(sorted(counts.items())),
        "example_count": len(examples),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl_out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=True) + "\n")

    print(f"Wrote annotation report to {args.json_out}")
    print(f"Wrote {len(examples)} AI-labeled examples to {args.jsonl_out}")
    for label, count in sorted(counts.items()):
        print(f"{label}\t{count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
