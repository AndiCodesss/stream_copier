from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TIMECODE_RE = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})(?:[.,](\d{3}))?")
_TIMING_LINE_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}[.,]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[.,]\d{3}")
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_JSON3_NEWLINE_RE = re.compile(r"\s*\n\s*")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(slots=True, frozen=True)
class CaptionLine:
    timestamp_seconds: int
    text: str


@dataclass(slots=True)
class DownloadResult:
    video_id: str
    title: str
    output_path: str | None
    status: str
    caption_source: str | None = None
    caption_language: str | None = None
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube channel captions and render them as timestamped plain text transcripts."
    )
    parser.add_argument("channel_url", help="YouTube channel/tab URL, for example https://www.youtube.com/@FlowZoneTrader/streams")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "transcripts" / "youtube_captions",
        help="Directory for transcript files and manifest.json",
    )
    parser.add_argument("--language", default="en", help="Preferred caption language (default: en)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos from the playlist")
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Attempt live/upcoming videos too. By default they are skipped because captions are often incomplete.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite transcript files even if they already exist",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Optional browser name for yt-dlp, for example firefox or chrome",
    )
    args = parser.parse_args(argv)

    yt_dlp_command = _resolve_yt_dlp_command()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    playlist_entries = _fetch_playlist_entries(
        yt_dlp_command=yt_dlp_command,
        channel_url=args.channel_url,
        limit=args.limit,
        cookies_from_browser=args.cookies_from_browser,
    )

    results: list[DownloadResult] = []
    for entry in playlist_entries:
        result = _process_video(
            yt_dlp_command=yt_dlp_command,
            video_url=entry["video_url"],
            out_dir=args.out_dir,
            language=args.language,
            overwrite=args.overwrite,
            include_live=args.include_live,
            cookies_from_browser=args.cookies_from_browser,
        )
        results.append(result)
        if result.status == "downloaded":
            print(f"downloaded  {result.video_id}  {result.output_path}")
        elif result.status == "skipped_existing":
            print(f"existing    {result.video_id}  {result.output_path}")
        elif result.status == "skipped_live":
            print(f"skipped     {result.video_id}  live/upcoming")
        elif result.status == "skipped_no_captions":
            print(f"skipped     {result.video_id}  no YouTube captions")
        else:
            print(f"error       {result.video_id}  {result.error or 'unknown error'}", file=sys.stderr)

    manifest_path = args.out_dir / "manifest.json"
    manifest = {
        "channel_url": args.channel_url,
        "language": args.language,
        "results": [asdict(result) for result in results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest    {manifest_path}")
    return 0


def _resolve_yt_dlp_command() -> list[str]:
    candidates = [
        shutil.which("yt-dlp"),
        shutil.which("yt-dlp.exe"),
        str(Path(sys.executable).with_name("yt-dlp")),
        str(Path(sys.executable).with_name("yt-dlp.exe")),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError(
        "yt-dlp is not installed. Install it into the project venv with './.venv/bin/python -m pip install yt-dlp'."
    )


def _fetch_playlist_entries(
    *,
    yt_dlp_command: list[str],
    channel_url: str,
    limit: int | None,
    cookies_from_browser: str | None,
) -> list[dict[str, str]]:
    command = [*yt_dlp_command, "--flat-playlist", "--dump-single-json"]
    if limit is not None:
        command.extend(["--playlist-end", str(limit)])
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.append(channel_url)

    payload = json.loads(_run_command(command))
    entries: list[dict[str, str]] = []
    for raw_entry in payload.get("entries") or []:
        video_id = (raw_entry or {}).get("id")
        if not video_id:
            continue
        video_url = (raw_entry or {}).get("url") or f"https://www.youtube.com/watch?v={video_id}"
        if not str(video_url).startswith("http"):
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        entries.append({"video_id": video_id, "video_url": str(video_url)})
    return entries


def _process_video(
    *,
    yt_dlp_command: list[str],
    video_url: str,
    out_dir: Path,
    language: str,
    overwrite: bool,
    include_live: bool,
    cookies_from_browser: str | None,
) -> DownloadResult:
    try:
        info_command = [*yt_dlp_command, "--skip-download", "--dump-single-json"]
        if cookies_from_browser:
            info_command.extend(["--cookies-from-browser", cookies_from_browser])
        info_command.append(video_url)
        info = json.loads(_run_command(info_command))

        video_id = str(info.get("id") or "unknown")
        title = str(info.get("title") or video_id)
        if not include_live and _is_live_or_upcoming(info):
            return DownloadResult(video_id=video_id, title=title, output_path=None, status="skipped_live")

        output_path = _build_output_path(out_dir=out_dir, info=info)
        if output_path.exists() and not overwrite:
            return DownloadResult(
                video_id=video_id,
                title=title,
                output_path=str(output_path.resolve()),
                status="skipped_existing",
            )

        caption_source, caption_language = _pick_caption_track(info, language)
        if caption_source is None or caption_language is None:
            return DownloadResult(
                video_id=video_id,
                title=title,
                output_path=None,
                status="skipped_no_captions",
            )

        with tempfile.TemporaryDirectory(prefix=f"{video_id}_", dir=out_dir) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _download_captions(
                yt_dlp_command=yt_dlp_command,
                video_url=video_url,
                temp_dir=temp_dir,
                source=caption_source,
                language=caption_language,
                cookies_from_browser=cookies_from_browser,
            )
            caption_file = _find_caption_file(temp_dir=temp_dir, video_id=video_id)
            if caption_file is None:
                raise RuntimeError("yt-dlp did not produce a caption file")
            transcript_text = _caption_file_to_plaintext(caption_file)
            if not transcript_text.strip():
                raise RuntimeError("caption file contained no usable transcript lines")

        output_path.write_text(transcript_text, encoding="utf-8")
        return DownloadResult(
            video_id=video_id,
            title=title,
            output_path=str(output_path.resolve()),
            status="downloaded",
            caption_source=caption_source,
            caption_language=caption_language,
        )
    except Exception as error:
        message = str(error)
        if _looks_like_upcoming_live_error(message):
            video_id = _extract_video_id(video_url)
            return DownloadResult(
                video_id=video_id,
                title=video_id,
                output_path=None,
                status="skipped_live",
            )
        return DownloadResult(
            video_id=_extract_video_id(video_url),
            title=_extract_video_id(video_url),
            output_path=None,
            status="error",
            error=message,
        )


def _is_live_or_upcoming(info: dict[str, Any]) -> bool:
    live_status = str(info.get("live_status") or "")
    return live_status in {"is_live", "is_upcoming", "post_live"} or bool(info.get("is_live"))


def _pick_caption_track(info: dict[str, Any], preferred_language: str) -> tuple[str | None, str | None]:
    manual = _pick_language_key(info.get("subtitles") or {}, preferred_language)
    if manual is not None:
        return "manual", manual
    automatic = _pick_language_key(info.get("automatic_captions") or {}, preferred_language)
    if automatic is not None:
        return "automatic", automatic
    return None, None


def _pick_language_key(tracks: dict[str, Any], preferred_language: str) -> str | None:
    if not tracks:
        return None

    preferred = preferred_language.lower()
    candidates = [key for key in tracks if "live_chat" not in key.lower()]
    if not candidates:
        return None

    def sort_key(value: str) -> tuple[int, int, int, str]:
        lowered = value.lower()
        exact = 0 if lowered == preferred else 1
        prefix = 0 if lowered.startswith(f"{preferred}-") or lowered.startswith(f"{preferred}_") else 1
        orig = 1 if "orig" in lowered else 0
        return (exact, prefix, orig, lowered)

    matches = [key for key in candidates if key.lower() == preferred or key.lower().startswith(f"{preferred}-") or key.lower().startswith(f"{preferred}_")]
    pool = matches or candidates
    return sorted(pool, key=sort_key)[0]


def _download_captions(
    *,
    yt_dlp_command: list[str],
    video_url: str,
    temp_dir: Path,
    source: str,
    language: str,
    cookies_from_browser: str | None,
) -> None:
    command = [
        *yt_dlp_command,
        "--skip-download",
        "--sub-format",
        "json3/vtt/best",
        "--sub-langs",
        language,
        "--output",
        str(temp_dir / "%(id)s"),
    ]
    if source == "manual":
        command.append("--write-subs")
    else:
        command.append("--write-auto-subs")
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.append(video_url)
    _run_command(command)


def _find_caption_file(*, temp_dir: Path, video_id: str) -> Path | None:
    candidates = sorted(temp_dir.glob(f"{video_id}.*"))
    for suffix in (".json3", ".vtt"):
        for candidate in candidates:
            if candidate.suffix == suffix:
                return candidate
    return candidates[0] if candidates else None


def _caption_file_to_plaintext(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix == ".json3":
        lines = _parse_json3(raw_text)
    else:
        lines = _parse_vtt(raw_text)
    return _render_transcript(lines)


def _parse_json3(raw_text: str) -> list[CaptionLine]:
    payload = json.loads(raw_text)
    lines: list[CaptionLine] = []
    for event in payload.get("events") or []:
        timestamp_ms = event.get("tStartMs")
        segs = event.get("segs") or []
        if timestamp_ms is None or not segs:
            continue
        text = "".join(str(segment.get("utf8") or "") for segment in segs)
        cleaned = _normalize_caption_text(text)
        if not cleaned:
            continue
        _append_caption_line(lines, CaptionLine(timestamp_seconds=max(0, int(int(timestamp_ms) / 1000)), text=cleaned))
    return lines


def _parse_vtt(raw_text: str) -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    current_timestamp: int | None = None
    current_text: list[str] = []

    def flush() -> None:
        nonlocal current_timestamp, current_text
        if current_timestamp is None:
            current_text = []
            return
        combined = _normalize_caption_text(" ".join(current_text))
        if combined:
            _append_caption_line(lines, CaptionLine(timestamp_seconds=current_timestamp, text=combined))
        current_timestamp = None
        current_text = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip("\ufeff")
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped == "WEBVTT" or stripped.startswith("Kind:") or stripped.startswith("Language:"):
            continue
        if stripped.isdigit():
            continue
        if "-->" in stripped and _TIMING_LINE_RE.match(stripped):
            flush()
            current_timestamp = _parse_timecode(stripped.split("-->", 1)[0].strip())
            continue
        if stripped.startswith("NOTE"):
            continue
        current_text.append(stripped)

    flush()
    return lines


def _append_caption_line(lines: list[CaptionLine], item: CaptionLine) -> None:
    if not lines:
        lines.append(item)
        return
    previous = lines[-1]
    if item.text == previous.text:
        return
    if item.timestamp_seconds <= previous.timestamp_seconds + 1 and item.text.startswith(previous.text):
        lines[-1] = CaptionLine(timestamp_seconds=previous.timestamp_seconds, text=item.text)
        return
    lines.append(item)


def _normalize_caption_text(value: str) -> str:
    cleaned = html.unescape(value)
    cleaned = cleaned.replace("\u200b", " ")
    cleaned = _JSON3_NEWLINE_RE.sub(" ", cleaned)
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _render_transcript(lines: list[CaptionLine]) -> str:
    return "\n".join(f"[{_format_timestamp(item.timestamp_seconds)}] {item.text}" for item in lines) + ("\n" if lines else "")


def _parse_timecode(value: str) -> int:
    match = _TIMECODE_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid timecode: {value}")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return (hours * 3600) + (minutes * 60) + seconds


def _format_timestamp(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _build_output_path(*, out_dir: Path, info: dict[str, Any]) -> Path:
    video_id = str(info.get("id") or "unknown")
    title = str(info.get("title") or video_id)
    upload_date = str(info.get("upload_date") or "")
    if len(upload_date) == 8 and upload_date.isdigit():
        date_prefix = f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    else:
        date_prefix = "unknown-date"
    safe_title = _SAFE_FILENAME_RE.sub("-", title).strip("-.")
    safe_title = re.sub(r"-{2,}", "-", safe_title) or video_id
    safe_title = safe_title[:80]
    return out_dir / f"{date_prefix}__{safe_title}__{video_id}.txt"


def _extract_video_id(video_url: str) -> str:
    if "v=" in video_url:
        return video_url.split("v=", 1)[1].split("&", 1)[0]
    return video_url.rstrip("/").rsplit("/", 1)[-1]


def _looks_like_upcoming_live_error(message: str) -> bool:
    lowered = message.lower()
    return "this live event will begin" in lowered or "this live event has not started" in lowered


def _run_command(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "command failed"
        raise RuntimeError(stderr)
    return completed.stdout


if __name__ == "__main__":
    raise SystemExit(main())
