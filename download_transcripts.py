"""Download YouTube auto-generated transcripts for all missing livestream videos."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TRANSCRIPTS_DIR = Path("transcripts")
MISSING_IDS_FILE = Path("/tmp/missing_ids.txt")
ALL_STREAMS_FILE = Path("/tmp/all_streams.txt")


def _title_to_filename_part(title: str) -> str:
    """Convert video title to a filesystem-safe string."""
    # Remove emoji and special chars
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"\s+", "-", title.strip())
    # Remove leading/trailing hyphens
    title = title.strip("-")
    return title


def _get_upload_date(video_id: str) -> str | None:
    """Get upload date via yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "%(upload_date)s", f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True, timeout=30,
        )
        date = result.stdout.strip()
        if date and len(date) == 8:
            return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    except Exception:
        pass
    return None


def _download_transcript(video_id: str, title: str) -> tuple[bool, str]:
    """Download auto-generated transcript for a video. Returns (success, message)."""
    # Try to get upload date for filename
    date = _get_upload_date(video_id)
    if not date:
        date = "unknown-date"

    title_part = _title_to_filename_part(title)
    filename = f"{date}__{title_part}__{video_id}.txt"
    output_path = TRANSCRIPTS_DIR / filename

    if output_path.exists():
        return True, f"Already exists: {filename}"

    # Download subtitles using yt-dlp
    tmp_prefix = f"/tmp/yt_sub_{video_id}"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "-o", tmp_prefix,
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=60,
        )

        # Find the downloaded subtitle file
        vtt_path = None
        for ext in [".en.vtt", ".en-orig.vtt"]:
            candidate = Path(f"{tmp_prefix}{ext}")
            if candidate.exists():
                vtt_path = candidate
                break

        if vtt_path is None:
            return False, f"No English subtitles found for {video_id}"

        # Convert VTT to plain text with timecodes
        lines = _vtt_to_text(vtt_path)
        if not lines:
            vtt_path.unlink(missing_ok=True)
            return False, f"Empty transcript for {video_id}"

        output_path.write_text("\n".join(lines), encoding="utf-8")
        vtt_path.unlink(missing_ok=True)
        return True, f"Downloaded: {filename} ({len(lines)} lines)"

    except subprocess.TimeoutExpired:
        return False, f"Timeout for {video_id}"
    except Exception as e:
        return False, f"Error for {video_id}: {e}"


def _vtt_to_text(vtt_path: Path) -> list[str]:
    """Convert VTT subtitle file to plain text lines with timecodes."""
    raw = vtt_path.read_text(encoding="utf-8")
    lines: list[str] = []
    seen_text: set[str] = set()

    # Parse VTT format
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2})\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")

    current_time = None
    for raw_line in raw.split("\n"):
        raw_line = raw_line.strip()

        # Skip VTT header
        if raw_line.startswith("WEBVTT") or raw_line.startswith("Kind:") or raw_line.startswith("Language:"):
            continue

        # Skip blank lines and numeric cue IDs
        if not raw_line or raw_line.isdigit():
            continue

        # Check for timestamp
        time_match = time_pattern.match(raw_line)
        if time_match:
            current_time = time_match.group(1)
            continue

        # Clean HTML tags from text
        text = re.sub(r"<[^>]+>", "", raw_line).strip()
        if not text:
            continue

        # Deduplicate (VTT often repeats lines)
        if text in seen_text:
            continue
        seen_text.add(text)

        if current_time:
            lines.append(f"[{current_time}] {text}")
        else:
            lines.append(text)

    return lines


def main() -> int:
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Load stream titles
    titles: dict[str, str] = {}
    if ALL_STREAMS_FILE.exists():
        for line in ALL_STREAMS_FILE.read_text().splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2:
                titles[parts[0]] = parts[1]

    # Load missing IDs
    if not MISSING_IDS_FILE.exists():
        print("Missing IDs file not found. Run the discovery step first.")
        return 1

    missing_ids = [line.strip() for line in MISSING_IDS_FILE.read_text().splitlines() if line.strip()]
    print(f"Found {len(missing_ids)} missing transcripts to download.\n")

    success_count = 0
    fail_count = 0
    failed_ids: list[str] = []

    for i, video_id in enumerate(missing_ids, 1):
        title = titles.get(video_id, "Unknown")
        print(f"[{i}/{len(missing_ids)}] {video_id} - {title[:60]}...", flush=True)

        ok, msg = _download_transcript(video_id, title)
        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_ids.append(video_id)
        print(f"  {msg}", flush=True)

    print(f"\n{'='*60}")
    print(f"Done. Downloaded: {success_count}, Failed: {fail_count}")
    if failed_ids:
        print(f"Failed IDs: {', '.join(failed_ids[:20])}")
        Path("/tmp/failed_transcript_ids.txt").write_text("\n".join(failed_ids))
        print(f"Full list saved to /tmp/failed_transcript_ids.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
