"""Safe, CPU-only media primitives used by the video parser.

The parser deliberately invokes FFmpeg/FFprobe with argument arrays and never
through a shell.  Video bytes remain local to the LightRAG worker; only short
audio segments are sent to the configured ASR endpoint.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaToolError(RuntimeError):
    """Raised when FFmpeg/FFprobe cannot process a source."""


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool
    video_codec: str | None = None
    audio_codec: str | None = None


@dataclass(frozen=True)
class AudioSegment:
    start_seconds: float
    end_seconds: float
    data: bytes


def _run(args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise MediaToolError(
            "FFmpeg is not installed. Install the ffmpeg system package."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"media command timed out after {timeout:g}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown media error").strip()
        raise MediaToolError(detail[-1200:]) from exc


def probe_media(source: Path, *, timeout: float = 60.0) -> MediaInfo:
    """Read duration and stream capabilities using FFprobe."""

    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of",
            "json",
            str(source),
        ],
        timeout=timeout,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("ffprobe returned invalid JSON") from exc

    fmt = payload.get("format") or {}
    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError) as exc:
        raise MediaToolError("ffprobe did not return a valid duration") from exc
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps: float | None = None
    rate = str(video.get("r_frame_rate") or "")
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        try:
            if float(denominator):
                fps = float(numerator) / float(denominator)
        except ValueError:
            pass

    width = _optional_int(video.get("width"))
    height = _optional_int(video.get("height"))
    if duration <= 0 or not video:
        raise MediaToolError("source has no readable video stream or duration")
    return MediaInfo(
        duration_seconds=duration,
        width=width,
        height=height,
        fps=fps,
        has_audio=bool(audio),
        video_codec=str(video.get("codec_name")) if video.get("codec_name") else None,
        audio_codec=str(audio.get("codec_name")) if audio.get("codec_name") else None,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def extract_audio_segment(
    source: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    timeout: float = 180.0,
) -> AudioSegment:
    """Decode one bounded mono 16 kHz WAV segment to memory."""

    start = max(0.0, float(start_seconds))
    end = max(start, float(end_seconds))
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{end - start:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaToolError(detail[-1200:] or "ffmpeg audio extraction failed")
    if not result.stdout:
        raise MediaToolError("ffmpeg returned an empty audio segment")
    return AudioSegment(start, end, result.stdout)


def detect_scene_cut_times(
    source: Path, *, threshold: float = 0.30, timeout: float = 600.0
) -> list[float]:
    """Return FFmpeg scene-change timestamps.

    Failure is intentionally non-fatal: uniform windows remain a useful and
    deterministic fallback for unusual codecs or videos without cuts.
    """

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "info",
                "-i",
                str(source),
                "-vf",
                f"select='gt(scene,{max(0.05, min(0.95, threshold)):.3f})',showinfo",
                "-an",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    output = result.stderr.decode("utf-8", errors="replace")
    values: list[float] = []
    for match in re.finditer(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)", output):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return sorted(set(values))
