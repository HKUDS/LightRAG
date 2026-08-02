"""Deterministic scene windows and contact-sheet generation."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lightrag.parser.video.media import detect_scene_cut_times


@dataclass(frozen=True)
class SceneWindow:
    scene_id: str
    index: int
    start_seconds: float
    end_seconds: float
    frame_timestamps: tuple[float, ...]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def build_scene_windows(
    duration_seconds: float,
    cut_times: list[float],
    *,
    max_scene_seconds: float = 45.0,
    min_scene_seconds: float = 2.0,
    max_scenes: int = 128,
    frames_per_scene: int = 4,
) -> list[SceneWindow]:
    """Build bounded windows, splitting long scenes and capping work."""

    duration = max(0.1, float(duration_seconds))
    max_len = max(2.0, float(max_scene_seconds))
    min_len = max(0.25, min(float(min_scene_seconds), max_len))
    boundaries = [0.0]
    boundaries.extend(
        t for t in cut_times if min_len <= t <= duration - min_len
    )
    boundaries.append(duration)
    boundaries = sorted(set(boundaries))

    spans: list[tuple[float, float]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 0.25:
            continue
        pieces = max(1, math.ceil((right - left) / max_len))
        for piece in range(pieces):
            start = left + (right - left) * piece / pieces
            end = left + (right - left) * (piece + 1) / pieces
            if end - start >= 0.25:
                spans.append((start, end))

    if len(spans) > max_scenes:
        stride = len(spans) / max_scenes
        spans = [spans[min(len(spans) - 1, int(i * stride))] for i in range(max_scenes)]
        spans = sorted(set(spans))

    frame_count = max(1, min(8, int(frames_per_scene)))
    windows: list[SceneWindow] = []
    for index, (start, end) in enumerate(spans, start=1):
        if frame_count == 1:
            timestamps = ((start + end) / 2,)
        else:
            timestamps = tuple(
                start + (end - start) * position / (frame_count - 1)
                for position in range(frame_count)
            )
        windows.append(
            SceneWindow(
                scene_id=f"scene-{index:04d}",
                index=index,
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                frame_timestamps=tuple(round(value, 3) for value in timestamps),
            )
        )
    return windows


def detect_windows(
    source: Path,
    duration_seconds: float,
    *,
    threshold: float,
    max_scene_seconds: float,
    min_scene_seconds: float,
    max_scenes: int,
    frames_per_scene: int,
) -> list[SceneWindow]:
    cuts = detect_scene_cut_times(source, threshold=threshold)
    return build_scene_windows(
        duration_seconds,
        cuts,
        max_scene_seconds=max_scene_seconds,
        min_scene_seconds=min_scene_seconds,
        max_scenes=max_scenes,
        frames_per_scene=frames_per_scene,
    )


def _extract_frame(source: Path, timestamp: float, destination: Path) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=768:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "4",
            "-y",
            str(destination),
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail[-800:] or "frame extraction failed")


def create_contact_sheet(
    source: Path,
    window: SceneWindow,
    destination: Path,
    *,
    cell_width: int = 384,
    jpeg_quality: int = 82,
) -> list[int]:
    """Extract frames and compose one compact JPEG evidence asset."""

    from PIL import Image, ImageDraw, ImageFont

    temp_dir = Path(tempfile.mkdtemp(prefix=".video-frames-", dir=destination.parent))
    try:
        frames: list[Image.Image] = []
        kept_timestamps: list[int] = []
        for position, timestamp in enumerate(window.frame_timestamps):
            frame_path = temp_dir / f"frame-{position:02d}.jpg"
            try:
                _extract_frame(source, timestamp, frame_path)
                with Image.open(frame_path) as image:
                    frame = image.convert("RGB")
                    ratio = min(1.0, cell_width / max(1, frame.width))
                    frame = frame.resize(
                        (max(1, int(frame.width * ratio)), max(1, int(frame.height * ratio))),
                        Image.Resampling.LANCZOS,
                    )
                    frames.append(frame.copy())
                    kept_timestamps.append(round(timestamp * 1000))
            except (OSError, RuntimeError):
                continue
        if not frames:
            raise RuntimeError(f"no frames could be extracted for {window.scene_id}")

        columns = min(2, len(frames))
        rows = math.ceil(len(frames) / columns)
        label_height = 28
        cell_height = max(frame.height for frame in frames) + label_height
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#111827")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        for index, frame in enumerate(frames):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            sheet.paste(frame, (x, y))
            draw.rectangle((x, y + frame.height, x + cell_width, y + cell_height), fill="#111827")
            draw.text((x + 8, y + frame.height + 5), _format_time(kept_timestamps[index] / 1000), fill="white", font=font)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination, format="JPEG", quality=max(50, min(95, jpeg_quality)), optimize=True)
        return kept_timestamps
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _format_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
