"""Production video parser: FFmpeg scenes + Qwen ASR + multimodal sidecar."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lightrag.constants import PARSER_ENGINE_VIDEO
from lightrag.parser.native_base import NativeParserBase
from lightrag.parser.video.asr import VideoAsrClient
from lightrag.parser.video.ir_builder import build_video_ir
from lightrag.parser.video.manifest import write_video_manifest
from lightrag.parser.video.media import MediaInfo, extract_audio_segment, probe_media
from lightrag.parser.video.scene import create_contact_sheet, detect_windows
from lightrag.utils import logger


class VideoParser(NativeParserBase):
    """Parse video locally while delegating speech recognition to Qwen ASR."""

    engine_name = PARSER_ENGINE_VIDEO
    sidecar_path_style = "with_prefix"
    empty_content_label = "Video"

    def validate_source(self, source: Path, file_path: str) -> None:
        suffix = source.suffix.lower().lstrip(".")
        if not source.is_file() or suffix not in {"mp4", "mov", "mkv", "webm", "avi", "m4v"}:
            raise ValueError(f"Video parser does not support pending file: {file_path}")

    def extract(
        self, source: Path, *, parsed_dir: Path, asset_dir: Path, base_name: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        config = _VideoConfig.from_env()
        media = probe_media(source, timeout=config.ffprobe_timeout)
        if media.duration_seconds > config.max_duration_seconds:
            raise ValueError(
                f"video duration {media.duration_seconds:.1f}s exceeds "
                f"VIDEO_MAX_DURATION_SECONDS={config.max_duration_seconds:.1f}"
            )

        warnings: list[str] = []
        windows = detect_windows(
            source,
            media.duration_seconds,
            threshold=config.scene_threshold,
            max_scene_seconds=config.max_scene_seconds,
            min_scene_seconds=config.min_scene_seconds,
            max_scenes=config.max_scenes,
            frames_per_scene=config.frames_per_scene,
        )
        if not windows:
            raise ValueError("video scene detection produced no usable windows")

        asr_client = None
        if media.has_audio:
            if not config.asr_host:
                warnings.append("audio track found but VIDEO_ASR_BINDING_HOST is not configured")
            else:
                asr_client = VideoAsrClient(
                    config.asr_host,
                    config.asr_api_key,
                    config.asr_model,
                    timeout_seconds=config.asr_timeout,
                )

        scene_records: list[dict[str, Any]] = []
        transcript_segments: list[dict[str, Any]] = []
        output_blocks: list[dict[str, Any]] = [
            {
                "heading": "Video overview",
                "level": 0,
                "content": _overview_text(source.name, media, len(windows)),
            }
        ]
        full_transcript: list[str] = []

        for window in windows:
            asset_name = f"{window.scene_id}.jpg"
            asset_path = asset_dir / asset_name
            frame_timestamps = create_contact_sheet(
                source,
                window,
                asset_path,
                cell_width=config.contact_sheet_cell_width,
                jpeg_quality=config.contact_sheet_quality,
            )
            transcript = ""
            scene_segments: list[dict[str, Any]] = []
            if asr_client:
                for start, end in _bounded_segments(
                    window.start_seconds, window.end_seconds, config.asr_segment_seconds
                ):
                    audio = extract_audio_segment(
                        source, start, end, timeout=config.ffmpeg_timeout
                    )
                    try:
                        text = asr_client.transcribe(audio.data, language=config.language)
                    except Exception as exc:
                        warnings.append(f"{window.scene_id} ASR failed: {exc}")
                        logger.warning("[video] %s", warnings[-1])
                        continue
                    if text:
                        scene_segments.append(
                            {"start_ms": round(start * 1000), "end_ms": round(end * 1000), "text": text}
                        )
                        transcript_segments.append(scene_segments[-1] | {"scene_id": window.scene_id})
                transcript = " ".join(str(item["text"]) for item in scene_segments).strip()
            elif media.has_audio:
                transcript = "[Audio track present; ASR is not configured.]"
            else:
                transcript = "[No audio track.]"
            if transcript:
                full_transcript.append(transcript)

            manifest_ref = f"{base_name}.video.manifest.json#/scenes/{window.index - 1}"
            caption = f"{window.scene_id}: {_format_time(window.start_seconds)} – {_format_time(window.end_seconds)}"
            output_blocks.append(
                {
                    "heading": f"Scene {window.index:04d} — {caption}",
                    "level": 1,
                    "parent_headings": ["Video overview"],
                    "placeholder_key": window.scene_id,
                    "asset_ref": asset_name,
                    "scene_id": window.scene_id,
                    "start_ms": round(window.start_seconds * 1000),
                    "end_ms": round(window.end_seconds * 1000),
                    "frame_timestamps_ms": frame_timestamps,
                    "manifest_ref": manifest_ref,
                    "caption": caption,
                    "content": (
                        f"[VIDEO_SCENE scene_id={window.scene_id} "
                        f"start={_format_time(window.start_seconds)} "
                        f"end={_format_time(window.end_seconds)}]\n"
                        f"Transcript: {transcript}\n\n{{{{IMG:{window.scene_id}}}}}"
                    ),
                }
            )
            scene_records.append(
                {
                    "scene_id": window.scene_id,
                    "index": window.index,
                    "start_ms": round(window.start_seconds * 1000),
                    "end_ms": round(window.end_seconds * 1000),
                    "frame_timestamps_ms": frame_timestamps,
                    "contact_sheet": f"{asset_dir.name}/{asset_name}",
                    "transcript": transcript,
                }
            )

        manifest_name = f"{base_name}.video.manifest.json"
        write_video_manifest(
            parsed_dir / manifest_name,
            document_name=source.name,
            media=_media_dict(media),
            scenes=scene_records,
            transcript_segments=transcript_segments,
            warnings=warnings,
        )
        (parsed_dir / f"{base_name}.video.transcript.txt").write_text(
            "\n\n".join(full_transcript) + "\n", encoding="utf-8"
        )
        metadata = {
            "title": source.stem,
            "manifest_name": manifest_name,
            "media": _media_dict(media),
            "scene_count": len(scene_records),
        }
        return output_blocks, {"warnings": warnings}, metadata

    def build_ir(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        metadata: dict[str, Any],
    ) -> Any:
        return build_video_ir(
            blocks,
            document_name=document_name,
            asset_dir_name=asset_dir_name,
            metadata=metadata,
        )

    def surface_warnings(self, warnings: dict[str, Any], source: Path) -> dict[str, Any] | None:
        values = warnings.get("warnings") if isinstance(warnings, dict) else None
        return {"warnings": list(values)} if values else None


class _VideoConfig:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    @classmethod
    def from_env(cls) -> "_VideoConfig":
        return cls(
            asr_host=os.getenv("VIDEO_ASR_BINDING_HOST", "").strip(),
            asr_api_key=os.getenv("VIDEO_ASR_BINDING_API_KEY", "").strip(),
            asr_model=os.getenv("VIDEO_ASR_MODEL", "qwen3-asr-1.7b").strip(),
            language=os.getenv("VIDEO_ASR_LANGUAGE", "").strip(),
            max_duration_seconds=_env_float("VIDEO_MAX_DURATION_SECONDS", 21600.0),
            scene_threshold=_env_float("VIDEO_SCENE_THRESHOLD", 0.30),
            max_scene_seconds=_env_float("VIDEO_MAX_SCENE_SECONDS", 45.0),
            min_scene_seconds=_env_float("VIDEO_MIN_SCENE_SECONDS", 2.0),
            max_scenes=_env_int("VIDEO_MAX_SCENES", 128),
            frames_per_scene=_env_int("VIDEO_FRAMES_PER_SCENE", 4),
            asr_segment_seconds=_env_float("VIDEO_ASR_SEGMENT_SECONDS", 30.0),
            asr_timeout=_env_float("VIDEO_ASR_TIMEOUT", 180.0),
            ffmpeg_timeout=_env_float("VIDEO_FFMPEG_TIMEOUT", 180.0),
            ffprobe_timeout=_env_float("VIDEO_FFPROBE_TIMEOUT", 60.0),
            contact_sheet_cell_width=_env_int("VIDEO_CONTACT_SHEET_CELL_WIDTH", 384),
            contact_sheet_quality=_env_int("VIDEO_CONTACT_SHEET_JPEG_QUALITY", 82),
        )


def _env_float(name: str, fallback: float) -> float:
    try:
        return max(0.01, float(os.getenv(name, str(fallback))))
    except ValueError:
        return fallback


def _env_int(name: str, fallback: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(fallback))))
    except ValueError:
        return fallback


def _bounded_segments(start: float, end: float, segment_seconds: float):
    size = max(1.0, segment_seconds)
    current = start
    while current < end - 0.01:
        next_end = min(end, current + size)
        yield current, next_end
        current = next_end


def _overview_text(name: str, media: MediaInfo, scene_count: int) -> str:
    audio = "yes" if media.has_audio else "no"
    return (
        f"Video file: {name}\n"
        f"Duration: {_format_time(media.duration_seconds)}\n"
        f"Resolution: {media.width or '?'}x{media.height or '?'}\n"
        f"Audio track: {audio}\n"
        f"Scene windows: {scene_count}"
    )


def _media_dict(media: MediaInfo) -> dict[str, Any]:
    return {
        "duration_seconds": media.duration_seconds,
        "width": media.width,
        "height": media.height,
        "fps": media.fps,
        "has_audio": media.has_audio,
        "video_codec": media.video_codec,
        "audio_codec": media.audio_codec,
    }


def _format_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
