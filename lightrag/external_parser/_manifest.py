"""Shared ``_manifest.json`` schema for ``external_parser/<engine>/`` bundles.

The manifest is the *atomic success marker* for a raw bundle. Its presence
implies "all files in this directory finished downloading"; its content is
the cache key for "is this bundle for the same source file, the same engine
version, the same endpoint, and the same option signature we are using right
now?".

Write path: :func:`write_manifest` writes a temp file then atomically renames
to ``_manifest.json``. A crash mid-download leaves no manifest, so the next
parse call cleanly invalidates and re-downloads.

Read path: :func:`load_manifest` returns ``None`` if absent, malformed, or
recorded under a different engine — either way the bundle is treated as
stale.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_FILENAME = "_manifest.json"
MANIFEST_VERSION = "1.0"


@dataclass
class ManifestFile:
    """One file entry inside the bundle. Size always; sha256 only for files
    where silent corruption would break the adapter (the "critical" file).
    """

    path: str  # relative to the raw dir
    size: int
    sha256: str | None = None  # ``"sha256:<hex>"`` or ``None``


@dataclass
class Manifest:
    """Generic manifest schema. ``engine`` is filled by the caller (docling /
    mineru / etc.); ``options_signature`` lets per-engine cache layers detect
    when env-driven request parameters changed without bumping the version.
    """

    engine: str
    source_content_hash: str
    source_size_bytes: int
    source_filename_at_parse: str
    critical_file: ManifestFile
    files: list[ManifestFile]
    total_size_bytes: int
    task_id: str = ""
    api_mode: str = ""
    engine_version: str = ""
    endpoint_signature: str = ""
    options_signature: str = ""
    downloaded_at: str = ""
    extras: dict = field(default_factory=dict)
    version: str = MANIFEST_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "engine": self.engine,
            "api_mode": self.api_mode,
            "engine_version": self.engine_version,
            "endpoint_signature": self.endpoint_signature,
            "options_signature": self.options_signature,
            "source_content_hash": self.source_content_hash,
            "source_size_bytes": int(self.source_size_bytes),
            "source_filename_at_parse": self.source_filename_at_parse,
            "task_id": self.task_id,
            "downloaded_at": self.downloaded_at,
            "critical_file": asdict(self.critical_file),
            "files": [asdict(f) for f in self.files],
            "total_size_bytes": int(self.total_size_bytes),
            "extras": dict(self.extras or {}),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Manifest":
        critical_raw = payload.get("critical_file") or {}
        files_raw = payload.get("files") or []
        return cls(
            version=str(payload.get("version") or MANIFEST_VERSION),
            engine=str(payload.get("engine") or ""),
            api_mode=str(payload.get("api_mode") or ""),
            engine_version=str(payload.get("engine_version") or ""),
            endpoint_signature=str(payload.get("endpoint_signature") or ""),
            options_signature=str(payload.get("options_signature") or ""),
            source_content_hash=str(payload.get("source_content_hash") or ""),
            source_size_bytes=int(payload.get("source_size_bytes") or 0),
            source_filename_at_parse=str(payload.get("source_filename_at_parse") or ""),
            task_id=str(payload.get("task_id") or ""),
            downloaded_at=str(payload.get("downloaded_at") or ""),
            critical_file=ManifestFile(
                path=str(critical_raw.get("path") or ""),
                size=int(critical_raw.get("size") or 0),
                sha256=(
                    str(critical_raw["sha256"]) if critical_raw.get("sha256") else None
                ),
            ),
            files=[
                ManifestFile(
                    path=str(f.get("path") or ""),
                    size=int(f.get("size") or 0),
                    sha256=(str(f["sha256"]) if f.get("sha256") else None),
                )
                for f in files_raw
                if isinstance(f, dict)
            ],
            total_size_bytes=int(payload.get("total_size_bytes") or 0),
            extras=dict(payload.get("extras") or {}),
        )


def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / MANIFEST_FILENAME


def load_manifest(raw_dir: Path, *, expected_engine: str) -> Manifest | None:
    """Return the parsed manifest or ``None`` if absent / malformed / for a
    different engine. ``expected_engine`` is required so a future shared raw
    dir cannot serve a bundle that belongs to another engine.
    """
    p = manifest_path(raw_dir)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != MANIFEST_VERSION:
        return None
    if payload.get("engine") != expected_engine:
        return None
    try:
        return Manifest.from_dict(payload)
    except (TypeError, ValueError):
        return None


def write_manifest(raw_dir: Path, manifest: Manifest) -> None:
    """Atomically write the manifest using temp-file + rename."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    final = manifest_path(raw_dir)
    tmp = final.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, final)


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestFile",
    "load_manifest",
    "manifest_path",
    "write_manifest",
]
