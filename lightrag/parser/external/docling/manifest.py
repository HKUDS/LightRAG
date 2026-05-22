"""Helpers for building ``_manifest.json`` for docling raw bundles.

Wraps the generic :class:`Manifest` schema with docling-specific knowledge:

- the critical file is the main ``<stem>.json`` produced by docling-serve,
- non-critical files are the markdown + every entry under ``artifacts/``,
- ``extras`` carries the fixed pipeline constants so the options signature
  remains reproducible across runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.external._manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestFile,
    write_manifest,
)
from lightrag.parser.external.docling import MANIFEST_ENGINE


def select_main_json(raw_dir: Path, source_file_path: Path) -> Path:
    """Locate the primary docling JSON inside ``raw_dir``.

    Priority: ``<source_stem>.json`` if present, else the single ``*.json``
    sitting at ``raw_dir`` root (excluding ``_manifest.json``, which always
    sits in the bundle once a download has completed and would otherwise
    collide with the bundle JSON in the fallback). Raises ``RuntimeError``
    if zero or multiple candidates exist.
    """
    preferred = raw_dir / f"{source_file_path.stem}.json"
    if preferred.is_file():
        return preferred

    candidates = sorted(
        p for p in raw_dir.glob("*.json") if p.is_file() and p.name != MANIFEST_FILENAME
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"Docling raw bundle at {raw_dir} contains no .json file")
    names = ", ".join(p.name for p in candidates)
    raise RuntimeError(
        f"Docling raw bundle at {raw_dir} has multiple .json candidates ({names}); "
        f"expected exactly one to derive the critical file from"
    )


def select_main_md(raw_dir: Path, source_file_path: Path) -> Path | None:
    """Locate the markdown twin of the main JSON. Returns ``None`` if no
    markdown was produced (defensive — docling-serve always emits one for
    ``to_formats=["json","md"]`` but we don't want to crash if it is
    missing)."""
    preferred = raw_dir / f"{source_file_path.stem}.md"
    if preferred.is_file():
        return preferred
    candidates = sorted(p for p in raw_dir.glob("*.md") if p.is_file())
    return candidates[0] if candidates else None


def build_and_write_docling_manifest(
    raw_dir: Path,
    *,
    source_file_path: Path,
    task_id: str,
    endpoint_signature: str,
    engine_version: str,
    options_signature: str,
    fixed_constants: dict[str, object],
    recorded_filename: str | None = None,
) -> Manifest:
    """Construct the manifest for a freshly downloaded docling bundle and
    persist it atomically. Returns the in-memory manifest for callers that
    need the task_id / signatures for logging.

    ``recorded_filename`` is the name passed to docling-serve at upload
    time (canonical, hint-stripped form when called from the pipeline).
    It governs both the preferred-path lookup for the bundle JSON and the
    value persisted as ``source_filename_at_parse``. When ``None``, falls
    back to ``source_file_path.name`` for backward compatibility.
    """
    lookup_path = Path(recorded_filename) if recorded_filename else source_file_path
    main_json = select_main_json(raw_dir, lookup_path)
    crit_size, crit_hash = compute_size_and_hash(main_json)
    critical = ManifestFile(
        path=main_json.relative_to(raw_dir).as_posix(),
        size=crit_size,
        sha256=crit_hash,
    )

    others: list[ManifestFile] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_dir).as_posix()
        if rel == critical.path or rel.startswith("_manifest"):
            continue
        others.append(ManifestFile(path=rel, size=path.stat().st_size))

    source_size, source_hash = compute_size_and_hash(source_file_path)
    total = crit_size + sum(f.size for f in others)

    manifest = Manifest(
        engine=MANIFEST_ENGINE,
        source_content_hash=source_hash,
        source_size_bytes=source_size,
        source_filename_at_parse=recorded_filename or source_file_path.name,
        critical_file=critical,
        files=others,
        total_size_bytes=total,
        task_id=task_id,
        endpoint_signature=endpoint_signature,
        engine_version=engine_version,
        options_signature=options_signature,
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extras={"fixed_constants": dict(fixed_constants)},
    )
    write_manifest(raw_dir, manifest)
    return manifest


__all__ = [
    "build_and_write_docling_manifest",
    "select_main_json",
    "select_main_md",
]
