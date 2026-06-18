from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ALLOWED_PATCH_TARGET_PREFIXES = (
    "prompts/",
    "lightrag/medical_kg/",
    "docs/",
    "lightrag_webui/src/components/kg-maintenance/",
)
DISALLOWED_PATCH_TARGETS = {".env", "uv.lock"}
PROPOSAL_ID_PATTERN = re.compile(r"^[a-z0-9_.-]+$")


@dataclass(frozen=True)
class PatchCandidate:
    proposal_id: str
    target_path: str
    diff_text: str


def _normalize_target(target_path: str) -> str:
    normalized = target_path.strip().replace("\\", "/")
    path = PurePosixPath(normalized)

    if not normalized:
        raise ValueError("Patch target cannot be empty")

    if ":" in normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError("Patch target must be a relative repository path")

    return path.as_posix()


def validate_patch_candidate(candidate: PatchCandidate) -> None:
    if not PROPOSAL_ID_PATTERN.fullmatch(candidate.proposal_id):
        raise ValueError("Unsafe proposal id")

    if not candidate.diff_text.strip():
        raise ValueError("Patch diff text cannot be empty")

    target_path = _normalize_target(candidate.target_path)

    if target_path in DISALLOWED_PATCH_TARGETS:
        raise ValueError("Patch target is disallowed")

    if not any(target_path.startswith(prefix) for prefix in ALLOWED_PATCH_TARGET_PREFIXES):
        raise ValueError("Patch target is outside allowed prefixes")


def write_patch_candidate(candidate: PatchCandidate, output_dir: str | Path) -> Path:
    validate_patch_candidate(candidate)

    patch_dir = Path(output_dir) / "patch_candidates"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / f"{candidate.proposal_id}.patch"
    patch_path.write_text(candidate.diff_text, encoding="utf-8")
    return patch_path
