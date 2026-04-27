#!/usr/bin/env python3
"""Seed Little Bull Knowledge demo documents into workspace input folders."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIR = REPO_ROOT / "fixtures" / "little_bull_knowledge"
DEFAULT_INPUT_DIR = REPO_ROOT / "inputs"
WORKSPACE_PATTERN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


@dataclass(frozen=True)
class SeededFile:
    workspace: str
    source: Path
    target: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "workspace": self.workspace,
            "source": display_path(self.source),
            "target": display_path(self.target),
        }


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sanitize_workspace(name: str) -> str:
    sanitized = "".join(char if char in WORKSPACE_PATTERN else "_" for char in name)
    return sanitized.strip("_")


def discover_workspace_dirs(fixture_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in fixture_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def copy_workspace_documents(
    fixture_workspace_dir: Path,
    input_dir: Path,
    overwrite: bool,
) -> list[SeededFile]:
    workspace = sanitize_workspace(fixture_workspace_dir.name)
    if not workspace:
        raise ValueError(f"Invalid workspace name: {fixture_workspace_dir.name}")

    target_dir = input_dir / workspace
    target_dir.mkdir(parents=True, exist_ok=True)

    seeded: list[SeededFile] = []
    for source in sorted(fixture_workspace_dir.glob("*.md")):
        target = target_dir / source.name
        if target.exists() and not overwrite:
            continue
        shutil.copy2(source, target)
        seeded.append(SeededFile(workspace=workspace, source=source, target=target))
    return seeded


def seed_demo_documents(
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    input_dir: Path = DEFAULT_INPUT_DIR,
    overwrite: bool = False,
) -> list[SeededFile]:
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")

    input_dir.mkdir(parents=True, exist_ok=True)
    seeded: list[SeededFile] = []
    for workspace_dir in discover_workspace_dirs(fixture_dir):
        seeded.extend(copy_workspace_documents(workspace_dir, input_dir, overwrite))
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy Little Bull Knowledge demo documents into inputs/<workspace>/."
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory containing workspace fixture folders.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="LightRAG input directory that receives workspace folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing seeded files.",
    )
    args = parser.parse_args()

    seeded = seed_demo_documents(args.fixture_dir, args.input_dir, args.overwrite)
    print(
        json.dumps(
            {
                "status": "success",
                "seeded_count": len(seeded),
                "files": [item.as_dict() for item in seeded],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
