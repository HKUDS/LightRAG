#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


VERSION_FILE = Path(__file__).resolve().parents[2] / "lightrag" / "_version.py"


def normalize_core_version(raw_version: str) -> str:
    if raw_version.startswith("v") and len(raw_version) > 1:
        return raw_version[1:]
    return raw_version


def update_assignment(content: str, name: str, value: str) -> str:
    pattern = rf'^{name}\s*=\s*"[^"]*"$'
    updated, count = re.subn(
        pattern,
        f'{name} = "{value}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Could not update {name} in {VERSION_FILE}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Update LightRAG version constants.")
    parser.add_argument(
        "--core-version",
        required=True,
        help="Core package version. A leading 'v' is stripped automatically.",
    )
    parser.add_argument(
        "--api-version",
        help="Optional API compatibility version override.",
    )
    args = parser.parse_args()

    core_version = normalize_core_version(args.core_version)
    content = VERSION_FILE.read_text(encoding="utf-8")
    content = update_assignment(content, "__version__", core_version)
    if args.api_version is not None:
        content = update_assignment(content, "__api_version__", args.api_version)

    VERSION_FILE.write_text(content, encoding="utf-8")

    print(f"Updated {VERSION_FILE}")
    print(f"__version__={core_version}")
    if args.api_version is not None:
        print(f"__api_version__={args.api_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
