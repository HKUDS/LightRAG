#!/usr/bin/env python
"""Regenerate the byte-equivalence golden fixtures for the native docx → IR
migration.

The fixtures live at
``tests/parser_adapters/golden/native_docx/<scenario>/``
and capture the exact on-disk artifacts the legacy
``parse_docx_to_lightrag_document`` produces for each scenario in
``tests/parser_adapters/_native_docx_fixtures.py``.

After the migration to ``NativeDocxAdapter`` + ``write_sidecar``, the
golden tests assert that the new flow yields byte-identical files.

Usage::

    python scripts/regen_native_docx_golden.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


_FROZEN_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)


def _ensure_clean_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


async def _regen() -> None:
    # Lazy imports so the deterministic clock patch takes effect before
    # the adapter module captures ``datetime``.
    from lightrag.native_parser.docx import lightrag_adapter

    from parser_adapters._native_docx_fixtures import SCENARIOS

    fixtures_root = (
        PROJECT_ROOT / "tests" / "parser_adapters" / "golden" / "native_docx"
    )
    fixtures_root.mkdir(parents=True, exist_ok=True)

    for scenario in SCENARIOS:
        scenario_dir = fixtures_root / scenario.name
        _ensure_clean_dir(scenario_dir)

        def _stub_extract(
            file_path,
            *,
            fixlevel=None,
            drawing_context=None,
            parse_warnings=None,
            parse_metadata=None,
            **_kwargs,
        ):
            # Bring the upstream parser's side effects in: extract assets
            # into the asset dir and forward parse_metadata.
            if drawing_context is not None and scenario.assets:
                drawing_context.export_dir_path.mkdir(parents=True, exist_ok=True)
                for name, data in scenario.assets.items():
                    (drawing_context.export_dir_path / name).write_bytes(data)
            if parse_metadata is not None:
                parse_metadata.update(scenario.parse_metadata)
            return [dict(b) for b in scenario.blocks]

        with (
            mock.patch.object(lightrag_adapter, "extract_docx_blocks", _stub_extract),
            mock.patch(
                # ``parse_time`` is stamped inside ``write_sidecar``; freezing
                # ``datetime`` there is what keeps regen output deterministic.
                "lightrag.sidecar.writer.datetime",
                _FrozenDateTime,
            ),
        ):
            await lightrag_adapter.parse_docx_to_lightrag_document(
                file_bytes=b"fake-docx",
                file_path=scenario.file_path,
                doc_id=scenario.doc_id,
                source_path=str(scenario_dir / scenario.file_path),
                output_dir=scenario_dir / "out",
            )

        # Move every file under out/ into scenario_dir/ root for easier
        # comparison. Drop empty asset dirs (current adapter prunes them).
        produced_dir = scenario_dir / "out"
        for item in produced_dir.rglob("*"):
            rel = item.relative_to(produced_dir)
            target = scenario_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                target.mkdir(exist_ok=True)
            else:
                shutil.copyfile(item, target)
        shutil.rmtree(produced_dir)
        print(f"[regen] {scenario.name}: wrote {scenario_dir}")


def main() -> None:
    asyncio.run(_regen())


if __name__ == "__main__":
    main()
