#!/usr/bin/env python
"""Regenerate the byte-equivalence golden fixtures for the native docx
sidecar pipeline.

The fixtures live at
``tests/native_parser/docx/golden/native_docx/<scenario>/``
and capture the exact on-disk artifacts ``LightRAG.parse_native`` produces
for each scenario in ``tests/native_parser/docx/_native_docx_fixtures.py``.

Usage::

    python scripts/regen_native_docx_golden.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "native_parser" / "docx"))


async def _regen() -> None:
    from lightrag.constants import (
        FULL_DOCS_FORMAT_PENDING_PARSE,
        PARSED_DIR_NAME,
    )
    from lightrag.parser_debug import (
        FrozenDateTime,
        build_debug_rag,
    )
    import lightrag.pipeline as pipeline_module

    from _native_docx_fixtures import SCENARIOS  # type: ignore[import]

    fixtures_root = (
        PROJECT_ROOT / "tests" / "native_parser" / "docx" / "golden" / "native_docx"
    )
    fixtures_root.mkdir(parents=True, exist_ok=True)

    async def _noop_archive(_p: str) -> None:
        return None

    for scenario in SCENARIOS:
        scenario_dir = fixtures_root / scenario.name
        if scenario_dir.exists():
            shutil.rmtree(scenario_dir)
        scenario_dir.mkdir(parents=True, exist_ok=True)

        def _stub_extract(
            file_path,
            *,
            fixlevel=None,
            drawing_context=None,
            parse_warnings=None,
            parse_metadata=None,
            **_kwargs,
        ):
            if drawing_context is not None and scenario.assets:
                drawing_context.export_dir_path.mkdir(parents=True, exist_ok=True)
                for name, data in scenario.assets.items():
                    (drawing_context.export_dir_path / name).write_bytes(data)
            if parse_metadata is not None:
                parse_metadata.update(scenario.parse_metadata)
            return [dict(b) for b in scenario.blocks]

        with tempfile.TemporaryDirectory(prefix="regen_") as tmp_root:
            input_dir = Path(tmp_root) / "inputs"
            input_dir.mkdir()
            source_path = input_dir / scenario.file_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"fake-docx")

            rag = build_debug_rag()

            with (
                mock.patch.dict("os.environ", {"INPUT_DIR": str(input_dir)}),
                mock.patch(
                    "lightrag.native_parser.docx.parse_document.extract_docx_blocks",
                    _stub_extract,
                ),
                mock.patch.object(
                    pipeline_module,
                    "archive_docx_source_after_full_docs_sync",
                    _noop_archive,
                ),
                mock.patch("lightrag.sidecar.writer.datetime", FrozenDateTime),
            ):
                await rag.parse_native(
                    scenario.doc_id,
                    str(source_path),
                    {
                        "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                        "content": "",
                        "source_path": str(source_path),
                    },
                )

            produced_dir = input_dir / PARSED_DIR_NAME / f"{scenario.file_path}.parsed"
            for item in produced_dir.rglob("*"):
                rel = item.relative_to(produced_dir)
                target = scenario_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.is_dir():
                    target.mkdir(exist_ok=True)
                else:
                    shutil.copyfile(item, target)
        print(f"[regen] {scenario.name}: wrote {scenario_dir}")


def main() -> None:
    asyncio.run(_regen())


if __name__ == "__main__":
    main()
