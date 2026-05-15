#!/usr/bin/env python
"""Regenerate the byte-equivalence golden fixtures for the native docx
sidecar pipeline.

The fixtures live at
``tests/parser_adapters/golden/native_docx/<scenario>/``
and capture the exact on-disk artifacts ``LightRAG.parse_native`` produces
for each scenario in ``tests/parser_adapters/_native_docx_fixtures.py``.

Usage::

    python scripts/regen_native_docx_golden.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))


_FROZEN_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)


class _DebugFullDocs:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def upsert(self, payload: dict[str, Any]) -> None:
        self.data.update(payload)

    async def get_by_id(self, doc_id: str) -> Any:
        return self.data.get(doc_id)

    async def index_done_callback(self) -> None:
        return None


class _DebugDocStatus:
    async def get_by_id(self, doc_id: str) -> Any:
        return None

    async def upsert(self, data: dict[str, Any]) -> None:
        return None


def _build_debug_rag():
    from lightrag import LightRAG

    class _DebugRag:
        _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
        parse_native = LightRAG.parse_native

        def __init__(self) -> None:
            self.full_docs = _DebugFullDocs()
            self.doc_status = _DebugDocStatus()

        def _resolve_source_file_for_parser(self, file_path: str) -> str:
            return file_path

    return _DebugRag()


async def _regen() -> None:
    # Lazy import so the frozen-clock patch can take effect inside the
    # writer when ``parse_native`` resolves it.
    from lightrag.constants import (
        FULL_DOCS_FORMAT_PENDING_PARSE,
        PARSED_DIR_NAME,
    )
    import lightrag.pipeline as pipeline_module

    from parser_adapters._native_docx_fixtures import SCENARIOS  # type: ignore[import]

    fixtures_root = (
        PROJECT_ROOT / "tests" / "parser_adapters" / "golden" / "native_docx"
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
            import os

            input_dir = Path(tmp_root) / "inputs"
            input_dir.mkdir()
            os.environ["INPUT_DIR"] = str(input_dir)
            source_path = input_dir / scenario.file_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(b"fake-docx")

            rag = _build_debug_rag()

            with (
                mock.patch(
                    "lightrag.native_parser.docx.parse_document.extract_docx_blocks",
                    _stub_extract,
                ),
                mock.patch.object(
                    pipeline_module,
                    "archive_docx_source_after_full_docs_sync",
                    _noop_archive,
                ),
                mock.patch("lightrag.sidecar.writer.datetime", _FrozenDateTime),
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
