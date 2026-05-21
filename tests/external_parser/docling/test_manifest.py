"""Tests for ``lightrag/external_parser/docling/manifest.py`` helpers.

Targets the contract guarantees that the rest of the docling flow relies on:
``select_main_json`` must find the bundle's main JSON even when ``_manifest.json``
sits alongside it, and the preferred-path lookup must take priority over the
fallback glob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.external_parser.docling.manifest import select_main_json


def _touch(path: Path, content: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_select_main_json_preferred_path_hits(tmp_path: Path) -> None:
    # Manifest is present (the typical post-download state), but the preferred
    # ``<stem>.json`` exists, so the fallback glob is not consulted at all.
    _touch(tmp_path / "report.json")
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    assert select_main_json(tmp_path, Path("report.pdf")) == tmp_path / "report.json"


def test_select_main_json_fallback_ignores_manifest(tmp_path: Path) -> None:
    # Defensive: when the preferred path misses (e.g. docling-serve renamed
    # the stem for whatever reason), the fallback glob must NOT confuse
    # ``_manifest.json`` for a bundle JSON. Pre-fix this case raised
    # "multiple .json candidates".
    _touch(tmp_path / "report.json")
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    assert select_main_json(tmp_path, Path("other.pdf")) == tmp_path / "report.json"


def test_select_main_json_raises_when_only_manifest_present(tmp_path: Path) -> None:
    # If the bundle JSON is genuinely missing, the manifest alone is not a
    # valid substitute — the helper must still raise rather than silently
    # returning the manifest.
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    with pytest.raises(RuntimeError, match="contains no .json file"):
        select_main_json(tmp_path, Path("report.pdf"))


def test_select_main_json_raises_on_real_ambiguity(tmp_path: Path) -> None:
    # Two genuine bundle JSONs is still an error; the manifest filter must
    # not mask multi-candidate detection.
    _touch(tmp_path / "report.json")
    _touch(tmp_path / "extra.json")
    _touch(tmp_path / "_manifest.json", '{"engine":"docling"}')
    with pytest.raises(RuntimeError, match="multiple .json candidates"):
        select_main_json(tmp_path, Path("other.pdf"))
