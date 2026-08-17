"""Tests for the native-markdown downloaded-image cache (`.native_raw/`).

The cache lets a re-parse of an unchanged file reuse already-downloaded images
instead of hitting the network. ``_download`` is monkeypatched so a cache hit is
proven by it NOT being called (it raises) while assets still materialize.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lightrag.parser.markdown import parser as md_parser
from lightrag.parser.markdown.parser import NativeMarkdownParser

from tests.parser.markdown.conftest import PNG_BYTES as _PNG_BYTES

_URL = "http://host/y.png"
_MD = f"# H\n\n![x]({_URL})\n"


@pytest.fixture(autouse=True)
def _enable_download(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.delenv("LIGHTRAG_FORCE_REPARSE_NATIVE", raising=False)
    monkeypatch.delenv("NATIVE_MD_IMAGE_MAX_BYTES", raising=False)
    monkeypatch.delenv("NATIVE_MD_IMAGE_MAX_SVG_PIXELS", raising=False)
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)


def _patch_download(monkeypatch, payload=(_PNG_BYTES, "png")):
    """Patch ``_download`` to count calls and return fixed bytes."""
    counter = {"n": 0}

    def _fake(self, src):
        counter["n"] += 1
        return payload

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _fake)
    return counter


def _forbid_download(monkeypatch):
    """Patch ``_download`` to fail the test if the network is touched."""

    def _boom(self, src):  # pragma: no cover - must not be called on a cache hit
        raise AssertionError("network download must not run on a cache hit")

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _boom)


def _make_doc(tmp_path: Path, text: str = _MD) -> tuple[Path, Path]:
    src = tmp_path / "doc.md"
    src.write_text(text)
    parsed = tmp_path / "__parsed__" / "doc.md.parsed"
    parsed.mkdir(parents=True)
    (parsed / "doc.blocks.assets").mkdir()
    return src, parsed


def _extract(p: NativeMarkdownParser, src: Path, parsed: Path):
    return p.extract(
        src, parsed_dir=parsed, asset_dir=parsed / "doc.blocks.assets", base_name="doc"
    )


def _wipe_parsed(parsed: Path) -> None:
    """Mirror NativeParserBase.parse's rmtree(parsed_dir) before a re-extract."""
    shutil.rmtree(parsed)
    parsed.mkdir(parents=True)
    (parsed / "doc.blocks.assets").mkdir()


def _raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "__parsed__" / "doc.md.native_raw"


def test_first_parse_downloads_and_writes_bundle(tmp_path, monkeypatch):
    counter = _patch_download(monkeypatch)
    src, parsed = _make_doc(tmp_path)
    _, _, meta = _extract(NativeMarkdownParser(), src, parsed)
    assert counter["n"] == 1
    assert len(meta["md_assets"]) == 1
    raw = _raw_dir(tmp_path)
    assert (raw / "_manifest.json").is_file()
    # Exactly one cached image file alongside the manifest.
    files = sorted(c.name for c in raw.iterdir())
    assert len(files) == 2 and "_manifest.json" in files


def test_reparse_unchanged_source_is_cache_hit(tmp_path, monkeypatch):
    _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)  # populate cache
    _wipe_parsed(parsed)  # base parser would rmtree the .parsed dir

    _forbid_download(monkeypatch)  # any download now fails the test
    _, warnings, meta = _extract(p, src, parsed)
    assert warnings.get("images_cache_hit") == 1
    assert len(meta["md_assets"]) == 1
    (asset,) = meta["md_assets"].values()
    assert asset["data"] == _PNG_BYTES


def test_pure_cache_hit_does_not_touch_bundle(tmp_path, monkeypatch):
    # A pure hit must write nothing: the manifest + image mtimes stay frozen, so
    # on-disk timestamps alone reveal whether the cache was hit.
    _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)
    raw = _raw_dir(tmp_path)
    before = {c.name: c.stat().st_mtime_ns for c in raw.iterdir()}

    _wipe_parsed(parsed)
    _forbid_download(monkeypatch)  # pure hit: no download allowed
    _, warnings, _ = _extract(p, src, parsed)
    assert warnings.get("images_cache_hit") == 1

    after = {c.name: c.stat().st_mtime_ns for c in raw.iterdir()}
    assert after == before  # nothing rewritten


def test_source_change_invalidates_cache(tmp_path, monkeypatch):
    counter = _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)
    assert counter["n"] == 1
    src.write_text("# Changed\n\n![x](http://host/y.png)\n")
    _wipe_parsed(parsed)
    _extract(p, src, parsed)
    assert counter["n"] == 2  # re-downloaded


def test_options_signature_change_invalidates_cache(tmp_path, monkeypatch):
    counter = _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)
    assert counter["n"] == 1
    # Changing a byte-affecting knob busts the bundle.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_BYTES", "12345")
    _wipe_parsed(parsed)
    _extract(p, src, parsed)
    assert counter["n"] == 2


def test_force_reparse_discards_cache(tmp_path, monkeypatch):
    counter = _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)
    assert counter["n"] == 1
    monkeypatch.setenv("LIGHTRAG_FORCE_REPARSE_NATIVE", "true")
    _wipe_parsed(parsed)
    _extract(p, src, parsed)
    assert counter["n"] == 2


def test_tampered_cache_file_falls_back_to_download(tmp_path, monkeypatch):
    counter = _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)
    assert counter["n"] == 1
    # Corrupt the cached image bytes so the sha256 no longer matches the manifest.
    raw = _raw_dir(tmp_path)
    img = next(c for c in raw.iterdir() if c.name != "_manifest.json")
    img.write_bytes(b"corrupted")
    _wipe_parsed(parsed)
    _extract(p, src, parsed)
    assert counter["n"] == 2  # integrity mismatch -> miss -> re-download


def test_svg_cached_as_png_and_reused_without_rasterizing(tmp_path, monkeypatch):
    # _download returns the post-rasterization PNG, so a cache hit reuses PNG
    # bytes and never calls cairosvg.
    _patch_download(monkeypatch, payload=(_PNG_BYTES, "png"))
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path, text=f"# H\n\n![s]({_URL})\n")
    _, _, meta = _extract(p, src, parsed)
    (asset,) = meta["md_assets"].values()
    assert asset["fmt"] == "png" and asset["data"].startswith(b"\x89PNG")

    _wipe_parsed(parsed)
    _forbid_download(monkeypatch)

    import cairosvg

    def _boom(*a, **k):  # pragma: no cover - must not run on a cache hit
        raise AssertionError("cairosvg must not run on a cache hit")

    monkeypatch.setattr(cairosvg, "svg2png", _boom)
    _, warnings, meta = _extract(p, src, parsed)
    assert warnings.get("images_cache_hit") == 1


def test_a_failed_required_parse_persists_downloads_for_the_retry(
    tmp_path, monkeypatch
):
    """Fix-proof for the re-download loop: with DOWNLOAD_REQUIRED and a request
    budget smaller than the image count, attempt 1 dies over budget AFTER
    downloading what it could. Before the fix its bundle had no manifest, so
    attempt 2 wiped it, re-downloaded the same image and failed identically —
    forever. With the failure-path flush, attempt 2 serves image A from the
    cache (a hit costs no request budget), spends its one request on image B,
    and succeeds.
    """
    counter = {"n": 0}

    def _fake(self, src):
        counter["n"] += 1
        return (_PNG_BYTES + src.encode(), "png")

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _fake)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "1")
    p = NativeMarkdownParser()
    src, parsed = _make_doc(
        tmp_path, text="# H\n\n![a](http://host/a.png)\n\n![b](http://host/b.png)\n"
    )

    with pytest.raises(md_parser._ImageRequestBudgetExceeded):
        _extract(p, src, parsed)
    assert counter["n"] == 1  # image A was downloaded before the budget stop

    _wipe_parsed(parsed)
    _, warnings, meta = _extract(p, src, parsed)
    assert warnings.get("images_cache_hit") == 1  # A came from the bundle
    assert counter["n"] == 2  # only B hit the network on the retry
    assert len(meta["md_assets"]) == 2


def test_failure_flush_keeps_prior_entries_the_aborted_run_never_reached(tmp_path):
    """flush(prune=False) must union the prior valid bundle into the manifest:
    pruning keys on the entries referenced THIS run, and an aborted run never
    referenced the images it did not get to."""
    from lightrag.parser.markdown.raw_cache import (
        NativeImageRawCache,
        native_md_options_signature,
    )

    src = tmp_path / "doc.md"
    src.write_text(_MD)
    raw = tmp_path / "doc.md.native_raw"
    sig = native_md_options_signature()

    def _cache() -> NativeImageRawCache:
        cache = NativeImageRawCache(
            raw, source_path=src, options_signature=sig, force_reparse=False
        )
        cache.load()
        return cache

    first = _cache()
    first.put("http://host/a.png", b"A" * 8, "png")
    first.put("http://host/b.png", b"B" * 8, "png")
    first.flush()

    # A later aborted run reaches only a NEW image before dying: neither prior
    # entry is revisited, and neither may be pruned.
    aborted = _cache()
    aborted.put("http://host/c.png", b"C" * 8, "png")
    aborted.flush(prune=False)

    retry = _cache()
    assert retry.get("http://host/a.png") == (b"A" * 8, "png")
    assert retry.get("http://host/b.png") == (b"B" * 8, "png")
    assert retry.get("http://host/c.png") == (b"C" * 8, "png")


def test_an_aborted_first_parse_still_persists_its_downloads(tmp_path):
    from lightrag.parser.markdown.raw_cache import (
        NativeImageRawCache,
        native_md_options_signature,
    )

    src = tmp_path / "doc.md"
    src.write_text(_MD)
    raw = tmp_path / "doc.md.native_raw"
    sig = native_md_options_signature()

    aborted = NativeImageRawCache(
        raw, source_path=src, options_signature=sig, force_reparse=False
    )
    aborted.load()  # no manifest yet: the bundle is invalid
    aborted.put(_URL, _PNG_BYTES, "png")
    aborted.flush(prune=False)

    retry = NativeImageRawCache(
        raw, source_path=src, options_signature=sig, force_reparse=False
    )
    retry.load()
    assert retry.get(_URL) == (_PNG_BYTES, "png")


def test_an_over_budget_cache_hit_warns_once(tmp_path, monkeypatch, caplog):
    """The cache-hit degrade path used to warn twice for the same image (once
    in _local, once in _budget_degraded) while the mid-read path warned once."""
    import logging

    from lightrag.utils import logger as lightrag_logger

    _patch_download(monkeypatch)
    p = NativeMarkdownParser()
    src, parsed = _make_doc(tmp_path)
    _extract(p, src, parsed)  # populate the cache

    _wipe_parsed(parsed)
    _forbid_download(monkeypatch)
    # Tighter than the cached image: the hit is refused by the byte budget
    # (budgets are excluded from the options signature, so the bundle stays
    # valid) and degrades to an external link.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", "1")
    monkeypatch.setattr(lightrag_logger, "propagate", True)
    with caplog.at_level(logging.WARNING, logger="lightrag"):
        _, warnings, _ = _extract(p, src, parsed)

    assert warnings.get("images_byte_budget_exceeded") == 1
    budget_warnings = [
        r for r in caplog.records if "budget exhausted" in r.getMessage()
    ]
    assert len(budget_warnings) == 1


def test_native_raw_is_sibling_and_survives_parsed_rmtree(tmp_path, monkeypatch):
    _patch_download(monkeypatch)
    src, parsed = _make_doc(tmp_path)
    _extract(NativeMarkdownParser(), src, parsed)
    raw = _raw_dir(tmp_path)
    assert raw.is_dir()
    assert raw.parent == parsed.parent  # sibling of .parsed, not nested
    shutil.rmtree(parsed)  # base parser wipes parsed_dir
    assert raw.is_dir()  # cache survives
