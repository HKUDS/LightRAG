"""Tests for ``NativeMarkdownParser`` extract/resolver behaviour.

Covers base64 decoding, ``.textpack`` extraction + file-reference resolution
with a traversal guard, zip-bomb rejection, and the SSRF-guarded remote image
download (monkeypatched ``urllib``), including the env-controlled failure
policy.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from lightrag.parser.markdown import parser as md_parser
from lightrag.parser.markdown.parser import (
    NativeMarkdownParser,
    _host_is_public,
    _image_ext_from_magic,
)

# A 1x1 transparent PNG.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
    b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
    b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_B64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def _make_parser() -> NativeMarkdownParser:
    return NativeMarkdownParser()


def test_magic_detection():
    assert _image_ext_from_magic(_PNG_BYTES) == "png"
    assert _image_ext_from_magic(b"\xff\xd8\xff\xe0junk") == "jpg"
    assert _image_ext_from_magic(b"not an image") is None


def test_host_is_public_rejects_internal(monkeypatch):
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    assert _host_is_public("8.8.8.8") is True
    assert _host_is_public("127.0.0.1") is False
    assert _host_is_public("10.0.0.1") is False
    assert _host_is_public("169.254.0.1") is False
    assert _host_is_public("::1") is False
    # CGNAT 100.64.0.0/10 is not private/reserved in Python's flags but is not
    # globally routable — default-deny via is_global must still reject it.
    assert _host_is_public("100.64.0.1") is False
    # TEST-NET (192.0.2.0/24) is likewise non-global.
    assert _host_is_public("192.0.2.1") is False


def test_allowlist_permits_configured_non_public(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "100.64.0.0/10")
    assert _host_is_public("100.64.0.1") is True
    # An address outside the allowlist is still rejected.
    assert _host_is_public("10.0.0.1") is False


def test_redirect_to_non_public_host_blocked(monkeypatch):
    import email.message

    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    handler = md_parser._GuardedRedirectHandler()
    req = md_parser.urllib.request.Request("http://example.com/a")
    with pytest.raises(md_parser.urllib.error.HTTPError):
        handler.redirect_request(
            req, None, 302, "Found", email.message.Message(), "http://10.0.0.1/evil"
        )


def test_base64_image_decoded_to_asset():
    p = _make_parser()
    md = f"# H\n\n![x]({_PNG_B64})\n"
    blocks, warnings, meta = p._extract_text(md, bundle_root=None)
    assert len(meta["md_assets"]) == 1
    (asset,) = meta["md_assets"].values()
    assert asset["data"].startswith(b"\x89PNG")
    assert asset["fmt"] == "png"
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "local"
    assert not warnings


def test_base64_non_image_bytes_rejected():
    # ``data:image/png;base64,QUJD`` decodes to b"ABC" — declared PNG but not a
    # real image. Magic bytes are authoritative, so it must be skipped.
    p = _make_parser()
    md = "# H\n\n![x](data:image/png;base64,QUJD)\n"
    _, warnings, meta = p._extract_text(md, bundle_root=None)
    assert meta["md_drawings"] == {}
    assert meta["md_assets"] == {}
    assert warnings.get("images_skipped") == 1


def test_standalone_md_relative_image_is_skipped():
    p = _make_parser()
    blocks, warnings, meta = p._extract_text(
        "# H\n\n![x](assets/pic.png)\n", bundle_root=None
    )
    assert meta["md_drawings"] == {}
    assert warnings.get("images_skipped") == 1


def _write_textpack(path: Path, *, text: str, assets: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("text.markdown", text)
        for name, data in assets.items():
            zf.writestr(f"assets/{name}", data)


def test_textpack_file_reference_resolved(tmp_path: Path):
    pack = tmp_path / "note.textpack"
    _write_textpack(
        pack,
        text="# Note\n\n![pic](assets/pic.png)\n",
        assets={"pic.png": _PNG_BYTES},
    )
    p = _make_parser()
    blocks, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="note"
    )
    assert len(meta["md_assets"]) == 1
    (asset,) = meta["md_assets"].values()
    assert asset["data"] == _PNG_BYTES
    assert asset["suggested_name"] == "pic.png"
    assert not warnings


def test_textpack_image_outside_assets_dir_is_resolved(tmp_path: Path):
    # Compatibility: a reference may point anywhere under the bundle root, not
    # only ``assets/`` — here ``media/pic.png`` and a root-level ``cover.png``.
    pack = tmp_path / "note.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", "# N\n\n![a](media/pic.png)\n\n![b](cover.png)\n")
        zf.writestr("media/pic.png", _PNG_BYTES)
        zf.writestr("cover.png", _PNG_BYTES)
    p = _make_parser()
    _, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="note"
    )
    assert len(meta["md_drawings"]) == 2
    assert all(d["kind"] == "local" for d in meta["md_drawings"].values())
    # Same bytes → deduped to a single asset.
    assert len(meta["md_assets"]) == 1
    assert not warnings


def test_textpack_non_image_file_reference_is_skipped(tmp_path: Path):
    pack = tmp_path / "note.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", "# N\n\n![x](assets/fake.png)\n")
        zf.writestr("assets/fake.png", b"this is not an image")
    p = _make_parser()
    _, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="note"
    )
    assert meta["md_drawings"] == {}
    assert warnings.get("images_skipped") == 1


def test_textpack_traversal_reference_is_skipped(tmp_path: Path):
    pack = tmp_path / "evil.textpack"
    _write_textpack(
        pack,
        text="# Evil\n\n![x](../../../etc/passwd)\n",
        assets={"pic.png": _PNG_BYTES},
    )
    p = _make_parser()
    _, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="evil"
    )
    assert meta["md_drawings"] == {}
    assert warnings.get("images_skipped") == 1


def test_textpack_zip_bomb_entry_count_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(md_parser, "_TEXTPACK_MAX_ENTRIES", 3)
    pack = tmp_path / "bomb.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", "# H\n")
        for i in range(10):
            zf.writestr(f"assets/f{i}.bin", b"x")
    p = _make_parser()
    with pytest.raises(RuntimeError, match="entries"):
        p.extract(pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="bomb")


# --- remote download (monkeypatched urllib) --------------------------------


class _FakeHeaders:
    def __init__(self, content_type: str) -> None:
        self._ct = content_type

    def get_content_type(self) -> str:
        return self._ct


class _FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, content_type: str) -> None:
        super().__init__(data)
        self.headers = _FakeHeaders(content_type)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _patch_download(
    monkeypatch, *, data=_PNG_BYTES, content_type="image/png", host_public=True
):
    monkeypatch.setattr(md_parser, "_host_is_public", lambda host: host_public)

    class _Opener:
        def open(self, req, timeout=None):
            return _FakeResponse(data, content_type)

    monkeypatch.setattr(
        md_parser.urllib.request, "build_opener", lambda *a, **k: _Opener()
    )


def test_remote_image_kept_external_when_download_disabled(monkeypatch):
    monkeypatch.delenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", raising=False)
    p = _make_parser()
    _, warnings, meta = p._extract_text(
        "# H\n\n![x](http://host/y.png)\n", bundle_root=None
    )
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "external"
    assert drawing["url"] == "http://host/y.png"
    assert not meta["md_assets"]


def test_remote_image_downloaded_when_enabled(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    _patch_download(monkeypatch)
    p = _make_parser()
    _, warnings, meta = p._extract_text(
        "# H\n\n![x](http://host/y.png)\n", bundle_root=None
    )
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "local"
    (asset,) = meta["md_assets"].values()
    assert asset["data"] == _PNG_BYTES


def test_remote_image_failure_kept_external_by_default(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.delenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", raising=False)
    _patch_download(monkeypatch, content_type="text/html")  # explicit non-image
    p = _make_parser()
    _, warnings, meta = p._extract_text(
        "# H\n\n![x](http://host/y.png)\n", bundle_root=None
    )
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "external"
    assert warnings.get("images_download_failed") == 1


def test_remote_image_failure_raises_when_required(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", "true")
    _patch_download(monkeypatch, content_type="text/html")
    p = _make_parser()
    with pytest.raises(Exception):
        p._extract_text("# H\n\n![x](http://host/y.png)\n", bundle_root=None)


def test_remote_image_private_host_blocked(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    monkeypatch.delenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", raising=False)
    _patch_download(monkeypatch, host_public=False)
    p = _make_parser()
    _, warnings, meta = p._extract_text(
        "# H\n\n![x](http://169.254.0.1/y.png)\n", bundle_root=None
    )
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "external"
    assert warnings.get("images_download_failed") == 1
