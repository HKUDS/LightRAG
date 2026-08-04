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
    _image_bytes_and_ext,
    _image_ext_from_magic,
    _looks_like_svg,
    _pin_socket,
    _unwrap_embedded_ipv4,
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


_SVG_BYTES = (
    b'<?xml version="1.0"?>\n'
    b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4">'
    b'<rect width="4" height="4" fill="red"/></svg>'
)


def test_magic_detection():
    assert _image_ext_from_magic(_PNG_BYTES) == "png"
    assert _image_ext_from_magic(b"\xff\xd8\xff\xe0junk") == "jpg"
    assert _image_ext_from_magic(b"not an image") is None
    # SVG is text, not a raster magic — recognised only by the SVG sniffer.
    assert _image_ext_from_magic(_SVG_BYTES) is None


def test_looks_like_svg():
    assert _looks_like_svg(_SVG_BYTES) is True
    assert _looks_like_svg(b"\xef\xbb\xbf  \n<svg></svg>") is True  # BOM + space
    assert _looks_like_svg(_PNG_BYTES) is False
    assert _looks_like_svg(b"<html><body>no svg</body></html>") is False


def test_svg_rasterized_to_png_via_coerce():
    coerced = _image_bytes_and_ext(
        _SVG_BYTES, max_bytes=25 * 1024 * 1024, max_svg_pixels=16_000_000
    )
    assert coerced is not None
    data, ext = coerced
    assert ext == "png"
    assert data.startswith(b"\x89PNG")


def test_svg_coerce_skips_when_rasterization_fails(monkeypatch):
    # cairosvg unavailable / render failure -> coerce returns None (caller skips).
    monkeypatch.setattr(md_parser, "_rasterize_svg", lambda raw, **kw: None)
    assert (
        _image_bytes_and_ext(_SVG_BYTES, max_bytes=25 * 1024 * 1024, max_svg_pixels=1)
        is None
    )


def test_svg_coerce_skips_when_rasterized_png_too_large(monkeypatch):
    monkeypatch.setattr(md_parser, "_rasterize_svg", lambda raw, **kw: _PNG_BYTES)
    assert (
        _image_bytes_and_ext(_SVG_BYTES, max_bytes=4, max_svg_pixels=16_000_000) is None
    )


def test_svg_dimensions_parsed_from_width_height_and_viewbox():
    assert md_parser._svg_pixel_dimensions(_SVG_BYTES) == (4, 4)
    # Unit conversion: 1in = 96px, 2in = 192px.
    svg_in = b'<svg xmlns="http://www.w3.org/2000/svg" width="1in" height="2in"/>'
    assert md_parser._svg_pixel_dimensions(svg_in) == (96, 192)
    # width/height missing or relative (%) -> fall back to viewBox w/h.
    svg_vb = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100%" viewBox="0 0 30 20"/>'
    )
    assert md_parser._svg_pixel_dimensions(svg_vb) == (30, 20)
    # Neither resolvable -> None (skipped by the rasterizer).
    svg_none = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
    assert md_parser._svg_pixel_dimensions(svg_none) is None


def test_svg_oversized_canvas_rejected_before_render(monkeypatch):
    # A tiny SVG declaring a huge canvas is rejected on the pre-render pixel
    # budget — cairosvg.svg2png must never be reached.
    import cairosvg

    def _boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("cairosvg.svg2png should not run for oversized canvas")

    monkeypatch.setattr(cairosvg, "svg2png", _boom)
    big = b'<svg xmlns="http://www.w3.org/2000/svg" width="100000" height="100000"/>'
    assert md_parser._rasterize_svg(big, max_pixels=16_000_000) is None


def test_base64_svg_decoded_and_rasterized():
    import base64 as _b64

    p = _make_parser()
    b64 = _b64.b64encode(_SVG_BYTES).decode()
    md = f"# H\n\n![x](data:image/svg+xml;base64,{b64})\n"
    _, warnings, meta = p._extract_text(md, bundle_root=None)
    (asset,) = meta["md_assets"].values()
    assert asset["fmt"] == "png"
    assert asset["data"].startswith(b"\x89PNG")
    (drawing,) = meta["md_drawings"].values()
    assert drawing["kind"] == "local"
    assert not warnings


def test_textpack_svg_file_rasterized_to_png(tmp_path: Path):
    pack = tmp_path / "note.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", "# N\n\n![logo](assets/logo.svg)\n")
        zf.writestr("assets/logo.svg", _SVG_BYTES)
    p = _make_parser()
    _, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="note"
    )
    (asset,) = meta["md_assets"].values()
    assert asset["fmt"] == "png"
    assert asset["data"].startswith(b"\x89PNG")
    # The on-disk name takes the rasterized extension.
    assert asset["suggested_name"] == "logo.png"
    assert not warnings


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


def test_host_is_public_rejects_ipv6_transition_wrapped_internal(monkeypatch):
    # GHSA-vv3m-f8x4-7377: an internal IPv4 smuggled inside an IPv6 transition
    # wrapper must be judged by its embedded IPv4, not the wrapper's own
    # ``is_global``. On a NAT64/DNS64 host these literals route to the internal
    # target, so every wrapped form of a non-public IPv4 must be blocked.
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    # 127.0.0.1 (loopback) via decoded wrappers whose literal is globally
    # routable (NAT64 well-known /96, IPv4-compatible) — blocked by the embedded
    # IPv4 — plus 6to4, blocked as a force-denied block.
    assert _host_is_public("64:ff9b::7f00:1") is False  # NAT64 well-known /96
    assert _host_is_public("::7f00:1") is False  # IPv4-compatible ::a.b.c.d
    assert _host_is_public("2002:7f00:1::") is False  # 6to4 2002::/16
    # IPv4-mapped form of the same loopback (already blocked pre-fix).
    assert _host_is_public("::ffff:7f00:1") is False
    # Cloud instance-metadata endpoint (169.254.169.254) via NAT64.
    assert _host_is_public("64:ff9b::a9fe:a9fe") is False
    # RFC1918 host (10.66.0.2) via NAT64 and via IPv4-compatible.
    assert _host_is_public("64:ff9b::a42:2") is False
    assert _host_is_public("::a42:2") is False


def test_host_is_public_rejects_6to4_block_even_wrapping_public(monkeypatch):
    # 6to4 (2002::/16) is not globally reachable per IANA and current CPython
    # classifies the whole block non-global. The guard must never be more
    # permissive than the stdlib, so decoding the embedded IPv4 must not
    # re-permit it: 2002::/16 is default-denied as a whole block — including a
    # 6to4 wrapper of a *public* IPv4 — regardless of interpreter. (RFC 7526
    # deprecated only the 6to4 anycast relay path, not the 2002::/16 prefix.)
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    assert _host_is_public("2002:808:808::") is False  # 6to4 of public 8.8.8.8
    assert _host_is_public("2002:7f00:1::") is False  # 6to4 of loopback
    # Explicit opt-in for a legacy 6to4 deployment still works via the allowlist.
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "2002::/16")
    assert _host_is_public("2002:808:808::") is True


def test_host_is_public_rejects_rfc8215_local_use_nat64_slash48(monkeypatch):
    # The RFC 8215 local-use NAT64 prefix 64:ff9b:1::/48 is technology-agnostic:
    # the embedded-IPv4 position is not guaranteed (RFC 8215 §5) and IANA marks
    # the whole /48 not-globally-reachable, so the entire block is default-denied
    # rather than decoded. A naive "low 32 bits" decode is a bypass — a proper
    # RFC 6052 §2.2 /48 encoding carries the IPv4 in bits 48-63 + 72-87, so the
    # low 32 bits are an attacker-chosen suffix.
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    # RFC 6052 /48 encoding of 127.0.0.1 with a public-looking low-32 suffix
    # (8.8.8.8) — a low-32 decode would wrongly read it as public and pass.
    assert _host_is_public("64:ff9b:1:7f00:0:100:808:808") is False
    # RFC 6052 /48 encoding of a *public* address (8.8.8.8) is also denied:
    # a not-globally-reachable local-use block is off by default policy.
    assert _host_is_public("64:ff9b:1:808:8:800::") is False
    # The "low 32 bits" style literals in the same /48 are blocked too.
    assert _host_is_public("64:ff9b:1::7f00:1") is False
    assert _host_is_public("64:ff9b:1::808:808") is False


def test_rfc8215_slash48_permitted_only_via_allowlist(monkeypatch):
    # An operator running a genuine local NAT64 in 64:ff9b:1::/48 can still opt
    # in explicitly; nothing outside the configured block is affected.
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "64:ff9b:1::/48")
    assert _host_is_public("64:ff9b:1::808:808") is True
    assert _host_is_public("64:ff9b:1:7f00:0:100:808:808") is True
    # The well-known /96 (a different prefix) is unaffected by the /48 allowlist.
    assert _host_is_public("64:ff9b::7f00:1") is False


def test_host_is_public_allows_genuine_and_nat64_wrapped_public(monkeypatch):
    # The fix must not create false positives: genuine global IPv6 and a NAT64
    # wrapper of a *public* IPv4 both resolve to a globally routable address and
    # must still be fetchable.
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    assert _host_is_public("2606:4700:4700::1111") is True  # Cloudflare DNS
    assert _host_is_public("2001:4860:4860::8888") is True  # Google DNS
    assert _host_is_public("64:ff9b::808:808") is True  # NAT64 of 8.8.8.8


def test_unwrap_embedded_ipv4_isolated():
    from ipaddress import ip_address

    # Fixed-position wrappers decode to the embedded (non-global) IPv4.
    assert _unwrap_embedded_ipv4(ip_address("64:ff9b::7f00:1")) == ip_address(
        "127.0.0.1"
    )
    assert _unwrap_embedded_ipv4(ip_address("::7f00:1")) == ip_address("127.0.0.1")
    assert _unwrap_embedded_ipv4(ip_address("::ffff:7f00:1")) == ip_address("127.0.0.1")
    # NAT64 of a public IPv4 decodes to that public IPv4 (still global).
    assert _unwrap_embedded_ipv4(ip_address("64:ff9b::808:808")) == ip_address(
        "8.8.8.8"
    )
    # ``::`` and ``::1`` are not IPv4 wrappers — left untouched.
    assert _unwrap_embedded_ipv4(ip_address("::1")) == ip_address("::1")
    assert _unwrap_embedded_ipv4(ip_address("::")) == ip_address("::")
    # 6to4 (2002::/16) and the RFC 8215 local-use /48 are deliberately NOT decoded
    # here — they are force-denied as whole blocks at the guard instead, so the
    # helper returns them unchanged (decoding either could loosen the verdict).
    sixtofour = ip_address("2002:7f00:1::")
    assert _unwrap_embedded_ipv4(sixtofour) == sixtofour
    slash48 = ip_address("64:ff9b:1:7f00:0:100:808:808")
    assert _unwrap_embedded_ipv4(slash48) == slash48
    # Genuine global IPv6 and plain IPv4 pass through unchanged.
    genuine = ip_address("2606:4700:4700::1111")
    assert _unwrap_embedded_ipv4(genuine) == genuine
    assert _unwrap_embedded_ipv4(ip_address("8.8.8.8")) == ip_address("8.8.8.8")


def test_allowlist_permits_configured_non_public(monkeypatch):
    monkeypatch.setenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", "100.64.0.0/10")
    assert _host_is_public("100.64.0.1") is True
    # An address outside the allowlist is still rejected.
    assert _host_is_public("10.0.0.1") is False


def test_pin_socket_refuses_internal_host(monkeypatch):
    # The connection pins to the same resolution that authorised the host, so
    # an internal target cannot be reached even if it would resolve.
    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    with pytest.raises(OSError, match="non-public host"):
        _pin_socket("127.0.0.1", 80, None, None)


def test_redirect_to_non_public_host_blocked(monkeypatch):
    import email.message

    monkeypatch.delenv("NATIVE_MD_IMAGE_ALLOWED_NON_PUBLIC_CIDRS", raising=False)
    handler = md_parser._GuardedRedirectHandler()
    req = md_parser.urllib.request.Request("http://example.com/a")
    with pytest.raises(md_parser.urllib.error.HTTPError):
        handler.redirect_request(
            req, None, 302, "Found", email.message.Message(), "http://10.0.0.1/evil"
        )


def test_guarded_https_handler_open_does_not_raise_attribute_error(monkeypatch):
    # Regression: ``https_open`` must forward only the kwargs the stdlib handler
    # actually has on this Python version. On 3.12 ``HTTPSHandler`` has no
    # ``_check_hostname`` attribute, so a hard reference raised AttributeError
    # and every real download silently fell back to an external link.
    handler = md_parser._GuardedHTTPSHandler()
    captured: dict = {}

    def _fake_do_open(conn_cls, req, **kwargs):
        captured["conn_cls"] = conn_cls
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(handler, "do_open", _fake_do_open)
    req = md_parser.urllib.request.Request("https://example.com/x.png")
    assert handler.https_open(req) == "ok"
    assert captured["conn_cls"] is md_parser._GuardedHTTPSConnection
    assert "context" in captured["kwargs"]


def test_build_guarded_opener_constructs_with_guarded_handlers():
    opener = md_parser._build_guarded_opener()
    handler_types = {type(h) for h in opener.handlers}
    assert md_parser._GuardedHTTPSHandler in handler_types
    assert md_parser._GuardedHTTPHandler in handler_types


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


def test_textpack_oversized_image_is_skipped(tmp_path: Path, monkeypatch):
    # A bundled asset larger than NATIVE_MD_IMAGE_MAX_BYTES is skipped before it
    # is read into memory (the per-file cap, separate from the zip-bomb total).
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_BYTES", "16")
    pack = tmp_path / "big.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", "# N\n\n![x](assets/big.png)\n")
        zf.writestr("assets/big.png", _PNG_BYTES)  # well over 16 bytes
    p = _make_parser()
    _, warnings, meta = p.extract(
        pack, parsed_dir=tmp_path, asset_dir=tmp_path, base_name="big"
    )
    assert meta["md_drawings"] == {}
    assert meta["md_assets"] == {}
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


def test_remote_image_dropped_when_download_disabled(monkeypatch):
    # With downloading explicitly disabled an external image is dropped entirely
    # — no drawing, no asset — so a doc whose only image is an external link
    # produces no drawings.json. (Downloading is ON by default.)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "false")
    p = _make_parser()
    _, warnings, meta = p._extract_text(
        "# H\n\n![x](http://host/y.png)\n", bundle_root=None
    )
    assert meta["md_drawings"] == {}
    assert meta["md_assets"] == {}
    assert warnings.get("images_external_dropped") == 1


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
