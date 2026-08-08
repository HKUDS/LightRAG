"""Per-document image budgets for the native markdown parser.

GHSA-25c3-j78v-83qx defects 1 and 3: ``NATIVE_MD_IMAGE_MAX_BYTES`` bounds one
image, and nothing bounded the document. A 4 KB ``.textpack`` produced 800
outbound GETs and held 800 MiB of image bytes, because the resolver kept every
image's bytes resident until extraction returned and the src-keyed memo was
defeated by a ``?u=<i>`` query parameter.
"""

from __future__ import annotations

import base64
import zipfile
from pathlib import Path

import pytest

from lightrag.parser.markdown import parser as md_parser

_PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _png(index: int, size: int = 1024) -> bytes:
    """A distinct 'PNG' of ``size`` bytes: distinct bytes, distinct sha256."""
    body = str(index).encode().rjust(16, b"0")
    return _PNG_HEADER + body + b"\x00" * (size - len(_PNG_HEADER) - len(body))


class _FakeHeaders:
    def __init__(self, content_type: str) -> None:
        self._ct = content_type

    def get_content_type(self) -> str:
        return self._ct


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.headers = _FakeHeaders("image/png")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_network(monkeypatch):
    """Serve a distinct PNG per URL and count opener.open() calls."""
    state = {"opens": 0, "size": 1024, "fail": False}

    class _Opener:
        def open(self, req, timeout=None):
            state["opens"] += 1
            if state["fail"]:
                raise OSError("connection refused")
            url = req.full_url if hasattr(req, "full_url") else str(req)
            index = url.rsplit("=", 1)[-1]
            return _FakeResponse(
                _png(int(index) if index.isdigit() else 0, state["size"])
            )

    monkeypatch.setattr(md_parser, "_host_is_public", lambda host: True)
    monkeypatch.setattr(md_parser, "_build_guarded_opener", lambda: _Opener())
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")
    return state


def _markdown(count: int) -> str:
    return "\n\n".join(
        f"![i{i}](http://host.example/img.png?u={i})" for i in range(count)
    )


def _extract(md: str, *, bundle_root: Path | None = None):
    parser = md_parser.NativeMarkdownParser()
    _, warnings, meta = parser._extract_text(md, bundle_root=bundle_root)
    kinds = [d["kind"] for d in meta["md_drawings"].values()]
    return kinds, warnings, meta


# --------------------------------------------------------------------------
# Byte budget (defect 1)
# --------------------------------------------------------------------------


def test_total_byte_budget_degrades_later_images_to_external(monkeypatch, fake_network):
    # Room for exactly two 1 KiB images; the other three must not be retained.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(2 * 1024))
    kinds, warnings, meta = _extract(_markdown(5))

    assert kinds.count("local") == 2
    assert kinds.count("external") == 3
    assert warnings.get("images_byte_budget_exceeded") == 3
    assert sum(len(a["data"]) for a in meta["md_assets"].values()) == 2 * 1024


def test_two_urls_with_identical_bytes_are_both_charged(monkeypatch, fake_network):
    # md_assets deduplicates by sha256, but the resolver holds one bytes object
    # per src and BOTH are resident — which is precisely the ?u=0..N shape an
    # attacker builds. Charging the deduplicated set would under-count it.
    monkeypatch.setattr(
        md_parser, "_build_guarded_opener", lambda: _SameBytesOpener(fake_network)
    )
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(1024 + 512))
    kinds, warnings, meta = _extract(_markdown(2))

    assert kinds == ["local", "external"]
    assert warnings.get("images_byte_budget_exceeded") == 1
    # One asset entry (identical bytes), but two charges.
    assert len(meta["md_assets"]) == 1


class _SameBytesOpener:
    def __init__(self, state):
        self._state = state

    def open(self, req, timeout=None):
        self._state["opens"] += 1
        return _FakeResponse(_png(0, self._state["size"]))


def test_byte_budget_charges_data_urls(monkeypatch, fake_network):
    # Distinct payloads: the same data URL repeated would hit the src memo and
    # be charged once, which is not what this test is about.
    md = "\n\n".join(
        "![d{}](data:image/png;base64,{})".format(
            i, base64.b64encode(_png(i, 1024)).decode()
        )
        for i in range(3)
    )
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(2 * 1024))
    kinds, warnings, _ = _extract(md)

    # No URL to fall back to, so an over-budget data URL is dropped outright…
    assert kinds == ["local", "local"]
    assert warnings.get("images_byte_budget_exceeded") == 1
    # …and it must NOT be lumped in with the generic skip counter.
    assert "images_skipped" not in warnings


def test_byte_budget_charges_bundle_files(monkeypatch, tmp_path, fake_network):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for i in range(3):
        (bundle / f"a{i}.png").write_bytes(_png(i, 1024))
    md = "\n\n".join(f"![b{i}](a{i}.png)" for i in range(3))
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(2 * 1024))

    kinds, warnings, _ = _extract(md, bundle_root=bundle)

    assert kinds == ["local", "local"]
    assert warnings.get("images_byte_budget_exceeded") == 1
    assert "images_skipped" not in warnings


def test_partial_remaining_budget_reports_a_budget_stop_not_a_download_failure(
    monkeypatch, fake_network
):
    # The remaining budget is not a whole multiple of the image size, so the
    # last fetch is capped mid-body. That must still read as a budget stop:
    # a plain ValueError here would be relabelled images_download_failed and
    # the dedicated counter would never appear.
    fake_network["size"] = 1024
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(1024 + 500))
    kinds, warnings, _ = _extract(_markdown(2))

    assert kinds == ["local", "external"]
    assert warnings.get("images_byte_budget_exceeded") == 1
    assert "images_download_failed" not in warnings


def test_budget_exceptions_are_not_rewrapped_as_urlerror():
    # urllib's do_open turns OSError into URLError, which would erase the
    # distinction between a budget stop and a network failure.
    assert not issubclass(md_parser._ImageBudgetExceeded, OSError)
    assert not issubclass(md_parser._ImageByteBudgetExceeded, OSError)
    assert not issubclass(md_parser._ImageRequestBudgetExceeded, OSError)


# --------------------------------------------------------------------------
# Request budget (defect 3)
# --------------------------------------------------------------------------


def test_request_budget_caps_outbound_attempts(monkeypatch, fake_network):
    # ?u=<i> defeats the src-keyed memo, which is the whole trick: ten distinct
    # srcs used to mean ten fetches with nothing counting them.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "3")
    kinds, warnings, _ = _extract(_markdown(10))

    assert fake_network["opens"] == 3
    assert kinds.count("local") == 3
    assert kinds.count("external") == 7
    assert warnings.get("images_request_budget_exceeded") == 7


def test_failed_downloads_consume_the_request_budget(monkeypatch, fake_network):
    # Charging only successes would make a few thousand unresolvable URLs free.
    fake_network["fail"] = True
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "2")
    kinds, warnings, _ = _extract(_markdown(5))

    assert fake_network["opens"] == 2
    assert kinds == ["external"] * 5
    assert warnings.get("images_download_failed") == 2
    assert warnings.get("images_request_budget_exceeded") == 3


def test_repeated_same_url_is_charged_once(monkeypatch, fake_network):
    md = "\n\n".join("![x](http://host.example/img.png?u=1)" for _ in range(5))
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(1024))
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "1")
    kinds, warnings, meta = _extract(md)

    # The src memo short-circuits before any accounting, so a repeated
    # reference cannot exhaust either budget.
    assert kinds == ["local"] * 5
    assert fake_network["opens"] == 1
    assert len(meta["md_assets"]) == 1
    assert "images_byte_budget_exceeded" not in warnings
    assert "images_request_budget_exceeded" not in warnings


def test_budgets_are_independent(monkeypatch, fake_network):
    # A generous byte budget must not mask an exhausted request budget…
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(100 * 1024 * 1024))
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "1")
    _, warnings, _ = _extract(_markdown(3))
    assert warnings.get("images_request_budget_exceeded") == 2
    assert "images_byte_budget_exceeded" not in warnings


def test_byte_budget_fires_without_the_request_budget(monkeypatch, fake_network):
    # …and vice versa.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(1024))
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "1000")
    _, warnings, _ = _extract(_markdown(3))
    assert warnings.get("images_byte_budget_exceeded") == 2
    assert "images_request_budget_exceeded" not in warnings


# --------------------------------------------------------------------------
# REQUIRED semantics, env parsing, and the shipped defaults
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", "1024"),
        ("NATIVE_MD_IMAGE_MAX_REQUESTS", "1"),
    ],
)
def test_download_required_raises_when_a_budget_is_exhausted(
    monkeypatch, fake_network, env, value
):
    # REQUIRED's documented contract is "a download error fails the document".
    # An image the operator demanded be embedded, and which was not embedded,
    # is exactly that.
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_REQUIRED", "true")
    monkeypatch.setenv(env, value)
    with pytest.raises(md_parser._ImageBudgetExceeded):
        _extract(_markdown(3))


@pytest.mark.parametrize(
    ("env", "value"),
    [
        ("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", "1024"),
        ("NATIVE_MD_IMAGE_MAX_REQUESTS", "1"),
    ],
)
def test_without_required_a_budget_stop_only_degrades(
    monkeypatch, fake_network, env, value
):
    monkeypatch.setenv(env, value)
    kinds, _, _ = _extract(_markdown(3))
    assert kinds.count("external") == 2  # degraded, document intact


@pytest.mark.parametrize("raw", ["abc", "0", "-5", ""])
def test_invalid_budget_env_falls_back_to_the_default(monkeypatch, fake_network, raw):
    # 0 and negatives are NOT "unlimited": an operator typing 0 into a ceiling
    # means strictest, and inverting that would turn a typo into the very
    # exhaustion the budget prevents.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", raw)
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", raw)
    kinds, warnings, _ = _extract(_markdown(3))

    assert kinds == ["local"] * 3
    assert "images_byte_budget_exceeded" not in warnings
    assert "images_request_budget_exceeded" not in warnings


def test_a_legitimate_document_is_unaffected_by_the_shipped_defaults(
    monkeypatch, tmp_path, fake_network
):
    """Defaults canary: a mixed, realistic document must not trip any budget."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    parts = []
    for i in range(8):
        (bundle / f"local{i}.png").write_bytes(_png(i, 64 * 1024))
        parts.append(f"![l{i}](local{i}.png)")
    for i in range(8):
        parts.append(
            "![d{}](data:image/png;base64,{})".format(
                i, base64.b64encode(_png(100 + i, 32 * 1024)).decode()
            )
        )
    fake_network["size"] = 64 * 1024
    for i in range(8):
        parts.append(f"![r{i}](http://host.example/img.png?u={i})")

    kinds, warnings, _ = _extract("\n\n".join(parts), bundle_root=bundle)

    assert kinds == ["local"] * 24
    assert "images_byte_budget_exceeded" not in warnings
    assert "images_request_budget_exceeded" not in warnings
    assert "images_skipped" not in warnings


def test_shipped_defaults_are_derived_from_the_parse_concurrency():
    from lightrag.constants import DEFAULT_MAX_PARALLEL_PARSE_NATIVE

    worst_case = (
        md_parser.DEFAULT_NATIVE_MD_IMAGE_MAX_TOTAL_BYTES
        * DEFAULT_MAX_PARALLEL_PARSE_NATIVE
    )
    # Pins the reasoning, not the number: the point of the default is that all
    # parse workers together cannot hold an unreasonable amount of image bytes.
    assert worst_case <= 512 * 1024 * 1024


# --------------------------------------------------------------------------
# Cache interaction
# --------------------------------------------------------------------------


def _textpack(tmp_path: Path, md: str) -> Path:
    pack = tmp_path / "doc.textpack"
    with zipfile.ZipFile(pack, "w") as zf:
        zf.writestr("text.markdown", md)
    return pack


def test_a_budget_rejected_download_is_not_cached(monkeypatch, tmp_path, fake_network):
    """Rejected bytes must not be served back for free on the next parse.

    raw_cache.put() runs only after _local() accepts, so the ordering is
    validate -> charge -> cache, never cache-then-charge.
    """
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_TOTAL_BYTES", str(1024))
    pack = _textpack(tmp_path, _markdown(2))
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()

    parser = md_parser.NativeMarkdownParser()
    _, warnings, _ = parser.extract(
        pack, parsed_dir=parsed, asset_dir=assets, base_name="doc"
    )
    assert warnings.get("images_byte_budget_exceeded") == 1

    raw_dir = md_parser.raw_dir_for_parsed_dir(
        parsed, suffix=md_parser.NATIVE_RAW_DIR_SUFFIX
    )
    cached = (
        [p for p in raw_dir.iterdir() if p.suffix != ".json"]
        if raw_dir.exists()
        else []
    )
    # Exactly the one image that was accepted.
    assert len(cached) == 1


def test_cache_hits_do_not_consume_the_request_budget(monkeypatch, tmp_path):
    """A hit issues no request, so it costs no request budget — but it is
    still charged bytes, since those bytes are just as resident and on a
    re-parse the byte budget is the only thing bounding memory."""
    payload = (_png(7, 1024), "png")
    calls = {"n": 0}

    def _fake_download(self, src):
        calls["n"] += 1
        return payload

    monkeypatch.setattr(md_parser._MarkdownImageResolver, "_download", _fake_download)
    monkeypatch.setenv("NATIVE_MD_IMAGE_DOWNLOAD_ENABLED", "true")

    pack = _textpack(tmp_path, "![x](http://host.example/img.png?u=1)")
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    parser = md_parser.NativeMarkdownParser()

    _, warnings, _ = parser.extract(
        pack, parsed_dir=parsed, asset_dir=assets, base_name="doc"
    )
    assert calls["n"] == 1

    # Re-parse with a request budget of zero-after-one: a hit must still work.
    monkeypatch.setenv("NATIVE_MD_IMAGE_MAX_REQUESTS", "1")
    _, warnings, meta = parser.extract(
        pack, parsed_dir=parsed, asset_dir=assets, base_name="doc"
    )
    assert calls["n"] == 1  # no second download
    assert warnings.get("images_cache_hit") == 1
    assert [d["kind"] for d in meta["md_drawings"].values()] == ["local"]
