"""Tests for shared helpers in ``lightrag/parser/external/``.

These cover the pure functions reused across engine integrations:

- ``compute_size_and_hash`` — single-read (size, hash) pair
- ``clear_dir_contents`` — empty a directory while keeping it
- ``raw_dir_for_parsed_dir`` — suffix-bound raw dir naming
- ``safe_extract_zip`` — refuses path traversal and absolute paths
- ``env_bool`` / ``env_int`` / ``env_json`` — env parsing
- ``Manifest`` round-trip via ``write_manifest`` / ``load_manifest``
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from lightrag.parser.external import (
    Manifest,
    ManifestFile,
    clear_dir_contents,
    compute_size_and_hash,
    env_bool,
    env_int,
    env_json,
    load_manifest,
    raw_dir_for_parsed_dir,
    safe_extract_zip,
    write_manifest,
)


pytestmark = pytest.mark.offline

# ---------------------------------------------------------------------------
# compute_size_and_hash
# ---------------------------------------------------------------------------


def test_compute_size_and_hash_stable(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    payload = b"hello-external-parser" * 1024
    p.write_bytes(payload)

    size_a, hash_a = compute_size_and_hash(p)
    size_b, hash_b = compute_size_and_hash(p)

    assert size_a == len(payload) == size_b
    assert hash_a == hash_b
    assert hash_a.startswith("sha256:") and len(hash_a) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# clear_dir_contents
# ---------------------------------------------------------------------------


def test_clear_dir_contents_keeps_dir_and_removes_children(tmp_path: Path) -> None:
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.txt").write_text("hi")
    sub = d / "nested"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"x" * 10)

    clear_dir_contents(d)

    assert d.is_dir()
    assert list(d.iterdir()) == []


def test_clear_dir_contents_noop_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    clear_dir_contents(missing)
    assert not missing.exists()


# ---------------------------------------------------------------------------
# raw_dir_for_parsed_dir
# ---------------------------------------------------------------------------


def test_raw_dir_for_parsed_dir_with_suffix(tmp_path: Path) -> None:
    parsed = tmp_path / "demo.pdf.parsed"
    raw = raw_dir_for_parsed_dir(parsed, suffix=".docling_raw")
    assert raw == tmp_path / "demo.pdf.docling_raw"


def test_raw_dir_for_parsed_dir_without_parsed_suffix(tmp_path: Path) -> None:
    parsed = tmp_path / "other_dir"
    raw = raw_dir_for_parsed_dir(parsed, suffix=".docling_raw")
    assert raw == tmp_path / "other_dir.docling_raw"


def test_raw_dir_for_parsed_dir_rejects_bad_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        raw_dir_for_parsed_dir(tmp_path / "x.parsed", suffix="docling_raw")


# ---------------------------------------------------------------------------
# safe_extract_zip
# ---------------------------------------------------------------------------


def _make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def test_safe_extract_zip_extracts_flat_bundle(tmp_path: Path) -> None:
    payload = _make_zip(
        {
            "demo.json": b'{"schema_name": "DoclingDocument"}',
            "demo.md": b"# demo",
            "artifacts/image_000000.png": b"\x89PNG fake",
        }
    )
    dest = tmp_path / "raw"
    names = safe_extract_zip(payload, dest)

    assert (dest / "demo.json").read_bytes().startswith(b'{"schema_name"')
    assert (dest / "demo.md").read_text(encoding="utf-8") == "# demo"
    assert (dest / "artifacts" / "image_000000.png").is_file()
    assert sorted(names) == sorted(
        ["demo.json", "demo.md", "artifacts/image_000000.png"]
    )


@pytest.mark.parametrize(
    "name", ["..notes.md", "...report.md", "..assets/figure.png", "nested/..notes.md"]
)
def test_safe_extract_zip_allows_dot_prefixed_names(tmp_path: Path, name: str) -> None:
    payload = _make_zip({name: b"Document content"})

    assert safe_extract_zip(payload, tmp_path) == [name]
    assert (tmp_path / name).read_bytes() == b"Document content"


# One case per branch of the unsafe-path check: the bare parent, a parent
# prefix, a normalized escape and an absolute path.
@pytest.mark.parametrize(
    "name", ["..", "../outside.md", "nested/../../outside.md", "/outside.md"]
)
def test_safe_extract_zip_rejects_unsafe_bundle_before_writing(
    tmp_path: Path, name: str
) -> None:
    destination = tmp_path / "bundle"
    payload = _make_zip({"safe.md": b"safe", name: b"unsafe"})

    with pytest.raises(RuntimeError, match="unsafe path"):
        safe_extract_zip(payload, destination)

    assert not (destination / "safe.md").exists()
    assert not (tmp_path / "outside.md").exists()


# ---------------------------------------------------------------------------
# env coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "yes", "ON"])
def test_env_bool_truthy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("X", raw)
    assert env_bool("X", False) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
def test_env_bool_falsy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("X", raw)
    assert env_bool("X", True) is False


def test_env_bool_falls_back_on_unrecognized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "maybe")
    assert env_bool("X", True) is True
    assert env_bool("X", False) is False


def test_env_int_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "not-an-int")
    assert env_int("X", 7) == 7


def test_env_json_returns_default_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", "{bad json")
    assert env_json("X", {"origin": "LEFTBOTTOM"}) == {"origin": "LEFTBOTTOM"}


def test_env_json_parses_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X", '{"a": 1, "b": [2, 3]}')
    assert env_json("X", None) == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    raw = tmp_path / "demo.docling_raw"
    raw.mkdir()
    crit = ManifestFile(path="demo.json", size=42, sha256="sha256:" + "a" * 64)
    other = ManifestFile(path="demo.md", size=10)
    manifest = Manifest(
        engine="docling",
        source_content_hash="sha256:" + "b" * 64,
        source_size_bytes=100,
        source_filename_at_parse="demo.pdf",
        critical_file=crit,
        files=[other],
        total_size_bytes=52,
        task_id="task-xyz",
        endpoint_signature="http://l4ai:5001",
        engine_version="1.18.0",
        options_signature="sha256:" + "c" * 64,
        downloaded_at="2026-05-18T00:00:00Z",
        extras={"to_formats": ["json", "md"]},
    )
    write_manifest(raw, manifest)

    payload = json.loads((raw / "_manifest.json").read_text(encoding="utf-8"))
    assert payload["engine"] == "docling"
    assert payload["options_signature"] == "sha256:" + "c" * 64
    assert payload["extras"] == {"to_formats": ["json", "md"]}

    loaded = load_manifest(raw, expected_engine="docling")
    assert loaded is not None
    assert loaded.task_id == "task-xyz"
    assert loaded.critical_file.size == 42
    assert loaded.files[0].path == "demo.md"


def test_manifest_load_treats_truncated_json_as_missing(tmp_path: Path) -> None:
    # Truncated JSON must read as a stale bundle instead of escaping and
    # failing the parse on every retry. Invalid UTF-8 is pinned per engine by
    # the ``not_utf8`` cases in ``docling/`` and ``mineru/test_cache.py``.
    raw = tmp_path / "demo.docling_raw"
    raw.mkdir()
    (raw / "_manifest.json").write_bytes(b'{"engine":')
    assert load_manifest(raw, expected_engine="docling") is None


def _load_docling_manifest(raw: Path) -> Manifest | None:
    return load_manifest(raw, expected_engine="docling")


def _load_mineru_manifest(raw: Path):
    from lightrag.parser.external.mineru.manifest import (
        load_manifest as load_mineru_manifest,
    )

    return load_mineru_manifest(raw)


@pytest.mark.parametrize(
    "engine, loader",
    [("docling", _load_docling_manifest), ("mineru", _load_mineru_manifest)],
    ids=["docling", "mineru"],
)
@pytest.mark.parametrize(
    "bad_field",
    [
        # A non-object critical_file has no .get -> AttributeError.
        {"critical_file": "oops"},
        # JSON Infinity in a size field -> OverflowError from int().
        {"source_size_bytes": float("inf")},
    ],
    ids=["non_dict_critical_file", "infinite_size"],
)
def test_manifest_load_treats_malformed_field_as_missing(
    tmp_path: Path, engine: str, loader, bad_field: dict
) -> None:
    # A malformed field must read as a stale bundle instead of escaping:
    # the parse template checks the bundle before clearing raw_dir, so an
    # escaped error would fail every retry.
    raw = tmp_path / f"demo.{engine}_raw"
    raw.mkdir()
    payload = {"version": "1.0", "engine": engine, "files": [], **bad_field}
    (raw / "_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert loader(raw) is None


# ---------------------------------------------------------------------------
# stream_capped_get -- Accept-Encoding: identity + Content-Encoding rejection
#
# Uses genuine httpx.AsyncByteStream fixtures (not content=...-preloaded
# responses) so these tests prove a rejected stream is never consumed.
# ---------------------------------------------------------------------------


class _GenuineStream(httpx.AsyncByteStream):
    """A streamed byte source with nothing preloaded, unlike
    ``httpx.Response(..., content=...)``."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


class _NeverIterateStream(httpx.AsyncByteStream):
    """Fails the test immediately if ever iterated, proving a rejected
    response's body is never touched."""

    async def __aiter__(self):
        pytest.fail("body stream was iterated despite a rejected Content-Encoding")
        yield b""  # pragma: no cover -- unreachable; makes this an async generator

    async def aclose(self) -> None:
        pass


async def test_stream_capped_get_sends_accept_encoding_identity():
    """Accept-Encoding: identity must override httpx's own default
    ('gzip, deflate, zstd'), not merely be added alongside it."""
    from lightrag.parser.external._common import stream_capped_get

    seen_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(request.headers)
        return httpx.Response(200, stream=_GenuineStream([b"ok"]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        await stream_capped_get(
            client, "https://x/y", max_bytes=None, operation="identity header check"
        )

    assert seen_headers.get("accept-encoding") == "identity"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        # The canonical value, and a padded mixed-case one that goes red if
        # either the strip or the case fold is dropped from the comparison.
        {"content-encoding": "identity"},
        {"content-encoding": "  Identity  "},
    ],
    ids=["absent", "identity", "identity-padded-mixed-case"],
)
async def test_stream_capped_get_accepts_absent_or_identity_encoding(headers):
    from lightrag.parser.external._common import stream_capped_get

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, stream=_GenuineStream([b"data ", b"chunk"]), headers=headers
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        resp, body = await stream_capped_get(
            client, "https://x/y", max_bytes=None, operation="accept check"
        )

    assert body == b"data chunk"


@pytest.mark.parametrize(
    "content_encoding",
    # One "is not identity" comparison: a plain encoding, the identity-prefix
    # lookalike, and the empty value (present but empty is not absent).
    ["gzip", "identity, gzip", ""],
    ids=["gzip", "multiple-identity-gzip", "malformed-empty"],
)
async def test_stream_capped_get_rejects_non_identity_encoding(content_encoding):
    """Also proves the rejected stream is never iterated (via
    _NeverIterateStream) and that the error names the operation but not
    the URL, for every encoding/malformed-value case in one pass."""
    from lightrag.parser.external._common import stream_capped_get

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=_NeverIterateStream(),
            headers={"content-encoding": content_encoding},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError) as excinfo:
            await stream_capped_get(
                client,
                "https://x/result.zip?token=secret",
                max_bytes=None,
                operation="my download op",
            )

    message = str(excinfo.value)
    assert "my download op" in message
    assert "https://x/result.zip?token=secret" not in message
    assert "secret" not in message


async def test_stream_capped_get_cap_counts_across_chunks():
    """The cap applies to the running total, not to each chunk: every chunk
    here is far below the limit. The per-client oversized-response bounds
    tests deliver one preloaded chunk, so they stay green if the total is
    reset per chunk; this is the only multi-chunk input."""
    from lightrag.parser.external._common import stream_capped_get

    chunks = [b"x" * 1000 for _ in range(50)]  # 50,000 bytes total

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_GenuineStream(chunks))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError) as excinfo:
            await stream_capped_get(
                client, "https://x/y", max_bytes=10_000, operation="cap check"
            )

    assert "10000 bytes" in str(excinfo.value)
