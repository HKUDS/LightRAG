"""Unit tests for the sidecar URI helpers and document-path canonicalization
introduced when ``full_docs`` collapsed its four path fields to
``file_path`` + ``sidecar_location``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

import pytest

from lightrag.utils_pipeline import (
    SIDECAR_LOCATION_UNKNOWN,
    _windows_sidecar_path,
    normalize_document_file_path,
    parsed_artifact_dir_for,
    resolve_sidecar_uri,
    sidecar_assets_dir_for_uri,
    sidecar_blocks_path,
    sidecar_modality_path,
    sidecar_uri_for,
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Windows path semantics"
)
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path semantics")


@pytest.mark.offline
def test_normalize_strips_hint_and_directory():
    assert normalize_document_file_path("abc.[native-iet].docx") == "abc.docx"
    assert normalize_document_file_path("/tmp/sub/abc.docx") == "abc.docx"
    assert normalize_document_file_path("abc.docx") == "abc.docx"


@pytest.mark.offline
def test_normalize_idempotent():
    once = normalize_document_file_path("/tmp/abc.[native].docx")
    twice = normalize_document_file_path(once)
    assert once == twice == "abc.docx"


@pytest.mark.offline
@pytest.mark.parametrize(
    "value",
    ["", None, "no-file-path", "unknown_source", "  "],
)
def test_normalize_maps_placeholders_to_unknown(value):
    assert normalize_document_file_path(value) == "unknown_source"


@pytest.mark.offline
def test_sidecar_uri_round_trip_ascii(tmp_path):
    sidecar_dir = tmp_path / "abc.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert uri.endswith("/")
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_round_trip_unicode_and_spaces(tmp_path):
    sidecar_dir = tmp_path / "中文 报告.docx.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)
    assert uri.startswith("file://")
    assert " " not in uri  # spaces are percent-encoded
    resolved = resolve_sidecar_uri(uri)
    assert resolved == sidecar_dir.resolve()


@pytest.mark.offline
def test_sidecar_uri_has_empty_authority(tmp_path):
    sidecar_dir = tmp_path / "demo.parsed"
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)

    assert urlsplit(uri).netloc == ""
    assert urlsplit(uri).path.startswith("/")


# --- POSIX: the writer and reader must keep their pre-fix behavior ----------


@pytest.mark.offline
@posix_only
@pytest.mark.parametrize(
    "name",
    [
        "demo.pdf.parsed",
        "a b.parsed",
        "中文 é.parsed",
        "lit%20x.parsed",
        "q#?;@&=+$,.parsed",
    ],
)
def test_posix_writer_is_byte_identical_to_the_previous_form(name):
    # Existing POSIX rows must keep their meaning without a migration, so the
    # ``as_uri()`` writer has to reproduce the old hand-built string exactly.
    p = Path("/srv/data/__parsed__") / name
    previous = f"file://{quote(str(p.resolve()), safe='/')}/"
    assert sidecar_uri_for(p) == previous


@pytest.mark.offline
@posix_only
@pytest.mark.parametrize("path", ["/a:b/demo.parsed", "/C:/demo.parsed"])
def test_posix_drive_lookalike_first_component_stays_absolute(path):
    # A POSIX directory named like a drive is an ordinary absolute path; the
    # reader must not strip its leading slash and return a relative path.
    resolved = resolve_sidecar_uri(sidecar_uri_for(path))
    assert resolved == Path(path)
    assert resolved.is_absolute()


@pytest.mark.offline
@posix_only
def test_posix_reader_decodes_literal_escape_exactly_once():
    uri = "file:///data/report%2520.parsed/"
    assert resolve_sidecar_uri(uri) == Path("/data/report%20.parsed")


@pytest.mark.offline
@posix_only
def test_posix_reader_does_not_raise_on_a_double_slash_path():
    # ``url2pathname`` treats a leading ``//`` as an authority and raises on
    # Python 3.14; the reader must keep returning the path as before.
    assert resolve_sidecar_uri("file:////srv/demo.parsed/") == Path("//srv/demo.parsed")


@pytest.mark.offline
@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="needs a filesystem that accepts non-UTF-8 names",
)
def test_posix_non_utf8_name_round_trips(tmp_path):
    # ``as_uri()`` encodes undecodable bytes instead of raising as the old
    # writer did, so the reader has to decode in the filesystem encoding.
    sidecar_dir = tmp_path / os.fsdecode(b"bad\xff.parsed")
    sidecar_dir.mkdir()
    assert resolve_sidecar_uri(sidecar_uri_for(sidecar_dir)) == sidecar_dir.resolve()


# --- Windows authority handling: pure string logic, runs everywhere ---------


@pytest.mark.offline
@pytest.mark.parametrize(
    ("netloc", "expected"),
    [
        (
            "C%3A%5CUsers%5Cqhx%5Cdata%5C__parsed__%5Cdemo.pdf.parsed",
            r"C:\Users\qhx\data\__parsed__\demo.pdf.parsed",
        ),
        # Decoded exactly once: "%2520" is the literal text "%20", never a space.
        ("C%3A%5Cdata%5Cdemo%2520dir.parsed", r"C:\data\demo%20dir.parsed"),
    ],
)
def test_windows_legacy_authority_form_is_read(netloc, expected):
    # The previous writer put the whole drive path in the authority, leaving
    # "/" as the URI path.
    assert _windows_sidecar_path(netloc, "/") == expected


@pytest.mark.offline
@pytest.mark.parametrize(
    ("netloc", "path"),
    [
        ("server", "/share/demo.parsed/"),  # canonical UNC form
        ("%5C%5Cserver%5Cshare%5Cdemo.parsed", "/"),  # legacy UNC form
        ("C%3A%5Cdata%5Cdemo.parsed", "/more/"),  # not what the old writer made
    ],
)
def test_windows_unc_and_unknown_authorities_resolve_to_none(netloc, path):
    assert _windows_sidecar_path(netloc, path) is None


# --- Windows: native path semantics, run by the Windows CI job --------------


@pytest.mark.offline
@windows_only
def test_windows_reader_recovers_drive_letter():
    uri = "file:///C:/Users/qhx/data/__parsed__/demo.pdf.parsed/"
    assert resolve_sidecar_uri(uri) == Path(
        r"C:\Users\qhx\data\__parsed__\demo.pdf.parsed"
    )


@pytest.mark.offline
@windows_only
def test_windows_reader_reads_legacy_authority_form():
    uri = "file://C%3A%5CUsers%5Cqhx%5Cdata%5C__parsed__%5Cdemo.pdf.parsed/"
    assert resolve_sidecar_uri(uri) == Path(
        r"C:\Users\qhx\data\__parsed__\demo.pdf.parsed"
    )


@pytest.mark.offline
@windows_only
def test_windows_reader_decodes_literal_escape_exactly_once():
    uri = "file:///C:/data/report%2520.parsed/"
    assert resolve_sidecar_uri(uri) == Path(r"C:\data\report%20.parsed")


@pytest.mark.offline
@windows_only
def test_windows_reader_rejects_unc_host():
    assert resolve_sidecar_uri("file://server/share/demo.parsed/") is None


@pytest.mark.offline
@windows_only
def test_windows_reader_does_not_raise_on_a_misplaced_colon():
    # CPython < 3.14 raises OSError for this path; the reader must not.
    resolved = resolve_sidecar_uri("file:///C:/data/a:b.parsed/")
    assert resolved is None or isinstance(resolved, Path)


@pytest.mark.offline
def test_resolve_sidecar_uri_tolerates_missing_trailing_slash(tmp_path):
    sidecar_dir = tmp_path / "demo.parsed"
    sidecar_dir.mkdir()
    uri_no_slash = sidecar_uri_for(sidecar_dir).rstrip("/")
    assert resolve_sidecar_uri(uri_no_slash) == sidecar_dir.resolve()


@pytest.mark.offline
@pytest.mark.parametrize(
    "uri",
    [None, "", SIDECAR_LOCATION_UNKNOWN, "s3://bucket/path/"],
)
def test_resolve_sidecar_uri_returns_none_for_unsupported(uri):
    assert resolve_sidecar_uri(uri) is None


@pytest.mark.offline
@pytest.mark.parametrize(
    "name",
    [
        "with space.docx.parsed",
        "中文 报告.docx.parsed",
        "literal%20lookalike.parsed",
        "literal%2Flookalike.parsed",
        "literal%25lookalike.parsed",
    ],
)
def test_sidecar_uri_round_trips_awkward_names(tmp_path, name):
    """A name that *looks* percent-encoded must survive the round trip.

    The reader has to decode exactly once: a second decode would rewrite the
    literal ``%20`` in ``literal%20lookalike`` as a space and silently point at
    a different artifact directory.
    """
    sidecar_dir = tmp_path / name
    sidecar_dir.mkdir()
    uri = sidecar_uri_for(sidecar_dir)

    assert uri.endswith("/")
    # The bug this pins: a path must never end up in the URI authority, which
    # is what a hand-built ``file://`` + ``quote`` did with a drive letter.
    assert urlsplit(uri).netloc == ""
    assert resolve_sidecar_uri(uri) == sidecar_dir.resolve()


@pytest.mark.offline
@pytest.mark.parametrize(
    "name",
    [
        "C:/data/demo.parsed",
        "C:/data/a b.parsed",
        "C:/data/中文.parsed",
        "C:/data/report%20.parsed",
        "C:/data/a%2Fb.parsed",
        "C:/data/a%25b.parsed",
    ],
)
@windows_only
def test_windows_drive_paths_round_trip(name):
    """A drive-letter path must produce ``file:///C:/...``, not ``file://C%3A...``.

    Asserted from the URI shape rather than from ``as_uri`` so the contract is
    stated once: the authority is empty, the path is the path, and reading it
    back yields the same directory.
    """
    sidecar_dir = Path(name)
    uri = sidecar_uri_for(sidecar_dir)

    assert uri.startswith("file:///")
    assert urlsplit(uri).netloc == ""
    assert resolve_sidecar_uri(uri) == sidecar_dir


@pytest.mark.offline
def test_sidecar_blocks_path_locates_jsonl(tmp_path):
    sidecar_dir = tmp_path / "demo.docx.parsed"
    sidecar_dir.mkdir()
    blocks = sidecar_dir / "demo.blocks.jsonl"
    blocks.write_text("", encoding="utf-8")
    uri = sidecar_uri_for(sidecar_dir)

    assert sidecar_blocks_path(uri) == str(blocks)
    assert sidecar_modality_path(uri, "tables") == str(sidecar_dir / "demo.tables.json")
    assert sidecar_assets_dir_for_uri(uri) == Path(sidecar_dir / "demo.blocks.assets")


@pytest.mark.offline
def test_sidecar_blocks_path_returns_none_when_missing(tmp_path):
    empty = tmp_path / "empty.parsed"
    empty.mkdir()
    uri = sidecar_uri_for(empty)
    assert sidecar_blocks_path(uri) is None
    assert sidecar_modality_path(uri, "drawings") is None
    assert sidecar_assets_dir_for_uri(uri) is None


@pytest.mark.offline
def test_parsed_artifact_dir_for_uses_input_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("demo.docx")
    assert result == tmp_path / "__parsed__" / "demo.docx.parsed"


@pytest.mark.offline
def test_parsed_artifact_dir_for_strips_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    result = parsed_artifact_dir_for("abc.[native-iet].docx")
    assert result == tmp_path / "__parsed__" / "abc.docx.parsed"
