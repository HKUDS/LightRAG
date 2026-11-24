"""The plugin-facing sidecar resolver: what resolves, and what fails loudly.

The resolver delegates decoding to the pipeline's own reader
(``resolve_sidecar_uri``) and only adds refusals. Both branch on ``os.name``,
so each platform is exercised on any CI host by switching ``os.name`` in both
modules and giving the reader that platform's ``Path`` flavor and, for
Windows, the real Windows ``url2pathname`` (``nturl2path``). Native cases
exercise the real classes end to end.
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

import lightrag.chunker as chunker
from lightrag import utils_pipeline
from lightrag.chunker import registry
from lightrag.chunker.registry import resolve_sidecar_location
from lightrag.utils_pipeline import (
    SIDECAR_LOCATION_UNKNOWN,
    resolve_sidecar_uri,
    sidecar_uri_for,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # Deprecated in 3.14.
    try:
        import nturl2path
    except ImportError:  # pragma: no cover - removal is scheduled for 3.19
        nturl2path = None

pytestmark = pytest.mark.offline

FOREIGN = "not an absolute local path on this platform"
UNSUPPORTED = "unsupported sidecar location"


def _emulate(monkeypatch, platform):
    name, flavor = platform
    if name == "nt" and nturl2path is None:
        pytest.skip("nturl2path is unavailable on this Python")
    fake_os = SimpleNamespace(name=name)
    monkeypatch.setattr(registry, "os", fake_os)
    monkeypatch.setattr(utils_pipeline, "os", fake_os)
    monkeypatch.setattr(utils_pipeline, "Path", flavor)
    if name == "nt":
        monkeypatch.setattr(utils_pipeline, "url2pathname", nturl2path.url2pathname)


POSIX = ("posix", PurePosixPath)
WINDOWS = ("nt", PureWindowsPath)


@pytest.fixture(params=[POSIX, WINDOWS], ids=["posix", "windows"])
def flavor(request, monkeypatch):
    _emulate(monkeypatch, request.param)


@pytest.fixture
def posix(monkeypatch):
    _emulate(monkeypatch, POSIX)


@pytest.fixture
def windows(monkeypatch):
    _emulate(monkeypatch, WINDOWS)


def test_resolver_is_exported_from_the_chunker_package():
    assert chunker.resolve_sidecar_location is resolve_sidecar_location
    assert "resolve_sidecar_location" in chunker.__all__


def test_registry_import_stays_cheap():
    # The resolver defers its pipeline import; the registry must not pull it in.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lightrag.chunker.registry; assert 'lightrag.utils_pipeline' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_resolver_returns_a_native_absolute_path_end_to_end():
    location = (
        "file:///C:/nonexistent-lightrag-sidecar/report.pdf.parsed/"
        if os.name == "nt"
        else "file:///nonexistent-lightrag-sidecar/report.pdf.parsed/"
    )
    resolved = resolve_sidecar_location(location)
    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert not resolved.exists()  # Existence is the plugin's business.


NATIVE_SIDECAR_NAMES = [
    "report.pdf.parsed",
    "in put/report.pdf.parsed",
    "caf\u00e9/\u62a5\u544a.pdf.parsed",
    # A literal percent sign must survive: the path is decoded exactly once.
    "100%20done/report.pdf.parsed",
    "hash#and?query/report.pdf.parsed",
]


@pytest.mark.parametrize("name", NATIVE_SIDECAR_NAMES)
def test_resolver_round_trips_what_the_pipeline_writes(tmp_path, name):
    # Whatever sidecar_uri_for() persists, the plugin-facing resolver must read
    # back as the same directory the pipeline's own reader resolves.
    directory = tmp_path / name
    location = sidecar_uri_for(directory)
    resolved = resolve_sidecar_location(location)
    assert resolved == directory.resolve()
    assert resolved == resolve_sidecar_uri(location)


@pytest.mark.skipif(os.name == "nt", reason="POSIX names may hold non-UTF-8 bytes")
def test_resolver_keeps_a_non_utf8_posix_name(tmp_path):
    # Path.as_uri() quotes the filesystem-encoded bytes; a UTF-8 'replace'
    # decode would turn the 0xFF byte into U+FFFD, another directory.
    directory = tmp_path / os.fsdecode(b"bad\xffname.parsed")
    location = sidecar_uri_for(directory)
    assert "%FF" in location
    resolved = resolve_sidecar_location(location)
    assert os.fsencode(resolved).endswith(b"/bad\xffname.parsed")
    assert resolved == resolve_sidecar_uri(location)


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only file names")
@pytest.mark.parametrize(
    "path",
    [
        # A backslash and a colon are ordinary POSIX filename characters, in
        # any position; neither may be read as Windows syntax.
        "/tmp/sidecar-store\\",
        "/C:/inputs/__parsed__/doc.parsed",
        "/a:/b",
        "/C:tmp/doc.parsed",
        "/\\\\fileserver/share/doc.parsed",
        "/\\@/doc.parsed",
        "/\\",
    ],
)
def test_resolver_keeps_windows_lookalike_posix_names(path):
    location = sidecar_uri_for(path)
    resolved = resolve_sidecar_location(location)
    assert resolved == Path(os.path.realpath(path))
    assert resolved == resolve_sidecar_uri(location)


# None and "" share the one ``not location`` check.
@pytest.mark.parametrize("location", [None, SIDECAR_LOCATION_UNKNOWN])
def test_resolver_returns_none_without_a_known_sidecar(flavor, location):
    assert resolve_sidecar_location(location) is None


POSIX_LOCATIONS = [
    "file:///nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
    "file://localhost/nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
    # A host name matches in any case, as in the pipeline's own reader.
    "file://LocalHost/nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
]


@pytest.mark.parametrize("location", POSIX_LOCATIONS)
def test_posix_resolves_a_rooted_uri_path(posix, location):
    assert resolve_sidecar_location(location) == PurePosixPath(
        "/nonexistent-lightrag-sidecar/in put/report.pdf.parsed"
    )


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        # What a POSIX writer emits for names that look like Windows syntax:
        # ':' and '\' are percent-encoded, and the reader decodes them back.
        ("file:///C%3A/inputs/doc.parsed/", "/C:/inputs/doc.parsed"),
        ("file:///tmp/sidecar-store%5C/", "/tmp/sidecar-store\\"),
        ("file:///%5C%5Cfileserver/share/", "/\\\\fileserver/share"),
        # A leading double slash is still a local POSIX path.
        ("file:////srv/share/", "//srv/share"),
        ("file:///", "/"),
    ],
)
def test_posix_reads_windows_lookalike_paths_as_posix_names(posix, location, expected):
    assert resolve_sidecar_location(location) == PurePosixPath(expected)


@pytest.mark.parametrize("location", POSIX_LOCATIONS)
def test_windows_rejects_a_drive_less_uri_path(windows, location):
    # Read on Windows, /srv/x would take its drive from the current directory.
    with pytest.raises(ValueError, match=FOREIGN):
        resolve_sidecar_location(location)


DRIVE_LOCATIONS = [
    # The standard form sidecar_uri_for() writes on Windows: the slash before
    # the drive only separates authority from path.
    ("file:///C:/tmp/report.parsed/", "C:/tmp/report.parsed"),
    (
        "file://localhost/D:/inputs/in%20put/report.pdf.parsed/",
        "D:/inputs/in put/report.pdf.parsed",
    ),
    ("file:///Z:/", "Z:/"),
    # The legacy form older Windows rows carry, from before sidecar_uri_for()
    # kept the drive in the path: the whole percent-encoded drive path lands
    # in the netloc.
    ("file://C%3A%5Ctmp%5Cexample.parsed/", "C:\\tmp\\example.parsed"),
    (
        "file://D%3A%5Cinputs%5Cin%20put%5Creport.pdf.parsed/",
        "D:\\inputs\\in put\\report.pdf.parsed",
    ),
    ("file://Z%3A%5Cexample.parsed/", "Z:\\example.parsed"),
]


@pytest.mark.parametrize(("location", "expected"), DRIVE_LOCATIONS)
def test_windows_resolves_both_drive_uri_forms(windows, location, expected):
    resolved = resolve_sidecar_location(location)
    assert resolved == PureWindowsPath(expected)
    assert resolved.drive == expected[:2]
    assert resolved.is_absolute()


def test_windows_drops_a_trailing_backslash_in_the_uri_path(windows):
    resolved = resolve_sidecar_location("file:///C:/tmp/report.parsed%5C/")
    assert resolved == PureWindowsPath("C:/tmp/report.parsed")


# POSIX refuses through two branches only, the literal drive in the path and the
# legacy drive in the netloc, so one location of each form suffices here.
@pytest.mark.parametrize(
    ("location", "expected"), [DRIVE_LOCATIONS[0], DRIVE_LOCATIONS[3]]
)
def test_posix_rejects_both_drive_uri_forms(posix, location, expected):
    # Written on Windows: read on POSIX, C:/tmp would be another directory.
    with pytest.raises(ValueError, match=FOREIGN):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        # Decoded netloc is not unambiguously a Windows drive path: forward
        # slash after the colon, or a longer prefix before it.
        "file://C%3A%2Ftmp%2Fexample.parsed/",
        "file://CD%3A%5Ctmp%5Cexample.parsed/",
        "file://%5Ctmp%5Cexample.parsed/",
        "file://fileserver/C%3A%5Cshare%5Creport.pdf.parsed/",
        # Drive in the netloc AND a path of its own: not a form LightRAG wrote.
        "file://C%3A%5Ctmp/sub/",
    ],
)
def test_resolver_rejects_a_netloc_that_only_looks_like_a_windows_path(
    flavor, location
):
    with pytest.raises(ValueError, match=UNSUPPORTED):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        # Empty: would resolve to the current working directory.
        "file:",
        "file://",
        # Relative: would resolve against the current working directory.
        "file:relative/sidecar/",
    ],
)
def test_resolver_rejects_an_empty_or_relative_uri_path(flavor, location):
    with pytest.raises(ValueError, match=FOREIGN):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        # Drive-relative: C:tmp depends on the drive's current directory.
        "file:///C:tmp/report.parsed/",
        "file:///C|tmp/report.parsed/",
        # UNC through an empty or localhost authority: a network share, not a
        # local directory, once Windows reads the leading double separator.
        "file:////fileserver/share/report.parsed/",
        "file://localhost//fileserver/share/report.parsed/",
        "file:///%5C%5Cfileserver/share/report.parsed/",
    ],
)
def test_windows_rejects_a_uri_path_that_names_no_local_directory(windows, location):
    with pytest.raises(ValueError, match=UNSUPPORTED):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        "file:///C:/tmp/report.parsed/",
        "file:///C:%5Ctmp/report.parsed/",
        "file:///C:%2Ftmp/report.parsed/",
        "file:///C|%2Ftmp/report.parsed/",
    ],
)
def test_windows_reader_still_reads_a_rooted_drive_path(windows, location):
    assert resolve_sidecar_uri(location) == PureWindowsPath("C:/tmp/report.parsed")


@pytest.mark.parametrize(
    "location",
    [
        "s3://bucket/workspace/report.pdf.parsed/",
        "/srv/inputs/__parsed__/report.pdf.parsed",
        "C:\\inputs\\__parsed__\\report.pdf.parsed",
        "file://fileserver/share/report.pdf.parsed/",
    ],
)
def test_resolver_rejects_locations_it_cannot_resolve_locally(flavor, location):
    # Where the internal helper would return None and let a plugin skip the
    # sidecar silently, the public entry point fails loudly instead.
    with pytest.raises(ValueError, match=UNSUPPORTED):
        resolve_sidecar_location(location)
