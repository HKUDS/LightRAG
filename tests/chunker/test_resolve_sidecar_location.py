"""The plugin-facing sidecar resolver: what resolves, and what fails loudly.

Whether a path is absolute depends on the platform, so most cases run under
both path flavors by swapping the resolver's ``Path`` for ``PurePosixPath`` or
``PureWindowsPath``; the outcome is then the same on any CI host. One native
case per run exercises the real ``Path`` class end to end.
"""

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

import lightrag.chunker as chunker
from lightrag.chunker import registry
from lightrag.chunker.registry import resolve_sidecar_location
from lightrag.utils_pipeline import SIDECAR_LOCATION_UNKNOWN

pytestmark = pytest.mark.offline

FOREIGN = "not an absolute local path on this platform"


@pytest.fixture(params=[PurePosixPath, PureWindowsPath], ids=["posix", "windows"])
def flavor(request, monkeypatch):
    monkeypatch.setattr(registry, "Path", request.param)
    return request.param


@pytest.fixture
def posix(monkeypatch):
    monkeypatch.setattr(registry, "Path", PurePosixPath)


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(registry, "Path", PureWindowsPath)


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


@pytest.mark.parametrize("location", [None, "", SIDECAR_LOCATION_UNKNOWN])
def test_resolver_returns_none_without_a_known_sidecar(flavor, location):
    assert resolve_sidecar_location(location) is None


POSIX_LOCATIONS = [
    "file:///nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
    "file://localhost/nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
]


@pytest.mark.parametrize("location", POSIX_LOCATIONS)
def test_posix_resolves_a_rooted_uri_path(posix, location):
    assert resolve_sidecar_location(location) == PurePosixPath(
        "/nonexistent-lightrag-sidecar/in put/report.pdf.parsed"
    )


@pytest.mark.parametrize("location", POSIX_LOCATIONS)
def test_windows_rejects_a_drive_less_uri_path(windows, location):
    # Read on Windows, /srv/x would take its drive from the current directory.
    with pytest.raises(ValueError, match=FOREIGN):
        resolve_sidecar_location(location)


DRIVE_LOCATIONS = [
    # The standard form: the slash before the drive only separates authority
    # from path, and is dropped.
    ("file:///C:/tmp/report.parsed/", "C:/tmp/report.parsed"),
    (
        "file://localhost/D:/inputs/in%20put/report.pdf.parsed/",
        "D:/inputs/in put/report.pdf.parsed",
    ),
    ("file:///Z:/", "Z:/"),
    # The legacy form sidecar_uri_for() writes on Windows: the whole
    # percent-encoded drive path lands in the netloc.
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


@pytest.mark.parametrize(("location", "expected"), DRIVE_LOCATIONS)
def test_posix_rejects_both_drive_uri_forms(posix, location, expected):
    # Read on POSIX, C:/tmp would be a path relative to the current directory.
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
    with pytest.raises(ValueError, match="unsupported sidecar location"):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        # Empty: would resolve to the current working directory.
        "file:",
        "file://",
        "file:///",
        # Relative: would resolve against the current working directory.
        "file:relative/sidecar/",
        # Drive-relative: C:tmp depends on the drive's current directory.
        "file:///C:tmp/report.parsed/",
        # UNC through an empty or localhost authority: a network share, not a
        # local directory, once Windows reads the leading double separator.
        "file:////fileserver/share/report.parsed/",
        "file://localhost//fileserver/share/report.parsed/",
        "file:///%5C%5Cfileserver/share/report.parsed/",
    ],
)
def test_resolver_rejects_a_local_uri_path_that_names_no_local_directory(
    flavor, location
):
    with pytest.raises(ValueError, match="unsupported sidecar location"):
        resolve_sidecar_location(location)


@pytest.mark.parametrize(
    "location",
    [
        "s3://bucket/workspace/report.pdf.parsed/",
        "http://example.com/report.pdf.parsed/",
        "/srv/inputs/__parsed__/report.pdf.parsed",
        "__parsed__/report.pdf.parsed",
        "C:\\inputs\\__parsed__\\report.pdf.parsed",
        "file://fileserver/share/report.pdf.parsed/",
    ],
)
def test_resolver_rejects_locations_it_cannot_resolve_locally(flavor, location):
    # Where the internal helper would return None and let a plugin skip the
    # sidecar silently, the public entry point fails loudly instead.
    with pytest.raises(ValueError, match="unsupported sidecar location"):
        resolve_sidecar_location(location)
