"""The plugin-facing sidecar resolver: what resolves, and what fails loudly."""

import subprocess
import sys
from pathlib import Path

import pytest

import lightrag.chunker as chunker
from lightrag.chunker.registry import resolve_sidecar_location
from lightrag.utils_pipeline import SIDECAR_LOCATION_UNKNOWN, resolve_sidecar_uri

pytestmark = pytest.mark.offline


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


@pytest.mark.parametrize("location", [None, "", SIDECAR_LOCATION_UNKNOWN])
def test_resolver_returns_none_without_a_known_sidecar(location):
    assert resolve_sidecar_location(location) is None


@pytest.mark.parametrize(
    "location",
    [
        "file:///nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
        "file://localhost/nonexistent-lightrag-sidecar/in%20put/report.pdf.parsed/",
    ],
)
def test_resolver_returns_local_path_without_requiring_it_to_exist(location):
    resolved = resolve_sidecar_location(location)
    assert resolved == Path("/nonexistent-lightrag-sidecar/in put/report.pdf.parsed")
    assert resolved == resolve_sidecar_uri(location)
    assert not resolved.exists()


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        # sidecar_uri_for() percent-encodes a Windows path whole, so the drive
        # letter and every separator land in the netloc, not in the path.
        ("file://C%3A%5Ctmp%5Cexample.parsed/", "C:\\tmp\\example.parsed"),
        (
            "file://D%3A%5Cinputs%5Cin%20put%5Creport.pdf.parsed/",
            "D:\\inputs\\in put\\report.pdf.parsed",
        ),
        ("file://Z%3A%5Cexample.parsed/", "Z:\\example.parsed"),
    ],
)
def test_resolver_accepts_the_legacy_windows_uri_form(location, expected):
    # Compared against the same literal on every platform: on Linux this is a
    # PosixPath whose single name contains backslashes, which is still the
    # value a Windows deployment resolves.
    assert resolve_sidecar_location(location) == Path(expected)


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
def test_resolver_rejects_a_netloc_that_only_looks_like_a_windows_path(location):
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
def test_resolver_rejects_locations_it_cannot_resolve_locally(location):
    # Where the internal helper would return None and let a plugin skip the
    # sidecar silently, the public entry point fails loudly instead.
    with pytest.raises(ValueError, match="unsupported sidecar location"):
        resolve_sidecar_location(location)
