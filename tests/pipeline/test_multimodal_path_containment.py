"""Sidecar image paths are contained to their own document directory.

GHSA-8rgj-chc2-6chv. The ``path`` field of ``<doc>.drawings.json`` reaches
``Path.read_bytes`` and then the VLM. Before this guard the resolver returned
an absolute path unchanged and joined a relative one to ``sidecar_dir``
without ``resolve()`` or ``is_relative_to()``, so ``/etc/hostname`` and
``../../../tenantB/...`` both resolved and were read.

The producer-side half of that advisory is already fixed: since df29b1a6 the
native DOCX extractor carries an external relationship target in ``src``, and
the sink reads only ``path``. This file guards the *sink*, which is what makes
the property hold for sidecars this version did not write — an older version's,
an external engine's, or one restored from backup. ``ReuseParser`` re-uses an
existing sidecar on resume/retry instead of re-parsing, so those values are
still reachable.

The resolver returns ``(path, outcome)``: RESOLVED (a real contained file),
REFUSED (containment / malformed / non-string — never read), or MISSING
(contained but no such file). REFUSED and MISSING must stay distinct so a
legacy absolute path (a file that DOES exist, outside the doc dir) is not
mis-reported to the operator as "not found".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.pipeline import _SidecarPathOutcome, _resolve_sidecar_image_path

# CI runs only ``-m offline``; these use tmp files only, no live services.
pytestmark = pytest.mark.offline


@pytest.fixture
def sidecar_dir(tmp_path: Path) -> Path:
    """A document's parsed dir, with one legitimately-placed asset."""
    d = tmp_path / "ws" / "tenantA" / "__parsed__" / "report.docx.parsed"
    assets = d / "report.blocks.assets"
    assets.mkdir(parents=True)
    (assets / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\nLEGIT")
    return d


# --- fix proof --------------------------------------------------------------


def test_absolute_path_outside_the_sidecar_is_refused(sidecar_dir, tmp_path):
    """Pre-fix this returned the path unchanged: ``is_absolute()`` skipped the
    join entirely and the bare ``exists()`` accepted it."""
    victim = tmp_path / "victim_home" / "id_rsa_screenshot.png"
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"\x89PNG\r\n\x1a\nSECRET")

    path, outcome = _resolve_sidecar_image_path(str(victim), sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_dotdot_escape_is_refused(sidecar_dir, tmp_path):
    """The cross-tenant read from the advisory, laid out as DocumentManager
    lays workspaces out. Pre-fix ``sidecar_dir / '../../..'`` was accepted
    because ``/`` does not fold ``..`` and ``exists()`` follows it."""
    tenant_b = (
        tmp_path
        / "ws"
        / "tenantB"
        / "__parsed__"
        / "victim.docx.parsed"
        / "victim.blocks.assets"
    )
    tenant_b.mkdir(parents=True)
    (tenant_b / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\nTENANT-B-CONFIDENTIAL")

    traversal = (
        "../../../tenantB/__parsed__/victim.docx.parsed/victim.blocks.assets/image1.png"
    )
    path, outcome = _resolve_sidecar_image_path(traversal, sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_absolute_path_to_a_non_image_is_refused(sidecar_dir, tmp_path):
    """``/etc/hostname`` was accepted by the resolver and only rejected 29
    lines later by the extension gate — which is why the Tier-1 existence
    oracle was not limited to images. It must not resolve at all.

    Uses a file this test creates rather than a real system path: on a host
    where ``/etc/hostname`` is absent, ``exists()`` refuses it for the wrong
    reason and the test would pass against the pre-fix resolver too.
    """
    target = tmp_path / "hostname"
    target.write_text("victim-host\n")
    path, outcome = _resolve_sidecar_image_path(str(target), sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


def test_non_string_path_value_is_refused_not_raised(sidecar_dir):
    """A sidecar carrying ``"path": 123`` (or a list) must be refused, not
    raise. The caller's ``item.get("path") or ...`` chain passes any truthy
    JSON type through; ``sidecar_dir / 123`` raises TypeError, which — before
    the guard — escaped the resolver, hit the fail-fast wait loop, and marked
    the whole document FAILED instead of skipping one image."""
    for bad in (123, ["x"], {"a": 1}, 3.5, True):
        path, outcome = _resolve_sidecar_image_path(bad, sidecar_dir)
        assert path is None
        assert outcome is _SidecarPathOutcome.REFUSED


# --- the legitimate case still works ---------------------------------------


def test_asset_inside_the_document_directory_resolves(sidecar_dir):
    resolved, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/image1.png", sidecar_dir
    )
    assert outcome is _SidecarPathOutcome.RESOLVED
    assert resolved is not None
    assert resolved.read_bytes() == b"\x89PNG\r\n\x1a\nLEGIT"


def test_resolution_survives_a_symlinked_prefix(tmp_path):
    """Both sides must be resolved, not just the candidate.

    On macOS ``/tmp`` is a symlink to ``/private/tmp``; comparing a resolved
    candidate against an unresolved ``sidecar_dir`` would refuse every
    legitimate asset there. This pins that the check is symmetric.
    """
    real = tmp_path / "real" / "report.docx.parsed"
    (real / "report.blocks.assets").mkdir(parents=True)
    (real / "report.blocks.assets" / "image1.png").write_bytes(b"PNGDATA")

    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "real", target_is_directory=True)

    resolved, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/image1.png", link / "report.docx.parsed"
    )
    assert outcome is _SidecarPathOutcome.RESOLVED
    assert resolved is not None
    assert resolved.read_bytes() == b"PNGDATA"


# --- refused vs missing must stay distinct ---------------------------------


def test_missing_and_empty_values_are_refused(sidecar_dir):
    # None / "" are not usable references at all → REFUSED.
    assert _resolve_sidecar_image_path(None, sidecar_dir) == (
        None,
        _SidecarPathOutcome.REFUSED,
    )
    assert _resolve_sidecar_image_path("", sidecar_dir) == (
        None,
        _SidecarPathOutcome.REFUSED,
    )


def test_a_contained_but_absent_file_is_missing_not_refused(sidecar_dir):
    """A relative reference that resolves INSIDE the dir but names no file is
    MISSING — the honest "image file not found". This is what must stay apart
    from a containment refusal so the operator message is accurate."""
    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/absent.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.MISSING


def test_a_directory_inside_the_sidecar_is_not_a_file(sidecar_dir):
    path, outcome = _resolve_sidecar_image_path("report.blocks.assets", sidecar_dir)
    assert path is None
    assert outcome is _SidecarPathOutcome.MISSING


def test_symlink_escaping_the_sidecar_is_refused(sidecar_dir, tmp_path):
    """resolve() follows symlinks, so a link planted inside the asset dir
    must be judged by where it points, not by where it sits."""
    victim = tmp_path / "outside_secret.png"
    victim.write_bytes(b"\x89PNG\r\n\x1a\nSECRET")
    (sidecar_dir / "report.blocks.assets" / "escape.png").symlink_to(victim)

    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/escape.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


# --- a malformed reference is skipped, never a failed document -------------
#
# The containment check introduced resolve(), which unlike the bare exists()
# it replaced can RAISE on a malformed path. An escaping exception here is
# not a refusal: it propagates out of the drawing task, hits the fail-fast
# wait loop in analyze_multimodal, and marks the entire document FAILED —
# turning a skipped image into a failed ingest.


def test_embedded_nul_is_skipped_not_raised(sidecar_dir):
    # Path.resolve() raises ValueError("embedded null character") here;
    # the shared helper catches it and returns None → REFUSED.
    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/a\x00b.png", sidecar_dir
    )
    assert path is None
    assert outcome is _SidecarPathOutcome.REFUSED


# The exact OUTCOME label for a symlink loop is platform-dependent, so these
# two assert the invariant that matters (no path is handed back, so nothing is
# read) rather than pinning the label. On macOS ``Path.resolve()`` raises
# ("Symlink loop from ..."), which the shared guard turns into REFUSED; on
# Linux CPython's non-strict ``realpath`` does not raise — it returns the
# partially-resolved path — so containment passes and the loop surfaces as
# MISSING at the ``is_file()`` check. Both refuse, and containment still holds
# in the escaping case: a loop that leaves the directory first resolves to the
# outside prefix and is REFUSED on every platform (covered by
# ``test_symlink_escaping_the_sidecar_is_refused``).


def test_symlink_loop_is_skipped_not_raised(sidecar_dir):
    assets = sidecar_dir / "report.blocks.assets"
    (assets / "loopA.png").symlink_to(assets / "loopB.png")
    (assets / "loopB.png").symlink_to(assets / "loopA.png")

    path, outcome = _resolve_sidecar_image_path(
        "report.blocks.assets/loopA.png", sidecar_dir
    )
    assert path is None
    assert outcome is not _SidecarPathOutcome.RESOLVED


def test_symlink_loop_in_the_sidecar_dir_itself_is_skipped(tmp_path):
    # The guard must cover sidecar_dir.resolve() too, not just the candidate:
    # a reused sidecar can be reached through a broken directory prefix.
    a = tmp_path / "dirA"
    b = tmp_path / "dirB"
    a.symlink_to(b, target_is_directory=True)
    b.symlink_to(a, target_is_directory=True)

    path, outcome = _resolve_sidecar_image_path("report.blocks.assets/x.png", a)
    assert path is None
    assert outcome is not _SidecarPathOutcome.RESOLVED
