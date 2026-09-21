"""``lightrag.private_file`` — a file no other user can read, from creation.

Kept free of storage imports on purpose: this is the module a Windows CI job
exercises, and it must not need faiss, nano, a database or an embedding model
to say whether a file was created private.

The property is not "ends up private". It is "was never anything else".
Tightening a file after creating it leaves a window, and on Windows that
window is not closed by tightening at all: access is checked when a handle is
OPENED, so a process that got a handle during the window keeps reading through
it afterwards, including rows written later. That is why the descriptor goes
IN to ``CreateFileW`` here rather than being applied after, and why sharing is
disabled for the duration of the copy.

Platform split: the POSIX mode assertions describe nothing on Windows
(``st_mode`` carries no ACL there), and the native Windows checks cannot run
anywhere else. The DACL predicate in between is pure and runs everywhere,
which matters because it is the part that encodes what "private" means.
"""

from __future__ import annotations

import ctypes
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from lightrag.private_file import (
    PrivateFileError,
    assert_dacl_grants_only,
    open_private_file,
    windows_create_exclusive,
)

pytestmark = pytest.mark.offline

SID = "S-1-5-21-1111111111-2222222222-3333333333-1001"


# ---------------------------------------------------------------------------
# Portable behaviour
# ---------------------------------------------------------------------------


def test_it_writes_what_it_was_given(tmp_path):
    target = tmp_path / "copy.bin"
    with open_private_file(str(target)) as destination:
        destination.write(b"sensitive rows")
    assert target.read_bytes() == b"sensitive rows"


def test_it_refuses_to_create_over_an_existing_file(tmp_path):
    """The exclusivity callers rely on to keep one recovery's files apart."""
    target = tmp_path / "copy.bin"
    target.write_bytes(b"already here")
    with pytest.raises(FileExistsError):
        with open_private_file(str(target)) as destination:
            destination.write(b"clobbered")
    assert target.read_bytes() == b"already here"


def test_a_failure_while_writing_leaves_no_partial_copy(tmp_path):
    """A half-written backup that looks whole is worse than none: the caller
    drops the original next, trusting this file to hold it."""
    target = tmp_path / "copy.bin"
    with pytest.raises(RuntimeError, match="source vanished"):
        with open_private_file(str(target)) as destination:
            destination.write(b"first half")
            raise RuntimeError("source vanished")
    assert not target.exists()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="injects os.fstat, which only the POSIX branch calls; the Windows "
    "equivalent is the DACL verification injected in its own test",
)
def test_a_privacy_failure_leaves_no_file_at_all(tmp_path, monkeypatch):
    """Nothing is written before the check, so a refusal costs no data."""
    target = tmp_path / "copy.bin"

    real_fstat = os.fstat

    def _wrong_mode(fd):
        real = real_fstat(fd)
        return os.stat_result((0o100644, *tuple(real)[1:]))

    monkeypatch.setattr(os, "fstat", _wrong_mode)
    with pytest.raises(PrivateFileError, match="grants group or other access"):
        with open_private_file(str(target)) as destination:
            destination.write(b"never reached")
    assert not target.exists()


# ---------------------------------------------------------------------------
# Reading CreateFileW's return value — pure, so it runs on every platform
# ---------------------------------------------------------------------------


class _FakeCtypes:
    """Just enough ctypes for the classification, on any platform."""

    c_void_p = ctypes.c_void_p

    def __init__(self, last_error):
        self._last_error = last_error

    @staticmethod
    def byref(value):
        # The real byref needs a ctypes instance; the classification under
        # test never dereferences it, so identity is enough here.
        return value

    def get_last_error(self):
        return self._last_error

    def WinError(self, code):  # noqa: N802 - mirrors the ctypes spelling
        return OSError(code, f"mock win32 error {code}")


class _FakeKernel32:
    def __init__(self, handle):
        self._handle = handle

    def CreateFileW(self, *args):  # noqa: N802 - mirrors the Win32 spelling
        return self._handle


@pytest.mark.parametrize(
    "handle",
    [
        pytest.param(ctypes.c_void_p(-1).value, id="INVALID_HANDLE_VALUE"),
        pytest.param(0, id="null"),
        pytest.param(None, id="None"),
    ],
)
def test_a_failed_creation_is_recognised_as_one(handle, tmp_path):
    """``INVALID_HANDLE_VALUE`` is ``(HANDLE)-1`` in the SDK, but a pointer
    restype hands it back as an unsigned int, so ``handle == -1`` is False on
    64-bit. Comparing against the literal read every failure as a success."""
    existing = tmp_path / "already-there.bin"
    existing.write_bytes(b"somebody else's file")

    with pytest.raises(FileExistsError):
        windows_create_exclusive(
            _FakeKernel32(handle), _FakeCtypes(80), str(existing), object()
        )

    # The whole point: a collision must not take the existing file with it.
    assert existing.read_bytes() == b"somebody else's file"


def test_a_failed_creation_for_another_reason_refuses_without_deleting(tmp_path):
    existing = tmp_path / "already-there.bin"
    existing.write_bytes(b"somebody else's file")

    with pytest.raises(PrivateFileError, match="private security descriptor"):
        windows_create_exclusive(
            _FakeKernel32(ctypes.c_void_p(-1).value),
            _FakeCtypes(5),  # ERROR_ACCESS_DENIED
            str(existing),
            object(),
        )
    assert existing.read_bytes() == b"somebody else's file"


def test_a_real_handle_is_returned_unchanged():
    """A plausible handle value must not be mistaken for a failure."""
    assert (
        windows_create_exclusive(
            _FakeKernel32(0x1A4), _FakeCtypes(0), "C:\\x", object()
        )
        == 0x1A4
    )


# ---------------------------------------------------------------------------
# What "private" means — pure, so it runs on every platform
# ---------------------------------------------------------------------------


def test_the_expected_dacl_is_accepted():
    assert_dacl_grants_only(f"D:P(A;;FA;;;{SID})", SID, "x")


@pytest.mark.parametrize(
    "sddl, reason",
    [
        (f"D:(A;;FA;;;{SID})", "not protected"),
        (f"D:AI(A;;FA;;;{SID})", "not protected"),
        (f"D:P(A;;FA;;;{SID})(A;;FR;;;S-1-1-0)", "a second audience"),
        ("D:P(A;;FA;;;S-1-1-0)", "grants Everyone instead"),
        (f"D:P(D;;FA;;;{SID})", "a deny entry, so nothing is granted"),
        ("D:P", "no entry at all"),
        (f"O:{SID}", "no DACL"),
        ("", "empty"),
    ],
)
def test_a_dacl_that_is_not_exactly_this_sid_is_refused(sddl, reason):
    """Each of these ends with someone other than the owner able to read, or
    with the entries free to change under the file later."""
    with pytest.raises(PrivateFileError):
        assert_dacl_grants_only(sddl, SID, "x")


def test_a_well_known_sid_read_back_as_its_alias_is_still_this_sid():
    """The failure that only appears on some accounts (PR #4025 Windows CI).

    An SDDL SID field has no stable spelling. Windows abbreviates any account
    with a well-known alias on the way out, so a file created by the built-in
    Administrator (RID 500) is WRITTEN as ``S-1-5-21-...-500`` and READ BACK
    as ``LA`` — and comparing the text refuses every file that account
    creates. It passes for an ordinary user, whose SID has no alias, which is
    why local runs were green and the Windows runner (``runneradmin``, RID
    500) failed on every single creation.

    ``sid_matches`` is what the Windows path injects to compare by value
    instead; here it stands in for ``EqualSid``.
    """
    administrator = "S-1-5-21-3699639565-2515463329-295617607-500"
    aliases = {"LA": administrator}

    def by_value(field):
        return aliases.get(field, field) == administrator

    assert_dacl_grants_only("D:P(A;;FA;;;LA)", administrator, "x", sid_matches=by_value)
    # The structural rules still apply through the comparer.
    with pytest.raises(PrivateFileError, match="not protected"):
        assert_dacl_grants_only(
            "D:(A;;FA;;;LA)", administrator, "x", sid_matches=by_value
        )
    # And a DIFFERENT well-known account is still a different audience.
    with pytest.raises(PrivateFileError):
        assert_dacl_grants_only(
            "D:P(A;;FA;;;BA)", administrator, "x", sid_matches=by_value
        )


def test_an_inheritable_entry_for_the_right_sid_is_still_refused():
    """``D:`` without ``P`` takes whatever the parent dictates TODAY. The
    single correct-looking entry says nothing about tomorrow's."""
    with pytest.raises(PrivateFileError, match="not protected"):
        assert_dacl_grants_only(f"D:AI(A;ID;FA;;;{SID})", SID, "x")


# ---------------------------------------------------------------------------
# POSIX: the mode is the mechanism
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="st_mode carries no ACL on Windows")
def test_the_file_is_0600_regardless_of_umask(tmp_path):
    """A mode left to the umask publishes the copy on the common 022."""
    target = tmp_path / "copy.bin"
    saved = os.umask(0o022)
    try:
        with open_private_file(str(target)) as destination:
            destination.write(b"sensitive rows")
    finally:
        os.umask(saved)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="st_mode carries no ACL on Windows")
def test_it_is_private_before_the_first_byte_is_written(tmp_path):
    """Observed from inside the body, which runs before any caller writes."""
    target = tmp_path / "copy.bin"
    with open_private_file(str(target)) as destination:
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert target.stat().st_size == 0
        destination.write(b"sensitive rows")


# ---------------------------------------------------------------------------
# Windows: native, and not reproducible anywhere else
# ---------------------------------------------------------------------------

_WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32",
    reason="exercises Win32 ACL and sharing semantics; no POSIX equivalent",
)


def _icacls_entries(path):
    """The access entries ``icacls`` lists for ``path``, one per element.

    An ACE line is one containing ``:(`` — the same judgement the CI guard
    uses, and one that does not move with the display language the way
    "Successfully processed" does.
    """
    return [
        line.replace(str(path), "", 1).strip()
        for line in _icacls(path).splitlines()
        if ":(" in line
    ]


def _icacls(path):
    """``icacls`` output, with a failed query treated as a failed test.

    Without the exit-code check an empty stdout satisfies every "X is not in
    the listing" assertion below, so a query that never ran would read as
    proof that nothing unwanted is there.
    """
    result = subprocess.run(["icacls", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"icacls exited {result.returncode} for {path}; nothing was verified.\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert result.stdout.strip(), f"icacls produced no output for {path}"
    return result.stdout


@_WINDOWS_ONLY
def test_a_wide_parent_directory_does_not_widen_the_file(tmp_path):
    """The case the descriptor is passed IN to CreateFileW for: a directory
    whose inheritable entries would otherwise land on everything created in
    it."""
    subprocess.run(
        ["icacls", str(tmp_path), "/grant", "*S-1-1-0:(OI)(CI)(F)"],
        capture_output=True,
        text=True,
        check=True,
    )
    target = tmp_path / "copy.bin"
    with open_private_file(str(target)) as destination:
        destination.write(b"sensitive rows")

    listing = _icacls(target)
    assert "(I)" not in listing, f"inherited access survived:\n{listing}"
    assert "Everyone" not in listing and "S-1-1-0" not in listing, listing


@_WINDOWS_ONLY
def test_no_second_handle_can_be_opened_while_the_copy_is_in_flight(tmp_path):
    """``dwShareMode=0``. Even a reader the DACL would admit — this same user,
    in another process — must be refused for the duration, because a handle
    opened now survives every later change to the ACL."""
    target = tmp_path / "copy.bin"
    probe = textwrap.dedent(
        """
        import sys
        try:
            open(sys.argv[1], "rb").read()
        except OSError as error:
            print(f"refused:{error.errno}")
        else:
            print("opened")
        """
    )
    with open_private_file(str(target)) as destination:
        destination.write(b"sensitive rows")
        destination.flush()
        result = subprocess.run(
            [sys.executable, "-c", probe, str(target)],
            capture_output=True,
            text=True,
        )
    assert result.stdout.startswith("refused:"), result.stdout + result.stderr


@_WINDOWS_ONLY
def test_the_owner_can_still_read_and_archive_it_afterwards(tmp_path):
    """Full access, not read-only: these files are the operator's to keep,
    copy elsewhere and eventually delete."""
    target = tmp_path / "copy.bin"
    with open_private_file(str(target)) as destination:
        destination.write(b"sensitive rows")
    assert target.read_bytes() == b"sensitive rows"
    archived = tmp_path / "archived.bin"
    target.replace(archived)
    archived.unlink()


@_WINDOWS_ONLY
def test_the_restriction_persists_after_the_handle_closes(tmp_path):
    """The half sharing mode does NOT cover.

    ``dwShareMode=0`` lapses the moment the handle closes, so from then on
    the DACL is the only thing standing between the file and another
    identity. This checks that it is still there, still protected, and still
    naming exactly one account — read back from the closed file, and set
    against a CONTROL file created normally in the same directory.

    The control is what stops this passing vacuously, and what it shows is
    worth being precise about. A pytest tmp directory carries no inheritable
    entries, so an ordinary file there gets the access token's DEFAULT DACL
    rather than anything inherited — on the runner that is `SYSTEM`,
    `BUILTIN\Administrators` and `OWNER RIGHTS`, all explicit, none marked
    ``(I)``. So the comparison is not "escaped inheritance" but the stronger
    "every principal the platform would have added is absent".

    **What this does not do** is have a second identity attempt the read and
    be refused. Two ways were tried and neither works here. ``runas
    /trustlevel:0x20000`` exits 0 on a GitHub Windows runner and starts
    nothing — no output, no process, no probe verdict after 60s (measured on
    PR #4025); Secondary Logon does not run in that session. And a token from
    ``CreateRestrictedToken`` carries the SAME user SID, which this very ACE
    grants, so an ``AccessCheck`` against it is allowed by construction and
    tests nothing. Creating a real second account is not something a CI
    runner should have to do.

    So the boundary is explicit: this pins that the DACL we wrote is the DACL
    that persists. That a DACL granting one account denies the others is the
    operating system's guarantee, not this module's.
    """
    control = tmp_path / "control.bin"
    control.write_bytes(b"created the ordinary way, in the same directory")
    target = tmp_path / "copy.bin"
    with open_private_file(str(target)) as destination:
        destination.write(b"sensitive rows")
    assert target.exists()

    control_entries = _icacls_entries(control)
    target_entries = _icacls_entries(target)

    def principals(entries):
        return {entry.split(":", 1)[0] for entry in entries}

    # Without this the rest could pass on a platform that grants nothing to
    # anybody, which would say nothing about what this module removed.
    assert len(control_entries) > 1, (
        f"an ordinary file here has {len(control_entries)} entries, so there "
        f"is nothing to compare against: {control_entries}"
    )
    assert len(target_entries) == 1, (
        f"expected exactly one access entry after the handle closed, got "
        f"{target_entries}"
    )
    assert "(I)" not in target_entries[0], (
        f"the target carries inherited access: {target_entries[0]}"
    )
    # Every principal the platform would have granted is gone.
    removed = principals(control_entries) - principals(target_entries)
    assert removed, (
        f"the target grants the same principals an ordinary file does, so "
        f"nothing was restricted: {target_entries} vs {control_entries}"
    )


@_WINDOWS_ONLY
def test_a_refusal_removes_the_file_it_had_already_created(tmp_path, monkeypatch):
    """The verification runs after CreateFileW has produced an object, so its
    refusal has something to clean up — and must, or the next run's
    CREATE_NEW collides with a stub nobody can explain."""
    import lightrag.private_file as private_file

    def _refuse(*args, **kwargs):
        raise PrivateFileError("verification refused")

    monkeypatch.setattr(private_file, "_windows_verify_dacl", _refuse)
    target = tmp_path / "copy.bin"
    with pytest.raises(PrivateFileError, match="verification refused"):
        with open_private_file(str(target)) as destination:
            destination.write(b"never reached")
    assert not target.exists()
