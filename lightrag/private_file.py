"""Create a file no other user can read, from the instant it exists.

The caller is anything that writes a copy of stored data to a NEW path and
must not widen who can read it: today the offline rebuild tool's backups of a
corrupt vector container, which hold the same documents and metadata as the
store they copy and are kept indefinitely by design.

The two platforms need different mechanisms and only one of them is carried by
the file mode, so "create it, then tighten it" is not a portable shape -- on
Windows it is not even a correct one. See ``open_private_file``.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from typing import IO, Iterator

from lightrag.utils import logger

# Windows constants, from the SDK headers. Named here rather than inlined so
# the CreateFileW call below reads as the documented one.
_GENERIC_WRITE = 0x40000000
_READ_CONTROL = 0x00020000
_FILE_SHARE_NONE = 0x00000000
_CREATE_NEW = 1
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_ERROR_INSUFFICIENT_BUFFER = 122
_TOKEN_QUERY = 0x0008
_TOKEN_USER_CLASS = 1
_SE_FILE_OBJECT = 1
_DACL_SECURITY_INFORMATION = 0x00000004
_SDDL_REVISION_1 = 1
_ERROR_NO_TOKEN = 1008


def _invalid_handle_value() -> int:
    """``INVALID_HANDLE_VALUE`` as ctypes hands it back, not as C writes it.

    The SDK defines it ``(HANDLE)-1``, but a function whose ``restype`` is a
    pointer returns a Python int holding the unsigned bit pattern, so on 64-bit
    it arrives as 18446744073709551615 and ``handle == -1`` is FALSE. Comparing
    against the literal therefore let every failed creation through as a
    success -- including a name collision, whose cleanup then deleted the file
    that was already there.
    """
    import ctypes

    return ctypes.c_void_p(-1).value


class PrivateFileError(RuntimeError):
    """A file could not be created private, or could not be proven private.

    Always raised BEFORE anything is written into it, so a caller that lets
    this propagate has neither exposed data nor destroyed any.
    """


def _remove_quietly(path: str) -> None:
    """Delete a file WE created and no longer want.

    Only ever called on a path this module has just brought into existence.
    A failure is logged, never raised: it is always running while another
    exception is on its way out, and replacing that one with "could not clean
    up" would hide why anything was being cleaned up at all.
    """
    try:
        os.remove(path)
    except OSError as removal_error:
        logger.warning(f"Could not remove the unused file {path}: {removal_error}")


def _discard(destination, path: str) -> None:
    """Close and delete a file we created, without masking the reason."""
    try:
        destination.close()
    except OSError as close_error:
        logger.warning(f"Could not close the unused file {path}: {close_error}")
    _remove_quietly(path)


@contextlib.contextmanager
def open_private_file(path: str) -> Iterator[IO[bytes]]:
    """Create ``path`` and yield it open for binary writing, private.

    "Private" means: no user other than the one this process actually runs as
    can read the file, and that holds from the moment the file exists rather
    than from the moment some later call tightens it. The distinction is not
    academic on Windows, where access is checked when a handle is OPENED --
    a process that opened the file before a later tightening keeps reading
    through that handle afterwards, including everything written after.

    ``path`` must not already exist; the creation is exclusive, and an
    existing file raises ``FileExistsError`` rather than being reused or
    truncated.

    Three different failures, deliberately not merged into one type:

    * Being unable to CREATE the file privately, or to PROVE it private,
      raises ``PrivateFileError``. Nothing has been written, and nothing that
      was already on disk has been touched -- in particular a name collision
      raises ``FileExistsError`` and leaves the existing file alone.
    * An exception from the caller's own body (a source that could not be
      read, a disk that filled up) propagates unchanged. This does not
      reclassify it; it only cleans up.
    * In both of the above, this attempts to remove the file it created, so
      no partial copy is left looking like a complete one. That removal is
      best effort: if it fails it is logged and the original exception still
      wins, which can leave a zero-length or partial file behind under a
      ``.corrupt-<token>`` name no later run will reuse.

    POSIX: the mode comes from ``os.open``'s 0600, which the umask can only
    narrow, and is verified from the open descriptor.

    Windows: ``os.open``'s mode writes no ACL at all -- it sets the read-only
    ATTRIBUTE -- so the file would appear carrying the parent directory's
    inherited ACL, which may admit users the source's own ACL excluded. This
    uses ``CreateFileW`` instead, with a protected DACL granting only this
    process's SID and with sharing DISABLED, then reads the DACL back off the
    same handle before handing it to Python. The two settings cover different
    windows and neither replaces the other: the DACL governs every open after
    this one, including after the handle is closed, while ``dwShareMode=0``
    stops any second handle being opened while the copy is in flight.
    """
    if os.name == "nt":
        destination = _open_private_file_windows(path)
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            destination = os.fdopen(fd, "wb")
        except BaseException:
            # fdopen did not take ownership, so the descriptor is still ours.
            os.close(fd)
            _remove_quietly(path)
            raise
        try:
            mode = stat.S_IMODE(os.fstat(destination.fileno()).st_mode)
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise PrivateFileError(
                    f"{path} was created mode {oct(mode)}, which grants group "
                    "or other access. The filesystem did not honour the "
                    "requested 0600."
                )
        except BaseException:
            _discard(destination, path)
            raise
    try:
        yield destination
    except BaseException:
        _discard(destination, path)
        raise
    finally:
        if not destination.closed:
            destination.close()


def windows_create_exclusive(kernel32, ctypes, path: str, attributes):
    """``CreateFileW`` plus the ONLY correct reading of what it returned.

    Split out and given plain parameters so the failure classification can be
    exercised on any platform: it is the part where being wrong destroys a
    file rather than merely refusing one.

    A caller may delete ``path`` only if this RETURNS. Every raise here leaves
    the filesystem as it found it -- which matters most for the collision
    case, where the file that exists is somebody else's and deleting it was
    the bug this signature exists to prevent.
    """
    handle = kernel32.CreateFileW(
        path,
        _GENERIC_WRITE | _READ_CONTROL,
        _FILE_SHARE_NONE,
        ctypes.byref(attributes),
        _CREATE_NEW,
        _FILE_ATTRIBUTE_NORMAL,
        None,
    )
    # None (a NULL pointer restype), 0, and the unsigned form of (HANDLE)-1
    # are all "no handle"; see _invalid_handle_value for why the last one is
    # not spelled -1.
    if handle is None or handle in (0, _invalid_handle_value()):
        code = ctypes.get_last_error()
        if code in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
        raise PrivateFileError(
            f"Could not create {path} with a private security descriptor: "
            f"{ctypes.WinError(code)}"
        )
    return handle


def _open_private_file_windows(path: str) -> IO[bytes]:
    """``open_private_file``'s Windows half: create, verify, hand over.

    Ordering is the point. The security descriptor goes IN to the creation
    call, so the file is never briefly inheritable; the read-back happens on
    the handle that call returned, so it describes the object created and not
    whatever a later path lookup would find; and the handle becomes Python's
    only at the very end, so every failure before that has exactly one owner
    to clean up.
    """
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _declare_windows_signatures(kernel32, advapi32, wintypes)

    sid = _windows_current_user_sid(kernel32, advapi32, ctypes, wintypes)
    descriptor = _windows_security_descriptor(
        kernel32, advapi32, ctypes, f"O:{sid}D:P(A;;FA;;;{sid})"
    )
    try:

        class SECURITY_ATTRIBUTES(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", wintypes.BOOL),
            ]

        attributes = SECURITY_ATTRIBUTES()
        attributes.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
        attributes.lpSecurityDescriptor = descriptor
        attributes.bInheritHandle = False
        handle = windows_create_exclusive(kernel32, ctypes, path, attributes)
    finally:
        kernel32.LocalFree(descriptor)

    try:
        _windows_verify_dacl(kernel32, advapi32, ctypes, wintypes, handle, path, sid)
        # Past this point the fd owns the handle: closing both would close
        # whatever the OS reassigned the number to in between.
        fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except BaseException:
        kernel32.CloseHandle(handle)
        _remove_quietly(path)
        raise
    try:
        return os.fdopen(fd, "wb")
    except BaseException:
        os.close(fd)
        _remove_quietly(path)
        raise


def _declare_windows_signatures(kernel32, advapi32, wintypes) -> None:
    """Give every call an explicit signature.

    ctypes defaults an unnamed argument to C ``int``, which truncates a 64-bit
    HANDLE or pointer to 32 bits and turns a working call into one that fails
    or corrupts silently. Nothing below is optional.
    """
    import ctypes

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentThread.argtypes = []
    kernel32.GetCurrentThread.restype = wintypes.HANDLE

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.OpenThreadToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.BOOL,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenThreadToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD


def _windows_current_user_sid(kernel32, advapi32, ctypes, wintypes) -> str:
    """The SID of the identity this process ACTUALLY runs as, as a string.

    Read from the access token, never from ``USERNAME`` / ``USERDOMAIN``:
    those are inherited environment strings that a caller can set to anything,
    and the grant has to name whoever will really own the handle. The thread
    token is tried first so that a thread running under an impersonated
    identity grants to that identity rather than to the process's.

    That fallback is allowed for exactly ONE reason -- ``ERROR_NO_TOKEN``,
    which means the thread is simply not impersonating and the process token
    is the right answer. Any other failure (access denied, for one) says a
    token exists that could not be read, and falling back there would grant
    the file to a DIFFERENT identity than the one that will own it, quietly.
    """
    token = wintypes.HANDLE()
    if not advapi32.OpenThreadToken(
        kernel32.GetCurrentThread(), _TOKEN_QUERY, True, ctypes.byref(token)
    ):
        code = ctypes.get_last_error()
        if code != _ERROR_NO_TOKEN:
            raise PrivateFileError(
                "Could not read this thread's impersonation token, so the "
                "identity that would own the file cannot be established: "
                f"{ctypes.WinError(code)}"
            )
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
        ):
            raise PrivateFileError(
                "Could not open this process's access token, so there is no "
                f"identity to grant the file to: {ctypes.WinError(ctypes.get_last_error())}"
            )
    try:
        size = wintypes.DWORD(0)
        advapi32.GetTokenInformation(
            token, _TOKEN_USER_CLASS, None, 0, ctypes.byref(size)
        )
        code = ctypes.get_last_error()
        if size.value == 0 or code not in (0, _ERROR_INSUFFICIENT_BUFFER):
            raise PrivateFileError(
                f"Could not size the token's user information: {ctypes.WinError(code)}"
            )
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token, _TOKEN_USER_CLASS, buffer, size, ctypes.byref(size)
        ):
            raise PrivateFileError(
                "Could not read the token's user information: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        # TOKEN_USER is a SID_AND_ATTRIBUTES, whose first member is the PSID.
        sid_pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p)).contents
        text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(sid_pointer, ctypes.byref(text)):
            raise PrivateFileError(
                "Could not format the token's SID: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        try:
            return text.value
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.CloseHandle(token)


def _windows_security_descriptor(kernel32, advapi32, ctypes, sddl: str):
    """Build a self-relative security descriptor from SDDL.

    SDDL rather than a hand-assembled ACL because the whole descriptor is then
    one reviewable string: ``D:P`` protects the DACL, which is what stops the
    parent directory's entries being inherited, and the single ``A`` entry
    grants ``FA`` (full access) to one SID and to nobody else. Full access,
    not read: the operator has to be able to read, archive and eventually
    delete these files.

    The returned pointer is LocalAlloc'd and must be freed by the caller.
    """
    descriptor = ctypes.c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None
    ):
        raise PrivateFileError(
            f"Could not build a security descriptor from {sddl!r}: "
            f"{ctypes.WinError(ctypes.get_last_error())}"
        )
    return descriptor


def _windows_verify_dacl(
    kernel32, advapi32, ctypes, wintypes, handle, path: str, sid: str
) -> None:
    """Prove the created object carries the DACL that was asked for.

    Read back from the HANDLE, so it describes the object that was created.
    A filesystem that does not persist ACLs (FAT32, some network redirectors)
    answers here rather than silently storing the data unprotected -- either
    the call fails or the DACL that comes back is not the one requested, and
    both refuse.

    The check is on the SDDL round-trip rather than on a walk of the ACE
    array: the same string form the descriptor was built from, so a mismatch
    is legible in the error.
    """
    descriptor = ctypes.c_void_p()
    status = advapi32.GetSecurityInfo(
        handle,
        _SE_FILE_OBJECT,
        _DACL_SECURITY_INFORMATION,
        None,
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    if status != 0:
        raise PrivateFileError(
            f"Could not read back the security descriptor of {path}, so its "
            "privacy cannot be confirmed (a filesystem that does not persist "
            f"ACLs answers this way): {ctypes.WinError(status)}"
        )
    try:
        text = wintypes.LPWSTR()
        if not advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor,
            _SDDL_REVISION_1,
            _DACL_SECURITY_INFORMATION,
            ctypes.byref(text),
            None,
        ):
            raise PrivateFileError(
                f"Could not format the security descriptor of {path}: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        try:
            actual = text.value or ""
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.LocalFree(descriptor)
    assert_dacl_grants_only(
        actual,
        sid,
        path,
        sid_matches=lambda field: _windows_sids_equal(
            kernel32, advapi32, ctypes, field, sid
        ),
    )


def _windows_sids_equal(kernel32, advapi32, ctypes, left: str, right: str) -> bool:
    """Compare two SDDL SID fields by value, whatever they are spelled as.

    ``ConvertStringSidToSidW`` accepts both forms -- a full ``S-1-...`` string
    and a well-known alias like ``LA`` or ``BA`` -- and resolves the alias
    against this machine, which is the same machine that owns the file. So
    this answers "is this the same account", which the text never did.
    """

    def to_sid(text: str):
        sid = ctypes.c_void_p()
        if not advapi32.ConvertStringSidToSidW(text, ctypes.byref(sid)):
            raise PrivateFileError(
                f"Could not resolve the SID {text!r} read back from the file's "
                f"DACL: {ctypes.WinError(ctypes.get_last_error())}"
            )
        return sid

    first = to_sid(left)
    try:
        second = to_sid(right)
        try:
            return bool(advapi32.EqualSid(first, second))
        finally:
            kernel32.LocalFree(second)
    finally:
        kernel32.LocalFree(first)


def assert_dacl_grants_only(sddl: str, sid: str, path: str, sid_matches=None) -> None:
    """Refuse a DACL that is not exactly "protected, this SID alone".

    Split out from the Win32 calls so it can be exercised anywhere: it is the
    half of the verification that encodes what "private" means, and the half
    most likely to be wrong.

    Three conditions, each for its own reason. ``D:P`` -- without the protect
    flag the entries below are whatever the parent directory dictates today
    and can change under the file tomorrow. Exactly one entry -- a second one
    is a second audience, whatever it grants. And that entry must be an allow
    for this SID, because an entry for anyone else is the whole failure this
    guards.

    ``sid_matches`` decides the last one, and Windows callers MUST supply it.
    An SDDL SID field is not a stable spelling: the string this process wrote
    comes back abbreviated to a two-letter alias whenever the account has one,
    so a store created by the built-in Administrator (RID 500) is written as
    ``S-1-5-21-...-500`` and read back as ``LA``. Comparing the text refuses
    every such file -- while passing for an ordinary user, whose SID has no
    alias, which is exactly the shape of bug that survives local testing and
    fails on someone else's machine. The Windows path therefore compares by
    VALUE, through ``ConvertStringSidToSidW`` and ``EqualSid``. The default
    here is text equality, for callers testing the structural rules with SIDs
    they control.
    """
    if sid_matches is None:

        def sid_matches(field: str) -> bool:
            return field == sid

    dacl = sddl.strip()
    if not dacl.startswith("D:"):
        raise PrivateFileError(f"{path}: expected a DACL, got {sddl!r}")
    flags = dacl[2:].split("(", 1)[0]
    if "P" not in flags:
        raise PrivateFileError(
            f"{path}: its DACL is not protected ({sddl!r}), so it still "
            "inherits access from the parent directory"
        )
    entries = [entry for entry in dacl.split("(")[1:]]
    if len(entries) != 1:
        raise PrivateFileError(
            f"{path}: expected exactly one access entry, found {len(entries)} "
            f"({sddl!r})"
        )
    fields = entries[0].rstrip(")").split(";")
    if len(fields) < 6 or fields[0] != "A" or not sid_matches(fields[5]):
        raise PrivateFileError(
            f"{path}: its only access entry is not an allow for {sid} ({sddl!r})"
        )
