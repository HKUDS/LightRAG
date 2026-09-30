"""The configuration storage's identity anchor: which container this
deployment is bound to, kept where no storage selection can move it.

Full contract: *The anchor and the container identity* in
``docs/design/ConfigurationStorageContract.md``. The rules a caller meets:

* **Fixed path.** ``<working_dir>/config_storage_anchor.json``, directly
  under ``working_dir`` for every backend and every workspace: it binds the
  whole deployment, so it lives outside every workspace, the empty one
  included. No setting moves it.

* **One format.** ``schema_version`` (integer ``1``), ``backend`` and
  ``storage_uuid``; a ``JsonKVStorage`` anchor adds ``layout``
  (``"json_shards"``) and ``members``, the workspaces whose JSON snapshots
  belong to this group. No host, credential, path or connection string: the
  identity is compared, connection details are not.

* **Strict read.** Only a genuine "file does not exist" is the no-anchor
  branch. A permission error, a truncated or corrupt file, an unknown
  version, an unadmitted backend, a malformed UUID, a member list on a
  database anchor or a bad member name refuses; it is never read as absent.

* **Two write modes.** ``publish_anchor(..., replace=False)`` is the bind: it
  never overwrites an anchor that exists. ``replace=True`` is the offline
  migration's commit point and a JSON start's member append (under the bind
  lock, from a list re-read inside it) -- nothing else. A normal start never
  changes the backend or the UUID and never deletes an anchor. Both modes
  write a temp file in the same directory, ``fsync`` it, publish, then
  ``fsync`` the directory where the platform supports it; a failure anywhere
  is raised, never reported as a success.

* **Deleting the file is the sanctioned rebind**, with every server and
  maintenance writer stopped. The next start binds to whatever container
  the current configuration selects (for JSON, the surviving consistent
  snapshots), with a WARNING. Running processes do not re-read the file, so
  it must not be deleted while servers are up.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import uuid
from dataclasses import dataclass

from lightrag.exceptions import ConfigurationIdentityError
from lightrag.file_atomic import tmp_path_for
from lightrag.namespace import ANCHOR_FILE_NAME

ANCHOR_SCHEMA_VERSION = 1
_ANCHOR_FIELDS = frozenset({"schema_version", "backend", "storage_uuid"})
# The JSON backend's extra fields, and the one layout value it may carry.
_JSON_ANCHOR_FIELDS = _ANCHOR_FIELDS | {"layout", "members"}
JSON_CONFIG_BACKEND = "JsonKVStorage"
JSON_SHARDS_LAYOUT = "json_shards"

# ``ConfigurationIdentityError.cause`` values. A caller branches on these,
# never on message text.
IDENTITY_ANCHOR_UNREADABLE = "anchor_unreadable"
IDENTITY_ANCHOR_WRITE_FAILED = "anchor_write_failed"
IDENTITY_ANCHOR_APPEARED = "anchor_appeared"
IDENTITY_BACKEND_MISMATCH = "backend_mismatch"
IDENTITY_UUID_MISSING = "uuid_missing"
IDENTITY_UUID_MISMATCH = "uuid_mismatch"
IDENTITY_ROW_INVALID = "identity_invalid"
IDENTITY_WRITE_FAILED = "identity_write_failed"
# JSON shards: a snapshot that is not a valid member (wrong owner, foreign
# rows, a UUID that disagrees with the group), a registered member whose
# snapshot lost its identity, and a workspace a maintenance tool found
# unregistered.
IDENTITY_SHARD_INVALID = "shard_invalid"
IDENTITY_MEMBER_MISSING = "member_missing"
IDENTITY_MEMBER_UNREGISTERED = "member_unregistered"

# Directory-fsync errors that mean "this filesystem cannot fsync a
# directory", not "the fsync failed". Anything else is a failure.
_DIR_FSYNC_UNSUPPORTED = frozenset(
    code
    for code in (
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EBADF", None),
    )
    if code is not None
)

# ``os.link`` errors that mean "this filesystem has no hard links", which is
# when the bind falls back to an exclusive-create claim of the anchor name.
_LINK_UNSUPPORTED = frozenset(
    code
    for code in (
        getattr(errno, "EPERM", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EMLINK", None),
        getattr(errno, "ENOSYS", None),
    )
    if code is not None
)


def anchor_dir(working_dir: str) -> str:
    """The directory the anchor and its two locks live in: ``working_dir``
    itself, absolute."""
    return os.path.abspath(working_dir)


def anchor_path(working_dir: str) -> str:
    """``<working_dir>/config_storage_anchor.json``, absolute."""
    return os.path.join(anchor_dir(working_dir), ANCHOR_FILE_NAME)


def new_storage_uuid() -> str:
    """A fresh container identity."""
    return str(uuid.uuid4())


def canonical_storage_uuid(value: object) -> str | None:
    """``value`` if it is a UUID in canonical lowercase hyphenated form, else
    ``None``. Canonical only, so two spellings of one UUID can never compare
    unequal."""
    if not isinstance(value, str):
        return None
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return None
    return value if str(parsed) == value else None


def _admitted_backends() -> tuple[str, ...]:
    from lightrag.kg import STORAGE_IMPLEMENTATIONS

    return tuple(STORAGE_IMPLEMENTATIONS["CONFIG_STORAGE"]["implementations"])


@dataclass(frozen=True)
class StorageAnchor:
    """The binding the anchor file records: a backend type, the UUID of the
    configuration container it was bound to and, for ``JsonKVStorage`` only,
    the sorted tuple of registered workspace snapshots (``None`` otherwise)."""

    backend: str
    storage_uuid: str
    members: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (self.backend == JSON_CONFIG_BACKEND) != (self.members is not None):
            raise ValueError(
                f"a {self.backend} anchor "
                + (
                    "must carry a member tuple"
                    if self.backend == JSON_CONFIG_BACKEND
                    else "carries no members"
                )
            )
        if self.members is not None:
            object.__setattr__(self, "members", tuple(sorted(set(self.members))))

    def with_member(self, workspace: str) -> "StorageAnchor":
        """This anchor with ``workspace`` registered (JSON only)."""
        if self.members is None:
            raise ValueError(f"a {self.backend} anchor has no member list")
        return StorageAnchor(
            backend=self.backend,
            storage_uuid=self.storage_uuid,
            members=(*self.members, workspace),
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "backend": self.backend,
            "storage_uuid": self.storage_uuid,
        }
        if self.members is not None:
            payload["layout"] = JSON_SHARDS_LAYOUT
            payload["members"] = list(self.members)
        return payload


def _unreadable(path: str, detail: str) -> ConfigurationIdentityError:
    return ConfigurationIdentityError(
        f"The configuration storage anchor {path} could not be read: {detail}. "
        f"It is never treated as absent. Repair the file or restore it from "
        f"backup; if this deployment is meant to rebind to the container its "
        f"current configuration selects, delete the file and restart.",
        cause=IDENTITY_ANCHOR_UNREADABLE,
        anchor_path=path,
    )


def parse_anchor_payload(payload: object, *, path: str) -> StorageAnchor:
    """Validate a decoded anchor. Raises ``ConfigurationIdentityError``."""
    if not isinstance(payload, dict):
        raise _unreadable(path, f"expected a JSON object, got {type(payload).__name__}")
    keys = set(payload)
    # The backend decides which shape is expected; an unadmitted or missing
    # backend is refused below, after the structure check names what is off.
    expected = (
        _JSON_ANCHOR_FIELDS
        if payload.get("backend") == JSON_CONFIG_BACKEND
        else _ANCHOR_FIELDS
    )
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise _unreadable(
            path, f"unexpected structure (missing {missing}, unexpected {extra})"
        )
    version = payload["schema_version"]
    if type(version) is not int or version != ANCHOR_SCHEMA_VERSION:
        raise _unreadable(
            path,
            f"unsupported schema_version {version!r}; expected integer "
            f"{ANCHOR_SCHEMA_VERSION}",
        )
    backend = payload["backend"]
    admitted = _admitted_backends()
    if not isinstance(backend, str) or backend not in admitted:
        raise _unreadable(
            path,
            f"backend {backend!r} is not a configuration storage backend "
            f"(admitted: {', '.join(admitted)})",
        )
    storage_uuid = canonical_storage_uuid(payload["storage_uuid"])
    if storage_uuid is None:
        raise _unreadable(
            path, f"storage_uuid {payload['storage_uuid']!r} is not a canonical UUID"
        )
    if backend != JSON_CONFIG_BACKEND:
        return StorageAnchor(backend=backend, storage_uuid=storage_uuid)
    if payload["layout"] != JSON_SHARDS_LAYOUT:
        raise _unreadable(
            path,
            f"layout {payload['layout']!r} is not {JSON_SHARDS_LAYOUT!r}",
        )
    return StorageAnchor(
        backend=backend,
        storage_uuid=storage_uuid,
        members=_parse_members(payload["members"], path=path),
    )


def _parse_members(members: object, *, path: str) -> tuple[str, ...]:
    """A JSON anchor's member list: distinct legal workspace names."""
    from lightrag.config_shards import validate_config_workspace

    if not isinstance(members, list):
        raise _unreadable(path, f"members {members!r} is not a list")
    seen: set[str] = set()
    for member in members:
        if not isinstance(member, str):
            raise _unreadable(path, f"member {member!r} is not a string")
        try:
            validate_config_workspace(member)
        except ValueError as e:
            raise _unreadable(path, f"member {member!r} is not a workspace: {e}")
        if member in seen:
            raise _unreadable(path, f"member {member!r} is listed twice")
        seen.add(member)
    return tuple(members)


def read_anchor(working_dir: str) -> StorageAnchor | None:
    """The anchor on record, or ``None`` ONLY when the file does not exist.

    Raises ``ConfigurationIdentityError`` (cause
    ``IDENTITY_ANCHOR_UNREADABLE``) for every other outcome that is not a
    valid anchor: a permission error, a directory where the file should be,
    undecodable or truncated content, a wrong structure or version -- and
    anything that is not a regular file, a symlink included: every publish
    is an atomic replace of the path, which would swap the link for a
    private regular file and leave its target stale.
    """
    path = anchor_path(working_dir)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise _unreadable(path, f"{type(e).__name__}: {e}") from e
    if stat.S_ISLNK(info.st_mode):
        raise _unreadable(
            path,
            "it is a symlink, which the next atomic publish would replace with "
            "a regular file (leaving its target stale); replace it with a copy "
            "of the file it points to",
        )
    if not stat.S_ISREG(info.st_mode):
        raise _unreadable(path, "it is not a regular file")
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise _unreadable(path, f"{type(e).__name__}: {e}") from e
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise _unreadable(path, f"not valid JSON ({e})") from e
    return parse_anchor_payload(payload, path=path)


def _fsync_dir(directory: str) -> None:
    """Make a rename in ``directory`` durable where the platform can.

    Windows cannot open a directory for ``fsync``; some filesystems answer
    ``EINVAL`` / ``ENOTSUP``. Those are "not supported" and pass. Any other
    error is a failure the caller reports.
    """
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in _DIR_FSYNC_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def _write_temp(tmp: str, anchor: StorageAnchor) -> None:
    data = json.dumps(anchor.to_payload(), indent=2, sort_keys=True) + "\n"
    with open(tmp, "x", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _publish_no_clobber(tmp: str, path: str) -> None:
    """Move ``tmp`` to ``path`` unless ``path`` exists; ``FileExistsError`` if
    it does.

    POSIX: ``link`` then ``unlink`` -- ``link(2)`` never replaces its target.
    Windows: ``rename`` refuses an existing target natively. A filesystem
    without hard links claims the name with ``O_CREAT | O_EXCL`` and only
    then replaces the empty claim with the temp file, so the fallback is
    no-clobber on its own and does not lean on any lock (the bind lock fails
    open, and the keyed lock cannot see another process tree). Until the
    replace lands the anchor is an empty file, which every reader refuses as
    unreadable -- loud, never "absent" -- and a crash in that window leaves
    it for the operator to delete, the sanctioned rebind.
    """
    if os.name == "nt":
        os.rename(tmp, path)
        return
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise
    except OSError as e:
        if e.errno not in _LINK_UNSUPPORTED:
            raise
        # Raises FileExistsError itself when another start got there first.
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
        os.replace(tmp, path)
        return
    # Published. The temp name is a second link to the same file; failing to
    # remove it leaves a harmless orphan, not a failed publish.
    try:
        os.unlink(tmp)
    except OSError:
        pass


def publish_anchor(working_dir: str, anchor: StorageAnchor, *, replace: bool) -> str:
    """Durably write ``anchor``; returns the path.

    ``replace=False`` (the bind) never overwrites an existing anchor and
    raises ``ConfigurationIdentityError`` with cause
    ``IDENTITY_ANCHOR_APPEARED`` when one is found. ``replace=True``
    (``os.replace``) is the offline migration's atomic commit and a JSON
    start's member append, which re-reads the anchor under the bind lock
    first; it is used nowhere else. Every other failure -- the directory, the temp file, its fsync, the
    publish, the directory fsync -- raises with cause
    ``IDENTITY_ANCHOR_WRITE_FAILED``; the temp file is removed best effort.
    """
    if anchor.backend not in _admitted_backends():
        raise ValueError(f"{anchor.backend!r} is not a configuration storage backend")
    if canonical_storage_uuid(anchor.storage_uuid) is None:
        raise ValueError(f"{anchor.storage_uuid!r} is not a canonical UUID")
    path = anchor_path(working_dir)
    directory = os.path.dirname(path)
    tmp: str | None = None
    stage = "create the anchor directory"
    try:
        os.makedirs(directory, exist_ok=True)
        stage = "write the temporary anchor file"
        # Named before it is created, so a write that fails part-way still
        # leaves the ``finally`` below a file to remove.
        tmp = tmp_path_for(path)
        _write_temp(tmp, anchor)
        stage = "publish the anchor"
        if replace:
            os.replace(tmp, path)
        else:
            _publish_no_clobber(tmp, path)
        tmp = None
        stage = "fsync the anchor directory"
        _fsync_dir(directory)
    except FileExistsError as e:
        raise ConfigurationIdentityError(
            f"The configuration storage anchor {path} appeared while this "
            f"start was binding; another process bound this working directory "
            f"at the same time, which is unsupported. Nothing was overwritten. "
            f"Restart, one server at a time.",
            cause=IDENTITY_ANCHOR_APPEARED,
            anchor_path=path,
        ) from e
    except OSError as e:
        published = stage == "fsync the anchor directory"
        raise ConfigurationIdentityError(
            f"Could not {stage} for the configuration storage anchor {path} "
            f"({type(e).__name__}: {e}). "
            + (
                "The anchor was published but may not be durable; the next "
                "start re-checks it against the container."
                if published
                else "No anchor was published."
            )
            + " WORKING_DIR must be writable for the first bind, and must "
            "persist across restarts.",
            cause=IDENTITY_ANCHOR_WRITE_FAILED,
            anchor_path=path,
        ) from e
    finally:
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return path
