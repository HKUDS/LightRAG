"""Where the JSON configuration storage keeps each workspace's snapshot, and
the names no workspace may take.

Full contract: *JSON configuration shards* in
``docs/design/ConfigurationStorageContract.md``. The rules a caller meets:

* **Fixed paths.** Workspace ``w`` keeps its snapshot at
  ``WORKING_DIR/w/kv_workspace_config.json`` and its lifetime claim beside
  it; the empty workspace uses ``WORKING_DIR`` itself. No setting moves
  them. Resolve with ``json_config_dir``, never with a second rule.

* **Five reserved names.** A workspace may not be named after one of the
  deployment-wide files directly under ``WORKING_DIR`` (the anchor, its two
  locks, the empty workspace's snapshot and claim): its directory would
  collide with that file. ``validate_config_workspace`` refuses them, and the
  aliases Unicode case folding and normalization (``caseless_alias``) or
  Windows' trailing dot / space stripping maps onto them, as well as a name of only dots and spaces (Windows strips it to
  ``WORKING_DIR`` itself) and a drive-qualified name such as ``C:`` (Windows
  joins it outside ``WORKING_DIR``), on every platform so data stays
  portable; it is applied to every workspace an anchor member list, a
  discovered snapshot or a migration scope names, and to a start's workspace
  whenever one of its storages keeps a workspace directory under
  ``WORKING_DIR`` (``validate_deployment_workspace``). Names are never
  rewritten.

* **The scan is bounded.** ``discover_shards`` probes the root snapshot and
  one file in each direct child directory -- a symlinked child included, as
  the running storage follows it too, so registration and discovery agree.
  It never recurses and never reads a business file. The snapshot file itself
  must be a regular file -- a symlinked one is refused, by a start too, since
  the first atomic save would replace the link. A read error is an error,
  never absence; a child that holds a snapshot under an illegal name is
  refused by path, never skipped, and so is one snapshot reachable under two
  names (one physical directory holds one workspace; a start refuses to
  create that layout via ``check_snapshot_location``; both apply
  ``physical_directory_aliases``). Spellings are never renormalized: a
  listing that differs from a requested workspace or from its snapshot's
  owner only in Unicode normalization, and resolves to the same directory,
  is that workspace's own directory as a decomposing filesystem (HFS+)
  lists it. It reports what survives -- it cannot tell a never-created
  snapshot from a deleted one.

* **Offline reads are strict.** ``read_shard_file`` returns ``None`` only
  for a file that does not exist; undecodable content or a payload that is
  not a mapping of mappings raises.
"""

from __future__ import annotations

import json
import ntpath
import os
import stat
import unicodedata
from dataclasses import dataclass
from typing import Any

from lightrag.config_anchor import IDENTITY_SHARD_INVALID
from lightrag.exceptions import ConfigurationIdentityError
from lightrag.namespace import (
    ANCHOR_BIND_LOCK_FILE_NAME,
    ANCHOR_FILE_NAME,
    ANCHOR_LOCK_FILE_NAME,
    CONFIG_CLAIM_FILE_NAME,
    CONFIG_JSON_FILE_NAME,
)
from lightrag.utils import validate_workspace

RESERVED_WORKSPACE_NAMES = frozenset(
    {
        CONFIG_JSON_FILE_NAME,
        ANCHOR_FILE_NAME,
        CONFIG_CLAIM_FILE_NAME,
        ANCHOR_LOCK_FILE_NAME,
        ANCHOR_BIND_LOCK_FILE_NAME,
    }
)


def caseless_alias(name: str) -> str:
    """The key under which a case-insensitive or normalization-insensitive
    filesystem (NTFS, APFS) may resolve ``name`` to one directory: Unicode
    canonical caseless matching, after Windows' trailing dot / space strip.

    The setup wizard cannot decode ``\\u`` escapes, so it matches this rule
    for the reserved names by mapping the only non-ASCII characters whose
    fold is ASCII (``_ASCII_FOLDING_NON_ASCII``); keep the two in step.
    """
    stripped = unicodedata.normalize("NFD", name.rstrip(" ."))
    return unicodedata.normalize("NFD", stripped.casefold())


_RESERVED_CASELESS = frozenset(caseless_alias(n) for n in RESERVED_WORKSPACE_NAMES)
# Every non-ASCII character whose caseless fold is pure ASCII: the only ones
# a name can use to alias an ASCII reserved name. The setup wizard maps
# exactly these (as the lower-case ``\\uXXXX`` escapes json.dumps writes)
# before folding ASCII case; a test pins this table to ``caseless_alias``.
_ASCII_FOLDING_NON_ASCII = {
    "\u00df": "ss",
    "\u017f": "s",
    "\u037e": ";",
    "\u1e9e": "ss",
    "\u1fef": "`",
    "\u212a": "k",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


def validate_config_workspace(workspace: str) -> str:
    """``validate_workspace`` plus the five reserved root names.

    Raises ``ValueError``; returns the name unchanged.
    """
    validate_workspace(workspace)
    if ntpath.splitdrive(workspace)[0]:
        raise ValueError(
            f"Invalid workspace name {workspace!r}: its second character is "
            f"':', so Windows reads it as drive-qualified and joins it outside "
            f"WORKING_DIR. Choose another name."
        )
    if workspace and not caseless_alias(workspace):
        raise ValueError(
            f"Invalid workspace name {workspace!r}: it consists only of dots "
            f"and spaces, which Windows strips from a path component, so its "
            f"directory would be WORKING_DIR itself -- the empty workspace's. "
            f"Choose another name."
        )
    if caseless_alias(workspace) in _RESERVED_CASELESS:
        raise ValueError(
            f"Invalid workspace name {workspace!r}: it is the name of a "
            f"deployment-wide file directly under WORKING_DIR "
            f"({', '.join(sorted(RESERVED_WORKSPACE_NAMES))}, in any letter "
            f"case or Unicode normalization, or with trailing dots or spaces), "
            f"so its directory would collide with that file. Choose another name."
        )
    return workspace


def validate_deployment_workspace(workspace: str, *, storages) -> str:
    """The workspace check for one deployment's storage selection.

    ``validate_workspace`` (path traversal, the reserved ``$`` prefix) holds
    for every backend. The directory rules of ``validate_config_workspace``
    -- drive-qualified names, dot/space-only names, the five reserved root
    names -- exist because a workspace directory is created under
    ``WORKING_DIR``, so they apply only when one of ``storages`` (the
    configured backend names, configuration storage included) keeps its
    data there. A deployment entirely on database backends creates no such
    directory and keeps every name it could use before. Raises
    ``ValueError``; returns the name unchanged.
    """
    from lightrag.kg.working_dir_lock import uses_working_dir

    if uses_working_dir(*storages):
        return validate_config_workspace(workspace)
    return validate_workspace(workspace)


def json_config_dir(working_dir: str, workspace: str) -> str:
    """The absolute directory holding ``workspace``'s JSON configuration
    snapshot and its lifetime claim: ``WORKING_DIR/<workspace>``, or
    ``WORKING_DIR`` for the empty workspace."""
    validate_config_workspace(workspace)
    root = os.path.abspath(working_dir)
    return os.path.join(root, workspace) if workspace else root


def json_config_path(working_dir: str, workspace: str) -> str:
    """The absolute path of ``workspace``'s JSON configuration snapshot."""
    return os.path.join(json_config_dir(working_dir, workspace), CONFIG_JSON_FILE_NAME)


def _shard_error(path: str, detail: str) -> ConfigurationIdentityError:
    return ConfigurationIdentityError(
        f"The JSON configuration snapshot {path} is not usable: {detail}. It "
        f"is never skipped or treated as absent; repair or restore it.",
        cause=IDENTITY_SHARD_INVALID,
    )


@dataclass(frozen=True)
class DiscoveredShard:
    """A snapshot found by ``discover_shards``: the workspace its physical
    location names, and the file."""

    workspace: str
    path: str


def probe_snapshot(path: str) -> bool:
    """Whether ``path`` is a snapshot: ``False`` only when it does not exist.

    Anything present that is not a regular file -- a directory, a symlink --
    is refused rather than followed or skipped. A start applies the same
    check to its own snapshot, so a file it serves is never one discovery
    refuses.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as e:
        raise _shard_error(path, f"{type(e).__name__}: {e}") from e
    if stat.S_ISLNK(info.st_mode):
        raise _shard_error(
            path,
            "it is a symlink, which the first atomic save would replace with a "
            "regular file (leaving its target stale); replace it with a copy "
            "of the file it points to",
        )
    if not stat.S_ISREG(info.st_mode):
        raise _shard_error(path, "it is not a regular file")
    return True


def _physical_id(path: str) -> tuple[int, int] | None:
    """``path``'s directory identity (device, inode), following symlinks;
    ``None`` only when it does not exist."""
    try:
        info = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise _shard_error(path, f"{type(e).__name__}: {e}") from e
    return (info.st_dev, info.st_ino)


def _child_directories(root: str) -> list[os.DirEntry]:
    """Every direct child of ``root`` that is a directory, a symlink to one
    included (followed, like the running storage follows it)."""
    try:
        entries = list(os.scandir(root))
    except FileNotFoundError:
        return []
    except OSError as e:
        raise _shard_error(root, f"cannot list it ({type(e).__name__}: {e})") from e
    children = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=True):
                children.append(entry)
        except OSError as e:
            raise _shard_error(entry.path, f"{type(e).__name__}: {e}") from e
    return children


def _differ_only_in_normalization(a: str, b: str) -> bool:
    return a != b and unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b)


def _snapshot_owner(directory: str) -> str | None:
    """The workspace the snapshot in ``directory`` names as its owner;
    ``None`` when there is no snapshot or it names none readably.

    Used only to tell which spelling a listing that differs by Unicode
    normalization alone stands for. Never raises: the snapshot is validated
    in full where it is opened, and probed first so a FIFO is never read.
    """
    from lightrag.config_store import json_shard_owner_key

    path = os.path.join(directory, CONFIG_JSON_FILE_NAME)
    try:
        if not probe_snapshot(path):
            return None
        rows = read_shard_file(path) or {}
    except ConfigurationIdentityError:
        return None
    row = rows.get(json_shard_owner_key())
    value = row.get("value") if isinstance(row, dict) else None
    owner = value.get("workspace") if isinstance(value, dict) else None
    return owner if isinstance(owner, str) else None


def physical_directory_aliases(working_dir: str, names) -> list[list[str]]:
    """Every group of two or more names that is ONE physical directory.

    The names considered are ``names`` (the workspaces asked for),
    ``WORKING_DIR`` itself (as ``""``, the empty workspace's directory) and
    every existing direct child directory as listed; a name whose directory
    does not exist yet is skipped. Each group and the list are sorted. The
    one rule a start (``check_snapshot_location``) and the migration apply,
    so they cannot disagree about which spellings share a directory.

    A listed name is not a second name of a requested one when it differs
    from it only in Unicode normalization, the requested spelling is not
    listed itself, and the directory's snapshot names the requested
    workspace or no owner at all: that is how a decomposing filesystem
    (HFS+) lists the requested workspace's own directory. A snapshot owned
    by the listed spelling is another workspace that a normalization-
    insensitive filesystem (APFS) resolves both spellings to -- an alias. So
    is a second listed entry, such as a symlink named in the other
    normalization on a filesystem that keeps both. Every spelling is kept
    as given; nothing is renormalized.
    """
    root = os.path.abspath(working_dir)
    entries = {entry.name: entry.path for entry in _child_directories(root)}
    requested = set(names)
    candidates = requested | {""} | set(entries)
    physical: dict[tuple[int, int], set[str]] = {}
    for name in candidates:
        identity = _physical_id(os.path.join(root, name) if name else root)
        if identity is not None:
            physical.setdefault(identity, set()).add(name)
    groups: list[list[str]] = []
    for group in physical.values():
        kept = {
            name
            for name in group
            if not (
                name in entries
                and name not in requested
                and any(
                    other in group
                    and other not in entries
                    and _differ_only_in_normalization(name, other)
                    and _snapshot_owner(entries[name]) in (None, other)
                    for other in requested
                )
            )
        }
        if len(kept) > 1:
            groups.append(sorted(kept))
    return sorted(groups)


def _discovered_name(root: str, entry: os.DirEntry, listed: set[str]) -> str:
    """The workspace a listed child directory holding a snapshot stands for:
    its listed name, unless the snapshot's owner differs from it only in
    Unicode normalization, is not listed itself, and resolves to this very
    directory -- a decomposing filesystem (HFS+) listing the owner's
    directory in another form. The owner's spelling is then kept as stored,
    so discovery reports the name the anchor's members carry."""
    name = entry.name
    if unicodedata.normalize("NFC", name) == unicodedata.normalize("NFD", name):
        return name  # Nothing normalization could change, e.g. pure ASCII.
    owner = _snapshot_owner(entry.path)
    if (
        owner is not None
        and owner not in listed
        and _differ_only_in_normalization(name, owner)
        and _physical_id(os.path.join(root, owner)) == _physical_id(entry.path)
    ):
        return owner
    return name


def _one_directory_error(path: str, names: list[str]) -> ConfigurationIdentityError:
    listed = ", ".join(repr(name) if name else "(default workspace)" for name in names)
    return _shard_error(
        path,
        f"its directory is one physical directory reachable as {listed} "
        f"(a symlink to WORKING_DIR or to another workspace directory, or a "
        f"spelling a case-insensitive filesystem resolves to it), and one "
        f"directory holds one workspace -- replace the symlink with a "
        f"separate directory",
    )


def check_snapshot_location(working_dir: str, workspace: str) -> None:
    """A start's and a maintenance tool's check of ``workspace``'s own
    snapshot, before it is opened or bound: ``probe_snapshot``, and its
    directory must not be one physical directory with ``WORKING_DIR`` or
    another direct child -- ``discover_shards`` would find the snapshot under
    that name too and refuse it."""
    probe_snapshot(json_config_path(working_dir, workspace))
    if _physical_id(json_config_dir(working_dir, workspace)) is None:
        return
    for group in physical_directory_aliases(working_dir, [workspace]):
        if workspace in group:
            raise _one_directory_error(json_config_path(working_dir, workspace), group)


def discover_shards(working_dir: str) -> list[DiscoveredShard]:
    """Every JSON configuration snapshot under ``working_dir``, sorted by
    workspace: the root one (the empty workspace), then one per direct child
    directory (a symlink to a directory included) that holds one. One
    snapshot reachable under two names refuses. See the module rules."""
    root = os.path.abspath(working_dir)
    found: list[DiscoveredShard] = []
    seen: dict[tuple[int, int] | None, str] = {}
    root_file = os.path.join(root, CONFIG_JSON_FILE_NAME)
    if probe_snapshot(root_file):
        found.append(DiscoveredShard(workspace="", path=root_file))
        seen[_physical_id(root)] = ""
    children = _child_directories(root)
    listed = {entry.name for entry in children}
    for entry in children:
        # Only the one snapshot file inside is probed, so a link loop cannot
        # recurse.
        path = os.path.join(entry.path, CONFIG_JSON_FILE_NAME)
        if not probe_snapshot(path):
            continue
        name = _discovered_name(root, entry, listed)
        try:
            validate_config_workspace(name)
        except ValueError as e:
            raise _shard_error(path, f"its directory is not a legal workspace: {e}")
        physical = _physical_id(entry.path)
        if physical in seen:
            raise _one_directory_error(path, sorted([seen[physical], name]))
        seen[physical] = name
        found.append(DiscoveredShard(workspace=name, path=path))
    return sorted(found, key=lambda shard: shard.workspace)


def read_shard_file(path: str) -> dict[str, dict[str, Any]] | None:
    """A snapshot's rows as stored, ``None`` only when the file does not
    exist. An existing empty file reads as no rows, as the storage itself
    loads it."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            content = f.read()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise _shard_error(path, f"{type(e).__name__}: {e}") from e
    if not content.strip():
        return {}
    try:
        payload = json.loads(content)
    except ValueError as e:
        raise _shard_error(path, f"not valid JSON ({e})") from e
    if not isinstance(payload, dict):
        raise _shard_error(
            path, f"expected a JSON object, got {type(payload).__name__}"
        )
    for key, row in payload.items():
        if not isinstance(row, dict):
            raise _shard_error(path, f"record {key!r} is not a mapping")
    return payload
