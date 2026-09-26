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
  collide with that file. ``validate_config_workspace`` refuses them; it is
  applied to every workspace a start, an anchor member list, a discovered
  snapshot or a migration scope names. Names are never rewritten.

* **The scan is bounded.** ``discover_shards`` probes the root snapshot and
  one file in each direct child directory. It never recurses, never follows
  a symlink, and never reads a business file. A read error is an error,
  never absence; a child that holds a snapshot under an illegal name is
  refused by path, never skipped. It reports what survives -- it cannot
  tell a never-created snapshot from a deleted one.

* **Offline reads are strict.** ``read_shard_file`` returns ``None`` only
  for a file that does not exist; undecodable content or a payload that is
  not a mapping of mappings raises.
"""

from __future__ import annotations

import json
import os
import stat
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


def validate_config_workspace(workspace: str) -> str:
    """``validate_workspace`` plus the five reserved root names.

    Raises ``ValueError``; returns the name unchanged.
    """
    validate_workspace(workspace)
    if workspace in RESERVED_WORKSPACE_NAMES:
        raise ValueError(
            f"Invalid workspace name {workspace!r}: it is the name of a "
            f"deployment-wide file directly under WORKING_DIR "
            f"({', '.join(sorted(RESERVED_WORKSPACE_NAMES))}), so its "
            f"directory would collide with that file. Choose another name."
        )
    return workspace


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


def _probe(path: str) -> bool:
    """Whether ``path`` is a snapshot: ``False`` only when it does not exist.

    Anything present that is not a regular file -- a directory, a symlink --
    is refused rather than followed or skipped.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as e:
        raise _shard_error(path, f"{type(e).__name__}: {e}") from e
    if not stat.S_ISREG(info.st_mode):
        raise _shard_error(path, "it is not a regular file")
    return True


def discover_shards(working_dir: str) -> list[DiscoveredShard]:
    """Every JSON configuration snapshot under ``working_dir``, sorted by
    workspace: the root one (the empty workspace), then one per direct,
    non-symlink child directory that holds one. See the module rules."""
    root = os.path.abspath(working_dir)
    found: list[DiscoveredShard] = []
    root_file = os.path.join(root, CONFIG_JSON_FILE_NAME)
    if _probe(root_file):
        found.append(DiscoveredShard(workspace="", path=root_file))
    try:
        entries = list(os.scandir(root))
    except FileNotFoundError:
        return found
    except OSError as e:
        raise _shard_error(root, f"cannot list it ({type(e).__name__}: {e})") from e
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError as e:
            raise _shard_error(entry.path, f"{type(e).__name__}: {e}") from e
        path = os.path.join(entry.path, CONFIG_JSON_FILE_NAME)
        if not _probe(path):
            continue
        try:
            validate_config_workspace(entry.name)
        except ValueError as e:
            raise _shard_error(path, f"its directory is not a legal workspace: {e}")
        found.append(DiscoveredShard(workspace=entry.name, path=path))
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
