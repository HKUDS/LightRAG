#!/usr/bin/env python3
"""``lightrag-migrate-config``: move the configuration container to a
different backend TYPE, offline, and move the anchor only once the copy is
verified.

Full contract: *Offline migration* under *The anchor and the container
identity* in ``docs/design/ConfigurationStorageContract.md``; operator guide:
``lightrag/tools/README_MIGRATE_CONFIG.md``. The rules:

* **Cross-type only.** A same-type move (PostgreSQL to PostgreSQL, Mongo to
  Mongo, an OpenSearch snapshot, copying the whole WORKING_DIR) is the
  backend's own dump/restore: the identity row travels with the data and
  "same type, same UUID" passes. The PostgreSQL, MongoDB and OpenSearch client managers are
  process-wide singletons that read ``os.environ``, which is also why two
  containers of ONE type cannot be open here at once.
* **The UUID is kept.** The target receives the source's identity; the type
  already tells the two apart.
* **Exclusive.** The anchor lock is taken exclusively, so every server, SDK
  process and maintenance tool on this ``WORKING_DIR`` must be stopped; where
  the filesystem cannot lock, ``--assume-exclusive`` is required.
* **Ownership before data.** The target's identity row is written FIRST and
  is what makes a re-run safe without a journal: a target that holds this
  identity is this migration's to reconcile, anything else is refused.
* **The anchor is the commit point.** It is replaced atomically only after a
  strict flush and a full verification; before that the old configuration
  keeps working on the source. No source row is ever written or deleted, and
  no ``.env`` or environment is ever written.
* **JSON is the whole group.** A JSON side is every workspace snapshot of
  this ``WORKING_DIR`` (``JsonShardGroup``), never the invoking workspace's
  alone. As a source, the anchor's members and the snapshots on disk must
  agree in both directions before anything is written; as a target, every
  source workspace and every discovered snapshot is claimed and validated
  first, same-identity snapshots are converged, and a snapshot the source no
  longer has rows for is kept with its identity and owner only and stays a
  member. The owner rows are layout metadata and never leave JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from dotenv import dotenv_values, load_dotenv

from lightrag import config_store as cs
from lightrag.config_anchor import (
    JSON_CONFIG_BACKEND,
    StorageAnchor,
    anchor_path,
    publish_anchor,
    read_anchor,
)
from lightrag.config_shards import (
    discover_shards,
    caseless_alias,
    json_config_dir,
    read_shard_file,
    validate_config_workspace,
)
from lightrag.constants import DEFAULT_WORKING_DIR
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationIdentityError,
    ConfigurationStorageError,
    CorruptStorageRecordError,
    WorkingDirectoryInUseError,
)
from lightrag.kg.anchor_lock import (
    acquire_anchor_lock_exclusive,
    acquire_anchor_lock_shared,
    release_anchor_lock_shared,
)
from lightrag.utils import normalize_server_workspace, setup_logger

# Fields a backend adds to a row it returns and owns itself; never content.
BACKEND_METADATA_KEYS = frozenset({"_id", "create_time", "update_time"})

# Envelope fields an admitted backend overwrites on write and returns as its
# own value on read, so a row's OWN field of that name cannot be stored there:
# PostgreSQL returns the key as ``id``; OpenSearch writes the key as
# ``__mirrored_id`` and drops it from every read. Only PostgreSQL's appears in
# what it yields, and is stripped there (``row_payload``); a source row that
# owns either field is refused for that target before the claim.
RESERVED_FIELDS_BY_BACKEND: dict[str, frozenset[str]] = {
    "PGKVStorage": frozenset({"id"}),
    "OpenSearchKVStorage": frozenset({"__mirrored_id"}),
}
_RESERVED_FIELDS = frozenset().union(*RESERVED_FIELDS_BY_BACKEND.values())

# The environment variables each configuration backend reads, by prefix (or
# exact name). Two backends of different types read disjoint sets, which is
# what lets one process hold both connections.
BACKEND_ENV_PREFIXES: dict[str, tuple[str, ...]] = {
    "JsonKVStorage": (),
    "PGKVStorage": ("POSTGRES_",),
    "MongoKVStorage": ("MONGO_", "MONGODB_"),
    "OpenSearchKVStorage": ("OPENSEARCH_",),
}

DEFAULT_PAGE_SIZE = 200


class MigrationRefused(RuntimeError):
    """A precondition does not hold; nothing was written."""


class MigrationFailed(RuntimeError):
    """A step failed after the target was claimed; the anchor is unchanged
    and a re-run resumes."""


class MigrationIndeterminate(RuntimeError):
    """The anchor replace raised and the anchor could not be read back: it
    may name either backend, and must be inspected before anything starts."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def row_payload(row: dict[str, Any], *, id_mirror: bool = False) -> dict[str, Any]:
    """The row envelope, verbatim, without the backend-owned metadata.

    ``id_mirror`` is set only for rows read from a backend that adds the key
    as ``id`` (``PGKVStorage`` sets both ``id`` and ``_id`` to it), and only
    then is an ``id`` equal to the key dropped. Every other ``id`` is
    envelope content: copied and verified, never silently dropped.
    """
    payload = {k: v for k, v in row.items() if k not in BACKEND_METADATA_KEYS}
    if id_mirror and "id" in payload and payload["id"] == row.get("_id"):
        del payload["id"]
    return payload


# The admitted configuration backends that return the key as ``id`` too.
_ID_MIRROR_BACKENDS = frozenset({"PGKVStorage"})


def _mirrors_id(config: Any) -> bool:
    """Whether ``config`` is a backend whose rows carry the key as ``id``."""
    return type(config).__name__ in _ID_MIRROR_BACKENDS


def is_well_formed(row: Any) -> bool:
    """The fields of the uniform row shape a reader interprets (*Row shape*
    in the contract): an integer ``schema_version``, a string ``workspace``
    and a mapping ``value``.

    ``updated_at`` / ``updated_by`` are diagnostic, read by no verdict, so a
    row lacking them is not refused: it is copied verbatim and verified by
    digest like every other, and the target serves it exactly as the source
    did.
    """
    if not isinstance(row, dict):
        return False
    version = row.get("schema_version")
    return (
        type(version) is int
        and isinstance(row.get("workspace"), str)
        and isinstance(row.get("value"), dict)
    )


def row_digest(payload: dict[str, Any]) -> str:
    """A content fingerprint independent of key order and backend round trip."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row_key(row: dict[str, Any]) -> str | None:
    key = row.get("_id", row.get("id"))
    return key if isinstance(key, str) else None


@dataclass
class ContainerScan:
    """One full, paged enumeration of a configuration container."""

    digests: dict[str, str] = field(default_factory=dict)
    # Rows per workspace, server-global rows (``server_keys``) excluded.
    scopes: dict[str, int] = field(default_factory=dict)
    # Well-formed rows under a registered server-global key, told apart by
    # key: a tenant may be named like the server prefix.
    server_keys: list[str] = field(default_factory=list)
    # The key of every row that is not a well-formed row (``None`` when the
    # row carries no usable key at all).
    malformed: list[str | None] = field(default_factory=list)
    # For every field some backend reserves, the keys of the well-formed rows
    # whose envelope carries it as their own.
    reserved: dict[str, list[str]] = field(default_factory=dict)
    # The ``workspace`` field of every well-formed workspace row, by key.
    scope_of: dict[str, str] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.digests)

    def malformed_names(self) -> str:
        return ", ".join(repr(key) for key in self.malformed)


async def scan_container(config: Any, *, page_size: int) -> ContainerScan:
    """Enumerate every data row (the identity row excluded) by digest.

    A row that is not a well-formed row is listed, never skipped. A backend
    failure mid-stream propagates: a partial listing is never a complete one.
    """
    identity_key = cs.storage_identity_key()
    server_keys = cs.server_config_keys()
    id_mirror = _mirrors_id(config)
    scan = ContainerScan()
    try:
        async for row in config.iter_rows(page_size=page_size):
            key = _row_key(row) if isinstance(row, dict) else None
            if key == identity_key:
                continue
            payload = (
                row_payload(row, id_mirror=id_mirror) if isinstance(row, dict) else None
            )
            if key is None or not is_well_formed(payload):
                scan.malformed.append(key)
                continue
            scan.digests[key] = row_digest(payload)
            for name in _RESERVED_FIELDS.intersection(payload):
                scan.reserved.setdefault(name, []).append(key)
            if key in server_keys:
                scan.server_keys.append(key)
                continue
            scope = payload["workspace"]
            scan.scopes[scope] = scan.scopes.get(scope, 0) + 1
            scan.scope_of[key] = scope
    except ConfigurationStorageError:
        raise
    except CorruptStorageRecordError as e:
        # One damaged row, named by the backend -- not a store that failed.
        # A refusal before the claim; steps 5-6 turn it into a failure.
        raise MigrationRefused(
            f"a row in the configuration container is not a configuration "
            f"row ({e}); repair or remove it and re-run"
        ) from e
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not enumerate the configuration container ({type(e).__name__}: {e})"
        ) from e
    return scan


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetVerdict:
    # "empty" -> claim it; "resume" -> it already holds THIS identity;
    # "foreign_identity" / "foreign_rows" -> refuse.
    state: str
    identity: str | None
    rows: int


async def classify_target(
    target: Any, storage_uuid: str, *, page_size: int
) -> TargetVerdict:
    identity = await cs.read_storage_identity(target)
    try:
        scan = await scan_container(target, page_size=page_size)
    except MigrationRefused as e:
        if identity != storage_uuid:
            raise
        # The enumeration died at the damaged record, so there is no complete
        # listing to converge from; but the target is this migration's own
        # residue, which the operator may simply clear.
        raise MigrationRefused(
            f"{e}. The target holds this migration's identity, so it is an "
            f"earlier attempt's residue: removing that row, or discarding the "
            f"whole target container, is safe -- the source is untouched and "
            f"the anchor unchanged."
        ) from e
    count = scan.rows + len(scan.malformed)
    if identity == storage_uuid:
        return TargetVerdict("resume", identity, count)
    if identity is not None:
        return TargetVerdict("foreign_identity", identity, count)
    if count:
        return TargetVerdict("foreign_rows", None, count)
    return TargetVerdict("empty", None, 0)


# ---------------------------------------------------------------------------
# The JSON group
# ---------------------------------------------------------------------------


def _workspace_label(workspace: str) -> str:
    return repr(workspace) if workspace else "(default workspace)"


class JsonShardGroup:
    """Every JSON configuration snapshot of one ``WORKING_DIR``, presented to
    the migration as ONE configuration container.

    Rows are routed by their ``workspace`` field to that workspace's
    snapshot; the identity is the one UUID every non-empty snapshot carries
    (a disagreement raises); owner rows are hidden -- read, validated and
    written here, never copied. Each snapshot's directory is claimed before
    it is opened, in sorted order, and the claims are handed back by the
    caller's ``release_claims``. See the module rules.
    """

    supports_strict_point_reads = True

    def __init__(self, working_dir: str, claims: list[str]) -> None:
        self.working_dir = os.path.abspath(working_dir)
        self._claims = claims
        self._shards: dict[str, Any] = {}
        # The identity a claimed-but-still-empty target will stamp into each
        # snapshot it creates.
        self._identity: str | None = None
        # For the migration report: the members a source opened, and the
        # snapshots a target found on disk / is asked to hold.
        self.source_members: tuple[str, ...] = ()
        self._on_disk: set[str] = set()
        self._planned: set[str] = set()

    # -- opening --------------------------------------------------------

    def _refuse(self, detail: str) -> MigrationRefused:
        return MigrationRefused(
            f"The JSON configuration group under {self.working_dir} {detail}. "
            f"Nothing was written."
        )

    def _validated_disk(self) -> dict[str, cs.ShardContents]:
        """Every non-empty snapshot on disk, validated against the workspace
        its location names. One holding records but no identity or owner row
        is damaged and refuses, as it does everywhere else: automatic
        recovery is limited to consistent state."""
        found: dict[str, cs.ShardContents] = {}
        try:
            for shard in discover_shards(self.working_dir):
                contents = cs.inspect_shard_rows(
                    read_shard_file(shard.path) or {},
                    workspace=shard.workspace,
                    location=shard.path,
                )
                if contents.empty:
                    continue
                if contents.storage_uuid is None or contents.owner is None:
                    raise cs.shard_invalid_error(
                        shard.path,
                        "it holds records but no "
                        + ("identity" if contents.storage_uuid is None else "owner")
                        + " row, so it cannot be attributed to a configuration group",
                    )
                found[shard.workspace] = contents
        except ConfigurationIdentityError as e:
            raise MigrationRefused(str(e)) from e
        return found

    def _refuse_shared_directories(self) -> None:
        """Refuse, before any claim, a planned workspace whose directory is
        one physical directory with another name's: a symlink to another
        workspace's directory or to ``WORKING_DIR`` (the default workspace's),
        or another spelling a case-insensitive filesystem resolves to it. Any
        existing child directory counts, with or without a snapshot -- the
        next discovery would find the snapshot through it too -- and the
        claim is reentrant by realpath, so nothing later stops both names
        opening one snapshot."""
        names = set(self._planned) | {""}
        try:
            entries = list(os.scandir(self.working_dir))
        except FileNotFoundError:
            entries = []
        except OSError as e:
            raise self._refuse(f"cannot be listed ({type(e).__name__}: {e})") from e
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=True):
                    names.add(entry.name)
            except OSError as e:
                raise self._refuse(
                    f"cannot inspect {entry.path} ({type(e).__name__}: {e})"
                ) from e
        physical: dict[tuple[int, int], list[str]] = {}
        for name in names:
            path = os.path.join(self.working_dir, name) if name else self.working_dir
            try:
                info = os.stat(path)
            except FileNotFoundError:
                continue  # Created by this run; the lexical check covers it.
            except OSError as e:
                raise self._refuse(
                    f"cannot inspect {path} ({type(e).__name__}: {e})"
                ) from e
            physical.setdefault((info.st_dev, info.st_ino), []).append(name)
        shared = sorted(
            sorted(group)
            for group in physical.values()
            if len(group) > 1 and self._planned.intersection(group)
        )
        if shared:
            raise self._refuse(
                "cannot give these names separate snapshots, because their "
                "directories are one physical directory (a symlink, or a "
                "spelling a case-insensitive filesystem resolves to it): "
                + "; ".join(
                    ", ".join(_workspace_label(n) for n in group) for group in shared
                )
                + ". Replace the symlinks with separate directories, or keep "
                "the configuration on a database backend"
            )

    async def _open(self, workspace: str) -> None:
        from lightrag.kg.json_kv_impl import JsonKVStorage
        from lightrag.kg.working_dir_lock import acquire_working_dir_lock

        directory = json_config_dir(self.working_dir, workspace)
        # A server on this WORKING_DIR could still hold the snapshot.
        acquire_working_dir_lock(directory)
        self._claims.append(directory)
        storage = cs.create_configuration_storage(
            JsonKVStorage,
            global_config={"working_dir": self.working_dir, "workspace": workspace},
            embedding_func=None,
        )
        self._shards[workspace] = await _initialize_or_close(
            storage, JSON_CONFIG_BACKEND
        )

    async def open_source(self) -> None:
        """Open the anchored group as a migration source: the anchor's
        members and the snapshots on disk must agree in both directions,
        and every member must carry the group's identity and its owner."""
        anchor = read_anchor(self.working_dir)
        if anchor is None or anchor.members is None:
            raise self._refuse("has no JSON anchor to name its members")
        on_disk = self._validated_disk()
        members = set(anchor.members)
        missing = sorted(members - set(on_disk))
        extra = sorted(set(on_disk) - members)
        if missing or extra:
            raise self._refuse(
                f"does not match its anchor: registered members without a "
                f"snapshot {missing}, snapshots that are not registered "
                f"{extra}. A missing member must be restored; an unregistered "
                f"snapshot is a lost registration or a stray file -- stop every "
                f"server, back up and delete the anchor and start one server "
                f"to rebuild the member list, or remove the stray file"
            )
        for workspace in sorted(members):
            contents = on_disk[workspace]
            if contents.storage_uuid != anchor.storage_uuid:
                raise self._refuse(
                    f"has member {workspace!r} with identity "
                    f"{contents.storage_uuid!r} and owner {contents.owner!r}, "
                    f"not the anchored group {anchor.storage_uuid}"
                )
        for workspace in sorted(members):
            await self._open(workspace)
        self.source_members = tuple(sorted(members))
        # Every member was just shown to carry the anchored UUID; a group
        # with no member (a database container that held only its identity,
        # or a first start stopped before registering) has no snapshot to
        # carry it, and its identity is still the anchor's.
        self._identity = anchor.storage_uuid

    async def prepare_target(
        self, source_scan: "ContainerScan", *, dry_run: bool
    ) -> None:
        """Claim, validate and open every snapshot a JSON target could touch:
        the source's workspaces and every snapshot already on disk. A dry run
        opens only the snapshots that exist."""
        scopes = set(source_scan.scopes)
        if source_scan.server_keys:
            raise self._refuse(
                "cannot hold the source's server-scope rows besides the identity "
                f"({', '.join(repr(k) for k in sorted(source_scan.server_keys))}): "
                "JSON snapshots are per workspace, and this version defines no "
                "rule for splitting a server-wide setting across them"
            )
        for scope in sorted(scopes):
            try:
                validate_config_workspace(scope)
            except ValueError as e:
                raise self._refuse(f"cannot hold source scope {scope!r}: {e}") from e
        # Every row must be a key a JSON snapshot accepts for its scope, or
        # the verification after the copy would refuse a snapshot this run
        # already wrote -- one the next run could then not converge.
        foreign = sorted(
            key
            for key, scope in source_scan.scope_of.items()
            if key not in cs.workspace_config_keys(scope)
        )
        if foreign:
            raise self._refuse(
                f"cannot hold {len(foreign)} source row(s) whose key is not a "
                f"registered key of the row's workspace: "
                + ", ".join(repr(key) for key in foreign[:10])
                + (", ..." if len(foreign) > 10 else "")
                + ". Repair or remove them in the source and re-run"
            )
        on_disk = self._validated_disk()
        self._on_disk = set(on_disk)
        self._planned = set(on_disk) | scopes
        # Distinct scopes that one case-insensitive, normalization-insensitive
        # (or trailing dot / space stripping) filesystem resolves to one
        # directory would share one snapshot: refused for the complete set,
        # before any claim -- and before any of them exists to compare.
        aliases: dict[str, list[str]] = {}
        for workspace in self._planned:
            aliases.setdefault(caseless_alias(workspace), []).append(workspace)
        colliding = sorted(
            sorted(names) for names in aliases.values() if len(names) > 1
        )
        if colliding:
            raise self._refuse(
                "cannot give these workspaces separate snapshots, because they "
                "differ only in letter case, Unicode normalization or trailing "
                "dots/spaces and a case- or normalization-insensitive "
                "filesystem stores them in one directory: "
                + "; ".join(", ".join(repr(n) for n in names) for names in colliding)
                + ". Keep the configuration on a database backend, or rename "
                "one of each group in the source"
            )
        self._refuse_shared_directories()
        wanted = set(on_disk) if dry_run else self._planned
        for workspace in sorted(wanted):
            await self._open(workspace)

    def describe_target(self, source_scan: "ContainerScan") -> list[str]:
        """The per-workspace plan a JSON target reports before it writes."""
        lines = [f"- JSON snapshots under {self.working_dir} ({len(self._planned)}):"]
        unknown: list[str] = []
        for workspace in sorted(self._planned):
            path = json_config_dir(self.working_dir, workspace)
            if workspace not in source_scan.scopes:
                what = "kept with identity and owner only (no rows in the source)"
            elif workspace in self._on_disk:
                what = "existing snapshot, converged to the source"
            else:
                what = "new snapshot"
                if workspace and not os.path.isdir(path):
                    unknown.append(workspace)
            lines.append(f"    {_workspace_label(workspace)}: {what}")
        if unknown:
            lines.append(
                "  Note: "
                + ", ".join(_workspace_label(ws) for ws in unknown)
                + (" has" if len(unknown) == 1 else " have")
                + " no directory under WORKING_DIR yet. If a workspace "
                "belongs to another deployment sharing the source container, "
                "its snapshot here is only a stale copy; the source keeps the "
                "live rows."
            )
        return lines

    # -- the container interface ----------------------------------------

    async def _identity_row(self) -> dict[str, Any] | None:
        identity_key = cs.storage_identity_key()
        rows: dict[str, dict[str, Any]] = {}
        for workspace, shard in sorted(self._shards.items()):
            row = await shard.get_by_id_strict(identity_key)
            if row is None:
                if not await shard.is_empty():
                    raise cs.shard_invalid_error(
                        json_config_dir(self.working_dir, workspace),
                        "it holds records but no identity row",
                    )
                continue
            rows[cs.identity_from_row(row, key=identity_key)] = row
        if len(rows) > 1:
            raise ConfigurationIdentityError(
                f"the JSON configuration snapshots under {self.working_dir} "
                f"carry different identities ({', '.join(sorted(rows))})",
                cause=cs.IDENTITY_SHARD_INVALID,
            )
        if rows:
            return next(iter(rows.values()))
        if self._identity is not None:
            return cs.make_config_row(
                scope_workspace=cs.SERVER_SCOPE,
                suffix=cs.STORAGE_IDENTITY_SUFFIX,
                value={"uuid": self._identity},
                updated_by=cs.UPDATED_BY_MIGRATE,
            )
        return None

    async def get_by_id_strict(self, key: str) -> dict[str, Any] | None:
        if key == cs.storage_identity_key():
            return await self._identity_row()
        if key == cs.json_shard_owner_key():
            return None
        for _, shard in sorted(self._shards.items()):
            row = await shard.get_by_id_strict(key)
            if row is not None:
                return row
        return None

    async def iter_rows(self, *, page_size: int = 200):
        hidden = {cs.storage_identity_key(), cs.json_shard_owner_key()}
        for _, shard in sorted(self._shards.items()):
            async for row in shard.iter_rows(page_size=page_size):
                if row.get("_id") not in hidden:
                    yield row
        identity = await self._identity_row()
        if identity is not None:
            yield {**identity, "_id": cs.storage_identity_key()}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        identity_key = cs.storage_identity_key()
        batches: dict[str, dict[str, dict[str, Any]]] = {}
        for key, row in data.items():
            if key == identity_key:
                self._identity = cs.identity_from_row(row, key=key)
                continue
            scope = row.get("workspace") if isinstance(row, dict) else None
            if not isinstance(scope, str) or scope not in self._shards:
                raise ConfigurationStorageError(
                    f"no JSON snapshot was prepared for row {key!r} (scope {scope!r})"
                )
            batches.setdefault(scope, {})[key] = dict(row)
        if not batches:
            return
        identity = await self._identity_row()
        if identity is None:
            raise ConfigurationStorageError(
                "the JSON target has no identity to stamp into a new snapshot"
            )
        storage_uuid = cs.identity_from_row(identity, key=identity_key)
        for workspace, batch in sorted(batches.items()):
            shard = self._shards[workspace]
            if await shard.get_by_id_strict(cs.json_shard_owner_key()) is None:
                batch.update(
                    cs.json_shard_metadata_rows(
                        workspace, storage_uuid, updated_by=cs.UPDATED_BY_MIGRATE
                    )
                )
            await shard.upsert(batch)

    async def delete(self, ids: list[str]) -> None:
        hidden = {cs.storage_identity_key(), cs.json_shard_owner_key()}
        keys = [key for key in ids if key not in hidden]
        for _, shard in sorted(self._shards.items()):
            await shard.delete(keys)

    async def index_done_callback(self) -> None:
        for _, shard in sorted(self._shards.items()):
            await shard.index_done_callback()

    async def finalize(self) -> None:
        for _, shard in sorted(self._shards.items()):
            await shard.finalize()

    # -- the commit -----------------------------------------------------

    async def committed_members(self) -> list[str]:
        """Every opened snapshot that now holds rows: the new member list."""
        return [
            workspace
            for workspace, shard in sorted(self._shards.items())
            if not await shard.is_empty()
        ]

    async def verify_layout(self, storage_uuid: str) -> None:
        """Every member snapshot carries ``storage_uuid`` and names its own
        workspace as owner."""
        for workspace in await self.committed_members():
            shard = self._shards[workspace]
            contents = cs.inspect_shard_rows(
                await cs.read_shard_rows(shard),
                workspace=workspace,
                location=json_config_dir(self.working_dir, workspace),
            )
            if contents.storage_uuid != storage_uuid or contents.owner != workspace:
                raise MigrationFailed(
                    f"the JSON snapshot of workspace {workspace!r} reads back "
                    f"identity {contents.storage_uuid!r} and owner "
                    f"{contents.owner!r}, not {storage_uuid!r} / {workspace!r}"
                )


# ---------------------------------------------------------------------------
# The migration
# ---------------------------------------------------------------------------


@dataclass
class MigrationResult:
    anchor: StorageAnchor
    target_backend: str
    source_scan: ContainerScan
    verdict: TargetVerdict
    switched: bool = False
    copied: int = 0
    deleted: int = 0


OpenStorage = Callable[[str], Awaitable[Any]]


def _no_anchor_message(working_dir: str) -> str:
    return (
        f"No configuration storage anchor at {anchor_path(working_dir)}, so "
        f"nothing is bound and there is nothing to migrate. Start the server "
        f"once on the current configuration to bind it, or -- if no records "
        f"need to move -- simply select the new backend."
    )


async def _copy_rows(
    source: Any,
    target: Any,
    *,
    source_scan: ContainerScan,
    page_size: int,
) -> tuple[int, int]:
    """Converge the target's data rows onto the source's: delete what the
    source does not hold or holds differently, flush, then upsert what is
    missing. The delete lands BEFORE the upsert, so no backend that merges an
    upsert into an existing document can keep a stale field.

    Only called once the target is proven to hold this identity, which is the
    sole reason deleting target rows is permitted.
    """
    target_scan = await scan_container(target, page_size=page_size)
    stale = [
        key
        for key, digest in target_scan.digests.items()
        if source_scan.digests.get(key) != digest
    ]
    # A damaged row left in the target by an earlier attempt is this
    # migration's too, and could never verify.
    stale.extend(key for key in target_scan.malformed if key is not None)
    if stale:
        try:
            await target.delete(stale)
        except Exception as e:
            raise MigrationFailed(
                f"could not delete {len(stale)} stale target row(s) "
                f"({type(e).__name__}: {e})"
            ) from e
        await cs.flush_configuration_storage(target, "the stale target rows")
    stale_keys = set(stale)
    kept = {k for k in target_scan.digests if k not in stale_keys}

    identity_key = cs.storage_identity_key()
    copied = 0
    batch: dict[str, dict[str, Any]] = {}

    async def _flush_batch() -> None:
        nonlocal copied
        if not batch:
            return
        try:
            await target.upsert(dict(batch))
        except Exception as e:
            raise MigrationFailed(
                f"could not write {len(batch)} row(s) to the target "
                f"({type(e).__name__}: {e})"
            ) from e
        copied += len(batch)
        batch.clear()

    source_id_mirror = _mirrors_id(source)
    try:
        async for row in source.iter_rows(page_size=page_size):
            key = _row_key(row) if isinstance(row, dict) else None
            if key == identity_key or key in kept:
                continue
            payload = (
                row_payload(row, id_mirror=source_id_mirror)
                if isinstance(row, dict)
                else None
            )
            if key is None or not is_well_formed(payload):
                raise MigrationRefused(
                    f"source row {key!r} is not a well-formed configuration row; "
                    f"repair it and re-run"
                )
            batch[key] = payload
            if len(batch) >= page_size:
                await _flush_batch()
    except (MigrationRefused, MigrationFailed):
        raise
    except Exception as e:
        # The target may already be claimed: this must reach the handler that
        # says the anchor is unchanged and a re-run resumes, not a traceback.
        raise MigrationFailed(
            f"could not enumerate the source during the copy ({type(e).__name__}: {e})"
        ) from e
    await _flush_batch()
    return copied, len(stale)


async def _verify(
    source: Any, target: Any, *, storage_uuid: str, page_size: int
) -> None:
    identity = await cs.read_storage_identity(target)
    if identity != storage_uuid:
        raise MigrationFailed(
            f"the target's identity reads back as {identity!r}, not {storage_uuid!r}"
        )
    source_scan = await scan_container(source, page_size=page_size)
    target_scan = await scan_container(target, page_size=page_size)
    if source_scan.malformed or target_scan.malformed:
        raise MigrationFailed(
            f"malformed rows during verification: source "
            f"[{source_scan.malformed_names()}], target "
            f"[{target_scan.malformed_names()}]"
        )
    if source_scan.digests != target_scan.digests:
        missing = sorted(set(source_scan.digests) - set(target_scan.digests))
        extra = sorted(set(target_scan.digests) - set(source_scan.digests))
        changed = sorted(
            k
            for k in set(source_scan.digests) & set(target_scan.digests)
            if source_scan.digests[k] != target_scan.digests[k]
        )
        raise MigrationFailed(
            f"the target does not match the source: missing {missing[:10]}, "
            f"unexpected {extra[:10]}, different {changed[:10]}"
            + (
                " (lists truncated)"
                if max(len(missing), len(extra), len(changed)) > 10
                else ""
            )
            + ". The source may have changed during the copy; re-run to reconcile."
        )


async def migrate_configuration(
    *,
    working_dir: str,
    target_backend: str,
    open_source: OpenStorage,
    open_target: OpenStorage,
    dry_run: bool = False,
    assume_exclusive: bool = False,
    page_size: int = DEFAULT_PAGE_SIZE,
    out: Callable[[str], None] = print,
    release_claims: Callable[[], None] = lambda: None,
    expected_anchor: StorageAnchor | None = None,
    current_workspace: str | None = None,
) -> MigrationResult:
    """Run the seven steps; raise ``MigrationRefused`` / ``MigrationFailed``
    (or a ``ConfigurationStorageError``) on anything short of "switched".

    ``open_source`` / ``open_target`` return an INITIALIZED configuration
    storage for a backend name; this function finalizes what it opened.
    ``dry_run`` takes the anchor lock shared, reads everything and writes
    nothing.
    """
    admitted = cs.configuration_storage_implementations()
    exclusive = None
    shared = False
    opened: list[Any] = []
    try:
        # Step 1. The lock, the anchor, the target type.
        if dry_run:
            acquire_anchor_lock_shared(working_dir)
            shared = True
        else:
            exclusive = acquire_anchor_lock_exclusive(
                working_dir, assume_exclusive=assume_exclusive
            )
        anchor = read_anchor(working_dir)
        if anchor is None:
            raise MigrationRefused(_no_anchor_message(working_dir))
        if expected_anchor is not None and anchor != expected_anchor:
            # The caller chose which connection settings are the source's
            # from the anchor it read before this lock; another migration
            # has moved it since, so those settings describe another backend.
            raise MigrationRefused(
                f"The anchor changed while this command was starting (it read "
                f"{expected_anchor.backend}, it now binds {anchor.backend}); "
                f"another migration ran. Nothing was written; re-run against "
                f"the current anchor."
            )
        if target_backend not in admitted:
            raise MigrationRefused(
                f"{target_backend!r} is not a configuration storage backend; "
                f"choose one of {', '.join(admitted)}"
            )
        if target_backend == anchor.backend:
            raise MigrationRefused(
                f"The target backend {target_backend} is the anchored one. A "
                f"same-type move is done with the backend's own dump/restore "
                f"(or, for JSON, by copying the whole WORKING_DIR): the identity "
                f"row travels with the data, and the same type and UUID pass "
                f"the start-up check."
            )
        out(f"- Anchor:  {anchor.backend}, identity {anchor.storage_uuid}")

        # Step 2. The source, strictly. A source that cannot be read is the
        # contract's refusal (nothing is claimed yet), not a resumable failure.
        try:
            source = await open_source(anchor.backend)
            opened.append(source)
            stored = await cs.read_storage_identity(source)
            source_scan = await scan_container(source, page_size=page_size)
        except ConfigurationStorageError as e:
            raise MigrationRefused(
                f"The source {anchor.backend} could not be read ({e}). Nothing "
                f"was written; restore or reconnect the source and re-run -- "
                f"the anchor is never moved without a readable source."
            ) from e
        if stored != anchor.storage_uuid:
            raise MigrationRefused(
                f"The source container ({anchor.backend}) holds identity "
                f"{stored!r}, but the anchor binds {anchor.storage_uuid}. The "
                f"source settings do not point at the anchored container; "
                f"the anchor is never moved without a readable source."
            )
        scopes = ", ".join(
            [
                f"{_workspace_label(scope)} ({n})"
                for scope, n in sorted(source_scan.scopes.items())
            ]
            + (
                [f"server scope ({len(source_scan.server_keys)})"]
                if source_scan.server_keys
                else []
            )
        )
        out(
            f"- Source:  {anchor.backend}, {source_scan.rows} row(s) besides the "
            f"identity; scopes: {scopes or '(none)'}"
        )
        workspaces = sorted(
            set(source_scan.scopes) | set(getattr(source, "source_members", ()))
        )
        out(
            "- Scope:   the WHOLE configuration container moves, every workspace "
            "below"
            + (
                f" -- not only this server's WORKSPACE "
                f"({_workspace_label(current_workspace)})"
                if current_workspace is not None
                else ""
            )
        )
        out(
            f"- Workspaces to migrate ({len(workspaces)}): "
            + (", ".join(_workspace_label(ws) for ws in workspaces) or "(none)")
        )
        members = getattr(source, "source_members", None)
        if members is not None:
            empty = sorted(set(members) - set(source_scan.scopes))
            out(
                f"  (every registered JSON snapshot under {working_dir}"
                + (
                    "; "
                    + ", ".join(_workspace_label(ws) for ws in empty)
                    + " hold no configuration rows, so nothing is copied for them"
                    if empty
                    else ""
                )
                + ")"
            )
        if source_scan.malformed:
            raise MigrationRefused(
                f"The source holds {len(source_scan.malformed)} row(s) that are "
                f"not well-formed configuration rows: "
                f"{source_scan.malformed_names()}. They are never skipped; "
                f"repair or remove them and re-run."
            )
        clashing = {
            name: keys
            for name, keys in source_scan.reserved.items()
            if name in RESERVED_FIELDS_BY_BACKEND.get(target_backend, ())
        }
        if clashing:
            # The target overwrites these fields on write and answers its own
            # value on read: the row's field could never be read back, nor
            # told apart from the backend's on a later migration out of it.
            listed = "; ".join(
                f"'{name}' in {len(keys)} row(s): "
                + ", ".join(repr(k) for k in keys[:10])
                + (", ..." if len(keys) > 10 else "")
                for name, keys in sorted(clashing.items())
            )
            raise MigrationRefused(
                f"Source rows carry fields {target_backend} cannot hold, because "
                f"it reserves them for its own use ({listed}). Nothing was "
                f"written; remove the fields or choose another target."
            )

        # Step 3. The target, classified. A JSON target first claims and
        # validates every snapshot it could touch -- the source's workspaces
        # and the ones already on disk -- so nothing is written before all
        # of them are known to be this migration's to converge.
        target = await open_target(target_backend)
        opened.append(target)
        prepare = getattr(target, "prepare_target", None)
        if prepare is not None:
            await prepare(source_scan, dry_run=dry_run)
            for line in target.describe_target(source_scan):
                out(line)
        verdict = await classify_target(
            target, anchor.storage_uuid, page_size=page_size
        )
        result = MigrationResult(
            anchor=anchor,
            target_backend=target_backend,
            source_scan=source_scan,
            verdict=verdict,
        )
        if verdict.state == "empty":
            out(f"- Target:  {target_backend}, empty; will claim it")
        elif verdict.state == "resume":
            out(
                f"- Target:  {target_backend} already holds this identity with "
                f"{verdict.rows} row(s); will reconcile"
            )
        elif verdict.state == "foreign_identity":
            raise MigrationRefused(
                f"The target {target_backend} holds another identity "
                f"({verdict.identity}) with {verdict.rows} row(s). It belongs to "
                f"another container; nothing is overwritten or merged. Use a "
                f"dedicated, empty target."
            )
        else:
            raise MigrationRefused(
                f"The target {target_backend} holds {verdict.rows} row(s) and "
                f"no identity. It is not empty and not this migration's; nothing "
                f"is overwritten or merged. Use a dedicated, empty target."
            )
        if dry_run:
            out(
                "- Dry run: no row was written. Opening the target provisions "
                "its container if it is missing (a table, collection or "
                "index), as any start on that backend does."
            )
            return result

        # Step 4. Claim: the ownership marker first.
        if verdict.state == "empty":
            try:
                await cs.write_storage_identity(
                    target, anchor.storage_uuid, updated_by=cs.UPDATED_BY_MIGRATE
                )
            except ConfigurationIdentityError as e:
                # Not a refusal: the marker may be durable even though the
                # client saw the write fail. A re-run classifies the target
                # again and resumes through it, or claims it afresh.
                raise MigrationFailed(
                    f"could not claim the target with this identity: {e}"
                ) from e

        try:
            # Step 5. Copy, converging the target onto the current source.
            result.copied, result.deleted = await _copy_rows(
                source, target, source_scan=source_scan, page_size=page_size
            )

            # Step 6. Strict flush, then verify every row against the source.
            await cs.flush_configuration_storage(target, "the migrated rows")
            await _verify(
                source, target, storage_uuid=anchor.storage_uuid, page_size=page_size
            )
            verify_layout = getattr(target, "verify_layout", None)
            if verify_layout is not None:
                await verify_layout(anchor.storage_uuid)
        except (MigrationRefused, ConfigurationIdentityError) as e:
            # The target is claimed and may already be partly converged, so
            # a row -- or the target's own identity -- that turned bad since
            # the claim is not "nothing was written": the anchor is unchanged
            # and a re-run reconciles or refuses by name.
            raise MigrationFailed(str(e)) from e

        # Step 7. The commit point. A JSON target lists every snapshot it
        # now holds as a member.
        members: tuple[str, ...] | None = None
        if target_backend == JSON_CONFIG_BACKEND:
            committed = getattr(target, "committed_members", None)
            members = tuple(await committed()) if committed is not None else ()
        new_anchor = StorageAnchor(
            backend=target_backend,
            storage_uuid=anchor.storage_uuid,
            members=members,
        )
        try:
            publish_anchor(working_dir, new_anchor, replace=True)
        except ConfigurationIdentityError as e:
            # Only the directory fsync can fail after the replace landed; the
            # anchor on disk is then the answer, and it must be reported
            # truthfully either way.
            try:
                landed = read_anchor(working_dir) == new_anchor
            except ConfigurationIdentityError as read_error:
                # The replace may have landed before the failure: asserting
                # either binding here would report a durable write that may
                # have happened as one that did not, or the reverse.
                raise MigrationIndeterminate(
                    f"the anchor switch raised ({e}) and the anchor could not "
                    f"be read back ({read_error})"
                ) from e
            if not landed:
                raise MigrationFailed(
                    f"the verified copy is complete but the anchor could not "
                    f"be switched: {e}"
                ) from e
            out(
                f"  (the anchor was switched, but its directory could not be "
                f"fsynced: {e})"
            )
        result.switched = True
        return result
    finally:
        for storage in reversed(opened):
            try:
                await storage.finalize()
            except Exception as e:  # pragma: no cover - best effort
                out(f"  (could not close a configuration storage cleanly: {e})")
        # The directory claims the openers took go back BEFORE the anchor
        # lock, the order every starter uses: a start admitted by the freed
        # anchor lock must not then be refused by a claim still held here.
        release_claims()
        if exclusive is not None:
            exclusive.release()
        if shared:
            release_anchor_lock_shared(working_dir)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _reads(backend: str, name: str) -> bool:
    return any(
        name == p or (p.endswith("_") and name.startswith(p))
        for p in BACKEND_ENV_PREFIXES.get(backend, ())
    )


def resolve_working_dir(env: Mapping[str, str]) -> str:
    """The WORKING_DIR the server resolves from ``env``: the default only
    when the key is unset. ``WORKING_DIR=`` is the empty string, which
    ``abspath`` makes the start directory -- where that server's anchor is."""
    return os.path.abspath(env.get("WORKING_DIR", DEFAULT_WORKING_DIR))


def resolve_environments(
    *,
    source_backend: str,
    target_backend: str,
    source_env: dict[str, str | None],
    target_env: dict[str, str | None],
    current: dict[str, str],
) -> dict[str, str]:
    """The variables to set so each backend reads ITS side's connection.

    Refuses when the two files set conflicting values for a variable either
    selected backend reads, and when either file names a different
    ``WORKING_DIR``: the anchor and its lock are resolved from the current
    environment only.
    """
    working_dir = resolve_working_dir(current)
    for label, env in (("--source-env", source_env), ("--target-env", target_env)):
        named = env.get("WORKING_DIR")
        if named is not None and os.path.abspath(named) != working_dir:
            raise MigrationRefused(
                f"{label} names WORKING_DIR={named}, but the anchor is resolved "
                f"from the current environment's WORKING_DIR ({working_dir}). "
                f"Run the tool where WORKING_DIR is the deployment's."
            )
    conflicts = sorted(
        name
        for name in set(source_env) & set(target_env)
        if (_reads(source_backend, name) or _reads(target_backend, name))
        and source_env[name] != target_env[name]
    )
    if conflicts:
        raise MigrationRefused(
            f"--source-env and --target-env set different values for "
            f"{', '.join(conflicts)}, which a selected backend reads; one "
            f"process cannot give each side its own value. Remove the conflict."
        )
    overlay: dict[str, str] = {}
    for backend, env in ((source_backend, source_env), (target_backend, target_env)):
        for name, value in env.items():
            if value is not None and _reads(backend, name):
                overlay[name] = value
    return overlay


def _opener(working_dir: str, claims: list[str], *, source: bool) -> OpenStorage:
    async def _open(backend: str) -> Any:
        from lightrag.kg.factory import get_storage_class

        if backend in cs.FILE_BACKED_CONFIG_STORAGES:
            group = JsonShardGroup(working_dir, claims)
            if source:
                await group.open_source()
            return group
        try:
            # Resolution and construction fail on a missing driver or an
            # environment value the backend rejects; nothing is open yet.
            storage = cs.create_configuration_storage(
                get_storage_class(backend),
                global_config={"working_dir": working_dir, "kv_storage": backend},
                embedding_func=None,
            )
        except (ConfigurationStorageError, ConfigurationIdentityError):
            raise
        except Exception as e:
            raise ConfigurationStorageError(
                f"could not create the {backend} configuration storage "
                f"({type(e).__name__}: {e})"
            ) from e
        return await _initialize_or_close(storage, backend)

    return _open


async def _initialize_or_close(storage: Any, backend: str) -> Any:
    try:
        await storage.initialize()
    except BaseException as e:
        # The caller never receives a storage that failed to open, so it is
        # closed here: a client acquired before the failure must not leak.
        try:
            await storage.finalize()
        except Exception:
            pass
        if not isinstance(e, Exception) or isinstance(
            e, (ConfigurationStorageError, ConfigurationIdentityError)
        ):
            raise
        raise ConfigurationStorageError(
            f"could not open the {backend} configuration storage "
            f"({type(e).__name__}: {e})"
        ) from e
    return storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lightrag-migrate-config",
        description=(
            "Move the configuration container to a different backend type, "
            "offline. Every server, SDK process and maintenance tool using "
            "this WORKING_DIR must be stopped first. See "
            "lightrag/tools/README_MIGRATE_CONFIG.md."
        ),
    )
    parser.add_argument(
        "--target-backend",
        required=True,
        help="JsonKVStorage, PGKVStorage, MongoKVStorage or OpenSearchKVStorage; "
        "must differ from the anchored backend",
    )
    parser.add_argument(
        "--source-env",
        help="env file with the SOURCE connection (default: the current environment)",
    )
    parser.add_argument(
        "--target-env",
        help="env file with the TARGET connection (default: the current environment)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="read and report; write nothing"
    )
    parser.add_argument(
        "--assume-exclusive",
        action="store_true",
        help="proceed where the anchor lock cannot be taken, asserting that "
        "every reader and writer has been stopped",
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not ask for confirmation"
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    return parser


def _release_claims(claims: list[str]) -> None:
    from lightrag.kg.working_dir_lock import release_working_dir_lock

    while claims:
        release_working_dir_lock(claims.pop())


def _load_env_file(path: str | None) -> dict[str, str | None]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise MigrationRefused(f"env file {path!r} does not exist")
    return dict(dotenv_values(path))


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    working_dir = resolve_working_dir(os.environ)
    claims: list[str] = []
    initialize_share_data(workers=1)
    try:
        source_env = _load_env_file(args.source_env)
        target_env = _load_env_file(args.target_env)
        # Read here only to know which connection variables are the source's
        # and to refuse before asking anything; the migration re-reads it
        # under the exclusive lock.
        anchor = read_anchor(working_dir)
        if anchor is None:
            raise MigrationRefused(_no_anchor_message(working_dir))
        overlay = resolve_environments(
            source_backend=anchor.backend,
            target_backend=args.target_backend,
            source_env=source_env,
            target_env=target_env,
            current=dict(os.environ),
        )
        os.environ.update(overlay)
        print("\nlightrag-migrate-config")
        print(f"- Working dir: {working_dir}")
        current_workspace = normalize_server_workspace(os.environ.get("WORKSPACE", ""))
        if not args.dry_run and not args.yes:
            # The inventory first, read-only, so the operator confirms the
            # whole list -- the migration is never only this server's
            # workspace.
            print("\nWhat would be migrated (read-only preview):")
            await migrate_configuration(
                working_dir=working_dir,
                target_backend=args.target_backend,
                open_source=_opener(working_dir, claims, source=True),
                open_target=_opener(working_dir, claims, source=False),
                dry_run=True,
                page_size=max(1, args.page_size),
                release_claims=lambda: _release_claims(claims),
                expected_anchor=anchor,
                current_workspace=current_workspace or "",
            )
            print(
                "\nEvery workspace listed above is migrated, not only this "
                "server's. Every server, SDK process and maintenance tool using "
                "this working directory (and every deployment sharing the "
                "source container) must be stopped."
            )
            answer = input("Type 'migrate' to continue: ").strip()
            if answer != "migrate":
                print("Cancelled; nothing was written.")
                return 0
        result = await migrate_configuration(
            working_dir=working_dir,
            target_backend=args.target_backend,
            open_source=_opener(working_dir, claims, source=True),
            open_target=_opener(working_dir, claims, source=False),
            dry_run=args.dry_run,
            assume_exclusive=args.assume_exclusive,
            page_size=max(1, args.page_size),
            release_claims=lambda: _release_claims(claims),
            # The overlay above chose the source's settings from this anchor.
            expected_anchor=anchor,
            current_workspace=current_workspace or "",
        )
    except (
        MigrationRefused,
        ConfigurationAnchorLockError,
        ConfigurationIdentityError,
        WorkingDirectoryInUseError,
    ) as e:
        print(f"\n✗ Refused: {e}")
        return 1
    except (MigrationFailed, ConfigurationStorageError) as e:
        print(
            f"\n✗ Migration failed: {e}\n  The anchor is unchanged: the current "
            f"configuration still works on the source. Re-run to resume."
        )
        return 1
    except MigrationIndeterminate as e:
        print(
            f"\n✗ Outcome unknown: {e}\n  The verified copy is complete, but the "
            f"anchor {anchor_path(working_dir)} may bind either backend. Start "
            f"nothing until it reads back: if it names {args.target_backend}, "
            f"the migration is complete; if it still names the source, re-run."
        )
        return 1
    finally:
        # Normally already given back inside the migration, before its anchor
        # lock; this covers a failure before it was reached.
        _release_claims(claims)
        finalize_share_data()

    if args.dry_run:
        return 0
    # Names only, never values: this tool never logs a credential.
    # What --target-env supplied for the target backend exists only in this
    # process; the deployment environment has to carry it too.
    target_settings = sorted(
        name
        for name, value in target_env.items()
        if value is not None and _reads(args.target_backend, name)
    )
    print(
        f"\n✓ Switched: the anchor now binds {result.target_backend} (identity "
        f"{result.anchor.storage_uuid}); {result.copied} row(s) written, "
        f"{result.deleted} stale row(s) removed, every row verified.\n"
        f"  Next: set LIGHTRAG_CONFIG_STORAGE={result.target_backend} explicitly "
        f"(this tool never edits .env)"
        + (
            f", and apply the target connection settings "
            f"({', '.join(target_settings)}) from --target-env to the "
            f"deployment environment -- without them the server opens a "
            f"different {result.target_backend} container and is refused"
            if target_settings
            else ""
        )
        + ". Then start the server.\n"
        f"  The source {result.anchor.backend} container was kept; removing it "
        f"is a separate, explicit step."
    )
    return 0


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    setup_logger("lightrag", level="INFO")
    raise SystemExit(asyncio.run(async_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
