#!/usr/bin/env python3
"""``lightrag-migrate-config``: move the configuration container to a
different backend TYPE, offline, and move the anchor only once the copy is
verified.

Full contract: *Offline migration* under *The anchor and the container
identity* in ``docs/design/ConfigurationStorage.md``; operator guide:
``lightrag/tools/README_MIGRATE_CONFIG.md``. The rules:

* **Cross-type only.** A same-type move (PostgreSQL to PostgreSQL, Mongo to
  Mongo, an OpenSearch snapshot, copying the JSON file) is the backend's own
  dump/restore: the identity row travels with the data and "same type, same
  UUID" passes. The PostgreSQL, MongoDB and OpenSearch client managers are
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
  keeps working on the source. The source is never modified or deleted, and
  no ``.env`` or environment is ever written.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from dotenv import dotenv_values, load_dotenv

from lightrag import config_store as cs
from lightrag.config_anchor import (
    StorageAnchor,
    anchor_path,
    publish_anchor,
    read_anchor,
)
from lightrag.constants import DEFAULT_WORKING_DIR
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationIdentityError,
    ConfigurationStorageError,
    WorkingDirectoryInUseError,
)
from lightrag.kg.anchor_lock import (
    acquire_anchor_lock_exclusive,
    acquire_anchor_lock_shared,
    release_anchor_lock_shared,
)
from lightrag.utils import setup_logger

# Fields a backend adds to a row it returns and owns itself; never content.
BACKEND_METADATA_KEYS = frozenset(
    {"_id", "id", "create_time", "update_time", "__mirrored_id"}
)

# The environment variables each configuration backend reads, by prefix (or
# exact name). Two backends of different types read disjoint sets, which is
# what lets one process hold both connections.
BACKEND_ENV_PREFIXES: dict[str, tuple[str, ...]] = {
    "JsonKVStorage": ("LIGHTRAG_CONFIG_DIR",),
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


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """The row envelope, verbatim, without the backend-owned metadata."""
    return {k: v for k, v in row.items() if k not in BACKEND_METADATA_KEYS}


def is_well_formed(row: Any) -> bool:
    """The uniform row shape (*Row shape* in the contract)."""
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
    scopes: dict[str, int] = field(default_factory=dict)
    # The key of every row that is not a well-formed row (``None`` when the
    # row carries no usable key at all).
    malformed: list[str | None] = field(default_factory=list)

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
    scan = ContainerScan()
    try:
        async for row in config.iter_rows(page_size=page_size):
            key = _row_key(row) if isinstance(row, dict) else None
            if key == identity_key:
                continue
            if key is None or not is_well_formed(row_payload(row)):
                scan.malformed.append(key)
                continue
            payload = row_payload(row)
            scan.digests[key] = row_digest(payload)
            scope = payload["workspace"]
            scan.scopes[scope] = scan.scopes.get(scope, 0) + 1
    except ConfigurationStorageError:
        raise
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
    scan = await scan_container(target, page_size=page_size)
    count = scan.rows + len(scan.malformed)
    if identity == storage_uuid:
        return TargetVerdict("resume", identity, count)
    if identity is not None:
        return TargetVerdict("foreign_identity", identity, count)
    if count:
        return TargetVerdict("foreign_rows", None, count)
    return TargetVerdict("empty", None, 0)


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
    kept = {k for k in target_scan.digests if k not in set(stale)}

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

    async for row in source.iter_rows(page_size=page_size):
        key = _row_key(row) if isinstance(row, dict) else None
        if key == identity_key or key in kept:
            continue
        payload = row_payload(row) if isinstance(row, dict) else None
        if key is None or not is_well_formed(payload):
            raise MigrationRefused(
                f"source row {key!r} is not a well-formed configuration row; "
                f"repair it and re-run"
            )
        batch[key] = payload
        if len(batch) >= page_size:
            await _flush_batch()
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
        if target_backend not in admitted:
            raise MigrationRefused(
                f"{target_backend!r} is not a configuration storage backend; "
                f"choose one of {', '.join(admitted)}"
            )
        if target_backend == anchor.backend:
            raise MigrationRefused(
                f"The target backend {target_backend} is the anchored one. A "
                f"same-type move is done with the backend's own dump/restore "
                f"(or by copying the JSON file): the identity row travels with "
                f"the data, and the same type and UUID pass the start-up check."
            )
        out(f"- Anchor:  {anchor.backend}, identity {anchor.storage_uuid}")

        # Step 2. The source, strictly.
        source = await open_source(anchor.backend)
        opened.append(source)
        stored = await cs.read_storage_identity(source)
        if stored != anchor.storage_uuid:
            raise MigrationRefused(
                f"The source container ({anchor.backend}) holds identity "
                f"{stored!r}, but the anchor binds {anchor.storage_uuid}. The "
                f"source settings do not point at the anchored container; "
                f"the anchor is never moved without a readable source."
            )
        source_scan = await scan_container(source, page_size=page_size)
        scopes = ", ".join(
            f"{scope} ({n})" for scope, n in sorted(source_scan.scopes.items())
        )
        out(
            f"- Source:  {anchor.backend}, {source_scan.rows} row(s) besides the "
            f"identity; scopes: {scopes or '(none)'}"
        )
        if source_scan.malformed:
            raise MigrationRefused(
                f"The source holds {len(source_scan.malformed)} row(s) that are "
                f"not well-formed configuration rows: "
                f"{source_scan.malformed_names()}. They are never skipped; "
                f"repair or remove them and re-run."
            )

        # Step 3. The target, classified.
        target = await open_target(target_backend)
        opened.append(target)
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
            out("- Dry run: nothing was written.")
            return result

        # Step 4. Claim: the ownership marker first.
        if verdict.state == "empty":
            await cs.write_storage_identity(
                target, anchor.storage_uuid, updated_by=cs.UPDATED_BY_MIGRATE
            )

        # Step 5. Copy, converging the target onto the current source.
        result.copied, result.deleted = await _copy_rows(
            source, target, source_scan=source_scan, page_size=page_size
        )

        # Step 6. Strict flush, then verify every row against the source.
        await cs.flush_configuration_storage(target, "the migrated rows")
        await _verify(
            source, target, storage_uuid=anchor.storage_uuid, page_size=page_size
        )

        # Step 7. The commit point.
        new_anchor = StorageAnchor(
            backend=target_backend, storage_uuid=anchor.storage_uuid
        )
        try:
            publish_anchor(working_dir, new_anchor, replace=True)
        except ConfigurationIdentityError as e:
            # Only the directory fsync can fail after the replace landed; the
            # anchor on disk is then the answer, and it must be reported
            # truthfully either way.
            try:
                landed = read_anchor(working_dir) == new_anchor
            except ConfigurationIdentityError:
                landed = False
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
    working_dir = os.path.abspath(current.get("WORKING_DIR") or DEFAULT_WORKING_DIR)
    for label, env in (("--source-env", source_env), ("--target-env", target_env)):
        named = env.get("WORKING_DIR")
        if named and os.path.abspath(named) != working_dir:
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


def _config_dir_for(env: dict[str, str | None], working_dir: str) -> str:
    named = env.get("LIGHTRAG_CONFIG_DIR")
    if named is None:
        named = os.environ.get("LIGHTRAG_CONFIG_DIR", "")
    return cs.resolve_config_dir(named, working_dir)


def _opener(working_dir: str, config_dir: str, claims: list[str]) -> OpenStorage:
    async def _open(backend: str) -> Any:
        from lightrag.kg.factory import get_storage_class
        from lightrag.kg.working_dir_lock import acquire_working_dir_lock

        if backend in cs.FILE_BACKED_CONFIG_STORAGES:
            # A server on another WORKING_DIR could still hold this directory.
            acquire_working_dir_lock(config_dir)
            claims.append(config_dir)
        storage = cs.create_configuration_storage(
            get_storage_class(backend),
            global_config={
                "working_dir": working_dir,
                "config_dir": config_dir,
                "kv_storage": backend,
            },
            embedding_func=None,
        )
        await storage.initialize()
        return storage

    return _open


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


def _load_env_file(path: str | None) -> dict[str, str | None]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise MigrationRefused(f"env file {path!r} does not exist")
    return dict(dotenv_values(path))


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
    from lightrag.kg.working_dir_lock import release_working_dir_lock

    working_dir = os.path.abspath(os.environ.get("WORKING_DIR") or DEFAULT_WORKING_DIR)
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
        if not args.dry_run and not args.yes:
            print(
                "\nEvery server, SDK process and maintenance tool using this "
                "working directory (and every deployment sharing the source "
                "container) must be stopped."
            )
            answer = input("Type 'migrate' to continue: ").strip()
            if answer != "migrate":
                print("Cancelled; nothing was written.")
                return 0
        result = await migrate_configuration(
            working_dir=working_dir,
            target_backend=args.target_backend,
            open_source=_opener(
                working_dir, _config_dir_for(source_env, working_dir), claims
            ),
            open_target=_opener(
                working_dir, _config_dir_for(target_env, working_dir), claims
            ),
            dry_run=args.dry_run,
            assume_exclusive=args.assume_exclusive,
            page_size=max(1, args.page_size),
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
    finally:
        for config_dir in claims:
            release_working_dir_lock(config_dir)
        finalize_share_data()

    if args.dry_run:
        return 0
    print(
        f"\n✓ Switched: the anchor now binds {result.target_backend} (identity "
        f"{result.anchor.storage_uuid}); {result.copied} row(s) written, "
        f"{result.deleted} stale row(s) removed, every row verified.\n"
        f"  Next: set LIGHTRAG_CONFIG_STORAGE={result.target_backend} explicitly "
        f"(this tool never edits .env), then start the server.\n"
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
