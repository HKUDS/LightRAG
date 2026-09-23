"""The configuration storage: one place, independent of any knowledge-base
workspace, for the server's settings and every workspace's.

Full contract: ``docs/design/ConfigurationStorageContract.md``. Read it before adding
a key, before binding a storage to the configuration workspace, and before
moving anything out of an environment variable. The rules this module
enforces, in the order a caller meets them:

* **The container** is the KV namespace ``config`` in a container named in
  CODE, never derived from a workspace: the JSON backend puts its file in
  ``config_dir``, and PostgreSQL, MongoDB and OpenSearch name their table,
  collection and index after ``CONFIG_CONTAINER_TAG``. No ``*_WORKSPACE``
  variable reaches it, and no caller can name a workspace that lands on it.
  ``create_configuration_storage()`` is still the single door in.

* **The backend is its own category.** ``CONFIG_STORAGE`` admits
  ``JsonKVStorage``, ``MongoKVStorage``, ``PGKVStorage`` and
  ``OpenSearchKVStorage``; anything else -- a vector storage, Redis -- is
  refused at construction, by name. Unset, the selection follows
  ``kv_storage`` so an existing deployment lands where its rows already are.

* **Keys** are ``<workspace>/<suffix>`` or ``_lightrag_server/<suffix>``, with
  ``/`` as the separator (a workspace name cannot contain it). They are built
  by ``config_key`` and **never reparsed**: the row carries ``workspace`` as a
  field, and a reader classifies by the field. That scope is the CONTAINER's
  only discriminator -- two deployments sharing one PostgreSQL, MongoDB or
  OpenSearch share the container and are kept apart by their business
  workspace names, which the contract already requires to differ.

* **Every key is registered** in ``CONFIG_KEY_REGISTRY`` before it is written.
  An unregistered suffix is a programming error, not a runtime condition.

* **Reads are strict.** A record that decides whether the instance may serve
  is read with ``get_by_id_strict``; a read that could not complete raises
  ``ConfigurationStorageError`` and is never mistaken for absence.

* **A missing baseline is claimed atomically** under a keyed lock, with the
  flush and a strict read-back inside the lock. The claim covers workers of
  one Gunicorn master and nothing wider.

* **The container has an identity.** ``_lightrag_server/storage_identity``
  holds a UUID for the whole container, and the anchor file
  (``lightrag/config_anchor.py``) records which backend and UUID this
  deployment is bound to. ``bind_configuration_identity`` checks -- or, with
  no anchor, establishes -- that binding before any baseline is read; the
  row is never overwritten once valid and never deleted by workspace
  maintenance.

The first per-workspace keys are the three per-target embedding baselines.
What a baseline MEANS -- the space adopted for that target, not the space its
vectors were written in -- and why there are three of them is in the
contract; the ``origin`` field records how each claim was established and
never enters a verdict.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable

from lightrag.config_anchor import (
    IDENTITY_BACKEND_MISMATCH,
    IDENTITY_ROW_INVALID,
    IDENTITY_UUID_MISMATCH,
    IDENTITY_UUID_MISSING,
    IDENTITY_WRITE_FAILED,
    StorageAnchor,
    anchor_path,
    canonical_storage_uuid,
    new_storage_uuid,
    publish_anchor,
    read_anchor,
)
from lightrag.exceptions import (
    ConfigurationIdentityError,
    ConfigurationRecordMalformedError,
    ConfigurationStorageError,
    CorruptStorageRecordError,
    EmbeddingBaselineMismatchError,
    ReferencesIntactFlushError,
)
from lightrag.kg.vector_space import declared_dimension, declared_model_name
from lightrag.namespace import (
    CONFIG_CONTAINER_TAG,
    SERVER_CONFIG_SCOPE,
    SERVER_SCOPE,
    NameSpace,
    _ServerScope,
    default_config_dir,
)
from lightrag.utils import logger

# The only separator a key may carry between its scope and its suffix. Load
# bearing: ``validate_workspace`` forbids ``/`` in a workspace name, so a key
# can always be told apart from a name -- and it is still never reparsed.
KEY_SEPARATOR = "/"

# The three vector targets, in the order the rebuild tool rebuilds them. One
# baseline each: they are separate containers with separate histories.
EMBEDDING_TARGETS: tuple[str, ...] = (
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
    NameSpace.VECTOR_STORE_CHUNKS,
)

# The keyed-lock namespace a baseline claim is made under. Per key, so two
# workspaces (or two targets) never wait on each other.
CLAIM_LOCK_NAMESPACE = "configuration_embedding_claim"

# The keyed-lock namespace the identity bind runs under. One key: the
# container has one identity, and the lock spans the whole read-decide-write.
IDENTITY_LOCK_NAMESPACE = "configuration_identity"
IDENTITY_LOCK_KEY = "storage_identity"

# ``updated_by`` values the two writers stamp. Diagnostic only.
UPDATED_BY_STARTUP = "lightrag.initialize_storages"
UPDATED_BY_REBUILD = "lightrag-rebuild-vdb"

# The two callers that only ever DELETE these rows, named in the registry's
# ``writers`` so the declared list stays the whole list. Neither stamps an
# ``updated_by``: a delete leaves no row to carry one.
DELETED_BY_CLEAR_ENDPOINT = "/documents/clear"
DELETED_BY_CLEAR_TOOL = "lightrag-clear-storage"

# The offline migration copies EVERY row verbatim into another container and
# writes that container's identity, so it reads and writes every key.
UPDATED_BY_MIGRATE = "lightrag-migrate-config"


class ConfigScope(str, Enum):
    """Whether a key is filed under a workspace or under the server itself."""

    WORKSPACE = "workspace"
    SERVER = "server"


@dataclass(frozen=True)
class ConfigKeySpec:
    """One registry entry: the five fields the contract requires of a key."""

    suffix: str
    scope: ConfigScope
    schema_version: int
    schema: str
    readers: tuple[str, ...]
    writers: tuple[str, ...]
    sensitive: bool


class BaselineOrigin:
    """How a baseline claim was established. Diagnostic; never in a verdict."""

    PROBE = "probe"
    EMPTY = "empty"
    REBUILD = "rebuild"

    ALL = frozenset({PROBE, EMPTY, REBUILD})


def embedding_baseline_suffix(target: str) -> str:
    """The key suffix of one target's embedding baseline: ``embedding/<target>``."""
    if target not in EMBEDDING_TARGETS:
        raise ValueError(
            f"{target!r} is not a vector target; expected one of {EMBEDDING_TARGETS}"
        )
    return f"embedding{KEY_SEPARATOR}{target}"


_EMBEDDING_BASELINE_SCHEMA = (
    "{model: str (unfolded, as configured), dim: int | null, origin: one of "
    "probe | empty | rebuild}"
)

# The container's own identity: one row, server scope, shared by every
# workspace the container holds.
STORAGE_IDENTITY_SUFFIX = "storage_identity"

CONFIG_KEY_REGISTRY: dict[str, ConfigKeySpec] = {
    STORAGE_IDENTITY_SUFFIX: ConfigKeySpec(
        suffix=STORAGE_IDENTITY_SUFFIX,
        scope=ConfigScope.SERVER,
        schema_version=1,
        schema="{uuid: str (canonical UUID of the whole configuration container)}",
        # Written by a start that binds with no anchor on record, and by the
        # migration into its (empty) target; never overwritten once valid,
        # never deleted by workspace clear/delete or an embedding rebuild.
        # The rebuild and clear tools only VERIFY it.
        readers=(
            UPDATED_BY_STARTUP,
            UPDATED_BY_REBUILD,
            DELETED_BY_CLEAR_TOOL,
            UPDATED_BY_MIGRATE,
        ),
        writers=(UPDATED_BY_STARTUP, UPDATED_BY_MIGRATE),
        sensitive=False,
    ),
    **{
        embedding_baseline_suffix(target): ConfigKeySpec(
            suffix=embedding_baseline_suffix(target),
            scope=ConfigScope.WORKSPACE,
            schema_version=1,
            schema=_EMBEDDING_BASELINE_SCHEMA,
            readers=(
                UPDATED_BY_STARTUP,
                UPDATED_BY_REBUILD,
                DELETED_BY_CLEAR_TOOL,
                UPDATED_BY_MIGRATE,
            ),
            writers=(
                UPDATED_BY_STARTUP,
                UPDATED_BY_REBUILD,
                DELETED_BY_CLEAR_ENDPOINT,
                DELETED_BY_CLEAR_TOOL,
                UPDATED_BY_MIGRATE,
            ),
            sensitive=False,
        )
        for target in EMBEDDING_TARGETS
    },
}
"""Every key this namespace may hold. A write to an unregistered suffix is
refused by ``make_config_row``; a general namespace without a registry becomes
a junk drawer."""


def registry_spec(suffix: str) -> ConfigKeySpec:
    """The registry entry for ``suffix``; ``KeyError`` when it is not registered."""
    try:
        return CONFIG_KEY_REGISTRY[suffix]
    except KeyError:
        raise KeyError(
            f"configuration key suffix {suffix!r} is not registered in "
            f"CONFIG_KEY_REGISTRY; declare it (scope, schema, readers/writers, "
            f"sensitive) before writing it"
        ) from None


def scope_prefix(scope_workspace: str | _ServerScope) -> str:
    """What a scope is written as in a key and in the row's ``workspace``
    field: a workspace's own name, or the server prefix for ``SERVER_SCOPE``."""
    return SERVER_CONFIG_SCOPE if scope_workspace is SERVER_SCOPE else scope_workspace


def config_key(scope_workspace: str | _ServerScope, suffix: str) -> str:
    """Build a key. ``scope_workspace`` is a workspace NAME or ``SERVER_SCOPE``;
    the suffix must be registered.

    A workspace named like the server prefix is an ordinary tenant and gets
    its own per-workspace keys: the scope is the sentinel object, never the
    string, so no name can reach a server-global key (see ``_ServerScope``).
    """
    spec = registry_spec(suffix)
    is_server = scope_workspace is SERVER_SCOPE
    if spec.scope is ConfigScope.SERVER and not is_server:
        raise ValueError(
            f"{suffix!r} is a server-global key and must be filed under "
            f"SERVER_SCOPE, not {scope_workspace!r}"
        )
    if spec.scope is ConfigScope.WORKSPACE and is_server:
        raise ValueError(
            f"{suffix!r} is a per-workspace key; SERVER_SCOPE is not a workspace"
        )
    prefix = scope_prefix(scope_workspace)
    if KEY_SEPARATOR in prefix:
        raise ValueError(
            f"a workspace name cannot contain {KEY_SEPARATOR!r}: {prefix!r}"
        )
    return f"{prefix}{KEY_SEPARATOR}{suffix}"


def embedding_baseline_key(workspace: str, target: str) -> str:
    """``<workspace>/embedding/<target>``."""
    return config_key(workspace, embedding_baseline_suffix(target))


def make_config_row(
    *,
    scope_workspace: str | _ServerScope,
    suffix: str,
    value: dict[str, Any],
    updated_by: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The uniform row every configuration key is stored as.

    ``schema_version`` is the registry's for this suffix; ``workspace`` is the
    scope the row is ABOUT (a tenant, or ``_lightrag_server``), carried as a
    field so enumeration never has to reparse the key.
    """
    spec = registry_spec(suffix)
    if not isinstance(value, dict):
        raise TypeError(
            f"a configuration value is a mapping, got {type(value).__name__}"
        )
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "schema_version": spec.schema_version,
        "workspace": scope_prefix(scope_workspace),
        "updated_at": stamp,
        "updated_by": updated_by,
        "value": dict(value),
    }


# ---------------------------------------------------------------------------
# The embedding baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingBaseline:
    """The embedding space adopted as one target's active baseline."""

    model: str
    dim: int | None
    origin: str

    def to_value(self) -> dict[str, Any]:
        return {"model": self.model, "dim": self.dim, "origin": self.origin}

    @classmethod
    def from_row(
        cls, row: dict[str, Any], *, key: str, target: str
    ) -> "EmbeddingBaseline":
        """Parse a stored row. A row that does not parse is UNREADABLE, which
        is a startup failure, not an absence."""
        expected_version = registry_spec(
            embedding_baseline_suffix(target)
        ).schema_version
        version = row.get("schema_version") if isinstance(row, dict) else None
        if type(version) is not int or version != expected_version:
            raise ConfigurationStorageError(
                f"configuration row {key!r} has unsupported schema_version "
                f"{version!r}; expected integer {expected_version}"
            )
        value = row.get("value") if isinstance(row, dict) else None
        if not isinstance(value, dict):
            raise ConfigurationStorageError(
                f"configuration row {key!r} carries no value mapping: {row!r}"
            )
        model = value.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationStorageError(
                f"configuration row {key!r} records no embedding model: {value!r}"
            )
        dim = value.get("dim")
        if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int)):
            raise ConfigurationStorageError(
                f"configuration row {key!r} records a non-integer dimension: {dim!r}"
            )
        origin = value.get("origin")
        if not isinstance(origin, str):
            origin = "unknown"
        return cls(model=model.strip(), dim=dim, origin=origin)

    def differs_from(self, embedding_func: Any) -> bool:
        """Whether this baseline contradicts the configured embedding function.

        The model must be equal. The dimension is compared only when both
        sides declare one: absent evidence never refuses.
        """
        expected_model = declared_model_name(embedding_func)
        if expected_model is not None and expected_model != self.model:
            return True
        expected_dim = declared_dimension(embedding_func)
        if (
            expected_dim is not None
            and self.dim is not None
            and expected_dim != self.dim
        ):
            return True
        return False


def configured_baseline(
    embedding_func: Any, *, origin: str
) -> EmbeddingBaseline | None:
    """The baseline this process would record, or ``None`` when it cannot name
    its model -- in which case there is nothing to record and nothing to
    compare, exactly as for the per-container marker."""
    if origin not in BaselineOrigin.ALL:
        raise ValueError(f"unknown baseline origin {origin!r}")
    model = declared_model_name(embedding_func)
    if model is None:
        return None
    return EmbeddingBaseline(
        model=model, dim=declared_dimension(embedding_func), origin=origin
    )


# ---------------------------------------------------------------------------
# The container: its own category, its own directory or fixed name
# ---------------------------------------------------------------------------


# The backends the configuration category admits, in the order an error lists
# them. Mirrors ``STORAGE_IMPLEMENTATIONS["CONFIG_STORAGE"]``; the registry is
# the source of truth and this is what reads it.
def configuration_storage_implementations() -> tuple[str, ...]:
    """The backend names ``CONFIG_STORAGE`` admits."""
    from lightrag.kg import STORAGE_IMPLEMENTATIONS

    return tuple(STORAGE_IMPLEMENTATIONS["CONFIG_STORAGE"]["implementations"])


def resolve_configuration_storage(selected: str | None, *, kv_storage: str) -> str:
    """The configuration backend to use, refusing anything outside the four.

    Unset, the selection FOLLOWS ``kv_storage`` -- an existing deployment's
    rows are already in that backend's container, so any other default would
    orphan them silently. A ``kv_storage`` the category does not admit
    (``RedisKVStorage`` today) is refused here, by name, rather than failing
    later on a missing method: the operator picks one of the four explicitly.

    Raises:
        ValueError: the selection, or the ``kv_storage`` it was derived from,
            is not one of the four.
    """
    admitted = configuration_storage_implementations()
    name = (selected or "").strip()
    if name:
        if name not in admitted:
            raise ValueError(
                f"config_storage={name!r} is not a configuration storage "
                f"backend. The configuration storage is its own category and "
                f"admits {', '.join(admitted)}. A vector storage cannot serve "
                f"here (its container is named after the embedding model, "
                f"which is the assertion the baselines exist to be "
                f"independent of), and Redis is excluded."
            )
        return name
    if kv_storage not in admitted:
        raise ValueError(
            f"config_storage is unset, so it would follow kv_storage="
            f"{kv_storage!r} -- which the configuration category does not "
            f"admit. Set config_storage (LIGHTRAG_CONFIG_STORAGE) to one of "
            f"{', '.join(admitted)}."
        )
    return kv_storage


def resolve_config_dir(config_dir: str | None, working_dir: str) -> str:
    """The absolute directory a file-backed configuration storage uses.

    Absolute for the same reason ``working_dir`` is: the single-server claim
    keys on the REALPATH, so a relative spelling from a differently-rooted
    process would open a second descriptor on the same lock file and refuse
    itself.
    """
    return os.path.abspath(
        (config_dir or "").strip() or default_config_dir(working_dir)
    )


def configuration_selection_from_env(
    *, kv_storage: str, working_dir: str
) -> tuple[str, str]:
    """``(config_storage, config_dir)`` as the environment resolves them.

    The one answer three call sites must agree on: ``LightRAG`` (from its own
    fields), ``lightrag-rebuild-vdb``, and the **Gunicorn master**, which takes
    the directory claim before forking. They must not drift: a master that
    claims a different directory than its workers hands them no inheritable
    claim, and each worker then opens its own descriptor -- the first wins and
    every other one is refused at startup.

    Raises ``ValueError`` when the selection is outside the category.
    """
    config_storage = resolve_configuration_storage(
        os.environ.get("LIGHTRAG_CONFIG_STORAGE", ""), kv_storage=kv_storage
    )
    return config_storage, resolve_config_dir(
        os.environ.get("LIGHTRAG_CONFIG_DIR", ""), working_dir
    )


def describe_configuration_container(storage_name: str, config_dir: str) -> str:
    """One operator-readable phrase naming where configuration is kept."""
    if storage_name in FILE_BACKED_CONFIG_STORAGES:
        return f"{storage_name} at {config_dir}"
    return f"{storage_name} ({CONFIG_CONTAINER_TAG})"


# The configuration backends that keep their container on the local
# filesystem, which is what makes the directory claim necessary.
FILE_BACKED_CONFIG_STORAGES = frozenset({"JsonKVStorage"})


def create_configuration_storage(
    config_storage_cls: Callable[..., Any],
    *,
    global_config: dict[str, Any],
    embedding_func: Any,
) -> Any:
    """Construct the configuration storage on its fixed container.

    The single door in. What makes the container unreachable by accident is
    no longer a reserved NAME defended everywhere a name can be chosen -- it
    is that the container is not addressed by a workspace at all. Every
    backend in the category keys off the ``config`` namespace, which nothing
    else is ever opened on, and names its container in code: a directory for
    the JSON backend, ``CONFIG_CONTAINER_TAG`` for the other three. No
    ``*_WORKSPACE`` variable is consulted.

    The returned storage is NOT initialized; the caller owns its lifecycle.
    """
    storage = config_storage_cls(
        namespace=NameSpace.KV_STORE_CONFIG,
        workspace=CONFIG_CONTAINER_TAG,
        global_config=global_config,
        embedding_func=embedding_func,
    )
    # A backend that RE-BOUND the container elsewhere is refused. A stand-in
    # that records no workspace at all (test doubles handed to the factory)
    # has not remapped anything; every real backend carries the dataclass
    # field, so ``None`` here is never a real backend's answer.
    bound = getattr(storage, "workspace", None)
    if bound is not None and bound != CONFIG_CONTAINER_TAG:
        raise ConfigurationStorageError(
            f"{type(storage).__name__} bound the configuration container to "
            f"{bound!r} instead of {CONFIG_CONTAINER_TAG!r}; the configuration "
            f"container is named in code and must not be remapped"
        )
    return storage


def warn_about_unrecorded_baselines(
    absent: list[str], *, workspace: str, container: str
) -> bool:
    """Announce a start on which NO baseline is on record. Never refuses.

    A separately selected configuration backend is a new way to point a
    running deployment at an empty store -- a fresh database, a mistyped
    connection string, a ``config_dir`` that does not exist -- and every
    baseline then reads as absent, which is what lets a start bootstrap.

    Enforcing is not available: absent is also what a genuine first start
    looks like, and the two are indistinguishable from here. So this follows
    ``warn_about_workspace_overrides()`` -- announce, do not enforce -- and
    the actual protection stays where it already is: an absent baseline is
    only ever established on POSITIVE evidence (a confirmed-empty container,
    or an adoption probe that vouched for the stored vectors), never on the
    configured model alone.

    Returns whether it warned, so a caller (and a test) can tell.
    """
    if not absent or len(absent) != len(EMBEDDING_TARGETS):
        return False
    logger.warning(
        f"[{workspace}] No embedding baseline is recorded in {container} for "
        f"this workspace. On a first start that is expected. If this "
        f"deployment has run before, check that the configuration storage "
        f"selection (config_storage / config_dir) still points at the store "
        f"that recorded them -- an empty or different one reads exactly like "
        f"a first start. Nothing is adopted on the configured model alone: a "
        f"baseline is recorded only for a target whose container is confirmed "
        f"empty or whose stored vectors an adoption probe vouched for."
    )
    return True


# ---------------------------------------------------------------------------
# Strict reads and the precheck
# ---------------------------------------------------------------------------


async def read_config_row_strict(config: Any, key: str) -> dict[str, Any] | None:
    """``None`` means CONFIRMED absent; anything that could not be confirmed raises.

    Two kinds of raise, told apart by TYPE and not by message: the store could
    not answer (``ConfigurationStorageError``), or it answered with something
    that is not a row (``ConfigurationRecordMalformedError``). Both stop a
    start, so nothing that must refuse stops refusing. The difference is for
    the caller that may legitimately go on -- a record it is about to DELETE
    by key, which never reads the value.
    """
    if not getattr(type(config), "supports_strict_point_reads", False):
        raise ConfigurationStorageError(
            f"{type(config).__name__} does not declare strict point reads, so a "
            f"configuration record cannot be read to a definite answer"
        )
    try:
        row = await config.get_by_id_strict(key)
    except ConfigurationStorageError:
        raise
    except CorruptStorageRecordError as e:
        # The backend reached the row and found it was not one. It never got
        # far enough to RETURN the payload, so the shape check below cannot
        # see this case -- without this branch it arrives as the generic
        # "could not read" above and reads as an outage.
        raise ConfigurationRecordMalformedError(
            f"configuration record {key!r} is not a mapping: {e}"
        ) from e
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not read configuration record {key!r} ({type(e).__name__}: {e})"
        ) from e
    if row is not None and not isinstance(row, dict):
        raise ConfigurationRecordMalformedError(
            f"configuration record {key!r} is not a mapping: {row!r}"
        )
    return row


async def read_embedding_baselines(
    config: Any, workspace: str
) -> dict[str, EmbeddingBaseline | None]:
    """Strict-read the three baselines. ``None`` per target means confirmed absent."""
    out: dict[str, EmbeddingBaseline | None] = {}
    for target in EMBEDDING_TARGETS:
        key = embedding_baseline_key(workspace, target)
        row = await read_config_row_strict(config, key)
        out[target] = (
            None
            if row is None
            else EmbeddingBaseline.from_row(row, key=key, target=target)
        )
    return out


def precheck_embedding_baselines(
    baselines: dict[str, EmbeddingBaseline | None],
    embedding_func: Any,
    *,
    workspace: str,
) -> list[str]:
    """Compare every recorded baseline with the configuration.

    Returns the targets whose record is absent (the bootstrap targets). Raises
    ``EmbeddingBaselineMismatchError`` naming EVERY target that differs -- one
    refusal, the whole list.
    """
    absent: list[str] = []
    mismatches: list[dict[str, Any]] = []
    expected_model = declared_model_name(embedding_func)
    expected_dim = declared_dimension(embedding_func)
    for target in EMBEDDING_TARGETS:
        recorded = baselines.get(target)
        if recorded is None:
            absent.append(target)
            continue
        if recorded.differs_from(embedding_func):
            mismatches.append(
                {
                    "target": target,
                    "recorded_model": recorded.model,
                    "recorded_dim": recorded.dim,
                    "expected_model": expected_model,
                    "expected_dim": expected_dim,
                }
            )
    if mismatches:
        raise EmbeddingBaselineMismatchError(workspace=workspace, mismatches=mismatches)
    return absent


# ---------------------------------------------------------------------------
# Writes: the atomic claim, the rebuild record, the drop
# ---------------------------------------------------------------------------


async def flush_configuration_storage(config: Any, what: str) -> None:
    """Flush, and let the BUFFER decide what the flush did -- not the return.

    Two ways a flush misreports itself, and one question separates them.

    ``OpenSearchKVStorage.index_done_callback`` keeps per-item RETRYABLE
    failures (408 / 429 / 5xx) buffered and RETURNS NORMALLY, and its strict
    point read answers from that buffer -- a buffered upsert reads as present,
    a buffered tombstone as gone -- so flush-then-read-back would confirm a
    write or a delete the server never saw.

    The mirror case is a flush that RAISES over a write that landed. A backend
    able to prove its raise lost nothing raises ``ReferencesIntactFlushError``,
    and that type covers two situations its own docstring separates: every
    operation still buffered (a bulk transport error), or the commit landed and
    only a step after it failed (a refresh). Treating both as a failed flush
    reports a durable write as one that did not happen, which is the one thing
    this module exists to prevent.

    So the store is asked the same question either way -- is anything still
    buffered, tombstones included. Retained: the buffer is dropped (what the
    caller reports must be what is true) and the flush is the failure it is.
    Nothing retained: the flush landed, and every caller here strict-reads the
    row back, so the read-back is what confirms it. A backend that cannot
    answer keeps the conservative reading of its own raise. Backends without a
    buffer answer ``False`` and are unaffected.
    """
    intact_failure: ReferencesIntactFlushError | None = None
    try:
        await config.index_done_callback()
    except ReferencesIntactFlushError as e:
        # Nothing was lost -- but whether anything LANDED is the buffer's
        # answer to give, below.
        intact_failure = e
    except Exception as e:
        raise ConfigurationStorageError(
            f"the configuration storage could not flush {what} "
            f"({type(e).__name__}: {e})"
        ) from e
    has_pending = getattr(config, "has_pending_index_ops", None)
    if has_pending is None:
        if intact_failure is None:
            return
        raise ConfigurationStorageError(
            f"the configuration storage raised while flushing {what} and "
            f"cannot say whether the operation is still buffered "
            f"({type(intact_failure).__name__}: {intact_failure})"
        ) from intact_failure
    try:
        retained = bool(await has_pending(include_deletes=True))
    except Exception as e:
        raise ConfigurationStorageError(
            f"the configuration storage could not say whether {what} was "
            f"flushed ({type(e).__name__}: {e})"
        ) from e
    if not retained:
        if intact_failure is not None:
            logger.warning(
                f"The configuration storage raised while flushing {what} but "
                f"kept nothing buffered, so the write landed and only a step "
                f"after it failed ({type(intact_failure).__name__}: "
                f"{intact_failure}); the strict read-back decides."
            )
        return
    drop = getattr(config, "drop_pending_index_ops", None)
    if drop is not None:
        try:
            await drop()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"Could not discard the configuration operations the flush "
                f"retained ({type(e).__name__}: {e}); they may replay at shutdown"
            )
    raise ConfigurationStorageError(
        f"the configuration storage retained {what} after the flush (a "
        f"transient backend failure kept the operation buffered, whether the "
        f"flush returned or raised); the write is not durable and must not be "
        f"reported as one"
    ) from intact_failure


async def _write_baseline_row(
    config: Any,
    *,
    key: str,
    workspace: str,
    target: str,
    baseline: EmbeddingBaseline,
    updated_by: str,
) -> None:
    row = make_config_row(
        scope_workspace=workspace,
        suffix=embedding_baseline_suffix(target),
        value=baseline.to_value(),
        updated_by=updated_by,
    )
    try:
        await config.upsert({key: row})
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not write configuration record {key!r} ({type(e).__name__}: {e})"
        ) from e


def _mismatch(
    target: str, recorded: EmbeddingBaseline, embedding_func: Any
) -> dict[str, Any]:
    return {
        "target": target,
        "recorded_model": recorded.model,
        "recorded_dim": recorded.dim,
        "expected_model": declared_model_name(embedding_func),
        "expected_dim": declared_dimension(embedding_func),
    }


async def claim_embedding_baseline(
    config: Any,
    *,
    workspace: str,
    target: str,
    candidate: EmbeddingBaseline,
    embedding_func: Any,
    updated_by: str = UPDATED_BY_STARTUP,
) -> EmbeddingBaseline:
    """Establish ``candidate`` as ``target``'s baseline unless one already exists.

    ``read absent -> upsert`` is not atomic, so the claim runs under a keyed
    lock per (workspace, target): strict-read again inside it, write only when
    still absent, flush BEFORE releasing (``OpenSearchKVStorage.upsert``
    buffers in process memory until the flush), then strict-read back and
    validate what is actually stored against THIS process's configuration.
    The lock spans workers of one Gunicorn master and nothing wider; see *What
    the lock does and does not span* in ``docs/design/ConfigurationStorageContract.md``.

    Returns the baseline now on record (this process's, or the one another
    worker got in first with). Raises ``EmbeddingBaselineMismatchError`` if
    the record on file differs from the configuration.
    """
    from lightrag.kg.shared_storage import get_storage_keyed_lock

    key = embedding_baseline_key(workspace, target)
    async with get_storage_keyed_lock(key, namespace=CLAIM_LOCK_NAMESPACE):
        row = await read_config_row_strict(config, key)
        if row is None:
            await _write_baseline_row(
                config,
                key=key,
                workspace=workspace,
                target=target,
                baseline=candidate,
                updated_by=updated_by,
            )
            await flush_configuration_storage(config, f"the baseline claim for {key!r}")
            row = await read_config_row_strict(config, key)
            if row is None:
                raise ConfigurationStorageError(
                    f"the configuration storage read back nothing for {key!r} "
                    f"right after writing it; the write is not visible or not "
                    f"durable, and the instance must not serve on it"
                )
            logger.info(
                f"[{workspace}] Recorded the embedding baseline for {target}: "
                f"model={candidate.model!r} dim={candidate.dim} origin={candidate.origin}"
            )
        recorded = EmbeddingBaseline.from_row(row, key=key, target=target)
        if recorded.differs_from(embedding_func):
            raise EmbeddingBaselineMismatchError(
                workspace=workspace,
                mismatches=[_mismatch(target, recorded, embedding_func)],
            )
        return recorded


async def record_embedding_baseline(
    config: Any,
    *,
    workspace: str,
    target: str,
    embedding_func: Any,
    origin: str = BaselineOrigin.REBUILD,
    updated_by: str = UPDATED_BY_REBUILD,
) -> EmbeddingBaseline:
    """Overwrite ``target``'s baseline with the configured space; flush; read back.

    The rebuild tool's write, made AFTER a target's rebuild is durable and
    verified and for that one target only. Raises ``ConfigurationStorageError``
    if the process cannot name its model, or the write, flush or read-back
    fails -- the caller exits non-zero, and the stale record keeps refusing.
    """
    baseline = configured_baseline(embedding_func, origin=origin)
    if baseline is None:
        raise ConfigurationStorageError(
            "the configured embedding function declares no model_name, so no "
            "embedding baseline can be recorded"
        )
    key = embedding_baseline_key(workspace, target)
    await _write_baseline_row(
        config,
        key=key,
        workspace=workspace,
        target=target,
        baseline=baseline,
        updated_by=updated_by,
    )
    await flush_configuration_storage(config, f"the baseline record for {key!r}")
    row = await read_config_row_strict(config, key)
    if row is None:
        raise ConfigurationStorageError(
            f"the configuration storage read back nothing for {key!r} right "
            f"after writing it"
        )
    stored = EmbeddingBaseline.from_row(row, key=key, target=target)
    if stored.model != baseline.model or (
        stored.dim is not None
        and baseline.dim is not None
        and stored.dim != baseline.dim
    ):
        raise ConfigurationStorageError(
            f"the configuration storage read back {stored} for {key!r} after "
            f"writing {baseline}"
        )
    return stored


async def delete_workspace_configuration(config: Any, workspace: str) -> None:
    """Delete a workspace's configuration rows, flush, and confirm they are gone.

    Called ONLY after every data storage of the workspace dropped
    successfully: *configuration gone, data remains* is the residue that can
    never be accepted (a later start would bootstrap a wrong baseline over
    surviving vectors). Raises ``ConfigurationStorageError`` if a row survives
    the delete -- backends that swallow delete errors would otherwise report a
    removal that did not happen.

    Two callers: ``/documents/clear`` and ``lightrag-clear-storage``. A third
    one declares itself in ``CONFIG_KEY_REGISTRY``'s ``writers`` before it
    calls this; nothing enforces that list, which is exactly why it drifts.
    """
    keys = [embedding_baseline_key(workspace, target) for target in EMBEDDING_TARGETS]
    try:
        await config.delete(keys)
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not delete the configuration rows of workspace {workspace!r} "
            f"({type(e).__name__}: {e})"
        ) from e
    await flush_configuration_storage(
        config, f"the configuration rows of workspace {workspace!r}"
    )
    surviving = [
        key for key in keys if await read_config_row_strict(config, key) is not None
    ]
    if surviving:
        raise ConfigurationStorageError(
            f"configuration rows survived the delete for workspace "
            f"{workspace!r}: {surviving}"
        )


# ---------------------------------------------------------------------------
# The container identity and the anchor
# ---------------------------------------------------------------------------


def storage_identity_key() -> str:
    """``_lightrag_server/storage_identity``."""
    return config_key(SERVER_SCOPE, STORAGE_IDENTITY_SUFFIX)


def _identity_invalid(key: str, detail: str) -> ConfigurationIdentityError:
    return ConfigurationIdentityError(
        f"The configuration storage identity row {key!r} is unreadable: "
        f"{detail}. It is never treated as absent and never regenerated; "
        f"repair or restore the row.",
        cause=IDENTITY_ROW_INVALID,
    )


async def read_storage_identity(config: Any) -> str | None:
    """The container's UUID; ``None`` only when the row is CONFIRMED absent.

    A backend error raises ``ConfigurationStorageError`` (the strict read);
    a row with the wrong schema, scope or a malformed UUID raises
    ``ConfigurationIdentityError`` with cause ``IDENTITY_ROW_INVALID``.
    """
    key = storage_identity_key()
    row = await read_config_row_strict(config, key)
    if row is None:
        return None
    expected_version = registry_spec(STORAGE_IDENTITY_SUFFIX).schema_version
    version = row.get("schema_version")
    if type(version) is not int or version != expected_version:
        raise _identity_invalid(
            key,
            f"unsupported schema_version {version!r}; expected integer "
            f"{expected_version}",
        )
    if row.get("workspace") != SERVER_CONFIG_SCOPE:
        raise _identity_invalid(
            key, f"scope {row.get('workspace')!r} is not {SERVER_CONFIG_SCOPE!r}"
        )
    value = row.get("value")
    storage_uuid = canonical_storage_uuid(
        value.get("uuid") if isinstance(value, dict) else None
    )
    if storage_uuid is None:
        raise _identity_invalid(key, f"value {value!r} carries no canonical UUID")
    return storage_uuid


async def write_storage_identity(
    config: Any, storage_uuid: str, *, updated_by: str = UPDATED_BY_STARTUP
) -> None:
    """Write the identity row, flush strictly, read it back and compare.

    Through ``flush_configuration_storage`` like every baseline claim, so an
    OpenSearch write still sitting in the process-local buffer is a failure
    rather than a read-back answered from that buffer. Two callers: the bind
    of a container that has none, and the offline migration claiming its
    target (the ownership marker). Neither ever overwrites a valid identity.
    """
    key = storage_identity_key()
    row = make_config_row(
        scope_workspace=SERVER_SCOPE,
        suffix=STORAGE_IDENTITY_SUFFIX,
        value={"uuid": storage_uuid},
        updated_by=updated_by,
    )
    try:
        await config.upsert({key: row})
    except Exception as e:
        raise ConfigurationIdentityError(
            f"could not write the configuration storage identity {key!r} "
            f"({type(e).__name__}: {e})",
            cause=IDENTITY_WRITE_FAILED,
        ) from e
    await flush_configuration_storage(config, f"the storage identity {key!r}")
    stored = await read_storage_identity(config)
    if stored != storage_uuid:
        raise ConfigurationIdentityError(
            f"the configuration storage read back identity {stored!r} for "
            f"{key!r} right after writing {storage_uuid!r}; the write is not "
            f"visible or not durable, or another process wrote concurrently, "
            f"and the instance must not serve on it",
            cause=IDENTITY_WRITE_FAILED,
        )


def check_anchor_backend(
    anchor: StorageAnchor | None, backend: str, *, working_dir: str, container: str
) -> None:
    """Step 0c: refuse when an anchor binds another backend TYPE.

    Runs before the configuration storage is opened, so a refusal here opens
    nothing. The server never switches to the anchored backend on its own:
    type drift is the mistake the anchor exists to stop, so the advice is to
    select the anchored backend explicitly, and deleting the anchor comes
    last.
    """
    if anchor is None or anchor.backend == backend:
        return
    path = anchor_path(working_dir)
    raise ConfigurationIdentityError(
        f"Refusing to start: the configuration storage selected now is "
        f"{container}, but the anchor {path} binds this deployment to "
        f"{anchor.backend} (identity {anchor.storage_uuid}). The usual cause "
        f"is LIGHTRAG_KV_STORAGE changing while LIGHTRAG_CONFIG_STORAGE is "
        f"unset, so the configuration followed it to a container that does "
        f"not hold this deployment's records. Fix: set "
        f"LIGHTRAG_CONFIG_STORAGE={anchor.backend} explicitly (no migration "
        f"needed), or move the configuration container to {backend} with "
        f"`lightrag-migrate-config --target-backend {backend}` before "
        f"starting. Last resort: deleting {path} rebinds the "
        f"next start to {backend} and ABANDONS every record in the "
        f"{anchor.backend} container, embedding baselines included.",
        cause=IDENTITY_BACKEND_MISMATCH,
        anchor_path=path,
    )


def preflight_configuration_anchor(
    *, working_dir: str, backend: str, container: str
) -> StorageAnchor | None:
    """Steps 0b-0c for any starter: strict-read the anchor and refuse a
    backend type it does not bind. Returns the anchor (``None`` when there is
    none). Opens nothing; the caller holds the shared anchor lock."""
    anchor = read_anchor(working_dir)
    check_anchor_backend(anchor, backend, working_dir=working_dir, container=container)
    return anchor


def _check_identity_against_anchor(
    anchor: StorageAnchor, stored: str | None, *, working_dir: str, container: str
) -> None:
    path = anchor_path(working_dir)
    if stored is None:
        raise ConfigurationIdentityError(
            f"Refusing to start: the configuration container {container} "
            f"holds no storage identity, but the anchor {path} binds this "
            f"deployment to identity {anchor.storage_uuid}. If the container "
            f"was intentionally emptied, replaced, or restored from a backup "
            f"older than the identity, delete {path} and restart -- the next "
            f"start binds to this container. Otherwise check that the "
            f"connection settings (or LIGHTRAG_CONFIG_DIR) point at the "
            f"intended container. No identity was created.",
            cause=IDENTITY_UUID_MISSING,
            anchor_path=path,
        )
    if stored != anchor.storage_uuid:
        raise ConfigurationIdentityError(
            f"Refusing to start: the configuration container {container} has "
            f"identity {stored}, but the anchor {path} binds this deployment "
            f"to identity {anchor.storage_uuid}. Check that the connection "
            f"settings (or LIGHTRAG_CONFIG_DIR) point at the intended "
            f"database or directory. Last resort: deleting {path} rebinds the "
            f"next start to this container and ABANDONS every record in the "
            f"anchored one.",
            cause=IDENTITY_UUID_MISMATCH,
            anchor_path=path,
        )


@dataclass(frozen=True)
class IdentityBinding:
    """What a bind or a verification established."""

    storage_uuid: str | None
    # "verified" (anchor and container agree), "adopted" (no anchor, the
    # container's identity was bound), "created" (no anchor, no identity:
    # one was created and bound), "unanchored" (a tool found no anchor and
    # verified nothing).
    action: str


async def bind_configuration_identity(
    config: Any, *, working_dir: str, backend: str, container: str
) -> IdentityBinding:
    """Step 1b: verify the container against the anchor, or bind it.

    Under one keyed lock spanning the whole read-decide-write, so a second
    worker of the same Gunicorn master waits and then finds an anchor and an
    equal UUID. A start that finds NO anchor additionally takes the anchor
    bind lock (``lightrag/kg/anchor_lock.py``), which serializes the bind
    across process trees on this host -- several servers may share one
    ``working_dir`` with different workspaces -- and re-reads the anchor
    inside it. Concurrent first binds from different ``working_dir``s or
    hosts against one container remain unsupported.

    * Anchored: the backend type and the container's UUID must both equal
      the anchor's. A missing, different, invalid or unreadable identity
      refuses, and nothing is created.
    * Not anchored: a present identity is adopted; a confirmed-absent one is
      created (upsert, strict flush, strict read-back). Either way the
      anchor is then published no-clobber and a WARNING names the container
      and the UUID. The anchor goes last so every crash window heals by
      adoption on the next start.
    """
    from lightrag.kg.anchor_lock import anchor_bind_lock
    from lightrag.kg.shared_storage import get_storage_keyed_lock

    async with get_storage_keyed_lock(
        IDENTITY_LOCK_KEY, namespace=IDENTITY_LOCK_NAMESPACE
    ):
        anchor = read_anchor(working_dir)
        if anchor is not None:
            return await _verify_anchored(
                config,
                anchor,
                working_dir=working_dir,
                backend=backend,
                container=container,
            )
        async with anchor_bind_lock(working_dir):
            # Another server on this working directory may have bound while
            # this one waited: its anchor is then the one to verify against.
            anchor = read_anchor(working_dir)
            if anchor is not None:
                return await _verify_anchored(
                    config,
                    anchor,
                    working_dir=working_dir,
                    backend=backend,
                    container=container,
                )
            return await _bind_unanchored(
                config,
                working_dir=working_dir,
                backend=backend,
                container=container,
            )


async def _verify_anchored(
    config: Any,
    anchor: StorageAnchor,
    *,
    working_dir: str,
    backend: str,
    container: str,
) -> IdentityBinding:
    check_anchor_backend(anchor, backend, working_dir=working_dir, container=container)
    stored = await read_storage_identity(config)
    _check_identity_against_anchor(
        anchor, stored, working_dir=working_dir, container=container
    )
    return IdentityBinding(storage_uuid=stored, action="verified")


async def _bind_unanchored(
    config: Any, *, working_dir: str, backend: str, container: str
) -> IdentityBinding:
    stored = await read_storage_identity(config)
    if stored is None:
        stored = new_storage_uuid()
        await write_storage_identity(config, stored)
        action = "created"
    else:
        action = "adopted"
    path = publish_anchor(
        working_dir,
        StorageAnchor(backend=backend, storage_uuid=stored),
        replace=False,
    )
    logger.warning(
        f"Bound this deployment to the configuration container {container} "
        f"(identity {stored}, {action}); the anchor is {path}. No anchor was "
        f"on record, which is expected on a first start or after the anchor "
        f"was deliberately deleted to rebind. If neither is the case -- a "
        f"WORKING_DIR that does not persist, a replaced volume -- drift "
        f"between configuration containers was NOT checked on this start."
    )
    return IdentityBinding(storage_uuid=stored, action=action)


async def verify_configuration_identity(
    config: Any,
    anchor: StorageAnchor | None,
    *,
    working_dir: str,
    backend: str,
    container: str,
) -> IdentityBinding:
    """The maintenance tools' check: refuse exactly as a start would, write
    nothing. With no anchor there is nothing to verify, and the caller says
    so; only a server or SDK start binds."""
    if anchor is None:
        return IdentityBinding(storage_uuid=None, action="unanchored")
    check_anchor_backend(anchor, backend, working_dir=working_dir, container=container)
    stored = await read_storage_identity(config)
    _check_identity_against_anchor(
        anchor, stored, working_dir=working_dir, container=container
    )
    return IdentityBinding(storage_uuid=stored, action="verified")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


async def iter_configuration_rows(
    config: Any, *, page_size: int = 200
) -> AsyncIterator[dict[str, Any]]:
    """Stream every configuration row, classified by its ``workspace`` FIELD.

    Rows that do not carry the uniform shape are yielded with
    ``workspace=None`` rather than skipped, so an inventory can report them
    instead of silently under-counting. Never reparse the key.
    """
    async for row in config.iter_rows(page_size=page_size):
        if not isinstance(row, dict):
            continue
        scope = row.get("workspace")
        yield {
            "id": row.get("_id", row.get("id")),
            "workspace": scope if isinstance(scope, str) else None,
            "schema_version": row.get("schema_version"),
            "updated_at": row.get("updated_at"),
            "updated_by": row.get("updated_by"),
            "value": row.get("value"),
        }
