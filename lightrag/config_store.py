"""The configuration storage: one place, independent of any knowledge-base
workspace, for the server's settings and every workspace's.

Full contract: ``docs/design/ConfigurationStorage.md``. Read it before adding
a key, before binding a storage to the configuration workspace, and before
moving anything out of an environment variable. The rules this module
enforces, in the order a caller meets them:

* **The container** is the KV namespace ``config`` in the reserved workspace
  ``_lightrag_config``, and only ``create_configuration_storage()`` may bind
  it. The reservation is enforced by ``validate_workspace``; this module holds
  the one grant that lets the reserved name through, for one construction.

* **Keys** are ``<workspace>/<suffix>`` or ``_lightrag_server/<suffix>``, with
  ``/`` as the separator (a workspace name cannot contain it). They are built
  by ``config_key`` and **never reparsed**: the row carries ``workspace`` as a
  field, and a reader classifies by the field.

* **Every key is registered** in ``CONFIG_KEY_REGISTRY`` before it is written.
  An unregistered suffix is a programming error, not a runtime condition.

* **Reads are strict.** A record that decides whether the instance may serve
  is read with ``get_by_id_strict``; a read that could not complete raises
  ``ConfigurationStorageError`` and is never mistaken for absence.

* **A missing baseline is claimed atomically** under a keyed lock, with the
  flush and a strict read-back inside the lock. The claim covers workers of
  one Gunicorn master and nothing wider.

The first keys are the three per-target embedding baselines. What a baseline
MEANS -- the space adopted for that target, not the space its vectors were
written in -- and why there are three of them is in the contract; the
``origin`` field records how each claim was established and never enters a
verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable

from lightrag.exceptions import (
    ConfigurationStorageError,
    EmbeddingBaselineMismatchError,
    ReferencesIntactFlushError,
)
from lightrag.kg.vector_space import declared_dimension, declared_model_name
from lightrag.namespace import CONFIG_WORKSPACE, SERVER_CONFIG_SCOPE, NameSpace
from lightrag.utils import _grant_reserved_workspace, logger

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

# ``updated_by`` values the two writers stamp. Diagnostic only.
UPDATED_BY_STARTUP = "lightrag.initialize_storages"
UPDATED_BY_REBUILD = "lightrag-rebuild-vdb"


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

CONFIG_KEY_REGISTRY: dict[str, ConfigKeySpec] = {
    embedding_baseline_suffix(target): ConfigKeySpec(
        suffix=embedding_baseline_suffix(target),
        scope=ConfigScope.WORKSPACE,
        schema_version=1,
        schema=_EMBEDDING_BASELINE_SCHEMA,
        readers=(UPDATED_BY_STARTUP, UPDATED_BY_REBUILD),
        writers=(UPDATED_BY_STARTUP, UPDATED_BY_REBUILD, "/documents/clear"),
        sensitive=False,
    )
    for target in EMBEDDING_TARGETS
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


def config_key(scope_workspace: str, suffix: str) -> str:
    """Build a key. ``scope_workspace`` is a workspace name or
    ``SERVER_CONFIG_SCOPE``; the suffix must be registered."""
    spec = registry_spec(suffix)
    if spec.scope is ConfigScope.SERVER and scope_workspace != SERVER_CONFIG_SCOPE:
        raise ValueError(
            f"{suffix!r} is a server-global key and must be filed under "
            f"{SERVER_CONFIG_SCOPE!r}, not {scope_workspace!r}"
        )
    if spec.scope is ConfigScope.WORKSPACE and scope_workspace == SERVER_CONFIG_SCOPE:
        raise ValueError(
            f"{suffix!r} is a per-workspace key; {SERVER_CONFIG_SCOPE!r} is not a workspace"
        )
    if KEY_SEPARATOR in scope_workspace:
        raise ValueError(
            f"a workspace name cannot contain {KEY_SEPARATOR!r}: {scope_workspace!r}"
        )
    return f"{scope_workspace}{KEY_SEPARATOR}{suffix}"


def embedding_baseline_key(workspace: str, target: str) -> str:
    """``<workspace>/embedding/<target>``."""
    return config_key(workspace, embedding_baseline_suffix(target))


def make_config_row(
    *,
    scope_workspace: str,
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
        "workspace": scope_workspace,
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
    def from_row(cls, row: dict[str, Any], *, key: str) -> "EmbeddingBaseline":
        """Parse a stored row. A row that does not parse is UNREADABLE, which
        is a startup failure, not an absence."""
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
# The container
# ---------------------------------------------------------------------------


def create_configuration_storage(
    kv_storage_cls: Callable[..., Any],
    *,
    global_config: dict[str, Any],
    embedding_func: Any,
) -> Any:
    """Bind a KV storage to the reserved configuration workspace.

    The ONLY way to construct a storage on ``_lightrag_config``: the private
    grant it takes out lets ``validate_workspace`` accept the reserved name
    for this one construction, and every backend's ``*_WORKSPACE``
    environment remap is skipped for a reserved name, so the container's
    workspace is fixed rather than configured. The returned storage is NOT
    initialized; the caller owns its lifecycle.
    """
    with _grant_reserved_workspace(CONFIG_WORKSPACE):
        storage = kv_storage_cls(
            namespace=NameSpace.KV_STORE_CONFIG,
            workspace=CONFIG_WORKSPACE,
            global_config=global_config,
            embedding_func=embedding_func,
        )
    # A backend that RE-BOUND the container elsewhere is refused. A stand-in
    # that records no workspace at all (test doubles handed to the factory)
    # has not remapped anything; every real backend carries the dataclass
    # field, so ``None`` here is never a real backend's answer.
    bound = getattr(storage, "workspace", None)
    if bound is not None and bound != CONFIG_WORKSPACE:
        raise ConfigurationStorageError(
            f"{type(storage).__name__} bound the configuration container to "
            f"workspace {bound!r} instead of {CONFIG_WORKSPACE!r}; the "
            f"configuration workspace is fixed and must not be remapped"
        )
    return storage


# ---------------------------------------------------------------------------
# Strict reads and the precheck
# ---------------------------------------------------------------------------


async def read_config_row_strict(config: Any, key: str) -> dict[str, Any] | None:
    """``None`` means CONFIRMED absent; anything that could not be confirmed raises."""
    if not getattr(type(config), "supports_strict_point_reads", False):
        raise ConfigurationStorageError(
            f"{type(config).__name__} does not declare strict point reads, so a "
            f"configuration record cannot be read to a definite answer"
        )
    try:
        row = await config.get_by_id_strict(key)
    except ConfigurationStorageError:
        raise
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not read configuration record {key!r} ({type(e).__name__}: {e})"
        ) from e
    if row is not None and not isinstance(row, dict):
        raise ConfigurationStorageError(
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
        out[target] = None if row is None else EmbeddingBaseline.from_row(row, key=key)
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


async def _flush(config: Any, what: str) -> None:
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
    the lock does and does not span* in ``docs/design/ConfigurationStorage.md``.

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
            await _flush(config, f"the baseline claim for {key!r}")
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
        recorded = EmbeddingBaseline.from_row(row, key=key)
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
    await _flush(config, f"the baseline record for {key!r}")
    row = await read_config_row_strict(config, key)
    if row is None:
        raise ConfigurationStorageError(
            f"the configuration storage read back nothing for {key!r} right "
            f"after writing it"
        )
    stored = EmbeddingBaseline.from_row(row, key=key)
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
    """
    keys = [embedding_baseline_key(workspace, target) for target in EMBEDDING_TARGETS]
    try:
        await config.delete(keys)
    except Exception as e:
        raise ConfigurationStorageError(
            f"could not delete the configuration rows of workspace {workspace!r} "
            f"({type(e).__name__}: {e})"
        ) from e
    await _flush(config, f"the configuration rows of workspace {workspace!r}")
    surviving = [
        key for key in keys if await read_config_row_strict(config, key) is not None
    ]
    if surviving:
        raise ConfigurationStorageError(
            f"configuration rows survived the delete for workspace "
            f"{workspace!r}: {surviving}"
        )


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
