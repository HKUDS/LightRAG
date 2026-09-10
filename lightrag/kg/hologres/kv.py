"""Shared-table key-value storage for the isolated Hologres backend."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from typing import Any, ClassVar, final

from ...base import BaseKVStorage
from ...namespace import NameSpace
from ...utils import validate_workspace
from .capabilities import probe_production_capabilities
from .client import HologresClientManager, quote_qualified_identifier
from .config import HologresConfig
from .schema import KV_TABLE_NAME, HologresSchemaManager, kv_schema_descriptors


_ID_CHUNK_SIZE = 1000
_UPSERT_RECORD_LIMIT = 200
_UPSERT_BYTE_LIMIT = 4 * 1024 * 1024
_ALLOWED_NAMESPACES = frozenset(
    {
        NameSpace.KV_STORE_FULL_DOCS,
        NameSpace.KV_STORE_TEXT_CHUNKS,
        NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        NameSpace.KV_STORE_FULL_ENTITIES,
        NameSpace.KV_STORE_FULL_RELATIONS,
        NameSpace.KV_STORE_ENTITY_CHUNKS,
        NameSpace.KV_STORE_RELATION_CHUNKS,
    }
)
_SHARED_CLIENTS = HologresClientManager()
_MISSING = object()


class HologresKVError(RuntimeError):
    """Raised when a KV operation cannot return a trustworthy result."""


def _deterministic_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise HologresKVError("Hologres KV input is invalid") from None


_FULL_DOCS_PROTECTED = (
    "sidecar_location",
    "parse_format",
    "content_hash",
    "process_options",
    "parse_engine",
    "chunk_options",
)


def _protected_key_usable(payload: dict[str, Any], key: str) -> bool:
    if key not in payload:
        return False
    value = payload[key]
    if value is None:
        return False
    if key == "chunk_options":
        return value != {}
    return value != "" and value != '""'


def _merge_full_docs_payload(
    existing: dict[str, Any] | None, new: dict[str, Any]
) -> dict[str, Any]:
    protected_set = set(_FULL_DOCS_PROTECTED)
    if existing is not None:
        merged = {k: v for k, v in existing.items() if k not in protected_set}
    else:
        merged = {}
    merged.update({k: v for k, v in new.items() if k not in protected_set})
    for key in _FULL_DOCS_PROTECTED:
        if _protected_key_usable(new, key):
            merged[key] = new[key]
        elif existing is not None and key in existing:
            merged[key] = existing[key]
    return merged


def _decode_payload(payload: Any) -> dict[str, Any]:
    try:
        if isinstance(payload, str):
            decoded = json.loads(payload)
        elif isinstance(payload, Mapping):
            decoded = json.loads(_deterministic_json(dict(payload)))
        else:
            raise TypeError
    except (HologresKVError, TypeError, ValueError, json.JSONDecodeError):
        raise HologresKVError("Hologres KV row is corrupt") from None
    if not isinstance(decoded, dict):
        raise HologresKVError("Hologres KV row is corrupt")
    return decoded


def _row_field(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return _MISSING


def _chunks(values: Sequence[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def _release_shared_client(
    release: Any, config: HologresConfig, client: Any
) -> Any:
    release_task = asyncio.create_task(release(config, client))
    cancellation: asyncio.CancelledError | None = None
    while not release_task.done():
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError as error:
            if release_task.cancelled():
                raise
            if cancellation is None:
                cancellation = error
        except BaseException as release_error:
            if cancellation is not None:
                raise cancellation from release_error
            raise

    try:
        result = release_task.result()
    except BaseException as release_error:
        if cancellation is not None:
            raise cancellation from release_error
        raise
    if cancellation is not None:
        raise cancellation
    return result


@final
@dataclass(repr=False)
class HologresKVStorage(BaseKVStorage):
    """One logical KV namespace over the fixed shared Hologres table."""

    supports_strict_point_reads: ClassVar[bool] = True

    config: HologresConfig | None = field(default=None, repr=False)
    client: Any | None = field(default=None, repr=False)
    _active_client: Any | None = field(default=None, init=False, repr=False)
    _effective_config: HologresConfig | None = field(
        default=None, init=False, repr=False
    )
    _owns_shared_client: bool = field(default=False, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or self.namespace not in _ALLOWED_NAMESPACES:
            raise ValueError("Unsupported Hologres KV namespace")
        try:
            self.workspace = validate_workspace(self.workspace)
        except (TypeError, ValueError):
            raise ValueError("Invalid Hologres KV workspace") from None
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "HologresKVStorage(<redacted>)"

    async def initialize(self) -> None:
        """Acquire/probe one client and reconcile the fixed KV descriptor."""

        async with self._lifecycle_lock:
            if self._initialized:
                return

            actual_client = self.client
            config = self.config
            owns_shared = False
            if actual_client is None:
                if config is None:
                    config = HologresConfig.from_env()
                    self.config = config
                acquire = getattr(_SHARED_CLIENTS, "acquire")
                actual_client = await acquire(config)
                owns_shared = True
            elif config is None:
                config = getattr(actual_client, "config", None)
                if not isinstance(config, HologresConfig):
                    raise HologresKVError("Hologres KV configuration is unavailable")

            try:
                capabilities = await probe_production_capabilities(actual_client)
                apply_capabilities = getattr(actual_client, "apply_capabilities", None)
                if apply_capabilities is not None:
                    apply_capabilities(capabilities)
                schema_manager = HologresSchemaManager(
                    actual_client, schema=config.schema
                )
                await schema_manager.initialize(kv_schema_descriptors(config.schema))
            except BaseException as initialization_error:
                if owns_shared:
                    release = getattr(_SHARED_CLIENTS, "release")
                    try:
                        await _release_shared_client(release, config, actual_client)
                    except asyncio.CancelledError:
                        raise
                    except BaseException as release_error:
                        raise initialization_error from release_error
                raise

            self._active_client = actual_client
            self._effective_config = config
            self._owns_shared_client = owns_shared
            self._initialized = True

    async def finalize(self) -> None:
        """Detach this storage and release only manager-owned clients."""

        async with self._lifecycle_lock:
            actual_client = self._active_client
            config = self._effective_config
            owns_shared = self._owns_shared_client
            self._active_client = None
            self._effective_config = None
            self._owns_shared_client = False
            self._initialized = False
            if owns_shared and actual_client is not None and config is not None:
                release = getattr(_SHARED_CLIENTS, "release")
                await _release_shared_client(release, config, actual_client)

    def _ready(self) -> tuple[Any, str]:
        if not self._initialized or self._active_client is None:
            raise HologresKVError("Hologres KV storage is not initialized")
        if self._effective_config is None:
            raise HologresKVError("Hologres KV configuration is unavailable")
        table = quote_qualified_identifier(
            self._effective_config.schema, KV_TABLE_NAME
        )
        return self._active_client, table

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        client, table = self._ready()
        sql = (
            f"SELECT payload FROM {table} "
            "WHERE workspace = $1 AND namespace = $2 AND id = $3"
        )
        try:
            row = await client.fetch_one(
                sql,
                self.workspace,
                self.namespace,
                id,
                descriptor="kv.read.one",
            )
        except Exception:
            raise HologresKVError("Hologres KV point read failed") from None
        if row is None:
            return None
        payload = _row_field(row, "payload")
        if payload is _MISSING or payload is None:
            raise HologresKVError("Hologres KV row is corrupt")
        return _decode_payload(payload)

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        return await self.get_by_id(id)

    async def get_by_ids(
        self, ids: list[str]
    ) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        client, table = self._ready()
        sql = (
            "SELECT idx AS ordinality, stored.payload "
            "FROM generate_series(1, $3::int) AS idx "
            f"LEFT JOIN {table} AS stored "
            "ON stored.workspace = $1 AND stored.namespace = $2 "
            "AND stored.id = ($4::text[])[idx] "
            "ORDER BY idx"
        )
        result: list[dict[str, Any] | None] = []
        for chunk in _chunks(ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    self.namespace,
                    len(chunk),
                    list(chunk),
                    descriptor="kv.read.batch",
                )
            except Exception:
                raise HologresKVError("Hologres KV batch read failed") from None
            try:
                materialized = list(rows)
            except (TypeError, ValueError):
                raise HologresKVError("Hologres KV batch response is corrupt") from None
            if len(materialized) != len(chunk):
                raise HologresKVError("Hologres KV batch response is corrupt")
            for expected, row in enumerate(materialized, start=1):
                ordinality = _row_field(row, "ordinality")
                payload = _row_field(row, "payload")
                if (
                    type(ordinality) is not int
                    or ordinality != expected
                    or payload is _MISSING
                ):
                    raise HologresKVError("Hologres KV batch response is corrupt")
                result.append(None if payload is None else _decode_payload(payload))
        return result

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        client, table = self._ready()
        sql = (
            f"SELECT id FROM {table} "
            "WHERE workspace = $1 AND namespace = $2 "
            "AND id = ANY($3::text[]) ORDER BY id"
        )
        ordered = sorted(keys)
        existing: set[str] = set()
        for chunk in _chunks(ordered, _ID_CHUNK_SIZE):
            requested = set(chunk)
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="kv.filter",
                )
            except Exception:
                raise HologresKVError("Hologres KV key filter failed") from None
            if rows is None:
                raise HologresKVError("Hologres KV filter response is corrupt")
            try:
                materialized = list(rows)
            except (TypeError, ValueError):
                raise HologresKVError("Hologres KV filter response is corrupt") from None
            chunk_existing: set[str] = set()
            for row in materialized:
                identifier = _row_field(row, "id")
                if (
                    not isinstance(identifier, str)
                    or identifier not in requested
                    or identifier in chunk_existing
                ):
                    raise HologresKVError("Hologres KV filter response is corrupt")
                chunk_existing.add(identifier)
            existing.update(chunk_existing)
        return keys - existing

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        records: list[tuple[str, dict[str, Any]]] = []
        for identifier, payload in data.items():
            if not isinstance(identifier, str) or not isinstance(payload, Mapping):
                raise HologresKVError("Hologres KV input is invalid")
            normalized = dict(payload)
            _deterministic_json({identifier: normalized})
            records.append((identifier, normalized))

        chunks: list[str] = []
        current: dict[str, dict[str, Any]] = {}
        current_json = ""
        for identifier, payload in records:
            candidate = dict(current)
            candidate[identifier] = payload
            candidate_json = _deterministic_json(candidate)
            if current and (
                len(current) >= _UPSERT_RECORD_LIMIT
                or len(candidate_json.encode("utf-8")) > _UPSERT_BYTE_LIMIT
            ):
                chunks.append(current_json)
                current = {identifier: payload}
                current_json = _deterministic_json(current)
            else:
                current = candidate
                current_json = candidate_json
        if current:
            chunks.append(current_json)

        client, table = self._ready()
        is_full_docs = self.namespace == NameSpace.KV_STORE_FULL_DOCS
        if is_full_docs:
            descriptor = "kv.upsert.full_docs"
        else:
            descriptor = "kv.upsert.replace"
        sql = (
            f"INSERT INTO {table} AS current "
            "(workspace, namespace, id, payload, updated_at) "
            "SELECT $1, $2, unnest($3::text[]), "
            "unnest($4::jsonb[]), CURRENT_TIMESTAMP "
            "ON CONFLICT (workspace, namespace, id) DO UPDATE SET "
            "payload = EXCLUDED.payload, "
            "updated_at = CURRENT_TIMESTAMP"
        )
        for payload_json in chunks:
            payload_dict = json.loads(payload_json)
            keys = list(payload_dict.keys())
            if is_full_docs:
                existing = await self.get_by_ids(keys)
                for idx, (key, existing_payload) in enumerate(
                    zip(keys, existing)
                ):
                    payload_dict[key] = _merge_full_docs_payload(
                        existing_payload, payload_dict[key]
                    )
            values = [
                _deterministic_json(payload_dict[k]) for k in keys
            ]
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    keys,
                    values,
                    descriptor=descriptor,
                    replay_safe=True,
                )
            except Exception:
                raise HologresKVError("Hologres KV upsert failed") from None

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        client, table = self._ready()
        sql = (
            f"DELETE FROM {table} WHERE workspace = $1 AND namespace = $2 "
            "AND id = ANY($3::text[])"
        )
        for chunk in _chunks(ids, _ID_CHUNK_SIZE):
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="kv.delete",
                    replay_safe=True,
                )
            except Exception:
                raise HologresKVError("Hologres KV delete failed") from None

    async def is_empty(self) -> bool:
        client, table = self._ready()
        sql = (
            f"SELECT NOT EXISTS (SELECT 1 FROM {table} "
            "WHERE workspace = $1 AND namespace = $2)"
        )
        try:
            result = await client.fetch_value(
                sql,
                self.workspace,
                self.namespace,
                descriptor="kv.empty",
            )
        except Exception:
            raise HologresKVError("Hologres KV emptiness check failed") from None
        if type(result) is not bool:
            raise HologresKVError("Hologres KV emptiness response is corrupt")
        return result

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        client, table = self._ready()
        sql = f"DELETE FROM {table} WHERE workspace = $1 AND namespace = $2"
        try:
            await client.execute_one(
                sql,
                self.workspace,
                self.namespace,
                descriptor="kv.drop",
                replay_safe=True,
            )
        except Exception:
            raise HologresKVError("Hologres KV drop failed") from None
        return {"status": "success", "message": "data dropped"}
