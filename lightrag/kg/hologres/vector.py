"""Shared-table vector storage for the isolated Hologres backend."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from numbers import Real
from typing import Any, final

from ...base import BaseVectorStorage
from ...constants import DEFAULT_QUERY_PRIORITY
from ...namespace import NameSpace
from ...utils import compute_mdhash_id, validate_workspace
from .capabilities import (
    probe_production_capabilities,
    prove_stream_copy_capability,
)
from .client import STREAM_COPY_MIN_ROWS, quote_qualified_identifier
from .config import HologresConfig
from .kv import _SHARED_CLIENTS, _release_shared_client
from .schema import (
    VECTOR_TABLE_NAME,
    HologresSchemaManager,
    vector_schema_descriptors,
)


_ID_CHUNK_SIZE = 1000
_UPSERT_RECORD_LIMIT = 200
_UPSERT_BYTE_LIMIT = 4 * 1024 * 1024
_FLOAT4_MAX = 3.4028234663852886e38
_MISSING = object()
_ALLOWED_NAMESPACES = frozenset(
    {
        NameSpace.VECTOR_STORE_ENTITIES,
        NameSpace.VECTOR_STORE_RELATIONSHIPS,
        NameSpace.VECTOR_STORE_CHUNKS,
    }
)


class HologresVectorError(RuntimeError):
    """Raised when a vector operation cannot return a trustworthy result."""


def _row_field(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return _MISSING


def _deterministic_json(value: Any, message: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        raise HologresVectorError(message) from None


def _decode_payload(value: Any) -> dict[str, Any]:
    try:
        if isinstance(value, str):
            decoded = json.loads(value)
        elif isinstance(value, Mapping):
            decoded = json.loads(
                _deterministic_json(
                    dict(value), "Hologres vector row is corrupt"
                )
            )
        else:
            raise TypeError
    except (TypeError, ValueError, json.JSONDecodeError, HologresVectorError):
        raise HologresVectorError("Hologres vector row is corrupt") from None
    if not isinstance(decoded, dict):
        raise HologresVectorError("Hologres vector row is corrupt")
    return decoded


def _materialize_rows(rows: Any, message: str) -> list[Any]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise HologresVectorError(message)
    try:
        return list(rows)
    except (TypeError, ValueError):
        raise HologresVectorError(message) from None


def _normalize_vector(value: Any, dimension: int, message: str) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise HologresVectorError(message)
    try:
        values = list(value)
    except (TypeError, ValueError):
        raise HologresVectorError(message) from None
    if len(values) != dimension:
        raise HologresVectorError(message)

    normalized: list[float] = []
    for component in values:
        if isinstance(component, bool) or not isinstance(component, Real):
            raise HologresVectorError(message)
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError):
            raise HologresVectorError(message) from None
        if not math.isfinite(number) or abs(number) > _FLOAT4_MAX:
            raise HologresVectorError(message)
        normalized.append(number)
    return normalized


def _chunks(values: Sequence[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _upsert_payloads(records: Sequence[dict[str, Any]]) -> list[str]:
    payloads: list[str] = []
    current: list[dict[str, Any]] = []
    current_json = ""
    for record in records:
        single_json = _deterministic_json(
            [record], "Hologres vector input is invalid"
        )
        if len(single_json.encode("utf-8")) > _UPSERT_BYTE_LIMIT:
            raise HologresVectorError(
                "Hologres vector record exceeds the upsert batch limit"
            )
        candidate = [*current, record]
        candidate_json = _deterministic_json(
            candidate, "Hologres vector input is invalid"
        )
        if current and (
            len(current) >= _UPSERT_RECORD_LIMIT
            or len(candidate_json.encode("utf-8")) > _UPSERT_BYTE_LIMIT
        ):
            payloads.append(current_json)
            current = [record]
            current_json = single_json
        else:
            current = candidate
            current_json = candidate_json
    if current:
        payloads.append(current_json)
    return payloads


@final
@dataclass(repr=False)
class HologresVectorStorage(BaseVectorStorage):
    """One logical vector namespace over the fixed shared Hologres table."""

    config: HologresConfig | None = field(default=None, repr=False)
    client: Any | None = field(default=None, repr=False)
    _active_client: Any | None = field(default=None, init=False, repr=False)
    _effective_config: HologresConfig | None = field(
        default=None, init=False, repr=False
    )
    _owns_shared_client: bool = field(default=False, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _dimension: int = field(default=0, init=False, repr=False)
    _embedding_batch_size: int = field(default=0, init=False, repr=False)
    _lifecycle_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_embedding_func()

        dimension = getattr(self.embedding_func, "embedding_dim", None)
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Invalid Hologres vector embedding dimension")
        self._dimension = dimension

        if not isinstance(self.namespace, str) or self.namespace not in _ALLOWED_NAMESPACES:
            raise ValueError("Unsupported Hologres vector namespace")
        try:
            self.workspace = validate_workspace(self.workspace)
        except (TypeError, ValueError):
            raise ValueError("Invalid Hologres vector workspace") from None

        batch_size = self.global_config.get("embedding_batch_num")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Invalid Hologres vector embedding batch size")
        self._embedding_batch_size = batch_size

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        threshold = kwargs.get("cosine_better_than_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or not -1.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("Invalid Hologres vector cosine threshold")
        self.cosine_better_than_threshold = float(threshold)
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "HologresVectorStorage(<redacted>)"

    async def initialize(self) -> None:
        """Acquire one client and reconcile the dimension-bound table."""

        async with self._lifecycle_lock:
            if self._initialized:
                return

            actual_client = self.client
            config = self.config
            owns_shared = False
            try:
                if actual_client is None:
                    if config is None:
                        config = HologresConfig.from_env()
                        self.config = config
                    actual_client = await _SHARED_CLIENTS.acquire(config)
                    owns_shared = True
                elif config is None:
                    config = getattr(actual_client, "config", None)
                    if not isinstance(config, HologresConfig):
                        raise HologresVectorError(
                            "Hologres vector configuration is unavailable"
                        )

                capabilities = await probe_production_capabilities(actual_client)
                manager = HologresSchemaManager(actual_client, schema=config.schema)
                await manager.initialize(
                    vector_schema_descriptors(config.schema, self._dimension)
                )
                capabilities = await prove_stream_copy_capability(
                    actual_client, capabilities
                )
                apply_capabilities = getattr(actual_client, "apply_capabilities", None)
                if apply_capabilities is not None:
                    apply_capabilities(capabilities)
            except BaseException as initialization_error:
                release_error = None
                if owns_shared and actual_client is not None and config is not None:
                    try:
                        await _release_shared_client(
                            _SHARED_CLIENTS.release, config, actual_client
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException:
                        release_error = HologresVectorError(
                            "Hologres vector shared client release failed"
                        )
                if isinstance(initialization_error, asyncio.CancelledError):
                    if release_error is not None:
                        raise initialization_error from release_error
                    raise
                error = HologresVectorError(
                    "Hologres vector initialization failed"
                )
                if release_error is not None:
                    raise error from release_error
                raise error from None

            self._active_client = actual_client
            self._effective_config = config
            self._owns_shared_client = owns_shared
            self._initialized = True

    async def finalize(self) -> None:
        """Detach this storage and release only a manager-owned client."""

        async with self._lifecycle_lock:
            actual_client = self._active_client
            config = self._effective_config
            owns_shared = self._owns_shared_client
            self._active_client = None
            self._effective_config = None
            self._owns_shared_client = False
            self._initialized = False
            if owns_shared and actual_client is not None and config is not None:
                try:
                    await _release_shared_client(
                        _SHARED_CLIENTS.release, config, actual_client
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    raise HologresVectorError(
                        "Hologres vector finalization failed"
                    ) from None

    def _ready(self) -> tuple[Any, str]:
        if not self._initialized or self._active_client is None:
            raise HologresVectorError("Hologres vector storage is not initialized")
        if self._effective_config is None:
            raise HologresVectorError(
                "Hologres vector configuration is unavailable"
            )
        table = quote_qualified_identifier(
            self._effective_config.schema, VECTOR_TABLE_NAME
        )
        return self._active_client, table

    async def _embed_documents(self, contents: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        embedding_func = self.embedding_func
        if embedding_func is None:
            raise HologresVectorError("Hologres vector embedding is unavailable")
        for start in range(0, len(contents), self._embedding_batch_size):
            batch = list(contents[start : start + self._embedding_batch_size])
            try:
                raw_batch = await embedding_func(batch, context="document")
            except Exception:
                raise HologresVectorError(
                    "Hologres vector embedding failed"
                ) from None
            try:
                materialized = list(raw_batch)
            except (TypeError, ValueError):
                raise HologresVectorError(
                    "Hologres vector embedding result is invalid"
                ) from None
            if len(materialized) != len(batch):
                raise HologresVectorError(
                    "Hologres vector embedding result is invalid"
                )
            for vector in materialized:
                embeddings.append(
                    _normalize_vector(
                        vector,
                        self._dimension,
                        "Hologres vector embedding result is invalid",
                    )
                )
        return embeddings

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        prepared: list[dict[str, Any]] = []
        missing_positions: list[int] = []
        missing_contents: list[str] = []
        for identifier, raw_payload in data.items():
            if not isinstance(identifier, str) or not isinstance(raw_payload, Mapping):
                raise HologresVectorError("Hologres vector input is invalid")
            payload = dict(raw_payload)
            has_embedding = "embedding" in payload
            has_internal_vector = "__vector__" in payload
            if has_embedding and has_internal_vector:
                raise HologresVectorError("Hologres vector is invalid")
            supplied = payload.pop("embedding", _MISSING)
            if supplied is _MISSING:
                supplied = payload.pop("__vector__", _MISSING)
            else:
                payload.pop("__vector__", None)
            content = payload.get("content")
            if not isinstance(content, str):
                raise HologresVectorError("Hologres vector input is invalid")
            normalized_payload = json.loads(
                _deterministic_json(
                    payload, "Hologres vector input is invalid"
                )
            )
            record = {
                "id": identifier,
                "embedding": None,
                "content": content,
                "payload": normalized_payload,
            }
            if supplied is _MISSING:
                missing_positions.append(len(prepared))
                missing_contents.append(content)
            else:
                record["embedding"] = _normalize_vector(
                    supplied,
                    self._dimension,
                    "Hologres vector is invalid",
                )
            prepared.append(record)

        if missing_contents:
            generated = await self._embed_documents(missing_contents)
            if len(generated) != len(missing_positions):
                raise HologresVectorError(
                    "Hologres vector embedding result is invalid"
                )
            for position, vector in zip(
                missing_positions, generated, strict=True
            ):
                prepared[position]["embedding"] = vector

        payloads = _upsert_payloads(prepared)
        client, table = self._ready()
        sql = (
            f"INSERT INTO {table} AS current ("
            "workspace, namespace, id, embedding, content, payload, updated_at) "
            "SELECT $1, $2, ($3::text[])[g.idx], "
            "(($4::text[])[g.idx])::float4[], "
            "($5::text[])[g.idx], ($6::jsonb[])[g.idx], CURRENT_TIMESTAMP "
            "FROM generate_series(1, $7::int) AS g(idx) "
            "ON CONFLICT (workspace, namespace, id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, content = EXCLUDED.content, "
            "payload = EXCLUDED.payload, updated_at = CURRENT_TIMESTAMP"
        )
        use_stream_copy = bool(getattr(client, "stream_copy_available", False))
        for payload_json in payloads:
            records = json.loads(payload_json)
            ids = [r["id"] for r in records]
            contents = [r["content"] for r in records]
            payload_values = [
                _deterministic_json(r["payload"], "Hologres vector input is invalid")
                for r in records
            ]
            if use_stream_copy and len(ids) >= STREAM_COPY_MIN_ROWS:
                updated_at = datetime.now(timezone.utc)
                rows = [
                    (
                        self.workspace,
                        self.namespace,
                        record["id"],
                        [float(value) for value in record["embedding"]],
                        record["content"],
                        payload_value,
                        updated_at,
                    )
                    for record, payload_value in zip(records, payload_values)
                ]
                try:
                    await client.copy_rows(
                        VECTOR_TABLE_NAME,
                        (
                            "workspace",
                            "namespace",
                            "id",
                            "embedding",
                            "content",
                            "payload",
                            "updated_at",
                        ),
                        rows,
                        descriptor="vector.upsert",
                        replay_safe=True,
                    )
                except Exception:
                    raise HologresVectorError(
                        "Hologres vector upsert failed"
                    ) from None
                continue
            embeddings = [
                "{" + ",".join(str(float(x)) for x in r["embedding"]) + "}"
                for r in records
            ]
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    self.namespace,
                    ids,
                    embeddings,
                    contents,
                    payload_values,
                    len(ids),
                    descriptor="vector.upsert",
                    replay_safe=True,
                )
            except Exception:
                raise HologresVectorError(
                    "Hologres vector upsert failed"
                ) from None

    @staticmethod
    def _decode_row(row: Any, expected_id: str) -> dict[str, Any]:
        identifier = _row_field(row, "id")
        content = _row_field(row, "content")
        payload_value = _row_field(row, "payload")
        if (
            identifier != expected_id
            or not isinstance(identifier, str)
            or not isinstance(content, str)
            or payload_value is _MISSING
            or payload_value is None
        ):
            raise HologresVectorError("Hologres vector row is corrupt")
        payload = _decode_payload(payload_value)
        stored_content = payload.get("content", content)
        if stored_content != content:
            raise HologresVectorError("Hologres vector row is corrupt")
        payload["content"] = content
        payload["id"] = identifier
        return payload

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        client, table = self._ready()
        sql = (
            f"SELECT id, content, payload FROM {table} "
            "WHERE workspace = $1 AND namespace = $2 AND id = $3"
        )
        try:
            row = await client.fetch_one(
                sql,
                self.workspace,
                self.namespace,
                id,
                descriptor="vector.read.one",
            )
        except Exception:
            raise HologresVectorError("Hologres vector point read failed") from None
        if row is None:
            return None
        return self._decode_row(row, id)

    async def get_by_ids(
        self, ids: list[str]
    ) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        if any(not isinstance(identifier, str) for identifier in ids):
            raise HologresVectorError("Hologres vector input is invalid")
        client, table = self._ready()
        sql = (
            "SELECT idx AS ordinality, stored.id, stored.content, stored.payload "
            "FROM generate_series(1, $3::int) AS g(idx) "
            f"LEFT JOIN {table} AS stored "
            "ON stored.workspace = $1 AND stored.namespace = $2 "
            "AND stored.id = ($4::text[])[g.idx] ORDER BY idx"
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
                    descriptor="vector.read.batch",
                )
            except Exception:
                raise HologresVectorError(
                    "Hologres vector batch read failed"
                ) from None
            materialized = _materialize_rows(
                rows, "Hologres vector batch response is corrupt"
            )
            if len(materialized) != len(chunk):
                raise HologresVectorError(
                    "Hologres vector batch response is corrupt"
                )
            for ordinal, (requested, row) in enumerate(
                zip(chunk, materialized, strict=True), start=1
            ):
                identifier = _row_field(row, "id")
                content = _row_field(row, "content")
                payload = _row_field(row, "payload")
                observed_ordinality = _row_field(row, "ordinality")
                if (
                    isinstance(observed_ordinality, bool)
                    or not isinstance(observed_ordinality, int)
                    or observed_ordinality != ordinal
                ):
                    raise HologresVectorError(
                        "Hologres vector batch response is corrupt"
                    )
                if identifier is None:
                    if content is not None or payload is not None:
                        raise HologresVectorError(
                            "Hologres vector batch response is corrupt"
                        )
                    result.append(None)
                    continue
                if identifier != requested:
                    raise HologresVectorError(
                        "Hologres vector batch response is corrupt"
                    )
                try:
                    result.append(self._decode_row(row, requested))
                except HologresVectorError:
                    raise HologresVectorError(
                        "Hologres vector batch response is corrupt"
                    ) from None
        return result

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        if any(not isinstance(identifier, str) for identifier in ids):
            raise HologresVectorError("Hologres vector input is invalid")
        unique_ids = list(dict.fromkeys(ids))
        client, table = self._ready()
        sql = (
            f"SELECT id, embedding FROM {table} "
            "WHERE workspace = $1 AND namespace = $2 "
            "AND id = ANY($3::text[]) ORDER BY id"
        )
        result: dict[str, list[float]] = {}
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    self.namespace,
                    list(chunk),
                    descriptor="vector.read.vectors",
                )
            except Exception:
                raise HologresVectorError(
                    "Hologres vector read failed"
                ) from None
            materialized = _materialize_rows(
                rows, "Hologres vector response is corrupt"
            )
            requested = set(chunk)
            for row in materialized:
                identifier = _row_field(row, "id")
                embedding = _row_field(row, "embedding")
                if (
                    not isinstance(identifier, str)
                    or identifier not in requested
                    or identifier in result
                    or embedding is _MISSING
                ):
                    raise HologresVectorError(
                        "Hologres vector response is corrupt"
                    )
                result[identifier] = _normalize_vector(
                    embedding,
                    self._dimension,
                    "Hologres vector response is corrupt",
                )
        return result

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise HologresVectorError("Hologres vector top_k is invalid")

        if query_embedding is None:
            embedding_func = self.embedding_func
            if embedding_func is None:
                raise HologresVectorError(
                    "Hologres vector embedding is unavailable"
                )
            try:
                raw = await embedding_func(
                    [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
                )
                materialized = list(raw)
            except Exception:
                raise HologresVectorError(
                    "Hologres vector embedding failed"
                ) from None
            if len(materialized) != 1:
                raise HologresVectorError(
                    "Hologres vector embedding result is invalid"
                )
            vector = _normalize_vector(
                materialized[0],
                self._dimension,
                "Hologres vector embedding result is invalid",
            )
        else:
            vector = _normalize_vector(
                query_embedding,
                self._dimension,
                "Hologres vector is invalid",
            )

        client, table = self._ready()
        # approx_cosine_distance returns cosine SIMILARITY (higher is
        # closer); the contract is frozen by the live HGraph probe.
        sql = (
            "SELECT id, content, payload, "
            "approx_cosine_distance(embedding, $3::float4[]) AS score, "
            "EXTRACT(EPOCH FROM updated_at)::bigint AS created_at "
            f"FROM {table} "
            "WHERE workspace = $1 AND namespace = $2 "
            "AND approx_cosine_distance(embedding, $3::float4[]) "
            "> $4::float8 "
            "ORDER BY score DESC LIMIT $5::int"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                self.namespace,
                vector,
                self.cosine_better_than_threshold,
                top_k,
                descriptor="vector.query",
            )
        except Exception:
            raise HologresVectorError("Hologres vector query failed") from None
        materialized_rows = _materialize_rows(
            rows, "Hologres vector query response is corrupt"
        )
        results: list[dict[str, Any]] = []
        for row in materialized_rows:
            identifier = _row_field(row, "id")
            score = _row_field(row, "score")
            created_at = _row_field(row, "created_at")
            if (
                not isinstance(identifier, str)
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise HologresVectorError(
                    "Hologres vector query response is corrupt"
                )
            try:
                decoded = self._decode_row(row, identifier)
            except HologresVectorError:
                raise HologresVectorError(
                    "Hologres vector query response is corrupt"
                ) from None
            decoded["distance"] = float(score)
            decoded["created_at"] = (
                created_at
                if not isinstance(created_at, bool)
                and isinstance(created_at, int)
                else None
            )
            results.append(decoded)
        return results

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if any(not isinstance(identifier, str) for identifier in ids):
            raise HologresVectorError("Hologres vector input is invalid")
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
                    descriptor="vector.delete",
                    replay_safe=True,
                )
            except Exception:
                raise HologresVectorError("Hologres vector delete failed") from None

    async def delete_entity(self, entity_name: str) -> None:
        await self.delete([compute_mdhash_id(entity_name, prefix="ent-")])

    async def delete_entity_relation(self, entity_name: str) -> None:
        client, table = self._ready()
        sql = (
            f"DELETE FROM {table} WHERE workspace = $1 AND namespace = $2 "
            "AND (payload ->> 'src_id' = $3 OR payload ->> 'tgt_id' = $3)"
        )
        try:
            await client.execute_one(
                sql,
                self.workspace,
                self.namespace,
                entity_name,
                descriptor="vector.delete.relations",
                replay_safe=True,
            )
        except Exception:
            raise HologresVectorError(
                "Hologres vector relation delete failed"
            ) from None

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
                descriptor="vector.drop",
                replay_safe=True,
            )
        except Exception:
            raise HologresVectorError("Hologres vector drop failed") from None
        return {"status": "success", "message": "data dropped"}
