"""Document-status storage for the isolated Hologres backend."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, ClassVar, final

from ...base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceConflict,
    SourceConflictPage,
    SourceConflictRepairResult,
    SourceConflictSummary,
    SourceResolution,
    SourceUnique,
)
from ...constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from ...exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from ...namespace import NameSpace
from ...utils import logger, validate_workspace
from .capabilities import (
    probe_production_capabilities,
    prove_stream_copy_capability,
)
from .client import OperationKind, quote_qualified_identifier
from .config import HologresConfig
from .kv import _SHARED_CLIENTS, _release_shared_client
from .schema import (
    DOC_STATUS_TABLE_NAME,
    HologresSchemaManager,
    doc_status_schema_descriptors,
)


_ID_CHUNK_SIZE = 1000
_DELETE_CHUNK_SIZE = 1000
_UPSERT_RECORD_LIMIT = 200
_UPSERT_BYTE_LIMIT = 4 * 1024 * 1024
_CONFLICT_SAMPLE_CAP = 32
_MISSING = object()

_EXPLICIT_COLUMNS = (
    "status",
    "created_at",
    "updated_at",
    "file_path",
    "track_id",
    "content_hash",
    "content_summary",
    "content_length",
    "chunks_count",
    "chunks_list",
    "error_msg",
    "metadata",
    "multimodal_processed",
)
_FULL_COLUMNS = "id, " + ", ".join(_EXPLICIT_COLUMNS) + ", extra"
_SCHEDULING_COLUMNS = (
    "id, status, created_at, updated_at, file_path, track_id, metadata"
)
_UPDATABLE_COLUMNS = frozenset(_EXPLICIT_COLUMNS) - {"created_at"}
_TEXT_COLUMNS = frozenset(
    {
        "file_path",
        "track_id",
        "content_hash",
        "content_summary",
        "error_msg",
    }
)
_JSON_COLUMNS = frozenset({"chunks_list", "metadata"})


class HologresDocStatusError(RuntimeError):
    """Raised when a DocStatus operation cannot return a trustworthy result."""


def _row_field(row: Any, name: str) -> Any:
    try:
        return row[name]
    except (KeyError, TypeError, IndexError):
        return _MISSING


def _materialize_rows(rows: Any, label: str) -> list[Any]:
    if rows is None or isinstance(rows, (str, bytes, Mapping)):
        raise HologresDocStatusError(f"Hologres DocStatus {label} response is corrupt")
    try:
        return list(rows)
    except (TypeError, ValueError):
        raise HologresDocStatusError(
            f"Hologres DocStatus {label} response is corrupt"
        ) from None


def _materialize_control_rows(rows: Any, label: str) -> list[Any]:
    try:
        return _materialize_rows(rows, label)
    except HologresDocStatusError:
        raise StorageControlPlaneError(
            f"Hologres DocStatus {label} response is corrupt"
        ) from None


def _deterministic_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise HologresDocStatusError("Hologres DocStatus input is invalid") from None


def _decode_json(
    value: Any,
    expected_type: type,
    label: str,
    *,
    allow_none: bool = False,
) -> Any:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        elif isinstance(value, expected_type):
            value = copy.deepcopy(value)
        elif value is not None or not allow_none:
            raise TypeError
    except (TypeError, ValueError, json.JSONDecodeError):
        raise HologresDocStatusError(
            f"Hologres DocStatus row has corrupt {label}"
        ) from None
    if value is None and allow_none:
        return None
    if not isinstance(value, expected_type):
        raise HologresDocStatusError(
            f"Hologres DocStatus row has corrupt {label}"
        )
    return value


def _parse_datetime(value: Any, label: str) -> datetime:
    try:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            raise TypeError
    except (TypeError, ValueError, OverflowError):
        raise HologresDocStatusError(
            f"Hologres DocStatus {label} is invalid"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_datetime(value: Any, label: str) -> str:
    return _parse_datetime(value, label).isoformat()


def _chunks(values: Sequence[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _conflict_fingerprint(sorted_doc_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for doc_id in sorted_doc_ids:
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _encode_scheduling_cursor(created_at: str, doc_id: str) -> str:
    return json.dumps(
        {"v": 1, "created_at": created_at, "id": doc_id},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_scheduling_cursor(opaque: str) -> tuple[datetime, str]:
    try:
        decoded = json.loads(opaque)
        if not isinstance(decoded, dict) or set(decoded) != {
            "v",
            "created_at",
            "id",
        }:
            raise ValueError("invalid shape")
        if type(decoded["v"]) is not int or decoded["v"] != 1:
            raise ValueError("invalid version")
        if not isinstance(decoded["id"], str):
            raise ValueError("invalid field type")
        created_at = _parse_datetime(decoded["created_at"], "cursor created_at")
    except (HologresDocStatusError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StorageControlPlaneError(
            "Malformed scheduling cursor for HologresDocStatusStorage"
        ) from error
    return created_at, decoded["id"]


def _encode_conflict_cursor(key: str) -> str:
    return json.dumps(
        {"v": 1, "key": key},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_conflict_cursor(opaque: str) -> str:
    try:
        decoded = json.loads(opaque)
        if (
            not isinstance(decoded, dict)
            or set(decoded) != {"v", "key"}
            or type(decoded["v"]) is not int
            or decoded["v"] != 1
            or not isinstance(decoded["key"], str)
        ):
            raise ValueError("invalid conflict cursor")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise StorageControlPlaneError(
            "Malformed source-conflict cursor for HologresDocStatusStorage"
        ) from error
    return decoded["key"]


@final
@dataclass(repr=False)
class HologresDocStatusStorage(DocStatusStorage):
    """One workspace over the fixed shared Hologres DocStatus table."""

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
        if self.namespace != NameSpace.DOC_STATUS:
            raise ValueError("Unsupported Hologres DocStatus namespace")
        try:
            self.workspace = validate_workspace(self.workspace)
        except (TypeError, ValueError):
            raise ValueError("Invalid Hologres DocStatus workspace") from None
        self._lifecycle_lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "HologresDocStatusStorage(<redacted>)"

    async def initialize(self) -> None:
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
                actual_client = await _SHARED_CLIENTS.acquire(config)
                owns_shared = True
            elif config is None:
                config = getattr(actual_client, "config", None)
                if not isinstance(config, HologresConfig):
                    raise HologresDocStatusError(
                        "Hologres DocStatus configuration is unavailable"
                    )

            try:
                capabilities = await probe_production_capabilities(actual_client)
                manager = HologresSchemaManager(actual_client, schema=config.schema)
                await manager.initialize(doc_status_schema_descriptors(config.schema))
                capabilities = await prove_stream_copy_capability(
                    actual_client, capabilities
                )
                apply_capabilities = getattr(actual_client, "apply_capabilities", None)
                if apply_capabilities is not None:
                    apply_capabilities(capabilities)
            except BaseException as initialization_error:
                if owns_shared:
                    try:
                        await _release_shared_client(
                            _SHARED_CLIENTS.release, config, actual_client
                        )
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
        async with self._lifecycle_lock:
            actual_client = self._active_client
            config = self._effective_config
            owns_shared = self._owns_shared_client
            self._active_client = None
            self._effective_config = None
            self._owns_shared_client = False
            self._initialized = False
            if owns_shared and actual_client is not None and config is not None:
                await _release_shared_client(
                    _SHARED_CLIENTS.release, config, actual_client
                )

    def _ready(self) -> tuple[Any, str]:
        if not self._initialized or self._active_client is None:
            raise HologresDocStatusError(
                "Hologres DocStatus storage is not initialized"
            )
        if self._effective_config is None:
            raise HologresDocStatusError(
                "Hologres DocStatus configuration is unavailable"
            )
        table = quote_qualified_identifier(
            self._effective_config.schema, DOC_STATUS_TABLE_NAME
        )
        return self._active_client, table

    @staticmethod
    def _decode_raw_row(row: Any) -> tuple[str, dict[str, Any]]:
        identifier = _row_field(row, "id")
        if not isinstance(identifier, str):
            raise HologresDocStatusError("Hologres DocStatus row is corrupt")

        extra_value = _row_field(row, "extra")
        if extra_value is _MISSING:
            raise HologresDocStatusError("Hologres DocStatus row is corrupt")
        raw = _decode_json(extra_value, dict, "extra")

        for column in _EXPLICIT_COLUMNS:
            value = _row_field(row, column)
            if value is _MISSING:
                raise HologresDocStatusError("Hologres DocStatus row is corrupt")
            raw[column] = value

        try:
            status = raw["status"]
            raw["status"] = (
                status if isinstance(status, DocStatus) else DocStatus(status)
            )
        except (TypeError, ValueError):
            raise HologresDocStatusError(
                "Hologres DocStatus row has corrupt status"
            ) from None
        raw["created_at"] = _format_datetime(raw["created_at"], "created_at")
        raw["updated_at"] = _format_datetime(raw["updated_at"], "updated_at")
        raw["metadata"] = _decode_json(raw["metadata"], dict, "metadata")
        raw["chunks_list"] = _decode_json(
            raw["chunks_list"], list, "chunks_list", allow_none=True
        )
        try:
            DocProcessingStatus.from_stored(copy.deepcopy(raw))
        except (TypeError, ValueError):
            raise HologresDocStatusError("Hologres DocStatus row is corrupt") from None
        return identifier, raw

    @staticmethod
    def _full_status(row: Any) -> tuple[str, DocProcessingStatus]:
        identifier, raw = HologresDocStatusStorage._decode_raw_row(row)
        try:
            status = DocProcessingStatus.from_stored(copy.deepcopy(raw))
        except (TypeError, ValueError):
            raise HologresDocStatusError("Hologres DocStatus row is corrupt") from None
        return identifier, status

    @staticmethod
    def _row_key(row: Any) -> tuple[datetime, str, str]:
        identifier = _row_field(row, "id")
        created_value = _row_field(row, "created_at")
        if not isinstance(identifier, str) or created_value in (_MISSING, None):
            raise HologresDocStatusError(
                "Hologres DocStatus page response is corrupt"
            )
        created = _parse_datetime(created_value, "page created_at")
        return created, identifier, created.isoformat()

    @staticmethod
    def _scheduling_record(row: Any) -> DocSchedulingRecord:
        identifier = _row_field(row, "id")
        raw_status = _row_field(row, "status")
        created_value = _row_field(row, "created_at")
        updated_value = _row_field(row, "updated_at")
        metadata_value = _row_field(row, "metadata")
        if (
            not isinstance(identifier, str)
            or raw_status is _MISSING
            or created_value in (_MISSING, None)
            or updated_value in (_MISSING, None)
            or metadata_value is _MISSING
        ):
            raise HologresDocStatusError(
                "Hologres DocStatus scheduling row is corrupt"
            )
        try:
            status = (
                raw_status
                if isinstance(raw_status, DocStatus)
                else DocStatus(raw_status)
            )
        except (TypeError, ValueError):
            raise HologresDocStatusError(
                "Hologres DocStatus scheduling row has corrupt status"
            ) from None
        metadata = _decode_json(metadata_value, dict, "metadata")
        created_at = _format_datetime(created_value, "created_at")
        updated_at = _format_datetime(updated_value, "updated_at")
        file_path = _row_field(row, "file_path")
        track_id = _row_field(row, "track_id")
        if file_path is _MISSING or track_id is _MISSING:
            raise HologresDocStatusError(
                "Hologres DocStatus scheduling row is corrupt"
            )
        return DocSchedulingRecord(
            id=identifier,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            file_path=file_path,
            track_id=track_id,
            has_custom_chunk_journal=isinstance(
                metadata.get(CUSTOM_CHUNK_PATCH_METADATA_KEY), dict
            ),
        )

    @staticmethod
    def _normalize_complete_input(
        identifier: str, value: Any
    ) -> dict[str, Any]:
        if not isinstance(identifier, str) or not isinstance(value, Mapping):
            raise HologresDocStatusError("Hologres DocStatus input is invalid")
        raw = copy.deepcopy(dict(value))
        raw.pop("content", None)
        raw["file_path"] = raw.get("file_path") or "no-file-path"
        raw.setdefault("metadata", {})
        raw.setdefault("chunks_list", [])
        raw.setdefault("chunks_count", None)
        raw.setdefault("error_msg", None)
        raw.setdefault("track_id", None)
        raw.setdefault("content_hash", None)
        raw.setdefault("multimodal_processed", None)
        try:
            raw_status = raw["status"]
            raw["status"] = (
                raw_status
                if isinstance(raw_status, DocStatus)
                else DocStatus(raw_status)
            )
            raw["created_at"] = _format_datetime(raw["created_at"], "created_at")
            raw["updated_at"] = _format_datetime(raw["updated_at"], "updated_at")
            if not isinstance(raw["metadata"], dict):
                raise TypeError
            if raw["chunks_list"] is not None and not isinstance(
                raw["chunks_list"], list
            ):
                raise TypeError
            document = DocProcessingStatus.from_stored(raw)
        except (KeyError, TypeError, ValueError, HologresDocStatusError):
            raise HologresDocStatusError("Hologres DocStatus input is invalid") from None
        if isinstance(document.content_length, bool) or not isinstance(
            document.content_length, int
        ):
            raise HologresDocStatusError("Hologres DocStatus input is invalid")
        if document.chunks_count is not None and (
            isinstance(document.chunks_count, bool)
            or not isinstance(document.chunks_count, int)
        ):
            raise HologresDocStatusError("Hologres DocStatus input is invalid")

        known = set(_EXPLICIT_COLUMNS)
        extra = {key: value for key, value in raw.items() if key not in known}
        normalized = {
            "id": identifier,
            "status": document.status.value,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
            "file_path": document.file_path,
            "track_id": document.track_id,
            "content_hash": document.content_hash,
            "content_summary": document.content_summary,
            "content_length": document.content_length,
            "chunks_count": document.chunks_count,
            "chunks_list": document.chunks_list,
            "error_msg": document.error_msg,
            "metadata": document.metadata,
            "multimodal_processed": document.multimodal_processed,
            "extra": extra,
        }
        _deterministic_json(normalized)
        return normalized

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        client, table = self._ready()
        sql = (
            f"SELECT id FROM {table} WHERE workspace = $1 "
            "AND id = ANY($2::text[]) ORDER BY id"
        )
        existing: set[str] = set()
        for chunk in _chunks(sorted(keys), _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    list(chunk),
                    descriptor="doc_status.filter",
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus key filter failed"
                ) from None
            for item in _materialize_rows(rows, "filter"):
                identifier = _row_field(item, "id")
                if (
                    not isinstance(identifier, str)
                    or identifier not in chunk
                    or identifier in existing
                ):
                    raise HologresDocStatusError(
                        "Hologres DocStatus filter response is corrupt"
                    )
                existing.add(identifier)
        return keys - existing

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        client, table = self._ready()
        sql = f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 AND id = $2"
        try:
            stored = await client.fetch_one(
                sql, self.workspace, id, descriptor="doc_status.read.one"
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus point read failed"
            ) from None
        if stored is None:
            return None
        _, raw = self._decode_raw_row(stored)
        return raw

    async def get_by_id_strict(self, id: str) -> dict[str, Any] | None:
        return await self.get_by_id(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any] | None]:
        if not ids:
            return []
        client, table = self._ready()
        sql = (
            "SELECT g.idx AS ordinality, ($2::text[])[g.idx] AS requested_id, "
            f"stored.{_FULL_COLUMNS.replace(', ', ', stored.')} "
            "FROM generate_series(1, $3::int) AS g(idx) "
            f"LEFT JOIN {table} AS stored ON stored.workspace = $1 "
            "AND stored.id = ($2::text[])[g.idx] ORDER BY g.idx"
        )
        result: list[dict[str, Any] | None] = []
        for chunk in _chunks(ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    list(chunk),
                    len(chunk),
                    descriptor="doc_status.read.ordered_batch",
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus ordered batch read failed"
                ) from None
            materialized = _materialize_rows(rows, "ordered batch")
            if len(materialized) != len(chunk):
                raise HologresDocStatusError(
                    "Hologres DocStatus ordered batch response is corrupt"
                )
            for ordinal, (requested, item) in enumerate(
                zip(chunk, materialized, strict=True), start=1
            ):
                if (
                    _row_field(item, "ordinality") != ordinal
                    or _row_field(item, "requested_id") != requested
                ):
                    raise HologresDocStatusError(
                        "Hologres DocStatus ordered batch response is corrupt"
                    )
                stored_id = _row_field(item, "id")
                if stored_id is None:
                    result.append(None)
                elif stored_id != requested:
                    raise HologresDocStatusError(
                        "Hologres DocStatus ordered batch response is corrupt"
                    )
                else:
                    _, raw = self._decode_raw_row(item)
                    result.append(raw)
        return result

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        records = [
            self._normalize_complete_input(identifier, value)
            for identifier, value in data.items()
        ]

        client, table = self._ready()
        sql = (
            f"INSERT INTO {table} AS current ("
            "workspace, id, status, created_at, updated_at, file_path, track_id, "
            "content_hash, content_summary, content_length, chunks_count, "
            "chunks_list, error_msg, metadata, multimodal_processed, extra) "
            "VALUES ($1, $2, $3, $4::timestamptz, $5::timestamptz, "
            "$6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16) "
            "ON CONFLICT (workspace, id) DO UPDATE SET "
            "status = EXCLUDED.status, updated_at = EXCLUDED.updated_at, "
            "file_path = EXCLUDED.file_path, track_id = EXCLUDED.track_id, "
            "content_hash = EXCLUDED.content_hash, "
            "content_summary = EXCLUDED.content_summary, "
            "content_length = EXCLUDED.content_length, "
            "chunks_count = EXCLUDED.chunks_count, chunks_list = EXCLUDED.chunks_list, "
            "error_msg = EXCLUDED.error_msg, metadata = EXCLUDED.metadata, "
            "multimodal_processed = EXCLUDED.multimodal_processed, "
            "extra = EXCLUDED.extra"
        )
        for record in records:
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    record["id"],
                    record["status"],
                    datetime.fromisoformat(record["created_at"]),
                    datetime.fromisoformat(record["updated_at"]),
                    record["file_path"],
                    record.get("track_id"),
                    record.get("content_hash"),
                    record.get("content_summary", ""),
                    record.get("content_length"),
                    record.get("chunks_count"),
                    _deterministic_json(
                        record.get("chunks_list"),
                    ),
                    record.get("error_msg"),
                    _deterministic_json(record.get("metadata", {})),
                    record.get("multimodal_processed"),
                    _deterministic_json(record.get("extra", {})),
                    descriptor="doc_status.upsert",
                    replay_safe=True,
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus upsert failed"
                ) from None

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        client, table = self._ready()
        sql = f"DELETE FROM {table} WHERE workspace = $1 AND id = ANY($2::text[])"
        for chunk in _chunks(ids, _DELETE_CHUNK_SIZE):
            try:
                await client.execute_one(
                    sql,
                    self.workspace,
                    list(chunk),
                    descriptor="doc_status.delete",
                    replay_safe=True,
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus delete failed"
                ) from None

    async def is_empty(self) -> bool:
        client, table = self._ready()
        sql = (
            f"SELECT NOT EXISTS (SELECT 1 FROM {table} WHERE workspace = $1)"
        )
        try:
            result = await client.fetch_value(
                sql, self.workspace, descriptor="doc_status.empty"
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus emptiness check failed"
            ) from None
        if type(result) is not bool:
            raise HologresDocStatusError(
                "Hologres DocStatus emptiness response is corrupt"
            )
        return result

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        client, table = self._ready()
        sql = f"DELETE FROM {table} WHERE workspace = $1"
        try:
            await client.execute_one(
                sql,
                self.workspace,
                descriptor="doc_status.drop",
                replay_safe=True,
            )
        except Exception:
            raise HologresDocStatusError("Hologres DocStatus drop failed") from None
        return {"status": "success", "message": "data dropped"}

    async def get_status_counts(self) -> dict[str, int]:
        client, table = self._ready()
        sql = (
            f"SELECT status, COUNT(*)::bigint AS count FROM {table} "
            "WHERE workspace = $1 GROUP BY status ORDER BY status"
        )
        try:
            rows = await client.fetch_all(
                sql, self.workspace, descriptor="doc_status.read.counts"
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus count read failed"
            ) from None
        counts = {status.value: 0 for status in DocStatus}
        seen: set[str] = set()
        for item in _materialize_rows(rows, "counts"):
            status_value = _row_field(item, "status")
            count = _row_field(item, "count")
            if (
                not isinstance(status_value, str)
                or status_value not in counts
                or status_value in seen
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise HologresDocStatusError(
                    "Hologres DocStatus counts response is corrupt"
                )
            seen.add(status_value)
            counts[status_value] = count
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        counts = await self.get_status_counts()
        counts["all"] = sum(counts.values())
        return counts

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        if not statuses:
            return {}
        status_values = [status.value for status in statuses]
        client, table = self._ready()
        sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 "
            "AND status = ANY($2::text[]) ORDER BY created_at, id"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                status_values,
                descriptor="doc_status.read.statuses",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus status read failed"
            ) from None
        result: dict[str, DocProcessingStatus] = {}
        for item in _materialize_rows(rows, "status read"):
            try:
                identifier, document = self._full_status(item)
                if document.status.value not in status_values or identifier in result:
                    raise HologresDocStatusError(
                        "Hologres DocStatus status response is corrupt"
                    )
                result[identifier] = document
            except HologresDocStatusError:
                if strict:
                    raise HologresDocStatusError(
                        "Hologres DocStatus status response is corrupt"
                    ) from None
                logger.error(
                    "[%s] Skipping corrupt Hologres DocStatus status row",
                    self.workspace,
                )
        return result

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        client, table = self._ready()
        sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 "
            "AND track_id = $2 ORDER BY created_at, id"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                track_id,
                descriptor="doc_status.read.track",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus track read failed"
            ) from None
        result: dict[str, DocProcessingStatus] = {}
        for item in _materialize_rows(rows, "track read"):
            try:
                identifier, document = self._full_status(item)
                if document.track_id != track_id or identifier in result:
                    raise HologresDocStatusError(
                        "Hologres DocStatus track response is corrupt"
                    )
                result[identifier] = document
            except HologresDocStatusError:
                logger.error(
                    "[%s] Skipping corrupt Hologres DocStatus track row",
                    self.workspace,
                )
        return result

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        status_filters: list[DocStatus] | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        page = max(1, page)
        page_size = min(200, max(10, page_size))
        if sort_field not in {"created_at", "updated_at", "id", "file_path"}:
            sort_field = "updated_at"
        direction = sort_direction.lower()
        if direction not in {"asc", "desc"}:
            direction = "desc"
        status_values = self.resolve_status_filter_values(status_filter, status_filters)
        ordered_statuses = sorted(status_values) if status_values is not None else None

        client, table = self._ready()
        where = "workspace = $1"
        values: list[Any] = [self.workspace]
        if ordered_statuses is not None:
            where += " AND status = ANY($2::text[])"
            values.append(ordered_statuses)
        count_sql = f"SELECT COUNT(*)::bigint FROM {table} WHERE {where}"
        try:
            total = await client.fetch_value(
                count_sql,
                *values,
                descriptor="doc_status.read.paginated_count",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus paginated count failed"
            ) from None
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            raise HologresDocStatusError(
                "Hologres DocStatus paginated count response is corrupt"
            )
        offset = (page - 1) * page_size
        limit_index = len(values) + 1
        offset_index = limit_index + 1
        page_sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE {where} "
            f"ORDER BY {sort_field} {direction.upper()}, id {direction.upper()} "
            f"LIMIT ${limit_index} OFFSET ${offset_index}"
        )
        try:
            rows = await client.fetch_all(
                page_sql,
                *values,
                page_size,
                offset,
                descriptor="doc_status.read.paginated",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus paginated read failed"
            ) from None
        result: list[tuple[str, DocProcessingStatus]] = []
        seen: set[str] = set()
        for item in _materialize_rows(rows, "paginated"):
            try:
                identifier, document = self._full_status(item)
                if identifier in seen:
                    raise HologresDocStatusError(
                        "Hologres DocStatus paginated response is corrupt"
                    )
                seen.add(identifier)
                result.append((identifier, document))
            except HologresDocStatusError:
                logger.error(
                    "[%s] Skipping corrupt Hologres DocStatus paginated row",
                    self.workspace,
                )
        return result, total

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        client, table = self._ready()
        sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 "
            "AND file_path = $2 ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        try:
            stored = await client.fetch_one(
                sql,
                self.workspace,
                file_path,
                descriptor="doc_status.read.file_path",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus file-path read failed"
            ) from None
        if stored is None:
            return None
        return self._decode_raw_row(stored)[1]

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        if not basename or basename == "unknown_source":
            return None
        client, table = self._ready()
        sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 "
            "AND file_path = $2 "
            "AND NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        try:
            stored = await client.fetch_one(
                sql,
                self.workspace,
                basename,
                descriptor="doc_status.read.basename",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus basename read failed"
            ) from None
        if stored is None:
            return None
        identifier, raw = self._decode_raw_row(stored)
        return identifier, raw

    async def get_doc_by_content_hash(
        self, content_hash: str, *, exclude_doc_id: str | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        if not content_hash:
            return None
        client, table = self._ready()
        sql = (
            f"SELECT {_FULL_COLUMNS} FROM {table} WHERE workspace = $1 "
            "AND content_hash = $2 "
            "AND NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "AND ($3::text IS NULL OR (id <> $3 "
            "AND NOT (COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "AND metadata ->> 'original_doc_id' = $3))) "
            "ORDER BY created_at ASC, id ASC LIMIT 1"
        )
        try:
            stored = await client.fetch_one(
                sql,
                self.workspace,
                content_hash,
                exclude_doc_id,
                descriptor="doc_status.read.content_hash",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus content-hash read failed"
            ) from None
        if stored is None:
            return None
        identifier, raw = self._decode_raw_row(stored)
        return identifier, raw

    async def get_docs_by_statuses_page(
        self,
        statuses: list[DocStatus],
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
        strict: bool = False,
    ) -> DocStatusPage:
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END or not statuses:
            return DocStatusPage(docs={}, next_position=CURSOR_END)
        if position is CURSOR_START:
            cursor: tuple[datetime, str] | None = None
        elif isinstance(position, CursorAfter):
            cursor = _decode_scheduling_cursor(position.opaque)
        else:
            raise StorageControlPlaneError(
                "Malformed scheduling cursor for HologresDocStatusStorage"
            )

        client, table = self._ready()
        status_values = [status.value for status in statuses]
        values: list[Any] = [self.workspace, status_values]
        where = "workspace = $1 AND status = ANY($2::text[])"
        if cursor is not None:
            where += " AND (created_at, id) > ($3::timestamptz, $4::text)"
            values.extend(cursor)
        limit_index = len(values) + 1
        sql = (
            f"SELECT {_SCHEDULING_COLUMNS} FROM {table} WHERE {where} "
            f"ORDER BY created_at ASC, id ASC LIMIT ${limit_index}"
        )
        try:
            rows = await client.fetch_all(
                sql,
                *values,
                limit,
                descriptor="doc_status.read.page",
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus page read failed"
            ) from None
        materialized = _materialize_rows(rows, "page")
        if len(materialized) > limit:
            raise HologresDocStatusError(
                "Hologres DocStatus page response is corrupt"
            )

        documents: dict[str, DocSchedulingRecord] = {}
        previous = cursor
        last_consumed: tuple[datetime, str, str] | None = None
        for item in materialized:
            try:
                current = self._row_key(item)
            except HologresDocStatusError:
                raise HologresDocStatusError(
                    "Hologres DocStatus page response is corrupt"
                ) from None
            comparable = (current[0], current[1])
            if previous is not None and comparable <= previous:
                raise HologresDocStatusError(
                    "Hologres DocStatus page response is corrupt"
                )
            previous = comparable
            last_consumed = current
            try:
                record = self._scheduling_record(item)
                if record.status.value not in status_values or record.id in documents:
                    raise HologresDocStatusError(
                        "Hologres DocStatus page response is corrupt"
                    )
                documents[record.id] = record
            except HologresDocStatusError:
                if strict:
                    raise HologresDocStatusError(
                        "Hologres DocStatus page response is corrupt"
                    ) from None
                logger.error(
                    "[%s] Skipping corrupt Hologres DocStatus scheduling row",
                    self.workspace,
                )
        if len(materialized) < limit:
            next_position: CursorPosition = CURSOR_END
        elif last_consumed is None:
            raise HologresDocStatusError(
                "Hologres DocStatus page response is corrupt"
            )
        else:
            next_position = CursorAfter(
                _encode_scheduling_cursor(last_consumed[2], last_consumed[1])
            )
        return DocStatusPage(docs=documents, next_position=next_position)

    async def _batch_scheduling(
        self, doc_ids: Sequence[str], *, strict: bool
    ) -> dict[str, DocSchedulingRecord]:
        unique_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))
        if not unique_ids:
            return {}
        client, table = self._ready()
        sql = (
            "SELECT g.idx AS ordinality, ($2::text[])[g.idx] AS requested_id, "
            f"stored.{_SCHEDULING_COLUMNS.replace(', ', ', stored.')} "
            "FROM generate_series(1, $3::int) AS g(idx) "
            f"LEFT JOIN {table} AS stored ON stored.workspace = $1 "
            "AND stored.id = ($2::text[])[g.idx] ORDER BY g.idx"
        )
        result: dict[str, DocSchedulingRecord] = {}
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    list(chunk),
                    len(chunk),
                    descriptor="doc_status.read.scheduling_batch",
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus scheduling batch read failed"
                ) from None
            materialized = _materialize_rows(rows, "batch")
            if len(materialized) != len(chunk):
                raise HologresDocStatusError(
                    "Hologres DocStatus batch response is corrupt"
                )
            for ordinal, (requested, item) in enumerate(
                zip(chunk, materialized, strict=True), start=1
            ):
                if (
                    _row_field(item, "ordinality") != ordinal
                    or _row_field(item, "requested_id") != requested
                ):
                    raise HologresDocStatusError(
                        "Hologres DocStatus batch response is corrupt"
                    )
                stored_id = _row_field(item, "id")
                if stored_id is None:
                    continue
                if stored_id != requested or stored_id in result:
                    raise HologresDocStatusError(
                        "Hologres DocStatus batch response is corrupt"
                    )
                try:
                    result[stored_id] = self._scheduling_record(item)
                except HologresDocStatusError:
                    if strict:
                        raise HologresDocStatusError(
                            "Hologres DocStatus batch response is corrupt"
                        ) from None
                    logger.error(
                        "[%s] Skipping corrupt Hologres DocStatus batch row",
                        self.workspace,
                    )
        return result

    async def get_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocSchedulingRecord]:
        return await self._batch_scheduling(doc_ids, strict=strict)

    async def get_full_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        unique_ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids))
        if not unique_ids:
            return {}
        client, table = self._ready()
        sql = (
            "SELECT g.idx AS ordinality, ($2::text[])[g.idx] AS requested_id, "
            f"stored.{_FULL_COLUMNS.replace(', ', ', stored.')} "
            "FROM generate_series(1, $3::int) AS g(idx) "
            f"LEFT JOIN {table} AS stored ON stored.workspace = $1 "
            "AND stored.id = ($2::text[])[g.idx] ORDER BY g.idx"
        )
        result: dict[str, DocProcessingStatus] = {}
        for chunk in _chunks(unique_ids, _ID_CHUNK_SIZE):
            try:
                rows = await client.fetch_all(
                    sql,
                    self.workspace,
                    list(chunk),
                    len(chunk),
                    descriptor="doc_status.read.full_batch",
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus full batch read failed"
                ) from None
            materialized = _materialize_rows(rows, "batch")
            if len(materialized) != len(chunk):
                raise HologresDocStatusError(
                    "Hologres DocStatus batch response is corrupt"
                )
            for ordinal, (requested, item) in enumerate(
                zip(chunk, materialized, strict=True), start=1
            ):
                if (
                    _row_field(item, "ordinality") != ordinal
                    or _row_field(item, "requested_id") != requested
                ):
                    raise HologresDocStatusError(
                        "Hologres DocStatus batch response is corrupt"
                    )
                stored_id = _row_field(item, "id")
                if stored_id is None:
                    continue
                if stored_id != requested or stored_id in result:
                    raise HologresDocStatusError(
                        "Hologres DocStatus batch response is corrupt"
                    )
                try:
                    _, document = self._full_status(item)
                    result[stored_id] = document
                except HologresDocStatusError:
                    if strict:
                        raise HologresDocStatusError(
                            "Hologres DocStatus batch response is corrupt"
                        ) from None
                    logger.error(
                        "[%s] Skipping corrupt Hologres DocStatus full batch row",
                        self.workspace,
                    )
        return result

    async def count_docs_by_statuses(
        self, statuses: list[DocStatus], *, strict: bool = True
    ) -> int:
        if not statuses:
            return 0
        client, table = self._ready()
        sql = (
            f"SELECT COUNT(*)::bigint FROM {table} WHERE workspace = $1 "
            "AND status = ANY($2::text[])"
        )
        try:
            result = await client.fetch_value(
                sql,
                self.workspace,
                [status.value for status in statuses],
                descriptor="doc_status.read.count",
            )
        except Exception:
            raise StorageControlPlaneError(
                "Hologres DocStatus strict count failed"
            ) from None
        if isinstance(result, bool) or not isinstance(result, int) or result < 0:
            raise StorageControlPlaneError(
                "Hologres DocStatus strict count response is corrupt"
            )
        return result

    @staticmethod
    def _normalize_update_value(column: str, value: Any) -> tuple[Any, str]:
        if column == "status":
            try:
                normalized = value.value if isinstance(value, DocStatus) else DocStatus(value).value
            except (TypeError, ValueError):
                raise ValueError("Invalid DocStatus status value") from None
            return normalized, ""
        if column == "updated_at":
            try:
                return _parse_datetime(value, "updated_at"), ""
            except HologresDocStatusError:
                raise ValueError("Invalid DocStatus updated_at value") from None
        if column == "metadata":
            if not isinstance(value, dict):
                raise ValueError("Invalid DocStatus metadata value")
            return _deterministic_json(value), "::jsonb"
        if column == "chunks_list":
            if value is not None and not isinstance(value, list):
                raise ValueError("Invalid DocStatus chunks_list value")
            return _deterministic_json(value), "::jsonb"
        if column in {"content_length", "chunks_count"}:
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"Invalid DocStatus {column} value")
            if column == "content_length" and value is None:
                raise ValueError("Invalid DocStatus content_length value")
            return value, ""
        if column == "multimodal_processed":
            if value is not None and not isinstance(value, bool):
                raise ValueError("Invalid DocStatus multimodal_processed value")
            return value, ""
        if column in _TEXT_COLUMNS:
            if value is not None and not isinstance(value, str):
                raise ValueError(f"Invalid DocStatus {column} value")
            if column in {"file_path", "content_summary"} and value is None:
                raise ValueError(f"Invalid DocStatus {column} value")
            return value, ""
        raise ValueError("Unknown DocStatus update column")

    async def update_doc_status_fields(
        self,
        doc_id: str,
        fields: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> None:
        if "created_at" in fields:
            raise ValueError(
                "created_at is an immutable scheduling sort key and cannot be changed"
            )
        unknown = set(fields) - _UPDATABLE_COLUMNS
        if unknown:
            raise ValueError("update_doc_status_fields received unknown columns")
        client, table = self._ready()
        if not fields:
            sql = f"SELECT EXISTS (SELECT 1 FROM {table} WHERE workspace = $1 AND id = $2)"
            try:
                exists = await client.fetch_value(
                    sql,
                    self.workspace,
                    doc_id,
                    descriptor="doc_status.exists",
                )
            except Exception:
                raise HologresDocStatusError(
                    "Hologres DocStatus existence read failed"
                ) from None
            if type(exists) is not bool:
                raise HologresDocStatusError(
                    "Hologres DocStatus existence response is corrupt"
                )
            if not exists and not missing_ok:
                raise StorageRecordNotFoundError(doc_id)
            return

        values: list[Any] = [self.workspace, doc_id]
        assignments: list[str] = []
        for column, value in fields.items():
            normalized, cast = self._normalize_update_value(column, value)
            values.append(normalized)
            assignments.append(f"{column} = ${len(values)}{cast}")
        sql = (
            f"UPDATE {table} SET {', '.join(assignments)} "
            "WHERE workspace = $1 AND id = $2 RETURNING id"
        )
        try:
            updated = await client.fetch_one(
                sql,
                *values,
                descriptor="doc_status.update",
                operation_kind=OperationKind.WRITE,
                replay_safe=True,
            )
        except Exception:
            raise HologresDocStatusError(
                "Hologres DocStatus targeted update failed"
            ) from None
        if updated is None:
            if missing_ok:
                return
            raise StorageRecordNotFoundError(doc_id)
        if _row_field(updated, "id") != doc_id:
            raise HologresDocStatusError(
                "Hologres DocStatus targeted update response is corrupt"
            )

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:
        if not canonical_source_key or canonical_source_key == "unknown_source":
            return SourceAbsent()
        client, table = self._ready()
        sql = (
            f"SELECT {_SCHEDULING_COLUMNS}, COUNT(*) OVER()::bigint AS candidate_count "
            f"FROM {table} WHERE workspace = $1 AND file_path = $2 "
            "AND NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "ORDER BY id ASC LIMIT $3"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                canonical_source_key,
                _CONFLICT_SAMPLE_CAP,
                descriptor="doc_status.read.source",
            )
        except Exception:
            raise StorageControlPlaneError(
                "Hologres DocStatus source resolution failed"
            ) from None
        materialized = _materialize_control_rows(rows, "source resolution")
        if not materialized:
            return SourceAbsent()
        samples: list[str] = []
        candidate_count: int | None = None
        previous_id: str | None = None
        unique_record: DocSchedulingRecord | None = None
        for item in materialized:
            try:
                record = self._scheduling_record(item)
            except HologresDocStatusError:
                raise StorageControlPlaneError(
                    "Hologres DocStatus source resolution response is corrupt"
                ) from None
            count_value = _row_field(item, "candidate_count")
            if (
                count_value is _MISSING
                or isinstance(count_value, bool)
                or not isinstance(count_value, int)
                or count_value < 1
                or record.file_path != canonical_source_key
                or (candidate_count is not None and count_value != candidate_count)
                or (previous_id is not None and record.id <= previous_id)
            ):
                raise StorageControlPlaneError(
                    "Hologres DocStatus source resolution response is corrupt"
                )
            candidate_count = count_value
            previous_id = record.id
            unique_record = record
            samples.append(record.id)
        if candidate_count is None or len(samples) != min(
            candidate_count, _CONFLICT_SAMPLE_CAP
        ):
            raise StorageControlPlaneError(
                "Hologres DocStatus source resolution response is corrupt"
            )
        if candidate_count == 1:
            assert unique_record is not None
            return SourceUnique(doc_id=unique_record.id, doc=unique_record)
        return SourceConflict(
            candidate_count=candidate_count,
            sample_doc_ids=tuple(samples),
        )

    async def list_source_conflicts_page(
        self,
        *,
        limit: int,
        position: CursorPosition = CURSOR_START,
    ) -> SourceConflictPage:
        if limit <= 0:
            raise ValueError(f"page limit must be positive, got {limit}")
        if position is CURSOR_END:
            return SourceConflictPage(conflicts=(), next_position=CURSOR_END)
        if position is CURSOR_START:
            last_key: str | None = None
        elif isinstance(position, CursorAfter):
            last_key = _decode_conflict_cursor(position.opaque)
        else:
            raise StorageControlPlaneError(
                "Malformed source-conflict cursor for HologresDocStatusStorage"
            )
        client, table = self._ready()
        sql = (
            "WITH ranked AS (SELECT file_path, id, "
            "COUNT(*) OVER (PARTITION BY file_path) AS candidate_count, "
            "ROW_NUMBER() OVER (PARTITION BY file_path ORDER BY id) AS sample_rank "
            f"FROM {table} WHERE workspace = $1 "
            "AND file_path NOT IN ('', 'unknown_source', 'no-file-path') "
            "AND NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "AND ($2::text IS NULL OR file_path > $2)) "
            "SELECT file_path AS canonical_source_key, "
            "MAX(candidate_count)::bigint AS candidate_count, "
            "ARRAY_AGG(id ORDER BY id) "
            f"FILTER (WHERE sample_rank <= {_CONFLICT_SAMPLE_CAP}) AS sample_doc_ids "
            "FROM ranked GROUP BY file_path HAVING MAX(candidate_count) > 1 "
            "ORDER BY file_path ASC LIMIT $3"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                last_key,
                limit,
                descriptor="doc_status.read.conflicts",
            )
        except Exception:
            raise StorageControlPlaneError(
                "Hologres DocStatus conflict listing failed"
            ) from None
        materialized = _materialize_control_rows(rows, "conflict listing")
        if len(materialized) > limit:
            raise StorageControlPlaneError(
                "Hologres DocStatus conflict listing response is corrupt"
            )
        summaries: list[SourceConflictSummary] = []
        previous = last_key
        for item in materialized:
            key = _row_field(item, "canonical_source_key")
            count = _row_field(item, "candidate_count")
            sample = _row_field(item, "sample_doc_ids")
            if (
                not isinstance(key, str)
                or not key
                or (previous is not None and key <= previous)
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 2
                or not isinstance(sample, (list, tuple))
                or any(not isinstance(identifier, str) for identifier in sample)
                or list(sample) != sorted(set(sample))
                or len(sample) > _CONFLICT_SAMPLE_CAP
                or len(sample) > count
            ):
                raise StorageControlPlaneError(
                    "Hologres DocStatus conflict listing response is corrupt"
                )
            previous = key
            summaries.append(
                SourceConflictSummary(
                    canonical_source_key=key,
                    candidate_count=count,
                    sample_doc_ids=tuple(sample),
                )
            )
        if len(materialized) < limit:
            next_position: CursorPosition = CURSOR_END
        elif previous is None:
            raise StorageControlPlaneError(
                "Hologres DocStatus conflict listing response is corrupt"
            )
        else:
            next_position = CursorAfter(_encode_conflict_cursor(previous))
        return SourceConflictPage(
            conflicts=tuple(summaries), next_position=next_position
        )

    async def _primary_candidates(
        self, canonical_source_key: str
    ) -> list[str]:
        client, table = self._ready()
        sql = (
            f"SELECT id FROM {table} WHERE workspace = $1 AND file_path = $2 "
            "AND NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "ORDER BY id ASC"
        )
        try:
            rows = await client.fetch_all(
                sql,
                self.workspace,
                canonical_source_key,
                descriptor="doc_status.repair.candidates",
            )
        except Exception:
            raise StorageControlPlaneError(
                "Hologres DocStatus conflict repair candidate read failed"
            ) from None
        candidates: list[str] = []
        for item in _materialize_control_rows(rows, "conflict repair candidates"):
            identifier = _row_field(item, "id")
            if (
                not isinstance(identifier, str)
                or identifier in candidates
                or (candidates and identifier <= candidates[-1])
            ):
                raise StorageControlPlaneError(
                    "Hologres DocStatus conflict repair candidate response is corrupt"
                )
            candidates.append(identifier)
        return candidates

    async def repair_source_conflict(
        self,
        canonical_source_key: str,
        *,
        primary_doc_id: str,
        expected_candidate_count: int,
        expected_candidate_fingerprint: str,
        dry_run: bool = True,
    ) -> SourceConflictRepairResult:
        candidates = await self._primary_candidates(canonical_source_key)
        count = len(candidates)
        fingerprint = _conflict_fingerprint(candidates)
        if primary_doc_id not in candidates:
            raise ValueError("primary_doc_id is not a current primary candidate")
        demoted = [identifier for identifier in candidates if identifier != primary_doc_id]
        if dry_run:
            return SourceConflictRepairResult(
                canonical_source_key=canonical_source_key,
                primary_doc_id=primary_doc_id,
                candidate_count=count,
                fingerprint=fingerprint,
                demoted_sample_doc_ids=tuple(demoted[:_CONFLICT_SAMPLE_CAP]),
                committed=False,
            )
        if (
            count != expected_candidate_count
            or fingerprint != expected_candidate_fingerprint
        ):
            raise SourceConflictRepairCASError(
                "Hologres DocStatus source-conflict repair CAS failed"
            )

        client, table = self._ready()
        sql = (
            f"UPDATE {table} SET metadata = metadata || "
            "jsonb_build_object('is_duplicate', TRUE, 'original_doc_id', $4::text) "
            "WHERE workspace = $1 AND id = $2 AND file_path = $3 AND ("
            "NOT COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) OR ("
            "COALESCE((metadata ->> 'is_duplicate')::boolean, FALSE) "
            "AND metadata ->> 'original_doc_id' = $4)) RETURNING id"
        )
        for identifier in demoted:
            try:
                updated = await client.fetch_one(
                    sql,
                    self.workspace,
                    identifier,
                    canonical_source_key,
                    primary_doc_id,
                    descriptor="doc_status.repair.demote",
                    operation_kind=OperationKind.WRITE,
                    replay_safe=True,
                )
            except Exception:
                raise StorageControlPlaneError(
                    "Hologres DocStatus source-conflict repair failed"
                ) from None
            if _row_field(updated, "id") != identifier:
                raise StorageControlPlaneError(
                    "Hologres DocStatus source-conflict repair CAS was refused"
                )

        remaining = await self._primary_candidates(canonical_source_key)
        if remaining != [primary_doc_id]:
            raise StorageControlPlaneError(
                "Hologres DocStatus source-conflict repair could not be proven"
            )
        return SourceConflictRepairResult(
            canonical_source_key=canonical_source_key,
            primary_doc_id=primary_doc_id,
            candidate_count=count,
            fingerprint=fingerprint,
            demoted_sample_doc_ids=tuple(demoted[:_CONFLICT_SAMPLE_CAP]),
            committed=True,
        )
