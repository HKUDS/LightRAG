"""Unit tests for the PostgreSQL batch-limit helpers.

Covers the module-level batching primitives shared by the PG non-graph write
paths (mirrors the MONGO_* / OPENSEARCH_* contract):

- ``_resolve_pg_batch_limits``: env parsing, defaults, non-positive disable.
- ``_estimate_record_bytes``: per-field byte estimation for asyncpg tuples.
- ``_chunk_by_budget``: record-count + payload-byte splitting.
"""

import json
from unittest.mock import patch

import numpy as np

from lightrag.kg.postgres_impl import (
    DEFAULT_PG_DELETE_MAX_RECORDS_PER_BATCH,
    DEFAULT_PG_UPSERT_MAX_PAYLOAD_BYTES,
    DEFAULT_PG_UPSERT_MAX_RECORDS_PER_BATCH,
    _chunk_by_budget,
    _estimate_record_bytes,
    _resolve_pg_batch_limits,
)


# ---------------------------------------------------------------------------
# _resolve_pg_batch_limits
# ---------------------------------------------------------------------------


class TestResolvePgBatchLimits:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            payload, upserts, deletes = _resolve_pg_batch_limits()
        assert payload == DEFAULT_PG_UPSERT_MAX_PAYLOAD_BYTES
        assert upserts == DEFAULT_PG_UPSERT_MAX_RECORDS_PER_BATCH
        assert deletes == DEFAULT_PG_DELETE_MAX_RECORDS_PER_BATCH

    def test_default_record_cap_keeps_kv_historical_200(self):
        # The PG-wide upsert record default deliberately stays at 200 (KV's
        # historical batch size), not Mongo/OS's 128.
        assert DEFAULT_PG_UPSERT_MAX_RECORDS_PER_BATCH == 200

    def test_env_override(self):
        env = {
            "POSTGRES_UPSERT_MAX_PAYLOAD_BYTES": "12345",
            "POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH": "7",
            "POSTGRES_DELETE_MAX_RECORDS_PER_BATCH": "9",
        }
        with patch.dict("os.environ", env, clear=True):
            assert _resolve_pg_batch_limits() == (12345, 7, 9)

    def test_non_positive_disables_and_warns(self):
        env = {
            "POSTGRES_UPSERT_MAX_PAYLOAD_BYTES": "0",
            "POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH": "-1",
            "POSTGRES_DELETE_MAX_RECORDS_PER_BATCH": "0",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("lightrag.kg.postgres_impl.logger") as mock_logger:
                payload, upserts, deletes = _resolve_pg_batch_limits()
        assert (payload, upserts, deletes) == (0, -1, 0)
        warnings = [c.args[0] for c in mock_logger.warning.call_args_list]
        assert sum("non-positive" in msg for msg in warnings) == 3


# ---------------------------------------------------------------------------
# _estimate_record_bytes
# ---------------------------------------------------------------------------


class TestEstimateRecordBytes:
    def test_ndarray_uses_nbytes(self):
        vec = np.zeros(128, dtype=np.float32)
        # 128 * 4 bytes (nbytes) plus the 2-byte "id" str.
        assert _estimate_record_bytes(("id", vec)) == vec.nbytes + 2

    def test_str_uses_utf8_length(self):
        assert _estimate_record_bytes(("abc",)) == 3

    def test_multibyte_str(self):
        # Each CJK char is 3 bytes in UTF-8.
        assert _estimate_record_bytes(("中文",)) == 6

    def test_bytes_and_none(self):
        assert _estimate_record_bytes((b"abcd", None)) == 4

    def test_dict_and_list_use_compact_json(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        expected = len(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), default=str
            ).encode("utf-8")
        )
        assert _estimate_record_bytes((payload,)) == expected

    def test_scalar_constant(self):
        # int + float + None -> 16 + 16 + 0
        assert _estimate_record_bytes((1, 2.0, None)) == 32

    def test_mixed_tuple(self):
        vec = np.ones(4, dtype=np.float32)  # 16 bytes
        record = ("ws", "id", "hello", vec, None, 42)
        # 2 + 2 + 5 + 16 + 0 + 16
        assert _estimate_record_bytes(record) == 41


# ---------------------------------------------------------------------------
# _chunk_by_budget
# ---------------------------------------------------------------------------


class TestChunkByBudget:
    def test_empty(self):
        assert _chunk_by_budget([], lambda x: 1, 100, 100) == []

    def test_split_by_record_count(self):
        items = list(range(7))
        batches = _chunk_by_budget(
            items, lambda x: 1, max_payload_bytes=0, max_records_per_batch=3
        )
        sizes = [len(b) for b, _ in batches]
        assert sizes == [3, 3, 1]

    def test_split_by_payload_bytes(self):
        # Each item ~10 bytes; a 25-byte budget fits 2 per batch (2 overhead +
        # 10 + 1 + 10 = 23 <= 25; adding a third would exceed).
        items = ["x" * 10, "y" * 10, "z" * 10, "w" * 10]
        batches = _chunk_by_budget(
            items, lambda s: len(s), max_payload_bytes=25, max_records_per_batch=0
        )
        sizes = [len(b) for b, _ in batches]
        assert sizes == [2, 2]

    def test_both_dimensions_record_cap_wins(self):
        items = ["x" * 5] * 10
        # record cap 4 binds before the generous byte budget.
        batches = _chunk_by_budget(
            items, lambda s: len(s), max_payload_bytes=10_000, max_records_per_batch=4
        )
        sizes = [len(b) for b, _ in batches]
        assert sizes == [4, 4, 2]

    def test_oversized_single_item_becomes_own_batch(self):
        items = ["small", "X" * 1000, "small2"]
        batches = _chunk_by_budget(
            items, lambda s: len(s), max_payload_bytes=50, max_records_per_batch=0
        )
        # The 1000-byte item is emitted alone rather than raising.
        sizes = [len(b) for b, _ in batches]
        assert sizes == [1, 1, 1]
        assert batches[1][0] == ["X" * 1000]

    def test_non_positive_disables_both_dimensions(self):
        items = list(range(100))
        batches = _chunk_by_budget(
            items, lambda x: 999, max_payload_bytes=0, max_records_per_batch=0
        )
        assert len(batches) == 1
        assert len(batches[0][0]) == 100
