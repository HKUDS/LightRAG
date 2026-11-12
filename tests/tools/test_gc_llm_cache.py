"""Regression tests for the orphaned LLM cache GC.

The sweep deletes data, so the tests pin the two halves that decide what may
go: the candidate filter (a row must name a chunk) and the liveness check (the
chunk must be gone). A row whose chunk still exists must survive even when no
chunk lists it — deleting it would only re-bill the next reprocess — and a
deletion that reports success while the rows are still present must raise.
"""

import types

import pytest

from lightrag.exceptions import StorageControlPlaneError
from lightrag.tools.gc_llm_cache import SAMPLE_LIMIT, gc_orphan_llm_cache

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _stub_key_enumeration(monkeypatch):
    """Route the sweep's fake KVs through the shared enumeration seam.

    The real per-backend scans belong to the rebuild tool's tests; this tool
    only has to consume the seam, which
    ``test_key_enumeration_delegates_to_the_shared_backend_scan`` pins.
    """
    import lightrag.tools.rebuild_vdb as rebuild_vdb

    async def fake_enumerate(kv):
        return list(kv.rows)

    monkeypatch.setattr(rebuild_vdb, "enumerate_kv_keys", fake_enumerate)


class FakeCacheKV:
    """Cache namespace with just enough surface for the sweep."""

    def __init__(self, rows, *, delete_removes=True):
        self.rows = dict(rows)
        self.delete_removes = delete_removes
        self.deleted: list[str] = []
        self.delete_batches: list[int] = []
        self.commits = 0

    async def get_by_ids(self, ids):
        return [self.rows.get(key) for key in ids]

    async def filter_keys(self, keys):
        return {key for key in keys if key not in self.rows}

    async def delete(self, ids):
        self.deleted.extend(ids)
        self.delete_batches.append(len(ids))
        if self.delete_removes:
            for key in ids:
                self.rows.pop(key, None)

    async def index_done_callback(self):
        self.commits += 1


class FakeChunksKV:
    """Text chunks namespace reduced to the liveness question.

    ``filter_keys`` answers with the absent set (the base contract) and
    ``get_by_id_strict`` confirms one id under the strict contract: a row when
    the chunk exists, ``None`` only when it is confirmed gone.
    """

    supports_strict_point_reads = True

    def __init__(self, chunk_ids):
        self.chunk_ids = set(chunk_ids)
        self.filter_calls: list[set[str]] = []
        self.strict_calls: list[str] = []

    async def filter_keys(self, keys):
        self.filter_calls.append(set(keys))
        return {key for key in keys if key not in self.chunk_ids}

    async def get_by_id_strict(self, chunk_id):
        self.strict_calls.append(chunk_id)
        if chunk_id in self.chunk_ids:
            return {"content": "chunk text", "full_doc_id": "doc-1"}
        return None


def _rag(cache, chunks):
    return types.SimpleNamespace(llm_response_cache=cache, text_chunks=chunks)


def _extract_row(chunk_id):
    return {
        "return": "entities...",
        "cache_type": "extract",
        "chunk_id": chunk_id,
        "original_prompt": "chunk text",
        "queryparam": None,
    }


def _query_row():
    return {
        "return": "answer",
        "cache_type": "query",
        "chunk_id": None,
        "original_prompt": "question",
        "queryparam": {},
    }


async def test_orphan_is_reported_and_a_dry_run_deletes_nothing():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone"),
            "default:extract:bbb": _extract_row("chunk-alive"),
        }
    )
    chunks = FakeChunksKV({"chunk-alive"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks))

    assert report["cache_rows"] == 2
    assert report["chunk_scoped_rows"] == 2
    assert report["orphan_rows"] == 1
    assert report["live_rows"] == 1
    assert report["deleted_rows"] == 0
    assert report["orphan_keys_sample"] == ["default:extract:aaa"]
    assert cache.deleted == []
    # A report run must not commit anything either.
    assert cache.commits == 0


async def test_apply_deletes_only_the_orphans_and_commits():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone"),
            "default:extract:bbb": _extract_row("chunk-alive"),
            "default:query:ccc": _query_row(),
        }
    )
    chunks = FakeChunksKV({"chunk-alive"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["deleted_rows"] == 1
    assert cache.deleted == ["default:extract:aaa"]
    assert "default:extract:aaa" not in cache.rows
    assert set(cache.rows) == {"default:extract:bbb", "default:query:ccc"}
    # The delete is only durable once the storage commits.
    assert cache.commits == 1


async def test_rows_without_a_chunk_id_are_never_candidates():
    """Query/keywords answers belong to ``lightrag-clean-llmqc``, and the
    summary/smartheading/analysis artifacts carry no chunk reference at all, so
    no liveness question can be asked about any of them. They are counted and
    left alone: deleting what the sweep cannot justify is how it loses data
    the operator still wants."""
    cache = FakeCacheKV(
        {
            "local:query:aaa": _query_row(),
            "local:keywords:bbb": _query_row(),
            "default:summary:ccc": {**_query_row(), "cache_type": "summary"},
            "default:smartheading:ddd": {
                **_query_row(),
                "cache_type": "smartheading",
            },
            "default:analysis:eee": {**_query_row(), "cache_type": "analysis"},
        }
    )
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["unscoped_rows"] == 5
    assert report["chunk_scoped_rows"] == 0
    assert report["orphan_rows"] == 0
    assert cache.deleted == []
    # Nothing chunk-scoped was seen, so liveness was never asked.
    assert chunks.filter_calls == []


async def test_row_with_live_chunk_survives_even_when_no_chunk_lists_it():
    """The row's chunk exists, so its document is alive; leave the row alone."""
    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-alive")})
    chunks = FakeChunksKV({"chunk-alive"})  # no llm_cache_list anywhere

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["orphan_rows"] == 0
    assert cache.deleted == []
    assert "default:extract:aaa" in cache.rows


async def test_empty_or_non_string_chunk_id_is_not_a_candidate():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row(""),
            "default:extract:bbb": _extract_row(None),
        }
    )
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["unscoped_rows"] == 2
    assert cache.deleted == []


async def test_a_delete_that_did_not_stick_raises():
    cache = FakeCacheKV(
        {"default:extract:aaa": _extract_row("chunk-gone")}, delete_removes=False
    )
    chunks = FakeChunksKV(set())

    with pytest.raises(RuntimeError, match="still present"):
        await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)


async def test_batching_preserves_every_orphan():
    rows = {f"default:extract:{i}": _extract_row(f"chunk-{i}") for i in range(7)}
    rows["default:extract:live"] = _extract_row("chunk-live")
    cache = FakeCacheKV(rows)
    chunks = FakeChunksKV({"chunk-live"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True, batch_size=2)

    assert report["orphan_rows"] == 7
    assert report["deleted_rows"] == 7
    assert set(cache.deleted) == {f"default:extract:{i}" for i in range(7)}
    assert chunks.filter_calls and all(len(call) <= 2 for call in chunks.filter_calls)
    # The delete phase honours the same bound: 7 orphans at 2 per batch is four
    # rounds, each committed before the next.
    assert cache.delete_batches == [2, 2, 2, 1]
    assert cache.commits == 4


async def test_missing_storages_raise():
    chunks = FakeChunksKV(set())
    with pytest.raises(ValueError, match="cache storage"):
        await gc_orphan_llm_cache(
            types.SimpleNamespace(llm_response_cache=None, text_chunks=chunks)
        )

    cache = FakeCacheKV({})
    with pytest.raises(ValueError, match="text chunks storage"):
        await gc_orphan_llm_cache(
            types.SimpleNamespace(llm_response_cache=cache, text_chunks=None)
        )


async def test_non_positive_batch_size_raises():
    cache = FakeCacheKV({})
    chunks = FakeChunksKV(set())
    with pytest.raises(ValueError, match="batch_size"):
        await gc_orphan_llm_cache(_rag(cache, chunks), batch_size=0)


async def test_key_enumeration_delegates_to_the_shared_backend_scan(monkeypatch):
    """The sweep must not grow its own copy of the per-backend scans."""
    calls: list[object] = []

    async def fake_enumerate(kv):
        calls.append(kv)
        return ["default:extract:aaa"]

    import lightrag.tools.rebuild_vdb as rebuild_vdb

    monkeypatch.setattr(rebuild_vdb, "enumerate_kv_keys", fake_enumerate)

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-gone")})
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks))

    assert calls == [cache]
    assert report["orphan_rows"] == 1


async def test_every_row_of_an_orphaned_chunk_is_deleted():
    """One chunk can own several rows (a re-extraction with a different prompt
    hash adds another). Deleting one row per orphaned chunk would strand the
    rest."""
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone"),
            "default:extract:bbb": _extract_row("chunk-gone"),
            "default:extract:ccc": _extract_row("chunk-live"),
        }
    )
    chunks = FakeChunksKV({"chunk-live"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["orphan_rows"] == 2
    assert report["deleted_rows"] == 2
    assert set(cache.rows) == {"default:extract:ccc"}


async def test_the_cli_opens_the_backends_the_environment_names(monkeypatch, capsys):
    """LightRAG's constructor defaults are file-backed, so a CLI that named no
    storage classes would open a fresh local directory against a
    ``LIGHTRAG_KV_STORAGE=PGKVStorage`` deployment and report it as empty: a
    silent wrong answer, where a mismatch has to be visible instead."""
    import lightrag

    from lightrag.tools import gc_llm_cache

    opened: dict[str, object] = {}
    storage_calls: list[str] = []

    class _FakeRAG:
        def __init__(self, **kwargs):
            opened.update(kwargs)
            self.workspace = kwargs.get("workspace")
            self.llm_response_cache = FakeCacheKV({})
            self.text_chunks = FakeChunksKV(set())

        async def initialize_storages(self):
            storage_calls.append("initialize")

        async def finalize_storages(self):
            storage_calls.append("finalize")

    monkeypatch.setattr(lightrag, "LightRAG", _FakeRAG)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "PGKVStorage")
    monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "Neo4JStorage")
    monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "PGDocStatusStorage")
    monkeypatch.setenv("WORKSPACE", "acme")
    monkeypatch.setenv("WORKING_DIR", "/tmp/gc-cli-ws")

    assert (
        await gc_llm_cache._async_main(apply=False, batch_size=10, verbose=False)
        is None
    )

    assert opened["kv_storage"] == "PGKVStorage"
    assert opened["graph_storage"] == "Neo4JStorage"
    assert opened["doc_status_storage"] == "PGDocStatusStorage"
    assert opened["workspace"] == "acme"
    assert opened["working_dir"] == "/tmp/gc-cli-ws"
    # The sweep never reads vectors, so the CLI must not build a real store.
    assert opened["vector_storage"] == "NoopVectorDBStorage"
    assert storage_calls == ["initialize", "finalize"]
    out = capsys.readouterr().out
    assert "Cache storage: FakeCacheKV" in out
    # Printed before the sweep, so an aborting run still names the target.
    assert out.index("Cache storage:") < out.index("Cache rows:")

    with pytest.raises(ValueError, match="batch_size"):
        await gc_llm_cache._async_main(apply=False, batch_size=0, verbose=False)

    assert "Cache storage: FakeCacheKV" in capsys.readouterr().out
    # Storages are released on the way out, failure or not.
    assert storage_calls == ["initialize", "finalize", "initialize", "finalize"]


async def test_a_scan_that_repeats_a_key_reports_the_row_once(monkeypatch):
    """A repeated key used to be counted twice and "deleted" twice, so the
    report stopped describing the namespace it claimed to describe."""

    async def repeating_scan(kv):
        return ["default:extract:aaa", "default:extract:aaa"]

    import lightrag.tools.rebuild_vdb as rebuild_vdb

    monkeypatch.setattr(rebuild_vdb, "enumerate_kv_keys", repeating_scan)

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-gone")})

    report = await gc_orphan_llm_cache(_rag(cache, FakeChunksKV(set())), apply=True)

    assert report["cache_rows"] == 1
    assert report["orphan_rows"] == 1
    assert report["deleted_rows"] == 1
    assert cache.deleted == ["default:extract:aaa"]


async def test_a_liveness_failure_deletes_nothing():
    """Every liveness check completes before the first delete, so a
    chunks-namespace failure must not leave a half-swept cache.

    The fake fails on the SECOND confirmation, so the first candidate has
    already been classified as an orphan: an implementation that deleted inside
    the classification loop would have removed its rows by then and fails
    here."""

    class _BrokenOnSecondConfirmation(FakeChunksKV):
        def __init__(self, chunk_ids):
            super().__init__(chunk_ids)
            self.confirmations = 0

        async def get_by_id_strict(self, chunk_id):
            self.confirmations += 1
            if self.confirmations > 1:
                raise ConnectionError("chunks namespace unreachable")
            return await super().get_by_id_strict(chunk_id)

    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone-1"),
            "default:extract:bbb": _extract_row("chunk-gone-2"),
        }
    )

    with pytest.raises(ConnectionError):
        await gc_orphan_llm_cache(
            _rag(cache, _BrokenOnSecondConfirmation(set())),
            apply=True,
            batch_size=1,
        )

    assert cache.deleted == []
    assert cache.commits == 0
    assert len(cache.rows) == 2


async def test_apply_on_an_empty_namespace_deletes_nothing():
    cache = FakeCacheKV({})
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report == {
        "cache_rows": 0,
        "chunk_scoped_rows": 0,
        "unscoped_rows": 0,
        "orphan_rows": 0,
        "live_rows": 0,
        "deleted_rows": 0,
        "orphan_keys_sample": [],
        "deleted_keys_sample": [],
    }
    assert cache.deleted == []
    assert cache.commits == 0
    # Nothing chunk-scoped was seen, so liveness was never asked.
    assert chunks.filter_calls == []
    assert chunks.strict_calls == []


async def test_the_key_sample_is_bounded_while_the_counts_stay_exact():
    """The report is printed as well as returned, so a huge workspace must not
    turn a dry run into an unbounded one."""
    rows = {
        f"default:extract:{index:03d}": _extract_row(f"chunk-{index}")
        for index in range(25)
    }
    cache = FakeCacheKV(rows)

    report = await gc_orphan_llm_cache(_rag(cache, FakeChunksKV(set())), apply=True)

    assert report["chunk_scoped_rows"] == 25
    assert report["orphan_rows"] == 25
    assert report["deleted_rows"] == 25
    assert len(report["orphan_keys_sample"]) == SAMPLE_LIMIT
    assert len(report["deleted_keys_sample"]) == SAMPLE_LIMIT
    assert report["orphan_keys_sample"] == sorted(rows)[:SAMPLE_LIMIT]


async def test_a_fail_open_chunks_read_cannot_orphan_a_live_chunk():
    """``filter_keys`` reports every key absent while the chunks index is not
    ready. Reading that as proof of absence would classify every chunk-scoped
    row as an orphan and empty the cache, so absence is confirmed per chunk."""

    class _FailOpenChunks(FakeChunksKV):
        async def filter_keys(self, keys):
            self.filter_calls.append(set(keys))
            return set(keys)  # an unanswerable question, answered as "absent"

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-alive")})
    chunks = _FailOpenChunks({"chunk-alive"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["orphan_rows"] == 0
    assert report["live_rows"] == 1
    assert report["deleted_rows"] == 0
    assert cache.deleted == []
    assert "default:extract:aaa" in cache.rows
    assert chunks.strict_calls == ["chunk-alive"]


async def test_a_strict_read_that_cannot_confirm_aborts_before_any_delete():
    """The strict contract raises when absence cannot be positively confirmed
    (a dropped index, an unreachable backend). That must stop the sweep, not
    downgrade to "absent"."""

    class _UnconfirmableChunks(FakeChunksKV):
        async def get_by_id_strict(self, chunk_id):
            raise StorageControlPlaneError(f"cannot confirm absence of {chunk_id}")

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-gone")})

    with pytest.raises(StorageControlPlaneError):
        await gc_orphan_llm_cache(_rag(cache, _UnconfirmableChunks(set())), apply=True)

    assert cache.deleted == []
    assert cache.commits == 0
    assert "default:extract:aaa" in cache.rows


async def test_a_chunks_backend_without_strict_reads_is_refused():
    """No strict reads means no way to tell a real orphan from a failed read,
    and this tool may not delete on an unconfirmed miss."""

    class _NoStrictReads(FakeChunksKV):
        supports_strict_point_reads = False

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-gone")})

    with pytest.raises(ValueError, match="supports_strict_point_reads"):
        await gc_orphan_llm_cache(_rag(cache, _NoStrictReads(set())), apply=True)

    assert cache.deleted == []
    assert "default:extract:aaa" in cache.rows
