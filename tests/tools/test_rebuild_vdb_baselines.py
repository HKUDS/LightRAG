"""``lightrag-rebuild-vdb`` and the embedding baselines: configuration LAST.

After a target's rebuild is durable and verified the tool records that ONE
target's baseline; a rebuild that reported errors records nothing; a baseline
that could not be recorded is reported as a failed rebuild, which is what
makes the process exit non-zero. See *Rebuild* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from lightrag.namespace import CONFIG_CONTAINER_TAG

import lightrag.tools.rebuild_vdb as rebuild_vdb
from lightrag import config_store as cs

pytestmark = pytest.mark.offline


class _FakeConfigKV:
    supports_strict_point_reads = True

    def __init__(self, rows=None, *, write_error=None):
        self.rows = dict(rows or {})
        self.write_error = write_error
        self.flushes = 0

    async def get_by_id_strict(self, key):
        row = self.rows.get(key)
        return None if row is None else dict(row)

    async def upsert(self, data):
        if self.write_error is not None:
            raise self.write_error
        self.rows.update(data)

    async def index_done_callback(self):
        self.flushes += 1


def _tool(config, *, embedding_model="new-model", vdbs=None):
    tool = rebuild_vdb.RebuildTool()
    tool.workspace = "ws"
    tool.configuration_storage = config
    tool.embedding_func = SimpleNamespace(
        model_name=embedding_model, embedding_dim=1024, max_token_size=None
    )
    for target, vdb in (vdbs or {}).items():
        setattr(tool, f"{target}_vdb", vdb)
    return tool


class _FakeVectorStorage:
    """A vector storage that answers the durability question the way a per-item
    backend does: ``index_done_callback`` already returned, and whether anything
    is still buffered is a separate question only the storage can answer."""

    def __init__(self, *, retained=False, answer_error=None):
        self.retained = retained
        self.answer_error = answer_error
        self.asked = 0

    async def has_pending_index_ops(self, *, include_deletes: bool = False) -> bool:
        self.asked += 1
        if self.answer_error is not None:
            raise self.answer_error
        return self.retained


def _stats(label, errors=()):
    stats = rebuild_vdb._new_stats(label, 3)
    stats["errors"].extend(errors)
    return stats


def _old_record(target):
    return cs.make_config_row(
        scope_workspace="ws",
        suffix=cs.embedding_baseline_suffix(target),
        value={"model": "old-model", "dim": 1024, "origin": "rebuild"},
        updated_by="test",
    )


async def test_a_clean_rebuild_records_only_that_targets_baseline(capsys):
    """Scenario 8."""
    config = _FakeConfigKV(
        {
            cs.embedding_baseline_key("ws", t): _old_record(t)
            for t in cs.EMBEDDING_TARGETS
        }
    )
    tool = _tool(config)

    await tool.commit_baseline("chunks", _stats("chunks"))

    assert config.rows[cs.embedding_baseline_key("ws", "chunks")]["value"] == {
        "model": "new-model",
        "dim": 1024,
        "origin": "rebuild",
    }
    for untouched in ("entities", "relationships"):
        assert (
            config.rows[cs.embedding_baseline_key("ws", untouched)]["value"]["model"]
            == "old-model"
        )
    assert config.flushes == 1
    assert "baseline recorded" in capsys.readouterr().out


async def test_a_rebuild_with_errors_records_nothing(capsys):
    """A failed rebuild advances nothing: the stale record keeps the server
    refusing, which is the safe direction."""
    config = _FakeConfigKV(
        {cs.embedding_baseline_key("ws", "chunks"): _old_record("chunks")}
    )
    tool = _tool(config)
    stats = _stats("chunks", errors=[{"batch": 2, "records_lost": 5}])

    await tool.commit_baseline("chunks", stats)

    assert (
        config.rows[cs.embedding_baseline_key("ws", "chunks")]["value"]["model"]
        == "old-model"
    )
    assert config.flushes == 0
    assert "NOT recorded" in capsys.readouterr().out


async def test_a_failed_baseline_write_makes_the_rebuild_a_failure(capsys):
    """Scenario 9: the vectors were rebuilt but the record could not be
    written. The tool reports it as a rebuild error, so report_rebuild() says
    the session failed and run() exits non-zero; the previous record keeps
    refusing startup until a re-run converges."""
    config = _FakeConfigKV(write_error=OSError("configuration store read-only"))
    tool = _tool(config)
    stats = _stats("entities")

    await tool.commit_baseline("entities", stats)

    assert stats["errors"] and stats["errors"][-1]["batch"] == "baseline"
    assert stats["errors"][-1]["error_type"] == "ConfigurationStorageError"
    assert "could not be recorded" in capsys.readouterr().out
    assert tool.report_rebuild([stats]) is True, (
        "a baseline failure is a failed rebuild"
    )


async def test_print_baselines_names_the_targets_the_server_would_refuse(capsys):
    config = _FakeConfigKV(
        {
            cs.embedding_baseline_key("ws", "entities"): _old_record("entities"),
            cs.embedding_baseline_key("ws", "chunks"): cs.make_config_row(
                scope_workspace="ws",
                suffix=cs.embedding_baseline_suffix("chunks"),
                value={"model": "new-model", "dim": 1024, "origin": "probe"},
                updated_by="test",
            ),
        }
    )
    tool = _tool(config)

    assert await tool.print_baselines() is True

    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "entities" in out and "(none recorded)" in out
    assert "refuses to start" in out


async def test_print_baselines_aborts_when_the_store_cannot_be_read(capsys):
    class Unreadable(_FakeConfigKV):
        async def get_by_id_strict(self, key):
            raise ConnectionError("down")

    tool = _tool(Unreadable())
    assert await tool.print_baselines() is False
    assert "Could not read" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# A flush that returned is not a flush that landed
# ---------------------------------------------------------------------------


async def test_retained_vector_ops_record_no_baseline(capsys):
    """The defect: OpenSearch keeps per-doc 408/429/5xx failures buffered and
    returns normally from ``index_done_callback``, so the rebuild reports no
    errors. Recording the baseline there would claim the target was adopted in
    the configured space while its index is missing vectors -- and nothing
    later catches it, because the startup precheck sees a matching record and
    the coverage gate only refuses an EMPTY index."""
    config = _FakeConfigKV(
        {cs.embedding_baseline_key("ws", "chunks"): _old_record("chunks")}
    )
    vdb = _FakeVectorStorage(retained=True)
    tool = _tool(config, vdbs={"chunks": vdb})
    stats = _stats("chunks")

    await tool.commit_baseline("chunks", stats)

    assert vdb.asked == 1
    assert (
        config.rows[cs.embedding_baseline_key("ws", "chunks")]["value"]["model"]
        == "old-model"
    ), "the stale record must stay, so the server keeps refusing"
    assert config.flushes == 0
    assert [e["error_type"] for e in stats["errors"]] == ["RetainedVectorOps"]
    assert "still holds" in capsys.readouterr().out


async def test_an_unanswerable_durability_question_records_no_baseline(capsys):
    """An answer that could not be read is not a durable one."""
    config = _FakeConfigKV(
        {cs.embedding_baseline_key("ws", "entities"): _old_record("entities")}
    )
    vdb = _FakeVectorStorage(answer_error=ConnectionError("index unreachable"))
    tool = _tool(config, vdbs={"entities": vdb})
    stats = _stats("entities")

    await tool.commit_baseline("entities", stats)

    assert (
        config.rows[cs.embedding_baseline_key("ws", "entities")]["value"]["model"]
        == "old-model"
    )
    assert [e["error_type"] for e in stats["errors"]] == ["ConnectionError"]
    assert "could not say" in capsys.readouterr().out


async def test_a_storage_that_retained_nothing_records_its_baseline(capsys):
    """The other half: asking costs a method call and changes nothing when the
    flush really did land everything."""
    config = _FakeConfigKV(
        {cs.embedding_baseline_key("ws", "chunks"): _old_record("chunks")}
    )
    vdb = _FakeVectorStorage(retained=False)
    tool = _tool(config, vdbs={"chunks": vdb})
    stats = _stats("chunks")

    await tool.commit_baseline("chunks", stats)

    assert vdb.asked == 1
    assert (
        config.rows[cs.embedding_baseline_key("ws", "chunks")]["value"]["model"]
        == "new-model"
    )
    assert stats["errors"] == []
    assert "baseline recorded" in capsys.readouterr().out


class TestTheToolIsASecondProcessTree:
    """``lightrag-rebuild-vdb`` run against a directory a server already holds.

    Both keep a private in-memory copy of a file-backed configuration
    namespace and publish by rewriting the whole file, so whichever flushes
    second erases the other's baselines -- including the very record this tool
    writes to say the rebuild happened. The confirmation prompt asks the
    operator; this asks the filesystem.
    """

    async def test_setup_refuses_while_another_process_tree_holds_it(
        self, tmp_path, monkeypatch, capsys
    ):
        """The real lock, against a real holder, through the real method.

        The holder is simulated by taking the claim and then dropping this
        process's bookkeeping, so the tool's own ``acquire`` opens a second
        descriptor -- which ``flock`` refuses even within one process.

        The refusal must also come FIRST: nothing about the answer depends on
        the environment checks or on building an embedding function, so the
        operator should not wait for them, and this assertion should not
        depend on whatever the ambient environment happens to hold.
        """
        from lightrag.kg import working_dir_lock as wdl

        # The tool parses argv; pytest's own arguments would make argparse exit.
        monkeypatch.setattr(sys, "argv", ["lightrag-rebuild-vdb"])
        monkeypatch.setenv("WORKING_DIR", str(tmp_path))
        monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        monkeypatch.setenv("WORKSPACE", "rebuildws")

        # The claim is on the CONFIGURATION directory, which is where the
        # file the two would overwrite actually lives.
        config_dir = str(tmp_path / CONFIG_CONTAINER_TAG)
        wdl.acquire_working_dir_lock(config_dir)
        holder = dict(wdl._claims)
        wdl._claims.clear()  # the tool must look like a different tree

        tool = rebuild_vdb.RebuildTool()
        try:
            ok = await tool.setup_storages()

            assert ok is False, "the tool ran while another process tree held it"
            assert "already in use" in capsys.readouterr().out
            assert tool.configuration_storage is None, (
                "the configuration storage was opened despite the refusal"
            )
            assert tool._holds_working_dir is False
        finally:
            wdl._claims.update(holder)
            wdl.release_working_dir_lock(config_dir)

    def test_a_server_backed_configuration_claims_nothing(self):
        """Two processes against one PostgreSQL share a container by design;
        refusing there would invent a restriction."""
        assert rebuild_vdb.uses_working_dir("PGKVStorage") is False
        assert rebuild_vdb.uses_working_dir("MongoKVStorage") is False
        assert rebuild_vdb.uses_working_dir("OpenSearchKVStorage") is False
        assert rebuild_vdb.uses_working_dir("JsonKVStorage") is True


class TestAFlushThatRaisedMayHaveLanded:
    """``_flush`` asks the buffer, not the exception, what a flush did.

    The rule this PR states for the configuration flush applies to the data
    side too: ``ReferencesIntactFlushError`` covers BOTH "the bulk failed with
    every operation still buffered" AND "the bulk landed and only the refresh
    after it failed". Reading both as a failed batch reports durable vectors
    as lost -- and ``commit_baseline`` then withholds the baseline, so the
    server keeps refusing to start over an index that is in fact rebuilt.
    """

    class _Vdb:
        def __init__(self, *, retained: bool):
            self._retained = retained

        async def index_done_callback(self):
            raise rebuild_vdb.ReferencesIntactFlushError("refresh failed")

        async def has_pending_index_ops(self, *, include_deletes: bool = False):
            return self._retained

    class _UnanswerableVdb:
        async def index_done_callback(self):
            raise rebuild_vdb.ReferencesIntactFlushError("refresh failed")

    def _stats(self):
        return {
            "label": "entities",
            "staged": 5,
            "rebuilt": 0,
            "batches": 1,
            "failed_batches": 0,
            "errors": [],
        }

    async def test_an_empty_buffer_means_the_write_landed(self):
        stats = self._stats()
        await rebuild_vdb._flush(self._Vdb(retained=False), stats)

        assert stats["rebuilt"] == 5
        assert stats["failed_batches"] == 0
        assert stats["errors"] == []
        assert stats["staged"] == 0

    async def test_a_retained_buffer_is_still_a_failed_batch(self):
        stats = self._stats()
        await rebuild_vdb._flush(self._Vdb(retained=True), stats)

        assert stats["rebuilt"] == 0
        assert stats["failed_batches"] == 1
        assert stats["errors"][0]["records_lost"] == 5

    async def test_a_storage_that_cannot_be_asked_keeps_the_conservative_reading(self):
        """Unanswerable is treated as retained: a flush is never credited on
        an assumption."""
        stats = self._stats()
        await rebuild_vdb._flush(self._UnanswerableVdb(), stats)

        assert stats["rebuilt"] == 0
        assert stats["failed_batches"] == 1
