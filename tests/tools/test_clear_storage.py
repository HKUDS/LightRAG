"""Tests for the offline storage clear tool.

Covers what the operator relies on:
- the summary is read fail-loud (a count that cannot be read aborts with
  nothing dropped, never renders as zero);
- only the exact confirmation phrase proceeds, and nothing is touched before;
- the drop step mirrors ``/documents/clear``: every one of the eleven storages
  is attempted, a ``BaseException`` object or a non-success dict is a failure,
  and the configuration records go only when every drop succeeded;
- the kind rule: a server backend that cannot be opened or read refuses the
  run before anything is dropped, a file-backed storage whose file is
  corrupt is shown as UNREADABLE and dropped anyway;
- the LLM response cache is never instantiated or dropped;
- top-level input files go, subdirectories (``__parsed__``) stay;
- exit status is non-zero on any partial outcome.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import lightrag.tools.clear_storage as clear_storage
from lightrag.base import DocStatus
from lightrag.namespace import CONFIG_CONTAINER_TAG
from lightrag.tools.clear_storage import (
    CONFIRMATION_PHRASE,
    DATA_STORAGE_LABELS,
    ClearTool,
    DropOutcome,
    RemoteBackendUnavailableError,
    Unreadable,
    classify_drop_result,
    count_kv_rows,
    is_server_backed,
)
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_storage(label: str, *, drop_result=None, drop_error=None, workspace="ws"):
    """A storage whose ``drop()`` returns ``drop_result`` or raises ``drop_error``."""
    if drop_error is not None:
        drop = AsyncMock(side_effect=drop_error)
    else:
        drop = AsyncMock(
            return_value=(
                {"status": "success", "message": "data dropped"}
                if drop_result is None
                else drop_result
            )
        )
    storage = SimpleNamespace(
        label=label,
        workspace=workspace,
        drop=drop,
        finalize=AsyncMock(),
        is_empty=AsyncMock(return_value=True),
        persists_vectors=True,
    )
    return storage


FILE_BACKED_NAMES = {
    "graph": "NetworkXStorage",
    "vector": "NanoVectorDBStorage",
    "kv": "JsonKVStorage",
    "doc_status": "JsonDocStatusStorage",
    "config": "JsonKVStorage",
}


def make_doc(status: DocStatus, updated_at: str, file_path: str):
    return SimpleNamespace(status=status, updated_at=updated_at, file_path=file_path)


def make_tool(
    tmp_path,
    *,
    counts=None,
    recent=None,
    chunk_keys=("chunk-1", "chunk-2"),
    workspace="ws",
    **storage_overrides,
) -> ClearTool:
    """A ``ClearTool`` past ``setup_storages``, on fakes.

    ``storage_overrides`` maps a label to a fake built by ``make_storage``
    with the failure the test wants.
    """
    tool = ClearTool()
    tool.workspace = workspace
    tool.input_dir = str(tmp_path / "inputs")
    tool.global_config = {"working_dir": str(tmp_path / "wd")}
    tool.storage_names = dict(FILE_BACKED_NAMES)
    tool.storages = {
        label: storage_overrides.get(label) or make_storage(label, workspace=workspace)
        for label in DATA_STORAGE_LABELS
    }
    counts = counts if counts is not None else {DocStatus.PROCESSED: 2}

    async def count_docs_by_statuses(statuses, *, strict):
        assert strict is True, "the summary must count strictly"
        return counts.get(statuses[0], 0)

    async def get_docs_paginated(**kwargs):
        rows = recent or []
        return rows, len(rows)

    tool.storages["doc_status"].count_docs_by_statuses = count_docs_by_statuses
    tool.storages["doc_status"].get_docs_paginated = get_docs_paginated
    tool.storages["text_chunks"]._keys = list(chunk_keys)
    tool.configuration_storage = make_storage("config", workspace=workspace)
    return tool


@pytest.fixture
def stub_kv_count(monkeypatch):
    async def count_kv_rows(kv):
        return len(kv._keys)

    monkeypatch.setattr(clear_storage, "count_kv_rows", count_kv_rows)


async def _never_embed(*_args, **_kwargs):
    raise AssertionError("the clear tool must never embed")


def fixed_embedding_func(dim: int = 8, model_name: str = "dummy") -> EmbeddingFunc:
    """What the server factory would hand back, without importing the api."""
    return EmbeddingFunc(embedding_dim=dim, func=_never_embed, model_name=model_name)


@pytest.fixture
def stub_embedding_factory(monkeypatch):
    """Pin the server factory's output so end-to-end runs need no argv/env."""
    monkeypatch.setattr(
        ClearTool, "build_embedding_func", lambda self: fixed_embedding_func()
    )


@pytest.fixture
def stub_baselines(monkeypatch):
    """Every baseline row confirmed absent."""

    async def read_config_row_strict(config, key):
        return None

    monkeypatch.setattr(clear_storage, "read_config_row_strict", read_config_row_strict)


@pytest.fixture(autouse=True)
def no_workspace_overrides(monkeypatch):
    """The developer's own ``*_WORKSPACE`` variables must not reach the summary."""
    monkeypatch.setattr(clear_storage, "warn_about_workspace_overrides", lambda: [])


@pytest.fixture
def recorded_config_delete(monkeypatch):
    calls = []

    async def delete_workspace_configuration(config, workspace):
        calls.append(workspace)

    monkeypatch.setattr(
        clear_storage, "delete_workspace_configuration", delete_workspace_configuration
    )
    return calls


def stub_share_data(monkeypatch):
    """Keep a fake-backed ``run()`` off the process-wide shared storage."""
    monkeypatch.setattr(clear_storage, "initialize_share_data", lambda **kw: None)
    monkeypatch.setattr(clear_storage, "finalize_share_data", lambda: None)


def answer_prompts(monkeypatch, *answers):
    """Feed ``answers`` to successive ``input()`` calls, then fail loudly."""
    remaining = list(answers)

    def fake_input(prompt=""):
        if not remaining:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def seed_input_dir(tmp_path):
    inputs = tmp_path / "inputs"
    (inputs / "__parsed__").mkdir(parents=True)
    (inputs / "a.txt").write_text("a")
    (inputs / "b.pdf").write_text("b")
    (inputs / "__parsed__" / "a.md").write_text("parsed")
    return inputs


# ---------------------------------------------------------------------------
# Drop classification, mirrored from /documents/clear
# ---------------------------------------------------------------------------


class TestClassifyDropResult:
    def test_a_success_dict_is_success(self):
        assert classify_drop_result({"status": "success"}) is None

    def test_a_non_success_dict_is_a_failure_with_its_message(self):
        assert (
            classify_drop_result({"status": "error", "message": "legacy store kept"})
            == "legacy store kept"
        )

    def test_an_exception_object_is_a_failure(self):
        assert "boom" in classify_drop_result(RuntimeError("boom"))

    def test_a_cancelled_drop_is_a_drop_that_did_not_happen(self):
        """``gather(return_exceptions=True)`` hands back a ``CancelledError``
        OBJECT, which is a ``BaseException``: classified on ``Exception`` it
        would read as success, and the configuration records would go over
        whatever the cancelled drop left behind."""
        failure = classify_drop_result(asyncio.CancelledError())
        assert failure is not None
        assert failure != "", "a CancelledError stringifies to nothing"


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------


class TestSummary:
    async def test_it_shows_counts_recent_docs_chunks_and_input_files(
        self, tmp_path, stub_kv_count, stub_baselines, capsys
    ):
        seed_input_dir(tmp_path)
        recent = [
            ("doc-2", make_doc(DocStatus.FAILED, "2026-09-02T00:00:00", "two.pdf")),
            ("doc-1", make_doc(DocStatus.PROCESSED, "2026-09-01T00:00:00", "one.txt")),
        ]
        tool = make_tool(
            tmp_path,
            counts={DocStatus.PROCESSED: 1, DocStatus.FAILED: 1},
            recent=recent,
            chunk_keys=("c1", "c2", "c3"),
        )

        summary = await tool.collect_summary()
        tool.print_summary(summary)
        out = capsys.readouterr().out

        assert summary["counts"]["processed"] == 1
        assert summary["counts"]["failed"] == 1
        assert summary["total_docs"] == 2
        assert summary["chunk_count"] == 3
        assert [p.rsplit("/", 1)[-1] for p in summary["input_files"]] == [
            "a.txt",
            "b.pdf",
        ], "only top-level files; __parsed__ contents are preserved"
        assert "two.pdf" in out and "one.txt" in out
        assert "Text chunks: 3" in out
        assert "processed      1" in out
        assert "LLM response cache" in out, "the operator is told what survives"

    async def test_the_recent_list_asks_for_the_ten_most_recently_updated(
        self, tmp_path, stub_kv_count, stub_baselines
    ):
        tool = make_tool(tmp_path)
        seen = {}

        async def get_docs_paginated(**kwargs):
            seen.update(kwargs)
            return [], 0

        tool.storages["doc_status"].get_docs_paginated = get_docs_paginated
        await tool.collect_summary()

        assert seen["page"] == 1
        assert seen["page_size"] == 10
        assert seen["sort_field"] == "updated_at"
        assert seen["sort_direction"] == "desc"

    async def test_vector_states_name_refused_empty_and_populated(
        self, tmp_path, stub_kv_count, stub_baselines
    ):
        tool = make_tool(tmp_path)
        tool.refused_vdbs["entities_vdb"] = "foreign space"
        tool.storages["relationships_vdb"].is_empty = AsyncMock(return_value=False)

        summary = await tool.collect_summary()

        assert summary["vectors"]["entities_vdb"].startswith("refused")
        assert summary["vectors"]["relationships_vdb"] == "has vectors"
        assert summary["vectors"]["chunks_vdb"] == "EMPTY"

    async def test_a_workspace_override_in_effect_is_named(
        self, tmp_path, monkeypatch, stub_kv_count, stub_baselines, capsys
    ):
        """A backend ``*_WORKSPACE`` variable outranks ``WORKSPACE`` inside the
        storage constructor, and Redis / Qdrant never write the effective
        name back onto ``workspace`` -- so the environment, not the storage
        object, is what the summary asks."""
        monkeypatch.setattr(
            clear_storage,
            "warn_about_workspace_overrides",
            lambda: ["REDIS_WORKSPACE"],
        )
        tool = make_tool(tmp_path, workspace="ws")

        tool.print_summary(await tool.collect_summary())

        out = capsys.readouterr().out
        assert "Workspace override(s) in effect: REDIS_WORKSPACE" in out

    async def test_a_resolved_workspace_that_differs_is_flagged(
        self, tmp_path, stub_kv_count, stub_baselines, capsys
    ):
        """Backends that write the effective workspace back (PostgreSQL,
        MongoDB, Milvus) or expose it (Qdrant) are reported by name too."""
        tool = make_tool(tmp_path, workspace="ws")
        tool.storages["doc_status"].workspace = "other"
        tool.storages["chunks_vdb"].effective_workspace = "qdrant-legacy"

        tool.print_summary(await tool.collect_summary())

        out = capsys.readouterr().out
        assert "resolved to workspace(s)" in out
        assert "other" in out and "qdrant-legacy" in out

    async def test_an_unreadable_count_on_a_file_backed_store_is_shown_not_zero(
        self, tmp_path, stub_kv_count, stub_baselines, capsys
    ):
        """The failure this display exists to prevent is a swallowed read
        rendering as zero documents. A file-backed store that cannot answer
        is DATA about to be deleted, so the run goes on -- but the value is
        UNREADABLE on screen, never 0, and the operator is told."""
        tool = make_tool(tmp_path)

        async def broken_count(statuses, *, strict):
            raise OSError("doc_status file unreadable")

        tool.storages["doc_status"].count_docs_by_statuses = broken_count

        summary = await tool.collect_summary()
        tool.print_summary(summary)

        out = capsys.readouterr().out
        assert all(isinstance(c, Unreadable) for c in summary["counts"].values())
        assert "UNREADABLE" in out
        assert "could not be read" in out
        assert "documents in status processed" in out

    async def test_an_unreadable_count_on_a_server_backend_refuses_the_run(
        self, tmp_path, monkeypatch, stub_kv_count, stub_baselines, capsys
    ):
        """A server backend that cannot answer a read will not serve the
        drop either; clearing the others would leave it populated. Nothing
        is dropped and the phrase is never asked for."""
        seed_input_dir(tmp_path)
        tool = make_tool(tmp_path)
        tool.storage_names["doc_status"] = "PGDocStatusStorage"

        async def broken_count(statuses, *, strict):
            raise ConnectionError("doc_status unreachable")

        tool.storages["doc_status"].count_docs_by_statuses = broken_count
        monkeypatch.setattr(tool, "setup_storages", AsyncMock(return_value=True))
        stub_share_data(monkeypatch)
        answer_prompts(monkeypatch, "yes")  # the phrase prompt must never come

        ok = await tool.run()

        assert ok is False
        assert "nothing was deleted" in capsys.readouterr().out
        for label in DATA_STORAGE_LABELS:
            tool.storages[label].drop.assert_not_called()
        assert (tmp_path / "inputs" / "a.txt").exists()

    async def test_a_storage_that_did_not_open_is_not_asked(
        self, tmp_path, stub_kv_count, stub_baselines
    ):
        """A JSON storage whose load failed still holds an EMPTY shared dict;
        asking it would render a corrupt file as zero rows."""
        tool = make_tool(tmp_path)
        tool.unavailable["doc_status"] = "JSONDecodeError: boom"
        asked = []

        async def count(statuses, *, strict):
            asked.append(statuses)
            return 0

        tool.storages["doc_status"].count_docs_by_statuses = count

        summary = await tool.collect_summary()

        assert asked == []
        assert isinstance(summary["counts"]["processed"], Unreadable)
        assert "did not open" in summary["counts"]["processed"].reason

    async def test_a_missing_count_capability_is_unreadable_not_fatal(
        self, tmp_path, monkeypatch, stub_baselines
    ):
        from lightrag.exceptions import StorageCapabilityError

        async def count_kv_rows(kv):
            raise StorageCapabilityError("no row enumeration on this backend")

        monkeypatch.setattr(clear_storage, "count_kv_rows", count_kv_rows)
        tool = make_tool(tmp_path)
        tool.storage_names["kv"] = "PGKVStorage"  # server-backed, still tolerated

        summary = await tool.collect_summary()

        assert isinstance(summary["chunk_count"], Unreadable)


class TestCountKvRows:
    """The chunk count must not materialize every key: a workspace with
    millions of chunks would exhaust the process before the prompt."""

    async def test_an_unknown_backend_is_counted_through_the_row_stream(self):
        seen_page_sizes = []

        class StreamingKV:
            def iter_rows(self, *, page_size):
                seen_page_sizes.append(page_size)

                async def rows():
                    for i in range(2500):
                        yield {"_id": f"chunk-{i}"}

                return rows()

        assert await count_kv_rows(StreamingKV()) == 2500
        assert seen_page_sizes == [1000]

    async def test_a_backend_without_a_row_stream_raises_the_capability_error(self):
        from lightrag.base import BaseKVStorage
        from lightrag.exceptions import StorageCapabilityError

        class BareKV:
            iter_rows = BaseKVStorage.iter_rows

        with pytest.raises(StorageCapabilityError):
            await count_kv_rows(BareKV())

    async def test_the_json_backend_is_counted_without_copying_its_keys(self):
        import asyncio as _asyncio

        class JsonKVStorage:
            def __init__(self):
                self._storage_lock = _asyncio.Lock()
                self._data = {f"chunk-{i}": {} for i in range(7)}

        assert await count_kv_rows(JsonKVStorage()) == 7

    async def test_the_redis_backend_is_counted_by_scanning_not_collecting(self):
        from contextlib import asynccontextmanager

        class FakeRedis:
            def __init__(self):
                self.pages = [(5, ["a", "b", "c"]), (0, ["d"])]

            async def scan(self, cursor, *, match, count):
                assert match == "ws_text_chunks:*"
                return self.pages.pop(0)

        class RedisKVStorage:
            final_namespace = "ws_text_chunks"

            @asynccontextmanager
            async def _get_redis_connection(self):
                yield FakeRedis()

        assert await count_kv_rows(RedisKVStorage()) == 4

    async def test_a_baseline_row_that_does_not_parse_is_unreadable_not_fatal(
        self, tmp_path, monkeypatch, stub_kv_count
    ):
        """``delete_workspace_configuration`` deletes by key without reading
        the value, so a row the clear will remove anyway must not block it."""

        async def read_config_row_strict(config, key):
            if key.endswith("/chunks"):
                return {"value": "not-a-dict", "workspace": "ws", "_id": key}
            return None

        monkeypatch.setattr(
            clear_storage, "read_config_row_strict", read_config_row_strict
        )
        tool = make_tool(tmp_path)

        summary = await tool.collect_summary()

        assert isinstance(summary["baselines"]["chunks"], Unreadable)
        assert summary["baselines"]["entities"] is None
        assert "chunks embedding baseline" in tool.unreadable_items(summary)

    async def test_a_baseline_row_that_cannot_be_fetched_refuses_the_run(
        self, tmp_path, monkeypatch, stub_kv_count
    ):
        """Transport, not parsing: the records could not be deleted after
        the clear either."""
        from lightrag.exceptions import ConfigurationStorageError

        async def read_config_row_strict(config, key):
            raise ConfigurationStorageError("could not read configuration record")

        monkeypatch.setattr(
            clear_storage, "read_config_row_strict", read_config_row_strict
        )
        tool = make_tool(tmp_path)

        with pytest.raises(RemoteBackendUnavailableError):
            await tool.collect_summary()


class TestInputDirResolution:
    """``DocumentManager`` keeps a named workspace's uploads under
    ``INPUT_DIR/<workspace>``; the tool must delete THOSE, not the default
    workspace's files one level up."""

    def test_a_named_workspace_uses_its_subdirectory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
        tool = ClearTool()
        tool.workspace = "ws1"
        assert tool.resolve_input_dir() == str(tmp_path / "inputs" / "ws1")

    def test_the_default_workspace_uses_the_base_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
        tool = ClearTool()
        tool.workspace = ""
        assert tool.resolve_input_dir() == str(tmp_path / "inputs")

    def test_a_traversing_workspace_name_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
        tool = ClearTool()
        tool.workspace = "../other"
        with pytest.raises(ValueError):
            tool.resolve_input_dir()


class TestKindRule:
    @pytest.mark.parametrize(
        "name",
        [
            "JsonKVStorage",
            "JsonDocStatusStorage",
            "NetworkXStorage",
            "NanoVectorDBStorage",
            "FaissVectorDBStorage",
        ],
    )
    def test_file_backed_backends(self, name):
        assert is_server_backed(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "PGKVStorage",
            "RedisDocStatusStorage",
            "Neo4JStorage",
            "QdrantVectorDBStorage",
            "MilvusVectorDBStorage",
            "OpenSearchVectorDBStorage",
        ],
    )
    def test_server_backends(self, name):
        assert is_server_backed(name) is True

    def test_an_unknown_backend_is_treated_as_a_server(self):
        """Refusing a run is recoverable; a partial clear is not."""
        assert is_server_backed("SomeCustomStorage") is True


# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------


class TestConfirmation:
    @pytest.mark.parametrize(
        "answer", ["", "yes", "delete all", "DELETE ALL", "Delete all", "Delete All!"]
    )
    def test_only_the_exact_phrase_confirms(self, tmp_path, monkeypatch, answer):
        tool = make_tool(tmp_path)
        answer_prompts(monkeypatch, answer)
        assert tool.confirm_delete_all() is False

    @pytest.mark.parametrize(
        "answer", [CONFIRMATION_PHRASE, f"  {CONFIRMATION_PHRASE} "]
    )
    def test_the_phrase_confirms_with_surrounding_whitespace_ignored(
        self, tmp_path, monkeypatch, answer
    ):
        tool = make_tool(tmp_path)
        answer_prompts(monkeypatch, answer)
        assert tool.confirm_delete_all() is True

    async def test_declining_leaves_everything_untouched_and_exits_clean(
        self,
        tmp_path,
        monkeypatch,
        stub_kv_count,
        stub_baselines,
        recorded_config_delete,
    ):
        seed_input_dir(tmp_path)
        tool = make_tool(tmp_path)
        monkeypatch.setattr(tool, "setup_storages", AsyncMock(return_value=True))
        stub_share_data(monkeypatch)
        answer_prompts(monkeypatch, "yes", "no thanks")

        ok = await tool.run()

        assert ok is True, "a deliberate cancellation is not a failure"
        for label in DATA_STORAGE_LABELS:
            tool.storages[label].drop.assert_not_called()
        assert recorded_config_delete == []
        assert (tmp_path / "inputs" / "a.txt").exists()

    async def test_not_stopping_the_server_cancels_before_any_storage_opens(
        self, tmp_path, monkeypatch
    ):
        tool = ClearTool()
        setup = AsyncMock(return_value=True)
        monkeypatch.setattr(tool, "setup_storages", setup)
        stub_share_data(monkeypatch)
        answer_prompts(monkeypatch, "no")

        assert await tool.run() is True
        setup.assert_not_called()


# ---------------------------------------------------------------------------
# The clear
# ---------------------------------------------------------------------------


class TestClear:
    async def test_a_clean_clear_drops_all_eleven_then_records_then_input_files(
        self, tmp_path, recorded_config_delete, capsys
    ):
        inputs = seed_input_dir(tmp_path)
        tool = make_tool(tmp_path, workspace="ws")

        ok = await tool.clear()

        assert ok is True
        for label in DATA_STORAGE_LABELS:
            tool.storages[label].drop.assert_awaited_once()
        assert recorded_config_delete == ["ws"]
        assert not (inputs / "a.txt").exists()
        assert not (inputs / "b.pdf").exists()
        assert (inputs / "__parsed__" / "a.md").exists(), "__parsed__ is preserved"
        assert "Workspace cleared" in capsys.readouterr().out

    async def test_the_llm_cache_is_not_among_the_dropped_storages(self):
        assert "llm_response_cache" not in DATA_STORAGE_LABELS
        assert len(DATA_STORAGE_LABELS) == 11

    async def test_every_storage_is_attempted_even_after_one_fails(self, tmp_path):
        tool = make_tool(
            tmp_path,
            full_docs=make_storage("full_docs", drop_error=RuntimeError("down")),
        )

        outcome = await tool.drop_all()

        for label in DATA_STORAGE_LABELS:
            tool.storages[label].drop.assert_awaited_once()
        assert [label for label, _ in outcome.failed] == ["full_docs"]
        assert len(outcome.succeeded) == 10

    async def test_a_non_success_dict_counts_as_a_failed_drop(self, tmp_path):
        tool = make_tool(
            tmp_path,
            chunks_vdb=make_storage(
                "chunks_vdb", drop_result={"status": "error", "message": "kept"}
            ),
        )

        outcome = await tool.drop_all()

        assert outcome.failed == [("chunks_vdb", "kept")]

    async def test_a_partial_drop_keeps_the_configuration_records_and_fails(
        self, tmp_path, recorded_config_delete, capsys
    ):
        """Records gone with data still present is the residue *Workspace
        drop* never accepts; a failed drop keeps all three and the run exits
        non-zero so automation does not read it as clean."""
        seed_input_dir(tmp_path)
        tool = make_tool(
            tmp_path,
            doc_status=make_storage("doc_status", drop_error=RuntimeError("locked")),
        )

        ok = await tool.clear()

        assert ok is False
        assert recorded_config_delete == []
        out = capsys.readouterr().out
        assert "Kept the workspace configuration records" in out
        assert "doc_status" in out

    async def test_a_cancelled_drop_keeps_the_configuration_records(
        self, tmp_path, recorded_config_delete
    ):
        tool = make_tool(
            tmp_path,
            entities_vdb=make_storage(
                "entities_vdb", drop_error=asyncio.CancelledError()
            ),
        )

        ok = await tool.clear()

        assert ok is False
        assert recorded_config_delete == []

    async def test_a_failed_records_delete_is_a_failed_clear(
        self, tmp_path, monkeypatch, capsys
    ):
        async def failing_delete(config, workspace):
            raise RuntimeError("config store unreachable")

        monkeypatch.setattr(
            clear_storage, "delete_workspace_configuration", failing_delete
        )
        tool = make_tool(tmp_path)

        ok = await tool.clear()

        assert ok is False
        assert "Could not delete the workspace configuration records" in (
            capsys.readouterr().out
        )

    async def test_when_every_drop_fails_the_input_files_stay(
        self, tmp_path, recorded_config_delete
    ):
        """Files left in place can be re-ingested once the storages are
        reachable again; deleting them beside untouched storages loses the
        only copy for nothing."""
        inputs = seed_input_dir(tmp_path)
        tool = make_tool(
            tmp_path,
            **{
                label: make_storage(label, drop_error=RuntimeError("down"))
                for label in DATA_STORAGE_LABELS
            },
        )

        ok = await tool.clear()

        assert ok is False
        assert (inputs / "a.txt").exists()
        assert recorded_config_delete == []

    async def test_a_missing_input_dir_is_not_an_error(
        self, tmp_path, recorded_config_delete
    ):
        tool = make_tool(tmp_path)  # tmp_path/inputs never created
        assert await tool.clear() is True

    def test_drop_outcome_all_succeeded(self):
        assert DropOutcome(succeeded=["a"]).all_succeeded is True
        assert DropOutcome(failed=[("a", "x")]).all_succeeded is False


# ---------------------------------------------------------------------------
# Setup against the real file-backed storages
# ---------------------------------------------------------------------------


def setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes, *, names=None):
    """Point ``setup_storages`` at ``fakes`` (label -> storage or exception)."""
    tool.storage_names = dict(names or FILE_BACKED_NAMES)
    monkeypatch.setattr(tool, "resolve_storage_names", lambda: tool.storage_names)
    monkeypatch.setenv("WORKING_DIR", str(tmp_path))
    monkeypatch.setenv("WORKSPACE", "clearws")
    monkeypatch.setattr(clear_storage, "uses_working_dir", lambda name: False)

    def build_storage(label, embedding_func):
        fake = fakes[label]
        if isinstance(fake, BaseException):
            raise fake
        return fake

    monkeypatch.setattr(tool, "build_storage", build_storage)
    monkeypatch.setattr(tool, "build_embedding_func", lambda: fixed_embedding_func())
    config = make_storage("config")
    config.initialize = AsyncMock()
    monkeypatch.setattr(
        clear_storage, "create_configuration_storage", lambda *a, **k: config
    )
    return config


def openable_fakes():
    fakes = {label: make_storage(label) for label in DATA_STORAGE_LABELS}
    for fake in fakes.values():
        fake.initialize = AsyncMock()
    return fakes


class TestSetup:
    async def test_a_refused_vector_target_is_recorded_not_fatal(
        self, tmp_path, monkeypatch, capsys
    ):
        """The two typed refusals are the states this tool exists to clear,
        on every backend: ``drop()`` is servable while refused."""
        from lightrag.exceptions import VectorSpaceMismatchError

        refusal = VectorSpaceMismatchError(
            backend="FakeVectorStorage",
            container="chunks_index",
            expected_model="new-model",
            expected_dim=8,
            stored_model="old-model",
            stored_dim=8,
        )
        fakes = openable_fakes()
        fakes["chunks_vdb"].initialize = AsyncMock(side_effect=refusal)
        tool = ClearTool()
        names = {**FILE_BACKED_NAMES, "vector": "QdrantVectorDBStorage"}
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes, names=names)

        ok = await tool.setup_storages()

        assert ok is True
        assert list(tool.refused_vdbs) == ["chunks_vdb"]
        assert tool.unavailable == {}
        assert "refused to attach" in capsys.readouterr().out
        for label in DATA_STORAGE_LABELS:
            fakes[label].initialize.assert_awaited_once()

    async def test_a_server_backend_that_cannot_open_refuses_the_run(
        self, tmp_path, monkeypatch, capsys
    ):
        """Clearing the others would leave the unreachable one populated."""
        fakes = openable_fakes()
        fakes["entities_vdb"].initialize = AsyncMock(
            side_effect=ConnectionError("vector backend down")
        )
        tool = ClearTool()
        names = {**FILE_BACKED_NAMES, "vector": "QdrantVectorDBStorage"}
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes, names=names)

        assert await tool.setup_storages() is False
        out = capsys.readouterr().out
        assert "refuses the run" in out
        assert "vector backend down" in out

    async def test_a_file_backed_storage_that_cannot_open_is_dropped_anyway(
        self, tmp_path, monkeypatch, capsys
    ):
        """A corrupt local file is data the operator is deleting: shown,
        marked unavailable, and still handed to ``drop()``."""
        fakes = openable_fakes()
        fakes["text_chunks"].initialize = AsyncMock(
            side_effect=ValueError("Expecting value: line 2 column 1")
        )
        tool = ClearTool()
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes)

        assert await tool.setup_storages() is True
        assert list(tool.unavailable) == ["text_chunks"]
        assert "will be dropped anyway" in capsys.readouterr().out

        outcome = await tool.drop_all()

        fakes["text_chunks"].drop.assert_awaited_once()
        assert outcome.all_succeeded

    async def test_a_storage_that_cannot_be_constructed_is_a_failed_drop(
        self, tmp_path, monkeypatch, capsys
    ):
        """NetworkX parses its file in the constructor, so a corrupt GraphML
        leaves no instance to drop. Reported as a failed drop -- the records
        stay, the exit is non-zero -- never silently skipped."""
        fakes = openable_fakes()
        fakes["chunk_entity_relation_graph"] = SyntaxError("syntax error: line 1")
        tool = ClearTool()
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes)

        assert await tool.setup_storages() is True
        assert tool.storages["chunk_entity_relation_graph"] is None
        assert "chunk_entity_relation_graph" in tool.unavailable

        outcome = await tool.drop_all()

        failed = dict(outcome.failed)
        assert "not constructed" in failed["chunk_entity_relation_graph"]
        assert "by hand" in failed["chunk_entity_relation_graph"]
        assert len(outcome.succeeded) == 10

    async def test_a_server_backend_that_cannot_be_constructed_refuses_the_run(
        self, tmp_path, monkeypatch, capsys
    ):
        fakes = openable_fakes()
        fakes["doc_status"] = ConnectionError("cannot resolve host")
        tool = ClearTool()
        names = {**FILE_BACKED_NAMES, "doc_status": "PGDocStatusStorage"}
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes, names=names)

        assert await tool.setup_storages() is False
        assert "refuses the run" in capsys.readouterr().out

    async def test_the_storages_receive_the_server_factory_embedding_function(
        self, tmp_path, monkeypatch
    ):
        """Qdrant, PostgreSQL and Milvus name their container after the
        model and dimension, and the server derives an omitted EMBEDDING_DIM
        from the provider default. The tool must hand every storage exactly
        the factory's function -- a guessed dimension opens, and drops, a
        container the server never wrote to."""
        monkeypatch.delenv("EMBEDDING_DIM", raising=False)
        received = []
        fakes = openable_fakes()
        tool = ClearTool()
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes)
        from_factory = fixed_embedding_func(
            dim=1536, model_name="text-embedding-3-small"
        )
        monkeypatch.setattr(tool, "build_embedding_func", lambda: from_factory)

        def build_storage(label, embedding_func):
            received.append(embedding_func)
            return fakes[label]

        monkeypatch.setattr(tool, "build_storage", build_storage)

        assert await tool.setup_storages() is True
        assert len(received) == len(DATA_STORAGE_LABELS)
        assert all(f is from_factory for f in received)
        assert tool.global_config["embedding_func"] is from_factory

    async def test_build_embedding_func_uses_the_server_factory(self, monkeypatch):
        """Same factory as the server and lightrag-rebuild-vdb, called on the
        server's parsed arguments, so model and dimension can only agree."""
        # Importing the server package parses argv once for the process;
        # pytest's own arguments would make argparse exit.
        monkeypatch.setattr(sys, "argv", ["lightrag-clear-storage"])
        import lightrag.api.config as api_config
        import lightrag.api.lightrag_server as server

        fake_args = SimpleNamespace(embedding_binding="openai")
        from_factory = fixed_embedding_func(
            dim=1536, model_name="text-embedding-3-small"
        )
        seen = []

        def factory(args):
            seen.append(args)
            return from_factory

        monkeypatch.setattr(api_config, "global_args", fake_args)
        monkeypatch.setattr(server, "create_embedding_function_from_args", factory)

        assert ClearTool().build_embedding_func() is from_factory
        assert seen == [fake_args]

    async def test_without_the_api_extra_the_run_is_refused(
        self, tmp_path, monkeypatch, capsys
    ):
        """No stub fallback, unlike the rebuild's check-only mode: the tool
        is destructive, and a dimension it cannot resolve is a container it
        cannot name."""
        fakes = openable_fakes()
        tool = ClearTool()
        setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes)
        monkeypatch.setattr(tool, "build_embedding_func", lambda: None)

        assert await tool.setup_storages() is False
        assert tool.storages == {}
        for fake in fakes.values():
            fake.initialize.assert_not_called()

    async def test_the_import_failure_message_names_the_extra(
        self, monkeypatch, capsys
    ):
        import builtins

        real_import = builtins.__import__

        def refuse_api(name, *args, **kwargs):
            if name.startswith("lightrag.api"):
                raise ImportError("No module named 'fastapi'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_api)

        assert ClearTool().build_embedding_func() is None
        assert "lightrag-hku[api]" in capsys.readouterr().out

    async def test_a_configuration_storage_that_cannot_open_refuses_the_run(
        self, tmp_path, monkeypatch, capsys
    ):
        """The records are deleted LAST; a store that cannot be opened could
        not take that step, and file-backed or not that is not a clean clear."""
        fakes = openable_fakes()
        tool = ClearTool()
        config = setup_tool_on_fakes(tool, tmp_path, monkeypatch, fakes)
        config.initialize = AsyncMock(side_effect=OSError("config dir unwritable"))

        assert await tool.setup_storages() is False
        assert "configuration storage could not be opened" in capsys.readouterr().out
        assert tool.storages == {}

    async def test_setup_refuses_while_another_process_tree_holds_the_config_dir(
        self, tmp_path, monkeypatch, capsys
    ):
        """Same second-process-tree hazard ``lightrag-rebuild-vdb`` guards:
        a server holding a file-backed configuration would republish the
        records this tool deletes. The refusal comes before any storage is
        opened."""
        from lightrag.kg import working_dir_lock as wdl

        monkeypatch.setattr(sys, "argv", ["lightrag-clear-storage"])
        monkeypatch.setenv("WORKING_DIR", str(tmp_path))
        monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        monkeypatch.setenv("WORKSPACE", "clearws")

        config_dir = str(tmp_path / CONFIG_CONTAINER_TAG)
        wdl.acquire_working_dir_lock(config_dir)
        holder = dict(wdl._claims)
        wdl._claims.clear()  # the tool must look like a different tree

        tool = ClearTool()
        try:
            ok = await tool.setup_storages()

            assert ok is False
            assert "already in use" in capsys.readouterr().out
            assert tool.storages == {}, "no storage was opened despite the refusal"
            assert tool.configuration_storage is None
            assert tool._holds_working_dir is False
        finally:
            wdl._claims.update(holder)
            wdl.release_working_dir_lock(config_dir)


# ---------------------------------------------------------------------------
# End to end on the JSON backends
# ---------------------------------------------------------------------------


class TestEndToEndOnJsonBackends:
    async def test_a_seeded_workspace_is_summarized_then_emptied(
        self, tmp_path, monkeypatch, capsys, stub_embedding_factory
    ):
        """The real JSON storages, seeded through the tool's own construction:
        the summary reports the seeded rows, the phrase empties them, the
        LLM cache file is never created, __parsed__ survives."""
        from lightrag.kg.shared_storage import (
            finalize_share_data,
            initialize_share_data,
        )
        from lightrag.kg.working_dir_lock import release_working_dir_lock

        working_dir = tmp_path / "wd"
        base_inputs = tmp_path / "inputs"
        inputs = base_inputs / "e2e"
        (inputs / "__parsed__").mkdir(parents=True)
        (inputs / "a.txt").write_text("a")
        (inputs / "__parsed__" / "a.md").write_text("parsed")
        # The DEFAULT workspace's upload, one level up: not this run's to delete.
        (base_inputs / "default-ws.txt").write_text("keep")
        monkeypatch.setenv("WORKING_DIR", str(working_dir))
        monkeypatch.setenv("INPUT_DIR", str(base_inputs))
        monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        monkeypatch.setenv("LIGHTRAG_CONFIG_STORAGE", "")
        monkeypatch.setenv("WORKSPACE", "e2e")

        # Seed through a first tool instance, then release everything it held.
        initialize_share_data(workers=1)
        seeder = ClearTool()
        assert await seeder.setup_storages() is True
        await seeder.storages["doc_status"].upsert(
            {
                "doc-1": {
                    "status": "processed",
                    "content_summary": "x",
                    "content_length": 1,
                    "file_path": "one.txt",
                    "created_at": "2026-09-01T00:00:00",
                    "updated_at": "2026-09-01T00:00:00",
                    "chunks_list": [],
                },
                "doc-2": {
                    "status": "failed",
                    "content_summary": "y",
                    "content_length": 1,
                    "file_path": "two.txt",
                    "created_at": "2026-09-02T00:00:00",
                    "updated_at": "2026-09-02T00:00:00",
                    "chunks_list": [],
                },
            }
        )
        await seeder.storages["text_chunks"].upsert(
            {f"chunk-{i}": {"content": "c", "full_doc_id": "doc-1"} for i in range(3)}
        )
        await seeder.storages["doc_status"].index_done_callback()
        await seeder.storages["text_chunks"].index_done_callback()
        for storage in [*seeder.storages.values(), seeder.configuration_storage]:
            await storage.finalize()
        finalize_share_data()
        release_working_dir_lock(seeder.config_dir)

        answer_prompts(monkeypatch, "yes", CONFIRMATION_PHRASE)
        tool = ClearTool()
        ok = await tool.run()
        out = capsys.readouterr().out

        assert ok is True
        assert "processed      1" in out and "failed         1" in out
        assert "two.txt" in out and "one.txt" in out
        assert "Text chunks: 3" in out
        assert "Workspace cleared" in out
        assert (working_dir / "e2e" / "kv_store_doc_status.json").read_text() == "{}"
        assert (working_dir / "e2e" / "kv_store_text_chunks.json").read_text() == "{}"
        assert not (working_dir / "e2e" / "kv_store_llm_response_cache.json").exists()
        assert "e2e" in out.split("Input files to delete")[1].split("\n")[0]
        assert not (inputs / "a.txt").exists()
        assert (inputs / "__parsed__" / "a.md").exists()
        assert (base_inputs / "default-ws.txt").exists(), (
            "the default workspace's upload one level up was deleted"
        )

    async def test_corrupt_local_files_are_shown_unreadable_and_still_cleared(
        self, tmp_path, monkeypatch, capsys, stub_embedding_factory
    ):
        """Fix-proof for the kind rule on the real JSON backends: a corrupt
        ``text_chunks`` file and a baseline row that does not parse used to
        abort the run with nothing deleted. Both are data the operator is
        deleting, so the summary shows them UNREADABLE and the clear rewrites
        the file empty and removes the row."""
        import json

        working_dir = tmp_path / "wd"
        workspace_dir = working_dir / "e2e"
        config_dir = working_dir / CONFIG_CONTAINER_TAG
        workspace_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        (workspace_dir / "kv_store_text_chunks.json").write_text('{"chunk-1": ')
        (config_dir / "kv_store_config.json").write_text(
            json.dumps(
                {"e2e/embedding/chunks": {"value": "not-a-dict", "workspace": "e2e"}}
            )
        )
        monkeypatch.setenv("WORKING_DIR", str(working_dir))
        monkeypatch.setenv("INPUT_DIR", str(tmp_path / "inputs"))
        monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
        monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
        monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
        monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")
        monkeypatch.setenv("LIGHTRAG_CONFIG_STORAGE", "")
        monkeypatch.setenv("WORKSPACE", "e2e")
        answer_prompts(monkeypatch, "yes", CONFIRMATION_PHRASE)

        ok = await ClearTool().run()
        out = capsys.readouterr().out

        assert ok is True
        assert "text_chunks (JsonKVStorage) could not be opened" in out
        assert "UNREADABLE" in out
        assert "chunks embedding baseline" in out
        assert "Workspace cleared" in out
        assert (workspace_dir / "kv_store_text_chunks.json").read_text() == "{}"
        assert json.loads((config_dir / "kv_store_config.json").read_text()) == {}
