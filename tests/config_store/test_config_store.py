"""The configuration store's own rules, driven with doubles.

Keys, the row shape, the registry, strict reads, the precheck that names every
mismatched target at once, the atomic claim (flush BEFORE the lock is
released, then a strict read-back), the rebuild record, the data-first drop
and the enumeration surface. See docs/design/ConfigurationStorage.md; the
scenario numbers below are its acceptance list.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lightrag import config_store as cs
from lightrag.exceptions import (
    ConfigurationStorageError,
    EmbeddingBaselineMismatchError,
    StorageCapabilityError,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_WORKSPACE, SERVER_CONFIG_SCOPE

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def _embedding(model="bge-m3", dim=16):
    return SimpleNamespace(model_name=model, embedding_dim=dim)


class FakeConfigKV:
    """A KV storage with strict reads, a process-local write buffer that only
    becomes visible on flush (the OpenSearch shape), and a call log."""

    supports_strict_point_reads = True

    def __init__(
        self,
        rows=None,
        *,
        read_error=None,
        write_error=None,
        flush_error=None,
        delete_swallowed=False,
        visible_on_upsert=False,
        read_delay=0.0,
    ):
        self.visible = dict(rows or {})
        self.pending = {}
        self.calls = []
        self.read_error = read_error
        self.write_error = write_error
        self.flush_error = flush_error
        self.delete_swallowed = delete_swallowed
        self.visible_on_upsert = visible_on_upsert
        self.read_delay = read_delay
        self.finalized = 0

    async def get_by_id_strict(self, key):
        self.calls.append(("read", key))
        if self.read_delay:
            await asyncio.sleep(self.read_delay)
        if self.read_error is not None:
            raise self.read_error
        row = self.visible.get(key)
        return None if row is None else dict(row)

    async def upsert(self, data):
        self.calls.append(("upsert", tuple(data)))
        if self.write_error is not None:
            raise self.write_error
        if self.visible_on_upsert:
            self.visible.update(data)
        else:
            self.pending.update(data)

    async def index_done_callback(self):
        self.calls.append(("flush",))
        if self.flush_error is not None:
            raise self.flush_error
        self.visible.update(self.pending)
        self.pending.clear()

    async def delete(self, ids):
        self.calls.append(("delete", tuple(ids)))
        if self.delete_swallowed:
            return
        for key in ids:
            self.visible.pop(key, None)

    async def finalize(self):
        self.finalized += 1

    def iter_rows(self, *, page_size=200):
        async def _gen():
            for key, row in list(self.visible.items()):
                yield {**row, "_id": key}

        return _gen()


def _row(model="bge-m3", dim=16, origin="probe", workspace="ws", target="entities"):
    return cs.make_config_row(
        scope_workspace=workspace,
        suffix=cs.embedding_baseline_suffix(target),
        value={"model": model, "dim": dim, "origin": origin},
        updated_by="test",
    )


def _key(target, workspace="ws"):
    return cs.embedding_baseline_key(workspace, target)


# ---------------------------------------------------------------------------
# Keys, rows, registry
# ---------------------------------------------------------------------------


class TestKeysAndRows:
    def test_keys_use_the_slash_separator_and_are_never_dotted(self):
        assert _key("entities") == "ws/embedding/entities"
        # A dotted workspace is legitimate and stays intact: nothing reparses.
        assert cs.embedding_baseline_key("v1.0", "chunks") == "v1.0/embedding/chunks"

    def test_a_workspace_name_cannot_carry_the_separator(self):
        with pytest.raises(ValueError):
            cs.config_key("a/b", cs.embedding_baseline_suffix("entities"))

    def test_an_unregistered_suffix_is_refused(self):
        with pytest.raises(KeyError, match="not registered"):
            cs.config_key("ws", "embedding/something_else")
        with pytest.raises(KeyError):
            cs.make_config_row(
                scope_workspace="ws", suffix="nope", value={}, updated_by="t"
            )

    def test_a_per_workspace_key_is_not_filed_under_the_server_scope(self):
        with pytest.raises(ValueError):
            cs.config_key(SERVER_CONFIG_SCOPE, cs.embedding_baseline_suffix("entities"))

    def test_the_registry_declares_the_five_fields_for_every_key(self):
        assert set(cs.CONFIG_KEY_REGISTRY) == {
            "embedding/entities",
            "embedding/relationships",
            "embedding/chunks",
        }
        for spec in cs.CONFIG_KEY_REGISTRY.values():
            assert spec.scope is cs.ConfigScope.WORKSPACE
            assert spec.schema_version == 1
            assert spec.schema
            assert spec.readers and spec.writers
            assert spec.sensitive is False

    def test_the_row_carries_the_scope_as_a_field(self):
        row = _row(workspace="tenant-a")
        assert row["workspace"] == "tenant-a"
        assert row["schema_version"] == 1
        assert row["updated_by"] == "test"
        assert row["updated_at"]
        assert row["value"] == {"model": "bge-m3", "dim": 16, "origin": "probe"}

    def test_a_value_must_be_a_mapping(self):
        with pytest.raises(TypeError):
            cs.make_config_row(
                scope_workspace="ws",
                suffix="embedding/entities",
                value=["not", "a", "mapping"],
                updated_by="t",
            )


class TestEmbeddingBaseline:
    def test_a_stored_row_parses(self):
        baseline = cs.EmbeddingBaseline.from_row(_row(), key="k")
        assert baseline == cs.EmbeddingBaseline("bge-m3", 16, "probe")

    @pytest.mark.parametrize(
        "row",
        [
            {"value": "not a mapping"},
            {"value": {"dim": 16, "origin": "probe"}},
            {"value": {"model": "", "dim": 16}},
            {"value": {"model": "m", "dim": "sixteen"}},
            {"value": {"model": "m", "dim": True}},
            "a bare string",
        ],
    )
    def test_a_malformed_row_is_unreadable_not_absent(self, row):
        with pytest.raises(ConfigurationStorageError):
            cs.EmbeddingBaseline.from_row(row, key="k")

    def test_the_model_decides_and_the_dimension_only_when_both_declare_one(self):
        recorded = cs.EmbeddingBaseline("bge-m3", 16, "probe")
        assert recorded.differs_from(_embedding("bge-m3", 16)) is False
        assert recorded.differs_from(_embedding("other", 16)) is True
        assert recorded.differs_from(_embedding("bge-m3", 32)) is True
        # Absent evidence never refuses.
        assert recorded.differs_from(_embedding("bge-m3", None)) is False
        assert (
            cs.EmbeddingBaseline("bge-m3", None, "probe").differs_from(
                _embedding("bge-m3", 32)
            )
            is False
        )

    def test_a_process_without_a_model_name_has_nothing_to_record(self):
        assert cs.configured_baseline(_embedding(None), origin="empty") is None
        assert cs.configured_baseline(_embedding("  "), origin="empty") is None
        assert cs.configured_baseline(_embedding("m", 8), origin="rebuild") == (
            cs.EmbeddingBaseline("m", 8, "rebuild")
        )

    def test_an_unknown_origin_is_a_programming_error(self):
        with pytest.raises(ValueError):
            cs.configured_baseline(_embedding(), origin="guessed")


# ---------------------------------------------------------------------------
# Strict reads and the precheck
# ---------------------------------------------------------------------------


class TestStrictReads:
    async def test_a_backend_without_strict_reads_cannot_answer(self):
        class Loose:
            supports_strict_point_reads = False

            async def get_by_id_strict(self, key):
                return None

        with pytest.raises(ConfigurationStorageError, match="strict point reads"):
            await cs.read_config_row_strict(Loose(), "k")

    async def test_a_transport_failure_is_a_failure_not_an_absence(self):
        """Scenario 15."""
        kv = FakeConfigKV(read_error=ConnectionError("down"))
        with pytest.raises(ConfigurationStorageError) as excinfo:
            await cs.read_embedding_baselines(kv, "ws")
        assert isinstance(excinfo.value.__cause__, ConnectionError)

    async def test_a_confirmed_absence_reads_as_none(self):
        kv = FakeConfigKV({_key("entities"): _row()})
        recorded = await cs.read_embedding_baselines(kv, "ws")
        assert recorded["entities"] == cs.EmbeddingBaseline("bge-m3", 16, "probe")
        assert recorded["relationships"] is None
        assert recorded["chunks"] is None


class TestPrecheck:
    def test_all_three_matching_proceed_with_nothing_to_bootstrap(self):
        """Scenario 1."""
        recorded = {
            t: cs.EmbeddingBaseline("bge-m3", 16, "probe") for t in cs.EMBEDDING_TARGETS
        }
        assert (
            cs.precheck_embedding_baselines(recorded, _embedding(), workspace="ws")
            == []
        )

    def test_absent_targets_are_returned_for_bootstrap(self):
        recorded = {
            "entities": cs.EmbeddingBaseline("bge-m3", 16, "probe"),
            "relationships": None,
            "chunks": None,
        }
        assert cs.precheck_embedding_baselines(
            recorded, _embedding(), workspace="ws"
        ) == [
            "relationships",
            "chunks",
        ]

    @pytest.mark.parametrize("target", ["entities", "relationships", "chunks"])
    def test_one_mismatching_target_is_named(self, target):
        """Scenarios 2, 3, 4: the refusal says WHICH target."""
        recorded = {
            t: cs.EmbeddingBaseline("bge-m3", 16, "probe") for t in cs.EMBEDDING_TARGETS
        }
        recorded[target] = cs.EmbeddingBaseline("old-model", 16, "rebuild")
        with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
            cs.precheck_embedding_baselines(recorded, _embedding(), workspace="ws")
        assert excinfo.value.targets == [target]
        assert target in str(excinfo.value)
        assert "lightrag-rebuild-vdb" in str(excinfo.value)

    def test_several_mismatches_are_one_refusal_listing_all_of_them(self):
        """Scenario 5: an operator planning a rebuild needs the whole list."""
        recorded = {
            "entities": cs.EmbeddingBaseline("old", 16, "rebuild"),
            "relationships": cs.EmbeddingBaseline("bge-m3", 32, "rebuild"),
            "chunks": cs.EmbeddingBaseline("bge-m3", 16, "probe"),
        }
        with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
            cs.precheck_embedding_baselines(recorded, _embedding(), workspace="ws")
        assert excinfo.value.targets == ["entities", "relationships"]
        message = str(excinfo.value)
        assert "entities" in message and "relationships" in message
        assert "2 vector target(s)" in message


# ---------------------------------------------------------------------------
# The atomic claim
# ---------------------------------------------------------------------------


class TestClaim:
    async def test_the_claim_flushes_and_reads_back_before_the_lock_is_released(
        self, monkeypatch
    ):
        """Scenario 10. ``OpenSearchKVStorage.upsert`` buffers in process
        memory until the flush, so a flush after the release lets another
        worker read absent and claim again. The order inside the lock is
        read, upsert, flush, read-back -- and the release comes after all of
        them."""
        kv = FakeConfigKV()
        events: list[str] = []

        class RecordingLock:
            async def __aenter__(self):
                events.append("acquire")

            async def __aexit__(self, *exc):
                events.append("release")

        monkeypatch.setattr(
            "lightrag.kg.shared_storage.get_storage_keyed_lock",
            lambda keys, namespace="default", enable_logging=False: RecordingLock(),
        )
        original_calls = kv.calls
        kv.calls = events  # interleave the storage calls with the lock events
        candidate = cs.EmbeddingBaseline("bge-m3", 16, "empty")

        recorded = await cs.claim_embedding_baseline(
            kv,
            workspace="ws",
            target="entities",
            candidate=candidate,
            embedding_func=_embedding(),
        )

        kv.calls = original_calls
        assert recorded == candidate
        assert events[0] == "acquire" and events[-1] == "release"
        inner = events[1:-1]
        assert [e[0] if isinstance(e, tuple) else e for e in inner] == [
            "read",
            "upsert",
            "flush",
            "read",
        ]
        assert not kv.pending, "the write must be flushed, not left buffered"

    async def test_two_workers_claiming_at_once_produce_exactly_one_baseline(self):
        """Scenario 11: the claim covers the workers of one Gunicorn master,
        which share ``shared_storage`` -- and nothing wider (scenario 21 is
        that boundary, and nothing here asserts anything about two masters)."""
        kv = FakeConfigKV(read_delay=0.001)
        first = cs.EmbeddingBaseline("bge-m3", 16, "empty")
        second = cs.EmbeddingBaseline("bge-m3", 16, "probe")

        results = await asyncio.gather(
            cs.claim_embedding_baseline(
                kv,
                workspace="ws",
                target="chunks",
                candidate=first,
                embedding_func=_embedding(),
            ),
            cs.claim_embedding_baseline(
                kv,
                workspace="ws",
                target="chunks",
                candidate=second,
                embedding_func=_embedding(),
            ),
        )

        upserts = [c for c in kv.calls if c[0] == "upsert"]
        assert len(upserts) == 1
        # Both callers see the ONE record that exists, whoever wrote it.
        assert results[0] == results[1]
        assert results[0].origin in {"empty", "probe"}

    async def test_an_existing_record_that_differs_refuses_the_claim(self):
        kv = FakeConfigKV({_key("entities"): _row(model="old")})
        with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
            await cs.claim_embedding_baseline(
                kv,
                workspace="ws",
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 16, "probe"),
                embedding_func=_embedding(),
            )
        assert excinfo.value.targets == ["entities"]
        assert not [c for c in kv.calls if c[0] == "upsert"]

    async def test_a_read_back_that_finds_nothing_is_a_failure(self):
        class Amnesiac(FakeConfigKV):
            async def index_done_callback(self):
                self.calls.append(("flush",))
                self.pending.clear()  # the write vanishes instead of landing

        with pytest.raises(ConfigurationStorageError, match="read back nothing"):
            await cs.claim_embedding_baseline(
                Amnesiac(),
                workspace="ws",
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 16, "empty"),
                embedding_func=_embedding(),
            )

    async def test_a_flush_failure_is_a_failure(self):
        kv = FakeConfigKV(flush_error=OSError("disk full"))
        with pytest.raises(ConfigurationStorageError, match="could not flush"):
            await cs.claim_embedding_baseline(
                kv,
                workspace="ws",
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 16, "empty"),
                embedding_func=_embedding(),
            )


# ---------------------------------------------------------------------------
# The rebuild record and the drop
# ---------------------------------------------------------------------------


class TestRebuildRecord:
    async def test_recording_one_target_moves_only_that_key(self):
        """Scenario 8."""
        kv = FakeConfigKV(
            {
                _key("entities"): _row(model="old"),
                _key("relationships"): _row(model="old", target="relationships"),
            }
        )
        stored = await cs.record_embedding_baseline(
            kv, workspace="ws", target="chunks", embedding_func=_embedding()
        )
        assert stored == cs.EmbeddingBaseline("bge-m3", 16, "rebuild")
        assert kv.visible[_key("chunks")]["value"]["origin"] == "rebuild"
        assert kv.visible[_key("entities")]["value"]["model"] == "old"
        assert kv.visible[_key("relationships")]["value"]["model"] == "old"
        assert ("flush",) in kv.calls

    async def test_a_write_failure_raises_so_the_tool_exits_non_zero(self):
        """Scenario 9, the store's half: the failure surfaces typed; the tool
        turns it into a failed rebuild and a non-zero exit."""
        kv = FakeConfigKV(write_error=OSError("read-only"))
        with pytest.raises(ConfigurationStorageError):
            await cs.record_embedding_baseline(
                kv, workspace="ws", target="chunks", embedding_func=_embedding()
            )

    async def test_a_process_without_a_model_name_cannot_record(self):
        with pytest.raises(ConfigurationStorageError, match="model_name"):
            await cs.record_embedding_baseline(
                FakeConfigKV(),
                workspace="ws",
                target="chunks",
                embedding_func=_embedding(None),
            )


class TestDrop:
    async def test_the_three_records_are_deleted_and_flushed(self):
        kv = FakeConfigKV({_key(t): _row(target=t) for t in cs.EMBEDDING_TARGETS})
        await cs.delete_workspace_configuration(kv, "ws")
        assert kv.visible == {}
        deletes = [c for c in kv.calls if c[0] == "delete"]
        assert len(deletes) == 1
        assert set(deletes[0][1]) == {_key(t) for t in cs.EMBEDDING_TARGETS}
        assert kv.calls.index(("flush",)) > kv.calls.index(deletes[0])

    async def test_a_swallowed_delete_is_caught_by_the_read_back(self):
        """``PGKVStorage.delete`` logs and returns on failure; a removal that
        did not happen must not be reported as one."""
        kv = FakeConfigKV(
            {_key(t): _row(target=t) for t in cs.EMBEDDING_TARGETS},
            delete_swallowed=True,
        )
        with pytest.raises(ConfigurationStorageError, match="survived"):
            await cs.delete_workspace_configuration(kv, "ws")

    async def test_other_workspaces_are_untouched(self):
        kv = FakeConfigKV(
            {
                _key("entities"): _row(),
                _key("entities", workspace="other"): _row(workspace="other"),
            }
        )
        await cs.delete_workspace_configuration(kv, "ws")
        assert list(kv.visible) == [_key("entities", workspace="other")]


# ---------------------------------------------------------------------------
# A flush that retained its operation
# ---------------------------------------------------------------------------


class _RetainingKV(FakeConfigKV):
    """The OpenSearch shape when the server answers a retryable failure
    (429): ``index_done_callback`` keeps the operation buffered and returns
    normally, and the strict read answers from the buffer -- a buffered
    upsert as present, a buffered tombstone as gone. Flush-then-read-back
    alone would confirm a write the server never saw."""

    def __init__(self, rows=None):
        super().__init__(rows)
        self.tombstones = set()
        self.dropped = 0

    async def get_by_id_strict(self, key):
        self.calls.append(("read", key))
        if key in self.tombstones:
            return None
        if key in self.pending:
            return dict(self.pending[key])
        row = self.visible.get(key)
        return None if row is None else dict(row)

    async def index_done_callback(self):
        self.calls.append(("flush",))  # ...and retains everything: 429

    async def delete(self, ids):
        self.calls.append(("delete", tuple(ids)))
        self.tombstones.update(ids)

    async def has_pending_index_ops(self, *, include_deletes=False):
        return bool(self.pending) or (include_deletes and bool(self.tombstones))

    async def drop_pending_index_ops(self):
        self.dropped += 1
        self.pending.clear()
        self.tombstones.clear()


class TestRetainedFlush:
    async def test_a_claim_whose_flush_retained_the_write_fails(self):
        kv = _RetainingKV()
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.claim_embedding_baseline(
                kv,
                workspace="ws",
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 16, "empty"),
                embedding_func=_embedding(),
            )
        # Reported as not written, and made true: the buffer is dropped so a
        # later flush at shutdown cannot land an unvalidated record.
        assert kv.visible == {} and kv.pending == {} and kv.dropped == 1

    async def test_a_rebuild_record_whose_flush_retained_the_write_fails(self):
        kv = _RetainingKV({_key("chunks"): _row(model="old", target="chunks")})
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.record_embedding_baseline(
                kv, workspace="ws", target="chunks", embedding_func=_embedding()
            )
        assert kv.visible[_key("chunks")]["value"]["model"] == "old"

    async def test_a_drop_whose_flush_retained_the_tombstones_fails(self):
        """A buffered tombstone reads as "gone"; without the pending check the
        read-back would confirm a deletion the server still has rows for --
        the never-acceptable *configuration gone, data remains* direction."""
        kv = _RetainingKV({_key(t): _row(target=t) for t in cs.EMBEDDING_TARGETS})
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.delete_workspace_configuration(kv, "ws")
        assert len(kv.visible) == 3 and kv.tombstones == set()

    async def test_a_backend_without_a_buffer_is_unaffected(self):
        kv = FakeConfigKV()  # no has_pending_index_ops at all
        recorded = await cs.claim_embedding_baseline(
            kv,
            workspace="ws",
            target="entities",
            candidate=cs.EmbeddingBaseline("bge-m3", 16, "empty"),
            embedding_func=_embedding(),
        )
        assert recorded.model == "bge-m3"


# ---------------------------------------------------------------------------
# Enumeration and the factory
# ---------------------------------------------------------------------------


class TestEnumeration:
    async def test_rows_are_classified_by_their_workspace_field(self):
        kv = FakeConfigKV(
            {
                _key("entities", workspace="a"): _row(workspace="a"),
                _key("chunks", workspace="b"): _row(workspace="b", target="chunks"),
                "junk": {"value": {"x": 1}},  # no uniform shape: reported, not hidden
            }
        )
        rows = [row async for row in cs.iter_configuration_rows(kv)]
        assert [(r["id"], r["workspace"]) for r in rows] == [
            ("a/embedding/entities", "a"),
            ("b/embedding/chunks", "b"),
            ("junk", None),
        ]

    async def test_a_backend_without_enumeration_raises_on_first_row(self):
        from lightrag.base import BaseKVStorage

        class Bare:
            iter_rows = BaseKVStorage.iter_rows

        with pytest.raises(StorageCapabilityError):
            async for _ in cs.iter_configuration_rows(Bare()):
                pass


class TestFactory:
    def test_the_factory_binds_the_reserved_workspace(self, tmp_path):
        """Scenario 12, the accepting half."""
        from lightrag.kg.json_kv_impl import JsonKVStorage

        storage = cs.create_configuration_storage(
            JsonKVStorage,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_embedding(),
        )
        assert storage.workspace == CONFIG_WORKSPACE
        assert storage.namespace == "config"
        assert (tmp_path / CONFIG_WORKSPACE).is_dir()

    def test_a_backend_that_remaps_the_workspace_is_refused(self):
        class Remapping:
            def __init__(self, **kwargs):
                self.workspace = "prod"

        with pytest.raises(ConfigurationStorageError, match="must not be remapped"):
            cs.create_configuration_storage(
                Remapping, global_config={}, embedding_func=_embedding()
            )

    def test_the_grant_does_not_outlive_the_construction(self, tmp_path):
        from lightrag.kg.json_kv_impl import JsonKVStorage

        cs.create_configuration_storage(
            JsonKVStorage,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_embedding(),
        )
        with pytest.raises(ValueError, match="reserved"):
            JsonKVStorage(
                namespace="config",
                workspace=CONFIG_WORKSPACE,
                global_config={"working_dir": str(tmp_path)},
                embedding_func=_embedding(),
            )
