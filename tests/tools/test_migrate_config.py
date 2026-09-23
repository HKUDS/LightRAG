"""``lightrag-migrate-config``: the seven steps, driven with doubles.

The target's identity row is written first and is the ownership marker; the
anchor moves only after a strict flush and a full verification; a crash at
any step leaves the anchor on the source and a re-run converges the target to
the CURRENT source. See *Offline migration* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import os

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import (
    ConfigurationAnchorLockError,
    ConfigurationStorageError,
)
from lightrag.kg import anchor_lock as al
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import SERVER_SCOPE
from lightrag.tools import migrate_config as mc
from tests.config_store.test_config_store import FakeConfigKV

pytestmark = pytest.mark.offline

UUID_A = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"
UUID_B = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
IDENTITY_KEY = "_lightrag_server/storage_identity"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


class Container(FakeConfigKV):
    """A configuration container: rows become visible on flush (the
    OpenSearch shape), deletes and enumeration included, plus the metadata a
    backend adds to every row it returns."""

    def __init__(self, rows=None, *, merge=False, **kwargs):
        super().__init__(rows, **kwargs)
        # Mongo's ``$set`` and OpenSearch's scripted upsert MERGE into an
        # existing document: a field the new row lacks survives the upsert.
        self.merge = merge
        self.iter_error_after: int | None = None
        self.upsert_calls = 0
        self.fail_upsert_at: int | None = None
        self.corrupt_on_flush: dict | None = None

    async def upsert(self, data):
        self.upsert_calls += 1
        if self.fail_upsert_at is not None and self.upsert_calls >= self.fail_upsert_at:
            raise ConnectionError("target went away mid-copy")
        if self.merge and self.visible_on_upsert:
            self.calls.append(("upsert", tuple(data)))
            for key, row in data.items():
                self.visible[key] = {**self.visible.get(key, {}), **row}
            return
        await super().upsert(data)

    async def index_done_callback(self):
        await super().index_done_callback()
        if self.corrupt_on_flush:
            self.visible.update(self.corrupt_on_flush)

    def iter_rows(self, *, page_size=200):
        async def _gen():
            for n, (key, row) in enumerate(list(self.visible.items())):
                if self.iter_error_after is not None and n >= self.iter_error_after:
                    raise ConnectionError("cursor lost")
                if isinstance(row, dict):
                    yield {**row, "_id": key, "create_time": 1, "update_time": 2}
                else:
                    yield row

        return _gen()


def _identity_row(storage_uuid=UUID_A):
    return cs.make_config_row(
        scope_workspace=SERVER_SCOPE,
        suffix=cs.STORAGE_IDENTITY_SUFFIX,
        value={"uuid": storage_uuid},
        updated_by="lightrag.initialize_storages",
    )


def _baseline(workspace, target="entities", model="bge-m3"):
    return {
        cs.embedding_baseline_key(workspace, target): cs.make_config_row(
            scope_workspace=workspace,
            suffix=cs.embedding_baseline_suffix(target),
            value={"model": model, "dim": 1024, "origin": "probe"},
            updated_by="lightrag.initialize_storages",
        )
    }


def _source_rows():
    rows = {IDENTITY_KEY: _identity_row()}
    for ws in ("alpha", "beta"):
        for target in cs.EMBEDDING_TARGETS:
            rows.update(_baseline(ws, target))
    # A server-scope data row under a different, legal suffix spelling is not
    # registrable today, so a hand-built one stands in for slice 2's keys.
    rows["_lightrag_server/embedding.current"] = {
        "schema_version": 1,
        "workspace": "_lightrag_server",
        "updated_at": "2026-09-01T00:00:00+00:00",
        "updated_by": "future",
        "value": {"model": "bge-m3"},
    }
    return rows


def _anchor(tmp_path, backend="PGKVStorage", storage_uuid=UUID_A):
    ca.publish_anchor(
        str(tmp_path),
        ca.StorageAnchor(backend=backend, storage_uuid=storage_uuid),
        replace=False,
    )


def _data(container):
    return {
        k: mc.row_payload(v) for k, v in container.visible.items() if k != IDENTITY_KEY
    }


async def _migrate(tmp_path, source, target, *, target_backend="MongoKVStorage", **kw):
    async def _open_source(backend):
        assert backend == "PGKVStorage"
        return source

    async def _open_target(backend):
        assert backend == target_backend
        return target

    return await mc.migrate_configuration(
        working_dir=str(tmp_path),
        target_backend=target_backend,
        open_source=_open_source,
        open_target=_open_target,
        out=lambda line: None,
        **kw,
    )


class TestTheHappyPath:
    async def test_every_row_moves_and_the_anchor_follows_last(self, tmp_path):
        _anchor(tmp_path)
        source = Container(_source_rows())
        before = {k: dict(v) for k, v in source.visible.items()}
        target = Container()

        result = await _migrate(tmp_path, source, target)

        assert result.switched is True
        assert _data(target) == _data(source)
        assert result.source_scan.scopes == {
            "alpha": 3,
            "beta": 3,
            "_lightrag_server": 1,
        }
        # The same UUID, written by the migration, and the anchor on the target.
        assert target.visible[IDENTITY_KEY]["value"] == {"uuid": UUID_A}
        assert target.visible[IDENTITY_KEY]["updated_by"] == cs.UPDATED_BY_MIGRATE
        assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
            backend="MongoKVStorage", storage_uuid=UUID_A
        )
        # The source is kept, untouched.
        assert source.visible == before
        assert not [c for c in source.calls if c[0] in ("upsert", "delete")]
        # Both were released, and so was the lock.
        assert source.finalized == target.finalized == 1
        al.acquire_anchor_lock_exclusive(str(tmp_path)).release()

    async def test_the_identity_is_written_before_any_data_row(self, tmp_path):
        _anchor(tmp_path)
        target = Container()
        await _migrate(tmp_path, Container(_source_rows()), target)
        upserts = [c[1] for c in target.calls if c[0] == "upsert"]
        assert upserts[0] == (IDENTITY_KEY,)
        first_flush = target.calls.index(("flush",))
        first_data = next(
            i
            for i, c in enumerate(target.calls)
            if c[0] == "upsert" and c[1] != (IDENTITY_KEY,)
        )
        assert first_flush < first_data

    async def test_backend_metadata_is_not_content(self, tmp_path):
        _anchor(tmp_path)
        target = Container()
        await _migrate(tmp_path, Container(_source_rows()), target)
        for key, row in target.visible.items():
            assert "_id" not in row and "create_time" not in row, key

    async def test_a_dry_run_reports_and_writes_nothing(self, tmp_path):
        _anchor(tmp_path)
        before = open(ca.anchor_path(str(tmp_path)), "rb").read()
        source, target = Container(_source_rows()), Container()
        lines: list[str] = []

        async def _src(_):
            return source

        async def _tgt(_):
            return target

        result = await mc.migrate_configuration(
            working_dir=str(tmp_path),
            target_backend="MongoKVStorage",
            open_source=_src,
            open_target=_tgt,
            dry_run=True,
            out=lines.append,
        )
        assert result.switched is False
        assert result.verdict.state == "empty"
        assert target.calls and not _writes(target)
        assert open(ca.anchor_path(str(tmp_path)), "rb").read() == before
        report = "\n".join(lines)
        assert "7 row(s)" in report and "alpha (3)" in report
        assert "will claim" in report


def _writes(container):
    return [c for c in container.calls if c[0] in ("upsert", "delete", "flush")]


class TestRefusals:
    async def test_no_anchor_is_nothing_to_migrate(self, tmp_path):
        with pytest.raises(mc.MigrationRefused, match="nothing is bound"):
            await _migrate(tmp_path, Container(), Container())

    async def test_a_same_type_target_is_refused(self, tmp_path):
        _anchor(tmp_path)
        with pytest.raises(mc.MigrationRefused, match="dump/restore"):
            await _migrate(
                tmp_path, Container(), Container(), target_backend="PGKVStorage"
            )

    async def test_an_unadmitted_target_is_refused(self, tmp_path):
        _anchor(tmp_path)
        with pytest.raises(mc.MigrationRefused, match="not a configuration"):
            await _migrate(
                tmp_path, Container(), Container(), target_backend="RedisKVStorage"
            )

    @pytest.mark.parametrize(
        "rows", [{}, {IDENTITY_KEY: _identity_row(UUID_B)}], ids=["none", "other"]
    )
    async def test_a_source_that_is_not_the_anchored_container_is_refused(
        self, tmp_path, rows
    ):
        _anchor(tmp_path)
        target = Container()
        with pytest.raises(mc.MigrationRefused, match="never moved"):
            await _migrate(tmp_path, Container(rows), target)
        assert target.calls == []
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    async def test_an_unreadable_source_is_refused(self, tmp_path):
        _anchor(tmp_path)
        with pytest.raises(ConfigurationStorageError):
            await _migrate(
                tmp_path, Container(read_error=ConnectionError("down")), Container()
            )
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    async def test_a_malformed_source_row_is_refused_and_listed(self, tmp_path):
        _anchor(tmp_path)
        rows = _source_rows()
        rows["alpha/embedding/entities"] = {"value": "no schema"}
        rows["beta/embedding/chunks"] = "not even a row"
        target = Container()
        with pytest.raises(mc.MigrationRefused) as excinfo:
            await _migrate(tmp_path, Container(rows), target)
        message = str(excinfo.value)
        assert "'alpha/embedding/entities'" in message
        assert "2 row(s)" in message
        assert not _writes(target)

    async def test_a_target_with_another_identity_is_refused(self, tmp_path):
        _anchor(tmp_path)
        target = Container({IDENTITY_KEY: _identity_row(UUID_B), **_baseline("x")})
        with pytest.raises(mc.MigrationRefused, match="another identity"):
            await _migrate(tmp_path, Container(_source_rows()), target)
        assert not _writes(target)

    async def test_a_non_empty_target_without_an_identity_is_refused(self, tmp_path):
        _anchor(tmp_path)
        target = Container(_baseline("someone-else"))
        with pytest.raises(mc.MigrationRefused, match="no identity"):
            await _migrate(tmp_path, Container(_source_rows()), target)
        assert not _writes(target)
        assert "someone-else/embedding/entities" in target.visible

    async def test_a_running_starter_refuses_the_migration(self, tmp_path):
        pytest.importorskip("fcntl")
        _anchor(tmp_path)
        al.acquire_anchor_lock_shared(str(tmp_path))
        try:
            with pytest.raises(ConfigurationAnchorLockError):
                await _migrate(tmp_path, Container(_source_rows()), Container())
        finally:
            al.release_anchor_lock_shared(str(tmp_path))


class TestFailureAndResume:
    async def test_a_pagination_failure_leaves_the_anchor(self, tmp_path):
        _anchor(tmp_path)
        source = Container(_source_rows())
        source.iter_error_after = 3
        with pytest.raises(ConfigurationStorageError, match="enumerate"):
            await _migrate(tmp_path, source, Container())
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    async def test_a_copy_that_dies_midway_resumes_on_the_next_run(self, tmp_path):
        _anchor(tmp_path)
        source = Container(_source_rows())
        target = Container()
        target.fail_upsert_at = 2  # the identity lands, the first data batch dies
        with pytest.raises(mc.MigrationFailed):
            await _migrate(tmp_path, source, target, page_size=2)
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"
        assert target.visible[IDENTITY_KEY]["value"] == {"uuid": UUID_A}

        target.fail_upsert_at = None
        result = await _migrate(tmp_path, source, target, page_size=2)
        assert result.verdict.state == "resume"
        assert result.switched is True
        assert _data(target) == _data(source)

    async def test_a_flush_failure_leaves_the_anchor(self, tmp_path):
        _anchor(tmp_path)
        target = Container(flush_error=OSError("disk full"))
        with pytest.raises(ConfigurationStorageError):
            await _migrate(tmp_path, Container(_source_rows()), target)
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    async def test_a_verification_failure_leaves_the_anchor(self, tmp_path):
        _anchor(tmp_path)
        target = Container()
        # The server "lands" a different value than the one written.
        key = cs.embedding_baseline_key("alpha", "entities")
        target.corrupt_on_flush = _baseline("alpha", model="something-else")
        with pytest.raises(mc.MigrationFailed, match="does not match"):
            await _migrate(tmp_path, Container(_source_rows()), target)
        assert key in target.visible
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

    async def test_an_anchor_publish_failure_leaves_the_anchor_and_resumes(
        self, tmp_path, monkeypatch
    ):
        _anchor(tmp_path)
        source, target = Container(_source_rows()), Container()
        real = mc.publish_anchor

        def _fail(*args, **kwargs):
            raise ca.ConfigurationIdentityError(
                "injected", cause=ca.IDENTITY_ANCHOR_WRITE_FAILED
            )

        monkeypatch.setattr(mc, "publish_anchor", _fail)
        with pytest.raises(mc.MigrationFailed, match="could not be switched"):
            await _migrate(tmp_path, source, target)
        assert ca.read_anchor(str(tmp_path)).backend == "PGKVStorage"

        monkeypatch.setattr(mc, "publish_anchor", real)
        result = await _migrate(tmp_path, source, target)
        assert result.switched and result.verdict.state == "resume"

    async def test_a_directory_fsync_failure_after_the_replace_is_still_switched(
        self, tmp_path, monkeypatch
    ):
        _anchor(tmp_path)
        real = mc.publish_anchor

        def _landed_then_fsync_failed(working_dir, anchor, *, replace):
            real(working_dir, anchor, replace=replace)
            raise ca.ConfigurationIdentityError(
                "fsync failed", cause=ca.IDENTITY_ANCHOR_WRITE_FAILED
            )

        monkeypatch.setattr(mc, "publish_anchor", _landed_then_fsync_failed)
        result = await _migrate(tmp_path, Container(_source_rows()), Container())
        assert result.switched is True
        assert ca.read_anchor(str(tmp_path)).backend == "MongoKVStorage"

    async def test_a_source_that_changed_between_attempts_is_converged(self, tmp_path):
        _anchor(tmp_path)
        # Writes land as they are made, so the failed attempt leaves rows.
        source = Container(_source_rows())
        target = Container(visible_on_upsert=True, merge=True)
        target.fail_upsert_at = 3
        with pytest.raises(mc.MigrationFailed):
            await _migrate(tmp_path, source, target, page_size=2)
        target.fail_upsert_at = None

        # Between attempts: a row the target already copied is removed from
        # the source, one is changed, one is added -- and another copied row
        # carries a stale field from the first attempt.
        copied = [k for k in target.visible if k != IDENTITY_KEY]
        assert len(copied) == 2
        del source.visible[copied[0]]
        target.visible[copied[1]] = {**target.visible[copied[1]], "leftover": 1}
        source.visible.update(_baseline("alpha", "relationships", model="changed"))
        source.visible.update(_baseline("gamma"))

        result = await _migrate(tmp_path, source, target, page_size=2)
        assert result.switched is True
        assert _data(target) == _data(source)
        assert copied[0] not in target.visible
        assert result.deleted >= 2

    async def test_a_stale_copy_from_an_earlier_migration_is_reconciled(self, tmp_path):
        """PG -> Mongo, later Mongo -> OpenSearch, later OpenSearch -> Mongo:
        the old Mongo copy holds the same identity and is converged."""
        _anchor(tmp_path)
        old_copy = {IDENTITY_KEY: _identity_row(), **_baseline("alpha", model="old")}
        old_copy.update(_baseline("retired"))
        target = Container(old_copy)
        source = Container(_source_rows())
        result = await _migrate(tmp_path, source, target)
        assert result.verdict.state == "resume"
        assert _data(target) == _data(source)
        assert "retired/embedding/entities" not in target.visible


class TestEnvironments:
    def test_disjoint_connections_are_merged_per_side(self, tmp_path):
        overlay = mc.resolve_environments(
            source_backend="PGKVStorage",
            target_backend="MongoKVStorage",
            source_env={"POSTGRES_HOST": "old-db", "LLM_MODEL": "a"},
            target_env={"MONGO_URI": "mongodb://new", "LLM_MODEL": "b"},
            current={"WORKING_DIR": str(tmp_path)},
        )
        assert overlay == {"POSTGRES_HOST": "old-db", "MONGO_URI": "mongodb://new"}

    def test_a_conflicting_variable_either_backend_reads_is_refused(self, tmp_path):
        with pytest.raises(mc.MigrationRefused, match="POSTGRES_HOST"):
            mc.resolve_environments(
                source_backend="PGKVStorage",
                target_backend="MongoKVStorage",
                source_env={"POSTGRES_HOST": "a"},
                target_env={"POSTGRES_HOST": "b"},
                current={"WORKING_DIR": str(tmp_path)},
            )

    def test_an_irrelevant_difference_is_not_a_conflict(self, tmp_path):
        mc.resolve_environments(
            source_backend="PGKVStorage",
            target_backend="MongoKVStorage",
            source_env={"LLM_MODEL": "a"},
            target_env={"LLM_MODEL": "b"},
            current={"WORKING_DIR": str(tmp_path)},
        )

    @pytest.mark.parametrize("side", ["source_env", "target_env"])
    def test_another_working_dir_is_refused(self, tmp_path, side):
        envs = {"source_env": {}, "target_env": {}}
        envs[side] = {"WORKING_DIR": str(tmp_path / "other")}
        with pytest.raises(mc.MigrationRefused, match="WORKING_DIR"):
            mc.resolve_environments(
                source_backend="PGKVStorage",
                target_backend="MongoKVStorage",
                current={"WORKING_DIR": str(tmp_path)},
                **envs,
            )

    def test_the_same_working_dir_spelled_differently_passes(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path.parent)
        mc.resolve_environments(
            source_backend="PGKVStorage",
            target_backend="JsonKVStorage",
            source_env={"WORKING_DIR": tmp_path.name},
            target_env={},
            current={"WORKING_DIR": str(tmp_path)},
        )


class TestEndToEndOnJson:
    async def test_json_to_a_server_backend_and_the_next_start_passes(
        self, tmp_path, monkeypatch
    ):
        """A real JSON source bound by a real start; the target stands in for
        a server backend. After the switch, a start with the target explicitly
        selected is judged by the target's identity."""
        from lightrag.kg.json_kv_impl import JsonKVStorage

        source = cs.create_configuration_storage(
            JsonKVStorage,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=None,
        )
        await source.initialize()
        await cs.bind_configuration_identity(
            source,
            working_dir=str(tmp_path),
            backend="JsonKVStorage",
            container="json",
        )
        await source.upsert(_baseline("alpha"))
        await cs.flush_configuration_storage(source, "seed")
        storage_uuid = ca.read_anchor(str(tmp_path)).storage_uuid
        target = Container()

        async def _src(backend):
            assert backend == "JsonKVStorage"
            return source

        async def _tgt(backend):
            return target

        result = await mc.migrate_configuration(
            working_dir=str(tmp_path),
            target_backend="MongoKVStorage",
            open_source=_src,
            open_target=_tgt,
            out=lambda line: None,
        )
        assert result.switched
        assert os.path.exists(tmp_path / "_lightrag_config" / "kv_server_config.json")
        assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
            backend="MongoKVStorage", storage_uuid=storage_uuid
        )
        # A start that did NOT set the target explicitly is refused by type...
        with pytest.raises(ca.ConfigurationIdentityError) as excinfo:
            cs.preflight_configuration_anchor(
                working_dir=str(tmp_path), backend="JsonKVStorage", container="json"
            )
        assert excinfo.value.cause == ca.IDENTITY_BACKEND_MISMATCH
        # ...and one that did is verified against the migrated container.
        binding = await cs.bind_configuration_identity(
            target,
            working_dir=str(tmp_path),
            backend="MongoKVStorage",
            container="mongo",
        )
        assert binding.action == "verified"


class TestCommandLine:
    """``async_main``: the refusals and the switch as the operator sees them.
    The storages are the doubles above, behind the tool's own opener."""

    def _wire(self, monkeypatch, source, target):
        def _opener(working_dir, config_dir, claims):
            async def _open(backend):
                return source if backend == "PGKVStorage" else target

            return _open

        monkeypatch.setattr(mc, "_opener", _opener)

    async def test_no_anchor_is_refused_with_exit_1(self, capsys):
        assert await mc.async_main(["--target-backend", "MongoKVStorage"]) == 1
        assert "nothing is bound" in capsys.readouterr().out

    async def test_a_migration_prints_the_next_step_and_exits_0(
        self, monkeypatch, capsys
    ):
        working_dir = os.environ["WORKING_DIR"]
        _anchor(working_dir)
        target = Container()
        self._wire(monkeypatch, Container(_source_rows()), target)
        code = await mc.async_main(["--target-backend", "MongoKVStorage", "--yes"])
        out = capsys.readouterr().out
        assert code == 0
        assert "Switched" in out
        assert "LIGHTRAG_CONFIG_STORAGE=MongoKVStorage" in out
        assert "kept" in out
        assert ca.read_anchor(working_dir).backend == "MongoKVStorage"

    async def test_declining_the_prompt_writes_nothing(self, monkeypatch, capsys):
        working_dir = os.environ["WORKING_DIR"]
        _anchor(working_dir)
        target = Container()
        self._wire(monkeypatch, Container(_source_rows()), target)
        monkeypatch.setattr("builtins.input", lambda prompt="": "no")
        assert await mc.async_main(["--target-backend", "MongoKVStorage"]) == 0
        assert "nothing was written" in capsys.readouterr().out
        assert target.calls == []
        assert ca.read_anchor(working_dir).backend == "PGKVStorage"

    async def test_a_failure_says_the_anchor_is_unchanged(self, monkeypatch, capsys):
        working_dir = os.environ["WORKING_DIR"]
        _anchor(working_dir)
        target = Container(flush_error=OSError("disk full"))
        self._wire(monkeypatch, Container(_source_rows()), target)
        code = await mc.async_main(["--target-backend", "MongoKVStorage", "--yes"])
        assert code == 1
        assert "anchor is unchanged" in capsys.readouterr().out

    async def test_conflicting_env_files_are_refused(self, tmp_path, capsys):
        _anchor(os.environ["WORKING_DIR"])
        (tmp_path / "a.env").write_text("POSTGRES_HOST=a\n")
        (tmp_path / "b.env").write_text("POSTGRES_HOST=b\n")
        code = await mc.async_main(
            [
                "--target-backend",
                "MongoKVStorage",
                "--source-env",
                str(tmp_path / "a.env"),
                "--target-env",
                str(tmp_path / "b.env"),
                "--dry-run",
            ]
        )
        assert code == 1
        assert "POSTGRES_HOST" in capsys.readouterr().out


async def test_the_cli_opens_a_real_json_source_through_its_own_opener(
    monkeypatch, capsys
):
    """The tool's own opener on a real ``JsonKVStorage`` source (its
    ``config_dir`` claimed and handed back), with the server backend stood in
    for by a class behind the factory."""
    from lightrag.kg import factory
    from lightrag.kg.json_kv_impl import JsonKVStorage
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    working_dir = os.environ["WORKING_DIR"]
    source = cs.create_configuration_storage(
        JsonKVStorage, global_config={"working_dir": working_dir}, embedding_func=None
    )
    await source.initialize()
    await cs.bind_configuration_identity(
        source, working_dir=working_dir, backend="JsonKVStorage", container="json"
    )
    seeded = _baseline("alpha")
    await source.upsert(dict(seeded))
    await cs.flush_configuration_storage(source, "seed")
    await source.finalize()
    finalize_share_data()

    target = Container()

    class _TargetStorage:
        def __new__(cls, **kwargs):
            target.workspace = kwargs["workspace"]
            target.initialize = _noop
            return target

    async def _noop():
        return None

    real = factory.get_storage_class
    monkeypatch.setattr(
        factory,
        "get_storage_class",
        lambda name: _TargetStorage if name == "MongoKVStorage" else real(name),
    )
    code = await mc.async_main(["--target-backend", "MongoKVStorage", "--yes"])
    assert code == 0, capsys.readouterr().out
    assert _data(target) == {k: mc.row_payload(v) for k, v in seeded.items()}
    assert ca.read_anchor(working_dir).backend == "MongoKVStorage"
    assert not holds_working_dir_lock(cs.resolve_config_dir("", working_dir))
    initialize_share_data(workers=1)
