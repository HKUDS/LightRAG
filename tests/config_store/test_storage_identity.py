"""The configuration container's identity row and the bind that pairs it with
the anchor, driven with doubles.

Anchored: backend type and UUID must both match, and nothing is ever
created. Not anchored: the container's identity is adopted, or created
through the strict flush and read-back, and only THEN is the anchor
published -- so every crash window heals by adoption on the next start. See
*The anchor and the container identity* in docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from lightrag import config_anchor as ca
from lightrag import config_store as cs
from lightrag.exceptions import (
    ConfigurationIdentityError,
    ConfigurationRecordMalformedError,
    ConfigurationStorageError,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import SERVER_CONFIG_SCOPE, SERVER_SCOPE
from tests.config_store.test_config_store import (
    FakeConfigKV,
    _RetainingKV,
    _row,
    _key,
)

pytestmark = pytest.mark.offline

UUID_A = "3f2b8c1e-6a4d-4e2f-9b7a-1c2d3e4f5a6b"
UUID_B = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
IDENTITY_KEY = "_lightrag_server/storage_identity"
CONTAINER = "PGKVStorage (_lightrag_config)"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def _identity_row(storage_uuid=UUID_A, **overrides):
    row = cs.make_config_row(
        scope_workspace=SERVER_SCOPE,
        suffix=cs.STORAGE_IDENTITY_SUFFIX,
        value={"uuid": storage_uuid},
        updated_by="test",
    )
    row.update(overrides)
    return row


def _anchor(tmp_path, backend="PGKVStorage", storage_uuid=UUID_A):
    ca.publish_anchor(
        str(tmp_path),
        ca.StorageAnchor(backend=backend, storage_uuid=storage_uuid),
        replace=False,
    )
    return open(ca.anchor_path(str(tmp_path)), "rb").read()


async def _bind(config, tmp_path, backend="PGKVStorage"):
    return await cs.bind_configuration_identity(
        config, working_dir=str(tmp_path), backend=backend, container=CONTAINER
    )


def _writes(config):
    return [c for c in config.calls if c[0] in ("upsert", "flush", "delete")]


class TestTheRow:
    def test_the_key_is_server_scoped(self):
        assert cs.storage_identity_key() == IDENTITY_KEY
        spec = cs.CONFIG_KEY_REGISTRY[cs.STORAGE_IDENTITY_SUFFIX]
        assert spec.scope is cs.ConfigScope.SERVER

    def test_a_tenant_named_like_the_server_scope_cannot_reach_it(self):
        """The scope is compared by identity: a workspace legally named
        ``_lightrag_server`` is refused the server-global suffix, and its own
        keys never equal the identity key."""
        with pytest.raises(ValueError, match="server-global"):
            cs.config_key(SERVER_CONFIG_SCOPE, cs.STORAGE_IDENTITY_SUFFIX)
        tenant_keys = {
            cs.embedding_baseline_key(SERVER_CONFIG_SCOPE, t)
            for t in cs.EMBEDDING_TARGETS
        }
        assert IDENTITY_KEY not in tenant_keys

    async def test_a_confirmed_absence_reads_as_none(self):
        assert await cs.read_storage_identity(FakeConfigKV()) is None

    async def test_a_valid_row_reads_its_uuid(self):
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        assert await cs.read_storage_identity(config) == UUID_A

    @pytest.mark.parametrize(
        "row",
        [
            _identity_row(schema_version=2),
            _identity_row(schema_version="1"),
            _identity_row(schema_version=True),
            _identity_row(workspace="tenant"),
            _identity_row(value={"uuid": "not-a-uuid"}),
            _identity_row(value={"uuid": UUID_A.upper()}),
            _identity_row(value={}),
            _identity_row(value="uuid"),
        ],
    )
    async def test_an_invalid_row_is_never_absent(self, row):
        config = FakeConfigKV({IDENTITY_KEY: row})
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await cs.read_storage_identity(config)
        assert excinfo.value.cause == ca.IDENTITY_ROW_INVALID

    async def test_a_row_that_is_not_a_mapping_is_malformed(self):
        class _RawKV(FakeConfigKV):
            async def get_by_id_strict(self, key):
                return self.visible.get(key)

        config = _RawKV({IDENTITY_KEY: "uuid"})
        with pytest.raises(ConfigurationRecordMalformedError):
            await cs.read_storage_identity(config)

    async def test_a_transport_failure_is_a_failure(self):
        config = FakeConfigKV(read_error=ConnectionError("down"))
        with pytest.raises(ConfigurationStorageError):
            await cs.read_storage_identity(config)

    async def test_a_workspace_drop_never_touches_the_identity(self):
        """``delete_workspace_configuration`` deletes registered per-workspace
        suffixes by key; the identity is the whole container's."""
        config = FakeConfigKV(
            {IDENTITY_KEY: _identity_row(), _key("entities"): _row()},
            visible_on_upsert=True,
        )
        await cs.delete_workspace_configuration(config, "ws")
        assert await cs.read_storage_identity(config) == UUID_A
        deleted = [k for c in config.calls if c[0] == "delete" for k in c[1]]
        assert IDENTITY_KEY not in deleted


class TestAnchoredStarts:
    async def test_matching_type_and_uuid_verifies_and_writes_nothing(self, tmp_path):
        before = _anchor(tmp_path)
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        binding = await _bind(config, tmp_path)
        assert binding == cs.IdentityBinding(storage_uuid=UUID_A, action="verified")
        assert _writes(config) == []
        assert open(ca.anchor_path(str(tmp_path)), "rb").read() == before

    async def test_a_missing_identity_refuses_and_creates_nothing(self, tmp_path):
        before = _anchor(tmp_path)
        config = FakeConfigKV()
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_UUID_MISSING
        message = str(excinfo.value)
        # Primary recovery for this cause: delete the anchor, by path.
        assert ca.anchor_path(str(tmp_path)) in message
        assert "delete" in message and UUID_A in message
        assert _writes(config) == []
        assert open(ca.anchor_path(str(tmp_path)), "rb").read() == before

    async def test_a_different_uuid_refuses(self, tmp_path):
        _anchor(tmp_path)
        config = FakeConfigKV({IDENTITY_KEY: _identity_row(UUID_B)})
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_UUID_MISMATCH
        message = str(excinfo.value)
        assert UUID_A in message and UUID_B in message
        assert "connection settings" in message
        # Deleting the anchor is listed LAST and carries the warning.
        assert message.index("connection settings") < message.index("Last resort")
        assert "ABANDONS" in message
        assert _writes(config) == []

    async def test_another_backend_type_refuses_even_with_the_same_uuid(self, tmp_path):
        """The copy-the-row bypass: a same-UUID container of another type is
        still another container."""
        _anchor(tmp_path, backend="PGKVStorage")
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path, backend="MongoKVStorage")
        assert excinfo.value.cause == ca.IDENTITY_BACKEND_MISMATCH
        message = str(excinfo.value)
        assert "LIGHTRAG_CONFIG_STORAGE=PGKVStorage" in message
        assert message.index("LIGHTRAG_CONFIG_STORAGE=") < message.index("Last resort")
        assert config.calls == []

    async def test_an_invalid_identity_refuses(self, tmp_path):
        _anchor(tmp_path)
        config = FakeConfigKV({IDENTITY_KEY: _identity_row(value={"uuid": "x"})})
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_ROW_INVALID
        assert _writes(config) == []

    async def test_a_read_error_refuses(self, tmp_path):
        _anchor(tmp_path)
        config = FakeConfigKV(read_error=ConnectionError("down"))
        with pytest.raises(ConfigurationStorageError):
            await _bind(config, tmp_path)
        assert _writes(config) == []

    async def test_an_unreadable_anchor_refuses_before_the_store_is_asked(
        self, tmp_path
    ):
        path = ca.anchor_path(str(tmp_path))
        os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            f.write("{truncated")
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_UNREADABLE
        assert config.calls == []


class TestUnanchoredStarts:
    async def test_an_empty_container_gets_a_new_identity_then_the_anchor(
        self, tmp_path, monkeypatch
    ):
        config = FakeConfigKV()
        warnings: list[str] = []
        monkeypatch.setattr(cs.logger, "warning", warnings.append)
        binding = await _bind(config, tmp_path)
        assert binding.action == "created"
        assert ca.canonical_storage_uuid(binding.storage_uuid) == binding.storage_uuid
        # Written, flushed, read back -- and the anchor names the same UUID.
        kinds = [c[0] for c in config.calls]
        assert kinds.index("upsert") < kinds.index("flush")
        assert kinds[-1] == "read"
        assert config.visible[IDENTITY_KEY]["value"] == {"uuid": binding.storage_uuid}
        assert ca.read_anchor(str(tmp_path)) == ca.StorageAnchor(
            backend="PGKVStorage", storage_uuid=binding.storage_uuid
        )
        # The rebind WARNING names the container and the UUID.
        assert any(
            CONTAINER in w and binding.storage_uuid in w and "created" in w
            for w in warnings
        )

    async def test_an_existing_identity_is_adopted_without_writing(
        self, tmp_path, monkeypatch
    ):
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        warnings: list[str] = []
        monkeypatch.setattr(cs.logger, "warning", warnings.append)
        binding = await _bind(config, tmp_path)
        assert binding == cs.IdentityBinding(storage_uuid=UUID_A, action="adopted")
        assert _writes(config) == []
        assert ca.read_anchor(str(tmp_path)).storage_uuid == UUID_A
        assert any(UUID_A in w and "adopted" in w for w in warnings)

    async def test_a_transport_failure_creates_neither_uuid_nor_anchor(self, tmp_path):
        config = FakeConfigKV(read_error=ConnectionError("down"))
        with pytest.raises(ConfigurationStorageError):
            await _bind(config, tmp_path)
        assert _writes(config) == []
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_write_failure_publishes_no_anchor(self, tmp_path):
        config = FakeConfigKV(write_error=OSError("disk full"))
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_WRITE_FAILED
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_flush_failure_publishes_no_anchor(self, tmp_path):
        config = FakeConfigKV(flush_error=OSError("disk full"))
        with pytest.raises(ConfigurationStorageError):
            await _bind(config, tmp_path)
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_write_still_retained_in_the_buffer_is_not_durable(self, tmp_path):
        """The OpenSearch shape: the strict read would answer from the
        process-local buffer, so the flush's retained-buffer check is what
        refuses."""
        config = _RetainingKV()
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await _bind(config, tmp_path)
        assert config.dropped == 1
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_read_back_that_disagrees_publishes_no_anchor(self, tmp_path):
        class _OtherWriter(FakeConfigKV):
            async def index_done_callback(self):
                await super().index_done_callback()
                self.visible[IDENTITY_KEY] = _identity_row(UUID_B)

        config = _OtherWriter()
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_WRITE_FAILED
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_failed_anchor_publish_heals_by_adoption(
        self, tmp_path, monkeypatch
    ):
        """Crash window 1: identity durable, anchor not published. The next
        start finds no anchor and a present identity, and adopts it."""
        config = FakeConfigKV()
        real_publish = cs.publish_anchor

        def _crash(*args, **kwargs):
            raise ConfigurationIdentityError(
                "injected", cause=ca.IDENTITY_ANCHOR_WRITE_FAILED
            )

        monkeypatch.setattr(cs, "publish_anchor", _crash)
        with pytest.raises(ConfigurationIdentityError):
            await _bind(config, tmp_path)
        created = config.visible[IDENTITY_KEY]["value"]["uuid"]
        assert ca.read_anchor(str(tmp_path)) is None

        monkeypatch.setattr(cs, "publish_anchor", real_publish)
        binding = await _bind(config, tmp_path)
        assert binding == cs.IdentityBinding(storage_uuid=created, action="adopted")
        assert ca.read_anchor(str(tmp_path)).storage_uuid == created

    async def test_concurrent_binds_in_one_tree_produce_exactly_one_identity(
        self, tmp_path, monkeypatch
    ):
        """Workers of one Gunicorn master: the keyed lock spans the whole
        read-decide-write, so the second waits and then verifies."""

        class _SlowKV(FakeConfigKV):
            async def get_by_id_strict(self, key):
                await asyncio.sleep(0.01)
                return await super().get_by_id_strict(key)

        config = _SlowKV()
        publishes = []
        real_publish = cs.publish_anchor

        def _counting(*args, **kwargs):
            publishes.append(args)
            return real_publish(*args, **kwargs)

        monkeypatch.setattr(cs, "publish_anchor", _counting)
        results = await asyncio.gather(*(_bind(config, tmp_path) for _ in range(4)))
        assert len({r.storage_uuid for r in results}) == 1
        assert sorted(r.action for r in results) == [
            "created",
            "verified",
            "verified",
            "verified",
        ]
        assert [c[0] for c in config.calls].count("upsert") == 1
        assert len(publishes) == 1

    async def test_an_anchor_that_appears_mid_bind_is_never_overwritten(
        self, tmp_path, monkeypatch
    ):
        """Separate process trees binding at once are unsupported; what IS
        promised is that the loser never clobbers the winner's anchor."""
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        real_read = cs.read_anchor

        def _racing(working_dir):
            result = real_read(working_dir)
            ca.publish_anchor(
                working_dir,
                ca.StorageAnchor(backend="PGKVStorage", storage_uuid=UUID_B),
                replace=False,
            )
            return result

        monkeypatch.setattr(cs, "read_anchor", _racing)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await _bind(config, tmp_path)
        assert excinfo.value.cause == ca.IDENTITY_ANCHOR_APPEARED
        assert real_read(str(tmp_path)).storage_uuid == UUID_B


class TestToolVerification:
    async def test_no_anchor_verifies_nothing_and_writes_nothing(self, tmp_path):
        config = FakeConfigKV()
        binding = await cs.verify_configuration_identity(
            config,
            None,
            working_dir=str(tmp_path),
            backend="PGKVStorage",
            container=CONTAINER,
        )
        assert binding.action == "unanchored"
        assert config.calls == []
        assert ca.read_anchor(str(tmp_path)) is None

    async def test_a_match_verifies_without_writing(self, tmp_path):
        _anchor(tmp_path)
        config = FakeConfigKV({IDENTITY_KEY: _identity_row()})
        binding = await cs.verify_configuration_identity(
            config,
            ca.read_anchor(str(tmp_path)),
            working_dir=str(tmp_path),
            backend="PGKVStorage",
            container=CONTAINER,
        )
        assert binding == cs.IdentityBinding(storage_uuid=UUID_A, action="verified")
        assert _writes(config) == []

    @pytest.mark.parametrize(
        "rows, cause",
        [
            ({}, ca.IDENTITY_UUID_MISSING),
            ({IDENTITY_KEY: _identity_row(UUID_B)}, ca.IDENTITY_UUID_MISMATCH),
        ],
    )
    async def test_a_mismatch_refuses_exactly_as_a_start_would(
        self, tmp_path, rows, cause
    ):
        _anchor(tmp_path)
        config = FakeConfigKV(rows)
        with pytest.raises(ConfigurationIdentityError) as excinfo:
            await cs.verify_configuration_identity(
                config,
                ca.read_anchor(str(tmp_path)),
                working_dir=str(tmp_path),
                backend="PGKVStorage",
                container=CONTAINER,
            )
        assert excinfo.value.cause == cause
        assert _writes(config) == []


def test_the_anchor_holds_no_connection_detail(tmp_path):
    _anchor(tmp_path)
    with open(ca.anchor_path(str(tmp_path)), encoding="utf-8") as f:
        assert set(json.load(f)) == {"schema_version", "backend", "storage_uuid"}
