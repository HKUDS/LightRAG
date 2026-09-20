"""The configuration storage as its own CATEGORY: selection, ``config_dir``,
the fixed container names, and the upgrade from slice 1's layout.

Slice 1 kept configuration in a reserved workspace ``_lightrag_config`` on
whichever backend the business KV storage used. It is now selected
independently and lives in a container named in code. The three things that
carry real risk are pinned here:

* the ``config_dir`` default resolves to the file slice 1 already wrote, so an
  upgrade reads every baseline instead of reading absence -- and absence is the
  one answer that lets a start bootstrap;
* a fixed container name is SHARED by two deployments on one server, so what
  keeps their baselines apart is the row key's scope;
* a separately selected backend is a new way to point a running deployment at
  an empty store, which is announced rather than enforced.

See docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from lightrag import LightRAG, config_store as cs
from lightrag.exceptions import EmbeddingBaselineMismatchError
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_CONTAINER_TAG, default_config_dir
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_DIM = 8


class _SimpleTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _mock_llm(prompt, **kwargs):  # pragma: no cover - never called
    return "mock"


async def _embed(texts, **kwargs):
    return np.zeros((len(texts), _DIM), dtype=np.float32)


def _workspace(tmp_path) -> str:
    return f"cat-{tmp_path.name}"


def _rag(tmp_path, *, model_name="bge-m3", **kwargs):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=_workspace(tmp_path),
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=4096,
            func=_embed,
            model_name=model_name,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
        **kwargs,
    )


def _write_slice_one_layout(tmp_path, workspace, *, model_name, dim=_DIM):
    """The file exactly as slice 1 wrote it: the reserved workspace's
    subdirectory under ``working_dir``, one row per target.

    Written by hand on purpose -- driving it through today's code would prove
    only that the code agrees with itself.
    """
    path = tmp_path / CONFIG_CONTAINER_TAG / "kv_store_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {
        cs.embedding_baseline_key(workspace, target): {
            "schema_version": 1,
            "workspace": workspace,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "updated_by": cs.UPDATED_BY_STARTUP,
            "value": {"model": model_name, "dim": dim, "origin": "probe"},
            "_id": cs.embedding_baseline_key(workspace, target),
            "create_time": 1,
            "update_time": 1,
        }
        for target in cs.EMBEDDING_TARGETS
    }
    path.write_text(json.dumps(rows))
    return path


def _stored(path):
    return json.loads(path.read_text())


class TestTheUpgradeFromSliceOne:
    """Deliverable 2 and 7: the default does not orphan a single baseline."""

    def test_the_default_config_dir_is_slice_ones_location(self, tmp_path):
        assert default_config_dir(str(tmp_path)) == str(tmp_path / CONFIG_CONTAINER_TAG)

    async def test_every_recorded_baseline_is_still_read_after_the_upgrade(
        self, tmp_path
    ):
        """Start on the old layout, upgrade, and the recorded model is
        unchanged -- not re-established, not rewritten."""
        workspace = _workspace(tmp_path)
        path = _write_slice_one_layout(tmp_path, workspace, model_name="bge-m3")
        before = _stored(path)

        rag = _rag(tmp_path, model_name="bge-m3")
        await rag.initialize_storages()
        try:
            assert rag.config_storage == "JsonKVStorage"
            assert rag.config_dir == str(tmp_path / CONFIG_CONTAINER_TAG)
            recorded = await cs.read_embedding_baselines(
                rag.configuration_storage, workspace
            )
            assert {t: b.model for t, b in recorded.items()} == {
                t: "bge-m3" for t in cs.EMBEDDING_TARGETS
            }
            # ``origin=probe`` is what the old file said. A re-established
            # baseline would say ``empty`` here, which is the silent-bootstrap
            # symptom this test exists to catch.
            assert {b.origin for b in recorded.values()} == {"probe"}
        finally:
            await rag.finalize_storages()
        assert _stored(path) == before

    async def test_a_mismatching_model_still_refuses_after_the_upgrade(self, tmp_path):
        _write_slice_one_layout(tmp_path, _workspace(tmp_path), model_name="bge-m3")

        rag = _rag(tmp_path, model_name="text-embedding-3-large")
        with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
            await rag.initialize_storages()
        assert "bge-m3" in str(excinfo.value)
        await rag.finalize_storages()

    async def test_a_config_dir_pointed_elsewhere_reads_absence(self, tmp_path):
        """The counterexample that makes the default load bearing: the very
        same deployment, one setting different, and every baseline is gone."""
        workspace = _workspace(tmp_path)
        _write_slice_one_layout(tmp_path, workspace, model_name="bge-m3")

        rag = _rag(tmp_path, config_dir=str(tmp_path / "elsewhere"))
        await rag.initialize_storages()
        try:
            recorded = await cs.read_embedding_baselines(
                rag.configuration_storage, workspace
            )
            # Re-established from an empty deployment, not read.
            assert {b.origin for b in recorded.values()} == {"empty"}
        finally:
            await rag.finalize_storages()


class TestTheSelectionIsRefusedByName:
    """Deliverable 1. A backend outside the four never reaches a missing
    method: it is named at construction."""

    def test_a_vector_storage_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="NanoVectorDBStorage"):
            _rag(tmp_path, config_storage="NanoVectorDBStorage")

    def test_redis_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="RedisKVStorage"):
            _rag(tmp_path, config_storage="RedisKVStorage")

    def test_an_unknown_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="NoSuchStorage"):
            _rag(tmp_path, config_storage="NoSuchStorage")

    def test_a_redis_business_deployment_must_choose_one(self, tmp_path):
        """Unset, the selection follows ``kv_storage`` -- and Redis is not in
        the category, so the operator is told so rather than silently landing
        somewhere else."""
        with pytest.raises(ValueError, match="LIGHTRAG_CONFIG_STORAGE"):
            _rag(tmp_path, kv_storage="RedisKVStorage")

    def test_an_admitted_selection_is_independent_of_the_business_backend(
        self, tmp_path
    ):
        rag = _rag(tmp_path, config_storage="JsonKVStorage")
        assert rag.config_storage == "JsonKVStorage"
        assert type(rag.configuration_storage) is JsonKVStorage


class TestTwoDeploymentsShareTheContainer:
    """Deliverable 3. A fixed container name is the same name for everyone, so
    what keeps two deployments apart is the row KEY -- whose scope is the
    business workspace, which the contract already requires to differ.
    """

    def test_the_keys_of_two_workspaces_are_disjoint(self):
        a = {cs.embedding_baseline_key("alpha", t) for t in cs.EMBEDDING_TARGETS}
        b = {cs.embedding_baseline_key("beta", t) for t in cs.EMBEDDING_TARGETS}
        assert a & b == set()

    def test_json_two_deployments_one_file_disjoint_rows(self, tmp_path):
        shared = tmp_path / "shared"
        one = cs.create_configuration_storage(
            JsonKVStorage,
            global_config={"working_dir": str(tmp_path), "config_dir": str(shared)},
            embedding_func=None,
        )
        two = cs.create_configuration_storage(
            JsonKVStorage,
            global_config={
                "working_dir": str(tmp_path / "b"),
                "config_dir": str(shared),
            },
            embedding_func=None,
        )
        assert one._file_name == two._file_name

    def test_mongodb_one_collection_whatever_the_caller_named(self):
        from lightrag.kg.mongo_impl import MongoKVStorage

        names = {
            MongoKVStorage(
                namespace="config",
                global_config={},
                embedding_func=None,
                workspace=ws,
            )._collection_name
            for ws in ("alpha", "beta", "")
        }
        assert names == {f"{CONFIG_CONTAINER_TAG}_config"}

    def test_opensearch_one_index_whatever_the_caller_named(self):
        from lightrag.kg.opensearch_impl import _build_index_name

        names = {_build_index_name(ws, "config")[2] for ws in ("alpha", "beta", "")}
        assert len(names) == 1

    async def test_postgresql_one_partition_whatever_the_caller_named(self):
        from types import SimpleNamespace

        from lightrag.kg.postgres_impl import PGKVStorage

        for ws in ("alpha", "beta", ""):
            storage = PGKVStorage.__new__(PGKVStorage)
            storage.namespace = "config"
            storage.global_config = {}
            storage.db = SimpleNamespace(workspace="prod")
            storage.workspace = ws
            storage.__post_init__()
            await storage.initialize()
            assert storage.workspace == CONFIG_CONTAINER_TAG


class TestTheEmptyStoreGuard:
    """Deliverable 6. Announce, do not enforce -- the same posture as
    ``warn_about_workspace_overrides()``, for the same reason: an empty store
    and a genuine first start are indistinguishable from here."""

    @pytest.fixture
    def warnings_seen(self):
        import lightrag.utils as _utils

        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Collect(level=logging.WARNING)
        _utils.logger.addHandler(handler)
        try:
            yield records
        finally:
            _utils.logger.removeHandler(handler)

    def test_nothing_recorded_is_announced_with_the_container(self, warnings_seen):
        fired = cs.warn_about_unrecorded_baselines(
            list(cs.EMBEDDING_TARGETS), workspace="ws", container="JsonKVStorage at /x"
        )
        assert fired is True
        assert len(warnings_seen) == 1
        message = warnings_seen[0]
        assert "JsonKVStorage at /x" in message
        assert "config_storage / config_dir" in message

    def test_a_partial_absence_says_nothing(self, warnings_seen):
        """Two of three recorded is an interrupted first start, not a store
        pointed somewhere else; the per-target logging already covers it."""
        assert (
            cs.warn_about_unrecorded_baselines(
                ["entities"], workspace="ws", container="c"
            )
            is False
        )
        assert cs.warn_about_unrecorded_baselines(
            [], workspace="ws", container="c"
        ) is (False)
        assert warnings_seen == []

    async def test_a_started_deployment_with_records_stays_quiet(
        self, tmp_path, warnings_seen
    ):
        _write_slice_one_layout(tmp_path, _workspace(tmp_path), model_name="bge-m3")
        rag = _rag(tmp_path, model_name="bge-m3")
        await rag.initialize_storages()
        try:
            assert not [m for m in warnings_seen if "No embedding baseline" in m]
        finally:
            await rag.finalize_storages()

    async def test_a_store_pointed_away_from_the_records_is_announced(
        self, tmp_path, warnings_seen
    ):
        """The failure this guard exists for: the deployment HAS records, and
        one setting points the instance at a store that does not hold them."""
        _write_slice_one_layout(tmp_path, _workspace(tmp_path), model_name="bge-m3")
        rag = _rag(tmp_path, config_dir=str(tmp_path / "empty-store"))
        await rag.initialize_storages()
        try:
            assert [m for m in warnings_seen if "No embedding baseline" in m]
        finally:
            await rag.finalize_storages()


class TestTheClaimFollowsTheConfigurationStorage:
    """Deliverable: the single-server claim is taken in exactly the cases
    where the configuration storage is file-backed, and on the directory that
    storage actually writes to."""

    def test_only_the_file_backed_member_of_the_category_claims(self):
        from lightrag.kg.working_dir_lock import uses_working_dir

        assert uses_working_dir("JsonKVStorage") is True
        for server_backed in ("PGKVStorage", "MongoKVStorage", "OpenSearchKVStorage"):
            assert uses_working_dir(server_backed) is False

    async def test_the_claim_is_on_config_dir_not_working_dir(self, tmp_path):
        from lightrag.kg.working_dir_lock import holds_working_dir_lock

        elsewhere = tmp_path / "conf"
        rag = _rag(tmp_path, config_dir=str(elsewhere))
        await rag.initialize_storages()
        try:
            assert holds_working_dir_lock(str(elsewhere)) is True
            assert holds_working_dir_lock(str(tmp_path)) is False
        finally:
            await rag.finalize_storages()
        assert holds_working_dir_lock(str(elsewhere)) is False


@pytest.mark.asyncio
async def test_a_tenant_named_like_the_server_scope_gets_its_own_baselines(tmp_path):
    """Retiring the reserved family made ``_lightrag_server`` a legal tenant
    name -- and the server SCOPE was a string with that spelling, so keying a
    baseline for that tenant raised and the startup died on a legal name.

    The scope is an object now, so a name cannot be one. The rows still
    render under the same prefix, which is harmless: a suffix belongs to
    exactly one scope, so no tenant key can be a server-global key.
    """
    from lightrag.namespace import SERVER_CONFIG_SCOPE, SERVER_SCOPE

    tenant = SERVER_CONFIG_SCOPE

    for target in cs.EMBEDDING_TARGETS:
        assert cs.embedding_baseline_key(tenant, target) == (
            f"{tenant}/embedding/{target}"
        )

    # The sentinel is refused for a per-workspace suffix; the NAME is not.
    with pytest.raises(ValueError, match="not a workspace"):
        cs.config_key(SERVER_SCOPE, cs.embedding_baseline_suffix("entities"))

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=tenant,
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM, max_token_size=4096, func=_embed, model_name="bge-m3"
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )
    await rag.initialize_storages()
    try:
        recorded = await cs.read_embedding_baselines(
            rag.configuration_storage, workspace=tenant
        )
        assert {t for t, row in recorded.items() if row} == set(cs.EMBEDDING_TARGETS)
    finally:
        await rag.finalize_storages()
