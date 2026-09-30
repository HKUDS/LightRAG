"""Exact SDK workspace selection must not redirect a rebuild to a sibling."""

import json

import numpy as np
import pytest

from lightrag import config_store as cs
from lightrag.config_shards import json_config_dir, json_config_path
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data
from lightrag.kg.working_dir_lock import (
    holds_working_dir_lock,
    release_working_dir_lock,
)
from lightrag.tools import rebuild_vdb as rb
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "argv, expected",
    [([], None), (["--workspace", "team-a"], "team-a"), (["--workspace", ""], "")],
)
def test_cli_passes_exact_workspace(monkeypatch, argv, expected):
    selected = []

    async def run(self):
        selected.append(self._workspace_override)
        return True

    monkeypatch.setattr(rb.RebuildTool, "run", run)
    monkeypatch.setattr(rb, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(rb, "setup_logger", lambda *args, **kwargs: None)
    rb.main(argv)
    assert selected == [expected]


@pytest.mark.parametrize("workspace", ["../other", "config_storage_anchor.json"])
def test_cli_refuses_invalid_workspace_before_run(monkeypatch, workspace):
    def unexpected(**kwargs):
        pytest.fail("invalid workspace must fail before loading the environment")

    monkeypatch.setattr(rb, "load_dotenv", unexpected)
    with pytest.raises(SystemExit) as error:
        rb.main(["--workspace", workspace])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "override, expected", [(None, "team_a"), ("team-a", "team-a"), ("", "")]
)
async def test_rebuild_selects_one_workspace_and_preserves_siblings(
    monkeypatch, tmp_path, override, expected
):
    """Real JSON/NetworkX/Nano stores: read, rebuild and baseline share one scope."""
    for key, value in {
        "WORKING_DIR": str(tmp_path),
        "WORKSPACE": "team-a",
        "LIGHTRAG_CONFIG_STORAGE": "JsonKVStorage",
        "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
        "LIGHTRAG_GRAPH_STORAGE": "NetworkXStorage",
        "LIGHTRAG_VECTOR_STORAGE": "NanoVectorDBStorage",
    }.items():
        monkeypatch.setenv(key, value)

    async def embed(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)

    embedding = EmbeddingFunc(embedding_dim=8, func=embed, model_name="test-model")
    monkeypatch.setattr(rb.RebuildTool, "build_embedding_func", lambda self: embedding)
    initialize_share_data(workers=1)
    tool = rb.RebuildTool(workspace=override)
    try:
        for workspace in ("team-a", "team_a", ""):
            config = cs.create_configuration_storage(
                JsonKVStorage,
                global_config={"working_dir": str(tmp_path), "workspace": workspace},
                embedding_func=None,
            )
            await config.initialize()
            try:
                await cs.bind_configuration_identity(
                    config,
                    working_dir=str(tmp_path),
                    backend="JsonKVStorage",
                    container=json_config_path(str(tmp_path), workspace),
                    workspace=workspace,
                )
                await cs.record_embedding_baseline(
                    config,
                    workspace=workspace,
                    target="chunks",
                    embedding_func=EmbeddingFunc(
                        embedding_dim=8, func=embed, model_name="old-model"
                    ),
                )
            finally:
                await config.finalize()

        for sibling in {"team-a", "team_a", ""} - {expected}:
            vector = NanoVectorDBStorage(
                namespace="chunks",
                workspace=sibling,
                global_config={
                    "working_dir": str(tmp_path),
                    "embedding_batch_num": 10,
                    "vector_db_storage_cls_kwargs": {
                        "cosine_better_than_threshold": 0.2
                    },
                },
                embedding_func=embedding,
                meta_fields={"content"},
            )
            await vector.initialize()
            try:
                await vector.upsert({"sibling-chunk": {"content": "must survive"}})
                await vector.index_done_callback()
            finally:
                await vector.finalize()

        before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
        assert await tool.setup_storages() is True
        assert tool.workspace == expected
        assert tool.global_config["workspace"] == expected
        assert tool.config_dir == json_config_dir(str(tmp_path), expected)
        assert holds_working_dir_lock(tool.config_dir)
        assert all(
            storage.workspace == expected
            for storage in (
                tool.graph,
                tool.text_chunks,
                tool.entities_vdb,
                tool.relationships_vdb,
                tool.chunks_vdb,
            )
        )
        await tool.text_chunks.upsert(
            {"chunk-test": {"content": "selected workspace", "full_doc_id": "doc-test"}}
        )
        stats = await tool.run_rebuild_chunks()
        assert not stats[0]["errors"]
        assert stats[0]["prepared"] == 1
        rows = json.loads(
            (tmp_path / expected / "kv_workspace_config.json").read_text()
        )
        assert (
            rows[cs.embedding_baseline_key(expected, "chunks")]["value"]["model"]
            == "test-model"
        )
        assert (tmp_path / expected / "vdb_chunks.json").is_file()
        for path, content in before.items():
            if str(path) != json_config_path(str(tmp_path), expected):
                assert path.read_bytes() == content
        for sibling in {"team-a", "team_a", ""} - {expected}:
            assert (tmp_path / sibling / "vdb_chunks.json").is_file()
    finally:
        for storage in tool.all_storages():
            if storage is not None:
                await storage.finalize()
        if tool._holds_working_dir:
            release_working_dir_lock(tool.config_dir)
        tool.release_anchor_lock()
        finalize_share_data()
