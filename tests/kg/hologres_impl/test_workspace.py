import pytest

from lightrag.kg.hologres.doc_status import HologresDocStatusStorage
from lightrag.kg.hologres.graph import HologresGraphStorage
from lightrag.kg.hologres.graph_age import HologresAGEGraphStorage
from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.namespace import NameSpace
from lightrag.kg.hologres.vector import HologresVectorStorage
from lightrag.kg.hologres.workspace import WORKSPACE_ENV_VAR


CONFIG_KWARGS = {
    "host": "secret-host.example",
    "port": 80,
    "user": "secret-user",
    "password": "secret-password",
    "database": "secret-database",
    "schema": "lightrag_test_workspace",
    "connection_retries": 0,
}


class FakeEmbedding:
    embedding_dim = 3

    async def __call__(self, texts, **kwargs):
        return [[1.0, 0.0, 0.0] for _ in texts]


def _all_storages(workspace="instance"):
    graph_namespace = NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
    return [
        HologresKVStorage(
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=workspace,
            global_config={},
            embedding_func=FakeEmbedding(),
            config=None,
            client=object(),
        ),
        HologresVectorStorage(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=workspace,
            global_config={
                "embedding_batch_num": 1,
                "vector_db_storage_cls_kwargs": {
                    "cosine_better_than_threshold": 0.2
                },
            },
            embedding_func=FakeEmbedding(),
            config=None,
            client=object(),
        ),
        HologresDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=None,
            client=object(),
        ),
        HologresGraphStorage(
            namespace=graph_namespace,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=None,
            client=object(),
        ),
        HologresAGEGraphStorage(
            namespace=graph_namespace,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=None,
            client=object(),
        ),
    ]


def test_hologres_workspace_env_overrides_every_backend(monkeypatch):
    monkeypatch.setenv(WORKSPACE_ENV_VAR, "forced_workspace")
    for storage in _all_storages():
        assert storage.workspace == "forced_workspace"


def test_empty_instance_workspace_falls_back_to_default(monkeypatch):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    for storage in _all_storages(workspace=""):
        assert storage.workspace == "default"


def test_invalid_workspace_override_fails_closed(monkeypatch):
    monkeypatch.setenv(WORKSPACE_ENV_VAR, "../escape")
    with pytest.raises(ValueError, match="workspace"):
        _all_storages()
