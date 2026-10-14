"""Edit and delete must not hang on a self-loop edge that is already stored.

New self-loops are refused at every ingress, but graphs written by older
releases can still hold one. Both ``aedit_relation`` and
``adelete_by_relation`` lock ``sorted([src, tgt])``; for ``src == tgt`` that
list repeats a key, which used to deadlock the call (issue #4148). Deleting the
legacy edge is the way out, so it must complete.
"""

import asyncio

import pytest

from lightrag import utils_graph
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

NAME = "LED"


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _KV:
    def __init__(self):
        self.records: dict = {}

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(data)

    async def delete(self, ids):
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        pass


class _VDB:
    def __init__(self, global_config):
        self.global_config = global_config

    async def upsert(self, data):
        pass

    async def delete(self, ids):
        pass

    async def index_done_callback(self):
        pass


@pytest.fixture
async def graph_with_self_loop(tmp_path):
    config = {"working_dir": str(tmp_path), "workspace": "", "embedding_batch_num": 10}
    graph = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace="",
        global_config=config,
        embedding_func=None,
    )
    await graph.initialize()
    await graph.upsert_node(
        NAME, {"entity_id": NAME, "description": "d", "source_id": "chunk-1"}
    )
    await graph.upsert_edge(
        NAME, NAME, {"description": "legacy", "weight": 1.0, "source_id": "chunk-1"}
    )
    await graph.index_done_callback()
    yield graph, config
    await graph.finalize()


@pytest.mark.asyncio
async def test_delete_of_a_stored_self_loop_completes(graph_with_self_loop):
    graph, config = graph_with_self_loop

    result = await asyncio.wait_for(
        utils_graph.adelete_by_relation(
            graph, _VDB(config), NAME, NAME, relation_chunks_storage=_KV()
        ),
        timeout=5,
    )

    assert result.status == "success"
    assert not await graph.has_edge(NAME, NAME)


@pytest.mark.asyncio
async def test_edit_of_a_stored_self_loop_completes(graph_with_self_loop):
    graph, config = graph_with_self_loop

    await asyncio.wait_for(
        utils_graph.aedit_relation(
            graph,
            _VDB(config),
            _VDB(config),
            NAME,
            NAME,
            {"description": "edited"},
            relation_chunks_storage=_KV(),
        ),
        timeout=5,
    )

    edge = await graph.get_edge(NAME, NAME)
    assert edge["description"] == "edited"
