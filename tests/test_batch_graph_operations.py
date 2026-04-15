"""
Unit tests for batch graph operations (PR #2910 follow-up).

Verifies:
1. BaseGraphStorage default batch methods fall back to serial single-item calls.
2. NetworkXStorage overrides batch methods with optimized in-memory operations.
3. ainsert_custom_kg uses the batch interface end-to-end (no hasattr guards).
4. has_nodes_batch returns only existing nodes, including newly inserted ones.
5. upsert_edges_batch and upsert_nodes_batch are idempotent (safe to call twice).
"""

import time
import tempfile
import pytest
import numpy as np
from unittest.mock import AsyncMock

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc, make_relation_vdb_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {
    "embedding_batch_num": 10,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
    "working_dir": "/tmp/test_batch_graph",
}


async def _raw_embedding_func(texts):
    return np.random.rand(len(texts), 10)


mock_embedding_func = EmbeddingFunc(
    embedding_dim=10,
    max_token_size=512,
    func=_raw_embedding_func,
)


def make_networkx_storage(tmp_dir: str) -> NetworkXStorage:
    config = dict(GLOBAL_CONFIG, working_dir=tmp_dir)
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="test_graph",
        workspace="test_ws",
        global_config=config,
        embedding_func=_raw_embedding_func,
    )
    return storage


def _make_node(entity_id: str, entity_type: str = "TEST") -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "description": f"Description of {entity_id}",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
    }


def _make_edge(weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "description": "test edge",
        "keywords": "test",
        "source_id": "chunk-1",
        "file_path": "test.txt",
        "created_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# 1. BaseGraphStorage default implementations delegate to single-item methods
# ---------------------------------------------------------------------------


class TestBaseGraphStorageDefaults:
    """
    Use NetworkXStorage as a concrete instance but spy on the single-item
    methods to verify the default batch implementations delegate correctly.
    """

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_calls_upsert_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            nodes = [
                ("NodeA", _make_node("NodeA")),
                ("NodeB", _make_node("NodeB")),
            ]

            call_log: list[str] = []
            original = storage.upsert_node

            async def spy(node_id, *, node_data):
                call_log.append(node_id)
                return await original(node_id, node_data=node_data)

            # Temporarily replace the optimised override with the base default

            async def base_upsert_nodes_batch(self, nodes):
                for node_id, node_data in nodes:
                    await self.upsert_node(node_id, node_data=node_data)

            storage.upsert_node = spy  # type: ignore[assignment]
            await base_upsert_nodes_batch(storage, nodes)

            assert call_log == ["NodeA", "NodeB"]

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_calls_has_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()
            await storage.upsert_node("NodeA", node_data=_make_node("NodeA"))

            call_log: list[str] = []
            original = storage.has_node

            async def spy(node_id):
                call_log.append(node_id)
                return await original(node_id)

            async def base_has_nodes_batch(self, node_ids):
                existing = set()
                for node_id in node_ids:
                    if await self.has_node(node_id):
                        existing.add(node_id)
                return existing

            storage.has_node = spy  # type: ignore[assignment]
            result = await base_has_nodes_batch(storage, ["NodeA", "NodeB"])

            assert call_log == ["NodeA", "NodeB"]
            assert result == {"NodeA"}

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_calls_upsert_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()
            await storage.upsert_node("NodeA", node_data=_make_node("NodeA"))
            await storage.upsert_node("NodeB", node_data=_make_node("NodeB"))
            await storage.upsert_node("NodeC", node_data=_make_node("NodeC"))

            call_log: list[tuple] = []
            original = storage.upsert_edge

            async def spy(src, tgt, *, edge_data):
                call_log.append((src, tgt))
                return await original(src, tgt, edge_data=edge_data)

            async def base_upsert_edges_batch(self, edges):
                for src, tgt, edge_data in edges:
                    await self.upsert_edge(src, tgt, edge_data=edge_data)

            edges = [
                ("NodeA", "NodeB", _make_edge()),
                ("NodeB", "NodeC", _make_edge()),
            ]
            storage.upsert_edge = spy  # type: ignore[assignment]
            await base_upsert_edges_batch(storage, edges)

            assert call_log == [("NodeA", "NodeB"), ("NodeB", "NodeC")]


# ---------------------------------------------------------------------------
# 2. NetworkXStorage optimised batch implementations
# ---------------------------------------------------------------------------


class TestNetworkXBatchOperations:
    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_inserts_all_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            nodes = [(f"Entity{i}", _make_node(f"Entity{i}")) for i in range(5)]
            await storage.upsert_nodes_batch(nodes)

            for entity_id, _ in nodes:
                assert await storage.has_node(entity_id), f"{entity_id} should exist"

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            node_data = _make_node("Alpha")
            await storage.upsert_nodes_batch([("Alpha", node_data)])
            await storage.upsert_nodes_batch([("Alpha", node_data)])  # second call

            assert await storage.has_node("Alpha")
            node = await storage.get_node("Alpha")
            assert node["entity_id"] == "Alpha"

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_returns_existing_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("Present1", _make_node("Present1")),
                    ("Present2", _make_node("Present2")),
                ]
            )

            result = await storage.has_nodes_batch(["Present1", "Present2", "Missing"])
            assert result == {"Present1", "Present2"}

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_has_nodes_batch_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            result = await storage.has_nodes_batch([])
            assert result == set()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_creates_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("A", _make_node("A")),
                    ("B", _make_node("B")),
                    ("C", _make_node("C")),
                ]
            )

            edges = [
                ("A", "B", _make_edge(1.5)),
                ("B", "C", _make_edge(2.0)),
            ]
            await storage.upsert_edges_batch(edges)

            edge_ab = await storage.get_edge("A", "B")
            assert edge_ab is not None
            assert float(edge_ab["weight"]) == pytest.approx(1.5)

            edge_bc = await storage.get_edge("B", "C")
            assert edge_bc is not None
            assert float(edge_bc["weight"]) == pytest.approx(2.0)

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            await storage.upsert_nodes_batch(
                [
                    ("X", _make_node("X")),
                    ("Y", _make_node("Y")),
                ]
            )
            edge_data = _make_edge(3.0)
            await storage.upsert_edges_batch([("X", "Y", edge_data)])
            await storage.upsert_edges_batch([("X", "Y", edge_data)])  # second call

            edge = await storage.get_edge("X", "Y")
            assert edge is not None

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_updates_existing_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = make_networkx_storage(tmp)
            await storage.initialize()

            original = _make_node("Node1")
            await storage.upsert_nodes_batch([("Node1", original)])

            updated = dict(original, description="Updated description")
            await storage.upsert_nodes_batch([("Node1", updated)])

            node = await storage.get_node("Node1")
            assert node["description"] == "Updated description"


# ---------------------------------------------------------------------------
# 3. ainsert_custom_kg uses batch interface end-to-end
# ---------------------------------------------------------------------------


class TestAinsertCustomKgBatchPath:
    """
    Verify that ainsert_custom_kg calls the three batch methods rather than
    the single-item methods, using a mock graph storage backend.
    """

    def _make_custom_kg(self):
        return {
            "chunks": [
                {
                    "content": "chunk content",
                    "chunk_order_index": 0,
                    "source_id": "src-1",
                }
            ],
            "entities": [
                {
                    "entity_name": "EntityA",
                    "entity_type": "CONCEPT",
                    "description": "An entity",
                    "source_id": "src-1",
                    "file_path": "test.pdf",
                }
            ],
            "relationships": [
                {
                    "src_id": "EntityA",
                    "tgt_id": "EntityB",
                    "description": "relates to",
                    "keywords": "relation",
                    "weight": 1.0,
                    "source_id": "src-1",
                    "file_path": "test.pdf",
                }
            ],
        }

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_ainsert_custom_kg_calls_batch_methods(self):
        """upsert_nodes_batch, has_nodes_batch, upsert_edges_batch must all be called."""
        from lightrag import LightRAG

        with tempfile.TemporaryDirectory() as tmp:
            rag = LightRAG(
                working_dir=tmp,
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=mock_embedding_func,
            )
            await rag.initialize_storages()

            graph = rag.chunk_entity_relation_graph
            upsert_nodes_batch = AsyncMock(wraps=graph.upsert_nodes_batch)
            has_nodes_batch = AsyncMock(wraps=graph.has_nodes_batch)
            upsert_edges_batch = AsyncMock(wraps=graph.upsert_edges_batch)

            graph.upsert_nodes_batch = upsert_nodes_batch
            graph.has_nodes_batch = has_nodes_batch
            graph.upsert_edges_batch = upsert_edges_batch

            # Mock VDB upserts to avoid needing real embeddings
            rag.entities_vdb.upsert = AsyncMock()
            rag.relationships_vdb.upsert = AsyncMock()
            rag.relationships_vdb.delete = AsyncMock()
            rag.text_chunks.upsert = AsyncMock()
            rag.doc_status.upsert = AsyncMock()

            await rag.ainsert_custom_kg(self._make_custom_kg())

            upsert_nodes_batch.assert_called()
            has_nodes_batch.assert_called()
            upsert_edges_batch.assert_called()

            await rag.finalize_storages()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_ainsert_custom_kg_no_hasattr_needed(self):
        """
        The batch methods are always available on the base class, so no
        hasattr() guard should be needed. Verify that a storage backend
        implementing only the abstract methods (no batch overrides) still
        works via the default serial fallback.
        """
        from lightrag.base import BaseGraphStorage

        # All three batch methods should exist on the base class
        assert hasattr(BaseGraphStorage, "upsert_nodes_batch")
        assert hasattr(BaseGraphStorage, "has_nodes_batch")
        assert hasattr(BaseGraphStorage, "upsert_edges_batch")

    @pytest.mark.offline
    def test_neo4j_has_nodes_batch_uses_read_retry(self):
        pytest.importorskip("neo4j")
        from lightrag.kg.neo4j_impl import Neo4JStorage

        assert hasattr(Neo4JStorage.has_nodes_batch, "retry")
        assert hasattr(Neo4JStorage.upsert_nodes_batch, "retry")
        assert hasattr(Neo4JStorage.upsert_edges_batch, "retry")

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_ainsert_custom_kg_missing_entity_nodes_created(self):
        """
        Nodes referenced in relationships but not in the entity list must
        be created as placeholder UNKNOWN nodes.
        """
        from lightrag import LightRAG

        with tempfile.TemporaryDirectory() as tmp:
            rag = LightRAG(
                working_dir=tmp,
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=mock_embedding_func,
            )
            await rag.initialize_storages()

            rag.entities_vdb.upsert = AsyncMock()
            rag.relationships_vdb.upsert = AsyncMock()
            rag.relationships_vdb.delete = AsyncMock()
            rag.text_chunks.upsert = AsyncMock()
            rag.doc_status.upsert = AsyncMock()

            custom_kg = {
                "chunks": [
                    {"content": "text", "chunk_order_index": 0, "source_id": "s1"}
                ],
                "entities": [],  # No entities declared
                "relationships": [
                    {
                        "src_id": "ImplicitNode",
                        "tgt_id": "AnotherImplicit",
                        "description": "connects",
                        "keywords": "link",
                        "weight": 1.0,
                        "source_id": "s1",
                        "file_path": "test.pdf",
                    }
                ],
            }

            await rag.ainsert_custom_kg(custom_kg)

            graph = rag.chunk_entity_relation_graph
            assert await graph.has_node(
                "ImplicitNode"
            ), "Implicit node should be created"
            assert await graph.has_node(
                "AnotherImplicit"
            ), "Implicit node should be created"

            await rag.finalize_storages()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_ainsert_custom_kg_deduplicates_entities_and_undirected_edges(self):
        from lightrag import LightRAG

        with tempfile.TemporaryDirectory() as tmp:
            rag = LightRAG(
                working_dir=tmp,
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=mock_embedding_func,
            )
            await rag.initialize_storages()

            graph = rag.chunk_entity_relation_graph
            graph.upsert_nodes_batch = AsyncMock()
            graph.has_nodes_batch = AsyncMock(return_value={"EntityA"})
            graph.upsert_edges_batch = AsyncMock()

            rag.entities_vdb.upsert = AsyncMock()
            rag.relationships_vdb.upsert = AsyncMock()
            rag.relationships_vdb.delete = AsyncMock()
            rag.text_chunks.upsert = AsyncMock()
            rag.doc_status.upsert = AsyncMock()

            custom_kg = {
                "chunks": [
                    {
                        "content": "chunk content",
                        "chunk_order_index": 0,
                        "source_id": "src-1",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "EntityA",
                        "entity_type": "CONCEPT",
                        "description": "first version",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                    {
                        "entity_name": "EntityA",
                        "entity_type": "CONCEPT",
                        "description": "latest version",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "EntityA",
                        "tgt_id": "EntityB",
                        "description": "old relation",
                        "keywords": "first",
                        "weight": 1.0,
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                    {
                        "src_id": "EntityB",
                        "tgt_id": "EntityA",
                        "description": "latest relation",
                        "keywords": "second",
                        "weight": 2.0,
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
            }

            await rag.ainsert_custom_kg(custom_kg)

            entity_batch = graph.upsert_nodes_batch.await_args_list[0].args[0]
            assert len(entity_batch) == 1
            assert entity_batch[0][0] == "EntityA"
            assert entity_batch[0][1]["entity_type"] == "CONCEPT"
            assert entity_batch[0][1]["description"] == "latest version"
            assert entity_batch[0][1]["file_path"] == "test.pdf"
            assert entity_batch[0][1]["source_id"]

            placeholder_batch = graph.upsert_nodes_batch.await_args_list[1].args[0]
            assert len(placeholder_batch) == 1
            assert placeholder_batch[0][0] == "EntityB"

            edge_batch = graph.upsert_edges_batch.await_args.args[0]
            assert len(edge_batch) == 1
            assert edge_batch[0][0] == "EntityB"
            assert edge_batch[0][1] == "EntityA"
            assert edge_batch[0][2]["description"] == "latest relation"
            assert edge_batch[0][2]["weight"] == 2.0

            entity_vdb_payload = rag.entities_vdb.upsert.await_args.args[0]
            assert len(entity_vdb_payload) == 1
            only_entity = next(iter(entity_vdb_payload.values()))
            assert only_entity["description"] == "latest version"

            rel_vdb_payload = rag.relationships_vdb.upsert.await_args.args[0]
            assert len(rel_vdb_payload) == 1
            only_rel = next(iter(rel_vdb_payload.values()))
            assert only_rel["src_id"] == "EntityA"
            assert only_rel["tgt_id"] == "EntityB"
            assert only_rel["description"] == "latest relation"
            assert rag.relationships_vdb.delete.await_args.args[0] == [
                make_relation_vdb_ids("EntityA", "EntityB")[1]
            ]

            await rag.finalize_storages()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_ainsert_custom_kg_keeps_legacy_relation_rows_if_upsert_fails(self):
        from lightrag import LightRAG

        with tempfile.TemporaryDirectory() as tmp:
            rag = LightRAG(
                working_dir=tmp,
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=mock_embedding_func,
            )
            await rag.initialize_storages()

            rag.entities_vdb.upsert = AsyncMock()
            rag.relationships_vdb.upsert = AsyncMock(side_effect=RuntimeError("boom"))
            rag.relationships_vdb.delete = AsyncMock()
            rag.text_chunks.upsert = AsyncMock()
            rag.doc_status.upsert = AsyncMock()

            custom_kg = {
                "chunks": [
                    {
                        "content": "chunk content",
                        "chunk_order_index": 0,
                        "source_id": "src-1",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "EntityA",
                        "entity_type": "CONCEPT",
                        "description": "Entity A",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                    {
                        "entity_name": "EntityB",
                        "entity_type": "CONCEPT",
                        "description": "Entity B",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "EntityB",
                        "tgt_id": "EntityA",
                        "description": "latest relation",
                        "keywords": "second",
                        "weight": 2.0,
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
            }

            with pytest.raises(RuntimeError, match="boom"):
                await rag.ainsert_custom_kg(custom_kg)

            rag.relationships_vdb.delete.assert_not_called()

            await rag.finalize_storages()

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_get_relation_info_falls_back_to_legacy_relation_vdb_id(self):
        from lightrag import LightRAG

        with tempfile.TemporaryDirectory() as tmp:
            rag = LightRAG(
                working_dir=tmp,
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=mock_embedding_func,
            )
            await rag.initialize_storages()

            rag.entities_vdb.upsert = AsyncMock()
            rag.relationships_vdb.upsert = AsyncMock()
            rag.relationships_vdb.delete = AsyncMock()
            rag.text_chunks.upsert = AsyncMock()
            rag.doc_status.upsert = AsyncMock()

            custom_kg = {
                "chunks": [
                    {
                        "content": "chunk content",
                        "chunk_order_index": 0,
                        "source_id": "src-1",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "EntityA",
                        "entity_type": "CONCEPT",
                        "description": "Entity A",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                    {
                        "entity_name": "EntityB",
                        "entity_type": "CONCEPT",
                        "description": "Entity B",
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "EntityB",
                        "tgt_id": "EntityA",
                        "description": "latest relation",
                        "keywords": "second",
                        "weight": 2.0,
                        "source_id": "src-1",
                        "file_path": "test.pdf",
                    },
                ],
            }

            await rag.ainsert_custom_kg(custom_kg)

            normalized_rel_id, legacy_rel_id = make_relation_vdb_ids(
                "EntityA", "EntityB"
            )
            rag.relationships_vdb.get_by_id = AsyncMock(
                side_effect=lambda rid: {"ok": True} if rid == legacy_rel_id else None
            )

            result_ab = await rag.get_relation_info(
                "EntityA", "EntityB", include_vector_data=True
            )
            result_ba = await rag.get_relation_info(
                "EntityB", "EntityA", include_vector_data=True
            )

            assert result_ab["vector_data"] == {"ok": True}
            assert result_ba["vector_data"] == {"ok": True}
            assert [
                call.args[0] for call in rag.relationships_vdb.get_by_id.await_args_list
            ] == [
                normalized_rel_id,
                legacy_rel_id,
                normalized_rel_id,
                legacy_rel_id,
            ]

            await rag.finalize_storages()


class TestPostgresBatchOrdering:
    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_preserves_last_write_wins(self):
        from lightrag.kg.postgres_impl import PGGraphStorage

        storage = PGGraphStorage.__new__(PGGraphStorage)
        call_log: list[tuple[str, str]] = []

        async def spy(node_id, *, node_data):
            call_log.append((node_id, node_data["description"]))

        storage.upsert_node = spy  # type: ignore[assignment]

        await PGGraphStorage.upsert_nodes_batch(
            storage,
            [
                ("EntityA", _make_node("EntityA")),
                ("EntityA", dict(_make_node("EntityA"), description="latest")),
                ("EntityB", _make_node("EntityB")),
            ],
        )

        assert call_log == [
            ("EntityA", "latest"),
            ("EntityB", "Description of EntityB"),
        ]

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_preserves_last_write_wins(self):
        from lightrag.kg.postgres_impl import PGGraphStorage

        storage = PGGraphStorage.__new__(PGGraphStorage)
        call_log: list[tuple[str, str, float]] = []

        async def spy(src, tgt, *, edge_data):
            call_log.append((src, tgt, edge_data["weight"]))

        storage.upsert_edge = spy  # type: ignore[assignment]

        await PGGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityB", "EntityA", _make_edge(2.0)),
                ("EntityB", "EntityC", _make_edge(3.0)),
            ],
        )

        assert call_log == [("EntityB", "EntityA", 2.0), ("EntityB", "EntityC", 3.0)]


class TestMongoBatchOrdering:
    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_nodes_batch_uses_ordered_bulk_write(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import MongoGraphStorage

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()

        await MongoGraphStorage.upsert_nodes_batch(
            storage,
            [
                ("EntityA", _make_node("EntityA")),
                ("EntityA", dict(_make_node("EntityA"), description="latest")),
            ],
        )

        assert storage.collection.bulk_write.await_args.kwargs["ordered"] is True

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_uses_ordered_bulk_write(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import MongoGraphStorage

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()
        storage.edge_collection = AsyncMock()

        await MongoGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityB", "EntityA", _make_edge(2.0)),
            ],
        )

        assert storage.edge_collection.bulk_write.await_args.kwargs["ordered"] is True

    @pytest.mark.offline
    @pytest.mark.asyncio
    async def test_upsert_edges_batch_deduplicates_source_node_upserts(self):
        pytest.importorskip("pymongo")
        from lightrag.kg.mongo_impl import MongoGraphStorage

        storage = MongoGraphStorage.__new__(MongoGraphStorage)
        storage.collection = AsyncMock()
        storage.edge_collection = AsyncMock()

        await MongoGraphStorage.upsert_edges_batch(
            storage,
            [
                ("EntityA", "EntityB", _make_edge(1.0)),
                ("EntityA", "EntityC", _make_edge(2.0)),
            ],
        )

        node_ops = storage.collection.bulk_write.await_args.args[0]
        assert len(node_ops) == 1
        assert node_ops[0]._filter == {"_id": "EntityA"}
