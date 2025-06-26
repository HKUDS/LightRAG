import os
from dataclasses import dataclass
from typing import final
import configparser

from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..constants import GRAPH_FIELD_SEP
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (
    AsyncGraphDatabase,
    AsyncManagedTransaction,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)

MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MemgraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None

    async def initialize(self):
        URI = os.environ.get(
            "MEMGRAPH_URI",
            config.get("memgraph", "uri", fallback="bolt://localhost:7687"),
        )
        USERNAME = os.environ.get(
            "MEMGRAPH_USERNAME", config.get("memgraph", "username", fallback="")
        )
        PASSWORD = os.environ.get(
            "MEMGRAPH_PASSWORD", config.get("memgraph", "password", fallback="")
        )
        DATABASE = os.environ.get(
            "MEMGRAPH_DATABASE", config.get("memgraph", "database", fallback="memgraph")
        )

        self._driver = AsyncGraphDatabase.driver(
            URI,
            auth=(USERNAME, PASSWORD),
        )
        self._DATABASE = DATABASE
        try:
            async with self._driver.session(database=DATABASE) as session:
                # Create index for base nodes on entity_id if it doesn't exist
                try:
                    await session.run("""CREATE INDEX ON :base(entity_id)""")
                    logger.info("Created index on :base(entity_id) in Memgraph.")
                except Exception as e:
                    # Index may already exist, which is not an error
                    logger.warning(
                        f"Index creation on :base(entity_id) may have failed or already exists: {e}"
                    )
                await session.run("RETURN 1")
                logger.info(f"Connected to Memgraph at {URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph at {URI}: {e}")
            raise

    async def finalize(self):
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.finalize()

    async def index_done_callback(self):
        # Memgraph handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
            result = await session.run(query, entity_id=node_id)
            single_result = await result.single()
            await result.consume()
            return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = (
                "MATCH (a:base {entity_id: $source_entity_id})-[r]-(b:base {entity_id: $target_entity_id}) "
                "RETURN COUNT(r) > 0 AS edgeExists"
            )
            result = await session.run(
                query,
                source_entity_id=source_node_id,
                target_entity_id=target_node_id,
            )
            single_result = await result.single()
            await result.consume()
            return single_result["edgeExists"]

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
            result = await session.run(query, entity_id=node_id)
            records = await result.fetch(2)
            await result.consume()
            if records:
                node = records[0]["n"]
                node_dict = dict(node)
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        label for label in node_dict["labels"] if label != "base"
                    ]
                return node_dict
            return None

    async def get_all_labels(self) -> list[str]:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (n:base)
            WHERE n.entity_id IS NOT NULL
            RETURN DISTINCT n.entity_id AS label
            ORDER BY label
            """
            result = await session.run(query)
            labels = []
            async for record in result:
                labels.append(record["label"])
            await result.consume()
            return labels

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (n:base {entity_id: $entity_id})
            OPTIONAL MATCH (n)-[r]-(connected:base)
            WHERE connected.entity_id IS NOT NULL
            RETURN n, r, connected
            """
            results = await session.run(query, entity_id=source_node_id)
            edges = []
            async for record in results:
                source_node = record["n"]
                connected_node = record["connected"]
                if not source_node or not connected_node:
                    continue
                source_label = source_node.get("entity_id")
                target_label = connected_node.get("entity_id")
                if source_label and target_label:
                    edges.append((source_label, target_label))
            await results.consume()
            return edges

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
            RETURN properties(r) as edge_properties
            """
            result = await session.run(
                query,
                source_entity_id=source_node_id,
                target_entity_id=target_node_id,
            )
            records = await result.fetch(2)
            await result.consume()
            if records:
                edge_result = dict(records[0]["edge_properties"])
                for key, default_value in {
                    "weight": 0.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }.items():
                    if key not in edge_result:
                        edge_result[key] = default_value
                return edge_result
            return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        properties = node_data
        entity_type = properties.get("entity_type", "base")
        if "entity_id" not in properties:
            raise ValueError(
                "Memgraph: node properties must contain an 'entity_id' field"
            )
        async with self._driver.session(database=self._DATABASE) as session:

            async def execute_upsert(tx: AsyncManagedTransaction):
                query = f"""
                    MERGE (n:base {{entity_id: $entity_id}})
                    SET n += $properties
                    SET n:`{entity_type}`
                    """
                result = await tx.run(query, entity_id=node_id, properties=properties)
                await result.consume()

            await session.execute_write(execute_upsert)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        edge_properties = edge_data
        async with self._driver.session(database=self._DATABASE) as session:

            async def execute_upsert(tx: AsyncManagedTransaction):
                query = """
                MATCH (source:base {entity_id: $source_entity_id})
                WITH source
                MATCH (target:base {entity_id: $target_entity_id})
                MERGE (source)-[r:DIRECTED]-(target)
                SET r += $properties
                RETURN r, source, target
                """
                result = await tx.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                    properties=edge_properties,
                )
                await result.consume()

            await session.execute_write(execute_upsert)

    async def delete_node(self, node_id: str) -> None:
        async def _do_delete(tx: AsyncManagedTransaction):
            query = """
            MATCH (n:base {entity_id: $entity_id})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            await result.consume()

        async with self._driver.session(database=self._DATABASE) as session:
            await session.execute_write(_do_delete)

    async def remove_nodes(self, nodes: list[str]):
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        for source, target in edges:

            async def _do_delete_edge(tx: AsyncManagedTransaction):
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                await result.consume()

            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete_edge)

    async def drop(self) -> dict[str, str]:
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                query = "MATCH (n) DETACH DELETE n"
                result = await session.run(query)
                await result.consume()
                logger.info(
                    f"Process {os.getpid()} drop Memgraph database {self._DATABASE}"
                )
                return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(f"Error dropping Memgraph database {self._DATABASE}: {e}")
            return {"status": "error", "message": str(e)}

    async def node_degree(self, node_id: str) -> int:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
                MATCH (n:base {entity_id: $entity_id})
                OPTIONAL MATCH (n)-[r]-()
                RETURN COUNT(r) AS degree
            """
            result = await session.run(query, entity_id=node_id)
            record = await result.single()
            await result.consume()
            if not record:
                return 0
            return record["degree"]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree
        return int(src_degree) + int(trg_degree)

    async def get_nodes_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (n:base)
            WHERE n.source_id IS NOT NULL AND chunk_id IN split(n.source_id, $sep)
            RETURN DISTINCT n
            """
            result = await session.run(query, chunk_ids=chunk_ids, sep=GRAPH_FIELD_SEP)
            nodes = []
            async for record in result:
                node = record["n"]
                node_dict = dict(node)
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)
            await result.consume()
            return nodes

    async def get_edges_by_chunk_ids(self, chunk_ids: list[str]) -> list[dict]:
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (a:base)-[r]-(b:base)
            WHERE r.source_id IS NOT NULL AND chunk_id IN split(r.source_id, $sep)
            RETURN DISTINCT a.entity_id AS source, b.entity_id AS target, properties(r) AS properties
            """
            result = await session.run(query, chunk_ids=chunk_ids, sep=GRAPH_FIELD_SEP)
            edges = []
            async for record in result:
                edge_properties = record["properties"]
                edge_properties["source"] = record["source"]
                edge_properties["target"] = record["target"]
                edges.append(edge_properties)
            await result.consume()
            return edges

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = MAX_GRAPH_NODES,
    ) -> KnowledgeGraph:
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            if node_label == "*":
                count_query = "MATCH (n) RETURN count(n) as total"
                count_result = await session.run(count_query)
                count_record = await count_result.single()
                await count_result.consume()
                if count_record and count_record["total"] > max_nodes:
                    result.is_truncated = True
                    logger.info(
                        f"Graph truncated: {count_record['total']} nodes found, limited to {max_nodes}"
                    )
                main_query = """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                WITH n, COALESCE(count(r), 0) AS degree
                ORDER BY degree DESC
                LIMIT $max_nodes
                WITH collect({node: n}) AS filtered_nodes
                UNWIND filtered_nodes AS node_info
                WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                OPTIONAL MATCH (a)-[r]-(b)
                WHERE a IN kept_nodes AND b IN kept_nodes
                RETURN filtered_nodes AS node_info,
                       collect(DISTINCT r) AS relationships
                """
                result_set = await session.run(main_query, {"max_nodes": max_nodes})
                record = await result_set.single()
                await result_set.consume()
            else:
                # BFS fallback for Memgraph (no APOC)
                from collections import deque

                # Get the starting node
                start_query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
                node_result = await session.run(start_query, entity_id=node_label)
                node_record = await node_result.single()
                await node_result.consume()
                if not node_record:
                    return result
                start_node = node_record["n"]
                queue = deque([(start_node, 0)])
                visited = set()
                bfs_nodes = []
                while queue and len(bfs_nodes) < max_nodes:
                    current_node, depth = queue.popleft()
                    node_id = current_node.get("entity_id")
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    bfs_nodes.append(current_node)
                    if depth < max_depth:
                        # Get neighbors
                        neighbor_query = """
                        MATCH (n:base {entity_id: $entity_id})-[]-(m:base)
                        RETURN m
                        """
                        neighbors_result = await session.run(
                            neighbor_query, entity_id=node_id
                        )
                        neighbors = [
                            rec["m"] for rec in await neighbors_result.to_list()
                        ]
                        await neighbors_result.consume()
                        for neighbor in neighbors:
                            neighbor_id = neighbor.get("entity_id")
                            if neighbor_id not in visited:
                                queue.append((neighbor, depth + 1))
                # Build subgraph
                subgraph_ids = [n.get("entity_id") for n in bfs_nodes]
                # Nodes
                for n in bfs_nodes:
                    node_id = n.get("entity_id")
                    if node_id not in seen_nodes:
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_id],
                                properties=dict(n),
                            )
                        )
                        seen_nodes.add(node_id)
                # Edges
                if subgraph_ids:
                    edge_query = """
                    MATCH (a:base)-[r]-(b:base)
                    WHERE a.entity_id IN $ids AND b.entity_id IN $ids
                    RETURN DISTINCT r, a, b
                    """
                    edge_result = await session.run(edge_query, ids=subgraph_ids)
                    async for record in edge_result:
                        r = record["r"]
                        a = record["a"]
                        b = record["b"]
                        edge_id = f"{a.get('entity_id')}-{b.get('entity_id')}"
                        if edge_id not in seen_edges:
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=edge_id,
                                    type="DIRECTED",
                                    source=a.get("entity_id"),
                                    target=b.get("entity_id"),
                                    properties=dict(r),
                                )
                            )
                            seen_edges.add(edge_id)
                    await edge_result.consume()
        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result
