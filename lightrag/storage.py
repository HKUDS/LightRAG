import asyncio
import html
import os
from dataclasses import dataclass, field
from typing import Any, Union, cast, Tuple, List, Dict
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
from neo4j import AsyncGraphDatabase
from .utils import load_json, logger, write_json
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)

@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids
@dataclass
class Neo4jKVStorage(BaseKVStorage):
    def __post_init__(self):
        logger.debug(f"Global config: {self.global_config}")
        neo4j_config: dict = self.global_config.get("neo4j_config", {})
        self.uri = neo4j_config.get("uri")
        self.username = neo4j_config.get("username")
        self.password = neo4j_config.get("password")
        
        if not self.namespace:
            self.namespace = neo4j_config.get("namespace", "default_namespace")
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))

    async def close(self):
        await self.driver.close()

    async def all_keys(self) -> list[str]:
        query = f"MATCH (n:{self.namespace}) RETURN n.id AS id"
        async with self.driver.session() as session:
            result = await session.run(query)
            keys = []
            async for record in result:
                keys.append(record["id"])
            return keys

    async def get_by_id(self, id: str):
        query = f"MATCH (n:{self.namespace} {{id: $id}}) RETURN n"
        async with self.driver.session() as session:
            result = await session.run(query, id=id)
            record = await result.single()
            return dict(record["n"]) if record else None

    async def get_by_ids(self, ids: list[str], fields: list[str] = None):
        field_str = ", ".join([f"n.{field}" for field in fields]) if fields else "n"
        query = f"MATCH (n:{self.namespace}) WHERE n.id IN $ids RETURN {field_str}"
        async with self.driver.session() as session:
            result = await session.run(query, ids=ids)
            records = []
            async for record in result:
                records.append(dict(record["n"]))
            return records

    async def filter_keys(self, data: list[str]) -> set[str]:
        query = f"MATCH (n:{self.namespace}) WHERE n.id IN $data RETURN n.id AS id"
        async with self.driver.session() as session:
            result = await session.run(query, data=data)
            existing_keys = set()
            async for record in result:
                existing_keys.add(record["id"])
            return set(data) - existing_keys

    async def upsert(self, data: dict[str, dict]):
        query = f"""
        UNWIND $data AS row
        MERGE (n:{self.namespace} {{id: row.id}})
        SET n += row.properties
        RETURN n.id AS id
        """
        async with self.driver.session() as session:
            await session.run(query, data=[{"id": k, "properties": v} for k, v in data.items()])
        return data

    async def drop(self):
        query = f"MATCH (n:{self.namespace}) DETACH DELETE n"
        async with self.driver.session() as session:
            await session.run(query)
                      
@dataclass
class Neo4jGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        neo4j_config: dict = self.global_config.get("neo4j_config", {})
        self.uri = neo4j_config.get("uri")
        self.username = neo4j_config.get("username")
        self.password = neo4j_config.get("password")

        if not self.namespace:
            self.namespace = neo4j_config.get("namespace", "default_namespace")
        
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.username, self.password))
        logger.info(f"Connected to Neo4j at {self.uri} with namespace '{self.namespace}'")

        # Initialize node embedding algorithms
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        await self.driver.close()

    async def index_done_callback(self):
        # Since we don't have a graphml file in Neo4j, we can perform any necessary finalization here
        logger.info("Indexing done. You can add any finalization logic if needed.")

    async def has_node(self, node_id: str) -> bool:
        query = f"MATCH (n:{self.namespace} {{id: $id}}) RETURN n LIMIT 1"
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            return record is not None

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        query = f"""
        MATCH (n:{self.namespace} {{id: $source_id}})-[r]->(m:{self.namespace} {{id: $target_id}})
        RETURN r LIMIT 1
        """
        async with self.driver.session() as session:
            result = await session.run(query, source_id=source_node_id, target_id=target_node_id)
            record = await result.single()
            return record is not None

    async def get_node(self, node_id: str) -> Union[dict, None]:
        query = f"MATCH (n:{self.namespace} {{id: $id}}) RETURN properties(n) AS props"
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                return record["props"]
            else:
                return None

    async def node_degree(self, node_id: str) -> int:
        query = f"MATCH (n:{self.namespace} {{id: $id}})-[r]-() RETURN count(r) as degree"
        async with self.driver.session() as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record:
                return record["degree"]
            else:
                return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        query = f"""
        MATCH (n:{self.namespace} {{id: $source_id}})-[r]->(m:{self.namespace} {{id: $target_id}})
        RETURN properties(r) AS props
        """
        async with self.driver.session() as session:
            result = await session.run(query, source_id=source_node_id, target_id=target_node_id)
            record = await result.single()
            if record:
                return record["props"]
            else:
                return None

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        query = f"""
        MATCH (n:{self.namespace} {{id: $source_id}})-[r]->(m:{self.namespace})
        RETURN n.id AS source_id, m.id AS target_id
        """
        async with self.driver.session() as session:
            result = await session.run(query, source_id=source_node_id)
            edges = []
            async for record in result:
                source_id = record["source_id"]
                target_id = record["target_id"]
                edges.append((source_id, target_id))
            return edges

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        query = f"""
        MERGE (n:{self.namespace} {{id: $id}})
        SET n += $properties
        """
        async with self.driver.session() as session:
            await session.run(query, id=node_id, properties=node_data)
        logger.info(f"Upserted node with ID '{node_id}' and label '{self.namespace}'")

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, Any]):
        query = f"""
        MERGE (source:{self.namespace} {{id: $source_id}})
        MERGE (target:{self.namespace} {{id: $target_id}})
        MERGE (source)-[r:RELATION]->(target)
        SET r += $properties
        """
        async with self.driver.session() as session:
            await session.run(query, source_id=source_node_id, target_id=target_node_id, properties=edge_data)
        logger.info(f"Upserted edge from '{source_node_id}' to '{target_node_id}' with label '{self.namespace}'")

    async def embed_nodes(self, algorithm: str) -> Tuple[np.ndarray, List[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        graph_name = f"{self.namespace}_graph"
        async with self.driver.session() as session:
            # Drop existing in-memory graph if exists
            await session.run(f"CALL gds.graph.drop($graph_name, false) YIELD graphName", graph_name=graph_name)
            
            # Create an in-memory graph
            await session.run(f"""
            CALL gds.graph.project(
                $graph_name,
                $node_label,
                {{
                    RELATION: {{
                        orientation: 'UNDIRECTED'
                    }}
                }}
            ) YIELD graphName, nodeCount, relationshipCount
            """, graph_name=graph_name, node_label=self.namespace)

            # Run node2vec embedding
            result = await session.run(f"""
            CALL gds.node2vec.stream($graph_name, {{
                embeddingDimension: $dimensions,
                walkLength: $walk_length,
                walksPerNode: $num_walks,
                windowSize: $window_size,
                iterations: $iterations
            }})
            YIELD nodeId, embedding
            RETURN gds.util.asNode(nodeId).id AS id, embedding
            """, 
            graph_name=graph_name,
            dimensions=self.global_config["node2vec_params"]["dimensions"],
            walk_length=self.global_config["node2vec_params"]["walk_length"],
            num_walks=self.global_config["node2vec_params"]["num_walks"],
            window_size=self.global_config["node2vec_params"]["window_size"],
            iterations=self.global_config["node2vec_params"]["iterations"],
            )
            embeddings = []
            node_ids = []
            async for record in result:
                node_ids.append(record["id"])
                embeddings.append(record["embedding"])
            embeddings = np.array(embeddings)
            return embeddings, node_ids

    async def drop(self):
        # Deletes all nodes and relationships with the given namespace label
        query = f"MATCH (n:{self.namespace}) DETACH DELETE n"
        async with self.driver.session() as session:
            await session.run(query)