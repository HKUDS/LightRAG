import asyncio
import inspect
import json
import os
import pipmaster as pm
from dataclasses import dataclass
from typing import Any, Dict, List, final

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from lightrag.utils import logger

from ..base import BaseGraphStorage

if not pm.is_installed("gremlinpython"):
    pm.install("gremlinpython")

from gremlin_python.driver import client, serializer  # type: ignore
from gremlin_python.driver.aiohttp.transport import AiohttpTransport  # type: ignore
from gremlin_python.driver.protocol import GremlinServerError  # type: ignore


@final
@dataclass
class GremlinStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with Gremlin in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

        self._driver = None
        self._driver_lock = asyncio.Lock()

        USER = os.environ.get("GREMLIN_USER", "")
        PASSWORD = os.environ.get("GREMLIN_PASSWORD", "")
        HOST = os.environ["GREMLIN_HOST"]
        PORT = int(os.environ["GREMLIN_PORT"])

        # TraversalSource, a custom one has to be created manually,
        # default it "g"
        SOURCE = os.environ.get("GREMLIN_TRAVERSE_SOURCE", "g")

        # All vertices will have graph={GRAPH} property, so that we can
        # have several logical graphs for one source
        GRAPH = GremlinStorage._to_value_map(
            os.environ.get("GREMLIN_GRAPH", "LightRAG")
        )

        self.graph_name = GRAPH

        self._driver = client.Client(
            f"ws://{HOST}:{PORT}/gremlin",
            SOURCE,
            username=USER,
            password=PASSWORD,
            message_serializer=serializer.GraphSONSerializersV3d0(),
            transport_factory=lambda: AiohttpTransport(call_from_event_loop=True),
        )

    async def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            self._driver.close()

    async def index_done_callback(self) -> None:
        # Gremlin handles persistence automatically
        pass

    @staticmethod
    def _to_value_map(value: Any) -> str:
        """Dump supported Python object as Gremlin valueMap"""
        json_str = json.dumps(value, ensure_ascii=False, sort_keys=False)
        parsed_str = json_str.replace("'", r"\'")

        # walk over the string and replace curly brackets with square brackets
        # outside of strings, as well as replace double quotes with single quotes
        # and "deescape" double quotes inside of strings
        outside_str = True
        escaped = False
        remove_indices = []
        for i, c in enumerate(parsed_str):
            if escaped:
                # previous character was an "odd" backslash
                escaped = False
                if c == '"':
                    # we want to "deescape" double quotes: store indices to delete
                    remove_indices.insert(0, i - 1)
            elif c == "\\":
                escaped = True
            elif c == '"':
                outside_str = not outside_str
                parsed_str = parsed_str[:i] + "'" + parsed_str[i + 1 :]
            elif c == "{" and outside_str:
                parsed_str = parsed_str[:i] + "[" + parsed_str[i + 1 :]
            elif c == "}" and outside_str:
                parsed_str = parsed_str[:i] + "]" + parsed_str[i + 1 :]
        for idx in remove_indices:
            parsed_str = parsed_str[:idx] + parsed_str[idx + 1 :]
        return parsed_str

    @staticmethod
    def _convert_properties(properties: Dict[str, Any]) -> str:
        """Create chained .property() commands from properties dict"""
        props = []
        for k, v in properties.items():
            prop_name = GremlinStorage._to_value_map(k)
            props.append(f".property({prop_name}, {GremlinStorage._to_value_map(v)})")
        return "".join(props)

    @staticmethod
    def _fix_name(name: str) -> str:
        """Strip double quotes and format as a proper field name"""
        name = GremlinStorage._to_value_map(name.strip('"').replace(r"\'", "'"))

        return name

    async def _query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the Gremlin graph

        Args:
            query (str): a query to be executed

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """

        result = list(await asyncio.wrap_future(self._driver.submit_async(query)))
        if result:
            result = result[0]

        return result

    async def has_node(self, node_id: str) -> bool:
        entity_name = GremlinStorage._fix_name(node_id)

        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name})
                 .limit(1)
                 .count()
                 .project('has_node')
                    .by(__.choose(__.is(gt(0)), constant(true), constant(false)))
                 """
        result = await self._query(query)
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            result[0]["has_node"],
        )

        return result[0]["has_node"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_source = GremlinStorage._fix_name(source_node_id)
        entity_name_target = GremlinStorage._fix_name(target_node_id)

        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name_source})
                 .outE()
                 .inV().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name_target})
                 .limit(1)
                 .count()
                 .project('has_edge')
                    .by(__.choose(__.is(gt(0)), constant(true), constant(false)))
                 """
        result = await self._query(query)
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            result[0]["has_edge"],
        )

        return result[0]["has_edge"]

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        entity_name = GremlinStorage._fix_name(node_id)
        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name})
                 .limit(1)
                 .project('properties')
                    .by(elementMap())
                 """
        result = await self._query(query)
        if result:
            node = result[0]
            node_dict = node["properties"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query.format,
                node_dict,
            )
            return node_dict

    async def node_degree(self, node_id: str) -> int:
        entity_name = GremlinStorage._fix_name(node_id)
        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name})
                 .outE()
                 .inV().has('graph', {self.graph_name})
                 .count()
                 .project('total_edge_count')
                    .by()
                 """
        result = await self._query(query)
        edge_count = result[0]["total_edge_count"]

        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            edge_count,
        )

        return edge_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            "{%s}:query:src_Degree+trg_degree:result:{%s}",
            inspect.currentframe().f_code.co_name,
            degrees,
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        entity_name_source = GremlinStorage._fix_name(source_node_id)
        entity_name_target = GremlinStorage._fix_name(target_node_id)
        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name_source})
                 .outE()
                 .inV().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name_target})
                 .limit(1)
                 .project('edge_properties')
                 .by(__.bothE().elementMap())
                 """
        result = await self._query(query)
        if result:
            edge_properties = result[0]["edge_properties"]
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_properties,
            )
            return edge_properties

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        node_name = GremlinStorage._fix_name(source_node_id)
        query = f"""g
                 .E()
                 .filter(
                     __.or(
                         __.outV().has('graph', {self.graph_name})
                           .has('entity_name', {node_name}),
                         __.inV().has('graph', {self.graph_name})
                           .has('entity_name', {node_name})
                     )
                 )
                 .project('source_name', 'target_name')
                 .by(__.outV().values('entity_name'))
                 .by(__.inV().values('entity_name'))
                 """
        result = await self._query(query)
        edges = [(res["source_name"], res["target_name"]) for res in result]

        return edges

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GremlinServerError,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Gremlin graph.

        Args:
            node_id: The unique identifier for the node (used as name)
            node_data: Dictionary of node properties
        """
        name = GremlinStorage._fix_name(node_id)
        properties = GremlinStorage._convert_properties(node_data)

        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {name})
                 .fold()
                 .coalesce(
                     __.unfold(),
                     __.addV('ENTITY')
                         .property('graph', {self.graph_name})
                         .property('entity_name', {name})
                 )
                 {properties}
                 """

        try:
            await self._query(query)
            logger.debug(
                "Upserted node with name {%s} and properties: {%s}",
                name,
                properties,
            )
        except Exception as e:
            logger.error("Error during upsert: {%s}", e)
            raise

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GremlinServerError,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their names.

        Args:
            source_node_id (str): Name of the source node (used as identifier)
            target_node_id (str): Name of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_name = GremlinStorage._fix_name(source_node_id)
        target_node_name = GremlinStorage._fix_name(target_node_id)
        edge_properties = GremlinStorage._convert_properties(edge_data)

        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {source_node_name}).as('source')
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {target_node_name}).as('target')
                 .coalesce(
                      __.select('source').outE('DIRECTED').where(__.inV().as('target')),
                      __.select('source').addE('DIRECTED').to(__.select('target'))
                  )
                  .property('graph', {self.graph_name})
                 {edge_properties}
                 """
        try:
            await self._query(query)
            logger.debug(
                "Upserted edge from {%s} to {%s} with properties: {%s}",
                source_node_name,
                target_node_name,
                edge_properties,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified entity_name

        Args:
            node_id: The entity_name of the node to delete
        """
        entity_name = GremlinStorage._fix_name(node_id)

        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .has('entity_name', {entity_name})
                 .drop()
                 """
        try:
            await self._query(query)
            logger.debug(
                "{%s}: Deleted node with entity_name '%s'",
                inspect.currentframe().f_code.co_name,
                entity_name,
            )
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    async def get_all_labels(self) -> list[str]:
        """
        Get all node entity_names in the graph
        Returns:
            [entity_name1, entity_name2, ...]  # Alphabetically sorted entity_name list
        """
        query = f"""g
                 .V().has('graph', {self.graph_name})
                 .values('entity_name')
                 .dedup()
                 .order()
                 """
        try:
            result = await self._query(query)
            labels = result if result else []
            logger.debug(
                "{%s}: Retrieved %d labels",
                inspect.currentframe().f_code.co_name,
                len(labels),
            )
            return labels
        except Exception as e:
            logger.error(f"Error retrieving labels: {str(e)}")
            return []

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the entity_name includes the specified `node_label`.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).

        Args:
            node_label: Entity name of the starting node
            max_depth: Maximum depth of the subgraph

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        # Get maximum number of graph nodes from environment variable, default is 1000
        MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

        entity_name = GremlinStorage._fix_name(node_label)

        # Handle special case for "*" label
        if node_label == "*":
            # For "*", get all nodes and their edges (limited by MAX_GRAPH_NODES)
            query = f"""g
                     .V().has('graph', {self.graph_name})
                     .limit({MAX_GRAPH_NODES})
                     .elementMap()
                     """
            nodes_result = await self._query(query)

            # Add nodes to result
            for node_data in nodes_result:
                node_id = node_data.get("entity_name", str(node_data.get("id", "")))
                if str(node_id) in seen_nodes:
                    continue

                # Create node with properties
                node_properties = {
                    k: v for k, v in node_data.items() if k not in ["id", "label"]
                }

                result.nodes.append(
                    KnowledgeGraphNode(
                        id=str(node_id),
                        labels=[str(node_id)],
                        properties=node_properties,
                    )
                )
                seen_nodes.add(str(node_id))

            # Get and add edges
            if nodes_result:
                query = f"""g
                         .V().has('graph', {self.graph_name})
                         .limit({MAX_GRAPH_NODES})
                         .outE()
                         .inV().has('graph', {self.graph_name})
                         .limit({MAX_GRAPH_NODES})
                         .path()
                         .by(elementMap())
                         .by(elementMap())
                         .by(elementMap())
                         """
                edges_result = await self._query(query)

                for path in edges_result:
                    if len(path) >= 3:  # source -> edge -> target
                        source = path[0]
                        edge_data = path[1]
                        target = path[2]

                        source_id = source.get("entity_name", str(source.get("id", "")))
                        target_id = target.get("entity_name", str(target.get("id", "")))

                        edge_id = f"{source_id}-{target_id}"
                        if edge_id in seen_edges:
                            continue

                        # Create edge with properties
                        edge_properties = {
                            k: v
                            for k, v in edge_data.items()
                            if k not in ["id", "label"]
                        }

                        result.edges.append(
                            KnowledgeGraphEdge(
                                id=edge_id,
                                type="DIRECTED",
                                source=str(source_id),
                                target=str(target_id),
                                properties=edge_properties,
                            )
                        )
                        seen_edges.add(edge_id)
        else:
            # Search for specific node and get its neighborhood
            query = f"""g
                     .V().has('graph', {self.graph_name})
                     .has('entity_name', {entity_name})
                     .repeat(__.both().simplePath().dedup())
                     .times({max_depth})
                     .emit()
                     .dedup()
                     .limit({MAX_GRAPH_NODES})
                     .elementMap()
                     """
            nodes_result = await self._query(query)

            # Add nodes to result
            for node_data in nodes_result:
                node_id = node_data.get("entity_name", str(node_data.get("id", "")))
                if str(node_id) in seen_nodes:
                    continue

                # Create node with properties
                node_properties = {
                    k: v for k, v in node_data.items() if k not in ["id", "label"]
                }

                result.nodes.append(
                    KnowledgeGraphNode(
                        id=str(node_id),
                        labels=[str(node_id)],
                        properties=node_properties,
                    )
                )
                seen_nodes.add(str(node_id))

            # Get edges between the nodes in the result
            if nodes_result:
                node_ids = [
                    n.get("entity_name", str(n.get("id", ""))) for n in nodes_result
                ]
                node_ids_query = ", ".join(
                    [GremlinStorage._to_value_map(nid) for nid in node_ids]
                )

                query = f"""g
                         .V().has('graph', {self.graph_name})
                         .has('entity_name', within({node_ids_query}))
                         .outE()
                         .where(inV().has('graph', {self.graph_name})
                                .has('entity_name', within({node_ids_query})))
                         .path()
                         .by(elementMap())
                         .by(elementMap())
                         .by(elementMap())
                         """
                edges_result = await self._query(query)

                for path in edges_result:
                    if len(path) >= 3:  # source -> edge -> target
                        source = path[0]
                        edge_data = path[1]
                        target = path[2]

                        source_id = source.get("entity_name", str(source.get("id", "")))
                        target_id = target.get("entity_name", str(target.get("id", "")))

                        edge_id = f"{source_id}-{target_id}"
                        if edge_id in seen_edges:
                            continue

                        # Create edge with properties
                        edge_properties = {
                            k: v
                            for k, v in edge_data.items()
                            if k not in ["id", "label"]
                        }

                        result.edges.append(
                            KnowledgeGraphEdge(
                                id=edge_id,
                                type="DIRECTED",
                                source=str(source_id),
                                target=str(target_id),
                                properties=edge_properties,
                            )
                        )
                        seen_edges.add(edge_id)

        logger.info(
            "Subgraph query successful | Node count: %d | Edge count: %d",
            len(result.nodes),
            len(result.edges),
        )
        return result

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node entity_names to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            entity_name_source = GremlinStorage._fix_name(source)
            entity_name_target = GremlinStorage._fix_name(target)

            query = f"""g
                     .V().has('graph', {self.graph_name})
                     .has('entity_name', {entity_name_source})
                     .outE()
                     .where(inV().has('graph', {self.graph_name})
                            .has('entity_name', {entity_name_target}))
                     .drop()
                     """
            try:
                await self._query(query)
                logger.debug(
                    "{%s}: Deleted edge from '%s' to '%s'",
                    inspect.currentframe().f_code.co_name,
                    entity_name_source,
                    entity_name_target,
                )
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def drop(self) -> dict[str, str]:
        """Drop the storage by removing all nodes and relationships in the graph.

        This function deletes all nodes with the specified graph name property,
        which automatically removes all associated edges.

        Returns:
            dict[str, str]: Status of the operation with keys 'status' and 'message'
        """
        try:
            query = f"""g
                    .V().has('graph', {self.graph_name})
                    .drop()
                    """
            await self._query(query)
            logger.info(f"Successfully dropped all data from graph {self.graph_name}")
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            logger.error(f"Error dropping graph {self.graph_name}: {e}")
            return {"status": "error", "message": str(e)}
