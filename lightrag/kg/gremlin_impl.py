import asyncio
import inspect
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from gremlin_python.driver import client, serializer
from gremlin_python.driver.aiohttp.transport import AiohttpTransport
from gremlin_python.driver.protocol import GremlinServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import logger

from ..base import BaseGraphStorage


@dataclass
class GremlinStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with Gremlin in production")

    # Will use this to make sure single quotes are properly escaped
    escape_rx = re.compile(r"(^|[^\\])((\\\\)*\\)\\'")

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
        GRAPH = GremlinStorage.escape_rx.sub(
            r"\1\2'",
            os.environ["GREMLIN_GRAPH"].replace("'", r"\'"),
        )

        self.traverse_source_name = SOURCE
        self.graph_name = GRAPH

        self._driver = client.Client(
            f"ws://{HOST}:{PORT}/gremlin",
            SOURCE,
            username=USER,
            password=PASSWORD,
            message_serializer=serializer.GraphSONSerializersV3d0(),
            transport_factory=lambda: AiohttpTransport(call_from_event_loop=True),
        )

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            self._driver.close()

    async def index_done_callback(self):
        print("KG successfully indexed.")

    @staticmethod
    def _to_value_map(value: Any) -> str:
        """Dump Python dict as Gremlin valueMap"""
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
            prop_name = GremlinStorage.escape_rx.sub(r"\1\2'", k.replace("'", r"\'"))
            props.append(f".property('{prop_name}', {GremlinStorage._to_value_map(v)})")
        return "".join(props)

    @staticmethod
    def _fix_label(label: str) -> str:
        """Strip double quotes and make sure single quotes are escaped"""
        label = label.strip('"').replace("'", r"\'")
        label = GremlinStorage.escape_rx.sub(r"\1\2'", label)

        return label

    async def _query(self, query: str) -> List[Dict[str, Any]]:
        """
        Query the Gremlin graph

        Args:
            query (str): a query to be executed

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """

        result = list(await asyncio.wrap_future(self._driver.submit_async(query)))

        return result

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = GremlinStorage._fix_label(node_id)

        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label}')
                 .limit(1)
                 .hasNext()
                 """
        result = await self._query(query)
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            result[0][0],
        )

        return result[0][0]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = GremlinStorage._fix_label(source_node_id)
        entity_name_label_target = GremlinStorage._fix_label(target_node_id)

        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label_source}')
                 .bothE()
                 .otherV().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label_target}')
                 .limit(1)
                 .hasNext()
                 """
        result = await self._query(query)
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            result[0][0],
        )

        return result[0][0]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        entity_name_label = GremlinStorage._fix_label(node_id)
        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label}')
                 .limit(1)
                 .project('properties')
                    .by(elementMap())
                 """
        result = await self._query(query)
        if result:
            node = result[0][0]
            node_dict = node["properties"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query.format,
                node_dict,
            )
            return node_dict

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = GremlinStorage._fix_label(node_id)
        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label}')
                 .outE()
                 .inV().has('graph', '{self.graph_name}')
                 .count()
                 .project('total_edge_count')
                    .by()
                 """
        result = await self._query(query)
        edge_count = result[0][0]["total_edge_count"]

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
    ) -> Union[dict, None]:
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes

        Returns:
            dict|None: Dict of found edge properties, or None of not found
        """
        entity_name_label_source = GremlinStorage._fix_label(source_node_id)
        entity_name_label_target = GremlinStorage._fix_label(target_node_id)
        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label_source}')
                 .outE()
                 .inV().has('graph', '{self.graph_name}')
                 .hasLabel('{entity_name_label_target}')
                 .limit(1)
                 .project('edge_properties')
                 .by(__.bothE().elementMap())
                 """
        result = await self._query(query)
        if result:
            edge_properties = result[0][0]["edge_properties"]
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_properties,
            )
            return edge_properties

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of tuples containing edge sources and targets
        """
        node_label = GremlinStorage._fix_label(source_node_id)
        query1 = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{node_label}')
                 .out().has('graph', '{self.graph_name}')
                 .project('connected_label')
                    .by(__.label())
                 """
        query2 = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .as('connected')
                 .out().has('graph', '{self.graph_name}')
                 .hasLabel('{node_label}')
                 .project('connected_label')
                    .by(__.select('connected').label())
                 """
        result1, result2 = await asyncio.gather(
            self._query(query1), self._query(query2)
        )
        edges1 = (
            [(node_label, res["connected_label"]) for res in result1[0]]
            if result1
            else []
        )
        edges2 = (
            [(res["connected_label"], node_label) for res in result2[0]]
            if result2
            else []
        )

        return edges1 + edges2

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GremlinServerError,)),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the Gremlin graph.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = GremlinStorage._fix_label(node_id)
        properties = GremlinStorage._convert_properties(node_data)

        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{label}').fold()
                 .coalesce(
                     unfold(),
                     addV('{label}'))
                 .property('graph', '{self.graph_name}')
                 {properties}
                 """

        try:
            await self._query(query)
            logger.debug(
                "Upserted node with label '{%s}' and properties: {%s}",
                label,
                properties,
            )
        except Exception as e:
            logger.error("Error during upsert: {%s}", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GremlinServerError,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = GremlinStorage._fix_label(source_node_id)
        target_node_label = GremlinStorage._fix_label(target_node_id)
        edge_properties = GremlinStorage._convert_properties(edge_data)

        query = f"""
                 {self.traverse_source_name}
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{source_node_label}').as('source')
                 .V().has('graph', '{self.graph_name}')
                 .hasLabel('{target_node_label}').as('target')
                 .coalesce(
                    select('source').outE('DIRECTED').where(inV().as('target')),
                    select('source').addE('DIRECTED').to(select('target'))
                 )
                 .property('graph', '{self.graph_name}')
                 {edge_properties}
                 """
        try:
            await self._query(query)
            logger.debug(
                "Upserted edge from '{%s}' to '{%s}' with properties: {%s}",
                source_node_label,
                target_node_label,
                edge_properties,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")
