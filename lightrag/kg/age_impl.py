import asyncio
import inspect
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Union, final
import numpy as np
import pipmaster as pm
from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import logger

from ..base import BaseGraphStorage

if sys.platform.startswith("win"):
    import asyncio.windows_events

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if not pm.is_installed("psycopg-pool"):
    pm.install("psycopg-pool")
    pm.install("psycopg[binary,pool]")

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import psycopg
from psycopg.rows import namedtuple_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout


class AGEQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@final
@dataclass
class AGEStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with AGE in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        self._driver_lock = asyncio.Lock()
        DB = os.environ["AGE_POSTGRES_DB"].replace("\\", "\\\\").replace("'", "\\'")
        USER = os.environ["AGE_POSTGRES_USER"].replace("\\", "\\\\").replace("'", "\\'")
        PASSWORD = (
            os.environ["AGE_POSTGRES_PASSWORD"]
            .replace("\\", "\\\\")
            .replace("'", "\\'")
        )
        HOST = os.environ["AGE_POSTGRES_HOST"].replace("\\", "\\\\").replace("'", "\\'")
        PORT = os.environ.get("AGE_POSTGRES_PORT", "8529")
        self.graph_name = namespace or os.environ.get("AGE_GRAPH_NAME", "lightrag")

        connection_string = f"dbname='{DB}' user='{USER}' password='{PASSWORD}' host='{HOST}' port={PORT}"

        self._driver = AsyncConnectionPool(connection_string, open=False)

        return None

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    @staticmethod
    def _record_to_dict(record: NamedTuple) -> Dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record._fields:
            v = getattr(record, k)
            # agtype comes back '{key: value}::type' which must be parsed
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
                if dtype == "vertex":
                    vertex = json.loads(v)
                    vertices[vertex["id"]] = vertex.get("properties")

        # iterate returned fields and parse appropriately
        for k in record._fields:
            v = getattr(record, k)
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
            else:
                dtype = ""

            if dtype == "vertex":
                vertex = json.loads(v)
                field = json.loads(v).get("properties")
                if not field:
                    field = {}
                field["label"] = AGEStorage._decode_graph_label(vertex["label"])
                d[k] = field
            # convert edge from id-label->id by replacing id with node information
            # we only do this if the vertex was also returned in the query
            # this is an attempt to be consistent with neo4j implementation
            elif dtype == "edge":
                edge = json.loads(v)
                d[k] = (
                    vertices.get(edge["start_id"], {}),
                    edge[
                        "label"
                    ],  # we don't use decode_graph_label(), since edge label is always "DIRECTED"
                    vertices.get(edge["end_id"], {}),
                )
            else:
                d[k] = json.loads(v) if isinstance(v, str) else v

        return d

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], _id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def _encode_graph_label(label: str) -> str:
        """
        Since AGE suports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            label (str): the original label

        Returns:
            str: the encoded label
        """
        return "x" + label.encode().hex()

    @staticmethod
    def _decode_graph_label(encoded_label: str) -> str:
        """
        Since AGE suports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            encoded_label (str): the encoded label

        Returns:
            str: the decoded label
        """
        return bytes.fromhex(encoded_label.removeprefix("x")).decode()

    @staticmethod
    def _get_col_name(field: str, idx: int) -> str:
        """
        Convert a cypher return field to a pgsql select field
        If possible keep the cypher column name, but create a generic name if necessary

        Args:
            field (str): a return field from a cypher query to be formatted for pgsql
            idx (int): the position of the field in the return statement

        Returns:
            str: the field to be used in the pgsql select statement
        """
        # remove white space
        field = field.strip()
        # if an alias is provided for the field, use it
        if " as " in field:
            return field.split(" as ")[-1].strip()
        # if the return value is an unnamed primitive, give it a generic name
        if field.isnumeric() or field in ("true", "false", "null"):
            return f"column_{idx}"
        # otherwise return the value stripping out some common special chars
        return field.replace("(", "_").replace(")", "")

    @staticmethod
    def _wrap_query(query: str, graph_name: str, **params: str) -> str:
        """
        Convert a cypher query to an Apache Age compatible
        sql query by wrapping the cypher query in ag_catalog.cypher,
        casting results to agtype and building a select statement

        Args:
            query (str): a valid cypher query
            graph_name (str): the name of the graph to query
            params (dict): parameters for the query

        Returns:
            str: an equivalent pgsql query
        """

        # pgsql template
        template = """SELECT {projection} FROM ag_catalog.cypher('{graph_name}', $$
            {query}
        $$) AS ({fields});"""

        # if there are any returned fields they must be added to the pgsql query
        if "return" in query.lower():
            # parse return statement to identify returned fields
            fields = (
                query.lower()
                .split("return")[-1]
                .split("distinct")[-1]
                .split("order by")[0]
                .split("skip")[0]
                .split("limit")[0]
                .split(",")
            )

            # raise exception if RETURN * is found as we can't resolve the fields
            if "*" in [x.strip() for x in fields]:
                raise ValueError(
                    "AGE graph does not support 'RETURN *'"
                    + " statements in Cypher queries"
                )

            # get pgsql formatted field names
            fields = [
                AGEStorage._get_col_name(field, idx) for idx, field in enumerate(fields)
            ]

            # build resulting pgsql relation
            fields_str = ", ".join(
                [field.split(".")[-1] + " agtype" for field in fields]
            )

        # if no return statement we still need to return a single field of type agtype
        else:
            fields_str = "a agtype"

        select_str = "*"

        return template.format(
            graph_name=graph_name,
            query=query.format(**params),
            fields=fields_str,
            projection=select_str,
        )

    async def _query(self, query: str, **params: str) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        # convert cypher query to pgsql/age query
        wrapped_query = self._wrap_query(query, self.graph_name, **params)

        await self._driver.open()

        # create graph if it doesn't exist
        async with self._get_pool_connection() as conn:
            async with conn.cursor() as curs:
                try:
                    await curs.execute('SET search_path = ag_catalog, "$user", public')
                    await curs.execute(f"SELECT create_graph('{self.graph_name}')")
                    await conn.commit()
                except (
                    psycopg.errors.InvalidSchemaName,
                    psycopg.errors.UniqueViolation,
                ):
                    await conn.rollback()

        # execute the query, rolling back on an error
        async with self._get_pool_connection() as conn:
            async with conn.cursor(row_factory=namedtuple_row) as curs:
                try:
                    await curs.execute('SET search_path = ag_catalog, "$user", public')
                    await curs.execute(wrapped_query)
                    await conn.commit()
                except psycopg.Error as e:
                    await conn.rollback()
                    raise AGEQueryException(
                        {
                            "message": f"Error executing graph query: {query.format(**params)}",
                            "detail": str(e),
                        }
                    ) from e

                data = await curs.fetchall()
                if data is None:
                    result = []
                # decode records
                else:
                    result = [AGEStorage._record_to_dict(d) for d in data]

                return result

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('"')

        query = """
                MATCH (n:`{label}`) RETURN count(n) > 0 AS node_exists
                """
        params = {"label": AGEStorage._encode_graph_label(entity_name_label)}
        single_result = (await self._query(query, **params))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query.format(**params),
            single_result["node_exists"],
        )

        return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        query = """
                MATCH (a:`{src_label}`)-[r]-(b:`{tgt_label}`)
                RETURN COUNT(r) > 0 AS edge_exists
                """
        params = {
            "src_label": AGEStorage._encode_graph_label(entity_name_label_source),
            "tgt_label": AGEStorage._encode_graph_label(entity_name_label_target),
        }
        single_result = (await self._query(query, **params))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query.format(**params),
            single_result["edge_exists"],
        )
        return single_result["edge_exists"]

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        entity_name_label = node_id.strip('"')
        query = """
                MATCH (n:`{label}`) RETURN n
                """
        params = {"label": AGEStorage._encode_graph_label(entity_name_label)}
        record = await self._query(query, **params)
        if record:
            node = record[0]
            node_dict = node["n"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query.format(**params),
                node_dict,
            )
            return node_dict
        return None

    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('"')

        query = """
                MATCH (n:`{label}`)-[]->(x)
                RETURN count(x) AS total_edge_count
                """
        params = {"label": AGEStorage._encode_graph_label(entity_name_label)}
        record = (await self._query(query, **params))[0]
        if record:
            edge_count = int(record["total_edge_count"])
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query.format(**params),
                edge_count,
            )
            return edge_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('"')
        entity_name_label_target = tgt_id.strip('"')
        src_degree = await self.node_degree(entity_name_label_source)
        trg_degree = await self.node_degree(entity_name_label_target)

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
        entity_name_label_source = source_node_id.strip('"')
        entity_name_label_target = target_node_id.strip('"')

        query = """
                MATCH (a:`{src_label}`)-[r]->(b:`{tgt_label}`)
                RETURN properties(r) as edge_properties
                LIMIT 1
                """
        params = {
            "src_label": AGEStorage._encode_graph_label(entity_name_label_source),
            "tgt_label": AGEStorage._encode_graph_label(entity_name_label_target),
        }
        record = await self._query(query, **params)
        if record and record[0] and record[0]["edge_properties"]:
            result = record[0]["edge_properties"]
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query.format(**params),
                result,
            )
            return result

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        node_label = source_node_id.strip('"')

        query = """
                MATCH (n:`{label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected
                """
        params = {"label": AGEStorage._encode_graph_label(node_label)}
        results = await self._query(query, **params)
        edges = []
        for record in results:
            source_node = record["n"] if record["n"] else None
            connected_node = record["connected"] if record["connected"] else None

            source_label = (
                source_node["label"] if source_node and source_node["label"] else None
            )
            target_label = (
                connected_node["label"]
                if connected_node and connected_node["label"]
                else None
            )

            if source_label and target_label:
                edges.append((source_label, target_label))

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AGEQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the AGE database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = node_id.strip('"')
        properties = node_data

        query = """
                MERGE (n:`{label}`)
                SET n += {properties}
                """
        params = {
            "label": AGEStorage._encode_graph_label(label),
            "properties": AGEStorage._format_properties(properties),
        }
        try:
            await self._query(query, **params)
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
        retry=retry_if_exception_type((AGEQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        source_node_label = source_node_id.strip('"')
        target_node_label = target_node_id.strip('"')
        edge_properties = edge_data

        query = """
                MATCH (source:`{src_label}`)
                WITH source
                MATCH (target:`{tgt_label}`)
                MERGE (source)-[r:DIRECTED]->(target)
                SET r += {properties}
                RETURN r
                """
        params = {
            "src_label": AGEStorage._encode_graph_label(source_node_label),
            "tgt_label": AGEStorage._encode_graph_label(target_node_label),
            "properties": AGEStorage._format_properties(edge_properties),
        }
        try:
            await self._query(query, **params)
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

    @asynccontextmanager
    async def _get_pool_connection(self, timeout: Optional[float] = None):
        """Workaround for a psycopg_pool bug"""

        try:
            connection = await self._driver.getconn(timeout=timeout)
        except PoolTimeout:
            await self._driver._add_connection(None)  # workaround...
            connection = await self._driver.getconn(timeout=timeout)

        try:
            async with connection:
                yield connection
        finally:
            await self._driver.putconn(connection)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """
        entity_name_label = node_id.strip('"')

        query = """
        MATCH (n:`{label}`)
        DETACH DELETE n
        """
        params = {"label": AGEStorage._encode_graph_label(entity_name_label)}
        try:
            await self._query(query, **params)
            logger.debug(f"Deleted node with label '{entity_name_label}'")
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:
            entity_name_label_source = source.strip('"')
            entity_name_label_target = target.strip('"')

            query = """
            MATCH (source:`{src_label}`)-[r]->(target:`{tgt_label}`)
            DELETE r
            """
            params = {
                "src_label": AGEStorage._encode_graph_label(entity_name_label_source),
                "tgt_label": AGEStorage._encode_graph_label(entity_name_label_target),
            }
            try:
                await self._query(query, **params)
                logger.debug(
                    f"Deleted edge from '{entity_name_label_source}' to '{entity_name_label_target}'"
                )
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        """Embed nodes using the specified algorithm

        Args:
            algorithm: Name of the embedding algorithm

        Returns:
            tuple: (embedding matrix, list of node identifiers)
        """
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def get_all_labels(self) -> list[str]:
        """Get all node labels in the database

        Returns:
            ["label1", "label2", ...]  # Alphabetically sorted label list
        """
        query = """
        MATCH (n)
        RETURN DISTINCT labels(n) AS node_labels
        """
        results = await self._query(query)

        all_labels = []
        for record in results:
            if record and "node_labels" in record:
                for label in record["node_labels"]:
                    if label:
                        # Decode label
                        decoded_label = AGEStorage._decode_graph_label(label)
                        all_labels.append(decoded_label)

        # Remove duplicates and sort
        return sorted(list(set(all_labels)))

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified 'node_label'.
        Maximum number of nodes is constrained by the environment variable 'MAX_GRAPH_NODES' (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Label matching nodes take precedence (nodes containing the specified label string)
            2. Followed by nodes directly connected to the matching nodes
            3. Finally, the degree of the nodes

        Args:
            node_label: String to match in node labels (will match any node containing this string in its label)
            max_depth: Maximum depth of the graph. Defaults to 5.

        Returns:
            KnowledgeGraph: Complete connected subgraph for specified node
        """
        max_graph_nodes = int(os.getenv("MAX_GRAPH_NODES", 1000))
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        # Handle special case for "*" label
        if node_label == "*":
            # Query all nodes and sort by degree
            query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT {max_nodes}
            RETURN n, degree
            """
            params = {"max_nodes": max_graph_nodes}
            nodes_result = await self._query(query, **params)

            # Add nodes to result
            node_ids = []
            for record in nodes_result:
                if "n" in record:
                    node = record["n"]
                    node_id = str(node.get("id", ""))
                    if node_id not in seen_nodes:
                        node_properties = {k: v for k, v in node.items()}
                        node_label = node.get("label", "")
                        result.nodes.append(
                            KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_label],
                                properties=node_properties,
                            )
                        )
                        seen_nodes.add(node_id)
                        node_ids.append(node_id)

            # Query edges between these nodes
            if node_ids:
                edges_query = """
                MATCH (a)-[r]->(b)
                WHERE a.id IN {node_ids} AND b.id IN {node_ids}
                RETURN a, r, b
                """
                edges_params = {"node_ids": node_ids}
                edges_result = await self._query(edges_query, **edges_params)

                # Add edges to result
                for record in edges_result:
                    if "r" in record and "a" in record and "b" in record:
                        source = record["a"].get("id", "")
                        target = record["b"].get("id", "")
                        edge_id = f"{source}-{target}"
                        if edge_id not in seen_edges:
                            edge_properties = {k: v for k, v in record["r"].items()}
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=edge_id,
                                    type="DIRECTED",
                                    source=source,
                                    target=target,
                                    properties=edge_properties,
                                )
                            )
                            seen_edges.add(edge_id)
        else:
            # For specific label, use partial matching
            entity_name_label = node_label.strip('"')
            encoded_label = AGEStorage._encode_graph_label(entity_name_label)

            # Find matching start nodes
            start_query = """
            MATCH (n:`{label}`)
            RETURN n
            """
            start_params = {"label": encoded_label}
            start_nodes = await self._query(start_query, **start_params)

            if not start_nodes:
                logger.warning(f"No nodes found with label '{entity_name_label}'!")
                return result

            # Traverse graph from each start node
            for start_node_record in start_nodes:
                if "n" in start_node_record:
                    # Use BFS to traverse graph
                    query = """
                    MATCH (start:`{label}`)
                    CALL {
                        MATCH path = (start)-[*0..{max_depth}]->(n)
                        RETURN nodes(path) AS path_nodes, relationships(path) AS path_rels
                    }
                    RETURN DISTINCT path_nodes, path_rels
                    """
                    params = {"label": encoded_label, "max_depth": max_depth}
                    results = await self._query(query, **params)

                    # Extract nodes and edges from results
                    for record in results:
                        if "path_nodes" in record:
                            # Process nodes
                            for node in record["path_nodes"]:
                                node_id = str(node.get("id", ""))
                                if (
                                    node_id not in seen_nodes
                                    and len(seen_nodes) < max_graph_nodes
                                ):
                                    node_properties = {k: v for k, v in node.items()}
                                    node_label = node.get("label", "")
                                    result.nodes.append(
                                        KnowledgeGraphNode(
                                            id=node_id,
                                            labels=[node_label],
                                            properties=node_properties,
                                        )
                                    )
                                    seen_nodes.add(node_id)

                        if "path_rels" in record:
                            # Process edges
                            for rel in record["path_rels"]:
                                source = str(rel.get("start_id", ""))
                                target = str(rel.get("end_id", ""))
                                edge_id = f"{source}-{target}"
                                if edge_id not in seen_edges:
                                    edge_properties = {k: v for k, v in rel.items()}
                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type=rel.get("label", "DIRECTED"),
                                            source=source,
                                            target=target,
                                            properties=edge_properties,
                                        )
                                    )
                                    seen_edges.add(edge_id)

        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def index_done_callback(self) -> None:
        # AGES handles persistence automatically
        pass
