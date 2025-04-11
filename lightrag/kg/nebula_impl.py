import asyncio
import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, final
import numpy as np
import configparser

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
import pipmaster as pm
if not pm.is_installed("nebula3-python"):
    pm.install("nebula3-python")

from nebula3.gclient.net import ConnectionPool  # type: ignore
from nebula3.Config import Config  # type: ignore
from nebula3.Exception import IOErrorException, AuthFailedException  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# Set nebula logger level to ERROR to suppress warning logs
logging.getLogger("nebula3").setLevel(logging.ERROR)


@final
@dataclass
class NebulaStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        global config
        self._connection_pool = None
        self._connection_pool_lock = asyncio.Lock()
        self._space_name = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)

        # Nebula configuration
        HOSTS = os.environ.get(
            "NEBULA_HOSTS", config.get("nebula", "hosts", fallback="127.0.0.1:9669")
        ).split(":")
        self.USERNAME = os.environ.get(
            "NEBULA_USER", config.get("nebula", "user", fallback="root")
        )
        self.PASSWORD = os.environ.get(
            "NEBULA_PASSWORD", config.get("nebula", "password", fallback="nebula")
        )
        POOL_SIZE = int(
            os.environ.get(
                "NEBULA_POOL_SIZE",
                config.get("nebula", "pool_size", fallback=10),
            )
        )
        # TIMEOUT = int(
        #     os.environ.get(
        #         "NEBULA_TIMEOUT",
        #         config.get("nebula", "timeout", fallback=30000),
        #     )
        # )

        # Initialize connection pool
       
        config.max_connection_pool_size = POOL_SIZE
        self._connection_pool = ConnectionPool()
        
        # Try to connect to Nebula
        try:
            self._connection_pool.init([HOSTS])
            
            # Check if space exists
            with self._connection_pool.session_context(self.USERNAME, self.PASSWORD) as session:
                session.execute(f"USE {self._space_name}")
                result = session.execute("SHOW SPACES").as_data_frame()
                spaces = [row[0] for _,row in result.iterrows()]
                
                if self._space_name not in spaces:
                    logger.info(f"Space {self._space_name} not found, creating...")
                    # Create space with default schema
                    session.execute(f"CREATE SPACE IF NOT EXISTS {self._space_name} (vid_type=FIXED_STRING(256))")
                    session.execute(f"USE {self._space_name}")                    # Create tag and edge types
                   
                    logger.info(f"Space {self._space_name} created successfully")
                else:
                    session.execute(f"USE {self._space_name}")
                    logger.info(f"Connected to space {self._space_name}")
                session.execute(" CREATE TAG IF NOT EXISTS base (    entity_id string,    entity_type string,    description string,    source_id string,    file_path string);")
                session.execute("CREATE EDGE IF NOT EXISTS DIRECTED(weight float, description string,keywords string ,source_id string ,file_path string) ")
                    
        except (IOErrorException, AuthFailedException) as e:
            logger.error(f"Failed to connect to Nebula Graph: {str(e)}")
            raise

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        """Close the Nebula connection pool"""
        if self._connection_pool:
            self._connection_pool.close()
            self._connection_pool = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure connection pool is closed when context manager exits"""
        await self.close()

    async def index_done_callback(self) -> None:
        # Nebula handles persistence automatically
        pass

    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given entity_id exists in the database
        """
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query = f""" MATCH (n) 
                        WHERE id(n) == '{node_id}' return count(n) as icount """
                result = session.execute(query)
                return result.is_succeeded() and len(result.as_data_frame())> 0

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes
        """
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query =  f''' MATCH (src)-[r]-(neighbor)
                       WHERE id(src) =='{source_node_id}' AND id(neighbor)=='{target_node_id}'
                      RETURN count(r) as icount
                   '''   
                result = session.execute(query)
                return result.is_succeeded() and len(result.as_data_frame()) > 0

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its entity_id"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query = f""" MATCH (n) 
                        WHERE id(n) == '{node_id}' 
                        RETURN id(n) AS id, properties(n) AS props """
                try:                
                    result = session.execute(query).as_data_frame()
                   
                        
                    if len(result) > 1:
                        logger.warning(
                            f"Multiple nodes found with label '{node_id}'. Using first node."
                        )
                    if len(result) ==1:
                        node_dict = result['props'][0]
                        # Remove base label from labels list if it exists
                        if "labels" in node_dict:
                            node_dict["labels"] = [
                                label
                                for label in node_dict["labels"]
                                if label != "base"
                            ]
                        logger.debug(f"nebula query node {query} return: {node_dict}")
                        return node_dict
                    return None                    
                except Exception as e:
                    logger.error(f"Error getting node for {node_id}: {str(e)}")
                    raise

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query = f" MATCH (v:base)-[r]-(n) WHERE id(v) == '{node_id}' RETURN count(r)  as degree"
                result = session.execute(query).as_data_frame()
                return  int(result['degree'][0]) if len(result)>0 else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes"""
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return src_degree + trg_degree

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query =  f''' MATCH (src)-[r]-(neighbor)
                       WHERE id(src) =='{source_node_id}' AND id(neighbor)=='{target_node_id}'
                      RETURN properties(r) as edge_properties
                   '''   
                result = session.execute(query).as_data_frame()

                if len(result) > 1:
                    logger.warning(
                        f"Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                    )
                
                try:
                    edge_result = result['edge_properties'][0]
                    logger.debug(f"Result: {edge_result}")
                    # Ensure required keys exist with defaults
                    required_keys = {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
                    for key, default_value in required_keys.items():
                        if key not in edge_result:
                            edge_result[key] = default_value
                            logger.warning(
                                f"Edge between {source_node_id} and {target_node_id} "
                                f"missing {key}, using default: {default_value}"
                            )

                    logger.debug(
                        f"{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_result}"
                    )
                    return edge_result
                except (KeyError, TypeError, ValueError) as e:
                    logger.error(
                        f"Error processing edge properties between {source_node_id} "
                        f"and {target_node_id}: {str(e)}"
                    )
                    # Return default edge properties on error
                    return {
                        "weight": 0.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }

                   


    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found

        Raises:
            ValueError: If source_node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                try:
                    session.execute(f"USE {self._space_name}")
                
                    query = """MATCH (n:base)-[r]-(connected:base)
                              WHERE  id(n)=='{source_node_id}' and connected.entity_id IS NOT NULL 
                            RETURN n, r, connected"""
                    results =  session.execute(query).as_data_frame()

                    edges = []
                    for _,record in results.iterrows():
                        source_node = record["n"]
                        connected_node = record["connected"]

                        # Skip if either node is None
                        if not source_node or not connected_node:
                            continue

                        source_label = (
                            source_node.get("entity_id")
                            if source_node.get("entity_id")
                            else None
                        )
                        target_label = (
                            connected_node.get("entity_id")
                            if connected_node.get("entity_id")
                            else None
                        )

                        if source_label and target_label:
                            edges.append((source_label, target_label))

                   
                    return edges
                except Exception as e:
                    logger.error(
                        f"Error getting edges for node {source_node_id}: {str(e)}"
                    )                   
                    raise
        except Exception as e:
            logger.error(f"Error in get_node_edges for {source_node_id}: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in Nebula"""
        if "entity_id" not in node_data:
            raise ValueError("Node properties must contain an 'entity_id' field")
        fields=  ", ".join(f"{k}" for k, v in node_data.items())   
        properties = ", ".join(f"{repr(v)}" for k, v in node_data.items())
        
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                
                query = (
                    f"INSERT VERTEX IF NOT EXISTS base({fields}) "
                    f"VALUES '{node_id}':({properties})"
                )
                session.execute(query)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert an edge between two nodes"""
        fields=", ".join(f"{k}" for k, v in edge_data.items())
        properties = ", ".join(f"{repr(v)}" for k, v in edge_data.items())
        
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")                
                # Nebula requires edges to have a rank (0 by default)
                query = (
                    f"INSERT EDGE IF NOT EXISTS DIRECTED({fields}) "
                    f"VALUES '{source_node_id}' -> '{target_node_id}'@0:({properties})"
                )
                session.execute(query)

    async def _node2vec_embed(self):
        print("Implemented but never called.")

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                
                if node_label == "*":
                    # Get all nodes with minimum degree
                    query = (
                        f"MATCH (n:base) "
                        f"WHERE id(n) >= '' "
                        f"RETURN id(n) AS id, properties(n) AS props "
                        f"LIMIT {MAX_GRAPH_NODES}"
                    )
                else:
                    # Start from specific node and traverse
                    query = f"""
                        MATCH (n)
                        WHERE id(n) {'CONTAINS' if inclusive else '=='} "{node_label}"
                        RETURN id(n) AS id, properties(n) AS props 
                        LIMIT {MAX_GRAPH_NODES}
                        """
                
                try:
                    result_set = session.execute(query).as_data_frame()
                    
                   
                    matched_ids=[]    # Process nodes from direct query
                    for _,row in result_set.iterrows():
                        node_id = row['id']
                        props = row['props']
                        # props = {k: v for k, v in props}
                        matched_ids.append(node_id)
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[node_id],
                                    properties=props,
                                )
                            )
                            seen_nodes.add(node_id)
                   
                            
                    
                      
                    neighbors_query = f"""
                    MATCH (src)-[e*1..{max_depth}]-(neighbor)
                    WHERE id(src) IN {matched_ids}
                    RETURN DISTINCT id(neighbor) AS id,properties(neighbor) AS props 
                    LIMIT {MAX_GRAPH_NODES - len(matched_ids)}
                    """
                    
                    result_set = session.execute(neighbors_query).as_data_frame()
                    
                    for _,row in result_set.iterrows():
                        node_id = row['id']
                        
                        if node_id not in seen_nodes:
                            props = row['props']                                
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[node_id],
                                    properties=props,
                                )
                            )
                            seen_nodes.add(node_id)
                    
                        # Process edges
                        
                    edge_query=f'''
                       MATCH (src)-[e]->(neighbor)
                       WHERE id(src) IN {seen_nodes} AND id(neighbor) IN {seen_nodes}
                        RETURN DISTINCT id(src) AS srcId,id(neighbor) AS destId  ,properties(e) as edgeProps ,type(e) as edgeType
                       
                    
                    '''
                    result_set = session.execute(edge_query).as_data_frame() 
                    for _,row in result_set.iterrows():       
                            edge_id = f"{row['srcId']}-{row['destId']}-{row['edgeType']}"
                            if edge_id not in seen_edges:
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type=row['edgeType'],
                                        source=row['srcId'],
                                        target=row['destId'],
                                        properties=row['edgeProps'],
                                    )
                                )
                                seen_edges.add(edge_id)
                    
                        
                       
                       
                    
                    logger.info(
                        f"Graph query return: {len(result.nodes)} nodes, {len(result.edges)} edges"
                    )
                    
                except Exception as e:
                    logger.error(f"Error executing graph query: {str(e)}")
                    raise

        return result

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node entity_ids in the database
        """
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query = "MATCH (n:base) RETURN id(n) AS label ORDER BY label"
                result = session.execute(query).as_data_frame()
                return [row[0] for _,row in result.iterrows()]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified entity_id"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                query = f"DELETE VERTEX '{node_id}' WITH EDGE"
                session.execute(query)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)),
    )
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                for node in nodes:
                    query = f"DELETE VERTEX '{node}' WITH EDGE"
                    session.execute(query)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((IOErrorException,)),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges"""
        async with self._connection_pool_lock:
            with self._connection_pool.session_context(
                self.USERNAME, self.PASSWORD
            ) as session:
                session.execute(f"USE {self._space_name}")
                for source, target in edges:
                    # Nebula requires edge type to delete specific edges
                    query = (
                        f"DELETE EDGE DIRECTED '{source}' -> '{target}'@0"
                    )
                    session.execute(query)

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        raise NotImplementedError