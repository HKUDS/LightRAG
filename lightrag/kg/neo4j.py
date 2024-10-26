import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import load_json, logger, write_json
from ..base import (
    BaseGraphStorage
)
from neo4j import GraphDatabase
# Replace with your actual URI, username, and password
URI = "neo4j://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "your_password"
# Create a driver object


@dataclass
class GraphStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    def __post_init__(self):
        # self._graph = preloaded_graph or nx.Graph()
        self._driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        print ("KG successfully indexed.")
    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id
        with self._driver.session() as session:  
            return session.read_transaction(self._check_node_exists, entity_name_label)

        @staticmethod  
        def _check_node_exists(tx, label):  
            query = f"MATCH (n:{label}) RETURN count(n) > 0 AS node_exists"  
            result = tx.run(query)  
            return result.single()["node_exists"]
        
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id
        entity_name_label_target = target_node_id
        #hard code relaitionship type 
        with self._driver.session() as session:  
                result = session.read_transaction(self._check_edge_existence, entity_name_label_source, entity_name_label_target)  
                return result

        @staticmethod  
        def _check_edge_existence(tx, label1, label2):  
            query = (  
                f"MATCH (a:{label1})-[r]-(b:{label2}) "  
                "RETURN COUNT(r) > 0 AS edgeExists"  
            )  
            result = tx.run(query)  
            return result.single()["edgeExists"]
        def close(self):  
            self._driver.close()      
        


    async def get_node(self, node_id: str) -> Union[dict, None]:
        entity_name_label = node_id
        with self._driver.session() as session:  
            result = session.run("MATCH (n:{entity_name_label}) RETURN n".format(entity_name_label=entity_name_label))
            for record in result:
                return record["n"]
            


    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id
        with self._driver.session()  as session:
            degree = self._find_node_degree(session, entity_name_label)
            return degree

        @staticmethod  
        def _find_node_degree(session, label):  
            with session.begin_transaction() as tx:  
                result = tx.run("MATCH (n:`{label}`) RETURN n, size((n)--()) AS degree".format(label=label))  
                record = result.single()  
                if record:  
                    return record["degree"]  
                else:  
                    return None


    # degree = session.read_transaction(get_edge_degree, 1, 2)
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name__label_source = src_id
        entity_name_label_target = tgt_id
        with self._driver.session()  as session:
            result = session.run(
                """MATCH (n1:{node_label1})-[r]-(n2:{node_label2})
                RETURN count(r) AS degree"""
                .format(node_label1=node_label1, node_label2=node_label2)
            )        
            record = result.single()        
            return record["degree"]
    
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        entity_name__label_source = source_node_id
        entity_name_label_target = target_node_id
        """
        Find all edges between nodes of two given labels
        
        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes
            
        Returns:
            list: List of all relationships/edges found
        """
        with self._driver.session() as session:
            query = f"""
            MATCH (source:{entity_name__label_source})-[r]-(target:{entity_name_label_target})
            RETURN r
            """
            
            result = session.run(query)
            return [record["r"] for record in result]


#upsert_node
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        label = node_id
        properties = node_data
        """
        Upsert a node with the given label and properties within a transaction.
        If a node with the same label exists, it will:
        - Update existing properties with new values
        - Add new properties that don't exist
        If no node exists, creates a new node with all properties.
        
        Args:
            label: The node label to search for and apply
            properties: Dictionary of node properties
            
        Returns:
            Dictionary containing the node's properties after upsert, or None if operation fails
        """
        with self._driver.session() as session:
            # Execute the upsert within a transaction
            result = session.execute_write(
                self._do_upsert,
                label,
                properties
            )
            return result
    

        @staticmethod
        def _do_upsert(tx: Transaction, label: str, properties: Dict[str, Any]):
            """
            Static method to perform the actual upsert operation within a transaction
            
            Args:
                tx: Neo4j transaction object
                label: The node label to search for and apply
                properties: Dictionary of node properties
                
            Returns:
                Dictionary containing the node's properties after upsert, or None if operation fails
            """
            # Create the dynamic property string for SET clause
            property_string = ", ".join([
                f"n.{key} = ${key}" 
                for key in properties.keys()
            ])
            
            # Cypher query that either matches existing node or creates new one
            query = f"""
            MATCH (n:{label})
            WITH n LIMIT 1
            CALL {{
                WITH n
                WHERE n IS NOT NULL
                SET {property_string}
                RETURN n
                UNION
                WITH n
                WHERE n IS NULL
                CREATE (n:{label})
                SET {property_string}
                RETURN n
            }}
            RETURN n
            """
        
        # Execute the query with properties as parameters
        result = tx.run(query, properties)
        record = result.single()
        
        if record:
            return dict(record["n"])
        return None
                
   

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        source_node_label = source_node_id
        target_node_label = target_node_id
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        
        Args:
            source_node_label (str): Label of the source node (used as identifier)
            target_node_label (str): Label of the target node (used as identifier)
            edge_properties (dict): Dictionary of properties to set on the edge
        """
        with self._driver.session() as session:
            session.execute_write(
                self._do_upsert_edge,
                source_node_label,
                target_node_label,
                edge_data
            )

        @staticmethod
        def _do_upsert_edge(tx, source_node_label: str, target_node_label: str, edge_properties: Dict[str, Any]) -> None:
            """
            Static method to perform the edge upsert within a transaction.
            
            The query will:
            1. Match the source and target nodes by their labels
            2. Merge the DIRECTED relationship
            3. Set all properties on the relationship, updating existing ones and adding new ones
            """
            # Convert edge properties to Cypher parameter string
            props_string = ", ".join(f"r.{key} = ${key}" for key in edge_properties.keys())
            
            query = """
            MATCH (source)
            WHERE source.label = $source_node_label
            MATCH (target)
            WHERE target.label = $target_node_label
            MERGE (source)-[r:DIRECTED]->(target)
            SET {}
            """.format(props_string)

            # Prepare parameters dictionary
            params = {
                "source_node_label": source_node_label,
                "target_node_label": target_node_label,
                **edge_properties
            }
            
            # Execute the query
            tx.run(query, params)


    async def _node2vec_embed(self):
        # async def _node2vec_embed(self):
        with self._driver.session()  as session:
            #Define the Cypher query
            options = self.global_config["node2vec_params"]
            query = f"""CALL gds.node2vec.stream('myGraph', {**options})
                    YIELD nodeId, embedding 
                    RETURN nodeId, embedding"""
            # Run the query and process the results
            results = session.run(query)
        for record in results:
            node_id = record["nodeId"]
            embedding = record["embedding"]
            print(f"Node ID: {node_id}, Embedding: {embedding}")
        #need to return two lists here.



