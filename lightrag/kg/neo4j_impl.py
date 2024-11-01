import asyncio
import html
import os
from dataclasses import dataclass
from typing import Any, Union, cast
import numpy as np
import inspect
from lightrag.utils import load_json, logger, write_json
from ..base import (
    BaseGraphStorage
)
from neo4j import GraphDatabase, exceptions as neo4jExceptions


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)



@dataclass
class GraphStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name):
       print ("no preloading of graph with neo4j in production")

    def __post_init__(self):
        # self._graph = preloaded_graph or nx.Graph()
        credetial_parts = ['URI', 'USERNAME','PASSWORD']
        credentials_set = all(x in os.environ for x in credetial_parts  )
        if credentials_set:
            URI = os.environ["URI"]
            USERNAME = os.environ["USERNAME"]
            PASSWORD = os.environ["PASSWORD"]
        else:
            raise Exception (f"One or more Neo4J Credentials, {credetial_parts}, not found in the environment")

        self._driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        print ("KG successfully indexed.")
    async def has_node(self, node_id: str) -> bool:
        entity_name_label = node_id.strip('\"')

        def _check_node_exists(tx, label):  
            query = f"MATCH (n:`{label}`) RETURN count(n) > 0 AS node_exists"  
            result = tx.run(query)  
            single_result = result.single()
            logger.debug(
                    f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["node_exists"]}'
            )  
            
            return single_result["node_exists"]
        
        with self._driver.session() as session:  
            return session.read_transaction(_check_node_exists, entity_name_label)

        
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        entity_name_label_source = source_node_id.strip('\"')
        entity_name_label_target = target_node_id.strip('\"')
       

        def _check_edge_existence(tx, label1, label2):  
            query = (  
                f"MATCH (a:`{label1}`)-[r]-(b:`{label2}`) "  
                "RETURN COUNT(r) > 0 AS edgeExists"  
            )  
            result = tx.run(query)  
            single_result = result.single()
            # if result.single() == None:
            #     print (f"this should not happen: ---- {label1}/{label2}   {query}")

            logger.debug(
                    f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{single_result["edgeExists"]}'
            )  
            
            return single_result["edgeExists"]
        def close(self):  
            self._driver.close()   
        #hard code relaitionship type 
        with self._driver.session() as session:  
                result = session.read_transaction(_check_edge_existence, entity_name_label_source, entity_name_label_target)  
                return result   
        


    async def get_node(self, node_id: str) -> Union[dict, None]:
        entity_name_label = node_id.strip('\"')
        with self._driver.session() as session:  
            query = "MATCH (n:`{entity_name_label}`) RETURN n".format(entity_name_label=entity_name_label)
            result = session.run(query)
            for record in result:
                result = record["n"]
                logger.debug(
                    f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}'
                )  
                return result
            


    async def node_degree(self, node_id: str) -> int:
        entity_name_label = node_id.strip('\"')


        def _find_node_degree(session, label):  
            with session.begin_transaction() as tx:  
                # query = "MATCH (n:`{label}`) RETURN n, size((n)--()) AS degree".format(label=label)
                query = f"""
                    MATCH (n:`{label}`)
                    RETURN COUNT{{ (n)--() }} AS totalEdgeCount
                """
                result = tx.run(query)  
                record = result.single()  
                if record:
                    edge_count = record["totalEdgeCount"]  
                    logger.debug(
                        f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{edge_count}'
                    )  
                    return edge_count
                else:  
                    return None
                
        with self._driver.session()  as session:
            degree = _find_node_degree(session, entity_name_label)
            return degree


    # degree = session.read_transaction(get_edge_degree, 1, 2)
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        entity_name_label_source = src_id.strip('\"')
        entity_name_label_target = tgt_id.strip('\"')
        with self._driver.session()  as session:
            query =  f"""MATCH (n1:`{entity_name_label_source}`)-[r]-(n2:`{entity_name_label_target}`)
                RETURN count(r) AS degree"""
            result = session.run(query)        
            record = result.single()
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{record["degree"]}'
            )       
            return record["degree"]
    
    async def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        entity_name_label_source = source_node_id.strip('\"')
        entity_name_label_target = target_node_id.strip('\"')
        """
        Find all edges between nodes of two given labels
        
        Args:
            source_node_label (str): Label of the source nodes
            target_node_label (str): Label of the target nodes
            
        Returns:
            list: List of all relationships/edges found
        """
        with self._driver.session()  as session:
            query = f"""
            MATCH (start:`{entity_name_label_source}`)-[r]->(end:`{entity_name_label_target}`)
            RETURN properties(r) as edge_properties
            LIMIT 1
            """.format(entity_name_label_source=entity_name_label_source, entity_name_label_target=entity_name_label_target)
            
            result = session.run(query)           
            record = result.single()
            if record:
                result = dict(record["edge_properties"])
                logger.debug(
                    f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{result}'
                )     
                return result
            else:
                return None
        

    async def get_node_edges(self, source_node_id: str):
        node_label = source_node_id.strip('\"')
          
        """
        Retrieves all edges (relationships) for a particular node identified by its label and ID.
        
        :param uri: Neo4j database URI
        :param username: Neo4j username
        :param password: Neo4j password
        :param node_label: Label of the node
        :param node_id: ID property of the node
        :return: List of dictionaries containing edge information
        """
        
        def fetch_edges(tx, label):
            query = f"""MATCH (n:`{label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, r, connected"""
            
            results = tx.run(query)

            edges = []
            for record in results:
                source_node = record['n']
                connected_node = record['connected']
                
                source_label = list(source_node.labels)[0] if source_node.labels else None
                target_label = list(connected_node.labels)[0] if connected_node and connected_node.labels else None
                
                if source_label and target_label:
                    print (f"appending: {(source_label, target_label)}")
                    edges.append((source_label, target_label))
            
            return edges

        with self._driver.session() as session:
            edges = session.read_transaction(fetch_edges,node_label)
            return edges


    
    # from typing import List, Tuple
    # async def get_node_connections(driver: GraphDatabase.driver, label: str) -> List[Tuple[str, str]]:
    #     def get_connections_for_node(tx):
    #         query = f"""
    #         MATCH (n:`{label}`)
    #         OPTIONAL MATCH (n)-[r]-(connected)
    #         RETURN n, r, connected
    #         """
    #         results = tx.run(query)
            
            
    #         connections = []
    #         for record in results:
    #             source_node = record['n']
    #             connected_node = record['connected']
                
    #             source_label = list(source_node.labels)[0] if source_node.labels else None
    #             target_label = list(connected_node.labels)[0] if connected_node and connected_node.labels else None
                
    #             if source_label and target_label:
    #                 connections.append((source_label, target_label))

    #         logger.debug(
    #             f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{connections}'
    #         ) 
    #         return connections

    #     with driver.session() as session:
              
    #         return session.read_transaction(get_connections_for_node)

        



    #upsert_node

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((neo4jExceptions.ServiceUnavailable, neo4jExceptions.TransientError, neo4jExceptions.WriteServiceUnavailable)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        label = node_id.strip('\"')
        properties = node_data
        """
        Upsert a node with the given label and properties within a transaction.
        Args:
            label: The node label to search for and apply
            properties: Dictionary of node properties
            
        Returns:
            Dictionary containing the node's properties after upsert, or None if operation fails
        """
        def _do_upsert(tx, label: str, properties: dict[str, Any]):

            """            
            Args:
                tx: Neo4j transaction object
                label: The node label to search for and apply
                properties: Dictionary of node properties
                
            Returns:
                Dictionary containing the node's properties after upsert, or None if operation fails
            """

            query = f"""
            MERGE (n:`{label}`)
            SET n += $properties
            RETURN n
            """
            # Execute the query with properties as parameters
            # with session.begin_transaction() as tx:  
            result = tx.run(query, properties=properties)
            record = result.single()
            if record:
                logger.debug(
                    f'{inspect.currentframe().f_code.co_name}:query:{query}:result:{dict(record["n"])}'
                )   
                return dict(record["n"])
            return None


        with self._driver.session() as session:
            with session.begin_transaction() as tx:
                try:
                    result = _do_upsert(tx,label,properties)
                    tx.commit()
                    return result
                except Exception as e:
                    raise  # roll back

               

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        source_node_label = source_node_id.strip('\"')
        target_node_label = target_node_id.strip('\"')
        edge_properties = edge_data
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        
        Args:
            source_node_label (str): Label of the source node (used as identifier)
            target_node_label (str): Label of the target node (used as identifier)
            edge_properties (dict): Dictionary of properties to set on the edge
        """
       

        
        def _do_upsert_edge(tx, source_node_label: str, target_node_label: str, edge_properties: dict[str, Any]) -> None:
            """
            Static method to perform the edge upsert within a transaction.
            
            The query will:
            1. Match the source and target nodes by their labels
            2. Merge the DIRECTED relationship
            3. Set all properties on the relationship, updating existing ones and adding new ones
            """
            # Convert edge properties to Cypher parameter string
            # props_string = ", ".join(f"r.{key} = ${key}" for key in edge_properties.keys())

            # """.format(props_string)
            query = f"""
            MATCH (source:`{source_node_label}`)
            WITH source
            MATCH (target:`{target_node_label}`)
            MERGE (source)-[r:DIRECTED]->(target)
            SET r += $properties
            RETURN r
            """

            result = tx.run(query, properties=edge_properties)
            logger.debug(
                f'{inspect.currentframe().f_code.co_name}:query:{query}:edge_properties:{edge_properties}'
            )               
            return result.single()
            
        with self._driver.session() as session:
            session.execute_write(
                _do_upsert_edge,
                source_node_label,
                target_node_label,
                edge_properties
            )
            # return result

    async def _node2vec_embed(self):
        print ("this is never called.  checking to be sure.")
        
        # async def _node2vec_embed(self):
        with self._driver.session()  as session:
            #Define the Cypher query
            options = self.global_config["node2vec_params"]
            logger.debug(f"building embeddings with options {options}")
            query = f"""CALL gds.node2vec.write('91fbae6c', {
                options
                })
                YIELD nodeId, labels, embedding
                RETURN 
                nodeId AS id, 
                labels[0] AS distinctLabel, 
                embedding AS nodeToVecEmbedding
                """
            # Run the query and process the results
            results = session.run(query)
            embeddings = []
            node_labels = []
        for record in results:
            node_id = record["id"]
            embedding = record["nodeToVecEmbedding"]
            label = record["distinctLabel"]
            print(f"Node id/label: {label}/{node_id}, Embedding: {embedding}")
            embeddings.append(embedding)
            node_labels.append(label)
        return embeddings, node_labels

