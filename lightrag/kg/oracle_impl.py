import asyncio

# import html
# import os
from dataclasses import dataclass
from typing import Union
import numpy as np
import array

from ..utils import logger
from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)

import oracledb


class OracleDB:
    def __init__(self, config, **kwargs):
        self.host = config.get("host", None)
        self.port = config.get("port", None)
        self.user = config.get("user", None)
        self.password = config.get("password", None)
        self.dsn = config.get("dsn", None)
        self.config_dir = config.get("config_dir", None)
        self.wallet_location = config.get("wallet_location", None)
        self.wallet_password = config.get("wallet_password", None)
        self.workspace = config.get("workspace", None)
        self.max = 12
        self.increment = 1
        logger.info(f"Using the label {self.workspace} for Oracle Graph as identifier")
        if self.user is None or self.password is None:
            raise ValueError("Missing database user or password in addon_params")

        try:
            oracledb.defaults.fetch_lobs = False

            self.pool = oracledb.create_pool_async(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                config_dir=self.config_dir,
                wallet_location=self.wallet_location,
                wallet_password=self.wallet_password,
                min=1,
                max=self.max,
                increment=self.increment,
            )
            logger.info(f"Connected to Oracle database at {self.dsn}")
        except Exception as e:
            logger.error(f"Failed to connect to Oracle database at {self.dsn}")
            logger.error(f"Oracle database error: {e}")
            raise

    def numpy_converter_in(self, value):
        """Convert numpy array to array.array"""
        if value.dtype == np.float64:
            dtype = "d"
        elif value.dtype == np.float32:
            dtype = "f"
        else:
            dtype = "b"
        return array.array(dtype, value)

    def input_type_handler(self, cursor, value, arraysize):
        """Set the type handler for the input data"""
        if isinstance(value, np.ndarray):
            return cursor.var(
                oracledb.DB_TYPE_VECTOR,
                arraysize=arraysize,
                inconverter=self.numpy_converter_in,
            )

    def numpy_converter_out(self, value):
        """Convert array.array to numpy array"""
        if value.typecode == "b":
            dtype = np.int8
        elif value.typecode == "f":
            dtype = np.float32
        else:
            dtype = np.float64
        return np.array(value, copy=False, dtype=dtype)

    def output_type_handler(self, cursor, metadata):
        """Set the type handler for the output data"""
        if metadata.type_code is oracledb.DB_TYPE_VECTOR:
            return cursor.var(
                metadata.type_code,
                arraysize=cursor.arraysize,
                outconverter=self.numpy_converter_out,
            )

    async def check_tables(self):
        for k, v in TABLES.items():
            try:
                if k.lower() == "lightrag_graph":
                    await self.query(
                        "SELECT id FROM GRAPH_TABLE (lightrag_graph MATCH (a) COLUMNS (a.id)) fetch first row only"
                    )
                else:
                    await self.query("SELECT 1 FROM {k}".format(k=k))
            except Exception as e:
                logger.error(f"Failed to check table {k} in Oracle database")
                logger.error(f"Oracle database error: {e}")
                try:
                    # print(v["ddl"])
                    await self.execute(v["ddl"])
                    logger.info(f"Created table {k} in Oracle database")
                except Exception as e:
                    logger.error(f"Failed to create table {k} in Oracle database")
                    logger.error(f"Oracle database error: {e}")

        logger.info("Finished check all tables in Oracle database")

    async def query(
        self, sql: str, params: dict = None, multirows: bool = False
    ) -> Union[dict, None]:
        async with self.pool.acquire() as connection:
            connection.inputtypehandler = self.input_type_handler
            connection.outputtypehandler = self.output_type_handler
            with connection.cursor() as cursor:
                try:
                    await cursor.execute(sql, params)
                except Exception as e:
                    logger.error(f"Oracle database error: {e}")
                    print(sql)
                    print(params)
                    raise
                columns = [column[0].lower() for column in cursor.description]
                if multirows:
                    rows = await cursor.fetchall()
                    if rows:
                        data = [dict(zip(columns, row)) for row in rows]
                    else:
                        data = []
                else:
                    row = await cursor.fetchone()
                    if row:
                        data = dict(zip(columns, row))
                    else:
                        data = None
                return data

    async def execute(self, sql: str, data: list | dict = None):
        # logger.info("go into OracleDB execute method")
        try:
            async with self.pool.acquire() as connection:
                connection.inputtypehandler = self.input_type_handler
                connection.outputtypehandler = self.output_type_handler
                with connection.cursor() as cursor:
                    if data is None:
                        await cursor.execute(sql)
                    else:
                        # print(data)
                        # print(sql)
                        await cursor.execute(sql, data)
                    await connection.commit()
        except Exception as e:
            logger.error(f"Oracle database error: {e}")
            print(sql)
            print(data)
            raise


@dataclass
class OracleKVStorage(BaseKVStorage):
    # should pass db object to self.db
    def __post_init__(self):
        self._data = {}
        self._max_batch_size = self.global_config["embedding_batch_num"]

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> Union[dict, None]:
        """根据 id 获取 doc_full 数据."""
        SQL = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "id": id}
        # print("get_by_id:"+SQL)
        res = await self.db.query(SQL, params)
        if res:
            data = res  # {"data":res}
            # print (data)
            return data
        else:
            return None

    # Query by id
    async def get_by_ids(self, ids: list[str], fields=None) -> Union[list[dict], None]:
        """根据 id 获取 doc_chunks 数据"""
        SQL = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.db.workspace}
        # print("get_by_ids:"+SQL)
        # print(params)
        res = await self.db.query(SQL, params, multirows=True)
        if res:
            data = res  # [{"data":i} for i in res]
            # print(data)
            return data
        else:
            return None

    async def filter_keys(self, keys: list[str]) -> set[str]:
        """过滤掉重复内容"""
        SQL = SQL_TEMPLATES["filter_keys"].format(
            table_name=N_T[self.namespace], ids=",".join([f"'{id}'" for id in keys])
        )
        params = {"workspace": self.db.workspace}
        try:
            await self.db.query(SQL, params)
        except Exception as e:
            logger.error(f"Oracle database error: {e}")
            print(SQL)
            print(params)
        res = await self.db.query(SQL, params, multirows=True)
        data = None
        if res:
            exist_keys = [key["id"] for key in res]
            data = set([s for s in keys if s not in exist_keys])
        else:
            exist_keys = []
            data = set([s for s in keys if s not in exist_keys])
        return data

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        # print(self._data)
        # values = []
        if self.namespace == "text_chunks":
            list_data = [
                {
                    "__id__": k,
                    **{k1: v1 for k1, v1 in v.items()},
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
            # print(list_data)
            for item in list_data:
                merge_sql = SQL_TEMPLATES["merge_chunk"]
                data = {
                    "check_id": item["__id__"],
                    "id": item["__id__"],
                    "content": item["content"],
                    "workspace": self.db.workspace,
                    "tokens": item["tokens"],
                    "chunk_order_index": item["chunk_order_index"],
                    "full_doc_id": item["full_doc_id"],
                    "content_vector": item["__vector__"],
                }
                # print(merge_sql)
                await self.db.execute(merge_sql, data)

        if self.namespace == "full_docs":
            for k, v in self._data.items():
                # values.clear()
                merge_sql = SQL_TEMPLATES["merge_doc_full"]
                data = {
                    "check_id": k,
                    "id": k,
                    "content": v["content"],
                    "workspace": self.db.workspace,
                }
                # print(merge_sql)
                await self.db.execute(merge_sql, data)
        return left_data

    async def index_done_callback(self):
        if self.namespace in ["full_docs", "text_chunks"]:
            logger.info("full doc and chunk data had been saved into oracle db!")


@dataclass
class OracleVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        pass

    async def upsert(self, data: dict[str, dict]):
        """向向量数据库中插入数据"""
        pass

    async def index_done_callback(self):
        pass

    #################### query method ###############
    async def query(self, query: str, top_k=5) -> Union[dict, list[dict]]:
        """从向量数据库中查询数据"""
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]
        # 转换精度
        dtype = str(embedding.dtype).upper()
        dimension = embedding.shape[0]
        embedding_string = "[" + ", ".join(map(str, embedding.tolist())) + "]"

        SQL = SQL_TEMPLATES[self.namespace].format(dimension=dimension, dtype=dtype)
        params = {
            "embedding_string": embedding_string,
            "workspace": self.db.workspace,
            "top_k": top_k,
            "better_than_threshold": self.cosine_better_than_threshold,
        }
        # print(SQL)
        results = await self.db.query(SQL, params=params, multirows=True)
        # print("vector search result:",results)
        return results


@dataclass
class OracleGraphStorage(BaseGraphStorage):
    """基于Oracle的图存储模块"""

    def __post_init__(self):
        """从graphml文件加载图"""
        self._max_batch_size = self.global_config["embedding_batch_num"]

    #################### insert method ################

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """插入或更新节点"""
        # print("go into upsert node method")
        entity_name = node_id
        entity_type = node_data["entity_type"]
        description = node_data["description"]
        source_id = node_data["source_id"]
        logger.debug(f"entity_name:{entity_name}, entity_type:{entity_type}")

        content = entity_name + description
        contents = [content]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        content_vector = embeddings[0]
        merge_sql = SQL_TEMPLATES["merge_node"]
        data = {
            "workspace": self.db.workspace,
            "name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_chunk_id": source_id,
            "content": content,
            "content_vector": content_vector,
        }
        # print(merge_sql)
        await self.db.execute(merge_sql, data)
        # self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """插入或更新边"""
        # print("go into upsert edge method")
        source_name = source_node_id
        target_name = target_node_id
        weight = edge_data["weight"]
        keywords = edge_data["keywords"]
        description = edge_data["description"]
        source_chunk_id = edge_data["source_id"]
        logger.debug(
            f"source_name:{source_name}, target_name:{target_name}, keywords: {keywords}"
        )

        content = keywords + source_name + target_name + description
        contents = [content]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        content_vector = embeddings[0]
        merge_sql = SQL_TEMPLATES["merge_edge"]
        data = {
            "workspace": self.db.workspace,
            "source_name": source_name,
            "target_name": target_name,
            "weight": weight,
            "keywords": keywords,
            "description": description,
            "source_chunk_id": source_chunk_id,
            "content": content,
            "content_vector": content_vector,
        }
        # print(merge_sql)
        await self.db.execute(merge_sql, data)
        # self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """为节点生成向量"""
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        """为节点生成向量"""
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    async def index_done_callback(self):
        """写入graphhml图文件"""
        logger.info(
            "Node and edge data had been saved into oracle db already, so nothing to do here!"
        )

    #################### query method #################
    async def has_node(self, node_id: str) -> bool:
        """根据节点id检查节点是否存在"""
        SQL = SQL_TEMPLATES["has_node"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        # print(SQL)
        # print(self.db.workspace, node_id)
        res = await self.db.query(SQL, params)
        if res:
            # print("Node exist!",res)
            return True
        else:
            # print("Node not exist!")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """根据源和目标节点id检查边是否存在"""
        SQL = SQL_TEMPLATES["has_edge"]
        params = {
            "workspace": self.db.workspace,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        # print(SQL)
        res = await self.db.query(SQL, params)
        if res:
            # print("Edge exist!",res)
            return True
        else:
            # print("Edge not exist!")
            return False

    async def node_degree(self, node_id: str) -> int:
        """根据节点id获取节点的度"""
        SQL = SQL_TEMPLATES["node_degree"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        # print(SQL)
        res = await self.db.query(SQL, params)
        if res:
            # print("Node degree",res["degree"])
            return res["degree"]
        else:
            # print("Edge not exist!")
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """根据源和目标节点id获取边的度"""
        degree = await self.node_degree(src_id) + await self.node_degree(tgt_id)
        # print("Edge degree",degree)
        return degree

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """根据节点id获取节点数据"""
        SQL = SQL_TEMPLATES["get_node"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        # print(self.db.workspace, node_id)
        # print(SQL)
        res = await self.db.query(SQL, params)
        if res:
            # print("Get node!",self.db.workspace, node_id,res)
            return res
        else:
            # print("Can't get node!",self.db.workspace, node_id)
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """根据源和目标节点id获取边"""
        SQL = SQL_TEMPLATES["get_edge"]
        params = {
            "workspace": self.db.workspace,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        res = await self.db.query(SQL, params)
        if res:
            # print("Get edge!",self.db.workspace, source_node_id, target_node_id,res[0])
            return res
        else:
            # print("Edge not exist!",self.db.workspace, source_node_id, target_node_id)
            return None

    async def get_node_edges(self, source_node_id: str):
        """根据节点id获取节点的所有边"""
        if await self.has_node(source_node_id):
            SQL = SQL_TEMPLATES["get_node_edges"]
            params = {"workspace": self.db.workspace, "source_node_id": source_node_id}
            res = await self.db.query(sql=SQL, params=params, multirows=True)
            if res:
                data = [(i["source_name"], i["target_name"]) for i in res]
                # print("Get node edge!",self.db.workspace, source_node_id,data)
                return data
            else:
                # print("Node Edge not exist!",self.db.workspace, source_node_id)
                return []

    async def get_all_nodes(self, limit: int):
        """查询所有节点"""
        SQL = SQL_TEMPLATES["get_all_nodes"]
        params = {"workspace": self.db.workspace, "limit": str(limit)}
        res = await self.db.query(sql=SQL, params=params, multirows=True)
        if res:
            return res

    async def get_all_edges(self, limit: int):
        """查询所有边"""
        SQL = SQL_TEMPLATES["get_all_edges"]
        params = {"workspace": self.db.workspace, "limit": str(limit)}
        res = await self.db.query(sql=SQL, params=params, multirows=True)
        if res:
            return res


N_T = {
    "full_docs": "LIGHTRAG_DOC_FULL",
    "text_chunks": "LIGHTRAG_DOC_CHUNKS",
    "chunks": "LIGHTRAG_DOC_CHUNKS",
    "entities": "LIGHTRAG_GRAPH_NODES",
    "relationships": "LIGHTRAG_GRAPH_EDGES",
}

TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id varchar(256)PRIMARY KEY,
                    workspace varchar(1024),
                    doc_name varchar(1024),
                    content CLOB,
                    meta JSON,
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id varchar(256) PRIMARY KEY,
                    workspace varchar(1024),
                    full_doc_id varchar(256),
                    chunk_order_index NUMBER,
                    tokens NUMBER,
                    content CLOB,
                    content_vector VECTOR,
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL
                    )"""
    },
    "LIGHTRAG_GRAPH_NODES": {
        "ddl": """CREATE TABLE LIGHTRAG_GRAPH_NODES (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    workspace varchar(1024),
                    name varchar(2048),
                    entity_type varchar(1024),
                    description CLOB,
                    source_chunk_id varchar(256),
                    content CLOB,
                    content_vector VECTOR,
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL
                    )"""
    },
    "LIGHTRAG_GRAPH_EDGES": {
        "ddl": """CREATE TABLE LIGHTRAG_GRAPH_EDGES (
                    id NUMBER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                    workspace varchar(1024),
                    source_name varchar(2048),
                    target_name varchar(2048),
                    weight NUMBER,
                    keywords CLOB,
                    description CLOB,
                    source_chunk_id varchar(256),
                    content CLOB,
                    content_vector VECTOR,
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
                    id varchar(256) PRIMARY KEY,
                    send clob,
                    return clob,
                    model varchar(1024),
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL
                    )"""
    },
    "LIGHTRAG_GRAPH": {
        "ddl": """CREATE OR REPLACE PROPERTY GRAPH lightrag_graph
                VERTEX TABLES (
                    lightrag_graph_nodes KEY (id)
                        LABEL entity
                        PROPERTIES (id,workspace,name) -- ,entity_type,description,source_chunk_id)
                )
                EDGE TABLES (
                    lightrag_graph_edges KEY (id)
                        SOURCE KEY (source_name) REFERENCES lightrag_graph_nodes(name)
                        DESTINATION KEY (target_name) REFERENCES lightrag_graph_nodes(name)
                        LABEL  has_relation
                        PROPERTIES (id,workspace,source_name,target_name) -- ,weight, keywords,description,source_chunk_id)
                ) OPTIONS(ALLOW MIXED PROPERTY TYPES)"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": "select ID,NVL(content,'') as content from LIGHTRAG_DOC_FULL where workspace=:workspace and ID=:id",
    "get_by_id_text_chunks": "select ID,TOKENS,NVL(content,'') as content,CHUNK_ORDER_INDEX,FULL_DOC_ID from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and ID=:id",
    "get_by_ids_full_docs": "select ID,NVL(content,'') as content from LIGHTRAG_DOC_FULL where workspace=:workspace and ID in ({ids})",
    "get_by_ids_text_chunks": "select ID,TOKENS,NVL(content,'') as content,CHUNK_ORDER_INDEX,FULL_DOC_ID  from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and ID in ({ids})",
    "filter_keys": "select id from {table_name} where workspace=:workspace and id in ({ids})",
    "merge_doc_full": """ MERGE INTO LIGHTRAG_DOC_FULL a
                    USING DUAL
                    ON (a.id = :check_id)
                    WHEN NOT MATCHED THEN
                    INSERT(id,content,workspace) values(:id,:content,:workspace)
                    """,
    "merge_chunk": """MERGE INTO LIGHTRAG_DOC_CHUNKS a
                    USING DUAL
                    ON (a.id = :check_id)
                    WHEN NOT MATCHED THEN
                    INSERT(id,content,workspace,tokens,chunk_order_index,full_doc_id,content_vector)
                    values (:id,:content,:workspace,:tokens,:chunk_order_index,:full_doc_id,:content_vector) """,
    # SQL for VectorStorage
    "entities": """SELECT name as entity_name FROM
        (SELECT id,name,VECTOR_DISTANCE(content_vector,vector(:embedding_string,{dimension},{dtype}),COSINE) as distance
        FROM LIGHTRAG_GRAPH_NODES WHERE workspace=:workspace)
        WHERE distance>:better_than_threshold ORDER BY distance ASC FETCH FIRST :top_k ROWS ONLY""",
    "relationships": """SELECT source_name as src_id, target_name as tgt_id FROM
        (SELECT id,source_name,target_name,VECTOR_DISTANCE(content_vector,vector(:embedding_string,{dimension},{dtype}),COSINE) as distance
        FROM LIGHTRAG_GRAPH_EDGES WHERE workspace=:workspace)
        WHERE distance>:better_than_threshold ORDER BY distance ASC FETCH FIRST :top_k ROWS ONLY""",
    "chunks": """SELECT id FROM
        (SELECT id,VECTOR_DISTANCE(content_vector,vector(:embedding_string,{dimension},{dtype}),COSINE) as distance
        FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=:workspace)
        WHERE distance>:better_than_threshold ORDER BY distance ASC FETCH FIRST :top_k ROWS ONLY""",
    # SQL for GraphStorage
    "has_node": """SELECT * FROM GRAPH_TABLE (lightrag_graph
        MATCH (a)
        WHERE a.workspace=:workspace AND a.name=:node_id
        COLUMNS (a.name))""",
    "has_edge": """SELECT * FROM GRAPH_TABLE (lightrag_graph
        MATCH (a) -[e]-> (b)
        WHERE e.workspace=:workspace and a.workspace=:workspace and b.workspace=:workspace
        AND a.name=:source_node_id AND b.name=:target_node_id
        COLUMNS (e.source_name,e.target_name)  )""",
    "node_degree": """SELECT count(1) as degree FROM GRAPH_TABLE (lightrag_graph
        MATCH (a)-[e]->(b)
        WHERE a.workspace=:workspace and a.workspace=:workspace and b.workspace=:workspace
        AND a.name=:node_id or b.name = :node_id
        COLUMNS (a.name))""",
    "get_node": """SELECT t1.name,t2.entity_type,t2.source_chunk_id as source_id,NVL(t2.description,'') AS description
        FROM GRAPH_TABLE (lightrag_graph
        MATCH (a)
        WHERE a.workspace=:workspace AND a.name=:node_id
        COLUMNS (a.name)
        ) t1 JOIN LIGHTRAG_GRAPH_NODES t2 on t1.name=t2.name
        WHERE t2.workspace=:workspace""",
    "get_edge": """SELECT t1.source_id,t2.weight,t2.source_chunk_id as source_id,t2.keywords,
        NVL(t2.description,'') AS description,NVL(t2.KEYWORDS,'') AS keywords
        FROM GRAPH_TABLE (lightrag_graph
        MATCH (a)-[e]->(b)
        WHERE e.workspace=:workspace and a.workspace=:workspace and b.workspace=:workspace
        AND a.name=:source_node_id and b.name = :target_node_id
        COLUMNS (e.id,a.name as source_id)
        ) t1 JOIN LIGHTRAG_GRAPH_EDGES t2 on t1.id=t2.id""",
    "get_node_edges": """SELECT source_name,target_name
            FROM GRAPH_TABLE (lightrag_graph
            MATCH (a)-[e]->(b)
            WHERE e.workspace=:workspace and a.workspace=:workspace and b.workspace=:workspace
            AND a.name=:source_node_id
            COLUMNS (a.name as source_name,b.name as target_name))""",
    "merge_node": """MERGE INTO LIGHTRAG_GRAPH_NODES a
                    USING DUAL
                    ON (a.workspace = :workspace and a.name=:name and a.source_chunk_id=:source_chunk_id)
                WHEN NOT MATCHED THEN
                    INSERT(workspace,name,entity_type,description,source_chunk_id,content,content_vector)
                    values (:workspace,:name,:entity_type,:description,:source_chunk_id,:content,:content_vector) """,
    "merge_edge": """MERGE INTO LIGHTRAG_GRAPH_EDGES a
                    USING DUAL
                    ON (a.workspace = :workspace and a.source_name=:source_name and a.target_name=:target_name and a.source_chunk_id=:source_chunk_id)
                WHEN NOT MATCHED THEN
                    INSERT(workspace,source_name,target_name,weight,keywords,description,source_chunk_id,content,content_vector)
                    values (:workspace,:source_name,:target_name,:weight,:keywords,:description,:source_chunk_id,:content,:content_vector) """,
    "get_all_nodes": """SELECT t1.name as id,t1.entity_type as label,t1.DESCRIPTION,t2.content
                FROM LIGHTRAG_GRAPH_NODES t1
                LEFT JOIN LIGHTRAG_DOC_CHUNKS t2 on t1.source_chunk_id=t2.id
                WHERE t1.workspace=:workspace
                order by t1.CREATETIME DESC
                fetch first :limit rows only
                """,
    "get_all_edges": """SELECT t1.id,t1.keywords as label,t1.keywords, t1.source_name as source, t1.target_name as target,
                t1.weight,t1.DESCRIPTION,t2.content
                FROM LIGHTRAG_GRAPH_EDGES t1
                LEFT JOIN LIGHTRAG_DOC_CHUNKS t2 on t1.source_chunk_id=t2.id
                WHERE t1.workspace=:workspace
                order by t1.CREATETIME DESC
                fetch first :limit rows only""",
}
