import array
import asyncio

# import html
import os
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np
import configparser

from lightrag.types import KnowledgeGraph

from ..base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger

import pipmaster as pm

if not pm.is_installed("graspologic"):
    pm.install("graspologic")

if not pm.is_installed("oracledb"):
    pm.install("oracledb")

from graspologic import embed
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
            raise ValueError("Missing database user or password")

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
                    await self.query(f"SELECT 1 FROM {k}")
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

    async def execute(self, sql: str, data: Union[list, dict] = None):
        # logger.info("go into OracleDB execute method")
        try:
            async with self.pool.acquire() as connection:
                connection.inputtypehandler = self.input_type_handler
                connection.outputtypehandler = self.output_type_handler
                with connection.cursor() as cursor:
                    if data is None:
                        await cursor.execute(sql)
                    else:
                        await cursor.execute(sql, data)
                    await connection.commit()
        except Exception as e:
            logger.error(f"Oracle database error: {e}")
            raise


class ClientManager:
    _instances: dict[str, Any] = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        config = configparser.ConfigParser()
        config.read("config.ini", "utf-8")

        return {
            "user": os.environ.get(
                "ORACLE_USER",
                config.get("oracle", "user", fallback=None),
            ),
            "password": os.environ.get(
                "ORACLE_PASSWORD",
                config.get("oracle", "password", fallback=None),
            ),
            "dsn": os.environ.get(
                "ORACLE_DSN",
                config.get("oracle", "dsn", fallback=None),
            ),
            "config_dir": os.environ.get(
                "ORACLE_CONFIG_DIR",
                config.get("oracle", "config_dir", fallback=None),
            ),
            "wallet_location": os.environ.get(
                "ORACLE_WALLET_LOCATION",
                config.get("oracle", "wallet_location", fallback=None),
            ),
            "wallet_password": os.environ.get(
                "ORACLE_WALLET_PASSWORD",
                config.get("oracle", "wallet_password", fallback=None),
            ),
            "workspace": os.environ.get(
                "ORACLE_WORKSPACE",
                config.get("oracle", "workspace", fallback="default"),
            ),
        }

    @classmethod
    async def get_client(cls) -> OracleDB:
        async with cls._lock:
            if cls._instances["db"] is None:
                config = ClientManager.get_config()
                db = OracleDB(config)
                await db.check_tables()
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: OracleDB):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        await db.pool.close()
                        logger.info("Closed OracleDB database connection pool")
                        cls._instances["db"] = None
                else:
                    await db.pool.close()


@final
@dataclass
class OracleKVStorage(BaseKVStorage):
    db: OracleDB = field(default=None)
    meta_fields = None

    def __post_init__(self):
        self._data = {}
        self._max_batch_size = self.global_config.get("embedding_batch_num", 10)

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get doc_full data based on id."""
        SQL = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "id": id}
        # print("get_by_id:"+SQL)
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            array_res = await self.db.query(SQL, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            if res:
                return res
            else:
                return None
        else:
            return await self.db.query(SQL, params)

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        """Specifically for llm_response_cache."""
        SQL = SQL_TEMPLATES["get_by_mode_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "cache_mode": mode, "id": id}
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            array_res = await self.db.query(SQL, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            return res
        else:
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data based on id"""
        SQL = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.db.workspace}
        # print("get_by_ids:"+SQL)
        res = await self.db.query(SQL, params, multirows=True)
        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            modes = set()
            dict_res: dict[str, dict] = {}
            for row in res:
                modes.add(row["mode"])
            for mode in modes:
                if mode not in dict_res:
                    dict_res[mode] = {}
            for row in res:
                dict_res[row["mode"]][row["id"]] = row
            res = [{k: v} for k, v in dict_res.items()]
        return res

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Return keys that don't exist in storage"""
        SQL = SQL_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.db.workspace}
        res = await self.db.query(SQL, params, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
            data = set([s for s in keys if s not in exist_keys])
            return data
        else:
            return set(keys)

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            list_data = [
                {
                    "id": k,
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

            merge_sql = SQL_TEMPLATES["merge_chunk"]
            for item in list_data:
                _data = {
                    "id": item["id"],
                    "content": item["content"],
                    "workspace": self.db.workspace,
                    "tokens": item["tokens"],
                    "chunk_order_index": item["chunk_order_index"],
                    "full_doc_id": item["full_doc_id"],
                    "content_vector": item["__vector__"],
                    "status": item["status"],
                }
                await self.db.execute(merge_sql, _data)
        if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            for k, v in data.items():
                # values.clear()
                merge_sql = SQL_TEMPLATES["merge_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.db.workspace,
                }
                await self.db.execute(merge_sql, _data)

        if is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            for mode, items in data.items():
                for k, v in items.items():
                    upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                    _data = {
                        "workspace": self.db.workspace,
                        "id": k,
                        "original_prompt": v["original_prompt"],
                        "return_value": v["return"],
                        "cache_mode": mode,
                    }

                    await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self) -> None:
        # Oracle handles persistence automatically
        pass


@final
@dataclass
class OracleVectorDBStorage(BaseVectorStorage):
    db: OracleDB | None = field(default=None)

    def __post_init__(self):
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    #################### query method ###############
    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
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
        results = await self.db.query(SQL, params=params, multirows=True)
        return results

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        raise NotImplementedError

    async def index_done_callback(self) -> None:
        # Oracles handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError


@final
@dataclass
class OracleGraphStorage(BaseGraphStorage):
    db: OracleDB = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config.get("embedding_batch_num", 10)

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    #################### insert method ################

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
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
        await self.db.execute(merge_sql, data)
        # self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
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

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        """为节点生成向量"""
        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids

    async def index_done_callback(self) -> None:
        # Oracles handles persistence automatically
        pass

    #################### query method #################
    async def has_node(self, node_id: str) -> bool:
        """根据节点id检查节点是否存在"""
        SQL = SQL_TEMPLATES["has_node"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        res = await self.db.query(SQL, params)
        if res:
            # print("Node exist!",res)
            return True
        else:
            # print("Node not exist!")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        SQL = SQL_TEMPLATES["has_edge"]
        params = {
            "workspace": self.db.workspace,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        res = await self.db.query(SQL, params)
        if res:
            # print("Edge exist!",res)
            return True
        else:
            # print("Edge not exist!")
            return False

    async def node_degree(self, node_id: str) -> int:
        SQL = SQL_TEMPLATES["node_degree"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        res = await self.db.query(SQL, params)
        if res:
            return res["degree"]
        else:
            return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """根据源和目标节点id获取边的度"""
        degree = await self.node_degree(src_id) + await self.node_degree(tgt_id)
        return degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """根据节点id获取节点数据"""
        SQL = SQL_TEMPLATES["get_node"]
        params = {"workspace": self.db.workspace, "node_id": node_id}
        res = await self.db.query(SQL, params)
        if res:
            return res
        else:
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
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

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
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

    async def get_statistics(self):
        SQL = SQL_TEMPLATES["get_statistics"]
        params = {"workspace": self.db.workspace}
        res = await self.db.query(sql=SQL, params=params, multirows=True)
        if res:
            return res

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError

    async def get_all_labels(self) -> list[str]:
        raise NotImplementedError

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        raise NotImplementedError


N_T = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_GRAPH_NODES",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_GRAPH_EDGES",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in N_T.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id varchar(256),
                    workspace varchar(1024),
                    doc_name varchar(1024),
                    content CLOB,
                    meta JSON,
                    content_summary varchar(1024),
                    content_length NUMBER,
                    status varchar(256),
                    chunks_count NUMBER,
                    createtime TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updatetime TIMESTAMP DEFAULT NULL,
                    error varchar(4096)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id varchar(256),
                    workspace varchar(1024),
                    full_doc_id varchar(256),
                    status varchar(256),
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
                    workspace varchar(1024),
                    cache_mode varchar(256),
                    model_name varchar(256),
                    original_prompt clob,
                    return_value clob,
                    embedding CLOB,
                    embedding_shape NUMBER,
                    embedding_min NUMBER,
                    embedding_max NUMBER,
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
    "get_by_id_full_docs": "select ID,content,status from LIGHTRAG_DOC_FULL where workspace=:workspace and ID=:id",
    "get_by_id_text_chunks": "select ID,TOKENS,content,CHUNK_ORDER_INDEX,FULL_DOC_ID,status from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and ID=:id",
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, NVL(return_value, '') as "return", cache_mode as "mode"
        FROM LIGHTRAG_LLM_CACHE WHERE workspace=:workspace AND id=:id""",
    "get_by_mode_id_llm_response_cache": """SELECT id, original_prompt, NVL(return_value, '') as "return", cache_mode as "mode"
        FROM LIGHTRAG_LLM_CACHE WHERE workspace=:workspace AND cache_mode=:cache_mode AND id=:id""",
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, NVL(return_value, '') as "return", cache_mode as "mode"
        FROM LIGHTRAG_LLM_CACHE WHERE workspace=:workspace  AND id IN ({ids})""",
    "get_by_ids_full_docs": "select t.*,createtime as created_at from LIGHTRAG_DOC_FULL t where workspace=:workspace and ID in ({ids})",
    "get_by_ids_text_chunks": "select ID,TOKENS,content,CHUNK_ORDER_INDEX,FULL_DOC_ID  from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and ID in ({ids})",
    "get_by_status_ids_full_docs": "select id,status from LIGHTRAG_DOC_FULL t where workspace=:workspace AND status=:status and ID in ({ids})",
    "get_by_status_ids_text_chunks": "select id,status from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and status=:status ID in ({ids})",
    "get_by_status_full_docs": "select id,status from LIGHTRAG_DOC_FULL t where workspace=:workspace AND status=:status",
    "get_by_status_text_chunks": "select id,status from LIGHTRAG_DOC_CHUNKS where workspace=:workspace and status=:status",
    "filter_keys": "select id from {table_name} where workspace=:workspace and id in ({ids})",
    "merge_doc_full": """MERGE INTO LIGHTRAG_DOC_FULL a
        USING DUAL
        ON (a.id = :id and a.workspace = :workspace)
        WHEN NOT MATCHED THEN
        INSERT(id,content,workspace) values(:id,:content,:workspace)""",
    "merge_chunk": """MERGE INTO LIGHTRAG_DOC_CHUNKS
        USING DUAL
        ON (id = :id and workspace = :workspace)
        WHEN NOT MATCHED THEN INSERT
            (id,content,workspace,tokens,chunk_order_index,full_doc_id,content_vector,status)
            values (:id,:content,:workspace,:tokens,:chunk_order_index,:full_doc_id,:content_vector,:status) """,
    "upsert_llm_response_cache": """MERGE INTO LIGHTRAG_LLM_CACHE a
        USING DUAL
        ON (a.id = :id)
        WHEN NOT MATCHED THEN
        INSERT (workspace,id,original_prompt,return_value,cache_mode)
            VALUES (:workspace,:id,:original_prompt,:return_value,:cache_mode)
        WHEN MATCHED THEN UPDATE
            SET original_prompt = :original_prompt,
            return_value = :return_value,
            cache_mode = :cache_mode,
            updatetime = SYSDATE""",
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
        WHERE e.workspace=:workspace and a.workspace=:workspace and b.workspace=:workspace
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
                    ON (a.workspace=:workspace and a.name=:name)
                WHEN NOT MATCHED THEN
                    INSERT(workspace,name,entity_type,description,source_chunk_id,content,content_vector)
                    values (:workspace,:name,:entity_type,:description,:source_chunk_id,:content,:content_vector)
                WHEN MATCHED THEN
                    UPDATE SET
                    entity_type=:entity_type,description=:description,source_chunk_id=:source_chunk_id,content=:content,content_vector=:content_vector,updatetime=SYSDATE""",
    "merge_edge": """MERGE INTO LIGHTRAG_GRAPH_EDGES a
                    USING DUAL
                    ON (a.workspace=:workspace and a.source_name=:source_name and a.target_name=:target_name)
                WHEN NOT MATCHED THEN
                    INSERT(workspace,source_name,target_name,weight,keywords,description,source_chunk_id,content,content_vector)
                    values (:workspace,:source_name,:target_name,:weight,:keywords,:description,:source_chunk_id,:content,:content_vector)
                WHEN MATCHED THEN
                    UPDATE SET
                    weight=:weight,keywords=:keywords,description=:description,source_chunk_id=:source_chunk_id,content=:content,content_vector=:content_vector,updatetime=SYSDATE""",
    "get_all_nodes": """WITH t0 AS (
                        SELECT name AS id, entity_type AS label, entity_type, description,
                            '["' || replace(source_chunk_id, '<SEP>', '","') || '"]'     source_chunk_ids
                        FROM lightrag_graph_nodes
                        WHERE workspace = :workspace
                        ORDER BY createtime DESC fetch first :limit rows only
                    ), t1 AS (
                        SELECT t0.id, source_chunk_id
                        FROM t0, JSON_TABLE ( source_chunk_ids, '$[*]' COLUMNS ( source_chunk_id PATH '$' ) )
                    ), t2 AS (
                        SELECT t1.id, LISTAGG(t2.content, '\n') content
                        FROM t1 LEFT JOIN lightrag_doc_chunks t2 ON t1.source_chunk_id = t2.id
                        GROUP BY t1.id
                    )
                    SELECT t0.id, label, entity_type, description, t2.content
                    FROM t0 LEFT JOIN t2 ON t0.id = t2.id""",
    "get_all_edges": """SELECT t1.id,t1.keywords as label,t1.keywords, t1.source_name as source, t1.target_name as target,
                t1.weight,t1.DESCRIPTION,t2.content
                FROM LIGHTRAG_GRAPH_EDGES t1
                LEFT JOIN LIGHTRAG_DOC_CHUNKS t2 on t1.source_chunk_id=t2.id
                WHERE t1.workspace=:workspace
                order by t1.CREATETIME DESC
                fetch first :limit rows only""",
    "get_statistics": """select  count(distinct CASE WHEN type='node' THEN id END) as nodes_count,
                count(distinct CASE WHEN type='edge' THEN id END) as edges_count
                FROM (
                select 'node' as type, id FROM GRAPH_TABLE (lightrag_graph
                    MATCH (a) WHERE a.workspace=:workspace columns(a.name as id))
                UNION
                select 'edge' as type, TO_CHAR(id) id FROM GRAPH_TABLE (lightrag_graph
                    MATCH (a)-[e]->(b) WHERE e.workspace=:workspace columns(e.id))
                )""",
}
