import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Union, final

import numpy as np

from lightrag.types import KnowledgeGraph


from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from ..namespace import NameSpace, is_namespace
from ..utils import logger

import pipmaster as pm
import configparser

if not pm.is_installed("pymysql"):
    pm.install("pymysql")
if not pm.is_installed("sqlalchemy"):
    pm.install("sqlalchemy")

from sqlalchemy import create_engine, text


class TiDB:
    def __init__(self, config, **kwargs):
        self.host = config.get("host", None)
        self.port = config.get("port", None)
        self.user = config.get("user", None)
        self.password = config.get("password", None)
        self.database = config.get("database", None)
        self.workspace = config.get("workspace", None)
        connection_string = (
            f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            f"?ssl_verify_cert=true&ssl_verify_identity=true"
        )

        try:
            self.engine = create_engine(connection_string)
            logger.info(f"Connected to TiDB database at {self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to TiDB database at {self.database}")
            logger.error(f"TiDB database error: {e}")
            raise

    async def check_tables(self):
        for k, v in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {k}".format(k=k))
            except Exception as e:
                logger.error(f"Failed to check table {k} in TiDB database")
                logger.error(f"TiDB database error: {e}")
                try:
                    await self.execute(v["ddl"])
                    logger.info(f"Created table {k} in TiDB database")
                except Exception as e:
                    logger.error(f"Failed to create table {k} in TiDB database")
                    logger.error(f"TiDB database error: {e}")

    async def query(
        self, sql: str, params: dict = None, multirows: bool = False
    ) -> Union[dict, None]:
        if params is None:
            params = {"workspace": self.workspace}
        else:
            params.update({"workspace": self.workspace})
        with self.engine.connect() as conn, conn.begin():
            try:
                result = conn.execute(text(sql), params)
            except Exception as e:
                logger.error(f"Tidb database,\nsql:{sql},\nparams:{params},\nerror:{e}")
                raise
            if multirows:
                rows = result.all()
                if rows:
                    data = [dict(zip(result.keys(), row)) for row in rows]
                else:
                    data = []
            else:
                row = result.first()
                if row:
                    data = dict(zip(result.keys(), row))
                else:
                    data = None
            return data

    async def execute(self, sql: str, data: list | dict = None):
        # logger.info("go into TiDBDB execute method")
        try:
            with self.engine.connect() as conn, conn.begin():
                if data is None:
                    conn.execute(text(sql))
                else:
                    conn.execute(text(sql), parameters=data)
        except Exception as e:
            logger.error(f"Tidb database,\nsql:{sql},\ndata:{data},\nerror:{e}")
            raise


class ClientManager:
    _instances: dict[str, Any] = {"db": None, "ref_count": 0}
    _lock = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        config = configparser.ConfigParser()
        config.read("config.ini", "utf-8")

        return {
            "host": os.environ.get(
                "TIDB_HOST",
                config.get("tidb", "host", fallback="localhost"),
            ),
            "port": os.environ.get(
                "TIDB_PORT", config.get("tidb", "port", fallback=4000)
            ),
            "user": os.environ.get(
                "TIDB_USER",
                config.get("tidb", "user", fallback=None),
            ),
            "password": os.environ.get(
                "TIDB_PASSWORD",
                config.get("tidb", "password", fallback=None),
            ),
            "database": os.environ.get(
                "TIDB_DATABASE",
                config.get("tidb", "database", fallback=None),
            ),
            "workspace": os.environ.get(
                "TIDB_WORKSPACE",
                config.get("tidb", "workspace", fallback="default"),
            ),
        }

    @classmethod
    async def get_client(cls) -> TiDB:
        async with cls._lock:
            if cls._instances["db"] is None:
                config = ClientManager.get_config()
                db = TiDB(config)
                await db.check_tables()
                cls._instances["db"] = db
                cls._instances["ref_count"] = 0
            cls._instances["ref_count"] += 1
            return cls._instances["db"]

    @classmethod
    async def release_client(cls, db: TiDB):
        async with cls._lock:
            if db is not None:
                if db is cls._instances["db"]:
                    cls._instances["ref_count"] -= 1
                    if cls._instances["ref_count"] == 0:
                        cls._instances["db"] = None


@final
@dataclass
class TiDBKVStorage(BaseKVStorage):
    db: TiDB = field(default=None)

    def __post_init__(self):
        self._data = {}
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Fetch doc_full data by id."""
        SQL = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"id": id}
        response = await self.db.query(SQL, params)
        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch doc_chunks data by id"""
        SQL = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        return await self.db.query(SQL, multirows=True)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        SQL = SQL_TEMPLATES["filter_keys"].format(
            table_name=namespace_to_table_name(self.namespace),
            id_field=namespace_to_id(self.namespace),
            ids=",".join([f"'{id}'" for id in keys]),
        )
        try:
            await self.db.query(SQL)
        except Exception as e:
            logger.error(f"Tidb database,\nsql:{SQL},\nkeys:{keys},\nerror:{e}")
        res = await self.db.query(SQL, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
            data = set([s for s in keys if s not in exist_keys])
        else:
            exist_keys = []
            data = set([s for s in keys if s not in exist_keys])
        return data

    ################ INSERT full_doc AND chunks ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
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

            merge_sql = SQL_TEMPLATES["upsert_chunk"]
            data = []
            for item in list_data:
                data.append(
                    {
                        "id": item["__id__"],
                        "content": item["content"],
                        "tokens": item["tokens"],
                        "chunk_order_index": item["chunk_order_index"],
                        "full_doc_id": item["full_doc_id"],
                        "content_vector": f"{item['__vector__'].tolist()}",
                        "workspace": self.db.workspace,
                    }
                )
            await self.db.execute(merge_sql, data)

        if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            merge_sql = SQL_TEMPLATES["upsert_doc_full"]
            data = []
            for k, v in self._data.items():
                data.append(
                    {
                        "id": k,
                        "content": v["content"],
                        "workspace": self.db.workspace,
                    }
                )
            await self.db.execute(merge_sql, data)
        return left_data

    async def index_done_callback(self) -> None:
        # Ti handles persistence automatically
        pass


@final
@dataclass
class TiDBVectorDBStorage(BaseVectorStorage):
    db: TiDB | None = field(default=None)

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
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

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search from tidb vector"""
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]

        embedding_string = "[" + ", ".join(map(str, embedding.tolist())) + "]"

        params = {
            "embedding_string": embedding_string,
            "top_k": top_k,
            "better_than_threshold": self.cosine_better_than_threshold,
        }

        results = await self.db.query(
            SQL_TEMPLATES[self.namespace], params=params, multirows=True
        )
        print("vector search result:", results)
        if not results:
            return []
        return results

    ###### INSERT entities And relationships ######
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return
        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            return

        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")

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
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["content_vector"] = embeddings[i]

        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            data = []
            for item in list_data:
                param = {
                    "id": item["id"],
                    "name": item["entity_name"],
                    "content": item["content"],
                    "content_vector": f"{item['content_vector'].tolist()}",
                    "workspace": self.db.workspace,
                }
                # update entity_id if node inserted by graph_storage_instance before
                has = await self.db.query(SQL_TEMPLATES["has_entity"], param)
                if has["cnt"] != 0:
                    await self.db.execute(SQL_TEMPLATES["update_entity"], param)
                    continue

                data.append(param)
            if data:
                merge_sql = SQL_TEMPLATES["insert_entity"]
                await self.db.execute(merge_sql, data)

        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            data = []
            for item in list_data:
                param = {
                    "id": item["id"],
                    "source_name": item["src_id"],
                    "target_name": item["tgt_id"],
                    "content": item["content"],
                    "content_vector": f"{item['content_vector'].tolist()}",
                    "workspace": self.db.workspace,
                }
                # update relation_id if node inserted by graph_storage_instance before
                has = await self.db.query(SQL_TEMPLATES["has_relationship"], param)
                if has["cnt"] != 0:
                    await self.db.execute(SQL_TEMPLATES["update_relationship"], param)
                    continue

                data.append(param)
            if data:
                merge_sql = SQL_TEMPLATES["insert_relationship"]
                await self.db.execute(merge_sql, data)

    async def get_by_status(self, status: str) -> Union[list[dict[str, Any]], None]:
        SQL = SQL_TEMPLATES["get_by_status_" + self.namespace]
        params = {"workspace": self.db.workspace, "status": status}
        return await self.db.query(SQL, params, multirows=True)

    async def delete_entity(self, entity_name: str) -> None:
        raise NotImplementedError

    async def delete_entity_relation(self, entity_name: str) -> None:
        raise NotImplementedError

    async def index_done_callback(self) -> None:
        # Ti handles persistence automatically
        pass


@final
@dataclass
class TiDBGraphStorage(BaseGraphStorage):
    db: TiDB = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        if self.db is None:
            self.db = await ClientManager.get_client()

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    #################### upsert method ################
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
        sql = SQL_TEMPLATES["upsert_node"]
        data = {
            "workspace": self.db.workspace,
            "name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_chunk_id": source_id,
            "content": content,
            "content_vector": f"{content_vector.tolist()}",
        }
        await self.db.execute(sql, data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
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
        merge_sql = SQL_TEMPLATES["upsert_edge"]
        data = {
            "workspace": self.db.workspace,
            "source_name": source_name,
            "target_name": target_name,
            "weight": weight,
            "keywords": keywords,
            "description": description,
            "source_chunk_id": source_chunk_id,
            "content": content,
            "content_vector": f"{content_vector.tolist()}",
        }
        await self.db.execute(merge_sql, data)

    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    # Query

    async def has_node(self, node_id: str) -> bool:
        sql = SQL_TEMPLATES["has_entity"]
        param = {"name": node_id, "workspace": self.db.workspace}
        has = await self.db.query(sql, param)
        return has["cnt"] != 0

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        sql = SQL_TEMPLATES["has_relationship"]
        param = {
            "source_name": source_node_id,
            "target_name": target_node_id,
            "workspace": self.db.workspace,
        }
        has = await self.db.query(sql, param)
        return has["cnt"] != 0

    async def node_degree(self, node_id: str) -> int:
        sql = SQL_TEMPLATES["node_degree"]
        param = {"name": node_id, "workspace": self.db.workspace}
        result = await self.db.query(sql, param)
        return result["cnt"]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        degree = await self.node_degree(src_id) + await self.node_degree(tgt_id)
        return degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        sql = SQL_TEMPLATES["get_node"]
        param = {"name": node_id, "workspace": self.db.workspace}
        return await self.db.query(sql, param)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        sql = SQL_TEMPLATES["get_edge"]
        param = {
            "source_name": source_node_id,
            "target_name": target_node_id,
            "workspace": self.db.workspace,
        }
        return await self.db.query(sql, param)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        sql = SQL_TEMPLATES["get_node_edges"]
        param = {"source_name": source_node_id, "workspace": self.db.workspace}
        res = await self.db.query(sql, param, multirows=True)
        if res:
            data = [(i["source_name"], i["target_name"]) for i in res]
            return data
        else:
            return []

    async def index_done_callback(self) -> None:
        # Ti handles persistence automatically
        pass

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
N_ID = {
    NameSpace.KV_STORE_FULL_DOCS: "doc_id",
    NameSpace.KV_STORE_TEXT_CHUNKS: "chunk_id",
    NameSpace.VECTOR_STORE_CHUNKS: "chunk_id",
    NameSpace.VECTOR_STORE_ENTITIES: "entity_id",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "relation_id",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in N_T.items():
        if is_namespace(namespace, k):
            return v


def namespace_to_id(namespace: str) -> str:
    for k, v in N_ID.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """
        CREATE TABLE LIGHTRAG_DOC_FULL (
            `id` BIGINT PRIMARY KEY AUTO_RANDOM,
            `doc_id` VARCHAR(256) NOT NULL,
            `workspace` varchar(1024),
            `content` LONGTEXT,
            `meta` JSON,
            `createtime` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `updatetime` TIMESTAMP DEFAULT NULL,
            UNIQUE KEY (`doc_id`)
        );
        """
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """
        CREATE TABLE LIGHTRAG_DOC_CHUNKS (
            `id` BIGINT PRIMARY KEY AUTO_RANDOM,
            `chunk_id` VARCHAR(256) NOT NULL,
            `full_doc_id` VARCHAR(256) NOT NULL,
            `workspace` varchar(1024),
            `chunk_order_index` INT,
            `tokens` INT,
            `content` LONGTEXT,
            `content_vector` VECTOR,
            `createtime` DATETIME DEFAULT CURRENT_TIMESTAMP,
            `updatetime` DATETIME DEFAULT NULL,
            UNIQUE KEY (`chunk_id`)
        );
        """
    },
    "LIGHTRAG_GRAPH_NODES": {
        "ddl": """
        CREATE TABLE LIGHTRAG_GRAPH_NODES (
            `id` BIGINT PRIMARY KEY AUTO_RANDOM,
            `entity_id`  VARCHAR(256),
            `workspace` varchar(1024),
            `name` VARCHAR(2048),
            `entity_type` VARCHAR(1024),
            `description` LONGTEXT,
            `source_chunk_id` VARCHAR(256),
            `content` LONGTEXT,
            `content_vector` VECTOR,
            `createtime` DATETIME DEFAULT CURRENT_TIMESTAMP,
            `updatetime` DATETIME DEFAULT NULL,
            KEY (`entity_id`)
        );
        """
    },
    "LIGHTRAG_GRAPH_EDGES": {
        "ddl": """
        CREATE TABLE LIGHTRAG_GRAPH_EDGES (
            `id` BIGINT PRIMARY KEY AUTO_RANDOM,
            `relation_id`  VARCHAR(256),
            `workspace` varchar(1024),
            `source_name` VARCHAR(2048),
            `target_name` VARCHAR(2048),
            `weight` DECIMAL,
            `keywords` TEXT,
            `description` LONGTEXT,
            `source_chunk_id` varchar(256),
            `content` LONGTEXT,
            `content_vector` VECTOR,
            `createtime` DATETIME DEFAULT CURRENT_TIMESTAMP,
            `updatetime` DATETIME DEFAULT NULL,
            KEY (`relation_id`)
        );
        """
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """
        CREATE TABLE LIGHTRAG_LLM_CACHE (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            send TEXT,
            return TEXT,
            model VARCHAR(1024),
            createtime DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatetime DATETIME DEFAULT NULL
        );
        """
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": "SELECT doc_id as id, IFNULL(content, '') AS content FROM LIGHTRAG_DOC_FULL WHERE doc_id = :id AND workspace = :workspace",
    "get_by_id_text_chunks": "SELECT chunk_id as id, tokens, IFNULL(content, '') AS content, chunk_order_index, full_doc_id FROM LIGHTRAG_DOC_CHUNKS WHERE chunk_id = :id AND workspace = :workspace",
    "get_by_ids_full_docs": "SELECT doc_id as id, IFNULL(content, '') AS content FROM LIGHTRAG_DOC_FULL WHERE doc_id IN ({ids}) AND workspace = :workspace",
    "get_by_ids_text_chunks": "SELECT chunk_id as id, tokens, IFNULL(content, '') AS content, chunk_order_index, full_doc_id FROM LIGHTRAG_DOC_CHUNKS WHERE chunk_id IN ({ids}) AND workspace = :workspace",
    "filter_keys": "SELECT {id_field} AS id FROM {table_name} WHERE {id_field} IN ({ids}) AND workspace = :workspace",
    # SQL for Merge operations (TiDB version with INSERT ... ON DUPLICATE KEY UPDATE)
    "upsert_doc_full": """
        INSERT INTO LIGHTRAG_DOC_FULL (doc_id, content, workspace)
        VALUES (:id, :content, :workspace)
        ON DUPLICATE KEY UPDATE content = VALUES(content), workspace = VALUES(workspace), updatetime = CURRENT_TIMESTAMP
    """,
    "upsert_chunk": """
        INSERT INTO LIGHTRAG_DOC_CHUNKS(chunk_id, content, tokens, chunk_order_index, full_doc_id, content_vector, workspace)
        VALUES (:id, :content, :tokens, :chunk_order_index, :full_doc_id, :content_vector, :workspace)
        ON DUPLICATE KEY UPDATE
        content = VALUES(content), tokens = VALUES(tokens), chunk_order_index = VALUES(chunk_order_index),
        full_doc_id = VALUES(full_doc_id), content_vector = VALUES(content_vector), workspace = VALUES(workspace), updatetime = CURRENT_TIMESTAMP
    """,
    # SQL for VectorStorage
    "entities": """SELECT n.name as entity_name FROM
        (SELECT entity_id as id, name, VEC_COSINE_DISTANCE(content_vector,:embedding_string) as distance
        FROM LIGHTRAG_GRAPH_NODES WHERE workspace = :workspace) n
        WHERE n.distance>:better_than_threshold ORDER BY n.distance DESC LIMIT :top_k
    """,
    "relationships": """SELECT e.source_name as src_id, e.target_name as tgt_id FROM
        (SELECT source_name, target_name, VEC_COSINE_DISTANCE(content_vector, :embedding_string) as distance
        FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = :workspace) e
        WHERE e.distance>:better_than_threshold ORDER BY e.distance DESC LIMIT :top_k
    """,
    "chunks": """SELECT c.id FROM
        (SELECT chunk_id as id,VEC_COSINE_DISTANCE(content_vector, :embedding_string) as distance
        FROM LIGHTRAG_DOC_CHUNKS WHERE workspace = :workspace) c
        WHERE c.distance>:better_than_threshold ORDER BY c.distance DESC LIMIT :top_k
    """,
    "has_entity": """
        SELECT COUNT(id) AS cnt FROM LIGHTRAG_GRAPH_NODES WHERE name = :name AND workspace = :workspace
    """,
    "has_relationship": """
        SELECT COUNT(id) AS cnt FROM LIGHTRAG_GRAPH_EDGES WHERE source_name = :source_name AND target_name = :target_name AND workspace = :workspace
    """,
    "update_entity": """
        UPDATE LIGHTRAG_GRAPH_NODES SET
            entity_id = :id, content = :content, content_vector = :content_vector, updatetime = CURRENT_TIMESTAMP
        WHERE workspace = :workspace AND name = :name
    """,
    "update_relationship": """
        UPDATE LIGHTRAG_GRAPH_EDGES SET
            relation_id = :id, content = :content, content_vector = :content_vector, updatetime = CURRENT_TIMESTAMP
        WHERE workspace = :workspace AND source_name = :source_name AND target_name = :target_name
    """,
    "insert_entity": """
        INSERT INTO LIGHTRAG_GRAPH_NODES(entity_id, name, content, content_vector, workspace)
        VALUES(:id, :name, :content, :content_vector, :workspace)
    """,
    "insert_relationship": """
        INSERT INTO LIGHTRAG_GRAPH_EDGES(relation_id, source_name, target_name, content, content_vector, workspace)
        VALUES(:id, :source_name, :target_name, :content, :content_vector, :workspace)
    """,
    # SQL for GraphStorage
    "get_node": """
        SELECT entity_id AS id, workspace, name, entity_type, description, source_chunk_id AS source_id, content, content_vector
        FROM LIGHTRAG_GRAPH_NODES WHERE name = :name AND workspace = :workspace
    """,
    "get_edge": """
        SELECT relation_id AS id, workspace, source_name, target_name, weight, keywords, description, source_chunk_id AS source_id, content, content_vector
        FROM LIGHTRAG_GRAPH_EDGES WHERE source_name = :source_name AND target_name = :target_name AND workspace = :workspace
    """,
    "get_node_edges": """
        SELECT relation_id AS id, workspace, source_name, target_name, weight, keywords, description, source_chunk_id, content, content_vector
        FROM LIGHTRAG_GRAPH_EDGES WHERE source_name = :source_name AND workspace = :workspace
    """,
    "node_degree": """
        SELECT COUNT(id) AS cnt FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = :workspace AND :name IN (source_name, target_name)
    """,
    "upsert_node": """
        INSERT INTO LIGHTRAG_GRAPH_NODES(name, content, content_vector, workspace, source_chunk_id, entity_type, description)
        VALUES(:name, :content, :content_vector, :workspace, :source_chunk_id, :entity_type, :description)
        ON DUPLICATE KEY UPDATE
        name = VALUES(name), content = VALUES(content), content_vector = VALUES(content_vector),
        workspace = VALUES(workspace), updatetime = CURRENT_TIMESTAMP,
        source_chunk_id = VALUES(source_chunk_id), entity_type = VALUES(entity_type), description = VALUES(description)
    """,
    "upsert_edge": """
        INSERT INTO LIGHTRAG_GRAPH_EDGES(source_name, target_name, content, content_vector,
            workspace, weight, keywords, description, source_chunk_id)
        VALUES(:source_name, :target_name, :content, :content_vector,
            :workspace, :weight, :keywords, :description, :source_chunk_id)
        ON DUPLICATE KEY UPDATE
        source_name = VALUES(source_name), target_name = VALUES(target_name), content = VALUES(content),
        content_vector = VALUES(content_vector), workspace = VALUES(workspace), updatetime = CURRENT_TIMESTAMP,
        weight = VALUES(weight), keywords = VALUES(keywords), description = VALUES(description),
        source_chunk_id = VALUES(source_chunk_id)
    """,
}
