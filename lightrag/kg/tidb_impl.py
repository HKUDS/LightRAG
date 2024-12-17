import asyncio
import os
from dataclasses import dataclass
from typing import Union

import numpy as np
from sqlalchemy import create_engine, text
from tqdm import tqdm

from lightrag.base import BaseVectorStorage, BaseKVStorage, BaseGraphStorage
from lightrag.utils import logger


class TiDB(object):
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
                    # print(v["ddl"])
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
                logger.error(f"Tidb database error: {e}")
                print(sql)
                print(params)
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
            logger.error(f"TiDB database error: {e}")
            print(sql)
            print(data)
            raise


@dataclass
class TiDBKVStorage(BaseKVStorage):
    # should pass db object to self.db
    def __post_init__(self):
        self._data = {}
        self._max_batch_size = self.global_config["embedding_batch_num"]

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> Union[dict, None]:
        """根据 id 获取 doc_full 数据."""
        SQL = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"id": id}
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
        # print("get_by_ids:"+SQL)
        res = await self.db.query(SQL, multirows=True)
        if res:
            data = res  # [{"data":i} for i in res]
            # print(data)
            return data
        else:
            return None

    async def filter_keys(self, keys: list[str]) -> set[str]:
        """过滤掉重复内容"""
        SQL = SQL_TEMPLATES["filter_keys"].format(
            table_name=N_T[self.namespace],
            id_field=N_ID[self.namespace],
            ids=",".join([f"'{id}'" for id in keys]),
        )
        try:
            await self.db.query(SQL)
        except Exception as e:
            logger.error(f"Tidb database error: {e}")
            print(SQL)
        res = await self.db.query(SQL, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
            data = set([s for s in keys if s not in exist_keys])
        else:
            exist_keys = []
            data = set([s for s in keys if s not in exist_keys])
        return data

    ################ INSERT full_doc AND chunks ################
    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
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
                        "content_vector": f"{item["__vector__"].tolist()}",
                        "workspace": self.db.workspace,
                    }
                )
            await self.db.execute(merge_sql, data)

        if self.namespace == "full_docs":
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

    async def index_done_callback(self):
        if self.namespace in ["full_docs", "text_chunks"]:
            logger.info("full doc and chunk data had been saved into TiDB db!")


@dataclass
class TiDBVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    async def query(self, query: str, top_k: int) -> list[dict]:
        """search from tidb vector"""

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
    async def upsert(self, data: dict[str, dict]):
        # ignore, upsert in TiDBKVStorage already
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        if self.namespace == "chunks":
            return []
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
        embeddings_list = []
        for f in tqdm(
            asyncio.as_completed(embedding_tasks),
            total=len(embedding_tasks),
            desc="Generating embeddings",
            unit="batch",
        ):
            embeddings = await f
            embeddings_list.append(embeddings)
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["content_vector"] = embeddings[i]

        if self.namespace == "entities":
            data = []
            for item in list_data:
                param = {
                    "id": item["id"],
                    "name": item["entity_name"],
                    "content": item["content"],
                    "content_vector": f"{item["content_vector"].tolist()}",
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

        elif self.namespace == "relationships":
            data = []
            for item in list_data:
                param = {
                    "id": item["id"],
                    "source_name": item["src_id"],
                    "target_name": item["tgt_id"],
                    "content": item["content"],
                    "content_vector": f"{item["content_vector"].tolist()}",
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


@dataclass
class TiDBGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    #################### upsert method ################
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
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
    ):
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

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
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

    async def get_node(self, node_id: str) -> Union[dict, None]:
        sql = SQL_TEMPLATES["get_node"]
        param = {"name": node_id, "workspace": self.db.workspace}
        return await self.db.query(sql, param)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        sql = SQL_TEMPLATES["get_edge"]
        param = {
            "source_name": source_node_id,
            "target_name": target_node_id,
            "workspace": self.db.workspace,
        }
        return await self.db.query(sql, param)

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        sql = SQL_TEMPLATES["get_node_edges"]
        param = {"source_name": source_node_id, "workspace": self.db.workspace}
        res = await self.db.query(sql, param, multirows=True)
        if res:
            data = [(i["source_name"], i["target_name"]) for i in res]
            return data
        else:
            return []


N_T = {
    "full_docs": "LIGHTRAG_DOC_FULL",
    "text_chunks": "LIGHTRAG_DOC_CHUNKS",
    "chunks": "LIGHTRAG_DOC_CHUNKS",
    "entities": "LIGHTRAG_GRAPH_NODES",
    "relationships": "LIGHTRAG_GRAPH_EDGES",
}
N_ID = {
    "full_docs": "doc_id",
    "text_chunks": "chunk_id",
    "chunks": "chunk_id",
    "entities": "entity_id",
    "relationships": "relation_id",
}

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
