import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Union, final

import numpy as np

from lightrag.types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge


from ..base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage
from ..namespace import NameSpace, is_namespace
from ..utils import logger

import pipmaster as pm
import configparser

if not pm.is_installed("pymysql"):
    pm.install("pymysql")
if not pm.is_installed("sqlalchemy"):
    pm.install("sqlalchemy")

from sqlalchemy import create_engine, text  # type: ignore


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
    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage

        Returns:
            Dictionary containing all stored data
        """
        async with self._storage_lock:
            return dict(self._data)

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

    async def delete(self, ids: list[str]) -> None:
        """Delete records with specified IDs from the storage.

        Args:
            ids: List of record IDs to be deleted
        """
        if not ids:
            return

        try:
            table_name = namespace_to_table_name(self.namespace)
            id_field = namespace_to_id(self.namespace)

            if not table_name or not id_field:
                logger.error(f"Unknown namespace for deletion: {self.namespace}")
                return

            ids_list = ",".join([f"'{id}'" for id in ids])
            delete_sql = f"DELETE FROM {table_name} WHERE workspace = :workspace AND {id_field} IN ({ids_list})"

            await self.db.execute(delete_sql, {"workspace": self.db.workspace})
            logger.info(
                f"Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error deleting records from {self.namespace}: {e}")

    async def drop_cache_by_modes(self, modes: list[str] | None = None) -> bool:
        """Delete specific records from storage by cache mode

        Args:
            modes (list[str]): List of cache modes to be dropped from storage

        Returns:
            bool: True if successful, False otherwise
        """
        if not modes:
            return False

        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return False

            if table_name != "LIGHTRAG_LLM_CACHE":
                return False

            # 构建MySQL风格的IN查询
            modes_list = ", ".join([f"'{mode}'" for mode in modes])
            sql = f"""
            DELETE FROM {table_name}
            WHERE workspace = :workspace
            AND mode IN ({modes_list})
            """

            logger.info(f"Deleting cache by modes: {modes}")
            await self.db.execute(sql, {"workspace": self.db.workspace})
            return True
        except Exception as e:
            logger.error(f"Error deleting cache by modes {modes}: {e}")
            return False

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


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

    async def query(
        self, query: str, top_k: int, ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Search from tidb vector"""
        embeddings = await self.embedding_func(
            [query], _priority=5
        )  # higher priority for query
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

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage.

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        id_field = namespace_to_id(self.namespace)

        if not table_name or not id_field:
            logger.error(f"Unknown namespace for vector deletion: {self.namespace}")
            return

        ids_list = ",".join([f"'{id}'" for id in ids])
        delete_sql = f"DELETE FROM {table_name} WHERE workspace = :workspace AND {id_field} IN ({ids_list})"

        try:
            await self.db.execute(delete_sql, {"workspace": self.db.workspace})
            logger.debug(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Construct SQL to delete the entity
            delete_sql = """DELETE FROM LIGHTRAG_GRAPH_NODES
                            WHERE workspace = :workspace AND name = :entity_name"""

            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "entity_name": entity_name}
            )
            logger.debug(f"Successfully deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity.

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_GRAPH_EDGES
                            WHERE workspace = :workspace AND (source_name = :entity_name OR target_name = :entity_name)"""

            await self.db.execute(
                delete_sql, {"workspace": self.db.workspace, "entity_name": entity_name}
            )
            logger.debug(f"Successfully deleted relations for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for entity {entity_name}: {e}")

    async def index_done_callback(self) -> None:
        # Ti handles persistence automatically
        pass

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search_by_prefix(self, prefix: str) -> list[dict[str, Any]]:
        """Search for records with IDs starting with a specific prefix.

        Args:
            prefix: The prefix to search for in record IDs

        Returns:
            List of records with matching ID prefixes
        """
        # Determine which table to query based on namespace
        if self.namespace == NameSpace.VECTOR_STORE_ENTITIES:
            sql_template = """
                SELECT entity_id as id, name as entity_name, entity_type, description, content
                FROM LIGHTRAG_GRAPH_NODES
                WHERE entity_id LIKE :prefix_pattern AND workspace = :workspace
            """
        elif self.namespace == NameSpace.VECTOR_STORE_RELATIONSHIPS:
            sql_template = """
                SELECT relation_id as id, source_name as src_id, target_name as tgt_id,
                       keywords, description, content
                FROM LIGHTRAG_GRAPH_EDGES
                WHERE relation_id LIKE :prefix_pattern AND workspace = :workspace
            """
        elif self.namespace == NameSpace.VECTOR_STORE_CHUNKS:
            sql_template = """
                SELECT chunk_id as id, content, tokens, chunk_order_index, full_doc_id
                FROM LIGHTRAG_DOC_CHUNKS
                WHERE chunk_id LIKE :prefix_pattern AND workspace = :workspace
            """
        else:
            logger.warning(
                f"Namespace {self.namespace} not supported for prefix search"
            )
            return []

        # Add prefix pattern parameter with % for SQL LIKE
        prefix_pattern = f"{prefix}%"
        params = {"prefix_pattern": prefix_pattern, "workspace": self.db.workspace}

        try:
            results = await self.db.query(sql_template, params=params, multirows=True)
            logger.debug(
                f"Found {len(results) if results else 0} records with prefix '{prefix}'"
            )
            return results if results else []
        except Exception as e:
            logger.error(f"Error searching records with prefix '{prefix}': {e}")
            return []

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Determine which table to query based on namespace
            if self.namespace == NameSpace.VECTOR_STORE_ENTITIES:
                sql_template = """
                    SELECT entity_id as id, name as entity_name, entity_type, description, content
                    FROM LIGHTRAG_GRAPH_NODES
                    WHERE entity_id = :entity_id AND workspace = :workspace
                """
                params = {"entity_id": id, "workspace": self.db.workspace}
            elif self.namespace == NameSpace.VECTOR_STORE_RELATIONSHIPS:
                sql_template = """
                    SELECT relation_id as id, source_name as src_id, target_name as tgt_id,
                           keywords, description, content
                    FROM LIGHTRAG_GRAPH_EDGES
                    WHERE relation_id = :relation_id AND workspace = :workspace
                """
                params = {"relation_id": id, "workspace": self.db.workspace}
            elif self.namespace == NameSpace.VECTOR_STORE_CHUNKS:
                sql_template = """
                    SELECT chunk_id as id, content, tokens, chunk_order_index, full_doc_id
                    FROM LIGHTRAG_DOC_CHUNKS
                    WHERE chunk_id = :chunk_id AND workspace = :workspace
                """
                params = {"chunk_id": id, "workspace": self.db.workspace}
            else:
                logger.warning(
                    f"Namespace {self.namespace} not supported for get_by_id"
                )
                return None

            result = await self.db.query(sql_template, params=params)
            return result
        except Exception as e:
            logger.error(f"Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Format IDs for SQL IN clause
            ids_str = ", ".join([f"'{id}'" for id in ids])

            # Determine which table to query based on namespace
            if self.namespace == NameSpace.VECTOR_STORE_ENTITIES:
                sql_template = f"""
                    SELECT entity_id as id, name as entity_name, entity_type, description, content
                    FROM LIGHTRAG_GRAPH_NODES
                    WHERE entity_id IN ({ids_str}) AND workspace = :workspace
                """
            elif self.namespace == NameSpace.VECTOR_STORE_RELATIONSHIPS:
                sql_template = f"""
                    SELECT relation_id as id, source_name as src_id, target_name as tgt_id,
                           keywords, description, content
                    FROM LIGHTRAG_GRAPH_EDGES
                    WHERE relation_id IN ({ids_str}) AND workspace = :workspace
                """
            elif self.namespace == NameSpace.VECTOR_STORE_CHUNKS:
                sql_template = f"""
                    SELECT chunk_id as id, content, tokens, chunk_order_index, full_doc_id
                    FROM LIGHTRAG_DOC_CHUNKS
                    WHERE chunk_id IN ({ids_str}) AND workspace = :workspace
                """
            else:
                logger.warning(
                    f"Namespace {self.namespace} not supported for get_by_ids"
                )
                return []

            params = {"workspace": self.db.workspace}
            results = await self.db.query(sql_template, params=params, multirows=True)
            return results if results else []
        except Exception as e:
            logger.error(f"Error retrieving vector data for IDs {ids}: {e}")
            return []


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

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            drop_sql = """
                DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = :workspace;
                DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace = :workspace;
            """
            await self.db.execute(drop_sql, {"workspace": self.db.workspace})
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its related edges

        Args:
            node_id: The ID of the node to delete
        """
        # First delete all edges related to this node
        await self.db.execute(
            SQL_TEMPLATES["delete_node_edges"],
            {"name": node_id, "workspace": self.db.workspace},
        )

        # Then delete the node itself
        await self.db.execute(
            SQL_TEMPLATES["delete_node"],
            {"name": node_id, "workspace": self.db.workspace},
        )

        logger.debug(
            f"Node {node_id} and its related edges have been deleted from the graph"
        )

    async def get_all_labels(self) -> list[str]:
        """Get all entity types (labels) in the database

        Returns:
            List of labels sorted alphabetically
        """
        result = await self.db.query(
            SQL_TEMPLATES["get_all_labels"],
            {"workspace": self.db.workspace},
            multirows=True,
        )

        if not result:
            return []

        # Extract all labels
        return [item["label"] for item in result]

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 5
    ) -> KnowledgeGraph:
        """
        Get a connected subgraph of nodes matching the specified label
        Maximum number of nodes is limited by MAX_GRAPH_NODES environment variable (default: 1000)

        Args:
            node_label: The node label to match
            max_depth: Maximum depth of the subgraph

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        result = KnowledgeGraph()
        MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

        # Get matching nodes
        if node_label == "*":
            # Handle special case, get all nodes
            node_results = await self.db.query(
                SQL_TEMPLATES["get_all_nodes"],
                {"workspace": self.db.workspace, "max_nodes": MAX_GRAPH_NODES},
                multirows=True,
            )
        else:
            # Get nodes matching the label
            label_pattern = f"%{node_label}%"
            node_results = await self.db.query(
                SQL_TEMPLATES["get_matching_nodes"],
                {"workspace": self.db.workspace, "label_pattern": label_pattern},
                multirows=True,
            )

        if not node_results:
            logger.warning(f"No nodes found matching label {node_label}")
            return result

        # Limit the number of returned nodes
        if len(node_results) > MAX_GRAPH_NODES:
            node_results = node_results[:MAX_GRAPH_NODES]

        # Extract node names for edge query
        node_names = [node["name"] for node in node_results]
        node_names_str = ",".join([f"'{name}'" for name in node_names])

        # Add nodes to result
        for node in node_results:
            node_properties = {
                k: v for k, v in node.items() if k not in ["id", "name", "entity_type"]
            }
            result.nodes.append(
                KnowledgeGraphNode(
                    id=node["name"],
                    labels=[node["entity_type"]]
                    if node.get("entity_type")
                    else [node["name"]],
                    properties=node_properties,
                )
            )

        # Get related edges
        edge_results = await self.db.query(
            SQL_TEMPLATES["get_related_edges"].format(node_names=node_names_str),
            {"workspace": self.db.workspace},
            multirows=True,
        )

        if edge_results:
            # Add edges to result
            for edge in edge_results:
                # Only include edges related to selected nodes
                if (
                    edge["source_name"] in node_names
                    and edge["target_name"] in node_names
                ):
                    edge_id = f"{edge['source_name']}-{edge['target_name']}"
                    edge_properties = {
                        k: v
                        for k, v in edge.items()
                        if k not in ["id", "source_name", "target_name"]
                    }

                    result.edges.append(
                        KnowledgeGraphEdge(
                            id=edge_id,
                            type="RELATED",
                            source=edge["source_name"],
                            target=edge["target_name"],
                            properties=edge_properties,
                        )
                    )

        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node IDs to delete
        """
        for node_id in nodes:
            await self.delete_node(node_id)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to delete, each edge is a (source, target) tuple
        """
        for source, target in edges:
            await self.db.execute(
                SQL_TEMPLATES["remove_multiple_edges"],
                {"source": source, "target": target, "workspace": self.db.workspace},
            )


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
    "delete_node": """
        DELETE FROM LIGHTRAG_GRAPH_NODES
        WHERE name = :name AND workspace = :workspace
    """,
    "delete_node_edges": """
        DELETE FROM LIGHTRAG_GRAPH_EDGES
        WHERE (source_name = :name OR target_name = :name) AND workspace = :workspace
    """,
    "get_all_labels": """
        SELECT DISTINCT entity_type as label
        FROM LIGHTRAG_GRAPH_NODES
        WHERE workspace = :workspace
        ORDER BY entity_type
    """,
    "get_matching_nodes": """
        SELECT * FROM LIGHTRAG_GRAPH_NODES
        WHERE name LIKE :label_pattern AND workspace = :workspace
        ORDER BY name
    """,
    "get_all_nodes": """
        SELECT * FROM LIGHTRAG_GRAPH_NODES
        WHERE workspace = :workspace
        ORDER BY name
        LIMIT :max_nodes
    """,
    "get_related_edges": """
        SELECT * FROM LIGHTRAG_GRAPH_EDGES
        WHERE (source_name IN (:node_names) OR target_name IN (:node_names))
        AND workspace = :workspace
    """,
    "remove_multiple_edges": """
        DELETE FROM LIGHTRAG_GRAPH_EDGES
        WHERE (source_name = :source AND target_name = :target)
        AND workspace = :workspace
    """,
    # Search by prefix SQL templates
    "search_entity_by_prefix": """
        SELECT entity_id as id, name as entity_name, entity_type, description, content
        FROM LIGHTRAG_GRAPH_NODES
        WHERE entity_id LIKE :prefix_pattern AND workspace = :workspace
    """,
    "search_relationship_by_prefix": """
        SELECT relation_id as id, source_name as src_id, target_name as tgt_id, keywords, description, content
        FROM LIGHTRAG_GRAPH_EDGES
        WHERE relation_id LIKE :prefix_pattern AND workspace = :workspace
    """,
    "search_chunk_by_prefix": """
        SELECT chunk_id as id, content, tokens, chunk_order_index, full_doc_id
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE chunk_id LIKE :prefix_pattern AND workspace = :workspace
    """,
    # Drop tables
    "drop_specifiy_table_workspace": "DELETE FROM {table_name} WHERE workspace = :workspace",
}
