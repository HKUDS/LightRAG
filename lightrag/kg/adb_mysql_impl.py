import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Union, final
import numpy as np

from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger
from ..kg.shared_storage import get_data_init_lock

import pipmaster

if not pipmaster.is_installed("aiomysql"):
    pipmaster.install("aiomysql")

import aiomysql

from dotenv import load_dotenv

# use the .env that is inside the current folder
load_dotenv(dotenv_path=".env", override=False)


class AnalyticDB:
    """AnalyticDB MySQL

    For more information, please visit
        [AnalyticDB MySQL official site](https://www.alibabacloud.com/en/product/analyticdb-for-mysql)

    For Example:
    .. code-block:: python
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            ),
            #rerank_model_func=rerank_model_func,
            #tiktoken_model_name="gpt-4o-mini",
            #graph_storage="NetworkXStorage",
            kv_storage="ADBKVStorage",
            vector_storage="ADBVectorStorage",
            doc_status_storage="ADBDocStatusStorage",
        )
    """

    def __init__(self, **kwargs: Any):
        self._lock = asyncio.Lock()

        self.db_config = {
            "host": os.getenv("ADB_HOST", "localhost"),
            "port": int(os.getenv("ADB_PORT", "3306")),
            "user": os.getenv("ADB_USER"),
            "password": os.getenv("ADB_PASSWORD"),
            "db": os.getenv("ADB_DATABASE"),
            "maxsize": int(os.getenv("ADB_MAX_CONNECTIONS", "5")),
            "autocommit": True,
        }
        self.workspace = os.getenv("ADB_WORKSPACE", "graphrag")
        self.pool = None

        if not all([self.db_config["user"], self.db_config["password"], self.db_config["db"]]):
            raise ValueError("Missing database user, password, or database")

    async def initdb(self):
        # init pool
        try:
            self.pool = await aiomysql.create_pool(**self.db_config, cursorclass=aiomysql.DictCursor)
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, Failed to connect database, Got:{e}")
            raise e

        # check tables
        for k, v in TABLES.items():
            try:
                result = await self.query(
                    f"SELECT 1 FROM information_schema.kepler_meta_tables "
                    f"where table_schema='{self.db_config["db"]}' and table_name=lower('{k}')"
                )
                if result is None:
                    logger.info(f"AnalyticDB MySQL, Try Creating table {k} in database")
                    await self.execute(v["ddl"])
            except Exception as e:
                logger.error(f"AnalyticDB MySQL, Failed to create table {k} in database, Got: {e}")
                raise e

    async def close_pool(self):
        async with self._lock:
            if self.pool is not None and not self.pool.closed():
                self.pool.terminate()
                await self.pool.wait_closed()

    async def query(
            self,
            sql: str,
            params: dict[str, Any] | None = None,
            multirows: bool = False,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if params is None:
                        await cursor.execute(sql)
                    else:
                        await cursor.execute(sql, params)

                    if multirows:
                        rows = await cursor.fetchall()
                        if rows:
                            columns = rows[0].keys()
                            return [dict(zip(columns, row.values())) for row in rows]
                        return []
                    else:
                        row = await cursor.fetchone()
                        if row:
                            columns = row.keys()
                            return dict(zip(columns, row.values()))
                        return None
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, \nsql:{sql},\nparam:{params},\nerror:{e}")
            raise e

    async def execute(
            self,
            sql: str,
            datas: dict[str, Any] | list[dict[str, Any]] | None = None,
    ):
        try:
            async with self.pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    if datas is None:
                        await cursor.execute(sql)
                    else:
                        if isinstance(datas, list):
                            await cursor.executemany(sql, datas)
                        else:
                            await cursor.execute(sql, datas)
        except Exception as e:
            logger.error(f"AnalyticDB MySQL, \nsql:{sql},\ndata:{datas},\nerror:{e}")
            raise e


@final
@dataclass
class ADBKVStorage(BaseKVStorage):
    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.workspace, "id": id}

        response = await self.db.query(sql, params)

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type"),
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        # Special handling for FULL_ENTITIES namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Parse entity_names JSON string back to list
            entity_names = response.get("entity_names", [])
            if isinstance(entity_names, str):
                try:
                    entity_names = json.loads(entity_names)
                except json.JSONDecodeError:
                    entity_names = []
            response["entity_names"] = entity_names
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Parse relation_pairs JSON string back to list
            relation_pairs = response.get("relation_pairs", [])
            if isinstance(relation_pairs, str):
                try:
                    relation_pairs = json.loads(relation_pairs)
                except json.JSONDecodeError:
                    relation_pairs = []
            response["relation_pairs"] = relation_pairs
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        return response if response else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace]
        params = {"workspace": self.workspace, "ids": ids_str}

        results = await self.db.query(sql, params, multirows=True)

        def _order_results(rows: list[dict[str, Any]] | None, ) -> list[dict[str, Any] | None]:
            """Preserve the caller requested ordering for bulk id lookups."""
            if not rows:
                return [None for _ in ids]

            id_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = row

            ordered: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered.append(id_map.get(str(requested_id)))
            return ordered

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list for each result
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)

            return _order_results(processed_results)

        # Special handling for FULL_ENTITIES namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for result in results:
                # Parse entity_names JSON string back to list
                entity_names = result.get("entity_names", [])
                if isinstance(entity_names, str):
                    try:
                        entity_names = json.loads(entity_names)
                    except json.JSONDecodeError:
                        entity_names = []
                result["entity_names"] = entity_names
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for result in results:
                # Parse relation_pairs JSON string back to list
                relation_pairs = result.get("relation_pairs", [])
                if isinstance(relation_pairs, str):
                    try:
                        relation_pairs = json.loads(relation_pairs)
                    except json.JSONDecodeError:
                        relation_pairs = []
                result["relation_pairs"] = relation_pairs
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result["chunk_ids"] = chunk_ids
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        return _order_results(results)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)

        ids_str = ",".join([f"'{id}'" for id in keys])
        sql = f"SELECT id FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        res = await self.db.query(sql, params, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
        else:
            exist_keys = []
        new_keys = set([s for s in keys if s not in exist_keys])
        return new_keys

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_text_chunk"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "tokens": v["tokens"],
                    "chunk_order_index": v["chunk_order_index"],
                    "full_doc_id": v["full_doc_id"],
                    "content": v["content"],
                    "file_path": v["file_path"],
                    "llm_cache_list": json.dumps(v.get("llm_cache_list", [])),
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
            for k, v in data.items():
                _data = {
                    "id": k,
                    "content": v["content"],
                    "doc_name": v.get("file_path", ""),  # Map file_path to doc_name
                    "workspace": self.workspace,
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,  # Use flattened key as id
                    "original_prompt": v["original_prompt"],
                    "return_value": v["return"],
                    "chunk_id": v.get("chunk_id"),
                    "cache_type": v.get(
                        "cache_type", "extract"
                    ),  # Get cache_type from data
                    "queryparam": json.dumps(v.get("queryparam"))
                    if v.get("queryparam")
                    else None,
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_full_entities"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "entity_names": json.dumps(v["entity_names"]),
                    "count": v["count"],
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_full_relations"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "relation_pairs": json.dumps(v["relation_pairs"]),
                    "count": v["count"],
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_entity_chunks"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "chunk_ids": json.dumps(v["chunk_ids"]),
                    "count": v["count"],
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)
        elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            datas = []
            upsert_sql = SQL_TEMPLATES["upsert_relation_chunks"]
            for k, v in data.items():
                _data = {
                    "workspace": self.workspace,
                    "id": k,
                    "chunk_ids": json.dumps(v["chunk_ids"]),
                    "count": v["count"],
                }
                datas.append(_data)
                if len(datas) == self._max_batch_size:
                    await self.db.execute(upsert_sql, datas)
                    datas = []
            if len(datas) > 0:
                await self.db.execute(upsert_sql, datas)

    async def index_done_callback(self) -> None:
        pass

    async def is_empty(self) -> bool:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}")
            return True

        sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=%(workspace)s LIMIT 1) as has_data"

        try:
            result = await self.db.query(sql, {"workspace": self.workspace})

            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}")
            return

        ids_str = ",".join([f"'{id}'" for id in ids])
        delete_sql = f"DELETE FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        try:
            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}")

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(table_name=table_name)

            await self.db.execute(drop_sql, {"workspace": self.workspace})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBVectorStorage(BaseVectorStorage):
    db: AnalyticDB | None = field(default=None)

    def __post_init__(self):
        if self.embedding_func is None:
            raise ValueError("embedding_func is required for vector storage")
        self._max_batch_size = self.global_config["embedding_batch_num"]
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = config.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError("cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs")
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            # check vector tables
            for k, v in VECTOR_TABLES.items():
                try:
                    result = await self.db.query(
                        f"SELECT 1 FROM information_schema.kepler_meta_tables "
                        f"where table_schema='{self.db.db_config["db"]}' and table_name=lower('{k}')"
                    )
                    if result is None:
                        logger.info(f"AnalyticDB MySQL, Try Creating vector table {k} in database")
                        embeddings = await self.embedding_func(["adb"])
                        ddl = v["ddl"].replace(
                            "ARRAY<FLOAT>(EMBEDDING_DIM)",
                            f"ARRAY<FLOAT>({len(embeddings[0])})"
                        )
                        await self.db.execute(ddl)
                except Exception as e:
                    logger.error(f"AnalyticDB MySQL, Failed to create vector table {k} in database, Got: {e}")
                    raise e

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    def _upsert_chunks(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_chunk"]
        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "tokens": item["tokens"],
            "chunk_order_index": item["chunk_order_index"],
            "full_doc_id": item["full_doc_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "file_path": item["file_path"],
        }
        return upsert_sql, data

    def _upsert_entities(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_entity"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": json.dumps(chunk_ids),
            "file_path": item.get("file_path", None),
        }
        return upsert_sql, data

    def _upsert_relationships(self, item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        upsert_sql = SQL_TEMPLATES["upsert_relationship"]
        source_id = item["source_id"]
        if isinstance(source_id, str) and "<SEP>" in source_id:
            chunk_ids = source_id.split("<SEP>")
        else:
            chunk_ids = [source_id]

        data: dict[str, Any] = {
            "workspace": self.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
            "chunk_ids": json.dumps(chunk_ids),
            "file_path": item.get("file_path", None),
        }
        return upsert_sql, data

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i: i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]

        datas = []
        upsert_sql = ""
        for item in list_data:
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                upsert_sql, data = self._upsert_chunks(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                upsert_sql, data = self._upsert_entities(item)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")
            datas.append(data)
            if len(datas) == self._max_batch_size:
                await self.db.execute(upsert_sql, datas)
                datas = []
        if len(datas) > 0:
            await self.db.execute(upsert_sql, datas)

    async def query(
            self,
            query: str,
            top_k: int,
            query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func([query], _priority=5)
            embedding = embeddings[0]

        embedding_string = ",".join(map(str, embedding))

        sql = SQL_TEMPLATES[self.namespace].format(embedding_string=embedding_string)
        params = {
            "workspace": self.workspace,
            "closer_than_threshold": 1 - self.cosine_better_than_threshold,
            "top_k": top_k,
        }

        results = await self.db.query(sql, params=params, multirows=True)
        return results

    async def index_done_callback(self) -> None:
        pass

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}")
            return

        ids_str = ",".join([f"'{id}'" for id in ids])
        delete_sql = f"DELETE FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        try:
            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            # Construct SQL to delete the entity
            delete_sql = "DELETE FROM LIGHTRAG_VDB_ENTITY WHERE workspace=%(workspace)s AND entity_name=%(entity_name)s"
            params = {"workspace": self.workspace, "entity_name": entity_name}

            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                         WHERE workspace=%(workspace)s AND (source_id=%(entity_name)s OR target_id=%(entity_name)s)
                         """
            params = {"workspace": self.workspace, "entity_name": entity_name}

            await self.db.execute(delete_sql, params)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting relations for entity {entity_name}: {e}")

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}")
            return None

        query = (f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at FROM {table_name} "
                 f"WHERE workspace=%(workspace)s AND id=%(id)s")
        params = {"workspace": self.workspace, "id": id}

        try:
            result = await self.db.query(query, params)
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}")
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}")
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = (f"SELECT *, UNIX_TIMESTAMP(create_time) as created_at FROM {table_name} "
                 f"WHERE workspace=%(workspace)s AND id IN (%(ids)s)")
        params = {"workspace": self.workspace, "ids": ids_str}

        try:
            results = await self.db.query(query, params, multirows=True)
            if not results:
                return []

            # Preserve caller requested ordering while normalizing asyncpg rows to dicts.
            id_map: dict[str, dict[str, Any]] = {}
            for record in results:
                if record is None:
                    continue
                record_dict = dict(record)
                row_id = record_dict.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = record_dict

            ordered_results: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered_results.append(id_map.get(str(requested_id)))
            return ordered_results
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}")
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}")
            return {}

        ids_str = ",".join([f"'{id}'" for id in ids])
        query = f"SELECT id, content_vector FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        try:
            results = await self.db.query(query, params, multirows=True)
            vectors_dict = {}

            for result in results:
                if result and "content_vector" in result and "id" in result:
                    try:
                        # Parse JSON string to get vector as list of floats
                        vector_data = result["content_vector"]
                        if hasattr(vector_data, "tolist"):
                            vectors_dict[result["id"]] = vector_data.tolist()
                        else:
                            vectors_dict[result["id"]] = json.loads(vector_data)
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f"[{self.workspace}] Failed to parse vector data for ID {result['id']}: {e}")

            return vectors_dict
        except Exception as e:
            logger.error(f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}")
            return {}

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(table_name=table_name)
            await self.db.execute(drop_sql, {"workspace": self.workspace})

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
class ADBDocStatusStorage(DocStatusStorage):
    db: AnalyticDB | None = field(default=None)

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = AnalyticDB()
                await self.db.initdb()

            # Implement workspace priority: ADB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use ADB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

    async def finalize(self):
        if self.db is not None:
            await self.db.close_pool()
            self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)

        ids_str = ",".join([f"'{id}'" for id in keys])
        sql = f"SELECT id FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        res = await self.db.query(sql, params, multirows=True)
        if res:
            exist_keys = [key["id"] for key in res]
        else:
            exist_keys = []
        new_keys = set([s for s in keys if s not in exist_keys])

        return new_keys

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and id=%(id)s"
        params = {"workspace": self.workspace, "id": id}

        result = await self.db.query(sql, params, True)
        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=result[0]["created_at"],
                updated_at=result[0]["updated_at"],
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []

        ids_str = ",".join([f"'{id}'" for id in ids])
        sql = "SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=%(workspace)s AND id IN (%(ids)s)"
        params = {"workspace": self.workspace, "ids": ids_str}

        results = await self.db.query(sql, params, True)
        if not results:
            return []

        processed_map: dict[str, dict[str, Any]] = {}
        for row in results:
            # Parse chunks_list JSON string back to list
            chunks_list = row.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = row.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            processed_map[str(row.get("id"))] = {
                "content_length": row["content_length"],
                "content_summary": row["content_summary"],
                "status": row["status"],
                "chunks_count": row["chunks_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "file_path": row["file_path"],
                "chunks_list": chunks_list,
                "metadata": metadata,
                "error_msg": row.get("error_msg"),
                "track_id": row.get("track_id"),
            }

        ordered_results: list[dict[str, Any] | None] = []
        for requested_id in ids:
            ordered_results.append(processed_map.get(str(requested_id)))

        return ordered_results

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and file_path=%(file_path)s"
        params = {"workspace": self.workspace, "file_path": file_path}

        result = await self.db.query(sql, params, True)
        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            return dict(
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                chunks_count=result[0]["chunks_count"],
                created_at=result[0]["created_at"],
                updated_at=result[0]["updated_at"],
                file_path=result[0]["file_path"],
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=result[0].get("error_msg"),
                track_id=result[0].get("track_id"),
            )

    async def get_status_counts(self) -> dict[str, int]:
        sql = "SELECT status, count(1) as count FROM LIGHTRAG_DOC_STATUS where workspace=%(workspace)s GROUP BY status"
        params = {"workspace": self.workspace}

        result = await self.db.query(sql, params, True)

        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and status=%(status)s"
        params = {"workspace": self.workspace, "status": status.value}

        result = await self.db.query(sql, params, True)

        docs_by_status = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            docs_by_status[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=element.get("error_msg"),
                track_id=element.get("track_id"),
            )

        return docs_by_status

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=%(workspace)s and track_id=%(track_id)s"
        params = {"workspace": self.workspace, "track_id": track_id}

        result = await self.db.query(sql, params, True)

        docs_by_track_id = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get("file_path")
            if file_path is None:
                file_path = "no-file-path"

            docs_by_track_id[element["id"]] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=file_path,
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )

        return docs_by_track_id

    async def get_docs_paginated(
            self,
            status_filter: DocStatus | None = None,
            page: int = 1,
            page_size: int = 50,
            sort_field: str = "updated_at",
            sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        # Whitelist validation for sort_field to prevent SQL injection
        allowed_sort_fields = {"created_at", "updated_at", "id", "file_path"}
        if sort_field not in allowed_sort_fields:
            sort_field = "updated_at"

        # Whitelist validation for sort_direction to prevent SQL injection
        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"
        else:
            sort_direction = sort_direction.lower()

        # Calculate offset
        offset = (page - 1) * page_size

        # Build parameterized query components
        params = {"workspace": self.workspace}
        param_count = 1

        # Build WHERE clause with parameterized query
        if status_filter is not None:
            param_count += 1
            where_clause = "WHERE workspace=%(workspace)s AND status=%(status)s"
            params["status"] = status_filter.value
        else:
            where_clause = "WHERE workspace=%(workspace)s"

        # Build ORDER BY clause using validated whitelist values
        order_clause = f"ORDER BY {sort_field} {sort_direction.upper()}"

        # Query for total count
        count_sql = f"SELECT COUNT(*) as total FROM LIGHTRAG_DOC_STATUS {where_clause}"
        count_result = await self.db.query(count_sql, params)
        total_count = count_result["total"] if count_result else 0

        # Query for paginated data with parameterized LIMIT and OFFSET
        data_sql = f"""
                    SELECT * FROM LIGHTRAG_DOC_STATUS
                    {where_clause} 
                    {order_clause} 
                    LIMIT ${param_count + 1} OFFSET ${param_count + 2}
                    """
        params["limit"] = page_size
        params["offset"] = offset

        result = await self.db.query(data_sql, params, True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result:
            doc_id = element["id"]

            # Parse chunks_list JSON string back to list
            chunks_list = element.get("chunks_list", [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
                file_path=element["file_path"],
                chunks_list=chunks_list,
                track_id=element.get("track_id"),
                metadata=metadata,
                error_msg=element.get("error_msg"),
            )
            documents.append((doc_id, doc_status))

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        sql = "SELECT status, count(*) as count FROM LIGHTRAG_DOC_STATUS WHERE workspace=%(workspace)s GROUP BY status"
        params = {"workspace": self.workspace}

        result = await self.db.query(sql, params, True)

        counts = {}
        total_count = 0
        for row in result:
            counts[row["status"]] = row["count"]
            total_count += row["count"]

        # Add 'all' field with total count
        counts["all"] = total_count

        return counts

    async def index_done_callback(self) -> None:
        pass

    async def is_empty(self) -> bool:
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}")
            return True

        sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=%(workspace)s LIMIT 1) as has_data"
        try:
            result = await self.db.query(sql, {"workspace": self.workspace})

            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}")
            return

        ids_str = ",".join([f"'{id}'" for id in ids])
        delete_sql = f"DELETE FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)"

        try:
            await self.db.execute(delete_sql, {"workspace": self.workspace, "ids": ids_str})
        except Exception as e:
            logger.error(f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return

        # All fields are updated from the input data in both INSERT and UPDATE cases
        sql = """REPLACE INTO LIGHTRAG_DOC_STATUS(workspace, id, content_summary, content_length, chunks_count, 
               status, file_path, chunks_list, track_id, metadata, error_msg, updated_at)
               values(%(workspace)s, %(id)s, %(content_summary)s, %(content_length)s, %(chunks_count)s, 
               %(status)s, %(file_path)s, %(chunks_list)s, %(track_id)s, %(metadata)s, %(error_msg)s, CURRENT_TIMESTAMP)
              """
        for k, v in data.items():
            # chunks_count, chunks_list, track_id, metadata, and error_msg are optional
            data = {
                "workspace": self.workspace,
                "id": k,
                "content_summary": v["content_summary"],
                "content_length": v["content_length"],
                "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                "status": v["status"],
                "file_path": v["file_path"],
                "chunks_list": json.dumps(v.get("chunks_list", [])),
                "track_id": v.get("track_id"),  # Add track_id support
                "metadata": json.dumps(
                    v.get("metadata", {})
                ),  # Add metadata support
                "error_msg": v.get("error_msg"),  # Add error_msg support
            }
            await self.db.execute(sql, data)

    async def drop(self) -> dict[str, str]:
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specify_table_workspace"].format(table_name=table_name)
            await self.db.execute(drop_sql, {"workspace": self.workspace})

            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: "LIGHTRAG_DOC_FULL",
    NameSpace.KV_STORE_TEXT_CHUNKS: "LIGHTRAG_DOC_CHUNKS",
    NameSpace.KV_STORE_FULL_ENTITIES: "LIGHTRAG_FULL_ENTITIES",
    NameSpace.KV_STORE_FULL_RELATIONS: "LIGHTRAG_FULL_RELATIONS",
    NameSpace.KV_STORE_ENTITY_CHUNKS: "LIGHTRAG_ENTITY_CHUNKS",
    NameSpace.KV_STORE_RELATION_CHUNKS: "LIGHTRAG_RELATION_CHUNKS",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "LIGHTRAG_LLM_CACHE",
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
    NameSpace.DOC_STATUS: "LIGHTRAG_DOC_STATUS",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSON,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    llm_cache_list JSON NULL DEFAULT CAST('[]' as JSON),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSON NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length INTEGER NULL,
	               chunks_count INTEGER NULL,
	               status varchar(64) NULL,
	               file_path TEXT NULL,
	               chunks_list JSON NULL DEFAULT CAST('[]' as JSON),
	               track_id varchar(255) NULL,
	               metadata JSON NULL DEFAULT CAST('{}' as JSON),
	               error_msg TEXT NULL,
	               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	               PRIMARY KEY (workspace, id)
	              )"""
    },
    "LIGHTRAG_FULL_ENTITIES": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_FULL_RELATIONS": {
        "ddl": """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_ENTITY_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_ENTITY_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_RELATION_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_RELATION_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSON,
                    count INTEGER,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace, id)
                    )"""
    },
}

VECTOR_TABLES = {
    "LIGHTRAG_VDB_CHUNKS": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    file_path TEXT NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ANN INDEX idx_content_vector(content_vector),
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    ANN INDEX idx_content_vector(content_vector),
                    PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector ARRAY<FLOAT>(EMBEDDING_DIM),
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids ARRAY<VARCHAR(255)> NULL,
                    file_path TEXT NULL,
                    ANN INDEX idx_content_vector(content_vector),
                    PRIMARY KEY (workspace, id)
                    )"""
    },
}

SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content,
                             COALESCE(doc_name, '') as file_path
                             FROM LIGHTRAG_DOC_FULL WHERE workspace=%(workspace)s AND id=%(id)s
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path,
                                COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path,
                                  COALESCE(llm_cache_list, cast('[]' as json)) as llm_cache_list,
                                  UNIX_TIMESTAMP(create_time) as create_time,
                                  UNIX_TIMESTAMP(update_time) as update_time
                                  FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_id_full_entities": """SELECT id, entity_names, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_id_full_relations": """SELECT id, relation_pairs, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_full_entities": """SELECT id, entity_names, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_full_relations": """SELECT id, relation_pairs, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_id_entity_chunks": """SELECT id, chunk_ids, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_id_relation_chunks": """SELECT id, chunk_ids, count,
                                UNIX_TIMESTAMP(create_time) as create_time,
                                UNIX_TIMESTAMP(update_time) as update_time
                                FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=%(workspace)s AND id=%(id)s
                               """,
    "get_by_ids_entity_chunks": """SELECT id, chunk_ids, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "get_by_ids_relation_chunks": """SELECT id, chunk_ids, count,
                                 UNIX_TIMESTAMP(create_time) as create_time,
                                 UNIX_TIMESTAMP(update_time) as update_time
                                 FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=%(workspace)s AND id IN (%(ids)s)
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=%(workspace)s AND id IN (%(ids)s)",
    "upsert_doc_full": """REPLACE INTO LIGHTRAG_DOC_FULL (id, content, doc_name, workspace, update_time)
                        VALUES (%(id)s, %(content)s, %(doc_name)s, %(workspace)s, CURRENT_TIMESTAMP)
                       """,
    "upsert_llm_response_cache": """REPLACE INTO LIGHTRAG_LLM_CACHE(workspace, id, original_prompt, return_value, 
                                  chunk_id, cache_type, queryparam, update_time)
                                  VALUES (%(workspace)s, %(id)s, %(original_prompt)s, %(return_value)s, 
                                  %(chunk_id)s, %(cache_type)s, %(queryparam)s, CURRENT_TIMESTAMP)
                                 """,
    "upsert_text_chunk": """REPLACE INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, llm_cache_list, update_time)
                      VALUES (%(workspace)s, %(id)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s, 
                      %(content)s, %(file_path)s, %(llm_cache_list)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_entities": """REPLACE INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count, update_time)
                      VALUES (%(workspace)s, %(id)s, %(entity_names)s, %(count)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_full_relations": """REPLACE INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count, update_time)
                      VALUES (%(workspace)s, %(id)s, %(relation_pairs)s, %(count)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_entity_chunks": """REPLACE INTO LIGHTRAG_ENTITY_CHUNKS (workspace, id, chunk_ids, count, update_time)
                      VALUES (%(workspace)s, %(id)s, %(chunk_ids)s, %(count)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_relation_chunks": """REPLACE INTO LIGHTRAG_RELATION_CHUNKS (workspace, id, chunk_ids, count, update_time)
                      VALUES (%(workspace)s, %(id)s, %(chunk_ids)s, %(count)s, CURRENT_TIMESTAMP)
                     """,
    # SQL for VectorStorage
    "upsert_chunk": """REPLACE INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path, update_time)
                      VALUES (%(workspace)s, %(id)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s, 
                      %(content)s, %(content_vector)s, %(file_path)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_entity": """REPLACE INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, update_time)
                      VALUES (%(workspace)s, %(id)s, %(entity_name)s, %(content)s, 
                      %(content_vector)s, %(chunk_ids)s, %(file_path)s, CURRENT_TIMESTAMP)
                     """,
    "upsert_relationship": """REPLACE INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, update_time)
                      VALUES (%(workspace)s, %(id)s, %(source_id)s, %(target_id)s, 
                      %(content)s, %(content_vector)s, %(chunk_ids)s, %(file_path)s, CURRENT_TIMESTAMP)
                     """,
    "relationships": """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            UNIX_TIMESTAMP(r.create_time) AS created_at,
                            l2_distance(r.content_vector, '[{embedding_string}]') AS distance
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = %(workspace)s
                       AND l2_distance(r.content_vector, '[{embedding_string}]') < %(closer_than_threshold)s
                     ORDER BY distance
                     LIMIT %(top_k)s;
                     """,
    "entities": """
                SELECT e.entity_name,
                       UNIX_TIMESTAMP(e.create_time) AS created_at,
                       l2_distance(e.content_vector, '[{embedding_string}]') AS distance
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = %(workspace)s
                  AND l2_distance(e.content_vector, '[{embedding_string}]') < %(closer_than_threshold)s
                ORDER BY distance
                LIMIT %(top_k)s;
                """,
    "chunks": """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     UNIX_TIMESTAMP(c.create_time) AS created_at,
                     l2_distance(c.content_vector, '[{embedding_string}]') AS distance
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = %(workspace)s
                AND l2_distance(c.content_vector, '[{embedding_string}]') < %(closer_than_threshold)s
              ORDER BY distance
              LIMIT %(top_k)s;
              """,
    # DROP tables
    "drop_specify_table_workspace": """
        DELETE FROM {table_name} WHERE workspace=%(workspace)s
       """,
}
