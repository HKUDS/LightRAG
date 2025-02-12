import asyncio
import json
import os
import time
import sys
import numpy as np
import aiomysql
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from ..base import BaseVectorStorage
from ..utils import logger, EmbeddingFunc

class SingleStoreDB:
    def __init__(self, config, **kwargs):
        self.pool = None
        self.host = config.get("host", "svc-452cc4b1-df20-4130-9e2f-e72ba79e3d46-shared-dml.aws-virginia-hd2.svc.singlestore.com")
        self.port = config.get("port", 3333)
        self.user = config.get("user", "thilak-d4748")
        self.password = config.get("password", os.environ["S2DB_PASSWORD"])
        self.database = config.get("database", "bigbrain")
        self.workspace = config.get("workspace", "bigbrain-ws")
        self.max = 12
        self.increment = 1

        if self.user is None or self.password is None or self.database is None:
            raise ValueError(
                "Missing database user, password, or database in addon_params"
            )

    async def initdb(self):
        self.pool = await aiomysql.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.database,
            minsize=1,
            maxsize=self.max_size,
            autocommit=False
        )

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def check_tables(self):
        for k, v in TABLES.items():
            try:
                await self.query(f"SELECT 1 FROM {k} LIMIT 1")
            except Exception:
                try:
                    await self.execute(v["ddl"])
                except Exception as e:
                    logger.error(e)

    async def query(self, sql: str, params: dict = None, multirows: bool = False) -> Union[None, dict, List[dict]]:
        conn = await self.pool.acquire()
        try:
            cur = await conn.cursor()
            if params:
                await cur.execute(sql, tuple(params.values()))
            else:
                await cur.execute(sql)
            rows = await cur.fetchall()
            if not rows:
                return [] if multirows else None
            cols = [desc[0] for desc in cur.description]
            if multirows:
                return [dict(zip(cols, row)) for row in rows]
            return dict(zip(cols, rows[0]))
        finally:
            await cur.close()
            await self.pool.release(conn)

    async def execute(self, sql: str, data: Union[dict, None] = None):
        conn = await self.pool.acquire()
        try:
            cur = await conn.cursor()
            if data:
                await cur.execute(sql, tuple(data.values()))
            else:
                await cur.execute(sql)
            await conn.commit()
        finally:
            await cur.close()
            await self.pool.release(conn)

@dataclass
class SingleStoreVectorDBStorage(BaseVectorStorage):
    db: SingleStoreDB = None
    cosine_better_than_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.2"))
    vector_dimension: int = field(default=1536)

    def __post_init__(self):
        c = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = c.get("cosine_better_than_threshold", self.cosine_better_than_threshold)
        self.vector_dimension = c.get("vector_dimension", self.vector_dimension)
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)

    async def upsert(self, data: Dict[str, dict]):
        if not data:
            return
        items = list(data.items())
        need_embedding = []
        for _, v in items:
            if "__vector__" not in v and self.embedding_func:
                need_embedding.append(v["content"])
        if need_embedding and self.embedding_func:
            embedded = []
            for i in range(0, len(need_embedding), self._max_batch_size):
                batch = need_embedding[i : i + self._max_batch_size]
                emb = await self.embedding_func(batch)
                embedded.append(emb)
            merged = np.concatenate(embedded)
            idx = 0
            for _, v in items:
                if "__vector__" not in v:
                    v["__vector__"] = merged[idx]
                    idx += 1
        sql = f"""
        INSERT INTO LIGHTRAG_DOC_CHUNKS
        (id, workspace, tokens, chunk_order_index, full_doc_id, content, content_vector)
        VALUES
        (%(id)s, %(workspace)s, %(tokens)s, %(chunk_order_index)s, %(full_doc_id)s, %(content)s,
         CAST(%(vector_json)s AS VECTOR({self.vector_dimension}))
        )
        ON DUPLICATE KEY UPDATE
          tokens=VALUES(tokens),
          chunk_order_index=VALUES(chunk_order_index),
          full_doc_id=VALUES(full_doc_id),
          content=VALUES(content),
          content_vector=VALUES(content_vector)
        """
        for k, v in items:
            if isinstance(v["__vector__"], np.ndarray):
                vec_str = json.dumps(v["__vector__"].tolist())
            else:
                vec_str = json.dumps(v["__vector__"])
            p = {
                "id": k,
                "workspace": self.db.workspace,
                "tokens": v.get("tokens", 0),
                "chunk_order_index": v.get("chunk_order_index", 0),
                "full_doc_id": v.get("full_doc_id", ""),
                "content": v.get("content", ""),
                "vector_json": vec_str,
            }
            await self.db.execute(sql, p)

    async def query(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        vec = await self.embedding_func([query])
        emb = json.dumps(vec[0].tolist())
        sql = f"""
        SELECT id, content,
        (content_vector <*> CAST(%(qv)s AS VECTOR({self.vector_dimension}))) AS distance
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE workspace=%(workspace)s
        AND (content_vector <*> CAST(%(qv)s AS VECTOR({self.vector_dimension}))) > %(threshold)s
        ORDER BY distance DESC
        LIMIT %(top_k)s
        """
        params = {
            "qv": emb,
            "workspace": self.db.workspace,
            "threshold": self.cosine_better_than_threshold,
            "top_k": top_k,
        }
        rows = await self.db.query(sql, params, multirows=True)
        if not rows:
            return []
        return rows

TABLES = {
    "LIGHTRAG_DOC_CHUNKS": {
        "ddl": """
        CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_CHUNKS (
            id VARCHAR(255),
            workspace VARCHAR(255),
            tokens INT,
            chunk_order_index INT,
            full_doc_id VARCHAR(256),
            content TEXT,
            content_vector VECTOR(1536),
            PRIMARY KEY (workspace, id)
        )
        """
    }
}
