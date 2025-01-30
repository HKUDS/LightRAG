import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Union, List, Dict, Set, Any, Tuple
import numpy as np

import pipmaster as pm

if not pm.is_installed("asyncpg"):
    pm.install("asyncpg")

import asyncpg
import sys
from tqdm.asyncio import tqdm as tqdm_async
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..utils import logger
from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
    DocStatus,
    DocProcessingStatus,
    BaseGraphStorage,
)

if sys.platform.startswith("win"):
    import asyncio.windows_events

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class PostgreSQLDB:
    def __init__(self, config, **kwargs):
        self.pool = None
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.user = config.get("user", "postgres")
        self.password = config.get("password", None)
        self.database = config.get("database", "postgres")
        self.workspace = config.get("workspace", "default")
        self.max = 12
        self.increment = 1
        logger.info(f"Using the label {self.workspace} for PostgreSQL as identifier")

        if self.user is None or self.password is None or self.database is None:
            raise ValueError(
                "Missing database user, password, or database in addon_params"
            )

    async def initdb(self):
        try:
            self.pool = await asyncpg.create_pool(
                user=self.user,
                password=self.password,
                database=self.database,
                host=self.host,
                port=self.port,
                min_size=1,
                max_size=self.max,
            )

            logger.info(
                f"Connected to PostgreSQL database at {self.host}:{self.port}/{self.database}"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to PostgreSQL database at {self.host}:{self.port}/{self.database}"
            )
            logger.error(f"PostgreSQL database error: {e}")
            raise

    async def check_tables(self):
        for k, v in TABLES.items():
            try:
                await self.query("SELECT 1 FROM {k} LIMIT 1".format(k=k))
            except Exception as e:
                logger.error(f"Failed to check table {k} in PostgreSQL database")
                logger.error(f"PostgreSQL database error: {e}")
                try:
                    await self.execute(v["ddl"])
                    logger.info(f"Created table {k} in PostgreSQL database")
                except Exception as e:
                    logger.error(f"Failed to create table {k} in PostgreSQL database")
                    logger.error(f"PostgreSQL database error: {e}")

        logger.info("Finished checking all tables in PostgreSQL database")

    async def query(
        self,
        sql: str,
        params: dict = None,
        multirows: bool = False,
        for_age: bool = False,
        graph_name: str = None,
    ) -> Union[dict, None, list[dict]]:
        async with self.pool.acquire() as connection:
            try:
                if for_age:
                    await PostgreSQLDB._prerequisite(connection, graph_name)
                if params:
                    rows = await connection.fetch(sql, *params.values())
                else:
                    rows = await connection.fetch(sql)

                if multirows:
                    if rows:
                        columns = [col for col in rows[0].keys()]
                        data = [dict(zip(columns, row)) for row in rows]
                    else:
                        data = []
                else:
                    if rows:
                        columns = rows[0].keys()
                        data = dict(zip(columns, rows[0]))
                    else:
                        data = None
                return data
            except Exception as e:
                logger.error(f"PostgreSQL database error: {e}")
                print(sql)
                print(params)
                raise

    async def execute(
        self,
        sql: str,
        data: Union[list, dict] = None,
        for_age: bool = False,
        graph_name: str = None,
        upsert: bool = False,
    ):
        try:
            async with self.pool.acquire() as connection:
                if for_age:
                    await PostgreSQLDB._prerequisite(connection, graph_name)

                if data is None:
                    await connection.execute(sql)
                else:
                    await connection.execute(sql, *data.values())
        except (
            asyncpg.exceptions.UniqueViolationError,
            asyncpg.exceptions.DuplicateTableError,
        ) as e:
            if upsert:
                print("Key value duplicate, but upsert succeeded.")
            else:
                logger.error(f"Upsert error: {e}")
        except Exception as e:
            logger.error(f"PostgreSQL database error: {e.__class__} - {e}")
            print(sql)
            print(data)
            raise

    @staticmethod
    async def _prerequisite(conn: asyncpg.Connection, graph_name: str):
        try:
            await conn.execute('SET search_path = ag_catalog, "$user", public')
            await conn.execute(f"""select create_graph('{graph_name}')""")
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass


@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = None

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]

    ################ QUERY METHODS ################

    async def get_by_id(self, id: str) -> Union[dict, None]:
        """Get doc_full data by id."""
        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        params = {"workspace": self.db.workspace, "id": id}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
        else:
            res = await self.db.query(sql, params)
        if res:
            return res
        else:
            return None

    async def get_by_mode_and_id(self, mode: str, id: str) -> Union[dict, None]:
        """Specifically for llm_response_cache."""
        sql = SQL_TEMPLATES["get_by_mode_id_" + self.namespace]
        params = {"workspace": self.db.workspace, mode: mode, "id": id}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            res = {}
            for row in array_res:
                res[row["id"]] = row
            return res
        else:
            return None

    # Query by id
    async def get_by_ids(self, ids: List[str], fields=None) -> Union[List[dict], None]:
        """Get doc_chunks data by id"""
        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace].format(
            ids=",".join([f"'{id}'" for id in ids])
        )
        params = {"workspace": self.db.workspace}
        if "llm_response_cache" == self.namespace:
            array_res = await self.db.query(sql, params, multirows=True)
            modes = set()
            dict_res: dict[str, dict] = {}
            for row in array_res:
                modes.add(row["mode"])
            for mode in modes:
                if mode not in dict_res:
                    dict_res[mode] = {}
            for row in array_res:
                dict_res[row["mode"]][row["id"]] = row
            res = [{k: v} for k, v in dict_res.items()]
        else:
            res = await self.db.query(sql, params, multirows=True)
        if res:
            return res
        else:
            return None

    async def all_keys(self) -> list[dict]:
        if "llm_response_cache" == self.namespace:
            sql = "select workspace,mode,id from lightrag_llm_cache"
            res = await self.db.query(sql, multirows=True)
            return res
        else:
            logger.error(
                f"all_keys is only implemented for llm_response_cache, not for {self.namespace}"
            )

    async def filter_keys(self, keys: List[str]) -> Set[str]:
        """Filter out duplicated content"""
        sql = SQL_TEMPLATES["filter_keys"].format(
            table_name=NAMESPACE_TABLE_MAP[self.namespace],
            ids=",".join([f"'{id}'" for id in keys]),
        )
        params = {"workspace": self.db.workspace}
        try:
            res = await self.db.query(sql, params, multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            data = set([s for s in keys if s not in exist_keys])
            return data
        except Exception as e:
            logger.error(f"PostgreSQL database error: {e}")
            print(sql)
            print(params)

    ################ INSERT METHODS ################
    async def upsert(self, data: Dict[str, dict]):
        if self.namespace == "text_chunks":
            pass
        elif self.namespace == "full_docs":
            for k, v in data.items():
                upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
                _data = {
                    "id": k,
                    "content": v["content"],
                    "workspace": self.db.workspace,
                }
                await self.db.execute(upsert_sql, _data)
        elif self.namespace == "llm_response_cache":
            for mode, items in data.items():
                for k, v in items.items():
                    upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
                    _data = {
                        "workspace": self.db.workspace,
                        "id": k,
                        "original_prompt": v["original_prompt"],
                        "return_value": v["return"],
                        "mode": mode,
                    }

                    await self.db.execute(upsert_sql, _data)

    async def index_done_callback(self):
        if self.namespace in ["full_docs", "text_chunks"]:
            logger.info("full doc and chunk data had been saved into postgresql db!")


@dataclass
class PGVectorStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = float(os.getenv("COSINE_THRESHOLD", "0.2"))
    db: PostgreSQLDB = None

    def __post_init__(self):
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # Use global config value if specified, otherwise use default
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def _upsert_chunks(self, item: dict):
        try:
            upsert_sql = SQL_TEMPLATES["upsert_chunk"]
            data = {
                "workspace": self.db.workspace,
                "id": item["__id__"],
                "tokens": item["tokens"],
                "chunk_order_index": item["chunk_order_index"],
                "full_doc_id": item["full_doc_id"],
                "content": item["content"],
                "content_vector": json.dumps(item["__vector__"].tolist()),
            }
        except Exception as e:
            logger.error(f"Error to prepare upsert sql: {e}")
            print(item)
            raise e
        return upsert_sql, data

    def _upsert_entities(self, item: dict):
        upsert_sql = SQL_TEMPLATES["upsert_entity"]
        data = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "entity_name": item["entity_name"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
        }
        return upsert_sql, data

    def _upsert_relationships(self, item: dict):
        upsert_sql = SQL_TEMPLATES["upsert_relationship"]
        data = {
            "workspace": self.db.workspace,
            "id": item["__id__"],
            "source_id": item["src_id"],
            "target_id": item["tgt_id"],
            "content": item["content"],
            "content_vector": json.dumps(item["__vector__"].tolist()),
        }
        return upsert_sql, data

    async def upsert(self, data: Dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        for item in list_data:
            if self.namespace == "chunks":
                upsert_sql, data = self._upsert_chunks(item)
            elif self.namespace == "entities":
                upsert_sql, data = self._upsert_entities(item)
            elif self.namespace == "relationships":
                upsert_sql, data = self._upsert_relationships(item)
            else:
                raise ValueError(f"{self.namespace} is not supported")

            await self.db.execute(upsert_sql, data)

    async def index_done_callback(self):
        logger.info("vector data had been saved into postgresql db!")

    #################### query method ###############
    async def query(self, query: str, top_k=5) -> Union[dict, list[dict]]:
        """从向量数据库中查询数据"""
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]
        embedding_string = ",".join(map(str, embedding))

        sql = SQL_TEMPLATES[self.namespace].format(embedding_string=embedding_string)
        params = {
            "workspace": self.db.workspace,
            "better_than_threshold": self.cosine_better_than_threshold,
            "top_k": top_k,
        }
        results = await self.db.query(sql, params=params, multirows=True)
        return results


@dataclass
class PGDocStatusStorage(DocStatusStorage):
    """PostgreSQL implementation of document status storage"""

    db: PostgreSQLDB = None

    def __post_init__(self):
        pass

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return keys that don't exist in storage"""
        keys = ",".join([f"'{_id}'" for _id in data])
        sql = (
            f"SELECT id FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND id IN ({keys})"
        )
        result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
        # The result is like [{'id': 'id1'}, {'id': 'id2'}, ...].
        if result is None:
            return set(data)
        else:
            existed = set([element["id"] for element in result])
            return set(data) - existed

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of documents in each status"""
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  where workspace=$1 GROUP BY STATUS
                 """
        result = await self.db.query(sql, {"workspace": self.db.workspace}, True)
        # Result is like [{'status': 'PENDING', 'count': 1}, {'status': 'PROCESSING', 'count': 2}, ...]
        counts = {}
        for doc in result:
            counts[doc["status"]] = doc["count"]
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> Dict[str, DocProcessingStatus]:
        """Get all documents by status"""
        sql = "select * from LIGHTRAG_DOC_STATUS where workspace=$1 and status=$1"
        params = {"workspace": self.db.workspace, "status": status}
        result = await self.db.query(sql, params, True)
        # Result is like [{'id': 'id1', 'status': 'PENDING', 'updated_at': '2023-07-01 00:00:00'}, {'id': 'id2', 'status': 'PENDING', 'updated_at': '2023-07-01 00:00:00'}, ...]
        # Converting to be a dict
        return {
            element["id"]: DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=element["created_at"],
                updated_at=element["updated_at"],
                chunks_count=element["chunks_count"],
            )
            for element in result
        }

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        return await self.get_docs_by_status(DocStatus.FAILED)

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        return await self.get_docs_by_status(DocStatus.PENDING)

    async def index_done_callback(self):
        """Save data after indexing, but for PostgreSQL, we already saved them during the upsert stage, so no action to take here"""
        logger.info("Doc status had been saved into postgresql db!")

    async def upsert(self, data: dict[str, dict]):
        """Update or insert document status

        Args:
            data: Dictionary of document IDs and their status data
        """
        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status)
                 values($1,$2,$3,$4,$5,$6)
                  on conflict(id,workspace) do update set
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  updated_at = CURRENT_TIMESTAMP"""
        for k, v in data.items():
            # chunks_count is optional
            await self.db.execute(
                sql,
                {
                    "workspace": self.db.workspace,
                    "id": k,
                    "content_summary": v["content_summary"],
                    "content_length": v["content_length"],
                    "chunks_count": v["chunks_count"] if "chunks_count" in v else -1,
                    "status": v["status"],
                },
            )
        return data


class PGGraphQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, Dict]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@dataclass
class PGGraphStorage(BaseGraphStorage):
    db: PostgreSQLDB = None

    @staticmethod
    def load_nx_graph(file_name):
        print("no preloading of graph with AGE in production")

    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self.graph_name = os.environ["AGE_GRAPH_NAME"]
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def index_done_callback(self):
        print("KG successfully indexed.")

    @staticmethod
    def _record_to_dict(record: asyncpg.Record) -> Dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            Dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """
        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}
        for k in record.keys():
            v = record[k]
            # agtype comes back '{key: value}::type' which must be parsed
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
                if dtype == "vertex":
                    vertex = json.loads(v)
                    vertices[vertex["id"]] = vertex.get("properties")

        # iterate returned fields and parse appropriately
        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                dtype = v.split("::")[-1]
                v = v.split("::")[0]
            else:
                dtype = ""

            if dtype == "vertex":
                vertex = json.loads(v)
                field = vertex.get("properties")
                if not field:
                    field = {}
                field["label"] = PGGraphStorage._decode_graph_label(field["node_id"])
                d[k] = field
            # convert edge from id-label->id by replacing id with node information
            # we only do this if the vertex was also returned in the query
            # this is an attempt to be consistent with neo4j implementation
            elif dtype == "edge":
                edge = json.loads(v)
                d[k] = (
                    vertices.get(edge["start_id"], {}),
                    edge[
                        "label"
                    ],  # we don't use decode_graph_label(), since edge label is always "DIRECTED"
                    vertices.get(edge["end_id"], {}),
                )
            else:
                d[k] = json.loads(v) if isinstance(v, str) else v

        return d

    @staticmethod
    def _format_properties(
        properties: Dict[str, Any], _id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            _id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def _encode_graph_label(label: str) -> str:
        """
        Since AGE supports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            label (str): the original label

        Returns:
            str: the encoded label
        """
        return "x" + label.encode().hex()

    @staticmethod
    def _decode_graph_label(encoded_label: str) -> str:
        """
        Since AGE supports only alphanumerical labels, we will encode generic label as HEX string

        Args:
            encoded_label (str): the encoded label

        Returns:
            str: the decoded label
        """
        return bytes.fromhex(encoded_label.removeprefix("x")).decode()

    @staticmethod
    def _get_col_name(field: str, idx: int) -> str:
        """
        Convert a cypher return field to a pgsql select field
        If possible keep the cypher column name, but create a generic name if necessary

        Args:
            field (str): a return field from a cypher query to be formatted for pgsql
            idx (int): the position of the field in the return statement

        Returns:
            str: the field to be used in the pgsql select statement
        """
        # remove white space
        field = field.strip()
        # if an alias is provided for the field, use it
        if " as " in field:
            return field.split(" as ")[-1].strip()
        # if the return value is an unnamed primitive, give it a generic name
        if field.isnumeric() or field in ("true", "false", "null"):
            return f"column_{idx}"
        # otherwise return the value stripping out some common special chars
        return field.replace("(", "_").replace(")", "")

    async def _query(
        self, query: str, readonly: bool = True, upsert: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed
            params (dict): parameters for the query

        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        # convert cypher query to pgsql/age query
        wrapped_query = query

        # execute the query, rolling back on an error
        try:
            if readonly:
                data = await self.db.query(
                    wrapped_query,
                    multirows=True,
                    for_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await self.db.execute(
                    wrapped_query,
                    for_age=True,
                    graph_name=self.graph_name,
                    upsert=upsert,
                )
        except Exception as e:
            raise PGGraphQueryException(
                {
                    "message": f"Error executing graph query: {query}",
                    "wrapped": wrapped_query,
                    "detail": str(e),
                }
            ) from e

        if data is None:
            result = []
        # decode records
        else:
            result = [PGGraphStorage._record_to_dict(d) for d in data]

        return result

    async def has_node(self, node_id: str) -> bool:
        entity_name_label = PGGraphStorage._encode_graph_label(node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})
                     RETURN count(n) > 0 AS node_exists
                   $$) AS (node_exists bool)""" % (self.graph_name, entity_name_label)

        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["node_exists"],
        )

        return single_result["node_exists"]

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (a:Entity {node_id: "%s"})-[r]-(b:Entity {node_id: "%s"})
                     RETURN COUNT(r) > 0 AS edge_exists
                   $$) AS (edge_exists bool)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )

        single_result = (await self._query(query))[0]
        logger.debug(
            "{%s}:query:{%s}:result:{%s}",
            inspect.currentframe().f_code.co_name,
            query,
            single_result["edge_exists"],
        )
        return single_result["edge_exists"]

    async def get_node(self, node_id: str) -> Union[dict, None]:
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})
                     RETURN n
                   $$) AS (n agtype)""" % (self.graph_name, label)
        record = await self._query(query)
        if record:
            node = record[0]
            node_dict = node["n"]
            logger.debug(
                "{%s}: query: {%s}, result: {%s}",
                inspect.currentframe().f_code.co_name,
                query,
                node_dict,
            )
            return node_dict
        return None

    async def node_degree(self, node_id: str) -> int:
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:Entity {node_id: "%s"})-[]->(x)
                     RETURN count(x) AS total_edge_count
                   $$) AS (total_edge_count integer)""" % (self.graph_name, label)
        record = (await self._query(query))[0]
        if record:
            edge_count = int(record["total_edge_count"])
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                edge_count,
            )
            return edge_count

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        logger.debug(
            "{%s}:query:src_Degree+trg_degree:result:{%s}",
            inspect.currentframe().f_code.co_name,
            degrees,
        )
        return degrees

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """
        Find all edges between nodes of two given labels

        Args:
            source_node_id (str): Label of the source nodes
            target_node_id (str): Label of the target nodes

        Returns:
            list: List of all relationships/edges found
        """
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (a:Entity {node_id: "%s"})-[r]->(b:Entity {node_id: "%s"})
                     RETURN properties(r) as edge_properties
                     LIMIT 1
                   $$) AS (edge_properties agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
        )
        record = await self._query(query)
        if record and record[0] and record[0]["edge_properties"]:
            result = record[0]["edge_properties"]
            logger.debug(
                "{%s}:query:{%s}:result:{%s}",
                inspect.currentframe().f_code.co_name,
                query,
                result,
            )
            return result

    async def get_node_edges(self, source_node_id: str) -> List[Tuple[str, str]]:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: List of dictionaries containing edge information
        """
        label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))

        query = """SELECT * FROM cypher('%s', $$
                      MATCH (n:Entity {node_id: "%s"})
                      OPTIONAL MATCH (n)-[r]-(connected)
                      RETURN n, r, connected
                    $$) AS (n agtype, r agtype, connected agtype)""" % (
            self.graph_name,
            label,
        )

        results = await self._query(query)
        edges = []
        for record in results:
            source_node = record["n"] if record["n"] else None
            connected_node = record["connected"] if record["connected"] else None

            source_label = (
                source_node["node_id"]
                if source_node and source_node["node_id"]
                else None
            )
            target_label = (
                connected_node["node_id"]
                if connected_node and connected_node["node_id"]
                else None
            )

            if source_label and target_label:
                edges.append(
                    (
                        PGGraphStorage._decode_graph_label(source_label),
                        PGGraphStorage._decode_graph_label(target_label),
                    )
                )

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]):
        """
        Upsert a node in the AGE database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        label = PGGraphStorage._encode_graph_label(node_id.strip('"'))
        properties = node_data

        query = """SELECT * FROM cypher('%s', $$
                     MERGE (n:Entity {node_id: "%s"})
                     SET n += %s
                     RETURN n
                   $$) AS (n agtype)""" % (
            self.graph_name,
            label,
            PGGraphStorage._format_properties(properties),
        )

        try:
            await self._query(query, readonly=False, upsert=True)
            logger.debug(
                "Upserted node with label '{%s}' and properties: {%s}",
                label,
                properties,
            )
        except Exception as e:
            logger.error("Error during upsert: {%s}", e)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        src_label = PGGraphStorage._encode_graph_label(source_node_id.strip('"'))
        tgt_label = PGGraphStorage._encode_graph_label(target_node_id.strip('"'))
        edge_properties = edge_data

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (source:Entity {node_id: "%s"})
                     WITH source
                     MATCH (target:Entity {node_id: "%s"})
                     MERGE (source)-[r:DIRECTED]->(target)
                     SET r += %s
                     RETURN r
                   $$) AS (r agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
            PGGraphStorage._format_properties(edge_properties),
        )
        # logger.info(f"-- inserting edge after formatted: {params}")
        try:
            await self._query(query, readonly=False, upsert=True)
            logger.debug(
                "Upserted edge from '{%s}' to '{%s}' with properties: {%s}",
                src_label,
                tgt_label,
                edge_properties,
            )
        except Exception as e:
            logger.error("Error during edge upsert: {%s}", e)
            raise

    async def _node2vec_embed(self):
        print("Implemented but never called.")


NAMESPACE_TABLE_MAP = {
    "full_docs": "LIGHTRAG_DOC_FULL",
    "text_chunks": "LIGHTRAG_DOC_CHUNKS",
    "chunks": "LIGHTRAG_DOC_CHUNKS",
    "entities": "LIGHTRAG_VDB_ENTITY",
    "relationships": "LIGHTRAG_VDB_RELATION",
    "doc_status": "LIGHTRAG_DOC_STATUS",
    "llm_response_cache": "LIGHTRAG_LLM_CACHE",
}


TABLES = {
    "LIGHTRAG_DOC_FULL": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSONB,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
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
                    content_vector VECTOR,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_ENTITY": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(255),
                    content TEXT,
                    content_vector VECTOR,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_VDB_RELATION": {
        "ddl": """CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(256),
                    target_id VARCHAR(256),
                    content TEXT,
                    content_vector VECTOR,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    "LIGHTRAG_LLM_CACHE": {
        "ddl": """CREATE TABLE LIGHTRAG_LLM_CACHE (
	                workspace varchar(255) NOT NULL,
	                id varchar(255) NOT NULL,
	                mode varchar(32) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP,
	                CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, mode, id)
                    )"""
    },
    "LIGHTRAG_DOC_STATUS": {
        "ddl": """CREATE TABLE LIGHTRAG_DOC_STATUS (
	               workspace varchar(255) NOT NULL,
	               id varchar(255) NOT NULL,
	               content_summary varchar(255) NULL,
	               content_length int4 NULL,
	               chunks_count int4 NULL,
	               status varchar(64) NULL,
	               created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               updated_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	               CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
	              )"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT id, COALESCE(content, '') as content
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode=$2
                               """,
    "get_by_mode_id_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                           FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode=$2 AND id=$3
                          """,
    "get_by_ids_full_docs": """SELECT id, COALESCE(content, '') as content
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id IN ({ids})
                            """,
    "get_by_ids_text_chunks": """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id
                                   FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id IN ({ids})
                                """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, COALESCE(return_value, '') as "return", mode
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND mode= IN ({ids})
                                """,
    "filter_keys": "SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})",
    "upsert_doc_full": """INSERT INTO LIGHTRAG_DOC_FULL (id, content, workspace)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = $2, update_time = CURRENT_TIMESTAMP
                       """,
    "upsert_llm_response_cache": """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,mode)
                                      VALUES ($1, $2, $3, $4, $5)
                                      ON CONFLICT (workspace,mode,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      mode=EXCLUDED.mode,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    "upsert_chunk": """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector)
                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      update_time = CURRENT_TIMESTAMP
                     """,
    "upsert_entity": """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content, content_vector)
                      VALUES ($1, $2, $3, $4, $5)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      update_time=CURRENT_TIMESTAMP
                     """,
    "upsert_relationship": """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector, update_time = CURRENT_TIMESTAMP
                     """,
    # SQL for VectorStorage
    "entities": """SELECT entity_name FROM
        (SELECT id, entity_name, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
        FROM LIGHTRAG_VDB_ENTITY where workspace=$1)
        WHERE distance>$2 ORDER BY distance DESC  LIMIT $3
       """,
    "relationships": """SELECT source_id as src_id, target_id as tgt_id FROM
        (SELECT id, source_id,target_id, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
        FROM LIGHTRAG_VDB_RELATION where workspace=$1)
        WHERE distance>$2 ORDER BY distance DESC  LIMIT $3
       """,
    "chunks": """SELECT id FROM
        (SELECT id, 1 - (content_vector <=> '[{embedding_string}]'::vector) as distance
        FROM LIGHTRAG_DOC_CHUNKS where workspace=$1)
        WHERE distance>$2 ORDER BY distance DESC  LIMIT $3
       """,
}
