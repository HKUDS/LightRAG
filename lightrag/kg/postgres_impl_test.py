import asyncio
import asyncpg
import sys
import os

import psycopg
from psycopg_pool import AsyncConnectionPool
from lightrag.kg.postgres_impl import PostgreSQLDB, PGGraphStorage

DB = "rag"
USER = "rag"
PASSWORD = "rag"
HOST = "localhost"
PORT = "15432"
os.environ["AGE_GRAPH_NAME"] = "dickens"

if sys.platform.startswith("win"):
    import asyncio.windows_events

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def get_pool():
    return await asyncpg.create_pool(
        f"postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
        min_size=10,
        max_size=10,
        max_queries=5000,
        max_inactive_connection_lifetime=300.0,
    )


async def main1():
    connection_string = (
        f"dbname='{DB}' user='{USER}' password='{PASSWORD}' host='{HOST}' port={PORT}"
    )
    pool = AsyncConnectionPool(connection_string, open=False)
    await pool.open()

    try:
        conn = await pool.getconn(timeout=10)
        async with conn.cursor() as curs:
            try:
                await curs.execute('SET search_path = ag_catalog, "$user", public')
                await curs.execute("SELECT create_graph('dickens-2')")
                await conn.commit()
                print("create_graph success")
            except (
                psycopg.errors.InvalidSchemaName,
                psycopg.errors.UniqueViolation,
            ):
                print("create_graph already exists")
                await conn.rollback()
    finally:
        pass


db = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "r1",
    }
)


async def query_with_age():
    await db.initdb()
    graph = PGGraphStorage(
        namespace="chunk_entity_relation",
        global_config={},
        embedding_func=None,
    )
    graph.db = db
    res = await graph.get_node('"A CHRISTMAS CAROL"')
    print("Node is: ", res)
    res = await graph.get_edge('"A CHRISTMAS CAROL"', "PROJECT GUTENBERG")
    print("Edge is: ", res)
    res = await graph.get_node_edges('"SCROOGE"')
    print("Node Edges are: ", res)


async def create_edge_with_age():
    await db.initdb()
    graph = PGGraphStorage(
        namespace="chunk_entity_relation",
        global_config={},
        embedding_func=None,
    )
    graph.db = db
    await graph.upsert_node('"THE CRATCHITS"', {"hello": "world"})
    await graph.upsert_node('"THE GIRLS"', {"world": "hello"})
    await graph.upsert_edge(
        '"THE CRATCHITS"',
        '"THE GIRLS"',
        edge_data={
            "weight": 7.0,
            "description": '"The girls are part of the Cratchit family, contributing to their collective efforts and shared experiences.',
            "keywords": '"family, collective effort"',
            "source_id": "chunk-1d4b58de5429cd1261370c231c8673e8",
        },
    )
    res = await graph.get_edge("THE CRATCHITS", '"THE GIRLS"')
    print("Edge is: ", res)


async def main():
    pool = await get_pool()
    sql = r"SELECT * FROM ag_catalog.cypher('dickens', $$ MATCH (n:帅哥) RETURN n $$) AS (n ag_catalog.agtype)"
    # cypher = "MATCH (n:how_are_you_doing) RETURN n"
    async with pool.acquire() as conn:
        try:
            await conn.execute(
                """SET search_path = ag_catalog, "$user", public;select create_graph('dickens')"""
            )
        except asyncpg.exceptions.InvalidSchemaNameError:
            print("create_graph already exists")
        # stmt = await conn.prepare(sql)
        row = await conn.fetch(sql)
        print("row is: ", row)

        row = await conn.fetchrow("select '100'::int + 200 as result")
        print(row)  # <Record result=300>


if __name__ == "__main__":
    asyncio.run(query_with_age())
