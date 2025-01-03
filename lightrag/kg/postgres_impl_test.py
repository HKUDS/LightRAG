import asyncio
import asyncpg
import sys, os

import psycopg
from psycopg_pool import AsyncConnectionPool
from lightrag.kg.postgres_impl import PostgreSQLDB, PGGraphStorage

DB="rag"
USER="rag"
PASSWORD="rag"
HOST="localhost"
PORT="15432"
os.environ["AGE_GRAPH_NAME"] = "dickens"

if sys.platform.startswith("win"):
    import asyncio.windows_events
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def get_pool():
    return await asyncpg.create_pool(
        f"postgres://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}",
        min_size=10,  # 连接池初始化时默认的最小连接数, 默认为1 0
        max_size=10,  # 连接池的最大连接数, 默认为 10
        max_queries=5000,  # 每个链接最大查询数量, 超过了就换新的连接, 默认 5000
        # 最大不活跃时间, 默认 300.0, 超过这个时间的连接就会被关闭, 传入 0 的话则永不关闭
        max_inactive_connection_lifetime=300.0
    )

async def main1():
    connection_string = f"dbname='{DB}' user='{USER}' password='{PASSWORD}' host='{HOST}' port={PORT}"
    pool = AsyncConnectionPool(connection_string, open=False)
    await pool.open()

    try:
        conn = await pool.getconn(timeout=10)
        async with conn.cursor() as curs:
            try:
                await curs.execute('SET search_path = ag_catalog, "$user", public')
                await curs.execute(f"SELECT create_graph('dickens-2')")
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
        "database": "rag",
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
    res = await graph.get_node('"CHRISTMAS-TIME"')
    print("Node is: ", res)

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
    res = await graph.get_edge('THE CRATCHITS', '"THE GIRLS"')
    print("Edge is: ", res)


async def main():
    pool = await get_pool()
    # 如果还有其它什么特殊参数，也可以直接往里面传递，因为设置了 **connect_kwargs
    # 专门用来设置一些数据库独有的某些属性
    # 从池子中取出一个连接
    sql = r"SELECT * FROM ag_catalog.cypher('dickens', $$ MATCH (n:帅哥) RETURN n $$) AS (n ag_catalog.agtype)"
    # cypher = "MATCH (n:how_are_you_doing) RETURN n"
    async with pool.acquire() as conn:
            try:
                await conn.execute("""SET search_path = ag_catalog, "$user", public;select create_graph('dickens')""")
            except asyncpg.exceptions.InvalidSchemaNameError:
                print("create_graph already exists")
            # stmt = await conn.prepare(sql)
            row = await conn.fetch(sql)
            print("row is: ", row)

            # 解决办法就是起一个别名
            row = await conn.fetchrow("select '100'::int + 200 as result")
            print(row)  # <Record result=300>
    # 我们的连接是从池子里面取出的，上下文结束之后会自动放回到到池子里面


if __name__ == '__main__':
    asyncio.run(query_with_age())


