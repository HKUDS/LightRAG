import asyncio
import logging
import os
from dotenv import load_dotenv

from lightrag.kg.postgres_impl import PostgreSQLDB, PGKVStorage
from lightrag.storage import JsonKVStorage

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{ROOT_DIR}/dickens-pg"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# AGE
os.environ["AGE_GRAPH_NAME"] = "chinese"

postgres_db = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "r1",
    }
)


async def main():
    await postgres_db.initdb()

    from_llm_response_cache = PGKVStorage(
        namespace="llm_response_cache",
        global_config={"embedding_batch_num": 6},
        embedding_func=None,
        db=postgres_db,
    )

    to_llm_response_cache = JsonKVStorage(
        namespace="llm_response_cache",
        global_config={"working_dir": WORKING_DIR},
        embedding_func=None,
    )

    kv = {}
    for c_id in await from_llm_response_cache.all_keys():
        print(f"Copying {c_id}")
        workspace = c_id["workspace"]
        mode = c_id["mode"]
        _id = c_id["id"]
        postgres_db.workspace = workspace
        obj = await from_llm_response_cache.get_by_mode_and_id(mode, _id)
        if mode not in kv:
            kv[mode] = {}
        kv[mode][_id] = obj[_id]
        print(f"Object {obj}")
    await to_llm_response_cache.upsert(kv)
    await to_llm_response_cache.index_done_callback()
    print("Mission accomplished!")


if __name__ == "__main__":
    asyncio.run(main())
