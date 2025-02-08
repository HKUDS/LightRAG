"""
Sometimes you need to switch a storage solution, but you want to save LLM token and time.
This handy script helps you to copy the LLM caches from one storage solution to another.
(Not all the storage impl are supported)
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from lightrag.kg.postgres_impl import PostgreSQLDB, PGKVStorage
from lightrag.storage import JsonKVStorage
from lightrag.namespace import NameSpace

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{ROOT_DIR}/dickens"

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
        "database": "r2",
    }
)


async def copy_from_postgres_to_json():
    await postgres_db.initdb()

    from_llm_response_cache = PGKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={"embedding_batch_num": 6},
        embedding_func=None,
        db=postgres_db,
    )

    to_llm_response_cache = JsonKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
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


async def copy_from_json_to_postgres():
    await postgres_db.initdb()

    from_llm_response_cache = JsonKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={"working_dir": WORKING_DIR},
        embedding_func=None,
    )

    to_llm_response_cache = PGKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={"embedding_batch_num": 6},
        embedding_func=None,
        db=postgres_db,
    )

    for mode in await from_llm_response_cache.all_keys():
        print(f"Copying {mode}")
        caches = await from_llm_response_cache.get_by_id(mode)
        for k, v in caches.items():
            item = {mode: {k: v}}
            print(f"\tCopying {item}")
            await to_llm_response_cache.upsert(item)


if __name__ == "__main__":
    asyncio.run(copy_from_json_to_postgres())
