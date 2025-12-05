"""
Sometimes you need to switch a storage solution, but you want to save LLM token and time.
This handy script helps you to copy the LLM caches from one storage solution to another.
(Not all the storage impl are supported)
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.postgres_impl import PGKVStorage, PostgreSQLDB
from lightrag.namespace import NameSpace

load_dotenv()
ROOT_DIR = os.environ.get('ROOT_DIR')
WORKING_DIR = f'{ROOT_DIR}/dickens'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# AGE
os.environ['AGE_GRAPH_NAME'] = 'chinese'

postgres_db = PostgreSQLDB(
    config={
        'host': 'localhost',
        'port': 15432,
        'user': 'rag',
        'password': 'rag',
        'database': 'r2',
    }
)


async def copy_from_postgres_to_json():
    await postgres_db.initdb()

    from_llm_response_cache = PGKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={'embedding_batch_num': 6},
        embedding_func=None,
        db=postgres_db,
    )

    to_llm_response_cache = JsonKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={'working_dir': WORKING_DIR},
        embedding_func=None,
    )

    # Get all cache data using the new flattened structure
    all_data = await from_llm_response_cache.get_all()

    # Convert flattened data to hierarchical structure for JsonKVStorage
    kv = {}
    for flattened_key, cache_entry in all_data.items():
        # Parse flattened key: {mode}:{cache_type}:{hash}
        parts = flattened_key.split(':', 2)
        if len(parts) == 3:
            mode, _cache_type, hash_value = parts
            if mode not in kv:
                kv[mode] = {}
            kv[mode][hash_value] = cache_entry
            print(f'Copying {flattened_key} -> {mode}[{hash_value}]')
        else:
            print(f'Skipping invalid key format: {flattened_key}')

    await to_llm_response_cache.upsert(kv)
    await to_llm_response_cache.index_done_callback()
    print('Mission accomplished!')


async def copy_from_json_to_postgres():
    await postgres_db.initdb()

    from_llm_response_cache = JsonKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={'working_dir': WORKING_DIR},
        embedding_func=None,
    )

    to_llm_response_cache = PGKVStorage(
        namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
        global_config={'embedding_batch_num': 6},
        embedding_func=None,
        db=postgres_db,
    )

    # Get all cache data from JsonKVStorage (hierarchical structure)
    all_data = await from_llm_response_cache.get_all()

    # Convert hierarchical data to flattened structure for PGKVStorage
    flattened_data = {}
    for mode, mode_data in all_data.items():
        print(f'Processing mode: {mode}')
        for hash_value, cache_entry in mode_data.items():
            # Determine cache_type from cache entry or use default
            cache_type = cache_entry.get('cache_type', 'extract')
            # Create flattened key: {mode}:{cache_type}:{hash}
            flattened_key = f'{mode}:{cache_type}:{hash_value}'
            flattened_data[flattened_key] = cache_entry
            print(f'\tConverting {mode}[{hash_value}] -> {flattened_key}')

    # Upsert the flattened data
    await to_llm_response_cache.upsert(flattened_data)
    print('Mission accomplished!')


if __name__ == '__main__':
    asyncio.run(copy_from_json_to_postgres())
