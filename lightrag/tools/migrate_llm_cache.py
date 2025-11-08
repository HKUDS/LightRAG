#!/usr/bin/env python3
"""
LLM Cache Migration Tool for LightRAG

This tool migrates LLM response cache (default:extract:* and default:summary:*)
between different KV storage implementations while preserving workspace isolation.

Usage:
    python -m lightrag.tools.migrate_llm_cache
    # or
    python lightrag/tools/migrate_llm_cache.py

Supported KV Storage Types:
    - JsonKVStorage
    - RedisKVStorage
    - PGKVStorage
    - MongoKVStorage
"""

import asyncio
import os
import sys
import time
from typing import Any, Dict, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.namespace import NameSpace
from lightrag.utils import setup_logger

# Load environment variables
# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Setup logger
setup_logger("lightrag", level="INFO")

# Storage type configurations
STORAGE_TYPES = {
    "1": "JsonKVStorage",
    "2": "RedisKVStorage",
    "3": "PGKVStorage",
    "4": "MongoKVStorage",
}

# Workspace environment variable mapping
WORKSPACE_ENV_MAP = {
    "PGKVStorage": "POSTGRES_WORKSPACE",
    "MongoKVStorage": "MONGODB_WORKSPACE",
    "RedisKVStorage": "REDIS_WORKSPACE",
}

# Default batch size for migration
DEFAULT_BATCH_SIZE = 1000


# Default count batch size for efficient counting
DEFAULT_COUNT_BATCH_SIZE = 1000


@dataclass
class MigrationStats:
    """Migration statistics and error tracking"""

    total_source_records: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    successful_records: int = 0
    failed_records: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_error(self, batch_idx: int, error: Exception, batch_size: int):
        """Record batch error"""
        self.errors.append(
            {
                "batch": batch_idx,
                "error_type": type(error).__name__,
                "error_msg": str(error),
                "records_lost": batch_size,
                "timestamp": time.time(),
            }
        )
        self.failed_batches += 1
        self.failed_records += batch_size


class MigrationTool:
    """LLM Cache Migration Tool"""

    def __init__(self):
        self.source_storage = None
        self.target_storage = None
        self.source_workspace = ""
        self.target_workspace = ""
        self.batch_size = DEFAULT_BATCH_SIZE

    def get_workspace_for_storage(self, storage_name: str) -> str:
        """Get workspace for a specific storage type

        Priority: Storage-specific env var > WORKSPACE env var > empty string

        Args:
            storage_name: Storage implementation name

        Returns:
            Workspace name
        """
        # Check storage-specific workspace
        if storage_name in WORKSPACE_ENV_MAP:
            specific_workspace = os.getenv(WORKSPACE_ENV_MAP[storage_name])
            if specific_workspace:
                return specific_workspace

        # Check generic WORKSPACE
        workspace = os.getenv("WORKSPACE", "")
        return workspace

    def check_env_vars(self, storage_name: str) -> bool:
        """Check if all required environment variables exist

        Args:
            storage_name: Storage implementation name

        Returns:
            True if all required env vars exist, False otherwise
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            print(
                f"✗ Missing required environment variables: {', '.join(missing_vars)}"
            )
            return False

        print("✓ All required environment variables are set")
        return True

    def get_storage_class(self, storage_name: str):
        """Dynamically import and return storage class

        Args:
            storage_name: Storage implementation name

        Returns:
            Storage class
        """
        if storage_name == "JsonKVStorage":
            from lightrag.kg.json_kv_impl import JsonKVStorage

            return JsonKVStorage
        elif storage_name == "RedisKVStorage":
            from lightrag.kg.redis_impl import RedisKVStorage

            return RedisKVStorage
        elif storage_name == "PGKVStorage":
            from lightrag.kg.postgres_impl import PGKVStorage

            return PGKVStorage
        elif storage_name == "MongoKVStorage":
            from lightrag.kg.mongo_impl import MongoKVStorage

            return MongoKVStorage
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    async def initialize_storage(self, storage_name: str, workspace: str):
        """Initialize storage instance

        Args:
            storage_name: Storage implementation name
            workspace: Workspace name

        Returns:
            Initialized storage instance
        """
        storage_class = self.get_storage_class(storage_name)

        # Create global config
        global_config = {
            "working_dir": os.getenv("WORKING_DIR", "./rag_storage"),
            "embedding_batch_num": 10,
        }

        # Initialize storage
        storage = storage_class(
            namespace=NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
            workspace=workspace,
            global_config=global_config,
            embedding_func=None,
        )

        # Initialize the storage
        await storage.initialize()

        return storage

    async def get_default_caches_json(self, storage) -> Dict[str, Any]:
        """Get default caches from JsonKVStorage

        Args:
            storage: JsonKVStorage instance

        Returns:
            Dictionary of cache entries with default:extract:* or default:summary:* keys
        """
        # Access _data directly - it's a dict from shared_storage
        async with storage._storage_lock:
            filtered = {}
            for key, value in storage._data.items():
                if key.startswith("default:extract:") or key.startswith(
                    "default:summary:"
                ):
                    filtered[key] = value
            return filtered

    async def get_default_caches_redis(
        self, storage, batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Get default caches from RedisKVStorage with pagination

        Args:
            storage: RedisKVStorage instance
            batch_size: Number of keys to process per batch

        Returns:
            Dictionary of cache entries with default:extract:* or default:summary:* keys
        """
        import json

        cache_data = {}

        # Use _get_redis_connection() context manager
        async with storage._get_redis_connection() as redis:
            for pattern in ["default:extract:*", "default:summary:*"]:
                # Add namespace prefix to pattern
                prefixed_pattern = f"{storage.final_namespace}:{pattern}"
                cursor = 0

                while True:
                    # SCAN already implements cursor-based pagination
                    cursor, keys = await redis.scan(
                        cursor, match=prefixed_pattern, count=batch_size
                    )

                    if keys:
                        # Process this batch using pipeline with error handling
                        try:
                            pipe = redis.pipeline()
                            for key in keys:
                                pipe.get(key)
                            values = await pipe.execute()

                            for key, value in zip(keys, values):
                                if value:
                                    key_str = (
                                        key.decode() if isinstance(key, bytes) else key
                                    )
                                    # Remove namespace prefix to get original key
                                    original_key = key_str.replace(
                                        f"{storage.final_namespace}:", "", 1
                                    )
                                    cache_data[original_key] = json.loads(value)

                        except Exception as e:
                            # Pipeline execution failed, fall back to individual gets
                            print(
                                f"⚠️  Pipeline execution failed for batch, using individual gets: {e}"
                            )
                            for key in keys:
                                try:
                                    value = await redis.get(key)
                                    if value:
                                        key_str = (
                                            key.decode()
                                            if isinstance(key, bytes)
                                            else key
                                        )
                                        original_key = key_str.replace(
                                            f"{storage.final_namespace}:", "", 1
                                        )
                                        cache_data[original_key] = json.loads(value)
                                except Exception as individual_error:
                                    print(
                                        f"⚠️  Failed to get individual key {key}: {individual_error}"
                                    )
                                    continue

                    if cursor == 0:
                        break

                    # Yield control periodically to avoid blocking
                    await asyncio.sleep(0)

        return cache_data

    async def get_default_caches_pg(
        self, storage, batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Get default caches from PGKVStorage with pagination

        Args:
            storage: PGKVStorage instance
            batch_size: Number of records to fetch per batch

        Returns:
            Dictionary of cache entries with default:extract:* or default:summary:* keys
        """
        from lightrag.kg.postgres_impl import namespace_to_table_name

        cache_data = {}
        table_name = namespace_to_table_name(storage.namespace)
        offset = 0

        while True:
            # Use LIMIT and OFFSET for pagination
            query = f"""
                SELECT id as key, original_prompt, return_value, chunk_id, cache_type, queryparam,
                       EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                       EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                FROM {table_name}
                WHERE workspace = $1
                AND (id LIKE 'default:extract:%' OR id LIKE 'default:summary:%')
                ORDER BY id
                LIMIT $2 OFFSET $3
            """

            results = await storage.db.query(
                query, [storage.workspace, batch_size, offset], multirows=True
            )

            if not results:
                break

            for row in results:
                # Map PostgreSQL fields to cache format
                cache_entry = {
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": row.get("queryparam"),
                    "create_time": row.get("create_time", 0),
                    "update_time": row.get("update_time", 0),
                }
                cache_data[row["key"]] = cache_entry

            # If we got fewer results than batch_size, we're done
            if len(results) < batch_size:
                break

            offset += batch_size

            # Yield control periodically
            await asyncio.sleep(0)

        return cache_data

    async def get_default_caches_mongo(
        self, storage, batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Get default caches from MongoKVStorage with cursor-based pagination

        Args:
            storage: MongoKVStorage instance
            batch_size: Number of documents to process per batch

        Returns:
            Dictionary of cache entries with default:extract:* or default:summary:* keys
        """
        cache_data = {}

        # MongoDB query with regex - use _data not collection
        query = {"_id": {"$regex": "^default:(extract|summary):"}}

        # Use cursor without to_list() - process in batches
        cursor = storage._data.find(query).batch_size(batch_size)

        async for doc in cursor:
            # Process each document as it comes
            doc_copy = doc.copy()
            key = doc_copy.pop("_id")

            # Filter ALL MongoDB/database-specific fields
            # Following .clinerules: "Always filter deprecated/incompatible fields during deserialization"
            for field_name in ["namespace", "workspace", "_id", "content"]:
                doc_copy.pop(field_name, None)

            cache_data[key] = doc_copy

            # Periodically yield control (every batch_size documents)
            if len(cache_data) % batch_size == 0:
                await asyncio.sleep(0)

        return cache_data

    async def get_default_caches(self, storage, storage_name: str) -> Dict[str, Any]:
        """Get default caches from any storage type

        Args:
            storage: Storage instance
            storage_name: Storage type name

        Returns:
            Dictionary of cache entries
        """
        if storage_name == "JsonKVStorage":
            return await self.get_default_caches_json(storage)
        elif storage_name == "RedisKVStorage":
            return await self.get_default_caches_redis(storage)
        elif storage_name == "PGKVStorage":
            return await self.get_default_caches_pg(storage)
        elif storage_name == "MongoKVStorage":
            return await self.get_default_caches_mongo(storage)
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    async def count_default_caches_json(self, storage) -> int:
        """Count default caches in JsonKVStorage - O(N) but very fast in-memory

        Args:
            storage: JsonKVStorage instance

        Returns:
            Total count of cache records
        """
        async with storage._storage_lock:
            return sum(
                1
                for key in storage._data.keys()
                if key.startswith("default:extract:")
                or key.startswith("default:summary:")
            )

    async def count_default_caches_redis(self, storage) -> int:
        """Count default caches in RedisKVStorage using SCAN with progress display

        Args:
            storage: RedisKVStorage instance

        Returns:
            Total count of cache records
        """
        count = 0
        print("Scanning Redis keys...", end="", flush=True)

        async with storage._get_redis_connection() as redis:
            for pattern in ["default:extract:*", "default:summary:*"]:
                prefixed_pattern = f"{storage.final_namespace}:{pattern}"
                cursor = 0
                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=prefixed_pattern, count=DEFAULT_COUNT_BATCH_SIZE
                    )
                    count += len(keys)

                    # Show progress
                    print(
                        f"\rScanning Redis keys... found {count:,} records",
                        end="",
                        flush=True,
                    )

                    if cursor == 0:
                        break

        print()  # New line after progress
        return count

    async def count_default_caches_pg(self, storage) -> int:
        """Count default caches in PostgreSQL using COUNT(*) with progress indicator

        Args:
            storage: PGKVStorage instance

        Returns:
            Total count of cache records
        """
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table_name = namespace_to_table_name(storage.namespace)

        query = f"""
            SELECT COUNT(*) as count
            FROM {table_name}
            WHERE workspace = $1
            AND (id LIKE 'default:extract:%' OR id LIKE 'default:summary:%')
        """

        print("Counting PostgreSQL records...", end="", flush=True)
        start_time = time.time()

        result = await storage.db.query(query, [storage.workspace])

        elapsed = time.time() - start_time
        if elapsed > 1:
            print(f" (took {elapsed:.1f}s)", end="")
        print()  # New line

        return result["count"] if result else 0

    async def count_default_caches_mongo(self, storage) -> int:
        """Count default caches in MongoDB using count_documents with progress indicator

        Args:
            storage: MongoKVStorage instance

        Returns:
            Total count of cache records
        """
        query = {"_id": {"$regex": "^default:(extract|summary):"}}

        print("Counting MongoDB documents...", end="", flush=True)
        start_time = time.time()

        count = await storage._data.count_documents(query)

        elapsed = time.time() - start_time
        if elapsed > 1:
            print(f" (took {elapsed:.1f}s)", end="")
        print()  # New line

        return count

    async def count_default_caches(self, storage, storage_name: str) -> int:
        """Count default caches from any storage type efficiently

        Args:
            storage: Storage instance
            storage_name: Storage type name

        Returns:
            Total count of cache records
        """
        if storage_name == "JsonKVStorage":
            return await self.count_default_caches_json(storage)
        elif storage_name == "RedisKVStorage":
            return await self.count_default_caches_redis(storage)
        elif storage_name == "PGKVStorage":
            return await self.count_default_caches_pg(storage)
        elif storage_name == "MongoKVStorage":
            return await self.count_default_caches_mongo(storage)
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    async def stream_default_caches_json(self, storage, batch_size: int):
        """Stream default caches from JsonKVStorage - yields batches

        Args:
            storage: JsonKVStorage instance
            batch_size: Size of each batch to yield

        Yields:
            Dictionary batches of cache entries
        """
        async with storage._storage_lock:
            batch = {}
            for key, value in storage._data.items():
                if key.startswith("default:extract:") or key.startswith(
                    "default:summary:"
                ):
                    batch[key] = value
                    if len(batch) >= batch_size:
                        yield batch
                        batch = {}
            # Yield remaining items
            if batch:
                yield batch

    async def stream_default_caches_redis(self, storage, batch_size: int):
        """Stream default caches from RedisKVStorage - yields batches

        Args:
            storage: RedisKVStorage instance
            batch_size: Size of each batch to yield

        Yields:
            Dictionary batches of cache entries
        """
        import json

        async with storage._get_redis_connection() as redis:
            for pattern in ["default:extract:*", "default:summary:*"]:
                prefixed_pattern = f"{storage.final_namespace}:{pattern}"
                cursor = 0

                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=prefixed_pattern, count=batch_size
                    )

                    if keys:
                        try:
                            pipe = redis.pipeline()
                            for key in keys:
                                pipe.get(key)
                            values = await pipe.execute()

                            batch = {}
                            for key, value in zip(keys, values):
                                if value:
                                    key_str = (
                                        key.decode() if isinstance(key, bytes) else key
                                    )
                                    original_key = key_str.replace(
                                        f"{storage.final_namespace}:", "", 1
                                    )
                                    batch[original_key] = json.loads(value)

                            if batch:
                                yield batch

                        except Exception as e:
                            print(f"⚠️  Pipeline execution failed for batch: {e}")
                            # Fall back to individual gets
                            batch = {}
                            for key in keys:
                                try:
                                    value = await redis.get(key)
                                    if value:
                                        key_str = (
                                            key.decode()
                                            if isinstance(key, bytes)
                                            else key
                                        )
                                        original_key = key_str.replace(
                                            f"{storage.final_namespace}:", "", 1
                                        )
                                        batch[original_key] = json.loads(value)
                                except Exception as individual_error:
                                    print(
                                        f"⚠️  Failed to get individual key {key}: {individual_error}"
                                    )
                                    continue

                            if batch:
                                yield batch

                    if cursor == 0:
                        break

                    await asyncio.sleep(0)

    async def stream_default_caches_pg(self, storage, batch_size: int):
        """Stream default caches from PostgreSQL - yields batches

        Args:
            storage: PGKVStorage instance
            batch_size: Size of each batch to yield

        Yields:
            Dictionary batches of cache entries
        """
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table_name = namespace_to_table_name(storage.namespace)
        offset = 0

        while True:
            query = f"""
                SELECT id as key, original_prompt, return_value, chunk_id, cache_type, queryparam,
                       EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                       EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                FROM {table_name}
                WHERE workspace = $1
                AND (id LIKE 'default:extract:%' OR id LIKE 'default:summary:%')
                ORDER BY id
                LIMIT $2 OFFSET $3
            """

            results = await storage.db.query(
                query, [storage.workspace, batch_size, offset], multirows=True
            )

            if not results:
                break

            batch = {}
            for row in results:
                cache_entry = {
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type"),
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": row.get("queryparam"),
                    "create_time": row.get("create_time", 0),
                    "update_time": row.get("update_time", 0),
                }
                batch[row["key"]] = cache_entry

            if batch:
                yield batch

            if len(results) < batch_size:
                break

            offset += batch_size
            await asyncio.sleep(0)

    async def stream_default_caches_mongo(self, storage, batch_size: int):
        """Stream default caches from MongoDB - yields batches

        Args:
            storage: MongoKVStorage instance
            batch_size: Size of each batch to yield

        Yields:
            Dictionary batches of cache entries
        """
        query = {"_id": {"$regex": "^default:(extract|summary):"}}
        cursor = storage._data.find(query).batch_size(batch_size)

        batch = {}
        async for doc in cursor:
            doc_copy = doc.copy()
            key = doc_copy.pop("_id")

            # Filter MongoDB/database-specific fields
            for field_name in ["namespace", "workspace", "_id", "content"]:
                doc_copy.pop(field_name, None)

            batch[key] = doc_copy

            if len(batch) >= batch_size:
                yield batch
                batch = {}

        # Yield remaining items
        if batch:
            yield batch

    async def stream_default_caches(
        self, storage, storage_name: str, batch_size: int = None
    ):
        """Stream default caches from any storage type - unified interface

        Args:
            storage: Storage instance
            storage_name: Storage type name
            batch_size: Size of each batch to yield (defaults to self.batch_size)

        Yields:
            Dictionary batches of cache entries
        """
        if batch_size is None:
            batch_size = self.batch_size

        if storage_name == "JsonKVStorage":
            async for batch in self.stream_default_caches_json(storage, batch_size):
                yield batch
        elif storage_name == "RedisKVStorage":
            async for batch in self.stream_default_caches_redis(storage, batch_size):
                yield batch
        elif storage_name == "PGKVStorage":
            async for batch in self.stream_default_caches_pg(storage, batch_size):
                yield batch
        elif storage_name == "MongoKVStorage":
            async for batch in self.stream_default_caches_mongo(storage, batch_size):
                yield batch
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    async def count_cache_types(self, cache_data: Dict[str, Any]) -> Dict[str, int]:
        """Count cache entries by type

        Args:
            cache_data: Dictionary of cache entries

        Returns:
            Dictionary with counts for each cache type
        """
        counts = {
            "extract": 0,
            "summary": 0,
        }

        for key in cache_data.keys():
            if key.startswith("default:extract:"):
                counts["extract"] += 1
            elif key.startswith("default:summary:"):
                counts["summary"] += 1

        return counts

    def print_header(self):
        """Print tool header"""
        print("\n" + "=" * 50)
        print("LLM Cache Migration Tool - LightRAG")
        print("=" * 50)

    def print_storage_types(self):
        """Print available storage types"""
        print("\nSupported KV Storage Types:")
        for key, value in STORAGE_TYPES.items():
            print(f"[{key}] {value}")

    def get_user_choice(
        self, prompt: str, valid_choices: list, allow_exit: bool = False
    ) -> str:
        """Get user choice with validation

        Args:
            prompt: Prompt message
            valid_choices: List of valid choices
            allow_exit: If True, allow user to press Enter or input '0' to exit

        Returns:
            User's choice, or None if user chose to exit
        """
        exit_hint = " (Press Enter or 0 to exit)" if allow_exit else ""
        while True:
            choice = input(f"\n{prompt}{exit_hint}: ").strip()

            # Check for exit
            if allow_exit and (choice == "" or choice == "0"):
                return None

            if choice in valid_choices:
                return choice
            print(f"✗ Invalid choice, please enter one of: {', '.join(valid_choices)}")

    async def setup_storage(
        self, storage_type: str, use_streaming: bool = False
    ) -> tuple:
        """Setup and initialize storage

        Args:
            storage_type: Type label (source/target)
            use_streaming: If True, only count records without loading. If False, load all data (legacy mode)

        Returns:
            Tuple of (storage_instance, storage_name, workspace, total_count)
            Returns (None, None, None, 0) if user chooses to exit
        """
        print(f"\n=== {storage_type} Storage Setup ===")

        # Get storage type choice - allow exit for source storage
        allow_exit = storage_type == "Source"
        choice = self.get_user_choice(
            f"Select {storage_type} storage type (1-4)",
            list(STORAGE_TYPES.keys()),
            allow_exit=allow_exit,
        )

        # Handle exit
        if choice is None:
            print("\n✓ Migration cancelled by user")
            return None, None, None, 0

        storage_name = STORAGE_TYPES[choice]

        # Check environment variables
        print("\nChecking environment variables...")
        if not self.check_env_vars(storage_name):
            return None, None, None, 0

        # Get workspace
        workspace = self.get_workspace_for_storage(storage_name)

        # Initialize storage
        print(f"\nInitializing {storage_type} storage...")
        try:
            storage = await self.initialize_storage(storage_name, workspace)
            print(f"- Storage Type: {storage_name}")
            print(f"- Workspace: {workspace if workspace else '(default)'}")
            print("- Connection Status: ✓ Success")
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            return None, None, None, 0

        # Count cache records efficiently
        print(f"\n{'Counting' if use_streaming else 'Loading'} cache records...")
        try:
            if use_streaming:
                # Use efficient counting without loading data
                total_count = await self.count_default_caches(storage, storage_name)
                print(f"- Total: {total_count:,} records")
            else:
                # Legacy mode: load all data
                cache_data = await self.get_default_caches(storage, storage_name)
                counts = await self.count_cache_types(cache_data)
                total_count = len(cache_data)

                print(f"- default:extract: {counts['extract']:,} records")
                print(f"- default:summary: {counts['summary']:,} records")
                print(f"- Total: {total_count:,} records")
        except Exception as e:
            print(f"✗ {'Counting' if use_streaming else 'Loading'} failed: {e}")
            return None, None, None, 0

        return storage, storage_name, workspace, total_count

    async def migrate_caches(
        self, source_data: Dict[str, Any], target_storage, target_storage_name: str
    ) -> MigrationStats:
        """Migrate caches in batches with error tracking (Legacy mode - loads all data)

        Args:
            source_data: Source cache data
            target_storage: Target storage instance
            target_storage_name: Target storage type name

        Returns:
            MigrationStats object with migration results and errors
        """
        stats = MigrationStats()
        stats.total_source_records = len(source_data)

        if stats.total_source_records == 0:
            print("\nNo records to migrate")
            return stats

        # Convert to list for batching
        items = list(source_data.items())
        stats.total_batches = (
            stats.total_source_records + self.batch_size - 1
        ) // self.batch_size

        print("\n=== Starting Migration ===")

        for batch_idx in range(stats.total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, stats.total_source_records)
            batch_items = items[start_idx:end_idx]
            batch_data = dict(batch_items)

            # Determine current cache type for display
            current_key = batch_items[0][0]
            cache_type = "extract" if "extract" in current_key else "summary"

            try:
                # Attempt to write batch
                await target_storage.upsert(batch_data)

                # Success - update stats
                stats.successful_batches += 1
                stats.successful_records += len(batch_data)

                # Calculate progress
                progress = (end_idx / stats.total_source_records) * 100
                bar_length = 20
                filled_length = int(bar_length * end_idx // stats.total_source_records)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                print(
                    f"Batch {batch_idx + 1}/{stats.total_batches}: {bar} "
                    f"{end_idx:,}/{stats.total_source_records:,} ({progress:.0f}%) - "
                    f"default:{cache_type} ✓"
                )

            except Exception as e:
                # Error - record and continue
                stats.add_error(batch_idx + 1, e, len(batch_data))

                print(
                    f"Batch {batch_idx + 1}/{stats.total_batches}: ✗ FAILED - "
                    f"{type(e).__name__}: {str(e)}"
                )

        # Final persist
        print("\nPersisting data to disk...")
        try:
            await target_storage.index_done_callback()
            print("✓ Data persisted successfully")
        except Exception as e:
            print(f"✗ Persist failed: {e}")
            stats.add_error(0, e, 0)  # batch 0 = persist error

        return stats

    async def migrate_caches_streaming(
        self,
        source_storage,
        source_storage_name: str,
        target_storage,
        target_storage_name: str,
        total_records: int,
    ) -> MigrationStats:
        """Migrate caches using streaming approach - minimal memory footprint

        Args:
            source_storage: Source storage instance
            source_storage_name: Source storage type name
            target_storage: Target storage instance
            target_storage_name: Target storage type name
            total_records: Total number of records to migrate

        Returns:
            MigrationStats object with migration results and errors
        """
        stats = MigrationStats()
        stats.total_source_records = total_records

        if stats.total_source_records == 0:
            print("\nNo records to migrate")
            return stats

        # Calculate total batches
        stats.total_batches = (total_records + self.batch_size - 1) // self.batch_size

        print("\n=== Starting Streaming Migration ===")
        print(
            f"💡 Memory-optimized mode: Processing {self.batch_size:,} records at a time\n"
        )

        batch_idx = 0

        # Stream batches from source and write to target immediately
        async for batch in self.stream_default_caches(
            source_storage, source_storage_name
        ):
            batch_idx += 1

            # Determine current cache type for display
            if batch:
                first_key = next(iter(batch.keys()))
                cache_type = "extract" if "extract" in first_key else "summary"
            else:
                cache_type = "unknown"

            try:
                # Write batch to target storage
                await target_storage.upsert(batch)

                # Success - update stats
                stats.successful_batches += 1
                stats.successful_records += len(batch)

                # Calculate progress with known total
                progress = (stats.successful_records / total_records) * 100
                bar_length = 20
                filled_length = int(
                    bar_length * stats.successful_records // total_records
                )
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                print(
                    f"Batch {batch_idx}/{stats.total_batches}: {bar} "
                    f"{stats.successful_records:,}/{total_records:,} ({progress:.1f}%) - "
                    f"default:{cache_type} ✓"
                )

            except Exception as e:
                # Error - record and continue
                stats.add_error(batch_idx, e, len(batch))

                print(
                    f"Batch {batch_idx}/{stats.total_batches}: ✗ FAILED - "
                    f"{type(e).__name__}: {str(e)}"
                )

        # Final persist
        print("\nPersisting data to disk...")
        try:
            await target_storage.index_done_callback()
            print("✓ Data persisted successfully")
        except Exception as e:
            print(f"✗ Persist failed: {e}")
            stats.add_error(0, e, 0)  # batch 0 = persist error

        return stats

    def print_migration_report(self, stats: MigrationStats):
        """Print comprehensive migration report

        Args:
            stats: MigrationStats object with migration results
        """
        print("\n" + "=" * 60)
        print("Migration Complete - Final Report")
        print("=" * 60)

        # Overall statistics
        print("\n📊 Statistics:")
        print(f"  Total source records:    {stats.total_source_records:,}")
        print(f"  Total batches:           {stats.total_batches:,}")
        print(f"  Successful batches:      {stats.successful_batches:,}")
        print(f"  Failed batches:          {stats.failed_batches:,}")
        print(f"  Successfully migrated:   {stats.successful_records:,}")
        print(f"  Failed to migrate:       {stats.failed_records:,}")

        # Success rate
        success_rate = (
            (stats.successful_records / stats.total_source_records * 100)
            if stats.total_source_records > 0
            else 0
        )
        print(f"  Success rate:            {success_rate:.2f}%")

        # Error details
        if stats.errors:
            print(f"\n⚠️  Errors encountered: {len(stats.errors)}")
            print("\nError Details:")
            print("-" * 60)

            # Group errors by type
            error_types = {}
            for error in stats.errors:
                err_type = error["error_type"]
                error_types[err_type] = error_types.get(err_type, 0) + 1

            print("\nError Summary:")
            for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"  - {err_type}: {count} occurrence(s)")

            print("\nFirst 5 errors:")
            for i, error in enumerate(stats.errors[:5], 1):
                print(f"\n  {i}. Batch {error['batch']}")
                print(f"     Type: {error['error_type']}")
                print(f"     Message: {error['error_msg']}")
                print(f"     Records lost: {error['records_lost']:,}")

            if len(stats.errors) > 5:
                print(f"\n  ... and {len(stats.errors) - 5} more errors")

            print("\n" + "=" * 60)
            print("⚠️  WARNING: Migration completed with errors!")
            print("   Please review the error details above.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✓ SUCCESS: All records migrated successfully!")
            print("=" * 60)

    async def run(self):
        """Run the migration tool with streaming approach"""
        try:
            # Initialize shared storage (REQUIRED for storage classes to work)
            from lightrag.kg.shared_storage import initialize_share_data

            initialize_share_data(workers=1)

            # Print header
            self.print_header()
            self.print_storage_types()

            # Setup source storage with streaming (only count, don't load all data)
            (
                self.source_storage,
                source_storage_name,
                self.source_workspace,
                source_count,
            ) = await self.setup_storage("Source", use_streaming=True)

            # Check if user cancelled (setup_storage returns None for all fields)
            if self.source_storage is None:
                return

            if source_count == 0:
                print("\n⚠ Source storage has no cache records to migrate")
                # Cleanup
                await self.source_storage.finalize()
                return

            # Setup target storage with streaming (only count, don't load all data)
            (
                self.target_storage,
                target_storage_name,
                self.target_workspace,
                target_count,
            ) = await self.setup_storage("Target", use_streaming=True)

            if not self.target_storage:
                print("\n✗ Target storage setup failed")
                # Cleanup source
                await self.source_storage.finalize()
                return

            # Show migration summary
            print("\n" + "=" * 50)
            print("Migration Confirmation")
            print("=" * 50)
            print(
                f"Source: {source_storage_name} (workspace: {self.source_workspace if self.source_workspace else '(default)'}) - {source_count:,} records"
            )
            print(
                f"Target: {target_storage_name} (workspace: {self.target_workspace if self.target_workspace else '(default)'}) - {target_count:,} records"
            )
            print(f"Batch Size: {self.batch_size:,} records/batch")
            print("Memory Mode: Streaming (memory-optimized)")

            if target_count > 0:
                print(
                    f"\n⚠️ Warning: Target storage already has {target_count:,} records"
                )
                print("Migration will overwrite records with the same keys")

            # Confirm migration
            confirm = input("\nContinue? (y/n): ").strip().lower()
            if confirm != "y":
                print("\n✗ Migration cancelled")
                # Cleanup
                await self.source_storage.finalize()
                await self.target_storage.finalize()
                return

            # Perform streaming migration with error tracking
            stats = await self.migrate_caches_streaming(
                self.source_storage,
                source_storage_name,
                self.target_storage,
                target_storage_name,
                source_count,
            )

            # Print comprehensive migration report
            self.print_migration_report(stats)

            # Cleanup
            await self.source_storage.finalize()
            await self.target_storage.finalize()

        except KeyboardInterrupt:
            print("\n\n✗ Migration interrupted by user")
        except Exception as e:
            print(f"\n✗ Migration failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Ensure cleanup
            if self.source_storage:
                try:
                    await self.source_storage.finalize()
                except Exception:
                    pass
            if self.target_storage:
                try:
                    await self.target_storage.finalize()
                except Exception:
                    pass

            # Finalize shared storage
            try:
                from lightrag.kg.shared_storage import finalize_share_data

                finalize_share_data()
            except Exception:
                pass


async def main():
    """Main entry point"""
    tool = MigrationTool()
    await tool.run()


if __name__ == "__main__":
    asyncio.run(main())
