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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lightrag.kg import STORAGE_ENV_REQUIREMENTS
from lightrag.namespace import NameSpace
from lightrag.utils import setup_logger

# Load environment variables
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

    def get_user_choice(self, prompt: str, valid_choices: list) -> str:
        """Get user choice with validation

        Args:
            prompt: Prompt message
            valid_choices: List of valid choices

        Returns:
            User's choice
        """
        while True:
            choice = input(f"\n{prompt}: ").strip()
            if choice in valid_choices:
                return choice
            print(f"✗ Invalid choice, please enter one of: {', '.join(valid_choices)}")

    async def setup_storage(self, storage_type: str) -> tuple:
        """Setup and initialize storage

        Args:
            storage_type: Type label (source/target)

        Returns:
            Tuple of (storage_instance, storage_name, workspace, cache_data)
        """
        print(f"\n=== {storage_type} Storage Setup ===")

        # Get storage type choice
        choice = self.get_user_choice(
            f"Select {storage_type} storage type (1-4)", list(STORAGE_TYPES.keys())
        )
        storage_name = STORAGE_TYPES[choice]

        # Check environment variables
        print("\nChecking environment variables...")
        if not self.check_env_vars(storage_name):
            return None, None, None, None

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
            return None, None, None, None

        # Get cache data
        print("\nCounting cache records...")
        try:
            cache_data = await self.get_default_caches(storage, storage_name)
            counts = await self.count_cache_types(cache_data)

            print(f"- default:extract: {counts['extract']:,} records")
            print(f"- default:summary: {counts['summary']:,} records")
            print(f"- Total: {len(cache_data):,} records")
        except Exception as e:
            print(f"✗ Counting failed: {e}")
            return None, None, None, None

        return storage, storage_name, workspace, cache_data

    async def migrate_caches(
        self, source_data: Dict[str, Any], target_storage, target_storage_name: str
    ) -> MigrationStats:
        """Migrate caches in batches with error tracking

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
        """Run the migration tool"""
        try:
            # Initialize shared storage (REQUIRED for storage classes to work)
            from lightrag.kg.shared_storage import initialize_share_data
            initialize_share_data(workers=1)
            
            # Print header
            self.print_header()
            self.print_storage_types()

            # Setup source storage
            (
                self.source_storage,
                source_storage_name,
                self.source_workspace,
                source_data,
            ) = await self.setup_storage("Source")

            if not self.source_storage:
                print("\n✗ Source storage setup failed")
                return

            if not source_data:
                print("\n⚠ Source storage has no cache records to migrate")
                # Cleanup
                await self.source_storage.finalize()
                return

            # Setup target storage
            (
                self.target_storage,
                target_storage_name,
                self.target_workspace,
                target_data,
            ) = await self.setup_storage("Target")

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
                f"Source: {source_storage_name} (workspace: {self.source_workspace if self.source_workspace else '(default)'}) - {len(source_data):,} records"
            )
            print(
                f"Target: {target_storage_name} (workspace: {self.target_workspace if self.target_workspace else '(default)'}) - {len(target_data):,} records"
            )
            print(f"Batch Size: {self.batch_size:,} records/batch")

            if target_data:
                print(
                    f"\n⚠ Warning: Target storage already has {len(target_data):,} records"
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

            # Perform migration with error tracking
            stats = await self.migrate_caches(
                source_data, self.target_storage, target_storage_name
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
