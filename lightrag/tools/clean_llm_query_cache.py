#!/usr/bin/env python3
"""
LLM Query Cache Cleanup Tool for LightRAG

This tool cleans up LLM query cache (mix:*, hybrid:*, local:*, global:*)
from KV storage implementations while preserving workspace isolation.

Usage:
    python -m lightrag.tools.clean_llm_query_cache
    # or
    python lightrag/tools/clean_llm_query_cache.py

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
from lightrag.kg.shared_storage import set_all_update_flags
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

# Query cache modes
QUERY_MODES = ["mix", "hybrid", "local", "global"]

# Query cache types
CACHE_TYPES = ["query", "keywords"]

# Default batch size for deletion
DEFAULT_BATCH_SIZE = 1000

# ANSI color codes for terminal output
BOLD_CYAN = "\033[1;36m"
BOLD_RED = "\033[1;31m"
BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"


@dataclass
class CleanupStats:
    """Cleanup statistics and error tracking"""

    # Count by mode and cache_type before cleanup
    counts_before: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Deletion statistics
    total_to_delete: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    successfully_deleted: int = 0
    failed_to_delete: int = 0

    # Count by mode and cache_type after cleanup
    counts_after: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Error tracking
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
        self.failed_to_delete += batch_size

    def initialize_counts(self):
        """Initialize count dictionaries"""
        for mode in QUERY_MODES:
            self.counts_before[mode] = {"query": 0, "keywords": 0}
            self.counts_after[mode] = {"query": 0, "keywords": 0}


class CleanupTool:
    """LLM Query Cache Cleanup Tool"""

    def __init__(self):
        self.storage = None
        self.workspace = ""
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

    def check_config_ini_for_storage(self, storage_name: str) -> bool:
        """Check if config.ini has configuration for the storage type

        Args:
            storage_name: Storage implementation name

        Returns:
            True if config.ini has the necessary configuration
        """
        try:
            import configparser

            config = configparser.ConfigParser()
            config.read("config.ini", "utf-8")

            if storage_name == "RedisKVStorage":
                return config.has_option("redis", "uri")
            elif storage_name == "PGKVStorage":
                return (
                    config.has_option("postgres", "user")
                    and config.has_option("postgres", "password")
                    and config.has_option("postgres", "database")
                )
            elif storage_name == "MongoKVStorage":
                return config.has_option("mongodb", "uri") and config.has_option(
                    "mongodb", "database"
                )

            return False
        except Exception:
            return False

    def check_env_vars(self, storage_name: str) -> bool:
        """Check environment variables, show warnings if missing but don't fail

        Args:
            storage_name: Storage implementation name

        Returns:
            Always returns True (warnings only, no hard failure)
        """
        required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])

        if not required_vars:
            print("✓ No environment variables required")
            return True

        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            print(
                f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}"
            )

            # Check if config.ini has configuration
            has_config = self.check_config_ini_for_storage(storage_name)
            if has_config:
                print("   ✓ Found configuration in config.ini")
            else:
                print(f"   Will attempt to use defaults for {storage_name}")

            return True

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
        """Initialize storage instance with fallback to config.ini and defaults

        Args:
            storage_name: Storage implementation name
            workspace: Workspace name

        Returns:
            Initialized storage instance

        Raises:
            Exception: If initialization fails
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

        # Initialize the storage (may raise exception if connection fails)
        await storage.initialize()

        return storage

    async def count_query_caches_json(self, storage) -> Dict[str, Dict[str, int]]:
        """Count query caches in JsonKVStorage by mode and cache_type

        Args:
            storage: JsonKVStorage instance

        Returns:
            Dictionary with counts for each mode and cache_type
        """
        counts = {mode: {"query": 0, "keywords": 0} for mode in QUERY_MODES}

        async with storage._storage_lock:
            for key in storage._data.keys():
                for mode in QUERY_MODES:
                    if key.startswith(f"{mode}:query:"):
                        counts[mode]["query"] += 1
                    elif key.startswith(f"{mode}:keywords:"):
                        counts[mode]["keywords"] += 1

        return counts

    async def count_query_caches_redis(self, storage) -> Dict[str, Dict[str, int]]:
        """Count query caches in RedisKVStorage by mode and cache_type

        Args:
            storage: RedisKVStorage instance

        Returns:
            Dictionary with counts for each mode and cache_type
        """
        counts = {mode: {"query": 0, "keywords": 0} for mode in QUERY_MODES}

        print("Scanning Redis keys...", end="", flush=True)

        async with storage._get_redis_connection() as redis:
            for mode in QUERY_MODES:
                for cache_type in CACHE_TYPES:
                    pattern = f"{mode}:{cache_type}:*"
                    prefixed_pattern = f"{storage.final_namespace}:{pattern}"
                    cursor = 0

                    while True:
                        cursor, keys = await redis.scan(
                            cursor, match=prefixed_pattern, count=DEFAULT_BATCH_SIZE
                        )
                        counts[mode][cache_type] += len(keys)

                        if cursor == 0:
                            break

        print()  # New line after progress
        return counts

    async def count_query_caches_pg(self, storage) -> Dict[str, Dict[str, int]]:
        """Count query caches in PostgreSQL by mode and cache_type

        Args:
            storage: PGKVStorage instance

        Returns:
            Dictionary with counts for each mode and cache_type
        """
        from lightrag.kg.postgres_impl import namespace_to_table_name

        counts = {mode: {"query": 0, "keywords": 0} for mode in QUERY_MODES}
        table_name = namespace_to_table_name(storage.namespace)

        print("Counting PostgreSQL records...", end="", flush=True)
        start_time = time.time()

        for mode in QUERY_MODES:
            for cache_type in CACHE_TYPES:
                query = f"""
                    SELECT COUNT(*) as count
                    FROM {table_name}
                    WHERE workspace = $1
                    AND id LIKE $2
                """
                pattern = f"{mode}:{cache_type}:%"
                result = await storage.db.query(query, [storage.workspace, pattern])
                counts[mode][cache_type] = result["count"] if result else 0

        elapsed = time.time() - start_time
        if elapsed > 1:
            print(f" (took {elapsed:.1f}s)", end="")
        print()  # New line

        return counts

    async def count_query_caches_mongo(self, storage) -> Dict[str, Dict[str, int]]:
        """Count query caches in MongoDB by mode and cache_type

        Args:
            storage: MongoKVStorage instance

        Returns:
            Dictionary with counts for each mode and cache_type
        """
        counts = {mode: {"query": 0, "keywords": 0} for mode in QUERY_MODES}

        print("Counting MongoDB documents...", end="", flush=True)
        start_time = time.time()

        for mode in QUERY_MODES:
            for cache_type in CACHE_TYPES:
                pattern = f"^{mode}:{cache_type}:"
                query = {"_id": {"$regex": pattern}}
                count = await storage._data.count_documents(query)
                counts[mode][cache_type] = count

        elapsed = time.time() - start_time
        if elapsed > 1:
            print(f" (took {elapsed:.1f}s)", end="")
        print()  # New line

        return counts

    async def count_query_caches(
        self, storage, storage_name: str
    ) -> Dict[str, Dict[str, int]]:
        """Count query caches from any storage type efficiently

        Args:
            storage: Storage instance
            storage_name: Storage type name

        Returns:
            Dictionary with counts for each mode and cache_type
        """
        if storage_name == "JsonKVStorage":
            return await self.count_query_caches_json(storage)
        elif storage_name == "RedisKVStorage":
            return await self.count_query_caches_redis(storage)
        elif storage_name == "PGKVStorage":
            return await self.count_query_caches_pg(storage)
        elif storage_name == "MongoKVStorage":
            return await self.count_query_caches_mongo(storage)
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    async def delete_query_caches_json(
        self, storage, cleanup_type: str, stats: CleanupStats
    ):
        """Delete query caches from JsonKVStorage

        Args:
            storage: JsonKVStorage instance
            cleanup_type: 'all', 'query', or 'keywords'
            stats: CleanupStats object to track progress
        """
        # Collect keys to delete
        async with storage._storage_lock:
            keys_to_delete = []
            for key in storage._data.keys():
                should_delete = False
                for mode in QUERY_MODES:
                    if cleanup_type == "all":
                        if key.startswith(f"{mode}:query:") or key.startswith(
                            f"{mode}:keywords:"
                        ):
                            should_delete = True
                    elif cleanup_type == "query":
                        if key.startswith(f"{mode}:query:"):
                            should_delete = True
                    elif cleanup_type == "keywords":
                        if key.startswith(f"{mode}:keywords:"):
                            should_delete = True

                if should_delete:
                    keys_to_delete.append(key)

        # Delete in batches
        total_keys = len(keys_to_delete)
        stats.total_batches = (total_keys + self.batch_size - 1) // self.batch_size

        print("\n=== Starting Cleanup ===")
        print(
            f"💡 Processing {self.batch_size:,} records at a time from JsonKVStorage\n"
        )

        for batch_idx in range(stats.total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, total_keys)
            batch_keys = keys_to_delete[start_idx:end_idx]

            try:
                async with storage._storage_lock:
                    for key in batch_keys:
                        del storage._data[key]

                # CRITICAL: Set update flag so changes persist to disk
                # Without this, deletions remain in-memory only and are lost on exit
                await set_all_update_flags(
                    storage.namespace, workspace=storage.workspace
                )

                # Success
                stats.successful_batches += 1
                stats.successfully_deleted += len(batch_keys)

                # Calculate progress
                progress = (stats.successfully_deleted / total_keys) * 100
                bar_length = 20
                filled_length = int(
                    bar_length * stats.successfully_deleted // total_keys
                )
                bar = "█" * filled_length + "░" * (bar_length - filled_length)

                print(
                    f"Batch {batch_idx + 1}/{stats.total_batches}: {bar} "
                    f"{stats.successfully_deleted:,}/{total_keys:,} ({progress:.1f}%) ✓"
                )

            except Exception as e:
                stats.add_error(batch_idx + 1, e, len(batch_keys))
                print(
                    f"Batch {batch_idx + 1}/{stats.total_batches}: ✗ FAILED - "
                    f"{type(e).__name__}: {str(e)}"
                )

    async def delete_query_caches_redis(
        self, storage, cleanup_type: str, stats: CleanupStats
    ):
        """Delete query caches from RedisKVStorage

        Args:
            storage: RedisKVStorage instance
            cleanup_type: 'all', 'query', or 'keywords'
            stats: CleanupStats object to track progress
        """
        # Build patterns to delete
        patterns = []
        for mode in QUERY_MODES:
            if cleanup_type == "all":
                patterns.append(f"{mode}:query:*")
                patterns.append(f"{mode}:keywords:*")
            elif cleanup_type == "query":
                patterns.append(f"{mode}:query:*")
            elif cleanup_type == "keywords":
                patterns.append(f"{mode}:keywords:*")

        print("\n=== Starting Cleanup ===")
        print(f"💡 Processing Redis keys in batches of {self.batch_size:,}\n")

        batch_idx = 0
        total_deleted = 0

        async with storage._get_redis_connection() as redis:
            for pattern in patterns:
                prefixed_pattern = f"{storage.final_namespace}:{pattern}"
                cursor = 0

                while True:
                    cursor, keys = await redis.scan(
                        cursor, match=prefixed_pattern, count=self.batch_size
                    )

                    if keys:
                        batch_idx += 1
                        stats.total_batches += 1

                        try:
                            # Delete batch using pipeline
                            pipe = redis.pipeline()
                            for key in keys:
                                pipe.delete(key)
                            await pipe.execute()

                            # Success
                            stats.successful_batches += 1
                            stats.successfully_deleted += len(keys)
                            total_deleted += len(keys)

                            # Progress
                            print(
                                f"Batch {batch_idx}: Deleted {len(keys):,} keys "
                                f"(Total: {total_deleted:,}) ✓"
                            )

                        except Exception as e:
                            stats.add_error(batch_idx, e, len(keys))
                            print(
                                f"Batch {batch_idx}: ✗ FAILED - "
                                f"{type(e).__name__}: {str(e)}"
                            )

                    if cursor == 0:
                        break

                    await asyncio.sleep(0)

    async def delete_query_caches_pg(
        self, storage, cleanup_type: str, stats: CleanupStats
    ):
        """Delete query caches from PostgreSQL

        Args:
            storage: PGKVStorage instance
            cleanup_type: 'all', 'query', or 'keywords'
            stats: CleanupStats object to track progress
        """
        from lightrag.kg.postgres_impl import namespace_to_table_name

        table_name = namespace_to_table_name(storage.namespace)

        # Build WHERE conditions
        conditions = []
        for mode in QUERY_MODES:
            if cleanup_type == "all":
                conditions.append(f"id LIKE '{mode}:query:%'")
                conditions.append(f"id LIKE '{mode}:keywords:%'")
            elif cleanup_type == "query":
                conditions.append(f"id LIKE '{mode}:query:%'")
            elif cleanup_type == "keywords":
                conditions.append(f"id LIKE '{mode}:keywords:%'")

        where_clause = " OR ".join(conditions)

        print("\n=== Starting Cleanup ===")
        print("💡 Executing PostgreSQL DELETE query\n")

        try:
            query = f"""
                DELETE FROM {table_name}
                WHERE workspace = $1
                AND ({where_clause})
            """

            start_time = time.time()
            # Fix: Pass dict instead of list for execute() method
            await storage.db.execute(query, {"workspace": storage.workspace})
            elapsed = time.time() - start_time

            # PostgreSQL returns deletion count
            stats.total_batches = 1
            stats.successful_batches = 1
            stats.successfully_deleted = stats.total_to_delete

            print(f"✓ Deleted {stats.successfully_deleted:,} records in {elapsed:.2f}s")

        except Exception as e:
            stats.add_error(1, e, stats.total_to_delete)
            print(f"✗ DELETE failed: {type(e).__name__}: {str(e)}")

    async def delete_query_caches_mongo(
        self, storage, cleanup_type: str, stats: CleanupStats
    ):
        """Delete query caches from MongoDB

        Args:
            storage: MongoKVStorage instance
            cleanup_type: 'all', 'query', or 'keywords'
            stats: CleanupStats object to track progress
        """
        # Build regex patterns
        patterns = []
        for mode in QUERY_MODES:
            if cleanup_type == "all":
                patterns.append(f"^{mode}:query:")
                patterns.append(f"^{mode}:keywords:")
            elif cleanup_type == "query":
                patterns.append(f"^{mode}:query:")
            elif cleanup_type == "keywords":
                patterns.append(f"^{mode}:keywords:")

        print("\n=== Starting Cleanup ===")
        print("💡 Executing MongoDB deleteMany operations\n")

        total_deleted = 0
        for idx, pattern in enumerate(patterns, 1):
            try:
                query = {"_id": {"$regex": pattern}}
                result = await storage._data.delete_many(query)
                deleted_count = result.deleted_count

                stats.total_batches += 1
                stats.successful_batches += 1
                stats.successfully_deleted += deleted_count
                total_deleted += deleted_count

                print(
                    f"Pattern {idx}/{len(patterns)}: Deleted {deleted_count:,} records ✓"
                )

            except Exception as e:
                stats.add_error(idx, e, 0)
                print(
                    f"Pattern {idx}/{len(patterns)}: ✗ FAILED - "
                    f"{type(e).__name__}: {str(e)}"
                )

        print(f"\nTotal deleted: {total_deleted:,} records")

    async def delete_query_caches(
        self, storage, storage_name: str, cleanup_type: str, stats: CleanupStats
    ):
        """Delete query caches from any storage type

        Args:
            storage: Storage instance
            storage_name: Storage type name
            cleanup_type: 'all', 'query', or 'keywords'
            stats: CleanupStats object to track progress
        """
        if storage_name == "JsonKVStorage":
            await self.delete_query_caches_json(storage, cleanup_type, stats)
        elif storage_name == "RedisKVStorage":
            await self.delete_query_caches_redis(storage, cleanup_type, stats)
        elif storage_name == "PGKVStorage":
            await self.delete_query_caches_pg(storage, cleanup_type, stats)
        elif storage_name == "MongoKVStorage":
            await self.delete_query_caches_mongo(storage, cleanup_type, stats)
        else:
            raise ValueError(f"Unsupported storage type: {storage_name}")

    def print_header(self):
        """Print tool header"""
        print("\n" + "=" * 60)
        print("LLM Query Cache Cleanup Tool - LightRAG")
        print("=" * 60)

    def print_storage_types(self):
        """Print available storage types"""
        print("\nSupported KV Storage Types:")
        for key, value in STORAGE_TYPES.items():
            print(f"[{key}] {value}")

    def format_workspace(self, workspace: str) -> str:
        """Format workspace name with highlighting

        Args:
            workspace: Workspace name (may be empty)

        Returns:
            Formatted workspace string with ANSI color codes
        """
        if workspace:
            return f"{BOLD_CYAN}{workspace}{RESET}"
        else:
            return f"{BOLD_CYAN}(default){RESET}"

    def print_cache_statistics(self, counts: Dict[str, Dict[str, int]], title: str):
        """Print cache statistics in a formatted table

        Args:
            counts: Dictionary with counts for each mode and cache_type
            title: Title for the statistics display
        """
        print(f"\n{title}")
        print("┌" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┐")
        print(f"│ {'Mode':<10} │ {'Query':>10} │ {'Keywords':>10} │ {'Total':>10} │")
        print("├" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        total_query = 0
        total_keywords = 0

        for mode in QUERY_MODES:
            query_count = counts[mode]["query"]
            keywords_count = counts[mode]["keywords"]
            mode_total = query_count + keywords_count

            total_query += query_count
            total_keywords += keywords_count

            print(
                f"│ {mode:<10} │ {query_count:>10,} │ {keywords_count:>10,} │ {mode_total:>10,} │"
            )

        print("├" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")
        grand_total = total_query + total_keywords
        print(
            f"│ {'Total':<10} │ {total_query:>10,} │ {total_keywords:>10,} │ {grand_total:>10,} │"
        )
        print("└" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")

    def calculate_total_to_delete(
        self, counts: Dict[str, Dict[str, int]], cleanup_type: str
    ) -> int:
        """Calculate total number of records to delete

        Args:
            counts: Dictionary with counts for each mode and cache_type
            cleanup_type: 'all', 'query', or 'keywords'

        Returns:
            Total number of records to delete
        """
        total = 0
        for mode in QUERY_MODES:
            if cleanup_type == "all":
                total += counts[mode]["query"] + counts[mode]["keywords"]
            elif cleanup_type == "query":
                total += counts[mode]["query"]
            elif cleanup_type == "keywords":
                total += counts[mode]["keywords"]
        return total

    def print_cleanup_report(self, stats: CleanupStats):
        """Print comprehensive cleanup report

        Args:
            stats: CleanupStats object with cleanup results
        """
        print("\n" + "=" * 60)
        print("Cleanup Complete - Final Report")
        print("=" * 60)

        # Overall statistics
        print("\n📊 Statistics:")
        print(f"  Total records to delete:  {stats.total_to_delete:,}")
        print(f"  Total batches:            {stats.total_batches:,}")
        print(f"  Successful batches:       {stats.successful_batches:,}")
        print(f"  Failed batches:           {stats.failed_batches:,}")
        print(f"  Successfully deleted:     {stats.successfully_deleted:,}")
        print(f"  Failed to delete:         {stats.failed_to_delete:,}")

        # Success rate
        success_rate = (
            (stats.successfully_deleted / stats.total_to_delete * 100)
            if stats.total_to_delete > 0
            else 0
        )
        print(f"  Success rate:             {success_rate:.2f}%")

        # Before/After comparison
        print("\n📈 Before/After Comparison:")
        total_before = sum(
            counts["query"] + counts["keywords"]
            for counts in stats.counts_before.values()
        )
        total_after = sum(
            counts["query"] + counts["keywords"]
            for counts in stats.counts_after.values()
        )
        print(f"  Total caches before:      {total_before:,}")
        print(f"  Total caches after:       {total_after:,}")
        print(f"  Net reduction:            {total_before - total_after:,}")

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
            print(f"{BOLD_RED}⚠️  WARNING: Cleanup completed with errors!{RESET}")
            print("   Please review the error details above.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print(f"{BOLD_GREEN}✓ SUCCESS: All records cleaned up successfully!{RESET}")
            print("=" * 60)

    async def setup_storage(self) -> tuple:
        """Setup and initialize storage

        Returns:
            Tuple of (storage_instance, storage_name, workspace)
            Returns (None, None, None) if user chooses to exit
        """
        print("\n=== Storage Setup ===")
        self.print_storage_types()

        # Custom input handling with exit support
        while True:
            choice = input(
                "\nSelect storage type (1-4) (Press Enter to exit): "
            ).strip()

            # Check for exit
            if choice == "" or choice == "0":
                print("\n✓ Cleanup cancelled by user")
                return None, None, None

            # Check if choice is valid
            if choice in STORAGE_TYPES:
                break

            print(
                f"✗ Invalid choice. Please enter one of: {', '.join(STORAGE_TYPES.keys())}"
            )

        storage_name = STORAGE_TYPES[choice]

        # Special warning for JsonKVStorage about concurrent access
        if storage_name == "JsonKVStorage":
            print("\n" + "=" * 60)
            print(f"{BOLD_RED}⚠️  IMPORTANT WARNING - JsonKVStorage Concurrency{RESET}")
            print("=" * 60)
            print("\nJsonKVStorage is an in-memory database that does NOT support")
            print("concurrent access to the same file by multiple programs.")
            print("\nBefore proceeding, please ensure that:")
            print("  • LightRAG Server is completely shut down")
            print("  • No other programs are accessing the storage files")
            print("\n" + "=" * 60)

            confirm = (
                input("\nHas LightRAG Server been shut down? (yes/no): ")
                .strip()
                .lower()
            )
            if confirm != "yes":
                print(
                    "\n✓ Operation cancelled - Please shut down LightRAG Server first"
                )
                return None, None, None

            print("✓ Proceeding with JsonKVStorage cleanup...")

        # Check configuration (warnings only, doesn't block)
        print("\nChecking configuration...")
        self.check_env_vars(storage_name)

        # Get workspace
        workspace = self.get_workspace_for_storage(storage_name)

        # Initialize storage (real validation point)
        print("\nInitializing storage...")
        try:
            storage = await self.initialize_storage(storage_name, workspace)
            print(f"- Storage Type: {storage_name}")
            print(f"- Workspace: {workspace if workspace else '(default)'}")
            print("- Connection Status: ✓ Success")

        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            print(f"\nFor {storage_name}, you can configure using:")
            print("  1. Environment variables (highest priority)")

            # Show specific environment variable requirements
            if storage_name in STORAGE_ENV_REQUIREMENTS:
                for var in STORAGE_ENV_REQUIREMENTS[storage_name]:
                    print(f"     - {var}")

            print("  2. config.ini file (medium priority)")
            if storage_name == "RedisKVStorage":
                print("     [redis]")
                print("     uri = redis://localhost:6379")
            elif storage_name == "PGKVStorage":
                print("     [postgres]")
                print("     host = localhost")
                print("     port = 5432")
                print("     user = postgres")
                print("     password = yourpassword")
                print("     database = lightrag")
            elif storage_name == "MongoKVStorage":
                print("     [mongodb]")
                print("     uri = mongodb://root:root@localhost:27017/")
                print("     database = LightRAG")

            return None, None, None

        return storage, storage_name, workspace

    async def run(self):
        """Run the cleanup tool"""
        try:
            # Initialize shared storage (REQUIRED for storage classes to work)
            from lightrag.kg.shared_storage import initialize_share_data

            initialize_share_data(workers=1)

            # Print header
            self.print_header()

            # Setup storage
            self.storage, storage_name, self.workspace = await self.setup_storage()

            # Check if user cancelled
            if self.storage is None:
                return

            # Count query caches
            print("\nCounting query cache records...")
            try:
                counts = await self.count_query_caches(self.storage, storage_name)
            except Exception as e:
                print(f"✗ Counting failed: {e}")
                await self.storage.finalize()
                return

            # Initialize stats
            stats = CleanupStats()
            stats.initialize_counts()
            stats.counts_before = counts

            # Print statistics
            self.print_cache_statistics(
                counts, "📊 Query Cache Statistics (Before Cleanup):"
            )

            # Calculate total
            total_caches = sum(
                counts[mode]["query"] + counts[mode]["keywords"] for mode in QUERY_MODES
            )

            if total_caches == 0:
                print("\n⚠️  No query caches found in storage")
                await self.storage.finalize()
                return

            # Select cleanup type
            print("\n=== Cleanup Options ===")
            print("[1] Delete all query caches (both query and keywords)")
            print("[2] Delete query caches only (keep keywords)")
            print("[3] Delete keywords caches only (keep query)")
            print("[0] Cancel")

            while True:
                choice = input("\nSelect cleanup option (0-3): ").strip()

                if choice == "0" or choice == "":
                    print("\n✓ Cleanup cancelled")
                    await self.storage.finalize()
                    return
                elif choice == "1":
                    cleanup_type = "all"
                elif choice == "2":
                    cleanup_type = "query"
                elif choice == "3":
                    cleanup_type = "keywords"
                else:
                    print("✗ Invalid choice. Please enter 0, 1, 2, or 3")
                    continue

                # Calculate total to delete for the selected type
                stats.total_to_delete = self.calculate_total_to_delete(
                    counts, cleanup_type
                )

                # Check if there are any records to delete
                if stats.total_to_delete == 0:
                    if cleanup_type == "all":
                        print(f"\n{BOLD_RED}⚠️  No query caches found to delete!{RESET}")
                    elif cleanup_type == "query":
                        print(
                            f"\n{BOLD_RED}⚠️  No query caches found to delete! (Only keywords exist){RESET}"
                        )
                    elif cleanup_type == "keywords":
                        print(
                            f"\n{BOLD_RED}⚠️  No keywords caches found to delete! (Only query caches exist){RESET}"
                        )
                    print("   Please select a different cleanup option.\n")
                    continue

                # Valid selection with records to delete
                break

            # Confirm deletion
            print("\n" + "=" * 60)
            print("Cleanup Confirmation")
            print("=" * 60)
            print(
                f"Storage: {BOLD_CYAN}{storage_name}{RESET} "
                f"(workspace: {self.format_workspace(self.workspace)})"
            )
            print(f"Cleanup Type: {BOLD_CYAN}{cleanup_type}{RESET}")
            print(
                f"Records to Delete: {BOLD_RED}{stats.total_to_delete:,}{RESET} / {total_caches:,}"
            )

            if cleanup_type == "all":
                print(
                    f"\n{BOLD_RED}⚠️  WARNING: This will delete ALL query caches across all modes!{RESET}"
                )
            elif cleanup_type == "query":
                print("\n⚠️  This will delete query caches only (keywords will be kept)")
            elif cleanup_type == "keywords":
                print("\n⚠️  This will delete keywords caches only (query will be kept)")

            confirm = input("\nContinue with deletion? (y/n): ").strip().lower()
            if confirm != "y":
                print("\n✓ Cleanup cancelled")
                await self.storage.finalize()
                return

            # Perform deletion
            await self.delete_query_caches(
                self.storage, storage_name, cleanup_type, stats
            )

            # Persist changes
            print("\nPersisting changes to storage...")
            try:
                await self.storage.index_done_callback()
                print("✓ Changes persisted successfully")
            except Exception as e:
                print(f"✗ Persist failed: {e}")
                stats.add_error(0, e, 0)

            # Count again to verify
            print("\nVerifying cleanup results...")
            try:
                stats.counts_after = await self.count_query_caches(
                    self.storage, storage_name
                )
            except Exception as e:
                print(f"⚠️  Verification failed: {e}")
                # Use zero counts if verification fails
                stats.counts_after = {
                    mode: {"query": 0, "keywords": 0} for mode in QUERY_MODES
                }

            # Print final report
            self.print_cleanup_report(stats)

            # Print after statistics
            self.print_cache_statistics(
                stats.counts_after, "\n📊 Query Cache Statistics (After Cleanup):"
            )

            # Cleanup
            await self.storage.finalize()

        except KeyboardInterrupt:
            print("\n\n✗ Cleanup interrupted by user")
        except Exception as e:
            print(f"\n✗ Cleanup failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Ensure cleanup
            if self.storage:
                try:
                    await self.storage.finalize()
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
    tool = CleanupTool()
    await tool.run()


if __name__ == "__main__":
    asyncio.run(main())
