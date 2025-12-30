#!/usr/bin/env python3
"""
Qdrant Legacy Data Preparation Tool for LightRAG

This tool copies data from new collections to legacy collections for testing
the data migration logic in setup_collection function.

New Collections (with workspace_id):
    - lightrag_vdb_chunks
    - lightrag_vdb_entities
    - lightrag_vdb_relationships

Legacy Collections (without workspace_id, dynamically named as {workspace}_{suffix}):
    - {workspace}_chunks (e.g., space1_chunks)
    - {workspace}_entities (e.g., space1_entities)
    - {workspace}_relationships (e.g., space1_relationships)

The tool:
    1. Filters source data by workspace_id
    2. Verifies workspace data exists before creating legacy collections
    3. Removes workspace_id field to simulate legacy data format
    4. Copies only the specified workspace's data to legacy collections

Usage:
    python -m lightrag.tools.prepare_qdrant_legacy_data
    # or
    python lightrag/tools/prepare_qdrant_legacy_data.py

    # Specify custom workspace
    python -m lightrag.tools.prepare_qdrant_legacy_data --workspace space1

    # Process specific collection types only
    python -m lightrag.tools.prepare_qdrant_legacy_data --types chunks,entities

    # Dry run (preview only, no actual changes)
    python -m lightrag.tools.prepare_qdrant_legacy_data --dry-run
"""

import argparse
import asyncio
import configparser
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pipmaster as pm
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models  # type: ignore

# Add project root to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Ensure qdrant-client is installed
if not pm.is_installed("qdrant-client"):
    pm.install("qdrant-client")

# Collection namespace mapping: new collection pattern -> legacy suffix
# Legacy collection will be named as: {workspace}_{suffix}
COLLECTION_NAMESPACES = {
    "chunks": {
        "new": "lightrag_vdb_chunks",
        "suffix": "chunks",
    },
    "entities": {
        "new": "lightrag_vdb_entities",
        "suffix": "entities",
    },
    "relationships": {
        "new": "lightrag_vdb_relationships",
        "suffix": "relationships",
    },
}

# Default batch size for copy operations
DEFAULT_BATCH_SIZE = 500

# Field to remove from legacy data
WORKSPACE_ID_FIELD = "workspace_id"

# ANSI color codes for terminal output
BOLD_CYAN = "\033[1;36m"
BOLD_GREEN = "\033[1;32m"
BOLD_YELLOW = "\033[1;33m"
BOLD_RED = "\033[1;31m"
RESET = "\033[0m"


@dataclass
class CopyStats:
    """Copy operation statistics"""

    collection_type: str
    source_collection: str
    target_collection: str
    total_records: int = 0
    copied_records: int = 0
    failed_records: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_time: float = 0.0

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
        self.failed_records += batch_size


class QdrantLegacyDataPreparationTool:
    """Tool for preparing legacy data in Qdrant for migration testing"""

    def __init__(
        self,
        workspace: str = "space1",
        batch_size: int = DEFAULT_BATCH_SIZE,
        dry_run: bool = False,
        clear_target: bool = False,
    ):
        """
        Initialize the tool.

        Args:
            workspace: Workspace to use for filtering new collection data
            batch_size: Number of records to process per batch
            dry_run: If True, only preview operations without making changes
            clear_target: If True, delete target collection before copying data
        """
        self.workspace = workspace
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.clear_target = clear_target
        self._client: Optional[QdrantClient] = None

    def _get_client(self) -> QdrantClient:
        """Get or create QdrantClient instance"""
        if self._client is None:
            config = configparser.ConfigParser()
            config.read("config.ini", "utf-8")

            self._client = QdrantClient(
                url=os.environ.get(
                    "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
                ),
                api_key=os.environ.get(
                    "QDRANT_API_KEY",
                    config.get("qdrant", "apikey", fallback=None),
                ),
            )
        return self._client

    def print_header(self):
        """Print tool header"""
        print("\n" + "=" * 60)
        print("Qdrant Legacy Data Preparation Tool - LightRAG")
        print("=" * 60)
        if self.dry_run:
            print(f"{BOLD_YELLOW}⚠️  DRY RUN MODE - No changes will be made{RESET}")
        if self.clear_target:
            print(
                f"{BOLD_RED}⚠️  CLEAR TARGET MODE - Target collections will be deleted first{RESET}"
            )
        print(f"Workspace: {BOLD_CYAN}{self.workspace}{RESET}")
        print(f"Batch Size: {self.batch_size}")
        print("=" * 60)

    def check_connection(self) -> bool:
        """Check Qdrant connection"""
        try:
            client = self._get_client()
            # Try to list collections to verify connection
            client.get_collections()
            print(f"{BOLD_GREEN}✓{RESET} Qdrant connection successful")
            return True
        except Exception as e:
            print(f"{BOLD_RED}✗{RESET} Qdrant connection failed: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get collection information.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection info (vector_size, count) or None if not exists
        """
        client = self._get_client()

        if not client.collection_exists(collection_name):
            return None

        info = client.get_collection(collection_name)
        count = client.count(collection_name=collection_name, exact=True).count

        # Handle both object and dict formats for vectors config
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # Named vectors format or dict format
            if vectors_config:
                first_key = next(iter(vectors_config.keys()), None)
                if first_key and hasattr(vectors_config[first_key], "size"):
                    vector_size = vectors_config[first_key].size
                    distance = vectors_config[first_key].distance
                else:
                    # Try to get from dict values
                    first_val = next(iter(vectors_config.values()), {})
                    vector_size = (
                        first_val.get("size")
                        if isinstance(first_val, dict)
                        else getattr(first_val, "size", None)
                    )
                    distance = (
                        first_val.get("distance")
                        if isinstance(first_val, dict)
                        else getattr(first_val, "distance", None)
                    )
            else:
                vector_size = None
                distance = None
        else:
            # Standard single vector format
            vector_size = vectors_config.size
            distance = vectors_config.distance

        return {
            "name": collection_name,
            "vector_size": vector_size,
            "count": count,
            "distance": distance,
        }

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection if it exists.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deleted or doesn't exist
        """
        client = self._get_client()

        if not client.collection_exists(collection_name):
            return True

        if self.dry_run:
            target_info = self.get_collection_info(collection_name)
            count = target_info["count"] if target_info else 0
            print(
                f"  {BOLD_YELLOW}[DRY RUN]{RESET} Would delete collection '{collection_name}' ({count:,} records)"
            )
            return True

        try:
            target_info = self.get_collection_info(collection_name)
            count = target_info["count"] if target_info else 0
            client.delete_collection(collection_name=collection_name)
            print(
                f"  {BOLD_RED}✗{RESET} Deleted collection '{collection_name}' ({count:,} records)"
            )
            return True
        except Exception as e:
            print(f"  {BOLD_RED}✗{RESET} Failed to delete collection: {e}")
            return False

    def create_legacy_collection(
        self, collection_name: str, vector_size: int, distance: models.Distance
    ) -> bool:
        """
        Create legacy collection if it doesn't exist.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of vectors
            distance: Distance metric

        Returns:
            True if created or already exists
        """
        client = self._get_client()

        if client.collection_exists(collection_name):
            print(f"  Collection '{collection_name}' already exists")
            return True

        if self.dry_run:
            print(
                f"  {BOLD_YELLOW}[DRY RUN]{RESET} Would create collection '{collection_name}' with {vector_size}d vectors"
            )
            return True

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
            )
            print(
                f"  {BOLD_GREEN}✓{RESET} Created collection '{collection_name}' with {vector_size}d vectors"
            )
            return True
        except Exception as e:
            print(f"  {BOLD_RED}✗{RESET} Failed to create collection: {e}")
            return False

    def _get_workspace_filter(self) -> models.Filter:
        """Create workspace filter for Qdrant queries"""
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=WORKSPACE_ID_FIELD,
                    match=models.MatchValue(value=self.workspace),
                )
            ]
        )

    def get_workspace_count(self, collection_name: str) -> int:
        """
        Get count of records for the current workspace in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Count of records for the workspace
        """
        client = self._get_client()
        return client.count(
            collection_name=collection_name,
            count_filter=self._get_workspace_filter(),
            exact=True,
        ).count

    def copy_collection_data(
        self,
        source_collection: str,
        target_collection: str,
        collection_type: str,
        workspace_count: int,
    ) -> CopyStats:
        """
        Copy data from source to target collection.

        This filters by workspace_id and removes it from payload to simulate legacy data format.

        Args:
            source_collection: Source collection name
            target_collection: Target collection name
            collection_type: Type of collection (chunks, entities, relationships)
            workspace_count: Pre-computed count of workspace records

        Returns:
            CopyStats with operation results
        """
        client = self._get_client()
        stats = CopyStats(
            collection_type=collection_type,
            source_collection=source_collection,
            target_collection=target_collection,
        )

        start_time = time.time()
        stats.total_records = workspace_count

        if workspace_count == 0:
            print(f"  No records for workspace '{self.workspace}', skipping")
            stats.elapsed_time = time.time() - start_time
            return stats

        print(f"  Workspace records: {workspace_count:,}")

        if self.dry_run:
            print(
                f"  {BOLD_YELLOW}[DRY RUN]{RESET} Would copy {workspace_count:,} records to '{target_collection}'"
            )
            stats.copied_records = workspace_count
            stats.elapsed_time = time.time() - start_time
            return stats

        # Batch copy using scroll with workspace filter
        workspace_filter = self._get_workspace_filter()
        offset = None
        batch_idx = 0

        while True:
            # Scroll source collection with workspace filter
            result = client.scroll(
                collection_name=source_collection,
                scroll_filter=workspace_filter,
                limit=self.batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
            points, next_offset = result

            if not points:
                break

            batch_idx += 1

            # Transform points: remove workspace_id from payload
            new_points = []
            for point in points:
                new_payload = dict(point.payload or {})
                # Remove workspace_id to simulate legacy format
                new_payload.pop(WORKSPACE_ID_FIELD, None)

                # Use original id from payload if available, otherwise use point.id
                original_id = new_payload.get("id")
                if original_id:
                    # Generate a simple deterministic id for legacy format
                    # Use original id directly (legacy format didn't have workspace prefix)
                    import hashlib
                    import uuid

                    hashed = hashlib.sha256(original_id.encode("utf-8")).digest()
                    point_id = uuid.UUID(bytes=hashed[:16], version=4).hex
                else:
                    point_id = str(point.id)

                new_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=point.vector,
                        payload=new_payload,
                    )
                )

            try:
                # Upsert to target collection
                client.upsert(
                    collection_name=target_collection, points=new_points, wait=True
                )
                stats.copied_records += len(new_points)

                # Progress bar
                progress = (stats.copied_records / workspace_count) * 100
                bar_length = 30
                filled = int(bar_length * stats.copied_records // workspace_count)
                bar = "█" * filled + "░" * (bar_length - filled)

                print(
                    f"\r  Copying: {bar} {stats.copied_records:,}/{workspace_count:,} ({progress:.1f}%) ",
                    end="",
                    flush=True,
                )

            except Exception as e:
                stats.add_error(batch_idx, e, len(new_points))
                print(
                    f"\n  {BOLD_RED}✗{RESET} Batch {batch_idx} failed: {type(e).__name__}: {e}"
                )

            if next_offset is None:
                break
            offset = next_offset

        print()  # New line after progress bar
        stats.elapsed_time = time.time() - start_time

        return stats

    def process_collection_type(self, collection_type: str) -> Optional[CopyStats]:
        """
        Process a single collection type.

        Args:
            collection_type: Type of collection (chunks, entities, relationships)

        Returns:
            CopyStats or None if error
        """
        namespace_config = COLLECTION_NAMESPACES.get(collection_type)
        if not namespace_config:
            print(f"{BOLD_RED}✗{RESET} Unknown collection type: {collection_type}")
            return None

        source = namespace_config["new"]
        # Generate legacy collection name dynamically: {workspace}_{suffix}
        target = f"{self.workspace}_{namespace_config['suffix']}"

        print(f"\n{'=' * 50}")
        print(f"Processing: {BOLD_CYAN}{collection_type}{RESET}")
        print(f"{'=' * 50}")
        print(f"  Source: {source}")
        print(f"  Target: {target}")

        # Check source collection
        source_info = self.get_collection_info(source)
        if source_info is None:
            print(
                f"  {BOLD_YELLOW}⚠{RESET} Source collection '{source}' does not exist, skipping"
            )
            return None

        print(f"  Source vector dimension: {source_info['vector_size']}d")
        print(f"  Source distance metric: {source_info['distance']}")
        print(f"  Source total records: {source_info['count']:,}")

        # Check workspace data exists BEFORE creating legacy collection
        workspace_count = self.get_workspace_count(source)
        print(f"  Workspace '{self.workspace}' records: {workspace_count:,}")

        if workspace_count == 0:
            print(
                f"  {BOLD_YELLOW}⚠{RESET} No data found for workspace '{self.workspace}' in '{source}', skipping"
            )
            return None

        # Clear target collection if requested
        if self.clear_target:
            if not self.delete_collection(target):
                return None

        # Create target collection only after confirming workspace data exists
        if not self.create_legacy_collection(
            target, source_info["vector_size"], source_info["distance"]
        ):
            return None

        # Copy data with workspace filter
        stats = self.copy_collection_data(
            source, target, collection_type, workspace_count
        )

        # Print result
        if stats.failed_records == 0:
            print(
                f"  {BOLD_GREEN}✓{RESET} Copied {stats.copied_records:,} records in {stats.elapsed_time:.2f}s"
            )
        else:
            print(
                f"  {BOLD_YELLOW}⚠{RESET} Copied {stats.copied_records:,} records, "
                f"{BOLD_RED}{stats.failed_records:,} failed{RESET} in {stats.elapsed_time:.2f}s"
            )

        return stats

    def print_summary(self, all_stats: List[CopyStats]):
        """Print summary of all operations"""
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        total_copied = sum(s.copied_records for s in all_stats)
        total_failed = sum(s.failed_records for s in all_stats)
        total_time = sum(s.elapsed_time for s in all_stats)

        for stats in all_stats:
            status = (
                f"{BOLD_GREEN}✓{RESET}"
                if stats.failed_records == 0
                else f"{BOLD_YELLOW}⚠{RESET}"
            )
            print(
                f"  {status} {stats.collection_type}: {stats.copied_records:,}/{stats.total_records:,} "
                f"({stats.source_collection} → {stats.target_collection})"
            )

        print("-" * 60)
        print(f"  Total records copied: {BOLD_CYAN}{total_copied:,}{RESET}")
        if total_failed > 0:
            print(f"  Total records failed: {BOLD_RED}{total_failed:,}{RESET}")
        print(f"  Total time: {total_time:.2f}s")

        if self.dry_run:
            print(f"\n{BOLD_YELLOW}⚠️  DRY RUN - No actual changes were made{RESET}")

        # Print error details if any
        all_errors = []
        for stats in all_stats:
            all_errors.extend(stats.errors)

        if all_errors:
            print(f"\n{BOLD_RED}Errors ({len(all_errors)}){RESET}")
            for i, error in enumerate(all_errors[:5], 1):
                print(
                    f"  {i}. Batch {error['batch']}: {error['error_type']}: {error['error_msg']}"
                )
            if len(all_errors) > 5:
                print(f"  ... and {len(all_errors) - 5} more errors")

        print("=" * 60)

    async def run(self, collection_types: Optional[List[str]] = None):
        """
        Run the data preparation tool.

        Args:
            collection_types: List of collection types to process (default: all)
        """
        self.print_header()

        # Check connection
        if not self.check_connection():
            return

        # Determine which collection types to process
        if collection_types:
            types_to_process = [t.strip() for t in collection_types]
            invalid_types = [
                t for t in types_to_process if t not in COLLECTION_NAMESPACES
            ]
            if invalid_types:
                print(
                    f"{BOLD_RED}✗{RESET} Invalid collection types: {', '.join(invalid_types)}"
                )
                print(f"  Valid types: {', '.join(COLLECTION_NAMESPACES.keys())}")
                return
        else:
            types_to_process = list(COLLECTION_NAMESPACES.keys())

        print(f"\nCollection types to process: {', '.join(types_to_process)}")

        # Process each collection type
        all_stats = []
        for ctype in types_to_process:
            stats = self.process_collection_type(ctype)
            if stats:
                all_stats.append(stats)

        # Print summary
        if all_stats:
            self.print_summary(all_stats)
        else:
            print(f"\n{BOLD_YELLOW}⚠{RESET} No collections were processed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prepare legacy data in Qdrant for migration testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m lightrag.tools.prepare_qdrant_legacy_data
    python -m lightrag.tools.prepare_qdrant_legacy_data --workspace space1
    python -m lightrag.tools.prepare_qdrant_legacy_data --types chunks,entities
    python -m lightrag.tools.prepare_qdrant_legacy_data --dry-run
        """,
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="space1",
        help="Workspace name (default: space1)",
    )

    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated list of collection types (chunks, entities, relationships)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for copy operations (default: {DEFAULT_BATCH_SIZE})",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview operations without making changes",
    )

    parser.add_argument(
        "--clear-target",
        action="store_true",
        help="Delete target collections before copying (for clean test environment)",
    )

    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()

    collection_types = None
    if args.types:
        collection_types = [t.strip() for t in args.types.split(",")]

    tool = QdrantLegacyDataPreparationTool(
        workspace=args.workspace,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        clear_target=args.clear_target,
    )

    await tool.run(collection_types=collection_types)


if __name__ == "__main__":
    asyncio.run(main())
