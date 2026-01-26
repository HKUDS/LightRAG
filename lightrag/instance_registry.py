"""
Multi-instance coordination via PostgreSQL for graceful autoscaling.

This module provides a registry for LightRAG instances that allows:
- Instance registration and heartbeat
- Coordinated drain requests across instances
- Scale-down coordination without direct instance targeting

Architecture:
    PostgreSQL (shared)
    ┌─────────────────────────────────────────────────────────┐
    │ lightrag_instances: instance_id, drain_requested, etc.  │
    └─────────────────────────────────────────────────────────┘
             ▲                    ▲                    ▲
             │ poll               │ poll               │ poll
        ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
        │ Inst A  │          │ Inst B  │          │ Inst C  │
        └─────────┘          └─────────┘          └─────────┘
"""

import asyncio
import os
import socket
from typing import Any, Optional

from .utils import logger

# Table DDL for instance registry
INSTANCE_REGISTRY_TABLE = {
    "LIGHTRAG_INSTANCES": {
        "ddl": """CREATE TABLE IF NOT EXISTS LIGHTRAG_INSTANCES (
            instance_id VARCHAR(64) PRIMARY KEY,
            hostname VARCHAR(255),
            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            drain_requested BOOLEAN DEFAULT FALSE,
            drain_reason VARCHAR(255),
            processing_count INT DEFAULT 0,
            pipeline_busy BOOLEAN DEFAULT FALSE
        )""",
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_instances_heartbeat ON LIGHTRAG_INSTANCES(last_heartbeat)",
            "CREATE INDEX IF NOT EXISTS idx_instances_drain ON LIGHTRAG_INSTANCES(drain_requested)",
        ],
    }
}


def generate_instance_id() -> str:
    """Generate a unique instance ID.

    Priority:
    1. RENDER_INSTANCE_ID (if running on Render)
    2. hostname-pid as fallback
    """
    render_id = os.environ.get("RENDER_INSTANCE_ID")
    if render_id:
        return render_id

    hostname = socket.gethostname()
    pid = os.getpid()
    return f"{hostname}-{pid}"


class InstanceRegistry:
    """
    Registry for coordinating multiple LightRAG instances via PostgreSQL.

    Each instance registers itself at startup, sends periodic heartbeats,
    and polls for drain requests from the autoscaler.
    """

    def __init__(
        self,
        pool,  # asyncpg connection pool
        instance_id: Optional[str] = None,
        heartbeat_interval: int = 30,
        drain_poll_interval: int = 5,
        dead_instance_timeout: int = 300,  # 5 minutes
    ):
        """
        Initialize the instance registry.

        Args:
            pool: asyncpg connection pool
            instance_id: Unique ID for this instance (auto-generated if None)
            heartbeat_interval: Seconds between heartbeat updates
            drain_poll_interval: Seconds between drain status polls
            dead_instance_timeout: Seconds after which an instance is considered dead
        """
        self.pool = pool
        self.instance_id = instance_id or generate_instance_id()
        self.hostname = socket.gethostname()
        self.heartbeat_interval = heartbeat_interval
        self.drain_poll_interval = drain_poll_interval
        self.dead_instance_timeout = dead_instance_timeout

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._drain_poll_task: Optional[asyncio.Task] = None
        self._running = False

        # Callback for when drain is requested
        self._on_drain_requested: Optional[callable] = None

    async def initialize(self) -> None:
        """Initialize the registry table if it doesn't exist."""
        async with self.pool.acquire() as conn:
            # Create table
            await conn.execute(INSTANCE_REGISTRY_TABLE["LIGHTRAG_INSTANCES"]["ddl"])

            # Create indexes
            for index_sql in INSTANCE_REGISTRY_TABLE["LIGHTRAG_INSTANCES"]["indexes"]:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation note: {e}")

        logger.info(f"Instance registry initialized for instance: {self.instance_id}")

    async def register(self) -> None:
        """Register this instance in the database."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO LIGHTRAG_INSTANCES (instance_id, hostname, started_at, last_heartbeat)
                VALUES ($1, $2, NOW(), NOW())
                ON CONFLICT (instance_id) DO UPDATE SET
                    hostname = $2,
                    started_at = NOW(),
                    last_heartbeat = NOW(),
                    drain_requested = FALSE,
                    drain_reason = NULL
                """,
                self.instance_id,
                self.hostname,
            )

        logger.info(f"Instance registered: {self.instance_id} ({self.hostname})")

    async def unregister(self) -> None:
        """Unregister this instance from the database."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM LIGHTRAG_INSTANCES WHERE instance_id = $1",
                self.instance_id,
            )

        logger.info(f"Instance unregistered: {self.instance_id}")

    async def heartbeat(
        self,
        processing_count: int = 0,
        pipeline_busy: bool = False,
    ) -> None:
        """Send a heartbeat update."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE LIGHTRAG_INSTANCES
                SET last_heartbeat = NOW(),
                    processing_count = $2,
                    pipeline_busy = $3
                WHERE instance_id = $1
                """,
                self.instance_id,
                processing_count,
                pipeline_busy,
            )

    async def check_drain_requested(self) -> tuple[bool, Optional[str]]:
        """Check if drain has been requested for this instance.

        Returns:
            Tuple of (drain_requested, drain_reason)
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT drain_requested, drain_reason
                FROM LIGHTRAG_INSTANCES
                WHERE instance_id = $1
                """,
                self.instance_id,
            )

            if row:
                return row["drain_requested"], row["drain_reason"]
            return False, None

    async def request_drain(
        self,
        count: int = 1,
        reason: str = "scale-down",
    ) -> list[str]:
        """
        Request drain for N instances (excluding the oldest/primary).

        Uses FOR UPDATE SKIP LOCKED to avoid race conditions when
        multiple callers request drain simultaneously.

        Args:
            count: Number of instances to drain
            reason: Reason for drain request

        Returns:
            List of instance IDs marked for drain
        """
        async with self.pool.acquire() as conn:
            # Select N instances to drain (newest first, skip the oldest = primary)
            # Use FOR UPDATE SKIP LOCKED to prevent race conditions
            result = await conn.fetch(
                """
                WITH candidates AS (
                    SELECT instance_id
                    FROM LIGHTRAG_INSTANCES
                    WHERE NOT drain_requested
                        AND last_heartbeat > NOW() - INTERVAL '5 minutes'
                    ORDER BY started_at DESC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE LIGHTRAG_INSTANCES
                SET drain_requested = TRUE, drain_reason = $2
                WHERE instance_id IN (SELECT instance_id FROM candidates)
                RETURNING instance_id
                """,
                count,
                reason,
            )

            drained_ids = [row["instance_id"] for row in result]

            if drained_ids:
                logger.info(f"Drain requested for {len(drained_ids)} instances: {drained_ids}")

            return drained_ids

    async def cancel_drain(self) -> int:
        """
        Cancel all drain requests.

        Returns:
            Number of instances that had drain cancelled
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE LIGHTRAG_INSTANCES
                SET drain_requested = FALSE, drain_reason = NULL
                WHERE drain_requested = TRUE
                """
            )

            # Parse the result string like "UPDATE 3"
            count = int(result.split()[-1]) if result else 0

            if count > 0:
                logger.info(f"Drain cancelled for {count} instances")

            return count

    async def get_all_instances(self) -> list[dict[str, Any]]:
        """Get status of all registered instances."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    instance_id,
                    hostname,
                    started_at,
                    last_heartbeat,
                    drain_requested,
                    drain_reason,
                    processing_count,
                    pipeline_busy,
                    (last_heartbeat > NOW() - INTERVAL '5 minutes') AS alive
                FROM LIGHTRAG_INSTANCES
                ORDER BY started_at ASC
                """
            )

            return [
                {
                    "instance_id": row["instance_id"],
                    "hostname": row["hostname"],
                    "started_at": row["started_at"].isoformat() if row["started_at"] else None,
                    "last_heartbeat": row["last_heartbeat"].isoformat() if row["last_heartbeat"] else None,
                    "drain_requested": row["drain_requested"],
                    "drain_reason": row["drain_reason"],
                    "processing_count": row["processing_count"],
                    "pipeline_busy": row["pipeline_busy"],
                    "alive": row["alive"],
                }
                for row in rows
            ]

    async def get_drain_status(self) -> dict[str, Any]:
        """Get aggregated drain status for autoscaling decisions."""
        instances = await self.get_all_instances()

        alive_instances = [i for i in instances if i["alive"]]
        draining_instances = [i for i in alive_instances if i["drain_requested"]]
        active_instances = [i for i in alive_instances if not i["drain_requested"]]

        # Safe to scale down when all draining instances are idle
        safe_to_scale_down = all(
            not i["pipeline_busy"] and i["processing_count"] == 0
            for i in draining_instances
        ) if draining_instances else False

        return {
            "total_instances": len(instances),
            "alive_instances": len(alive_instances),
            "active_instances": len(active_instances),
            "draining_instances": len(draining_instances),
            "safe_to_scale_down": safe_to_scale_down,
            "instances": instances,
        }

    async def cleanup_dead_instances(self) -> int:
        """Remove instances that haven't sent a heartbeat recently.

        Returns:
            Number of instances removed
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM LIGHTRAG_INSTANCES
                WHERE last_heartbeat < NOW() - INTERVAL '{self.dead_instance_timeout} seconds'
                """
            )

            # Parse the result string like "DELETE 2"
            count = int(result.split()[-1]) if result else 0

            if count > 0:
                logger.info(f"Cleaned up {count} dead instances")

            return count

    def set_drain_callback(self, callback: callable) -> None:
        """Set callback to be called when drain is requested.

        The callback should accept (drain_requested: bool, reason: str).
        """
        self._on_drain_requested = callback

    async def start_background_tasks(self) -> None:
        """Start heartbeat and drain polling background tasks."""
        if self._running:
            return

        self._running = True

        # Register first
        await self.register()

        # Start tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._drain_poll_task = asyncio.create_task(self._drain_poll_loop())

        logger.info(
            f"Started background tasks (heartbeat: {self.heartbeat_interval}s, "
            f"drain poll: {self.drain_poll_interval}s)"
        )

    async def stop_background_tasks(self) -> None:
        """Stop background tasks and unregister."""
        self._running = False

        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._drain_poll_task:
            self._drain_poll_task.cancel()
            try:
                await self._drain_poll_task
            except asyncio.CancelledError:
                pass

        # Unregister
        try:
            await self.unregister()
        except Exception as e:
            logger.warning(f"Error unregistering instance: {e}")

        logger.info("Stopped background tasks")

    async def _heartbeat_loop(self) -> None:
        """Background task for sending heartbeats."""
        while self._running:
            try:
                # Get current pipeline status
                from .kg.shared_storage import is_any_pipeline_busy
                pipeline_status = is_any_pipeline_busy()

                await self.heartbeat(
                    processing_count=len(pipeline_status.get("busy_workspaces", [])),
                    pipeline_busy=pipeline_status.get("busy", False),
                )

                # Also cleanup dead instances occasionally
                await self.cleanup_dead_instances()

            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

            await asyncio.sleep(self.heartbeat_interval)

    async def _drain_poll_loop(self) -> None:
        """Background task for polling drain requests."""
        while self._running:
            try:
                drain_requested, drain_reason = await self.check_drain_requested()

                if drain_requested and self._on_drain_requested:
                    # Trigger callback (which should activate local drain mode)
                    self._on_drain_requested(drain_requested, drain_reason)

            except Exception as e:
                logger.warning(f"Drain poll error: {e}")

            await asyncio.sleep(self.drain_poll_interval)


# Global instance registry (initialized by the server)
_instance_registry: Optional[InstanceRegistry] = None


def get_instance_registry() -> Optional[InstanceRegistry]:
    """Get the global instance registry."""
    return _instance_registry


def set_instance_registry(registry: InstanceRegistry) -> None:
    """Set the global instance registry."""
    global _instance_registry
    _instance_registry = registry
