import asyncio
import configparser
import datetime
import itertools
import json
import os
import re
import ssl
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, ClassVar, Literal, TypeVar, cast, final, overload

import numpy as np
import pipmaster as pm
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from lightrag.kg.shared_storage import get_data_init_lock
from lightrag.namespace import NameSpace, is_namespace
from lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from lightrag.utils import logger

if not pm.is_installed('asyncpg'):
    pm.install('asyncpg')

import asyncpg  # type: ignore
from asyncpg import Connection, Pool  # type: ignore
from asyncpg.pool import PoolConnectionProxy  # type: ignore
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path='.env', override=False)

T = TypeVar('T')


class PostgreSQLDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config['host']
        self.port = config['port']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        self.workspace = config['workspace']
        self.max = int(config['max_connections'])
        self.min = int(config.get('min_connections', 5))
        self.increment = 1
        self.pool: Pool | None = None

        # SSL configuration
        self.ssl_mode = config.get('ssl_mode')
        self.ssl_cert = config.get('ssl_cert')
        self.ssl_key = config.get('ssl_key')
        self.ssl_root_cert = config.get('ssl_root_cert')
        self.ssl_crl = config.get('ssl_crl')

        # Vector configuration
        self.vector_index_type = config.get('vector_index_type')
        self.hnsw_m = config.get('hnsw_m')
        self.hnsw_ef = config.get('hnsw_ef')
        self.hnsw_ef_search = config.get('hnsw_ef_search')
        self.ivfflat_lists = config.get('ivfflat_lists')
        self.vchordrq_build_options = config.get('vchordrq_build_options')
        self.vchordrq_probes = config.get('vchordrq_probes')
        self.vchordrq_epsilon = config.get('vchordrq_epsilon')

        # Server settings
        self.server_settings = config.get('server_settings')

        # Statement LRU cache size (keep as-is, allow None for optional configuration)
        self.statement_cache_size = config.get('statement_cache_size')

        if self.user is None or self.password is None or self.database is None:
            raise ValueError('Missing database user, password, or database')

        # Guard concurrent pool resets
        self._pool_reconnect_lock = asyncio.Lock()

        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

        # Connection retry configuration
        self.connection_retry_attempts = config['connection_retry_attempts']
        self.connection_retry_backoff = config['connection_retry_backoff']
        self.connection_retry_backoff_max = max(
            self.connection_retry_backoff,
            config['connection_retry_backoff_max'],
        )
        self.pool_close_timeout = config['pool_close_timeout']
        logger.info(
            'PostgreSQL, Retry config: attempts=%s, backoff=%.1fs, backoff_max=%.1fs, pool_close_timeout=%.1fs',
            self.connection_retry_attempts,
            self.connection_retry_backoff,
            self.connection_retry_backoff_max,
            self.pool_close_timeout,
        )

        # Migration error tracking - stores (migration_name, error_message) tuples
        self._migration_failures: list[tuple[str, str]] = []

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Create SSL context based on configuration parameters."""
        if not self.ssl_mode:
            return None

        # Nothing to do if SSL mode is not configured
        if not self.ssl_mode:
            return None

        ssl_mode = self.ssl_mode.lower()

        # For simple modes that don't require custom context
        if ssl_mode in ['disable', 'allow', 'prefer', 'require']:
            if ssl_mode == 'disable':
                return None
            elif ssl_mode in ['require', 'prefer', 'allow']:
                # Return None for simple SSL requirement, handled in initdb
                return None

        # For modes that require certificate verification
        if ssl_mode in ['verify-ca', 'verify-full']:
            try:
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

                # Configure certificate verification
                if ssl_mode == 'verify-ca':
                    context.check_hostname = False
                elif ssl_mode == 'verify-full':
                    context.check_hostname = True

                # Load root certificate if provided
                if self.ssl_root_cert:
                    if os.path.exists(self.ssl_root_cert):
                        context.load_verify_locations(cafile=self.ssl_root_cert)
                        logger.info(f'PostgreSQL, Loaded SSL root certificate: {self.ssl_root_cert}')
                    else:
                        logger.warning(f'PostgreSQL, SSL root certificate file not found: {self.ssl_root_cert}')

                # Load client certificate and key if provided
                if self.ssl_cert and self.ssl_key:
                    if os.path.exists(self.ssl_cert) and os.path.exists(self.ssl_key):
                        context.load_cert_chain(self.ssl_cert, self.ssl_key)
                        logger.info(f'PostgreSQL, Loaded SSL client certificate: {self.ssl_cert}')
                    else:
                        logger.warning('PostgreSQL, SSL client certificate or key file not found')

                # Load certificate revocation list if provided
                if self.ssl_crl:
                    if os.path.exists(self.ssl_crl):
                        # ssl.SSLContext.load_verify_locations has cafile/capath/cadata params only;
                        # CRL files can be provided via cafile when bundled appropriately.
                        context.load_verify_locations(cafile=self.ssl_crl)
                        logger.info(f'PostgreSQL, Loaded SSL CRL: {self.ssl_crl}')
                    else:
                        logger.warning(f'PostgreSQL, SSL CRL file not found: {self.ssl_crl}')

                return context

            except Exception as e:
                logger.error(f'PostgreSQL, Failed to create SSL context: {e}')
                raise ValueError(f'SSL configuration error: {e}') from e

        # Unknown SSL mode
        logger.warning(f'PostgreSQL, Unknown SSL mode: {ssl_mode}, SSL disabled')
        return None

    async def initdb(self):
        # Prepare connection parameters
        connection_params = {
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'host': self.host,
            'port': self.port,
            'min_size': self.min,  # Configurable via POSTGRES_MIN_CONNECTIONS
            'max_size': self.max,
        }

        # Only add statement_cache_size if it's configured
        if self.statement_cache_size is not None:
            connection_params['statement_cache_size'] = int(self.statement_cache_size)
            logger.info(f'PostgreSQL, statement LRU cache size set as: {self.statement_cache_size}')

        # Add SSL configuration if provided
        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            connection_params['ssl'] = ssl_context
            logger.info('PostgreSQL, SSL configuration applied')
        elif self.ssl_mode:
            # Handle simple SSL modes without custom context
            if self.ssl_mode.lower() in ['require', 'prefer']:
                connection_params['ssl'] = True
            elif self.ssl_mode.lower() == 'disable':
                connection_params['ssl'] = False
            logger.info(f'PostgreSQL, SSL mode set to: {self.ssl_mode}')

        # Add server settings if provided
        if self.server_settings:
            try:
                settings = {}
                # The format is expected to be a query string, e.g., "key1=value1&key2=value2"
                pairs = self.server_settings.split('&')
                for pair in pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        settings[key] = value
                if settings:
                    connection_params['server_settings'] = settings
                    logger.info(f'PostgreSQL, Server settings applied: {settings}')
            except Exception as e:
                logger.warning(f'PostgreSQL, Failed to parse server_settings: {self.server_settings}, error: {e}')

        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async def _create_pool_once() -> None:
            pool = await asyncpg.create_pool(**connection_params)  # type: ignore
            try:
                async with pool.acquire() as connection:
                    await self.configure_vector_extension(connection)
            except Exception:
                await pool.close()
                raise
            self.pool = pool

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.connection_retry_attempts),
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_strategy,
                before_sleep=self._before_sleep,
                reraise=True,
            ):
                with attempt:
                    await _create_pool_once()

            ssl_status = 'with SSL' if connection_params.get('ssl') else 'without SSL'
            logger.info(f'PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database} {ssl_status}')
        except Exception as e:
            logger.error(f'PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}')
            raise

    async def _ensure_pool(self) -> None:
        """Ensure the connection pool is initialised."""
        if self.pool is None:
            async with self._pool_reconnect_lock:
                if self.pool is None:
                    await self.initdb()

    async def _reset_pool(self) -> None:
        async with self._pool_reconnect_lock:
            if self.pool is not None:
                try:
                    await asyncio.wait_for(self.pool.close(), timeout=self.pool_close_timeout)
                except asyncio.TimeoutError:
                    logger.error(
                        'PostgreSQL, Timed out closing connection pool after %.2fs',
                        self.pool_close_timeout,
                    )
                except Exception as close_error:  # pragma: no cover - defensive logging
                    logger.warning(f'PostgreSQL, Failed to close existing connection pool cleanly: {close_error!r}')
            self.pool = None

    async def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Hook invoked by tenacity before sleeping between retries."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            'PostgreSQL transient connection issue on attempt %s/%s: %r',
            retry_state.attempt_number,
            self.connection_retry_attempts,
            exc,
        )
        await self._reset_pool()

    async def _run_with_retry(
        self,
        operation: Callable[[Connection | PoolConnectionProxy], Awaitable[T]],
        *,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> T:
        """
        Execute a database operation with automatic retry for transient failures.

        Args:
            operation: Async callable that receives an active connection.
            with_age: Whether to configure Apache AGE on the connection.
            graph_name: AGE graph name; required when with_age is True.

        Returns:
            The result returned by the operation.

        Raises:
            Exception: Propagates the last error if all retry attempts fail or a non-transient error occurs.
        """
        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.connection_retry_attempts),
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_strategy,
            before_sleep=self._before_sleep,
            reraise=True,
        ):
            with attempt:
                await self._ensure_pool()
                assert self.pool is not None
                async with self.pool.acquire() as connection:  # type: ignore[arg-type]
                    if with_age and graph_name:
                        await self.configure_age(connection, graph_name)
                    elif with_age and not graph_name:
                        raise ValueError('Graph name is required when with_age is True')
                    if self.vector_index_type == 'HNSW':
                        await self.configure_hnsw(connection)
                    elif self.vector_index_type == 'VCHORDRQ':
                        await self.configure_vchordrq(connection)
                    return await operation(connection)

        # Should be unreachable because AsyncRetrying raises on final failure,
        # but keeps the type-checker happy about the return type.
        raise RuntimeError('Unexpected retry exhaustion')

    async def _timed_operation(
        self,
        coro: Awaitable[T],
        operation_name: str,
        slow_threshold: float = 1.0,
    ) -> T:
        """
        Execute an async operation with timing and slow query detection.

        Args:
            coro: The coroutine to execute
            operation_name: Name for logging
            slow_threshold: Log warning if operation takes longer (seconds)

        Returns:
            The result of the coroutine
        """
        import time

        start = time.perf_counter()
        try:
            result = await coro
            elapsed = time.perf_counter() - start
            if elapsed > slow_threshold:
                logger.warning(f'[{self.workspace}] Slow operation: {operation_name} took {elapsed:.2f}s')
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.error(f'[{self.workspace}] Failed operation: {operation_name} after {elapsed:.2f}s')
            raise

    @staticmethod
    async def configure_vector_extension(connection: Connection | PoolConnectionProxy) -> None:
        """Create VECTOR extension if it doesn't exist for vector similarity operations."""
        try:
            await connection.execute('CREATE EXTENSION IF NOT EXISTS vector')  # type: ignore
            logger.info('PostgreSQL, VECTOR extension enabled')
        except Exception as e:
            logger.warning(f'Could not create VECTOR extension: {e}')
            # Don't raise - let the system continue without vector extension

    @staticmethod
    async def configure_age_extension(connection: Connection | PoolConnectionProxy) -> None:
        """Create AGE extension if it doesn't exist for graph operations."""
        try:
            await connection.execute('CREATE EXTENSION IF NOT EXISTS AGE CASCADE')  # type: ignore
            logger.info('PostgreSQL, AGE extension enabled')
        except Exception as e:
            logger.warning(f'Could not create AGE extension: {e}')
            # Don't raise - let the system continue without AGE extension

    @staticmethod
    async def configure_age(connection: Connection | PoolConnectionProxy, graph_name: str) -> None:
        """Set the Apache AGE environment and creates a graph if it does not exist.

        This method:
        - Sets the PostgreSQL `search_path` to include `ag_catalog`, ensuring that Apache AGE functions can be used without specifying the schema.
        - Attempts to create a new graph with the provided `graph_name` if it does not already exist.
        - Silently ignores errors related to the graph already existing.

        """
        try:
            await connection.execute(  # type: ignore
                'SET search_path = ag_catalog, "$user", public'
            )
            await connection.execute(  # type: ignore
                f"select create_graph('{graph_name}')"
            )
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass

    async def configure_vchordrq(self, connection: Connection | PoolConnectionProxy) -> None:
        """Configure VCHORDRQ extension for vector similarity search.

        Raises:
            asyncpg.exceptions.UndefinedObjectError: If VCHORDRQ extension is not installed
            asyncpg.exceptions.InvalidParameterValueError: If parameter value is invalid

        Note:
            This method does not catch exceptions. Configuration errors will fail-fast,
            while transient connection errors will be retried by _run_with_retry.
        """
        # Handle probes parameter - only set if non-empty value is provided
        if self.vchordrq_probes and str(self.vchordrq_probes).strip():
            await connection.execute(f"SET vchordrq.probes TO '{self.vchordrq_probes}'")
            logger.debug(f'PostgreSQL, VCHORDRQ probes set to: {self.vchordrq_probes}')

        # Handle epsilon parameter independently - check for None to allow 0.0 as valid value
        if self.vchordrq_epsilon is not None:
            await connection.execute(f'SET vchordrq.epsilon TO {self.vchordrq_epsilon}')
            logger.debug(f'PostgreSQL, VCHORDRQ epsilon set to: {self.vchordrq_epsilon}')

    async def configure_hnsw(self, connection: Connection | PoolConnectionProxy) -> None:
        """Configure HNSW search parameters for this connection.

        Sets the ef_search parameter which controls the trade-off between
        search accuracy and speed. Higher values give better recall but
        slower queries.

        Note:
            This method does not catch exceptions. Configuration errors will fail-fast,
            while transient connection errors will be retried by _run_with_retry.
        """
        if self.hnsw_ef_search is not None:
            await connection.execute(f'SET hnsw.ef_search = {int(self.hnsw_ef_search)}')
            logger.debug(f'PostgreSQL, HNSW ef_search set to: {self.hnsw_ef_search}')

    async def _run_migration(self, migration_coro, migration_name: str, error_msg: str) -> bool:
        """Run a migration with error tracking.

        Args:
            migration_coro: The migration coroutine to run
            migration_name: Short name for tracking (e.g., "timestamp_columns")
            error_msg: Full error message for logging

        Returns:
            True if migration succeeded, False if it failed
        """
        try:
            await migration_coro
            return True
        except Exception as e:
            logger.error(f'PostgreSQL, {error_msg}: {e}')
            self._migration_failures.append((migration_name, str(e)))
            return False

    def get_migration_status(self) -> dict[str, Any]:
        """Get the current migration status including any failures.

        Returns:
            Dict with 'success' bool and 'failures' list of (name, error) tuples
        """
        return {
            'success': len(self._migration_failures) == 0,
            'failures': self._migration_failures.copy(),
        }

    async def _migrate_llm_cache_schema(self):
        """Migrate LLM cache schema: add new columns and remove deprecated mode field"""
        try:
            # Check if all columns exist
            check_columns_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_llm_cache'
            AND column_name IN ('chunk_id', 'cache_type', 'queryparam', 'mode')
            """

            existing_columns = await self.query(check_columns_sql, multirows=True)
            existing_column_names = {col['column_name'] for col in existing_columns} if existing_columns else set()

            # Add missing chunk_id column
            if 'chunk_id' not in existing_column_names:
                logger.info('Adding chunk_id column to LIGHTRAG_LLM_CACHE table')
                add_chunk_id_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                ADD COLUMN chunk_id VARCHAR(255) NULL
                """
                await self.execute(add_chunk_id_sql)
                logger.info('Successfully added chunk_id column to LIGHTRAG_LLM_CACHE table')
            else:
                logger.info('chunk_id column already exists in LIGHTRAG_LLM_CACHE table')

            # Add missing cache_type column
            if 'cache_type' not in existing_column_names:
                logger.info('Adding cache_type column to LIGHTRAG_LLM_CACHE table')
                add_cache_type_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                ADD COLUMN cache_type VARCHAR(32) NULL
                """
                await self.execute(add_cache_type_sql)
                logger.info('Successfully added cache_type column to LIGHTRAG_LLM_CACHE table')

                # Migrate existing data using optimized regex pattern
                logger.info('Migrating existing LLM cache data to populate cache_type field (optimized)')
                optimized_update_sql = """
                UPDATE LIGHTRAG_LLM_CACHE
                SET cache_type = CASE
                    WHEN id ~ '^[^:]+:[^:]+:' THEN split_part(id, ':', 2)
                    ELSE 'extract'
                END
                WHERE cache_type IS NULL
                """
                await self.execute(optimized_update_sql)
                logger.info('Successfully migrated existing LLM cache data')
            else:
                logger.info('cache_type column already exists in LIGHTRAG_LLM_CACHE table')

            # Add missing queryparam column
            if 'queryparam' not in existing_column_names:
                logger.info('Adding queryparam column to LIGHTRAG_LLM_CACHE table')
                add_queryparam_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                ADD COLUMN queryparam JSONB NULL
                """
                await self.execute(add_queryparam_sql)
                logger.info('Successfully added queryparam column to LIGHTRAG_LLM_CACHE table')
            else:
                logger.info('queryparam column already exists in LIGHTRAG_LLM_CACHE table')

            # Remove deprecated mode field if it exists
            if 'mode' in existing_column_names:
                logger.info('Removing deprecated mode column from LIGHTRAG_LLM_CACHE table')

                # First, drop the primary key constraint that includes mode
                drop_pk_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                DROP CONSTRAINT IF EXISTS LIGHTRAG_LLM_CACHE_PK
                """
                await self.execute(drop_pk_sql)
                logger.info('Dropped old primary key constraint')

                # Drop the mode column
                drop_mode_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                DROP COLUMN mode
                """
                await self.execute(drop_mode_sql)
                logger.info('Successfully removed mode column from LIGHTRAG_LLM_CACHE table')

                # Create new primary key constraint without mode
                add_pk_sql = """
                ALTER TABLE LIGHTRAG_LLM_CACHE
                ADD CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                """
                await self.execute(add_pk_sql)
                logger.info('Created new primary key constraint (workspace, id)')
            else:
                logger.info('mode column does not exist in LIGHTRAG_LLM_CACHE table')

        except Exception as e:
            logger.warning(f'Failed to migrate LLM cache schema: {e}')

    async def _migrate_timestamp_columns(self):
        """Migrate timestamp columns in tables to witimezone-free types, assuming original data is in UTC time"""
        # Tables and columns that need migration
        tables_to_migrate = {
            'LIGHTRAG_VDB_ENTITY': ['create_time', 'update_time'],
            'LIGHTRAG_VDB_RELATION': ['create_time', 'update_time'],
            'LIGHTRAG_DOC_CHUNKS': ['create_time', 'update_time'],
            'LIGHTRAG_DOC_STATUS': ['created_at', 'updated_at'],
        }

        try:
            # Optimization: Batch check all columns in one query instead of 8 separate queries
            table_names_lower = [t.lower() for t in tables_to_migrate]
            all_column_names = list({col for cols in tables_to_migrate.values() for col in cols})

            check_all_columns_sql = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_name = ANY($1)
            AND column_name = ANY($2)
            """

            all_columns_result = await self.query(
                check_all_columns_sql,
                [table_names_lower, all_column_names],
                multirows=True,
            )

            # Build lookup dict: (table_name, column_name) -> data_type
            column_types = {}
            if all_columns_result:
                column_types = {
                    (row['table_name'].upper(), row['column_name']): row['data_type'] for row in all_columns_result
                }

            # Now iterate and migrate only what's needed
            for table_name, columns in tables_to_migrate.items():
                for column_name in columns:
                    try:
                        data_type = column_types.get((table_name, column_name))

                        if not data_type:
                            logger.warning(f'Column {table_name}.{column_name} does not exist, skipping migration')
                            continue

                        # Check column type
                        if data_type == 'timestamp without time zone':
                            logger.debug(
                                f'Column {table_name}.{column_name} is already witimezone-free, no migration needed'
                            )
                            continue

                        # Execute migration, explicitly specifying UTC timezone for interpreting original data
                        logger.info(f'Migrating {table_name}.{column_name} from {data_type} to TIMESTAMP(0) type')
                        migration_sql = f"""
                        ALTER TABLE {table_name}
                        ALTER COLUMN {column_name} TYPE TIMESTAMP(0),
                        ALTER COLUMN {column_name} SET DEFAULT CURRENT_TIMESTAMP
                        """

                        await self.execute(migration_sql)
                        logger.info(f'Successfully migrated {table_name}.{column_name} to timezone-free type')
                    except Exception as e:
                        # Log error but don't interrupt the process
                        logger.warning(f'Failed to migrate {table_name}.{column_name}: {e}')
        except Exception as e:
            logger.error(f'Failed to batch check timestamp columns: {e}')

    @staticmethod
    def _safe_int_field(record: dict[str, Any] | None, key: str) -> int:
        """Return int value for key if record present, else 0."""
        if isinstance(record, dict):
            try:
                value = record.get(key, 0)
                return int(value) if value is not None else 0
            except (TypeError, ValueError):
                return 0
        return 0

    async def _migrate_doc_chunks_to_vdb_chunks(self):
        """
        Migrate data from LIGHTRAG_DOC_CHUNKS to LIGHTRAG_VDB_CHUNKS if specific conditions are met.
        This migration is intended for users who are upgrading and have an older table structure
        where LIGHTRAG_DOC_CHUNKS contained a `content_vector` column.

        """
        try:
            # 1. Check if the new table LIGHTRAG_VDB_CHUNKS is empty
            vdb_chunks_count_sql = 'SELECT COUNT(1) as count FROM LIGHTRAG_VDB_CHUNKS'
            vdb_chunks_count_result = await self.query(vdb_chunks_count_sql)
            if self._safe_int_field(vdb_chunks_count_result, 'count') > 0:
                logger.info('Skipping migration: LIGHTRAG_VDB_CHUNKS already contains data.')
                return

            # 2. Check if `content_vector` column exists in the old table
            check_column_sql = """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_chunks' AND column_name = 'content_vector'
            """
            column_exists = await self.query(check_column_sql)
            if not column_exists:
                logger.info('Skipping migration: `content_vector` not found in LIGHTRAG_DOC_CHUNKS')
                return

            # 3. Check if the old table LIGHTRAG_DOC_CHUNKS has data
            doc_chunks_count_sql = 'SELECT COUNT(1) as count FROM LIGHTRAG_DOC_CHUNKS'
            doc_chunks_count_result = await self.query(doc_chunks_count_sql)
            if self._safe_int_field(doc_chunks_count_result, 'count') == 0:
                logger.info('Skipping migration: LIGHTRAG_DOC_CHUNKS is empty.')
                return

            # 4. Perform the migration
            logger.info('Starting data migration from LIGHTRAG_DOC_CHUNKS to LIGHTRAG_VDB_CHUNKS...')
            migration_sql = """
            INSERT INTO LIGHTRAG_VDB_CHUNKS (
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            )
            SELECT
                id, workspace, full_doc_id, chunk_order_index, tokens, content,
                content_vector, file_path, create_time, update_time
            FROM LIGHTRAG_DOC_CHUNKS
            ON CONFLICT (workspace, id) DO NOTHING;
            """
            await self.execute(migration_sql)
            logger.info('Data migration to LIGHTRAG_VDB_CHUNKS completed successfully.')

        except Exception as e:
            logger.error(f'Failed during data migration to LIGHTRAG_VDB_CHUNKS: {e}')
            # Do not re-raise, to allow the application to start

    async def _check_llm_cache_needs_migration(self):
        """Check if LLM cache data needs migration by examining any record with old format"""
        try:
            # Optimized query: directly check for old format records without sorting
            check_sql = """
            SELECT 1 FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            LIMIT 1
            """
            result = await self.query(check_sql)

            # If any old format record exists, migration is needed
            return result is not None

        except Exception as e:
            logger.warning(f'Failed to check LLM cache migration status: {e}')
            return False

    async def _migrate_llm_cache_to_flattened_keys(self):
        """Optimized version: directly execute single UPDATE migration to migrate old format cache keys to flattened format"""
        try:
            # Check if migration is needed
            check_sql = """
            SELECT COUNT(*) as count FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            result = await self.query(check_sql)

            if self._safe_int_field(result, 'count') == 0:
                logger.info('No old format LLM cache data found, skipping migration')
                return
            old_count = self._safe_int_field(result, 'count')
            logger.info(f'Found {old_count} old format cache records')

            # Check potential primary key conflicts (optional but recommended)
            conflict_check_sql = """
            WITH new_ids AS (
                SELECT
                    workspace,
                    mode,
                    id as old_id,
                    mode || ':' ||
                    CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                    md5(original_prompt) as new_id
                FROM LIGHTRAG_LLM_CACHE
                WHERE id NOT LIKE '%:%'
            )
            SELECT COUNT(*) as conflicts
            FROM new_ids n1
            JOIN LIGHTRAG_LLM_CACHE existing
            ON existing.workspace = n1.workspace
            AND existing.mode = n1.mode
            AND existing.id = n1.new_id
            WHERE existing.id LIKE '%:%'  -- Only check conflicts with existing new format records
            """

            conflict_result = await self.query(conflict_check_sql)
            conflicts = self._safe_int_field(conflict_result, 'conflicts')
            if conflicts > 0:
                logger.warning(f'Found {conflicts} potential ID conflicts with existing records')
                # Can choose to continue or abort, here we choose to continue and log warning

            # Execute single UPDATE migration
            logger.info('Starting optimized LLM cache migration...')
            migration_sql = """
            UPDATE LIGHTRAG_LLM_CACHE
            SET
                id = mode || ':' ||
                     CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END || ':' ||
                     md5(original_prompt),
                cache_type = CASE WHEN mode = 'default' THEN 'extract' ELSE 'unknown' END,
                update_time = CURRENT_TIMESTAMP
            WHERE id NOT LIKE '%:%'
            """

            # Execute migration
            await self.execute(migration_sql)

            # Verify migration results
            verify_sql = """
            SELECT COUNT(*) as remaining_old FROM LIGHTRAG_LLM_CACHE
            WHERE id NOT LIKE '%:%'
            """
            verify_result = await self.query(verify_sql)
            remaining = self._safe_int_field(verify_result, 'remaining_old')

            if remaining == 0:
                logger.info(f'✅ Successfully migrated {old_count} LLM cache records to flattened format')
            else:
                logger.warning(f'⚠️ Migration completed but {remaining} old format records remain')

        except Exception as e:
            logger.error(f'Optimized LLM cache migration failed: {e}')
            raise

    async def _migrate_doc_status_add_chunks_list(self):
        """Add chunks_list column to LIGHTRAG_DOC_STATUS table if it doesn't exist"""
        try:
            # Check if chunks_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'chunks_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info('Adding chunks_list column to LIGHTRAG_DOC_STATUS table')
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN chunks_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info('Successfully added chunks_list column to LIGHTRAG_DOC_STATUS table')
            else:
                logger.info('chunks_list column already exists in LIGHTRAG_DOC_STATUS table')
        except Exception as e:
            logger.warning(f'Failed to add chunks_list column to LIGHTRAG_DOC_STATUS: {e}')

    async def _migrate_text_chunks_add_llm_cache_list(self):
        """Add llm_cache_list column to LIGHTRAG_DOC_CHUNKS table if it doesn't exist"""
        try:
            # Check if llm_cache_list column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_chunks'
            AND column_name = 'llm_cache_list'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info('Adding llm_cache_list column to LIGHTRAG_DOC_CHUNKS table')
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_CHUNKS
                ADD COLUMN llm_cache_list JSONB NULL DEFAULT '[]'::jsonb
                """
                await self.execute(add_column_sql)
                logger.info('Successfully added llm_cache_list column to LIGHTRAG_DOC_CHUNKS table')
            else:
                logger.info('llm_cache_list column already exists in LIGHTRAG_DOC_CHUNKS table')
        except Exception as e:
            logger.warning(f'Failed to add llm_cache_list column to LIGHTRAG_DOC_CHUNKS: {e}')

    async def _migrate_add_s3_key_columns(self):
        """Add s3_key column to LIGHTRAG_DOC_FULL, LIGHTRAG_DOC_CHUNKS, and LIGHTRAG_DOC_STATUS tables if they don't exist"""
        tables = [
            ('lightrag_doc_full', 'LIGHTRAG_DOC_FULL'),
            ('lightrag_doc_chunks', 'LIGHTRAG_DOC_CHUNKS'),
            ('lightrag_doc_status', 'LIGHTRAG_DOC_STATUS'),
        ]

        for table_name_lower, table_name in tables:
            try:
                check_column_sql = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name_lower}'
                AND column_name = 's3_key'
                """

                column_info = await self.query(check_column_sql)
                if not column_info:
                    logger.info(f'Adding s3_key column to {table_name} table')
                    add_column_sql = f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN s3_key TEXT NULL
                    """
                    await self.execute(add_column_sql)
                    logger.info(f'Successfully added s3_key column to {table_name} table')
                else:
                    logger.info(f's3_key column already exists in {table_name} table')
            except Exception as e:
                logger.warning(f'Failed to add s3_key column to {table_name}: {e}')

    async def _migrate_add_chunk_position_columns(self):
        """Add char_start and char_end columns to LIGHTRAG_DOC_CHUNKS table if they don't exist"""
        columns = [
            ('char_start', 'INTEGER NULL'),
            ('char_end', 'INTEGER NULL'),
        ]

        for column_name, column_type in columns:
            try:
                check_column_sql = f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'lightrag_doc_chunks'
                AND column_name = '{column_name}'
                """

                column_info = await self.query(check_column_sql)
                if not column_info:
                    logger.info(f'Adding {column_name} column to LIGHTRAG_DOC_CHUNKS table')
                    add_column_sql = f"""
                    ALTER TABLE LIGHTRAG_DOC_CHUNKS
                    ADD COLUMN {column_name} {column_type}
                    """
                    await self.execute(add_column_sql)
                    logger.info(f'Successfully added {column_name} column to LIGHTRAG_DOC_CHUNKS table')
                else:
                    logger.info(f'{column_name} column already exists in LIGHTRAG_DOC_CHUNKS table')
            except Exception as e:
                logger.warning(f'Failed to add {column_name} column to LIGHTRAG_DOC_CHUNKS: {e}')

    async def _migrate_doc_status_add_track_id(self):
        """Add track_id column to LIGHTRAG_DOC_STATUS table if it doesn't exist and create index"""
        try:
            # Check if track_id column exists
            check_column_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'track_id'
            """

            column_info = await self.query(check_column_sql)
            if not column_info:
                logger.info('Adding track_id column to LIGHTRAG_DOC_STATUS table')
                add_column_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN track_id VARCHAR(255) NULL
                """
                await self.execute(add_column_sql)
                logger.info('Successfully added track_id column to LIGHTRAG_DOC_STATUS table')
            else:
                logger.info('track_id column already exists in LIGHTRAG_DOC_STATUS table')

            # Check if track_id index exists
            check_index_sql = """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'lightrag_doc_status'
            AND indexname = 'idx_lightrag_doc_status_track_id'
            """

            index_info = await self.query(check_index_sql)
            if not index_info:
                logger.info('Creating index on track_id column for LIGHTRAG_DOC_STATUS table')
                create_index_sql = """
                CREATE INDEX idx_lightrag_doc_status_track_id ON LIGHTRAG_DOC_STATUS (track_id)
                """
                await self.execute(create_index_sql)
                logger.info('Successfully created index on track_id column for LIGHTRAG_DOC_STATUS table')
            else:
                logger.info('Index on track_id column already exists for LIGHTRAG_DOC_STATUS table')

        except Exception as e:
            logger.warning(f'Failed to add track_id column or index to LIGHTRAG_DOC_STATUS: {e}')

    async def _migrate_doc_status_add_metadata_error_msg(self):
        """Add metadata and error_msg columns to LIGHTRAG_DOC_STATUS table if they don't exist"""
        try:
            # Check if metadata column exists
            check_metadata_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'metadata'
            """

            metadata_info = await self.query(check_metadata_sql)
            if not metadata_info:
                logger.info('Adding metadata column to LIGHTRAG_DOC_STATUS table')
                add_metadata_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN metadata JSONB NULL DEFAULT '{}'::jsonb
                """
                await self.execute(add_metadata_sql)
                logger.info('Successfully added metadata column to LIGHTRAG_DOC_STATUS table')
            else:
                logger.info('metadata column already exists in LIGHTRAG_DOC_STATUS table')

            # Check if error_msg column exists
            check_error_msg_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'lightrag_doc_status'
            AND column_name = 'error_msg'
            """

            error_msg_info = await self.query(check_error_msg_sql)
            if not error_msg_info:
                logger.info('Adding error_msg column to LIGHTRAG_DOC_STATUS table')
                add_error_msg_sql = """
                ALTER TABLE LIGHTRAG_DOC_STATUS
                ADD COLUMN error_msg TEXT NULL
                """
                await self.execute(add_error_msg_sql)
                logger.info('Successfully added error_msg column to LIGHTRAG_DOC_STATUS table')
            else:
                logger.info('error_msg column already exists in LIGHTRAG_DOC_STATUS table')

        except Exception as e:
            logger.warning(f'Failed to add metadata/error_msg columns to LIGHTRAG_DOC_STATUS: {e}')

    async def _migrate_field_lengths(self):
        """Migrate database field lengths: entity_name, source_id, target_id, and file_path"""
        # Define the field changes needed
        field_migrations = [
            {
                'table': 'LIGHTRAG_VDB_ENTITY',
                'column': 'entity_name',
                'old_type': 'character varying(255)',
                'new_type': 'VARCHAR(512)',
                'description': 'entity_name from 255 to 512',
            },
            {
                'table': 'LIGHTRAG_VDB_RELATION',
                'column': 'source_id',
                'old_type': 'character varying(256)',
                'new_type': 'VARCHAR(512)',
                'description': 'source_id from 256 to 512',
            },
            {
                'table': 'LIGHTRAG_VDB_RELATION',
                'column': 'target_id',
                'old_type': 'character varying(256)',
                'new_type': 'VARCHAR(512)',
                'description': 'target_id from 256 to 512',
            },
            {
                'table': 'LIGHTRAG_DOC_CHUNKS',
                'column': 'file_path',
                'old_type': 'character varying(256)',
                'new_type': 'TEXT',
                'description': 'file_path to TEXT NULL',
            },
            {
                'table': 'LIGHTRAG_VDB_CHUNKS',
                'column': 'file_path',
                'old_type': 'character varying(256)',
                'new_type': 'TEXT',
                'description': 'file_path to TEXT NULL',
            },
        ]

        try:
            # Optimization: Batch check all columns in one query instead of 5 separate queries
            unique_tables = list({m['table'].lower() for m in field_migrations})
            unique_columns = list({m['column'] for m in field_migrations})

            check_all_columns_sql = """
            SELECT table_name, column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_name = ANY($1)
            AND column_name = ANY($2)
            """

            all_columns_result = await self.query(
                check_all_columns_sql, [unique_tables, unique_columns], multirows=True
            )

            # Build lookup dict: (table_name, column_name) -> column_info
            column_info_map = {}
            if all_columns_result:
                column_info_map = {(row['table_name'].upper(), row['column_name']): row for row in all_columns_result}

            # Now iterate and migrate only what's needed
            for migration in field_migrations:
                try:
                    column_info = column_info_map.get((migration['table'], migration['column']))

                    if not column_info:
                        logger.warning(
                            f'Column {migration["table"]}.{migration["column"]} does not exist, skipping migration'
                        )
                        continue

                    current_type = column_info.get('data_type', '').lower()
                    current_length = column_info.get('character_maximum_length')

                    # Check if migration is needed
                    needs_migration = False

                    if (
                        (migration['column'] == 'entity_name' and current_length == 255)
                        or (migration['column'] in ['source_id', 'target_id'] and current_length == 256)
                        or (migration['column'] == 'file_path' and current_type == 'character varying')
                    ):
                        needs_migration = True

                    if needs_migration:
                        logger.info(f'Migrating {migration["table"]}.{migration["column"]}: {migration["description"]}')

                        # Execute the migration
                        alter_sql = f"""
                        ALTER TABLE {migration['table']}
                        ALTER COLUMN {migration['column']} TYPE {migration['new_type']}
                        """

                        await self.execute(alter_sql)
                        logger.info(f'Successfully migrated {migration["table"]}.{migration["column"]}')
                    else:
                        logger.debug(
                            f'Column {migration["table"]}.{migration["column"]} already has correct type, no migration needed'
                        )

                except Exception as e:
                    # Log error but don't interrupt the process
                    logger.warning(f'Failed to migrate {migration["table"]}.{migration["column"]}: {e}')
        except Exception as e:
            logger.error(f'Failed to batch check field lengths: {e}')

    async def check_tables(self):
        # First create all tables
        for k, v in TABLES.items():
            try:
                await self.query(f'SELECT 1 FROM {k} LIMIT 1')
            except Exception:
                try:
                    logger.info(f'PostgreSQL, Try Creating table {k} in database')
                    await self.execute(v['ddl'])
                    logger.info(f'PostgreSQL, Creation success table {k} in PostgreSQL database')
                except Exception as e:
                    logger.error(
                        f'PostgreSQL, Failed to create table {k} in database, Please verify the connection with PostgreSQL database, Got: {e}'
                    )
                    raise e

        # Batch check all indexes at once (optimization: single query instead of N queries)
        existing_indexes: set[str] = set()
        try:
            table_names = list(TABLES.keys())
            table_names_lower = [t.lower() for t in table_names]

            # Get all existing indexes for our tables in one query
            check_all_indexes_sql = """
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE tablename = ANY($1)
            """
            existing_indexes_result = await self.query(check_all_indexes_sql, [table_names_lower], multirows=True)

            # Build a set of existing index names for fast lookup
            if existing_indexes_result:
                existing_indexes = {row['indexname'] for row in existing_indexes_result}

            # Create missing indexes
            # Tables that don't have an 'id' column (use different primary key structure)
            tables_without_id = {'LIGHTRAG_ENTITY_ALIASES'}

            for k in table_names:
                # Skip tables that don't have an 'id' column
                if k in tables_without_id:
                    continue

                # Create index for id column if missing
                index_name = f'idx_{k.lower()}_id'
                if index_name not in existing_indexes:
                    try:
                        create_index_sql = f'CREATE INDEX {index_name} ON {k}(id)'
                        logger.info(f'PostgreSQL, Creating index {index_name} on table {k}')
                        await self.execute(create_index_sql)
                    except Exception as e:
                        logger.error(f'PostgreSQL, Failed to create index {index_name}, Got: {e}')

                # Create composite index for (workspace, id) if missing
                composite_index_name = f'idx_{k.lower()}_workspace_id'
                if composite_index_name not in existing_indexes:
                    try:
                        create_composite_index_sql = f'CREATE INDEX {composite_index_name} ON {k}(workspace, id)'
                        logger.info(f'PostgreSQL, Creating composite index {composite_index_name} on table {k}')
                        await self.execute(create_composite_index_sql)
                    except Exception as e:
                        logger.error(f'PostgreSQL, Failed to create composite index {composite_index_name}, Got: {e}')
        except Exception as e:
            logger.error(f'PostgreSQL, Failed to batch check/create indexes: {e}')

        # Create additional performance indexes for common query patterns
        try:
            performance_indexes = [
                # Entity resolution lookups (used heavily during ingestion)
                ('idx_lightrag_vdb_entity_workspace_name', 'LIGHTRAG_VDB_ENTITY', '(workspace, entity_name)'),
                # Graph traversal queries (forward and backward edge lookups)
                ('idx_lightrag_vdb_relation_workspace_source', 'LIGHTRAG_VDB_RELATION', '(workspace, source_id)'),
                ('idx_lightrag_vdb_relation_workspace_target', 'LIGHTRAG_VDB_RELATION', '(workspace, target_id)'),
                # Document chunk lookups by document
                ('idx_lightrag_doc_chunks_workspace_doc', 'LIGHTRAG_DOC_CHUNKS', '(workspace, full_doc_id)'),
                # File path lookups in doc status
                ('idx_lightrag_doc_status_workspace_path', 'LIGHTRAG_DOC_STATUS', '(workspace, file_path)'),
            ]

            # GIN indexes for array membership queries (chunk_ids lookups)
            # and full-text search (BM25-style keyword search)
            gin_indexes = [
                ('idx_lightrag_vdb_entity_chunk_ids_gin', 'LIGHTRAG_VDB_ENTITY', 'USING gin (chunk_ids)'),
                ('idx_lightrag_vdb_relation_chunk_ids_gin', 'LIGHTRAG_VDB_RELATION', 'USING gin (chunk_ids)'),
                # Full-text search GIN index for BM25 keyword search on chunks
                (
                    'idx_lightrag_doc_chunks_content_fts_gin',
                    'LIGHTRAG_DOC_CHUNKS',
                    "USING gin (to_tsvector('english', content))",
                ),
            ]

            # Create GIN indexes separately (different syntax)
            for index_name, table_name, index_type in gin_indexes:
                if index_name not in existing_indexes:
                    try:
                        create_gin_sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} {index_type}'
                        logger.info(f'PostgreSQL, Creating GIN index {index_name} on {table_name}')
                        await self.execute(create_gin_sql)
                    except Exception as e:
                        logger.warning(f'PostgreSQL, Failed to create GIN index {index_name}: {e}')

            for index_name, table_name, columns in performance_indexes:
                if index_name not in existing_indexes:
                    try:
                        create_perf_index_sql = f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}{columns}'
                        logger.info(f'PostgreSQL, Creating performance index {index_name} on {table_name}')
                        await self.execute(create_perf_index_sql)
                    except Exception as e:
                        logger.warning(f'PostgreSQL, Failed to create performance index {index_name}: {e}')
        except Exception as e:
            logger.error(f'PostgreSQL, Failed to create performance indexes: {e}')

        # Create vector indexs
        if self.vector_index_type:
            logger.info(f'PostgreSQL, Create vector indexs, type: {self.vector_index_type}')
            try:
                if self.vector_index_type in ['HNSW', 'IVFFLAT', 'VCHORDRQ']:
                    await self._create_vector_indexes()
                else:
                    logger.warning(
                        "Doesn't support this vector index type: {self.vector_index_type}. "
                        'Supported types: HNSW, IVFFLAT, VCHORDRQ'
                    )
            except Exception as e:
                logger.error(f'PostgreSQL, Failed to create vector index, type: {self.vector_index_type}, Got: {e}')
        # Run schema migrations with error tracking
        # Failures are logged and tracked but don't stop initialization
        await self._run_migration(
            self._migrate_timestamp_columns(),
            'timestamp_columns',
            'Failed to migrate timestamp columns',
        )

        await self._run_migration(
            self._migrate_llm_cache_schema(),
            'llm_cache_schema',
            'Failed to migrate LLM cache schema',
        )

        await self._run_migration(
            self._migrate_doc_chunks_to_vdb_chunks(),
            'doc_chunks_to_vdb',
            'Failed to migrate doc_chunks to vdb_chunks',
        )

        # Check and migrate LLM cache to flattened keys if needed
        try:
            if await self._check_llm_cache_needs_migration():
                await self._run_migration(
                    self._migrate_llm_cache_to_flattened_keys(),
                    'llm_cache_flattened_keys',
                    'LLM cache migration to flattened keys failed',
                )
        except Exception as e:
            logger.error(f'PostgreSQL, LLM cache migration check failed: {e}')
            self._migration_failures.append(('llm_cache_check', str(e)))

        await self._run_migration(
            self._migrate_doc_status_add_chunks_list(),
            'doc_status_chunks_list',
            'Failed to migrate doc status chunks_list field',
        )

        await self._run_migration(
            self._migrate_text_chunks_add_llm_cache_list(),
            'text_chunks_llm_cache_list',
            'Failed to migrate text chunks llm_cache_list field',
        )

        await self._run_migration(
            self._migrate_add_s3_key_columns(),
            's3_key_columns',
            'Failed to add s3_key columns to doc tables',
        )

        await self._run_migration(
            self._migrate_add_chunk_position_columns(),
            'chunk_position_columns',
            'Failed to add char_start/char_end columns to doc chunks table',
        )

        await self._run_migration(
            self._migrate_field_lengths(),
            'field_lengths',
            'Failed to migrate field lengths',
        )

        await self._run_migration(
            self._migrate_doc_status_add_track_id(),
            'doc_status_track_id',
            'Failed to migrate doc status track_id field',
        )

        await self._run_migration(
            self._migrate_doc_status_add_metadata_error_msg(),
            'doc_status_metadata_error_msg',
            'Failed to migrate doc status metadata/error_msg fields',
        )

        await self._run_migration(
            self._create_pagination_indexes(),
            'pagination_indexes',
            'Failed to create pagination indexes',
        )

        await self._run_migration(
            self._migrate_create_full_entities_relations_tables(),
            'full_entities_relations_tables',
            'Failed to create full entities/relations tables',
        )

        await self._run_migration(
            self._migrate_entity_aliases_schema(),
            'entity_aliases_schema',
            'Failed to migrate entity aliases schema',
        )

        # Log migration summary
        if self._migration_failures:
            logger.warning(
                f'PostgreSQL, {len(self._migration_failures)} migration(s) failed: '
                f'{[f[0] for f in self._migration_failures]}'
            )
        else:
            logger.debug('PostgreSQL, All migrations completed successfully')

    async def _migrate_create_full_entities_relations_tables(self):
        """Create LIGHTRAG_FULL_ENTITIES and LIGHTRAG_FULL_RELATIONS tables if they don't exist"""
        tables_to_check = [
            {
                'name': 'LIGHTRAG_FULL_ENTITIES',
                'ddl': TABLES['LIGHTRAG_FULL_ENTITIES']['ddl'],
                'description': 'Full entities storage table',
            },
            {
                'name': 'LIGHTRAG_FULL_RELATIONS',
                'ddl': TABLES['LIGHTRAG_FULL_RELATIONS']['ddl'],
                'description': 'Full relations storage table',
            },
        ]

        for table_info in tables_to_check:
            table_name = table_info['name']
            try:
                # Check if table exists
                check_table_sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = $1
                AND table_schema = 'public'
                """
                params = {'table_name': table_name.lower()}
                table_exists = await self.query(check_table_sql, list(params.values()))

                if not table_exists:
                    logger.info(f'Creating table {table_name}')
                    await self.execute(table_info['ddl'])
                    logger.info(f'Successfully created {table_info["description"]}: {table_name}')

                    # Create basic indexes for the new table
                    try:
                        # Create index for id column
                        index_name = f'idx_{table_name.lower()}_id'
                        create_index_sql = f'CREATE INDEX {index_name} ON {table_name}(id)'
                        await self.execute(create_index_sql)
                        logger.info(f'Created index {index_name} on table {table_name}')

                        # Create composite index for (workspace, id) columns
                        composite_index_name = f'idx_{table_name.lower()}_workspace_id'
                        create_composite_index_sql = (
                            f'CREATE INDEX {composite_index_name} ON {table_name}(workspace, id)'
                        )
                        await self.execute(create_composite_index_sql)
                        logger.info(f'Created composite index {composite_index_name} on table {table_name}')

                    except Exception as e:
                        logger.warning(f'Failed to create indexes for table {table_name}: {e}')

                else:
                    logger.debug(f'Table {table_name} already exists')

            except Exception as e:
                logger.error(f'Failed to create table {table_name}: {e}')

    async def _migrate_entity_aliases_schema(self):
        """Migrate LIGHTRAG_ENTITY_ALIASES table to add update_time column, canonical index, and confidence constraint"""
        table_name = 'LIGHTRAG_ENTITY_ALIASES'

        # Check if table exists first
        check_table_sql = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = $1
        AND table_schema = 'public'
        """
        table_exists = await self.query(check_table_sql, [table_name.lower()])
        if not table_exists:
            logger.debug(f'Table {table_name} does not exist yet, skipping migration')
            return

        # 1. Add update_time column if it doesn't exist
        check_column_sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
        AND column_name = 'update_time'
        AND table_schema = 'public'
        """
        column_exists = await self.query(check_column_sql, [table_name.lower()])

        if not column_exists:
            try:
                # Three-step migration to add update_time column:
                # 1. Add column WITHOUT default - avoids full table rewrite on large tables
                # 2. Backfill existing rows with create_time values
                # 3. Set DEFAULT for future inserts
                # Note: There's a tiny race window between steps 1-3 where concurrent
                # inserts could get NULL. This is acceptable for this migration use case.
                #
                # Step 1: Add column WITHOUT default (existing rows get NULL)
                add_column_sql = f"""
                ALTER TABLE {table_name}
                ADD COLUMN update_time TIMESTAMP(0)
                """
                await self.execute(add_column_sql)
                logger.info(f'PostgreSQL, Added update_time column to {table_name}')

                # Step 2: Set existing rows' update_time to their create_time
                update_sql = f"""
                UPDATE {table_name}
                SET update_time = create_time
                WHERE update_time IS NULL
                """
                await self.execute(update_sql)
                logger.info(f'PostgreSQL, Initialized update_time values in {table_name}')

                # Step 3: Set default for future rows
                set_default_sql = f"""
                ALTER TABLE {table_name}
                ALTER COLUMN update_time SET DEFAULT CURRENT_TIMESTAMP
                """
                await self.execute(set_default_sql)
                logger.info(f'PostgreSQL, Set default for update_time column in {table_name}')
            except Exception as e:
                logger.error(f'PostgreSQL, Failed to add update_time column to {table_name}: {e}')

        # 2. Create index on (workspace, canonical_entity) for get_aliases_for_canonical query
        index_name = 'idx_lightrag_entity_aliases_canonical'
        check_index_sql = """
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = $1
        AND indexname = $2
        """
        index_exists = await self.query(check_index_sql, [table_name.lower(), index_name])

        if not index_exists:
            try:
                create_index_sql = f"""
                CREATE INDEX {index_name}
                ON {table_name} (workspace, canonical_entity)
                """
                await self.execute(create_index_sql)
                logger.info(f'PostgreSQL, Created index {index_name} on {table_name}')
            except Exception as e:
                logger.error(f'PostgreSQL, Failed to create index {index_name}: {e}')

        # 3. Add CHECK constraint for confidence range if it doesn't exist
        constraint_name = 'confidence_range'
        check_constraint_sql = """
        SELECT constraint_name
        FROM information_schema.table_constraints
        WHERE table_name = $1
        AND constraint_name = $2
        AND constraint_type = 'CHECK'
        AND table_schema = 'public'
        """
        constraint_exists = await self.query(check_constraint_sql, [table_name.lower(), constraint_name])

        if not constraint_exists:
            try:
                add_constraint_sql = f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT {constraint_name}
                CHECK (confidence >= 0 AND confidence <= 1)
                """
                await self.execute(add_constraint_sql)
                logger.info(f'PostgreSQL, Added CHECK constraint {constraint_name} to {table_name}')
            except Exception as e:
                logger.warning(f'PostgreSQL, Failed to add CHECK constraint {constraint_name} to {table_name}: {e}')

    async def _create_pagination_indexes(self):
        """Create indexes to optimize pagination queries for LIGHTRAG_DOC_STATUS"""
        indexes = [
            {
                'name': 'idx_lightrag_doc_status_workspace_status_updated_at',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_status_updated_at ON LIGHTRAG_DOC_STATUS (workspace, status, updated_at DESC)',
                'description': 'Composite index for workspace + status + updated_at pagination',
            },
            {
                'name': 'idx_lightrag_doc_status_workspace_status_created_at',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_status_created_at ON LIGHTRAG_DOC_STATUS (workspace, status, created_at DESC)',
                'description': 'Composite index for workspace + status + created_at pagination',
            },
            {
                'name': 'idx_lightrag_doc_status_workspace_updated_at',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_updated_at ON LIGHTRAG_DOC_STATUS (workspace, updated_at DESC)',
                'description': 'Index for workspace + updated_at pagination (all statuses)',
            },
            {
                'name': 'idx_lightrag_doc_status_workspace_created_at',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_created_at ON LIGHTRAG_DOC_STATUS (workspace, created_at DESC)',
                'description': 'Index for workspace + created_at pagination (all statuses)',
            },
            {
                'name': 'idx_lightrag_doc_status_workspace_id',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_id ON LIGHTRAG_DOC_STATUS (workspace, id)',
                'description': 'Index for workspace + id sorting',
            },
            {
                'name': 'idx_lightrag_doc_status_workspace_file_path',
                'sql': 'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_lightrag_doc_status_workspace_file_path ON LIGHTRAG_DOC_STATUS (workspace, file_path)',
                'description': 'Index for workspace + file_path sorting',
            },
        ]

        for index in indexes:
            try:
                # Check if index already exists
                check_sql = """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'lightrag_doc_status'
                AND indexname = $1
                """

                params = {'indexname': index['name']}
                existing = await self.query(check_sql, list(params.values()))

                if not existing:
                    logger.info(f'Creating pagination index: {index["description"]}')
                    await self.execute(index['sql'])
                    logger.info(f'Successfully created index: {index["name"]}')
                else:
                    logger.debug(f'Index already exists: {index["name"]}')

            except Exception as e:
                logger.warning(f'Failed to create index {index["name"]}: {e}')

    async def _create_vector_indexes(self):
        vdb_tables = [
            'LIGHTRAG_VDB_CHUNKS',
            'LIGHTRAG_VDB_ENTITY',
            'LIGHTRAG_VDB_RELATION',
        ]

        create_sql = {
            'HNSW': f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING hnsw (content_vector vector_cosine_ops)
                WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef})
            """,
            'IVFFLAT': f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING ivfflat (content_vector vector_cosine_ops)
                WITH (lists = {self.ivfflat_lists})
            """,
            'VCHORDRQ': f"""
                CREATE INDEX {{vector_index_name}}
                ON {{k}} USING vchordrq (content_vector vector_cosine_ops)
                {f'WITH (options = $${self.vchordrq_build_options}$$)' if self.vchordrq_build_options else ''}
            """,
        }

        embedding_dim = int(os.environ.get('EMBEDDING_DIM', 1024))
        vector_index_type = self.vector_index_type or 'HNSW'
        vit_lower = vector_index_type.lower()
        for k in vdb_tables:
            vector_index_name = f'idx_{k.lower()}_{vit_lower}_cosine'
            check_vector_index_sql = f"""
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = '{vector_index_name}' AND tablename = '{k.lower()}'
                """
            try:
                vector_index_exists = await self.query(check_vector_index_sql)
                if not vector_index_exists:
                    # Only set vector dimension when index doesn't exist
                    alter_sql = f'ALTER TABLE {k} ALTER COLUMN content_vector TYPE VECTOR({embedding_dim})'
                    await self.execute(alter_sql)
                    logger.debug(f'Ensured vector dimension for {k}')
                    logger.info(f'Creating {self.vector_index_type} index {vector_index_name} on table {k}')
                    await self.execute(create_sql[vector_index_type].format(vector_index_name=vector_index_name, k=k))
                    logger.info(f'Successfully created vector index {vector_index_name} on table {k}')
                else:
                    logger.info(f'{vector_index_type} vector index {vector_index_name} already exists on table {k}')
            except Exception as e:
                logger.error(f'Failed to create vector index on table {k}, Got: {e}')

    @overload
    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        multirows: Literal[True],
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        multirows: Literal[False] = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> dict[str, Any] | None: ...

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        multirows: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        async def _operation(connection: Connection | PoolConnectionProxy) -> Any:
            prepared_params = tuple(params) if params else ()
            if prepared_params:
                rows = await connection.fetch(sql, *prepared_params)
            else:
                rows = await connection.fetch(sql)

            if multirows:
                if rows:
                    columns = list(rows[0].keys())
                    return [dict(zip(columns, row, strict=False)) for row in rows]
                return []

            if rows:
                columns = rows[0].keys()
                return dict(zip(columns, rows[0], strict=False))
            return None

        try:
            return await self._run_with_retry(_operation, with_age=with_age, graph_name=graph_name)
        except Exception as e:
            logger.error(f'PostgreSQL database, error:{e}')
            raise

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | None = None,
        upsert: bool = False,
        ignore_if_exists: bool = False,
        with_age: bool = False,
        graph_name: str | None = None,
    ):
        async def _operation(connection: Connection | PoolConnectionProxy) -> Any:
            prepared_values = tuple(data.values()) if data else ()
            try:
                if not data:
                    return await connection.execute(sql)
                return await connection.execute(sql, *prepared_values)
            except (
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
                asyncpg.exceptions.InvalidSchemaNameError,
            ) as e:
                if ignore_if_exists:
                    logger.debug('PostgreSQL, ignoring duplicate during execute: %r', e)
                    return None
                if upsert:
                    logger.info(
                        'PostgreSQL, duplicate detected but treated as upsert success: %r',
                        e,
                    )
                    return None
                raise

        try:
            await self._run_with_retry(_operation, with_age=with_age, graph_name=graph_name)
        except Exception as e:
            logger.error(f'PostgreSQL database,\nsql:{sql},\ndata:{data},\nerror:{e}')
            raise

    async def executemany(
        self,
        sql: str,
        data_list: list[tuple],
        batch_size: int = 500,
    ) -> None:
        """Execute SQL with multiple parameter sets using asyncpg's executemany.

        This is significantly faster than calling execute() in a loop because it
        reduces database round-trips by batching multiple rows in a single operation.

        Args:
            sql: The SQL statement with positional parameters ($1, $2, etc.)
            data_list: List of tuples, each containing parameters for one row
            batch_size: Number of rows to process per batch (default 500)
        """
        if not data_list:
            return

        async def _operation(connection: Connection | PoolConnectionProxy) -> None:
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i : i + batch_size]
                await connection.executemany(sql, batch)

        try:
            await self._run_with_retry(_operation)
            logger.debug(f'PostgreSQL executemany: inserted {len(data_list)} rows in batches of {batch_size}')
        except Exception as e:
            logger.error(f'PostgreSQL executemany error: {e}, sql: {sql[:100]}...')
            raise

    async def full_text_search(
        self,
        query: str,
        workspace: str | None = None,
        limit: int = 10,
        language: str = 'english',
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Perform BM25-style full-text search on document chunks.

        Uses PostgreSQL's native full-text search with ts_rank for ranking.
        Results are cached by default for performance.

        Args:
            query: Search query string
            workspace: Optional workspace filter (uses self.workspace if not provided)
            limit: Maximum number of results to return
            language: Language for text search configuration (default: english)
            use_cache: Whether to use result caching (default: True)

        Returns:
            List of matching chunks with content, score, metadata, and s3_key
        """
        from lightrag.cache.fts_cache import get_cached_fts_results, store_fts_results

        ws = workspace or self.workspace

        # Check cache first
        if use_cache:
            cached_results = await get_cached_fts_results(query, ws, limit, language)
            if cached_results is not None:
                return cached_results

        # Use plainto_tsquery for simple query parsing (handles plain text queries)
        # websearch_to_tsquery could be used for more advanced syntax
        sql = f"""
            SELECT
                id,
                full_doc_id,
                chunk_order_index,
                tokens,
                content,
                file_path,
                s3_key,
                char_start,
                char_end,
                ts_rank(
                    to_tsvector('{language}', content),
                    plainto_tsquery('{language}', $1)
                ) AS score
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $2
              AND to_tsvector('{language}', content) @@ plainto_tsquery('{language}', $1)
            ORDER BY score DESC
            LIMIT $3
        """

        results = await self.query(sql, [query, ws, limit], multirows=True)
        results = results if results else []

        # Store in cache
        if use_cache and results:
            await store_fts_results(query, ws, limit, language, results)

        return results


class ClientManager:
    _db_instance: ClassVar[PostgreSQLDB | None] = None
    _ref_count: ClassVar[int] = 0
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @staticmethod
    def get_config() -> dict[str, Any]:
        config = configparser.ConfigParser()
        config.read('config.ini', 'utf-8')

        return {
            'host': os.environ.get(
                'POSTGRES_HOST',
                config.get('postgres', 'host', fallback='localhost'),
            ),
            'port': os.environ.get('POSTGRES_PORT', config.get('postgres', 'port', fallback=5432)),
            'user': os.environ.get('POSTGRES_USER', config.get('postgres', 'user', fallback='postgres')),
            'password': os.environ.get(
                'POSTGRES_PASSWORD',
                config.get('postgres', 'password', fallback=None),
            ),
            'database': os.environ.get(
                'POSTGRES_DATABASE',
                config.get('postgres', 'database', fallback='postgres'),
            ),
            'workspace': os.environ.get(
                'POSTGRES_WORKSPACE',
                config.get('postgres', 'workspace', fallback=None),
            ),
            'max_connections': int(
                os.environ.get(
                    'POSTGRES_MAX_CONNECTIONS',
                    config.get('postgres', 'max_connections', fallback=50),
                )
            ),
            'min_connections': int(
                os.environ.get(
                    'POSTGRES_MIN_CONNECTIONS',
                    config.get('postgres', 'min_connections', fallback=5),
                )
            ),
            # SSL configuration
            'ssl_mode': os.environ.get(
                'POSTGRES_SSL_MODE',
                config.get('postgres', 'ssl_mode', fallback=None),
            ),
            'ssl_cert': os.environ.get(
                'POSTGRES_SSL_CERT',
                config.get('postgres', 'ssl_cert', fallback=None),
            ),
            'ssl_key': os.environ.get(
                'POSTGRES_SSL_KEY',
                config.get('postgres', 'ssl_key', fallback=None),
            ),
            'ssl_root_cert': os.environ.get(
                'POSTGRES_SSL_ROOT_CERT',
                config.get('postgres', 'ssl_root_cert', fallback=None),
            ),
            'ssl_crl': os.environ.get(
                'POSTGRES_SSL_CRL',
                config.get('postgres', 'ssl_crl', fallback=None),
            ),
            'vector_index_type': os.environ.get(
                'POSTGRES_VECTOR_INDEX_TYPE',
                config.get('postgres', 'vector_index_type', fallback='HNSW'),
            ),
            'hnsw_m': int(
                os.environ.get(
                    'POSTGRES_HNSW_M',
                    config.get('postgres', 'hnsw_m', fallback='16'),
                )
            ),
            'hnsw_ef': int(
                os.environ.get(
                    'POSTGRES_HNSW_EF',
                    config.get('postgres', 'hnsw_ef', fallback='64'),
                )
            ),
            'hnsw_ef_search': int(
                os.environ.get(
                    'POSTGRES_HNSW_EF_SEARCH',
                    config.get('postgres', 'hnsw_ef_search', fallback='200'),
                )
            ),
            'ivfflat_lists': int(
                os.environ.get(
                    'POSTGRES_IVFFLAT_LISTS',
                    config.get('postgres', 'ivfflat_lists', fallback='100'),
                )
            ),
            'vchordrq_build_options': os.environ.get(
                'POSTGRES_VCHORDRQ_BUILD_OPTIONS',
                config.get('postgres', 'vchordrq_build_options', fallback=''),
            ),
            'vchordrq_probes': os.environ.get(
                'POSTGRES_VCHORDRQ_PROBES',
                config.get('postgres', 'vchordrq_probes', fallback=''),
            ),
            'vchordrq_epsilon': float(
                os.environ.get(
                    'POSTGRES_VCHORDRQ_EPSILON',
                    config.get('postgres', 'vchordrq_epsilon', fallback='1.9'),
                )
            ),
            # Server settings for Supabase
            'server_settings': os.environ.get(
                'POSTGRES_SERVER_SETTINGS',
                config.get('postgres', 'server_options', fallback=None),
            ),
            'statement_cache_size': os.environ.get(
                'POSTGRES_STATEMENT_CACHE_SIZE',
                config.get('postgres', 'statement_cache_size', fallback='500'),
            ),
            # Connection retry configuration
            'connection_retry_attempts': min(
                10,
                int(
                    os.environ.get(
                        'POSTGRES_CONNECTION_RETRIES',
                        config.get('postgres', 'connection_retries', fallback=3),
                    )
                ),
            ),
            'connection_retry_backoff': min(
                5.0,
                float(
                    os.environ.get(
                        'POSTGRES_CONNECTION_RETRY_BACKOFF',
                        config.get('postgres', 'connection_retry_backoff', fallback=0.5),
                    )
                ),
            ),
            'connection_retry_backoff_max': min(
                60.0,
                float(
                    os.environ.get(
                        'POSTGRES_CONNECTION_RETRY_BACKOFF_MAX',
                        config.get(
                            'postgres',
                            'connection_retry_backoff_max',
                            fallback=5.0,
                        ),
                    )
                ),
            ),
            'pool_close_timeout': min(
                30.0,
                float(
                    os.environ.get(
                        'POSTGRES_POOL_CLOSE_TIMEOUT',
                        config.get('postgres', 'pool_close_timeout', fallback=5.0),
                    )
                ),
            ),
        }

    @classmethod
    async def get_client(cls) -> PostgreSQLDB:
        async with cls._lock:
            if cls._db_instance is None:
                config = ClientManager.get_config()
                db = PostgreSQLDB(config)
                await db.initdb()
                await db.check_tables()
                cls._db_instance = db
                cls._ref_count = 0
            cls._ref_count += 1
            return cast(PostgreSQLDB, cls._db_instance)

    @classmethod
    async def release_client(cls, db: PostgreSQLDB):
        async with cls._lock:
            if db is not None:
                if db is cls._db_instance:
                    cls._ref_count -= 1
                    if cls._ref_count == 0:
                        if db.pool is not None:
                            await db.pool.close()
                            logger.info('Closed PostgreSQL database connection pool')
                        cls._db_instance = None
                else:
                    if db.pool is not None:
                        await db.pool.close()


@final
@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB | None = field(default=None)

    def _db_required(self) -> PostgreSQLDB:
        """Return initialized DB client or raise for clearer typing."""
        if self.db is None:
            raise RuntimeError('PostgreSQL client is not initialized')
        return self.db

    def __post_init__(self):
        self._max_batch_size = self.global_config['embedding_batch_num']

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, 'workspace') and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = 'default'

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id."""
        sql = SQL_TEMPLATES['get_by_id_' + self.namespace]
        params = {'workspace': self.workspace, 'id': id}
        db = self._db_required()
        response = await db.query(sql, list(params.values()))

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get('llm_cache_list', [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response['llm_cache_list'] = llm_cache_list
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            response['create_time'] = create_time
            response['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get('queryparam')
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                'return': response.get('return_value', ''),
                'cache_type': response.get('cache_type'),
                'original_prompt': response.get('original_prompt', ''),
                'chunk_id': response.get('chunk_id'),
                'queryparam': queryparam,
                'create_time': create_time,
                'update_time': create_time if update_time == 0 else update_time,
            }

        # Special handling for FULL_ENTITIES namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            # Parse entity_names JSON string back to list
            entity_names = response.get('entity_names', [])
            if isinstance(entity_names, str):
                try:
                    entity_names = json.loads(entity_names)
                except json.JSONDecodeError:
                    entity_names = []
            response['entity_names'] = entity_names
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            response['create_time'] = create_time
            response['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            # Parse relation_pairs JSON string back to list
            relation_pairs = response.get('relation_pairs', [])
            if isinstance(relation_pairs, str):
                try:
                    relation_pairs = json.loads(relation_pairs)
                except json.JSONDecodeError:
                    relation_pairs = []
            response['relation_pairs'] = relation_pairs
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            response['create_time'] = create_time
            response['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get('chunk_ids', [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response['chunk_ids'] = chunk_ids
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            response['create_time'] = create_time
            response['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if response and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get('chunk_ids', [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response['chunk_ids'] = chunk_ids
            create_time = response.get('create_time', 0)
            update_time = response.get('update_time', 0)
            response['create_time'] = create_time
            response['update_time'] = create_time if update_time == 0 else update_time

        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""
        if not ids:
            return []

        sql = SQL_TEMPLATES['get_by_ids_' + self.namespace]
        params = {'workspace': self.workspace, 'ids': ids}
        db = self._db_required()
        results = await db.query(sql, list(params.values()), multirows=True)

        def _order_results(
            rows: list[dict[str, Any]] | None,
        ) -> list[dict[str, Any]]:
            """Preserve the caller requested ordering for bulk id lookups."""
            if not rows:
                return [{} for _ in ids]

            id_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get('id')
                if row_id is not None:
                    id_map[str(row_id)] = row

            ordered: list[dict[str, Any]] = []
            for requested_id in ids:
                ordered.append(id_map.get(str(requested_id), {}))
            return ordered

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list for each result
            for result in results:
                llm_cache_list = result.get('llm_cache_list', [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result['llm_cache_list'] = llm_cache_list
                create_time = result.get('create_time', 0)
                update_time = result.get('update_time', 0)
                result['create_time'] = create_time
                result['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            processed_results = []
            for row in results:
                create_time = row.get('create_time', 0)
                update_time = row.get('update_time', 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get('queryparam')
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    'return': row.get('return_value', ''),
                    'cache_type': row.get('cache_type'),
                    'original_prompt': row.get('original_prompt', ''),
                    'chunk_id': row.get('chunk_id'),
                    'queryparam': queryparam,
                    'create_time': create_time,
                    'update_time': create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)
            return _order_results(processed_results)

        # Special handling for FULL_ENTITIES namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            for result in results:
                # Parse entity_names JSON string back to list
                entity_names = result.get('entity_names', [])
                if isinstance(entity_names, str):
                    try:
                        entity_names = json.loads(entity_names)
                    except json.JSONDecodeError:
                        entity_names = []
                result['entity_names'] = entity_names
                create_time = result.get('create_time', 0)
                update_time = result.get('update_time', 0)
                result['create_time'] = create_time
                result['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for FULL_RELATIONS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            for result in results:
                # Parse relation_pairs JSON string back to list
                relation_pairs = result.get('relation_pairs', [])
                if isinstance(relation_pairs, str):
                    try:
                        relation_pairs = json.loads(relation_pairs)
                    except json.JSONDecodeError:
                        relation_pairs = []
                result['relation_pairs'] = relation_pairs
                create_time = result.get('create_time', 0)
                update_time = result.get('update_time', 0)
                result['create_time'] = create_time
                result['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for ENTITY_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get('chunk_ids', [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result['chunk_ids'] = chunk_ids
                create_time = result.get('create_time', 0)
                update_time = result.get('update_time', 0)
                result['create_time'] = create_time
                result['update_time'] = create_time if update_time == 0 else update_time

        # Special handling for RELATION_CHUNKS namespace
        if results and is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            for result in results:
                # Parse chunk_ids JSON string back to list
                chunk_ids = result.get('chunk_ids', [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                result['chunk_ids'] = chunk_ids
                create_time = result.get('create_time', 0)
                update_time = result.get('update_time', 0)
                result['create_time'] = create_time
                result['update_time'] = create_time if update_time == 0 else update_time

        return _order_results(results)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()

        db = self._db_required()
        table_name = namespace_to_table_name(self.namespace)
        sql = f'SELECT id FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'
        params = {'workspace': self.workspace, 'ids': list(keys)}
        try:
            res = await db.query(sql, list(params.values()), multirows=True)
            exist_keys = [key['id'] for key in res] if res else []
            new_keys = {s for s in keys if s not in exist_keys}
            return new_keys
        except Exception as e:
            logger.error(f'[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}')
            raise

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f'[{self.workspace}] Inserting {len(data)} to {self.namespace}')
        if not data:
            return
        db = self._db_required()

        # Get current UTC time and convert to naive datetime for database storage
        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        db = self._db_required()

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            upsert_sql = SQL_TEMPLATES['upsert_text_chunk']
            # Collect all rows as tuples for batch insert
            batch_data = [
                (
                    self.workspace,
                    k,
                    v['tokens'],
                    v['chunk_order_index'],
                    v['full_doc_id'],
                    v['content'],
                    v['file_path'],
                    v.get('s3_key'),  # S3 key for document source
                    v.get('char_start'),  # Character offset start in source document
                    v.get('char_end'),  # Character offset end in source document
                    json.dumps(v.get('llm_cache_list', [])),
                    current_time,
                    current_time,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            upsert_sql = SQL_TEMPLATES['upsert_doc_full']
            batch_data = [
                (
                    k,
                    v['content'],
                    v.get('file_path', ''),  # Map file_path to doc_name
                    self.workspace,
                    v.get('s3_key'),  # S3 key for document source
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            upsert_sql = SQL_TEMPLATES['upsert_llm_response_cache']
            batch_data = [
                (
                    self.workspace,
                    k,  # Use flattened key as id
                    v['original_prompt'],
                    v['return'],
                    v.get('chunk_id'),
                    v.get('cache_type', 'extract'),
                    json.dumps(v.get('queryparam')) if v.get('queryparam') else None,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            upsert_sql = SQL_TEMPLATES['upsert_full_entities']
            batch_data = [
                (
                    self.workspace,
                    k,
                    json.dumps(v['entity_names']),
                    v['count'],
                    current_time,
                    current_time,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            upsert_sql = SQL_TEMPLATES['upsert_full_relations']
            batch_data = [
                (
                    self.workspace,
                    k,
                    json.dumps(v['relation_pairs']),
                    v['count'],
                    current_time,
                    current_time,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            upsert_sql = SQL_TEMPLATES['upsert_entity_chunks']
            batch_data = [
                (
                    self.workspace,
                    k,
                    json.dumps(v['chunk_ids']),
                    v['count'],
                    current_time,
                    current_time,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

        elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            upsert_sql = SQL_TEMPLATES['upsert_relation_chunks']
            batch_data = [
                (
                    self.workspace,
                    k,
                    json.dumps(v['chunk_ids']),
                    v['count'],
                    current_time,
                    current_time,
                )
                for k, v in data.items()
            ]
            await db.executemany(upsert_sql, batch_data)

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}')
            return True

        sql = f'SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=$1 LIMIT 1) as has_data'

        try:
            db = self._db_required()
            result = await db.query(sql, [self.workspace])
            return not result.get('has_data', False) if result else True
        except Exception as e:
            logger.error(f'[{self.workspace}] Error checking if storage is empty: {e}')
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for deletion: {self.namespace}')
            return

        delete_sql = f'DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'

        try:
            db = self._db_required()
            await db.execute(delete_sql, {'workspace': self.workspace, 'ids': ids})
            logger.debug(f'[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error while deleting records from {self.namespace}: {e}')

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    'status': 'error',
                    'message': f'Unknown namespace: {self.namespace}',
                }

            drop_sql = SQL_TEMPLATES['drop_specifiy_table_workspace'].format(table_name=table_name)
            db = self._db_required()
            await db.execute(drop_sql, {'workspace': self.workspace})
            return {'status': 'success', 'message': 'data dropped'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def update_s3_key_by_doc_id(self, full_doc_id: str, s3_key: str, archive_url: str | None = None) -> int:
        """Update s3_key for all chunks of a document after archiving.

        This method is called after a document is moved from S3 staging to archive,
        to update the database chunks with the new archive location.

        Args:
            full_doc_id: Document ID to update
            s3_key: Archive S3 key (e.g., 'archive/default/doc123/file.pdf')
            archive_url: Optional full S3 URL to update file_path

        Returns:
            Number of rows updated
        """
        if archive_url:
            # Update both s3_key and file_path
            sql = """
                UPDATE LIGHTRAG_DOC_CHUNKS
                SET s3_key = $1, file_path = $2, update_time = CURRENT_TIMESTAMP
                WHERE workspace = $3 AND full_doc_id = $4
            """
            params = {
                's3_key': s3_key,
                'file_path': archive_url,
                'workspace': self.workspace,
                'full_doc_id': full_doc_id,
            }
        else:
            # Update only s3_key
            sql = """
                UPDATE LIGHTRAG_DOC_CHUNKS
                SET s3_key = $1, update_time = CURRENT_TIMESTAMP
                WHERE workspace = $2 AND full_doc_id = $3
            """
            params = {
                's3_key': s3_key,
                'workspace': self.workspace,
                'full_doc_id': full_doc_id,
            }

        db = self._db_required()
        result = await db.execute(sql, params)

        # Parse the number of rows updated from result like "UPDATE 5"
        try:
            count = int(result.split()[-1]) if result else 0
        except (ValueError, AttributeError, IndexError):
            count = 0
        logger.debug(f'[{self.workspace}] Updated {count} chunks with s3_key for doc {full_doc_id}')
        return count


@final
@dataclass
class PGVectorStorage(BaseVectorStorage):
    db: PostgreSQLDB | None = field(default=None)

    def _db_required(self) -> PostgreSQLDB:
        if self.db is None:
            raise RuntimeError('PostgreSQL client is not initialized')
        return self.db

    def __post_init__(self):
        self._max_batch_size = self.global_config['embedding_batch_num']
        config = self.global_config.get('vector_db_storage_cls_kwargs', {})
        cosine_threshold = config.get('cosine_better_than_threshold')
        if cosine_threshold is None:
            raise ValueError('cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs')
        self.cosine_better_than_threshold = cosine_threshold

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, 'workspace') and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = 'default'

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    def _prepare_chunk_tuple(self, item: dict[str, Any], current_time: datetime.datetime) -> tuple:
        """Prepare a tuple for batch chunk upsert."""
        try:
            return (
                self.workspace,
                item['__id__'],
                item['tokens'],
                item['chunk_order_index'],
                item['full_doc_id'],
                item['content'],
                json.dumps(item['__vector__'].tolist()),
                item['file_path'],
                current_time,
                current_time,
            )
        except Exception as e:
            logger.error(f'[{self.workspace}] Error to prepare upsert,\nerror: {e}\nitem: {item}')
            raise

    def _prepare_entity_tuple(self, item: dict[str, Any], current_time: datetime.datetime) -> tuple:
        """Prepare a tuple for batch entity upsert."""
        source_id = item['source_id']
        chunk_ids = source_id.split('<SEP>') if isinstance(source_id, str) and '<SEP>' in source_id else [source_id]

        return (
            self.workspace,
            item['__id__'],
            item['entity_name'],
            item['content'],
            json.dumps(item['__vector__'].tolist()),
            chunk_ids,
            item.get('file_path'),
            current_time,
            current_time,
        )

    def _prepare_relationship_tuple(self, item: dict[str, Any], current_time: datetime.datetime) -> tuple:
        """Prepare a tuple for batch relationship upsert."""
        source_id = item['source_id']
        chunk_ids = source_id.split('<SEP>') if isinstance(source_id, str) and '<SEP>' in source_id else [source_id]

        return (
            self.workspace,
            item['__id__'],
            item['src_id'],
            item['tgt_id'],
            item['content'],
            json.dumps(item['__vector__'].tolist()),
            chunk_ids,
            item.get('file_path'),
            current_time,
            current_time,
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f'[{self.workspace}] Inserting {len(data)} to {self.namespace}')
        if not data:
            return
        db = self._db_required()

        # Get current UTC time and convert to naive datetime for database storage
        current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
        list_data = [
            {
                '__id__': k,
                **dict(v.items()),
            }
            for k, v in data.items()
        ]

        # Batch compute embeddings (already optimized)
        contents = [v['content'] for v in data.values()]
        batches = [contents[i : i + self._max_batch_size] for i in range(0, len(contents), self._max_batch_size)]
        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        # Assign embeddings to items
        for i, d in enumerate(list_data):
            d['__vector__'] = embeddings[i]

        # Prepare batch data based on namespace and execute in single batch
        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            upsert_sql = SQL_TEMPLATES['upsert_chunk']
            batch_data = [self._prepare_chunk_tuple(item, current_time) for item in list_data]
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            upsert_sql = SQL_TEMPLATES['upsert_entity']
            batch_data = [self._prepare_entity_tuple(item, current_time) for item in list_data]
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            upsert_sql = SQL_TEMPLATES['upsert_relationship']
            batch_data = [self._prepare_relationship_tuple(item, current_time) for item in list_data]
        else:
            raise ValueError(f'{self.namespace} is not supported')

        await db.executemany(upsert_sql, batch_data)

    #################### query method ###############
    async def query(self, query: str, top_k: int, query_embedding: list[float] | None = None) -> list[dict[str, Any]]:
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embeddings = await self.embedding_func([query], _priority=5)  # higher priority for query
            embedding = embeddings[0]

        embedding_string = ','.join(map(str, embedding))

        sql = SQL_TEMPLATES[self.namespace].format(embedding_string=embedding_string)
        params = {
            'workspace': self.workspace,
            'closer_than_threshold': 1 - self.cosine_better_than_threshold,
            'top_k': top_k,
        }
        db = self._db_required()
        results = await db.query(sql, params=list(params.values()), multirows=True)
        return results

    async def hybrid_search(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
        bm25_weight: float = 0.3,
        language: str = 'english',
    ) -> list[dict[str, Any]]:
        """Combine vector similarity search with BM25 full-text search using RRF.

        This hybrid approach leverages the strengths of both retrieval methods:
        - Vector search: Captures semantic similarity (good for paraphrases)
        - BM25: Captures exact keyword matches (good for names, dates, acronyms)

        Uses Reciprocal Rank Fusion (RRF) to combine the ranked results.

        Args:
            query: Search query string
            top_k: Number of final results to return
            query_embedding: Optional pre-computed query embedding
            bm25_weight: Weight for BM25 results (0.0-1.0). Higher = more BM25 influence.
                        Note: RRF naturally handles ranking, this affects how many BM25
                        results to retrieve (more weight = retrieve more BM25 results).
            language: Language for BM25 text search (default: 'english')

        Returns:
            List of chunks with content, metadata, and rrf_score
        """
        from lightrag.utils import reciprocal_rank_fusion

        # Determine how many results to fetch from each source
        # Fetch more than top_k to allow RRF to work effectively
        vector_fetch_k = int(top_k * 2)
        bm25_fetch_k = int(top_k * (1 + bm25_weight))  # Scale by weight

        # 1. Vector search (existing method)
        vector_results = await self.query(query, top_k=vector_fetch_k, query_embedding=query_embedding)

        # 2. BM25 full-text search (query LIGHTRAG_DOC_CHUNKS directly)
        bm25_sql = f"""
            SELECT
                id,
                full_doc_id,
                chunk_order_index,
                tokens,
                content,
                file_path,
                ts_rank(
                    to_tsvector('{language}', content),
                    plainto_tsquery('{language}', $1)
                ) AS bm25_score
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $2
              AND to_tsvector('{language}', content) @@ plainto_tsquery('{language}', $1)
            ORDER BY bm25_score DESC
            LIMIT $3
        """

        db = self._db_required()
        bm25_results = await db.query(
            bm25_sql,
            params=[query, self.workspace, bm25_fetch_k],
            multirows=True,
        )
        bm25_results = bm25_results if bm25_results else []

        # Mark source type for debugging
        for r in vector_results:
            r['source_type'] = 'vector'
        for r in bm25_results:
            r['source_type'] = 'bm25'

        # 3. Combine using Reciprocal Rank Fusion
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            id_key='id',
            k=60,  # Standard RRF constant
        )

        logger.debug(
            f'[{self.workspace}] Hybrid search: {len(vector_results)} vector + '
            f'{len(bm25_results)} BM25 → {len(fused_results[:top_k])} fused'
        )

        return fused_results[:top_k]

    async def get_entity_linked_chunk_ids(
        self,
        keywords: list[str],
        top_k_per_keyword: int = 5,
    ) -> set[str]:
        """Get chunk IDs linked to entities matching the given keywords.

        This is used for entity-aware retrieval boosting: when a query mentions
        specific entities (like "SARP"), we want to prioritize chunks that are
        linked to those entities in the knowledge graph.

        Args:
            keywords: List of keywords to match against entity names
            top_k_per_keyword: Max entities to match per keyword (controls expansion)

        Returns:
            Set of chunk IDs linked to matched entities
        """
        if not keywords:
            return set()

        db = self._db_required()
        all_chunk_ids: set[str] = set()

        # Search for entities matching each keyword
        # Use similarity search on entity names (case-insensitive, partial match)
        entity_search_sql = """
            SELECT entity_name, chunk_ids
            FROM LIGHTRAG_VDB_ENTITY
            WHERE workspace = $1
              AND (
                  entity_name ILIKE $2
                  OR entity_name ILIKE $3
              )
            LIMIT $4
        """

        for keyword in keywords:
            try:
                # Search for exact match and partial match
                exact_pattern = keyword
                partial_pattern = f'%{keyword}%'

                results = await db.query(
                    entity_search_sql,
                    params=[self.workspace, exact_pattern, partial_pattern, top_k_per_keyword],
                    multirows=True,
                )

                if results:
                    for row in results:
                        chunk_ids = row.get('chunk_ids', [])
                        if chunk_ids:
                            # chunk_ids is stored as a PostgreSQL array
                            if isinstance(chunk_ids, list):
                                all_chunk_ids.update(chunk_ids)
                            elif isinstance(chunk_ids, str):
                                # Handle JSON string format if needed
                                try:
                                    parsed = json.loads(chunk_ids)
                                    all_chunk_ids.update(parsed)
                                except json.JSONDecodeError:
                                    all_chunk_ids.add(chunk_ids)

                    logger.debug(
                        f'[{self.workspace}] Entity search for "{keyword}": '
                        f'{len(results)} entities, {len(all_chunk_ids)} total chunk_ids'
                    )

            except Exception as e:
                logger.warning(f'[{self.workspace}] Entity search error for "{keyword}": {e}')
                continue

        return all_chunk_ids

    async def hybrid_search_with_entity_boost(
        self,
        query: str,
        top_k: int,
        entity_keywords: list[str] | None = None,
        query_embedding: list[float] | None = None,
        bm25_weight: float = 0.3,
        entity_boost: float = 1.5,
        language: str = 'english',
    ) -> list[dict[str, Any]]:
        """Hybrid search with entity-aware boosting for improved precision.

        Extends hybrid_search by boosting chunks that are linked to entities
        mentioned in the query. This helps retrieve the RIGHT documents when
        multiple similar documents exist (e.g., different "lessons learned" docs).

        Args:
            query: Search query string
            top_k: Number of final results to return
            entity_keywords: Keywords to match against entity names for boosting
            query_embedding: Optional pre-computed query embedding
            bm25_weight: Weight for BM25 results (0.0-1.0)
            entity_boost: Multiplier for chunks linked to query entities (default: 1.5)
            language: Language for BM25 text search

        Returns:
            List of chunks with content, metadata, and boosted rrf_score
        """
        from lightrag.utils import reciprocal_rank_fusion

        # Get entity-linked chunk IDs for boosting
        entity_chunk_ids: set[str] = set()
        if entity_keywords:
            entity_chunk_ids = await self.get_entity_linked_chunk_ids(entity_keywords)
            if entity_chunk_ids:
                logger.info(
                    f'[{self.workspace}] Entity boost: {len(entity_chunk_ids)} chunks '
                    f'linked to entities matching {entity_keywords}'
                )

        # Fetch more results to allow for entity-based reranking
        vector_fetch_k = int(top_k * 2.5)  # Increased from 2 to allow more entity matches
        bm25_fetch_k = int(top_k * (1.5 + bm25_weight))

        # 1. Vector search
        vector_results = await self.query(query, top_k=vector_fetch_k, query_embedding=query_embedding)

        # 2. BM25 full-text search
        bm25_sql = f"""
            SELECT
                id,
                full_doc_id,
                chunk_order_index,
                tokens,
                content,
                file_path,
                ts_rank(
                    to_tsvector('{language}', content),
                    plainto_tsquery('{language}', $1)
                ) AS bm25_score
            FROM LIGHTRAG_DOC_CHUNKS
            WHERE workspace = $2
              AND to_tsvector('{language}', content) @@ plainto_tsquery('{language}', $1)
            ORDER BY bm25_score DESC
            LIMIT $3
        """

        db = self._db_required()
        bm25_results = await db.query(
            bm25_sql,
            params=[query, self.workspace, bm25_fetch_k],
            multirows=True,
        )
        bm25_results = bm25_results if bm25_results else []

        # Mark source type
        for r in vector_results:
            r['source_type'] = 'vector'
        for r in bm25_results:
            r['source_type'] = 'bm25'

        # 3. Combine using RRF
        fused_results = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            id_key='id',
            k=60,
        )

        # 4. Apply entity boost to chunks linked to query entities
        if entity_chunk_ids:
            boosted_count = 0
            for result in fused_results:
                chunk_id = result.get('id', '')
                if chunk_id in entity_chunk_ids:
                    original_score = result.get('rrf_score', 0)
                    result['rrf_score'] = original_score * entity_boost
                    result['entity_boosted'] = True
                    boosted_count += 1

            # Re-sort by boosted scores
            fused_results = sorted(fused_results, key=lambda x: x.get('rrf_score', 0), reverse=True)

            logger.info(
                f'[{self.workspace}] Entity boost applied to {boosted_count}/{len(fused_results)} chunks'
            )

        logger.debug(
            f'[{self.workspace}] Hybrid+entity search: {len(vector_results)} vector + '
            f'{len(bm25_results)} BM25 + entity boost → {len(fused_results[:top_k])} results'
        )

        return fused_results[:top_k]

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage.

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for vector deletion: {self.namespace}')
            return

        delete_sql = f'DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'

        try:
            db = self._db_required()
            await db.execute(delete_sql, {'workspace': self.workspace, 'ids': ids})
            logger.debug(f'[{self.workspace}] Successfully deleted {len(ids)} vectors from {self.namespace}')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}')

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage.

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Construct SQL to delete the entity
            delete_sql = """DELETE FROM LIGHTRAG_VDB_ENTITY
                            WHERE workspace=$1 AND entity_name=$2"""

            db = self._db_required()
            await db.execute(delete_sql, {'workspace': self.workspace, 'entity_name': entity_name})
            logger.debug(f'[{self.workspace}] Successfully deleted entity {entity_name}')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error deleting entity {entity_name}: {e}')

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity.

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Delete relations where the entity is either the source or target
            delete_sql = """DELETE FROM LIGHTRAG_VDB_RELATION
                            WHERE workspace=$1 AND (source_id=$2 OR target_id=$2)"""

            db = self._db_required()
            await db.execute(delete_sql, {'workspace': self.workspace, 'entity_name': entity_name})
            logger.debug(f'[{self.workspace}] Successfully deleted relations for entity {entity_name}')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error deleting relations for entity {entity_name}: {e}')

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for ID lookup: {self.namespace}')
            return None

        query = f'SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at FROM {table_name} WHERE workspace=$1 AND id=$2'
        params = {'workspace': self.workspace, 'id': id}

        try:
            db = self._db_required()
            result = await db.query(query, list(params.values()))
            if result:
                return dict(result)
            return None
        except Exception as e:
            logger.error(f'[{self.workspace}] Error retrieving vector data for ID {id}: {e}')
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for IDs lookup: {self.namespace}')
            return []

        # Use parameterized array for security and performance
        query = f'SELECT *, EXTRACT(EPOCH FROM create_time)::BIGINT as created_at FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'
        params = [self.workspace, list(ids)]

        try:
            db = self._db_required()
            results = await db.query(query, params, multirows=True)
            if not results:
                return []

            # Preserve caller requested ordering while normalizing asyncpg rows to dicts.
            id_map: dict[str, dict[str, Any]] = {}
            for record in results:
                if record is None:
                    continue
                record_dict = dict(record)
                row_id = record_dict.get('id')
                if row_id is not None:
                    id_map[str(row_id)] = record_dict

            ordered_results: list[dict[str, Any]] = []
            for requested_id in ids:
                ordered_results.append(id_map.get(str(requested_id), {}))
            return ordered_results
        except Exception as e:
            logger.error(f'[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}')
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for vector lookup: {self.namespace}')
            return {}

        # Use parameterized array for security and performance
        query = f'SELECT id, content_vector FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'
        params = [self.workspace, list(ids)]

        try:
            db = self._db_required()
            results = await db.query(query, params, multirows=True)
            vectors_dict = {}

            for result in results:
                if result and 'content_vector' in result and 'id' in result:
                    try:
                        # Parse JSON string to get vector as list of floats
                        vector_data = json.loads(result['content_vector'])
                        if isinstance(vector_data, list):
                            vectors_dict[result['id']] = vector_data
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning(f'[{self.workspace}] Failed to parse vector data for ID {result["id"]}: {e}')

            return vectors_dict
        except Exception as e:
            logger.error(f'[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}')
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    'status': 'error',
                    'message': f'Unknown namespace: {self.namespace}',
                }

            drop_sql = SQL_TEMPLATES['drop_specifiy_table_workspace'].format(table_name=table_name)
            db = self._db_required()
            await db.execute(drop_sql, {'workspace': self.workspace})
            return {'status': 'success', 'message': 'data dropped'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


@final
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    db: PostgreSQLDB | None = field(default=None)

    def _db_required(self) -> PostgreSQLDB:
        if self.db is None:
            raise RuntimeError('PostgreSQL client is not initialized')
        return self.db

    def _format_datetime_with_timezone(self, dt):
        """Convert datetime to ISO format string with timezone info"""
        if dt is None:
            return None
        # If no timezone info, assume it's UTC time (as stored in database)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If datetime already has timezone info, keep it as is
        return dt.isoformat()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, 'workspace') and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = 'default'

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()

        db = self._db_required()
        table_name = namespace_to_table_name(self.namespace)
        sql = f'SELECT id FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'
        params = {'workspace': self.workspace, 'ids': list(keys)}
        try:
            res = await db.query(sql, list(params.values()), multirows=True)
            exist_keys = [key['id'] for key in res] if res else []
            new_keys = {s for s in keys if s not in exist_keys}
            # print(f"keys: {keys}")
            # print(f"new_keys: {new_keys}")
            return new_keys
        except Exception as e:
            logger.error(f'[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}')
            raise

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        sql = 'select * from LIGHTRAG_DOC_STATUS where workspace=$1 and id=$2'
        params = {'workspace': self.workspace, 'id': id}
        db = self._db_required()
        result = await db.query(sql, list(params.values()), multirows=True)
        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]['created_at'])
            updated_at = self._format_datetime_with_timezone(result[0]['updated_at'])

            return {
                'content_length': result[0]['content_length'],
                'content_summary': result[0]['content_summary'],
                'status': result[0]['status'],
                'chunks_count': result[0]['chunks_count'],
                'created_at': created_at,
                'updated_at': updated_at,
                'file_path': result[0]['file_path'],
                'chunks_list': chunks_list,
                'metadata': metadata,
                'error_msg': result[0].get('error_msg'),
                'track_id': result[0].get('track_id'),
            }

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs."""
        if not ids:
            return []

        db = self._db_required()
        sql = 'SELECT * FROM LIGHTRAG_DOC_STATUS WHERE workspace=$1 AND id = ANY($2)'
        params = {'workspace': self.workspace, 'ids': ids}

        results = await db.query(sql, list(params.values()), multirows=True)

        if not results:
            return []

        processed_map: dict[str, dict[str, Any]] = {}
        for row in results:
            # Parse chunks_list JSON string back to list
            chunks_list = row.get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(row['created_at'])
            updated_at = self._format_datetime_with_timezone(row['updated_at'])

            processed_map[str(row.get('id'))] = {
                'content_length': row['content_length'],
                'content_summary': row['content_summary'],
                'status': row['status'],
                'chunks_count': row['chunks_count'],
                'created_at': created_at,
                'updated_at': updated_at,
                'file_path': row['file_path'],
                'chunks_list': chunks_list,
                'metadata': metadata,
                'error_msg': row.get('error_msg'),
                'track_id': row.get('track_id'),
            }

        ordered_results: list[dict[str, Any]] = []
        for requested_id in ids:
            ordered_results.append(processed_map.get(str(requested_id), {}))

        return ordered_results

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        """Get document by file path

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
            Returns the same format as get_by_id method
        """
        db = self._db_required()
        sql = 'select * from LIGHTRAG_DOC_STATUS where workspace=$1 and file_path=$2'
        params = {'workspace': self.workspace, 'file_path': file_path}
        result = await db.query(sql, list(params.values()), multirows=True)

        if result is None or result == []:
            return None
        else:
            # Parse chunks_list JSON string back to list
            chunks_list = result[0].get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = result[0].get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]['created_at'])
            updated_at = self._format_datetime_with_timezone(result[0]['updated_at'])

            return {
                'content_length': result[0]['content_length'],
                'content_summary': result[0]['content_summary'],
                'status': result[0]['status'],
                'chunks_count': result[0]['chunks_count'],
                'created_at': created_at,
                'updated_at': updated_at,
                'file_path': result[0]['file_path'],
                'chunks_list': chunks_list,
                'metadata': metadata,
                'error_msg': result[0].get('error_msg'),
                'track_id': result[0].get('track_id'),
            }

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status"""
        db = self._db_required()
        sql = """SELECT status as "status", COUNT(1) as "count"
                   FROM LIGHTRAG_DOC_STATUS
                  where workspace=$1 GROUP BY STATUS
                 """
        params: dict[str, Any] = {'workspace': self.workspace}
        result = await db.query(sql, list(params.values()), multirows=True)
        counts = {}
        for doc in result:
            counts[doc['status']] = doc['count']
        return counts

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """all documents with a specific status"""
        db = self._db_required()
        sql = 'select * from LIGHTRAG_DOC_STATUS where workspace=$1 and status=$2'
        params = {'workspace': self.workspace, 'status': status.value}
        result = await db.query(sql, list(params.values()), multirows=True)

        docs_by_status = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get('file_path')
            if file_path is None:
                file_path = 'no-file-path'

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element['created_at']) or ''
            updated_at = self._format_datetime_with_timezone(element['updated_at']) or ''

            docs_by_status[element['id']] = DocProcessingStatus(
                content_summary=element['content_summary'],
                content_length=element['content_length'],
                status=element['status'],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element['chunks_count'],
                file_path=file_path,
                chunks_list=chunks_list,
                metadata=metadata,
                error_msg=element.get('error_msg'),
                track_id=element.get('track_id'),
                s3_key=element.get('s3_key'),
            )

        return docs_by_status

    async def get_docs_by_track_id(self, track_id: str) -> dict[str, DocProcessingStatus]:
        """Get all documents with a specific track_id"""
        db = self._db_required()
        sql = 'select * from LIGHTRAG_DOC_STATUS where workspace=$1 and track_id=$2'
        params = {'workspace': self.workspace, 'track_id': track_id}
        result = await db.query(sql, list(params.values()), multirows=True)

        docs_by_track_id = {}
        for element in result:
            # Parse chunks_list JSON string back to list
            chunks_list = element.get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            # Ensure metadata is a dict
            if not isinstance(metadata, dict):
                metadata = {}

            # Safe handling for file_path
            file_path = element.get('file_path')
            if file_path is None:
                file_path = 'no-file-path'

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element['created_at']) or ''
            updated_at = self._format_datetime_with_timezone(element['updated_at']) or ''

            docs_by_track_id[element['id']] = DocProcessingStatus(
                content_summary=element['content_summary'],
                content_length=element['content_length'],
                status=element['status'],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element['chunks_count'],
                file_path=file_path,
                chunks_list=chunks_list,
                track_id=element.get('track_id'),
                metadata=metadata,
                error_msg=element.get('error_msg'),
                s3_key=element.get('s3_key'),
            )

        return docs_by_track_id

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = 'updated_at',
        sort_direction: str = 'desc',
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        # Whitelist validation for sort_field to prevent SQL injection
        allowed_sort_fields = {'created_at', 'updated_at', 'id', 'file_path'}
        if sort_field not in allowed_sort_fields:
            sort_field = 'updated_at'

        # Whitelist validation for sort_direction to prevent SQL injection
        sort_direction = 'desc' if sort_direction.lower() not in ['asc', 'desc'] else sort_direction.lower()

        # Calculate offset
        offset = (page - 1) * page_size

        # Build parameterized query components
        params: dict[str, Any] = {'workspace': self.workspace}
        param_count = 1

        # Build WHERE clause with parameterized query
        if status_filter is not None:
            param_count += 1
            where_clause = 'WHERE workspace=$1 AND status=$2'
            params['status'] = status_filter.value
        else:
            where_clause = 'WHERE workspace=$1'

        # Build ORDER BY clause using validated whitelist values
        order_clause = f'ORDER BY {sort_field} {sort_direction.upper()}'

        # Query for total count
        count_sql = f'SELECT COUNT(*) as total FROM LIGHTRAG_DOC_STATUS {where_clause}'
        db = self._db_required()
        count_result = await db.query(count_sql, list(params.values()))
        total_count = count_result['total'] if count_result else 0

        # Query for paginated data with parameterized LIMIT and OFFSET
        data_sql = f"""
            SELECT * FROM LIGHTRAG_DOC_STATUS
            {where_clause}
            {order_clause}
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        params['limit'] = page_size
        params['offset'] = offset
        param_values = list(params.values())

        result = await db.query(data_sql, param_values, multirows=True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result:
            doc_id = element['id']

            # Parse chunks_list JSON string back to list
            chunks_list = element.get('chunks_list', [])
            if isinstance(chunks_list, str):
                try:
                    chunks_list = json.loads(chunks_list)
                except json.JSONDecodeError:
                    chunks_list = []

            # Parse metadata JSON string back to dict
            metadata = element.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element['created_at']) or ''
            updated_at = self._format_datetime_with_timezone(element['updated_at']) or ''

            doc_status = DocProcessingStatus(
                content_summary=element['content_summary'],
                content_length=element['content_length'],
                status=element['status'],
                created_at=created_at,
                updated_at=updated_at,
                chunks_count=element['chunks_count'],
                file_path=element['file_path'],
                chunks_list=chunks_list,
                track_id=element.get('track_id'),
                metadata=metadata,
                error_msg=element.get('error_msg'),
                s3_key=element.get('s3_key'),
            )
            documents.append((doc_id, doc_status))

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        db = self._db_required()
        sql = """
            SELECT status, COUNT(*) as count
            FROM LIGHTRAG_DOC_STATUS
            WHERE workspace=$1
            GROUP BY status
        """
        params = {'workspace': self.workspace}
        result = await db.query(sql, list(params.values()), multirows=True)

        counts = {}
        total_count = 0
        for row in result:
            counts[row['status']] = row['count']
            total_count += row['count']

        # Add 'all' field with total count
        counts['all'] = total_count

        return counts

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}')
            return True

        sql = f'SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace=$1 LIMIT 1) as has_data'

        try:
            db = self._db_required()
            result = await db.query(sql, [self.workspace])
            return not result.get('has_data', False) if result else True
        except Exception as e:
            logger.error(f'[{self.workspace}] Error checking if storage is empty: {e}')
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs

        Args:
            ids (list[str]): List of document IDs to be deleted from storage

        Returns:
            None
        """
        if not ids:
            return

        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(f'[{self.workspace}] Unknown namespace for deletion: {self.namespace}')
            return

        delete_sql = f'DELETE FROM {table_name} WHERE workspace=$1 AND id = ANY($2)'

        try:
            db = self._db_required()
            await db.execute(delete_sql, {'workspace': self.workspace, 'ids': ids})
            logger.debug(f'[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error while deleting records from {self.namespace}: {e}')

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(f'[{self.workspace}] Inserting {len(data)} to {self.namespace}')
        if not data:
            return

        def parse_datetime(dt_str):
            """Parse datetime and ensure it's stored as UTC time in database"""
            if dt_str is None:
                return None
            if isinstance(dt_str, (datetime.date, datetime.datetime)):
                # If it's a datetime object
                if isinstance(dt_str, datetime.datetime):
                    # If no timezone info, assume it's UTC
                    if dt_str.tzinfo is None:
                        dt_str = dt_str.replace(tzinfo=timezone.utc)
                    # Convert to UTC and remove timezone info for storage
                    return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_str
            try:
                # Process ISO format string with timezone
                dt = datetime.datetime.fromisoformat(dt_str)
                # If no timezone info, assume it's UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Convert to UTC and remove timezone info for storage
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                logger.warning(f'[{self.workspace}] Unable to parse datetime string: {dt_str}')
                return None

        # Modified SQL to include created_at, updated_at, chunks_list, track_id, metadata, error_msg, and s3_key in both INSERT and UPDATE operations
        # All fields are updated from the input data in both INSERT and UPDATE cases
        sql = """insert into LIGHTRAG_DOC_STATUS(workspace,id,content_summary,content_length,chunks_count,status,file_path,chunks_list,track_id,metadata,error_msg,s3_key,created_at,updated_at)
                 values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                  on conflict(id,workspace) do update set
                  content_summary = EXCLUDED.content_summary,
                  content_length = EXCLUDED.content_length,
                  chunks_count = EXCLUDED.chunks_count,
                  status = EXCLUDED.status,
                  file_path = EXCLUDED.file_path,
                  chunks_list = EXCLUDED.chunks_list,
                  track_id = EXCLUDED.track_id,
                  metadata = EXCLUDED.metadata,
                  error_msg = EXCLUDED.error_msg,
                  s3_key = EXCLUDED.s3_key,
                  created_at = EXCLUDED.created_at,
                  updated_at = EXCLUDED.updated_at"""

        batch_data = []
        for k, v in data.items():
            # Remove timezone information, store utc time in db
            created_at = parse_datetime(v.get('created_at'))
            updated_at = parse_datetime(v.get('updated_at'))

            batch_data.append(
                (
                    self.workspace,
                    k,
                    v['content_summary'],
                    v['content_length'],
                    v.get('chunks_count', -1),
                    v['status'],
                    v['file_path'],
                    json.dumps(v.get('chunks_list', [])),
                    v.get('track_id'),
                    json.dumps(v.get('metadata', {})),
                    v.get('error_msg'),
                    v.get('s3_key'),
                    created_at,
                    updated_at,
                )
            )

        if batch_data:
            db = self._db_required()
            await db.executemany(sql, batch_data)

    async def update_s3_key(self, doc_id: str, s3_key: str) -> bool:
        """Update s3_key for a document after archiving.

        Args:
            doc_id: Document ID to update
            s3_key: S3 storage key (e.g., 'archive/default/doc123/file.pdf')

        Returns:
            True if update was successful
        """
        sql = """
            UPDATE LIGHTRAG_DOC_STATUS
            SET s3_key = $1, updated_at = CURRENT_TIMESTAMP
            WHERE workspace = $2 AND id = $3
        """
        params = {'s3_key': s3_key, 'workspace': self.workspace, 'id': doc_id}
        db = self._db_required()
        await db.execute(sql, params)
        logger.debug(f'[{self.workspace}] Updated s3_key for doc {doc_id}: {s3_key}')
        return True

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    'status': 'error',
                    'message': f'Unknown namespace: {self.namespace}',
                }

            drop_sql = SQL_TEMPLATES['drop_specifiy_table_workspace'].format(table_name=table_name)
            db = self._db_required()
            await db.execute(drop_sql, {'workspace': self.workspace})
            return {'status': 'success', 'message': 'data dropped'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


class PGGraphQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: str | dict[str, Any]) -> None:
        if isinstance(exception, dict):
            self.message = exception.get('message', 'unknown')
            self.details = exception.get('details', 'unknown')
        else:
            self.message = exception
            self.details = 'unknown'

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


@final
@dataclass
class PGGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        # Graph name will be dynamically generated in initialize() based on workspace
        self.db: PostgreSQLDB | None = None

    def _db_required(self) -> PostgreSQLDB:
        if self.db is None:
            raise RuntimeError('PostgreSQL client is not initialized')
        return self.db

    def _get_workspace_graph_name(self) -> str:
        """
        Generate graph name based on workspace and namespace for data isolation.
        Rules:
        - If workspace is empty or "default": graph_name = namespace
        - If workspace has other value: graph_name = workspace_namespace

        Args:
            None

        Returns:
            str: The graph name for the current workspace
        """
        workspace = self.workspace
        namespace = self.namespace

        if workspace and workspace.strip() and workspace.strip().lower() != 'default':
            # Ensure names comply with PostgreSQL identifier specifications
            safe_workspace = re.sub(r'[^a-zA-Z0-9_]', '_', workspace.strip())
            safe_namespace = re.sub(r'[^a-zA-Z0-9_]', '_', namespace)
            return f'{safe_workspace}_{safe_namespace}'
        else:
            # When the workspace is "default", use the namespace directly (for backward compatibility with legacy implementations)
            return re.sub(r'[^a-zA-Z0-9_]', '_', namespace)

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        """Best-effort sanitization for identifiers we interpolate into Cypher.

        This avoids common parse errors without altering the semantic value.
        Control chars are stripped, quotes/backticks are escaped, and we keep
        the result ASCII-only to match server expectations.
        """

        # Drop control characters that can break AGE parsing
        normalized_id = re.sub(r'[\x00-\x1F]', '', node_id)

        # Escape characters that matter for the interpolated Cypher literal
        normalized_id = normalized_id.replace('\\', '\\\\')  # backslash
        normalized_id = normalized_id.replace('"', '\\"')  # double quote
        normalized_id = normalized_id.replace('`', '\\`')  # backtick

        # Keep it compact and ASCII to avoid encoding surprises
        normalized_id = normalized_id.encode('ascii', 'ignore').decode('ascii')
        return normalized_id

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, 'workspace') and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = 'default'

            # Dynamically generate graph name based on workspace
            self.graph_name = self._get_workspace_graph_name()

            # Log the graph initialization for debugging
            logger.info(f"[{self.workspace}] PostgreSQL Graph initialized: graph_name='{self.graph_name}'")

            # Create AGE extension and configure graph environment once at initialization
            db = self._db_required()
            if db.pool is None:
                raise RuntimeError('PostgreSQL pool is not initialized')
            async with db.pool.acquire() as connection:
                # First ensure AGE extension is created
                await PostgreSQLDB.configure_age_extension(connection)

            # Execute each statement separately and ignore errors
            queries = [
                f"SELECT create_graph('{self.graph_name}')",
                f"SELECT create_vlabel('{self.graph_name}', 'base');",
                f"SELECT create_elabel('{self.graph_name}', 'DIRECTED');",
                # f'CREATE INDEX CONCURRENTLY vertex_p_idx ON {self.graph_name}."_ag_label_vertex" (id)',
                f'CREATE INDEX CONCURRENTLY vertex_idx_node_id ON {self.graph_name}."_ag_label_vertex" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
                # f'CREATE INDEX CONCURRENTLY edge_p_idx ON {self.graph_name}."_ag_label_edge" (id)',
                f'CREATE INDEX CONCURRENTLY edge_sid_idx ON {self.graph_name}."_ag_label_edge" (start_id)',
                f'CREATE INDEX CONCURRENTLY edge_eid_idx ON {self.graph_name}."_ag_label_edge" (end_id)',
                f'CREATE INDEX CONCURRENTLY edge_seid_idx ON {self.graph_name}."_ag_label_edge" (start_id,end_id)',
                f'CREATE INDEX CONCURRENTLY directed_p_idx ON {self.graph_name}."DIRECTED" (id)',
                f'CREATE INDEX CONCURRENTLY directed_eid_idx ON {self.graph_name}."DIRECTED" (end_id)',
                f'CREATE INDEX CONCURRENTLY directed_sid_idx ON {self.graph_name}."DIRECTED" (start_id)',
                f'CREATE INDEX CONCURRENTLY directed_seid_idx ON {self.graph_name}."DIRECTED" (start_id,end_id)',
                f'CREATE INDEX CONCURRENTLY entity_p_idx ON {self.graph_name}."base" (id)',
                f'CREATE INDEX CONCURRENTLY entity_idx_node_id ON {self.graph_name}."base" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
                f'CREATE INDEX CONCURRENTLY entity_node_id_gin_idx ON {self.graph_name}."base" using gin(properties)',
                f'CREATE UNIQUE INDEX CONCURRENTLY {self.graph_name}_entity_id_unique ON {self.graph_name}."base" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
                f'ALTER TABLE {self.graph_name}."DIRECTED" CLUSTER ON directed_sid_idx',
            ]

            for query in queries:
                # Use the new flag to silently ignore "already exists" errors
                # at the source, preventing log spam.
                await db.execute(
                    query,
                    upsert=True,
                    ignore_if_exists=True,  # Pass the new flag
                    with_age=True,
                    graph_name=self.graph_name,
                )

    async def health_check(self, max_retries: int = 3) -> dict[str, Any]:
        """Comprehensive health check for PostgreSQL graph connectivity and status.

        Returns a detailed health report including:
        - Overall status (healthy/degraded/unhealthy)
        - Individual check results with latency
        - Migration status
        - Graph information
        """
        import time
        from datetime import datetime

        result: dict[str, Any] = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
        }
        graph_names: list[str] = []
        db = self._db_required()

        # 1. Basic connectivity check with latency
        try:
            start = time.perf_counter()
            await db.execute('SELECT 1')
            latency_ms = (time.perf_counter() - start) * 1000
            result['checks']['connectivity'] = {
                'status': True,
                'latency_ms': round(latency_ms, 2),
            }
        except Exception as e:
            result['checks']['connectivity'] = {'status': False, 'error': str(e)}
            result['status'] = 'unhealthy'
            logger.debug(f'Graph health check - connectivity failed: {e}')
            return result  # Can't proceed without connectivity

        # 2. AGE extension check
        try:
            start = time.perf_counter()
            graphs = await db.query(
                'SELECT name FROM ag_catalog.ag_graph',
                multirows=True,
                with_age=False,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            graph_names: list[str] = []
            if graphs:
                for g in graphs:
                    if isinstance(g, dict):
                        name = g.get('name')
                        if isinstance(name, str):
                            graph_names.append(name)
            result['checks']['age_extension'] = {
                'status': True,
                'latency_ms': round(latency_ms, 2),
                'graphs': graph_names,
            }
        except Exception as e:
            result['checks']['age_extension'] = {'status': False, 'error': str(e)}
            result['status'] = 'degraded'
            logger.debug(f'Graph health check - AGE extension failed: {e}')

        # 3. Workspace graph check
        graph_exists = self.graph_name in graph_names
        if graph_exists:
            try:
                start = time.perf_counter()
                await db.query(
                    f"SELECT * FROM cypher('{self.graph_name}', $$ RETURN 1 $$) AS (one agtype)",
                    with_age=True,
                    graph_name=self.graph_name,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                result['checks']['workspace_graph'] = {
                    'status': True,
                    'graph_name': self.graph_name,
                    'latency_ms': round(latency_ms, 2),
                }
            except Exception as e:
                result['checks']['workspace_graph'] = {
                    'status': False,
                    'graph_name': self.graph_name,
                    'error': str(e),
                }
                result['status'] = 'degraded'
                logger.debug(f'Graph health check - workspace graph failed: {e}')
        else:
            result['checks']['workspace_graph'] = {
                'status': False,
                'graph_name': self.graph_name,
                'error': 'Graph does not exist yet',
            }

        # 4. Migration status check
        if db:
            migration_status = db.get_migration_status()
            result['checks']['migrations'] = {
                'status': migration_status['success'],
                'failures': [f[0] for f in migration_status['failures']],
            }
            if not migration_status['success']:
                result['status'] = 'degraded'

        return result

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    @staticmethod
    def _record_to_dict(record: Mapping[str, Any]) -> dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """

        @staticmethod
        def parse_agtype_string(agtype_str: str) -> tuple[str, str]:
            """
            Parse agtype string precisely, separating JSON content and type identifier

            Args:
                agtype_str: String like '{"json": "content"}::vertex'

            Returns:
                (json_content, type_identifier)
            """
            if not isinstance(agtype_str, str) or '::' not in agtype_str:
                return agtype_str, ''

            # Find the last :: from the right, which is the start of type identifier
            last_double_colon = agtype_str.rfind('::')

            if last_double_colon == -1:
                return agtype_str, ''

            # Separate JSON content and type identifier
            json_content = agtype_str[:last_double_colon]
            type_identifier = agtype_str[last_double_colon + 2 :]

            return json_content, type_identifier

        @staticmethod
        def safe_json_parse(json_str: str, context: str = '') -> dict:
            """
            Safe JSON parsing with simplified error logging
            """
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f'JSON parsing failed ({context}): {e}')
                logger.error(f'Raw data (first 100 chars): {json_str[:100]!r}')
                logger.error(f'Error position: line {e.lineno}, column {e.colno}')
                return {}

        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}

        # First pass: preprocess vertices
        for k in record:
            v = record[k]
            if isinstance(v, str) and '::' in v:
                if v.startswith('[') and v.endswith(']'):
                    # Handle vertex arrays
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == 'vertex':
                        vertexes = safe_json_parse(json_content, f'vertices array for {k}')
                        if vertexes:
                            for vertex in vertexes:
                                vertices[vertex['id']] = vertex.get('properties')
                else:
                    # Handle single vertex
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == 'vertex':
                        vertex = safe_json_parse(json_content, f'single vertex for {k}')
                        if vertex:
                            vertices[vertex['id']] = vertex.get('properties')

        # Second pass: process all fields
        for k in record:
            v = record[k]
            if isinstance(v, str) and '::' in v:
                if v.startswith('[') and v.endswith(']'):
                    # Handle array types
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ['vertex', 'edge']:
                        parsed_data = safe_json_parse(json_content, f'array {type_id} for field {k}')
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        logger.warning(f'Unknown array type: {type_id}')
                        d[k] = None
                else:
                    # Handle single objects
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ['vertex', 'edge']:
                        parsed_data = safe_json_parse(json_content, f'single {type_id} for field {k}')
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        # May be other types of agtype data, keep as is
                        d[k] = v
            else:
                d[k] = v  # Keep as string

        return d

    @staticmethod
    def _format_properties(properties: dict[str, Any], _id: str | None = None) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Uses json.dumps() for value escaping which handles quotes, backslashes,
        and other special characters safely.

        Args:
            properties: a dictionary containing node/edge properties
            _id: the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted Cypher map
        """
        MAX_KEY_LENGTH = 255  # Reasonable limit for property key names

        props = []
        for k, v in properties.items():
            # Truncate excessively long keys (defensive)
            if len(k) > MAX_KEY_LENGTH:
                k = k[:MAX_KEY_LENGTH]

            # Escape backticks within key names to prevent Cypher injection
            k_safe = k.replace('`', '\\`')

            # Wrap property key in backticks to escape special characters
            prop = f'`{k_safe}`: {json.dumps(v)}'
            props.append(prop)

        if _id is not None and 'id' not in properties:
            props.append(f'id: {json.dumps(_id)}' if isinstance(_id, str) else f'id: {_id}')
        return '{' + ', '.join(props) + '}'

    async def _query(
        self,
        query: str,
        readonly: bool = True,
        upsert: bool = False,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed

        Returns:
            list[dict[str, Any]]: a list of dictionaries containing the result set
        """
        db = self._db_required()
        try:
            if readonly:
                data = await db.query(
                    query,
                    list(params.values()) if params else None,
                    multirows=True,
                    with_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await db.execute(
                    query,
                    data=params,
                    upsert=upsert,
                    with_age=True,
                    graph_name=self.graph_name,
                )

        except Exception as e:
            raise PGGraphQueryException(
                {
                    'message': f'Error executing graph query: {query}',
                    'wrapped': query,
                    'detail': repr(e),
                    'error_type': e.__class__.__name__,
                }
            ) from e

        # decode records
        result = [] if data is None else [self._record_to_dict(d) for d in data]

        return result

    async def has_node(self, node_id: str) -> bool:
        query = f"""
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
              LIMIT 1
            ) AS node_exists;
        """

        params = {'node_id': node_id}
        row = (await self._query(query, params=params))[0]
        return bool(row['node_exists'])

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        query = f"""
            WITH a AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
            ),
            b AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($2::text)::text)::agtype
            )
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.start_id = a.vid
              JOIN b ON d.end_id   = b.vid
              LIMIT 1
            )
            OR EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.end_id   = a.vid
              JOIN b ON d.start_id = b.vid
              LIMIT 1
            ) AS edge_exists;
        """
        params = {
            'source_node_id': source_node_id,
            'target_node_id': target_node_id,
        }
        row = (await self._query(query, params=params))[0]
        return bool(row['edge_exists'])

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties"""

        result = await self.get_nodes_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]
        return None

    async def node_degree(self, node_id: str) -> int:
        result = await self.node_degrees_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]
        return 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        result = await self.edge_degrees_batch(edge_pairs=[(src_id, tgt_id)])
        if result and (src_id, tgt_id) in result:
            return result[(src_id, tgt_id)]
        return 0

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes"""
        result = await self.get_edges_batch([{'src': source_node_id, 'tgt': target_node_id}])
        if result and (source_node_id, target_node_id) in result:
            return result[(source_node_id, target_node_id)]
        return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: list of dictionaries containing edge information
        """
        label = self._normalize_node_id(source_node_id)

        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                      MATCH (n:base {{entity_id: $1}})
                      OPTIONAL MATCH (n)-[]-(connected:base)
                      RETURN n.entity_id AS source_id, connected.entity_id AS connected_id
                    $$, $1) AS (source_id text, connected_id text)"""

        results = await self._query(query, params={'node_id': label})
        edges = []
        for record in results:
            source_id = record['source_id']
            connected_id = record['connected_id']

            if source_id and connected_id:
                edges.append((source_id, connected_id))

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the PostgreSQL graph database using Apache AGE.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        if 'entity_id' not in node_data:
            raise ValueError("PostgreSQL: node properties must contain an 'entity_id' field")

        label = self._normalize_node_id(node_id)
        properties = self._format_properties(node_data)

        # Note: AGE doesn't support parameterized SET n += $map, so we use
        # _format_properties which uses json.dumps for safe value escaping
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                     MERGE (n:base {{entity_id: "{label}"}})
                     SET n += {properties}
                     RETURN n
                   $$) AS (n agtype)"""

        try:
            db = self._db_required()
            await db._timed_operation(
                self._query(query, readonly=False, upsert=True),
                f'upsert_node({node_id})',
                slow_threshold=2.0,  # Node upserts should be fast
            )

        except Exception:
            logger.error(f'[{self.workspace}] POSTGRES, upsert_node error on node_id: `{node_id}`')
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): dictionary of properties to set on the edge
        """
        src_label = self._normalize_node_id(source_node_id)
        tgt_label = self._normalize_node_id(target_node_id)
        edge_properties = self._format_properties(edge_data)

        # Note: AGE doesn't support parameterized SET r += $map, so we use
        # _format_properties which uses json.dumps for safe value escaping
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                     MATCH (source:base {{entity_id: "{src_label}"}})
                     WITH source
                     MATCH (target:base {{entity_id: "{tgt_label}"}})
                     MERGE (source)-[r:DIRECTED]-(target)
                     SET r += {edge_properties}
                     SET r += {edge_properties}
                     RETURN r
                   $$) AS (r agtype)"""

        try:
            db = self._db_required()
            await db._timed_operation(
                self._query(query, readonly=False, upsert=True),
                f'upsert_edge({source_node_id}->{target_node_id})',
                slow_threshold=2.0,  # Edge upserts should be fast
            )

        except Exception:
            logger.error(
                f'[{self.workspace}] POSTGRES, upsert_edge error on edge: `{source_node_id}`-`{target_node_id}`'
            )
            raise

    async def upsert_nodes_bulk(self, nodes: list[tuple[str, dict[str, str]]], batch_size: int = 500) -> None:
        """Bulk upsert nodes using UNWIND for efficient batching.

        Uses UNWIND with explicit SET clauses for each property, since AGE
        doesn't support SET += map in UNWIND context. Falls back to individual
        upserts if bulk query fails.

        Args:
            nodes: List of (node_id, properties) tuples
            batch_size: Number of nodes per batch (default 500)
        """
        if not nodes:
            return

        import time

        start = time.perf_counter()

        db = self._db_required()

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]

            # Build node data array with normalized properties
            node_data = []
            for node_id, data in batch:
                if 'entity_id' not in data:
                    logger.warning(f'[{self.workspace}] Skipping node {node_id}: missing entity_id')
                    continue

                label = self._normalize_node_id(node_id)
                node_data.append(
                    {
                        'entity_id': label,
                        'entity_name': data.get('entity_name', ''),
                        'entity_type': data.get('entity_type', ''),
                        'description': data.get('description', ''),
                        'source_id': data.get('source_id', ''),
                    }
                )

            if not node_data:
                continue

            # Build UNWIND query with explicit SET clauses
            # Note: json.dumps handles all value escaping safely
            query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND {json.dumps(node_data)} AS d
                MERGE (n:base {{entity_id: d.entity_id}})
                SET n.entity_name = d.entity_name
                SET n.entity_type = d.entity_type
                SET n.description = d.description
                SET n.source_id = d.source_id
                RETURN count(n)
            $$) AS (cnt agtype)"""

            try:
                await db._timed_operation(
                    self._query(query, readonly=False, upsert=True),
                    f'upsert_nodes_bulk_batch({len(node_data)})',
                    slow_threshold=10.0,  # Bulk ops can take longer
                )
            except Exception as e:
                logger.warning(f'[{self.workspace}] Bulk node upsert failed, falling back to individual: {e}')
                # Fallback to individual upserts
                for node_id, data in batch:
                    await self.upsert_node(node_id, data)

        elapsed = time.perf_counter() - start

        # Log summary for bulk operations
        if elapsed > 5.0 or len(nodes) > 100:
            logger.info(
                f'[{self.workspace}] upsert_nodes_bulk: {len(nodes)} nodes in {elapsed:.2f}s '
                f'({len(nodes) / elapsed:.1f} nodes/sec)'
            )

    async def upsert_edges_bulk(
        self,
        edges: list[tuple[str, str, dict[str, str]]],
        batch_size: int = 500,
    ) -> None:
        """Bulk upsert edges using UNWIND for efficient batching.

        Uses UNWIND with explicit SET clauses for each property, since AGE
        doesn't support SET += map in UNWIND context. Falls back to individual
        upserts if bulk query fails.

        Args:
            edges: List of (source_id, target_id, properties) tuples
            batch_size: Number of edges per batch (default 500)
        """
        if not edges:
            return

        import time

        start = time.perf_counter()
        db = self._db_required()

        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]

            # Build edge data array with normalized properties
            edge_data = []
            for src, tgt, props in batch:
                src_id = self._normalize_node_id(src)
                tgt_id = self._normalize_node_id(tgt)
                edge_data.append(
                    {
                        'src': src_id,
                        'tgt': tgt_id,
                        'weight': props.get('weight', 1.0),
                        'description': props.get('description', ''),
                        'keywords': props.get('keywords', ''),
                        'source_id': props.get('source_id', ''),
                    }
                )

            if not edge_data:
                continue

            # Build UNWIND query with explicit SET clauses
            # Note: json.dumps handles all value escaping safely
            query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND {json.dumps(edge_data)} AS d
                MATCH (s:base {{entity_id: d.src}})
                MATCH (t:base {{entity_id: d.tgt}})
                MERGE (s)-[r:DIRECTED]->(t)
                SET r.weight = d.weight
                SET r.description = d.description
                SET r.keywords = d.keywords
                SET r.source_id = d.source_id
                RETURN count(r)
            $$) AS (cnt agtype)"""

            try:
                await db._timed_operation(
                    self._query(query, readonly=False, upsert=True),
                    f'upsert_edges_bulk_batch({len(edge_data)})',
                    slow_threshold=10.0,  # Bulk ops can take longer
                )
            except Exception as e:
                logger.warning(f'[{self.workspace}] Bulk edge upsert failed, falling back to individual: {e}')
                # Fallback to individual upserts
                for src, tgt, props in batch:
                    await self.upsert_edge(src, tgt, props)

        elapsed = time.perf_counter() - start

        # Log summary for bulk operations
        if elapsed > 5.0 or len(edges) > 100:
            logger.info(
                f'[{self.workspace}] upsert_edges_bulk: {len(edges)} edges in {elapsed:.2f}s '
                f'({len(edges) / elapsed:.1f} edges/sec)'
            )

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node from the graph.

        Args:
            node_id (str): The ID of the node to delete.
        """
        label = self._normalize_node_id(node_id)

        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                     MATCH (n:base {{entity_id: $1}})
                     DETACH DELETE n
                   $$, $1) AS (n agtype)"""

        try:
            await self._query(query, readonly=False, params={'node_id': label})
        except Exception as e:
            logger.error(f'[{self.workspace}] Error during node deletion: {e}')
            raise

    async def remove_nodes(self, nodes: list[str]) -> None:
        """
        Remove multiple nodes from the graph.

        Args:
            node_ids (list[str]): A list of node IDs to remove.
        """
        node_ids = [self._normalize_node_id(node_id) for node_id in nodes]
        if not node_ids:
            return

        unique_ids = list(dict.fromkeys(node_ids))
        cy_params = {'params': json.dumps({'node_ids': unique_ids}, ensure_ascii=False)}

        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                     UNWIND $node_ids AS node_id
                     MATCH (n:base {{entity_id: node_id}})
                     DETACH DELETE n
                   $$, $1::agtype) AS (n agtype)"""

        try:
            await self._query(query, readonly=False, params=cy_params)
            logger.debug(f'[{self.workspace}] Removed {len(unique_ids)} nodes from graph')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error during node removal: {e}')
            raise

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph.

        Args:
            edges (list[tuple[str, str]]): A list of edges to remove, where each edge is a tuple of (source_node_id, target_node_id).
        """
        if not edges:
            return

        cleaned_edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for source, target in edges:
            pair = (self._normalize_node_id(source), self._normalize_node_id(target))
            if pair not in seen:
                seen.add(pair)
                cleaned_edges.append(pair)

        if not cleaned_edges:
            return

        literal_pairs = ', '.join([f'{{src: {json.dumps(src)}, tgt: {json.dumps(tgt)}}}' for src, tgt in cleaned_edges])

        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                         UNWIND [{literal_pairs}] AS pair
                         MATCH (a:base {{entity_id: pair.src}})-[r]-(b:base {{entity_id: pair.tgt}})
                         DELETE r
                       $$) AS (r agtype)"""

        try:
            await self._query(query, readonly=False)
            logger.debug(f'[{self.workspace}] Deleted {len(cleaned_edges)} edges (undirected)')
        except Exception as e:
            logger.error(f'[{self.workspace}] Error during edge deletion: {e!s}')
            raise

    async def get_nodes_batch(self, node_ids: list[str], batch_size: int = 1000) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        # Build result dictionary
        nodes_dict = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]

            query = f"""
                WITH input(v, ord) AS (
                  SELECT v, ord
                  FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                ),
                ids(node_id, ord) AS (
                  SELECT (to_json(v)::text)::agtype AS node_id, ord
                  FROM input
                )
                SELECT i.node_id::text AS node_id,
                       b.properties
                FROM {self.graph_name}.base AS b
                JOIN ids i
                  ON ag_catalog.agtype_access_operator(
                       VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                     ) = i.node_id
                ORDER BY i.ord;
            """

            results = await self._query(query, params={'ids': batch})

            for result in results:
                if result['node_id'] and result['properties']:
                    node_dict = result['properties']

                    # Process string result, parse it to JSON dictionary
                    if isinstance(node_dict, str):
                        try:
                            node_dict = json.loads(node_dict)
                        except json.JSONDecodeError:
                            logger.warning(f'[{self.workspace}] Failed to parse node string in batch: {node_dict}')

                    node_key = result['node_id']
                    original_key = lookup.get(node_key)
                    if original_key is None:
                        logger.warning(f'[{self.workspace}] Node {node_key} not found in lookup map')
                        original_key = node_key
                    if original_key in requested:
                        nodes_dict[original_key] = node_dict

        return nodes_dict

    async def node_degrees_batch(self, node_ids: list[str], batch_size: int = 500) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.
        Calculates the total degree by counting distinct relationships.
        Uses separate queries for outgoing and incoming edges.

        Args:
            node_ids: List of node labels (entity_id values) to look up.
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping each node_id to its degree (total number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        out_degrees = {}
        in_degrees = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]

            query = f"""
                    WITH input(v, ord) AS (
                      SELECT v, ord
                      FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                    ),
                    ids(node_id, ord) AS (
                      SELECT (to_json(v)::text)::agtype AS node_id, ord
                      FROM input
                    ),
                    vids AS (
                      SELECT b.id AS vid, i.node_id, i.ord
                      FROM {self.graph_name}.base AS b
                      JOIN ids i
                        ON ag_catalog.agtype_access_operator(
                             VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                           ) = i.node_id
                    ),
                    deg_out AS (
                      SELECT d.start_id AS vid, COUNT(*)::bigint AS out_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.start_id
                      GROUP BY d.start_id
                    ),
                    deg_in AS (
                      SELECT d.end_id AS vid, COUNT(*)::bigint AS in_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.end_id
                      GROUP BY d.end_id
                    )
                    SELECT v.node_id::text AS node_id,
                           COALESCE(o.out_degree, 0) AS out_degree,
                           COALESCE(n.in_degree, 0)  AS in_degree
                    FROM vids v
                    LEFT JOIN deg_out o ON o.vid = v.vid
                    LEFT JOIN deg_in  n ON n.vid = v.vid
                    ORDER BY v.ord;
                """

            combined_results = await self._query(query, params={'ids': batch})

            for row in combined_results:
                node_id = row['node_id']
                if not node_id:
                    continue
                node_key = node_id
                original_key = lookup.get(node_key)
                if original_key is None:
                    logger.warning(f'[{self.workspace}] Node {node_key} not found in lookup map')
                    original_key = node_key
                if original_key in requested:
                    out_degrees[original_key] = int(row.get('out_degree', 0) or 0)
                    in_degrees[original_key] = int(row.get('in_degree', 0) or 0)

        degrees_dict = {}
        for node_id in node_ids:
            out_degree = out_degrees.get(node_id, 0)
            in_degree = in_degrees.get(node_id, 0)
            degrees_dict[node_id] = out_degree + in_degree

        return degrees_dict

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edges: List of (source_node_id, target_node_id) tuples

        Returns:
            Dictionary mapping edge tuples to their combined degrees
        """
        if not edge_pairs:
            return {}

        # Use node_degrees_batch to get all node degrees efficiently
        all_nodes = set()
        for src, tgt in edge_pairs:
            all_nodes.add(src)
            all_nodes.add(tgt)

        node_degrees = await self.node_degrees_batch(list(all_nodes))

        # Calculate edge degrees
        edge_degrees_dict: dict[tuple[str, str], int] = {}
        for src, tgt in edge_pairs:
            src_degree = node_degrees.get(src, 0)
            tgt_degree = node_degrees.get(tgt, 0)
            edge_degrees_dict[(src, tgt)] = src_degree + tgt_degree

        return edge_degrees_dict

    async def get_edges_batch(self, pairs: list[dict[str, str]], batch_size: int = 500) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.
        Get forward and backward edges seperately and merge them before return

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        if not pairs:
            return {}

        seen = set()
        uniq_pairs: list[dict[str, str]] = []
        for p in pairs:
            s = self._normalize_node_id(p['src'])
            t = self._normalize_node_id(p['tgt'])
            key = (s, t)
            if s and t and key not in seen:
                seen.add(key)
                uniq_pairs.append(p)

        edges_dict: dict[tuple[str, str], dict] = {}

        for i in range(0, len(uniq_pairs), batch_size):
            batch = uniq_pairs[i : i + batch_size]

            pairs = [{'src': p['src'], 'tgt': p['tgt']} for p in batch]

            forward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)-[r]->(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""
            backward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)<-[r]-(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""

            def dollar_quote(s: str, tag_prefix='AGE'):
                s = '' if s is None else str(s)
                for i in itertools.count(1):
                    tag = f'{tag_prefix}{i}'
                    wrapper = f'${tag}$'
                    if wrapper not in s:
                        return f'{wrapper}{s}{wrapper}'

            sql_fwd = f"""
            SELECT * FROM cypher({dollar_quote(self.graph_name)}::name,
                                 {dollar_quote(forward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """

            sql_bwd = f"""
            SELECT * FROM cypher({dollar_quote(self.graph_name)}::name,
                                 {dollar_quote(backward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """

            pg_params = {'params': json.dumps({'pairs': pairs}, ensure_ascii=False)}

            forward_results, backward_results = await asyncio.gather(
                self._query(sql_fwd, params=pg_params), self._query(sql_bwd, params=pg_params)
            )

            for result in forward_results:
                if result['source'] and result['target'] and result['edge_properties']:
                    edge_props = result['edge_properties']

                    # Process string result, parse it to JSON dictionary
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            logger.warning(f'[{self.workspace}]Failed to parse edge properties string: {edge_props}')
                            continue

                    edges_dict[(result['source'], result['target'])] = edge_props

            for result in backward_results:
                if result['source'] and result['target'] and result['edge_properties']:
                    edge_props = result['edge_properties']

                    # Process string result, parse it to JSON dictionary
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            logger.warning(f'[{self.workspace}] Failed to parse edge properties string: {edge_props}')
                            continue

                    edges_dict[(result['source'], result['target'])] = edge_props

        return edges_dict

    async def get_nodes_edges_batch(
        self, node_ids: list[str], batch_size: int = 500
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Get all edges (both outgoing and incoming) for multiple nodes in a single batch operation.

        Args:
            node_ids: List of node IDs to get edges for
            batch_size: Batch size for the query

        Returns:
            Dictionary mapping node IDs to lists of (source, target) edge tuples
        """
        if not node_ids:
            return {}

        seen = set()
        unique_ids: list[str] = []
        for nid in node_ids:
            n = self._normalize_node_id(nid)
            if n and n not in seen:
                seen.add(n)
                unique_ids.append(n)

        edges_norm: dict[str, list[tuple[str, str]]] = {n: [] for n in unique_ids}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]
            cy_params = {'params': json.dumps({'node_ids': batch}, ensure_ascii=False)}

            outgoing_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                         UNWIND $node_ids AS node_id
                         MATCH (n:base {{entity_id: node_id}})
                         OPTIONAL MATCH (n:base)-[]->(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id
                       $$, $1::agtype) AS (node_id text, connected_id text)"""

            incoming_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                         UNWIND $node_ids AS node_id
                         MATCH (n:base {{entity_id: node_id}})
                         OPTIONAL MATCH (n:base)<-[]-(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id
                       $$, $1::agtype) AS (node_id text, connected_id text)"""

            outgoing_results, incoming_results = await asyncio.gather(
                self._query(outgoing_query, params=cy_params), self._query(incoming_query, params=cy_params)
            )

            for result in outgoing_results:
                if result['node_id'] and result['connected_id']:
                    edges_norm[result['node_id']].append((result['node_id'], result['connected_id']))

            for result in incoming_results:
                if result['node_id'] and result['connected_id']:
                    edges_norm[result['node_id']].append((result['connected_id'], result['node_id']))

        out: dict[str, list[tuple[str, str]]] = {}
        for orig in node_ids:
            n = self._normalize_node_id(orig)
            out[orig] = edges_norm.get(n, [])

        return out

    async def get_all_labels(self) -> list[str]:
        """
        Get all labels (node IDs) in the graph.

        Returns:
            list[str]: A list of all labels in the graph.
        """
        # Use native SQL for better performance
        query = f"""
            SELECT DISTINCT
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text AS label
            FROM {self.graph_name}.base
            WHERE ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]) IS NOT NULL
            ORDER BY label
        """

        results = await self._query(query)
        labels = []
        for result in results:
            if result and isinstance(result, dict) and 'label' in result:
                labels.append(result['label'])
        return labels

    async def _bfs_subgraph(self, node_label: str, max_depth: int | None, max_nodes: int | None) -> KnowledgeGraph:
        """
        Implements a true breadth-first search algorithm for subgraph retrieval.
        This method is used as a fallback when the standard Cypher query is too slow
        or when we need to guarantee BFS ordering.

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            max_nodes: Maximum number of nodes to return

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_node_ids = set()
        visited_edges = set()
        visited_edge_pairs = set()

        # Normalize limits
        max_depth = 0 if max_depth is None else int(max_depth)
        max_nodes = 0 if max_nodes is None else int(max_nodes)
        max_depth_int = max_depth
        max_nodes_int = max_nodes

        # Get starting node data
        label = self._normalize_node_id(node_label)
        query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base {{entity_id: $1}})
                    RETURN id(n) as node_id, n
                  $$, $1) AS (node_id bigint, n agtype)"""

        node_result = await self._query(query, params={'node_id': label})
        if not node_result or not node_result[0].get('n'):
            return result

        # Create initial KnowledgeGraphNode
        start_node_data = node_result[0]['n']
        entity_id = start_node_data['properties']['entity_id']
        internal_id = str(start_node_data['id'])

        start_node = KnowledgeGraphNode(
            id=internal_id,
            labels=[entity_id],
            properties=start_node_data['properties'],
        )

        # Initialize BFS queue, each element is a tuple of (node, depth)
        queue = deque([(start_node, 0)])

        visited_nodes.add(entity_id)
        visited_node_ids.add(internal_id)
        result.nodes.append(start_node)

        result.is_truncated = False

        # BFS search main loop
        while queue:
            # Get all nodes at the current depth
            current_level_nodes = []
            current_depth = None

            # Determine current depth
            if queue:
                current_depth = queue[0][1]
            if current_depth is None:
                break

            # Extract all nodes at current depth from the queue
            while queue and queue[0][1] == current_depth:
                node, depth = queue.popleft()
                if depth > max_depth_int:
                    continue
                current_level_nodes.append(node)

            if not current_level_nodes:
                continue

            # Check depth limit
            if current_depth > max_depth_int:
                continue

            # Prepare node IDs list
            node_ids = [self._normalize_node_id(node.labels[0]) for node in current_level_nodes]
            cy_params = {'params': json.dumps({'node_ids': node_ids}, ensure_ascii=False)}

            # Construct batch query for outgoing edges
            outgoing_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND $node_ids AS node_id
                MATCH (n:base {{entity_id: node_id}})
                OPTIONAL MATCH (n)-[r]->(neighbor:base)
                RETURN node_id AS current_id,
                       id(n) AS current_internal_id,
                       id(neighbor) AS neighbor_internal_id,
                       neighbor.entity_id AS neighbor_id,
                       id(r) AS edge_id,
                       r,
                       neighbor,
                       true AS is_outgoing
              $$, $1::agtype) AS (current_id text, current_internal_id bigint, neighbor_internal_id bigint,
                      neighbor_id text, edge_id bigint, r agtype, neighbor agtype, is_outgoing bool)"""

            # Construct batch query for incoming edges
            incoming_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND $node_ids AS node_id
                MATCH (n:base {{entity_id: node_id}})
                OPTIONAL MATCH (n)<-[r]-(neighbor:base)
                RETURN node_id AS current_id,
                       id(n) AS current_internal_id,
                       id(neighbor) AS neighbor_internal_id,
                       neighbor.entity_id AS neighbor_id,
                       id(r) AS edge_id,
                       r,
                       neighbor,
                       false AS is_outgoing
              $$, $1::agtype) AS (current_id text, current_internal_id bigint, neighbor_internal_id bigint,
                      neighbor_id text, edge_id bigint, r agtype, neighbor agtype, is_outgoing bool)"""

            # Execute queries concurrently
            outgoing_results, incoming_results = await asyncio.gather(
                self._query(outgoing_query, params=cy_params), self._query(incoming_query, params=cy_params)
            )

            # Combine results
            neighbors = outgoing_results + incoming_results

            # Create mapping from node ID to node object
            node_map = {node.labels[0]: node for node in current_level_nodes}

            # Process all results in a single loop
            for record in neighbors:
                if not record.get('neighbor') or not record.get('r'):
                    continue

                # Get current node information
                current_entity_id = record['current_id']
                current_node = node_map[current_entity_id]

                # Get neighbor node information
                neighbor_entity_id = record['neighbor_id']
                neighbor_internal_id = str(record['neighbor_internal_id'])
                is_outgoing = record['is_outgoing']

                # Determine edge direction
                if is_outgoing:
                    source_id = current_node.id
                    target_id = neighbor_internal_id
                else:
                    source_id = neighbor_internal_id
                    target_id = current_node.id

                if not neighbor_entity_id:
                    continue

                # Get edge and node information
                b_node = record['neighbor']
                rel = record['r']
                edge_id = str(record['edge_id'])

                # Create neighbor node object
                neighbor_node = KnowledgeGraphNode(
                    id=neighbor_internal_id,
                    labels=[neighbor_entity_id],
                    properties=b_node['properties'],
                )

                # Sort entity_ids to ensure (A,B) and (B,A) are treated as the same edge
                sorted_pair = tuple(sorted([current_entity_id, neighbor_entity_id]))

                # Create edge object
                edge = KnowledgeGraphEdge(
                    id=edge_id,
                    type=rel['label'],
                    source=source_id,
                    target=target_id,
                    properties=rel['properties'],
                )

                if neighbor_internal_id in visited_node_ids:
                    # Add backward edge if neighbor node is already visited
                    if edge_id not in visited_edges and sorted_pair not in visited_edge_pairs:
                        result.edges.append(edge)
                        visited_edges.add(edge_id)
                        visited_edge_pairs.add(sorted_pair)
                else:
                    if len(visited_node_ids) < max_nodes_int and current_depth < max_depth_int:
                        # Add new node to result and queue
                        result.nodes.append(neighbor_node)
                        visited_nodes.add(neighbor_entity_id)
                        visited_node_ids.add(neighbor_internal_id)

                        # Add node to queue with incremented depth
                        queue.append((neighbor_node, current_depth + 1))

                        # Add forward edge
                        if edge_id not in visited_edges and sorted_pair not in visited_edge_pairs:
                            result.edges.append(edge)
                            visited_edges.add(edge_id)
                            visited_edge_pairs.add(sorted_pair)
                    else:
                        if current_depth < max_depth_int:
                            result.is_truncated = True

        # Add db_degree to all nodes via bulk query
        if result.nodes:
            entity_ids = [self._normalize_node_id(node.labels[0]) for node in result.nodes]
            degree_params = {'params': json.dumps({'node_ids': entity_ids}, ensure_ascii=False)}
            degree_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND $node_ids AS entity_id
                MATCH (n:base {{entity_id: entity_id}})
                OPTIONAL MATCH (n)-[r]-()
                RETURN entity_id, count(r) as degree
            $$, $1::agtype) AS (entity_id text, degree bigint)"""
            degree_results = await self._query(degree_query, params=degree_params)
            degree_map = {row['entity_id']: int(row['degree']) for row in degree_results}
            # Update node properties with db_degree
            for node in result.nodes:
                entity_id = node.labels[0]
                node.properties['db_degree'] = degree_map.get(entity_id, 0)

        return result

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int | None = None,
        min_degree: int = 0,
        include_orphans: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return, Defaults to global_config max_graph_nodes
            min_degree: Minimum degree (connections) for nodes to be included. 0=all nodes
            include_orphans: Include orphan nodes (degree=0) even when min_degree > 0

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Use global_config max_graph_nodes as default if max_nodes is None
        max_nodes_default = int(self.global_config.get('max_graph_nodes', 1000))
        max_nodes = max_nodes_default if max_nodes is None else min(max_nodes, max_nodes_default)
        kg = KnowledgeGraph()

        # Handle wildcard query - get all nodes
        if node_label == '*':
            # First check total node count to determine if graph should be truncated
            count_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    RETURN count(distinct n) AS total_nodes
                    $$) AS (total_nodes bigint)"""

            count_result = await self._query(count_query)
            total_nodes = count_result[0]['total_nodes'] if count_result else 0
            is_truncated = total_nodes > max_nodes

            # Get max_nodes with highest degrees, applying min_degree filter
            # Build the degree filter condition
            if min_degree > 0:
                if include_orphans:
                    # Include nodes with degree >= min_degree OR degree = 0 (orphans)
                    degree_filter = f'WHERE degree >= {min_degree} OR degree = 0'
                else:
                    # Only include nodes with degree >= min_degree
                    degree_filter = f'WHERE degree >= {min_degree}'
            else:
                degree_filter = ''

            query_nodes = f"""SELECT * FROM (
                SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    OPTIONAL MATCH (n)-[r]->()
                    RETURN id(n) as node_id, count(r) as degree
                $$) AS (node_id BIGINT, degree BIGINT)
            ) AS subq
            {degree_filter}
            ORDER BY degree DESC
            LIMIT {max_nodes}"""
            node_results = await self._query(query_nodes)

            node_ids = [str(result['node_id']) for result in node_results]
            # Build degree map for db_degree property
            degree_map = {str(result['node_id']): int(result['degree']) for result in node_results}

            logger.info(f'[{self.workspace}] Total nodes: {total_nodes}, Selected nodes: {len(node_ids)}')

            if node_ids:
                cy_params = {'params': json.dumps({'node_ids': [int(n) for n in node_ids]}, ensure_ascii=False)}
                # Construct batch query for subgraph within max_nodes
                query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                        WITH $node_ids AS node_ids
                        MATCH (a)
                        WHERE id(a) IN node_ids
                        OPTIONAL MATCH (a)-[r]->(b)
                            WHERE id(b) IN node_ids
                        RETURN a, r, b
                    $$, $1::agtype) AS (a AGTYPE, r AGTYPE, b AGTYPE)"""
                results = await self._query(query, params=cy_params)

                # Process query results, deduplicate nodes and edges
                nodes_dict = {}
                edges_dict = {}
                for result in results:
                    # Process node a
                    if result.get('a') and isinstance(result['a'], dict):
                        node_a = result['a']
                        node_id = str(node_a['id'])
                        if node_id not in nodes_dict and 'properties' in node_a:
                            props = dict(node_a['properties'])
                            props['db_degree'] = degree_map.get(node_id, 0)
                            nodes_dict[node_id] = KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_a['properties']['entity_id']],
                                properties=props,
                            )

                    # Process node b
                    if result.get('b') and isinstance(result['b'], dict):
                        node_b = result['b']
                        node_id = str(node_b['id'])
                        if node_id not in nodes_dict and 'properties' in node_b:
                            props = dict(node_b['properties'])
                            props['db_degree'] = degree_map.get(node_id, 0)
                            nodes_dict[node_id] = KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_b['properties']['entity_id']],
                                properties=props,
                            )

                    # Process edge r
                    if result.get('r') and isinstance(result['r'], dict):
                        edge = result['r']
                        edge_id = str(edge['id'])
                        if edge_id not in edges_dict:
                            edges_dict[edge_id] = KnowledgeGraphEdge(
                                id=edge_id,
                                type=edge['label'],
                                source=str(edge['start_id']),
                                target=str(edge['end_id']),
                                properties=edge['properties'],
                            )

                kg = KnowledgeGraph(
                    nodes=list(nodes_dict.values()),
                    edges=list(edges_dict.values()),
                    is_truncated=is_truncated,
                )
            else:
                # For single node query, use BFS algorithm
                kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)

            logger.info(
                f'[{self.workspace}] Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}'
            )
        else:
            # For non-wildcard queries, use the BFS algorithm
            kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)
            logger.info(
                f"[{self.workspace}] Subgraph query for '{node_label}' successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
            )

        return kg

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        # Use native SQL to avoid Cypher wrapper overhead
        # Original: SELECT * FROM cypher(...) with MATCH (n:base)
        # Optimized: Direct table access for better performance
        query = f"""
            SELECT properties
            FROM {self.graph_name}.base
        """

        results = await self._query(query)
        nodes = []
        for result in results:
            if result.get('properties'):
                node_dict = result['properties']

                # Process string result, parse it to JSON dictionary
                if isinstance(node_dict, str):
                    try:
                        node_dict = json.loads(node_dict)
                    except json.JSONDecodeError:
                        logger.warning(f'[{self.workspace}] Failed to parse node string: {node_dict}')
                        continue

                # Add node id (entity_id) to the dictionary for easier access
                node_dict['id'] = node_dict.get('entity_id')
                nodes.append(node_dict)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
            (If 2 directional edges exist between the same pair of nodes, deduplication must be handled by the caller)
        """
        # Use native SQL to avoid Cartesian product (N×N) in Cypher MATCH
        # Original Cypher: MATCH (a:base)-[r]-(b:base) creates ~50 billion row combinations
        # Optimized: Start from edges table, join to nodes only to get entity_id
        # Performance: O(E) instead of O(N²), ~50,000x faster for large graphs
        query = f"""
            SELECT DISTINCT
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[a.properties, '"entity_id"'::agtype]))::text AS source,
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]))::text AS target,
                r.properties
            FROM {self.graph_name}."DIRECTED" r
            JOIN {self.graph_name}.base a ON r.start_id = a.id
            JOIN {self.graph_name}.base b ON r.end_id = b.id
        """

        results = await self._query(query)
        edges = []
        for result in results:
            edge_properties = result['properties']

            # Process string result, parse it to JSON dictionary
            if isinstance(edge_properties, str):
                try:
                    edge_properties = json.loads(edge_properties)
                except json.JSONDecodeError:
                    logger.warning(f'[{self.workspace}] Failed to parse edge properties string: {edge_properties}')
                    edge_properties = {}

            edge_properties['source'] = result['source']
            edge_properties['target'] = result['target']
            edges.append(edge_properties)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities) using native SQL for performance."""
        try:
            # Native SQL query to calculate node degrees directly from AGE's underlying tables
            # This is significantly faster than using the cypher() function wrapper
            query = f"""
            WITH node_degrees AS (
                SELECT
                    node_id,
                    COUNT(*) AS degree
                FROM (
                    SELECT start_id AS node_id FROM {self.graph_name}._ag_label_edge
                    UNION ALL
                    SELECT end_id AS node_id FROM {self.graph_name}._ag_label_edge
                ) AS all_edges
                GROUP BY node_id
            )
            SELECT
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[v.properties, '"entity_id"'::agtype]))::text AS label
            FROM
                node_degrees d
            JOIN
                {self.graph_name}._ag_label_vertex v ON d.node_id = v.id
            WHERE
                ag_catalog.agtype_access_operator(VARIADIC ARRAY[v.properties, '"entity_id"'::agtype]) IS NOT NULL
            ORDER BY
                d.degree DESC,
                label ASC
            LIMIT $1;
            """
            results = await self._query(query, params={'limit': limit})
            labels = [result['label'] for result in results if result and 'label' in result]

            logger.debug(f'[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})')
            return labels
        except Exception as e:
            logger.error(f'[{self.workspace}] Error getting popular labels: {e!s}')
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching using native, parameterized SQL for performance and security."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        try:
            # Re-implementing with the correct agtype access operator and full scoring logic.
            sql_query = f"""
            WITH ranked_labels AS (
                SELECT
                    (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text AS label,
                    LOWER((ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text) AS label_lower
                FROM
                    {self.graph_name}._ag_label_vertex
                WHERE
                    ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]) IS NOT NULL
                    AND LOWER((ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text) ILIKE $1
            )
            SELECT
                label
            FROM (
                SELECT
                    label,
                    CASE
                        WHEN label_lower = $2 THEN 1000
                        WHEN label_lower LIKE $3 THEN 500
                        ELSE (100 - LENGTH(label))
                    END +
                    CASE
                        WHEN label_lower LIKE $4 OR label_lower LIKE $5 THEN 50
                        ELSE 0
                    END AS score
                FROM
                    ranked_labels
            ) AS scored_labels
            ORDER BY
                score DESC,
                label ASC
            LIMIT $6;
            """
            params_tuple = (
                f'%{query_lower}%',  # For the main ILIKE clause ($1)
                query_lower,  # For exact match ($2)
                f'{query_lower}%',  # For prefix match ($3)
                f'% {query_lower}%',  # For word boundary (space) ($4)
                f'%_{query_lower}%',  # For word boundary (underscore) ($5)
                limit,  # For LIMIT ($6)
            )
            params = {str(i): v for i, v in enumerate(params_tuple, 1)}
            results = await self._query(sql_query, params=params)
            labels = [result['label'] for result in results if result and 'label' in result]

            logger.debug(f"[{self.workspace}] Search query '{query}' returned {len(labels)} results (limit: {limit})")
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error searching labels with query '{query}': {e!s}")
            return []

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            drop_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                            MATCH (n)
                            DETACH DELETE n
                            $$) AS (result agtype)"""

            await self._query(drop_query, readonly=False)
            return {
                'status': 'success',
                'message': f"workspace '{self.workspace}' graph data dropped",
            }
        except Exception as e:
            logger.error(f'[{self.workspace}] Error dropping graph: {e}')
            return {'status': 'error', 'message': str(e)}


# Note: Order matters! More specific namespaces (e.g., "full_entities") must come before
# more general ones (e.g., "entities") because is_namespace() uses endswith() matching
NAMESPACE_TABLE_MAP = {
    NameSpace.KV_STORE_FULL_DOCS: 'LIGHTRAG_DOC_FULL',
    NameSpace.KV_STORE_TEXT_CHUNKS: 'LIGHTRAG_DOC_CHUNKS',
    NameSpace.KV_STORE_FULL_ENTITIES: 'LIGHTRAG_FULL_ENTITIES',
    NameSpace.KV_STORE_FULL_RELATIONS: 'LIGHTRAG_FULL_RELATIONS',
    NameSpace.KV_STORE_ENTITY_CHUNKS: 'LIGHTRAG_ENTITY_CHUNKS',
    NameSpace.KV_STORE_RELATION_CHUNKS: 'LIGHTRAG_RELATION_CHUNKS',
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: 'LIGHTRAG_LLM_CACHE',
    NameSpace.VECTOR_STORE_CHUNKS: 'LIGHTRAG_VDB_CHUNKS',
    NameSpace.VECTOR_STORE_ENTITIES: 'LIGHTRAG_VDB_ENTITY',
    NameSpace.VECTOR_STORE_RELATIONSHIPS: 'LIGHTRAG_VDB_RELATION',
    NameSpace.DOC_STATUS: 'LIGHTRAG_DOC_STATUS',
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v
    return ''


TABLES = {
    'LIGHTRAG_DOC_FULL': {
        'ddl': """CREATE TABLE LIGHTRAG_DOC_FULL (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    doc_name VARCHAR(1024),
                    content TEXT,
                    meta JSONB,
                    s3_key TEXT NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_DOC_CHUNKS': {
        'ddl': """CREATE TABLE LIGHTRAG_DOC_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    file_path TEXT NULL,
                    s3_key TEXT NULL,
                    char_start INTEGER NULL,
                    char_end INTEGER NULL,
                    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_VDB_CHUNKS': {
        'ddl': f"""CREATE TABLE LIGHTRAG_VDB_CHUNKS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    full_doc_id VARCHAR(256),
                    chunk_order_index INTEGER,
                    tokens INTEGER,
                    content TEXT,
                    content_vector VECTOR({os.environ.get('EMBEDDING_DIM', 1024)}),
                    file_path TEXT NULL,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_VDB_ENTITY': {
        'ddl': f"""CREATE TABLE LIGHTRAG_VDB_ENTITY (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_name VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get('EMBEDDING_DIM', 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
                    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_VDB_RELATION': {
        'ddl': f"""CREATE TABLE LIGHTRAG_VDB_RELATION (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    source_id VARCHAR(512),
                    target_id VARCHAR(512),
                    content TEXT,
                    content_vector VECTOR({os.environ.get('EMBEDDING_DIM', 1024)}),
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    chunk_ids VARCHAR(255)[] NULL,
                    file_path TEXT NULL,
                    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_LLM_CACHE': {
        'ddl': """CREATE TABLE LIGHTRAG_LLM_CACHE (
                    workspace varchar(255) NOT NULL,
                    id varchar(255) NOT NULL,
                    original_prompt TEXT,
                    return_value TEXT,
                    chunk_id VARCHAR(255) NULL,
                    cache_type VARCHAR(32),
                    queryparam JSONB NULL,
                    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_DOC_STATUS': {
        'ddl': """CREATE TABLE LIGHTRAG_DOC_STATUS (
                   workspace varchar(255) NOT NULL,
                   id varchar(255) NOT NULL,
                   content_summary varchar(255) NULL,
                   content_length int4 NULL,
                   chunks_count int4 NULL,
                   status varchar(64) NULL,
                   file_path TEXT NULL,
                   chunks_list JSONB NULL DEFAULT '[]'::jsonb,
                   track_id varchar(255) NULL,
                   metadata JSONB NULL DEFAULT '{}'::jsonb,
                   error_msg TEXT NULL,
                   s3_key TEXT NULL,
                   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                   CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
                  )"""
    },
    'LIGHTRAG_FULL_ENTITIES': {
        'ddl': """CREATE TABLE LIGHTRAG_FULL_ENTITIES (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    entity_names JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_ENTITIES_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_FULL_RELATIONS': {
        'ddl': """CREATE TABLE LIGHTRAG_FULL_RELATIONS (
                    id VARCHAR(255),
                    workspace VARCHAR(255),
                    relation_pairs JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_FULL_RELATIONS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_ENTITY_CHUNKS': {
        'ddl': """CREATE TABLE LIGHTRAG_ENTITY_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_ENTITY_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_RELATION_CHUNKS': {
        'ddl': """CREATE TABLE LIGHTRAG_RELATION_CHUNKS (
                    id VARCHAR(512),
                    workspace VARCHAR(255),
                    chunk_ids JSONB,
                    count INTEGER,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_RELATION_CHUNKS_PK PRIMARY KEY (workspace, id)
                    )"""
    },
    'LIGHTRAG_ENTITY_ALIASES': {
        'ddl': """CREATE TABLE LIGHTRAG_ENTITY_ALIASES (
                    workspace VARCHAR(255),
                    alias VARCHAR(512),
                    canonical_entity VARCHAR(512),
                    method VARCHAR(50),
                    confidence FLOAT,
                    create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT LIGHTRAG_ENTITY_ALIASES_PK PRIMARY KEY (workspace, alias),
                    CONSTRAINT confidence_range CHECK (confidence >= 0 AND confidence <= 1)
                    )"""
    },
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    'get_by_id_full_docs': """SELECT id, COALESCE(content, '') as content,
                                COALESCE(doc_name, '') as file_path, s3_key
                                FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id=$2
                            """,
    'get_by_id_text_chunks': """SELECT id, tokens, COALESCE(content, '') as content,
                                chunk_order_index, full_doc_id, file_path, s3_key,
                                char_start, char_end,
                                COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id=$2
                            """,
    'get_by_id_llm_response_cache': """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id=$2
                               """,
    'get_by_ids_full_docs': """SELECT id, COALESCE(content, '') as content,
                                 COALESCE(doc_name, '') as file_path, s3_key
                                 FROM LIGHTRAG_DOC_FULL WHERE workspace=$1 AND id = ANY($2)
                            """,
    'get_by_ids_text_chunks': """SELECT id, tokens, COALESCE(content, '') as content,
                                  chunk_order_index, full_doc_id, file_path, s3_key,
                                  char_start, char_end,
                                  COALESCE(llm_cache_list, '[]'::jsonb) as llm_cache_list,
                                  EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                  EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                   FROM LIGHTRAG_DOC_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    'get_by_ids_llm_response_cache': """SELECT id, original_prompt, return_value, chunk_id, cache_type, queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_LLM_CACHE WHERE workspace=$1 AND id = ANY($2)
                                """,
    'get_by_id_full_entities': """SELECT id, entity_names, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id=$2
                               """,
    'get_by_id_full_relations': """SELECT id, relation_pairs, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id=$2
                               """,
    'get_by_ids_full_entities': """SELECT id, entity_names, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_FULL_ENTITIES WHERE workspace=$1 AND id = ANY($2)
                                """,
    'get_by_ids_full_relations': """SELECT id, relation_pairs, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_FULL_RELATIONS WHERE workspace=$1 AND id = ANY($2)
                                """,
    'get_by_id_entity_chunks': """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=$1 AND id=$2
                               """,
    'get_by_id_relation_chunks': """SELECT id, chunk_ids, count,
                                EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=$1 AND id=$2
                               """,
    'get_by_ids_entity_chunks': """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_ENTITY_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    'get_by_ids_relation_chunks': """SELECT id, chunk_ids, count,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT as create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT as update_time
                                 FROM LIGHTRAG_RELATION_CHUNKS WHERE workspace=$1 AND id = ANY($2)
                                """,
    'filter_keys': 'SELECT id FROM {table_name} WHERE workspace=$1 AND id IN ({ids})',
    'upsert_doc_full': """INSERT INTO LIGHTRAG_DOC_FULL (id, content, doc_name, workspace, s3_key)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (workspace,id) DO UPDATE
                           SET content = $2,
                               doc_name = $3,
                               s3_key = COALESCE($5, LIGHTRAG_DOC_FULL.s3_key),
                               update_time = CURRENT_TIMESTAMP
                       """,
    'upsert_llm_response_cache': """INSERT INTO LIGHTRAG_LLM_CACHE(workspace,id,original_prompt,return_value,chunk_id,cache_type,queryparam)
                                      VALUES ($1, $2, $3, $4, $5, $6, $7)
                                      ON CONFLICT (workspace,id) DO UPDATE
                                      SET original_prompt = EXCLUDED.original_prompt,
                                      return_value=EXCLUDED.return_value,
                                      chunk_id=EXCLUDED.chunk_id,
                                      cache_type=EXCLUDED.cache_type,
                                      queryparam=EXCLUDED.queryparam,
                                      update_time = CURRENT_TIMESTAMP
                                     """,
    'upsert_text_chunk': """INSERT INTO LIGHTRAG_DOC_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, file_path, s3_key,
                      char_start, char_end, llm_cache_list, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      file_path=EXCLUDED.file_path,
                      s3_key=COALESCE(EXCLUDED.s3_key, LIGHTRAG_DOC_CHUNKS.s3_key),
                      char_start=EXCLUDED.char_start,
                      char_end=EXCLUDED.char_end,
                      llm_cache_list=EXCLUDED.llm_cache_list,
                      update_time = EXCLUDED.update_time
                     """,
    'upsert_full_entities': """INSERT INTO LIGHTRAG_FULL_ENTITIES (workspace, id, entity_names, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_names=EXCLUDED.entity_names,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    'upsert_full_relations': """INSERT INTO LIGHTRAG_FULL_RELATIONS (workspace, id, relation_pairs, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET relation_pairs=EXCLUDED.relation_pairs,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    'upsert_entity_chunks': """INSERT INTO LIGHTRAG_ENTITY_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    'upsert_relation_chunks': """INSERT INTO LIGHTRAG_RELATION_CHUNKS (workspace, id, chunk_ids, count,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET chunk_ids=EXCLUDED.chunk_ids,
                      count=EXCLUDED.count,
                      update_time = EXCLUDED.update_time
                     """,
    # SQL for VectorStorage
    'upsert_chunk': """INSERT INTO LIGHTRAG_VDB_CHUNKS (workspace, id, tokens,
                      chunk_order_index, full_doc_id, content, content_vector, file_path,
                      create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET tokens=EXCLUDED.tokens,
                      chunk_order_index=EXCLUDED.chunk_order_index,
                      full_doc_id=EXCLUDED.full_doc_id,
                      content = EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    'upsert_entity': """INSERT INTO LIGHTRAG_VDB_ENTITY (workspace, id, entity_name, content,
                      content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6::varchar[], $7, $8, $9)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET entity_name=EXCLUDED.entity_name,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time=EXCLUDED.update_time
                     """,
    'upsert_relationship': """INSERT INTO LIGHTRAG_VDB_RELATION (workspace, id, source_id,
                      target_id, content, content_vector, chunk_ids, file_path, create_time, update_time)
                      VALUES ($1, $2, $3, $4, $5, $6, $7::varchar[], $8, $9, $10)
                      ON CONFLICT (workspace,id) DO UPDATE
                      SET source_id=EXCLUDED.source_id,
                      target_id=EXCLUDED.target_id,
                      content=EXCLUDED.content,
                      content_vector=EXCLUDED.content_vector,
                      chunk_ids=EXCLUDED.chunk_ids,
                      file_path=EXCLUDED.file_path,
                      update_time = EXCLUDED.update_time
                     """,
    'relationships': """
                     SELECT r.source_id AS src_id,
                            r.target_id AS tgt_id,
                            EXTRACT(EPOCH FROM r.create_time)::BIGINT AS created_at
                     FROM LIGHTRAG_VDB_RELATION r
                     WHERE r.workspace = $1
                       AND r.content_vector <=> '[{embedding_string}]'::vector < $2
                     ORDER BY r.content_vector <=> '[{embedding_string}]'::vector
                     LIMIT $3;
                     """,
    'entities': """
                SELECT e.entity_name,
                       EXTRACT(EPOCH FROM e.create_time)::BIGINT AS created_at
                FROM LIGHTRAG_VDB_ENTITY e
                WHERE e.workspace = $1
                  AND e.content_vector <=> '[{embedding_string}]'::vector < $2
                ORDER BY e.content_vector <=> '[{embedding_string}]'::vector
                LIMIT $3;
                """,
    'chunks': """
              SELECT c.id,
                     c.content,
                     c.file_path,
                     EXTRACT(EPOCH FROM c.create_time)::BIGINT AS created_at
              FROM LIGHTRAG_VDB_CHUNKS c
              WHERE c.workspace = $1
                AND c.content_vector <=> '[{embedding_string}]'::vector < $2
              ORDER BY c.content_vector <=> '[{embedding_string}]'::vector
              LIMIT $3;
              """,
    # DROP tables
    'drop_specifiy_table_workspace': """
        DELETE FROM {table_name} WHERE workspace=$1
       """,
    # Entity alias cache
    'get_alias': """
        SELECT canonical_entity, method, confidence
        FROM LIGHTRAG_ENTITY_ALIASES
        WHERE workspace=$1 AND alias=$2
        """,
    'upsert_alias': """
        INSERT INTO LIGHTRAG_ENTITY_ALIASES
            (workspace, alias, canonical_entity, method, confidence, create_time, update_time)
        VALUES ($1, $2, $3, $4, $5, $6, $6)
        ON CONFLICT (workspace, alias) DO UPDATE SET
            canonical_entity = EXCLUDED.canonical_entity,
            method = EXCLUDED.method,
            confidence = EXCLUDED.confidence,
            update_time = CURRENT_TIMESTAMP
        """,
    'get_aliases_for_canonical': """
        SELECT alias, method, confidence
        FROM LIGHTRAG_ENTITY_ALIASES
        WHERE workspace=$1 AND canonical_entity=$2
        """,
    # Orphan connection queries
    'get_orphan_entities': """
        SELECT e.id, e.entity_name, e.content, e.content_vector
        FROM LIGHTRAG_VDB_ENTITY e
        WHERE e.workspace = $1
          AND NOT EXISTS (
              SELECT 1 FROM LIGHTRAG_VDB_RELATION r
              WHERE r.workspace = $1
                AND (r.source_id = e.entity_name OR r.target_id = e.entity_name)
          )
        """,
    # Get entities with degree <= max_degree (sparse entities)
    # $1 = workspace, $2 = max_degree
    'get_sparse_entities': """
        SELECT e.id, e.entity_name, e.content, e.content_vector, COALESCE(degree_counts.degree, 0) as degree
        FROM LIGHTRAG_VDB_ENTITY e
        LEFT JOIN (
            SELECT entity_name, COUNT(*) as degree
            FROM (
                SELECT source_id as entity_name FROM LIGHTRAG_VDB_RELATION WHERE workspace = $1
                UNION ALL
                SELECT target_id as entity_name FROM LIGHTRAG_VDB_RELATION WHERE workspace = $1
            ) as edges
            GROUP BY entity_name
        ) degree_counts ON e.entity_name = degree_counts.entity_name
        WHERE e.workspace = $1
          AND COALESCE(degree_counts.degree, 0) <= $2
        ORDER BY COALESCE(degree_counts.degree, 0) ASC, e.entity_name
        """,
    'get_orphan_candidates': """
        SELECT e.id, e.entity_name, e.content,
               1 - (e.content_vector <=> $2::vector) AS similarity
        FROM LIGHTRAG_VDB_ENTITY e
        WHERE e.workspace = $1
          AND e.entity_name != $3
          AND 1 - (e.content_vector <=> $2::vector) >= $4
        ORDER BY e.content_vector <=> $2::vector
        LIMIT $5
        """,
    'get_connected_candidates': """
        SELECT e.id, e.entity_name, e.content,
               1 - (e.content_vector <=> $2::vector) AS similarity
        FROM LIGHTRAG_VDB_ENTITY e
        WHERE e.workspace = $1
          AND e.entity_name != $3
          AND 1 - (e.content_vector <=> $2::vector) >= $4
          AND EXISTS (
              SELECT 1 FROM LIGHTRAG_VDB_RELATION r
              WHERE r.workspace = $1
                AND (r.source_id = e.entity_name OR r.target_id = e.entity_name)
          )
        ORDER BY e.content_vector <=> $2::vector
        LIMIT $5
        """,
}
