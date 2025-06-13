"""
Query logging system for LightRAG with rotation, archiving, and detailed tracking.
"""

import asyncio
import gzip
import json
import shutil
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from lightrag.utils import logger as lightrag_logger


class LogLevel(Enum):
    """Query log detail levels."""

    MINIMAL = "minimal"  # Just query and response
    STANDARD = "standard"  # Add timing and basic metadata
    VERBOSE = "verbose"  # Full retrieval details


class QueryLogger:
    """
    Async-safe query logger with rotation and archiving capabilities.
    """

    def __init__(
        self,
        log_file_path: str = "lightrag_queries.log",
        max_file_size_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        log_level: LogLevel = LogLevel.STANDARD,
        archive_dir: Optional[str] = None,
        retention_days: Optional[int] = None,
    ):
        """
        Initialize the query logger.

        Args:
            log_file_path: Path to the main log file
            max_file_size_bytes: Maximum size before rotation
            backup_count: Number of backup files to keep
            log_level: Detail level for logging
            archive_dir: Directory for compressed archives
            retention_days: Days to retain archives (None = forever)
        """
        self.log_file_path = Path(log_file_path)
        self.max_file_size_bytes = max_file_size_bytes
        self.backup_count = backup_count
        self.log_level = log_level
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self.retention_days = retention_days

        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create archive directory if specified
        if self.archive_dir:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_errors": 0,
            "start_time": datetime.now().isoformat(),
        }

        lightrag_logger.info(f"QueryLogger initialized: {self.log_file_path}")

    async def log_query(
        self,
        query_text: str,
        response_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_parameters: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None,
        tokens_processed: Optional[int] = None,
        error_message: Optional[str] = None,
        retrieval_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Log a query with configurable detail level.

        Args:
            query_text: The user's query
            response_text: The generated response
            user_id: Optional user identifier
            session_id: Optional session identifier
            query_parameters: Query configuration parameters
            response_time_ms: Response generation time in milliseconds
            tokens_processed: Number of tokens processed
            error_message: Error message if query failed
            retrieval_details: Detailed retrieval information
        """
        async with self._lock:
            # Check if rotation is needed
            await self._check_rotation()

            # Build log entry based on level
            log_entry = self._build_log_entry(
                query_text=query_text,
                response_text=response_text,
                user_id=user_id,
                session_id=session_id,
                query_parameters=query_parameters,
                response_time_ms=response_time_ms,
                tokens_processed=tokens_processed,
                error_message=error_message,
                retrieval_details=retrieval_details,
            )

            # Write to file
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # Update statistics
                self.stats["total_queries"] += 1
                if error_message:
                    self.stats["total_errors"] += 1

            except Exception as e:
                lightrag_logger.error(f"Failed to write query log: {e}")

    def _build_log_entry(
        self,
        query_text: str,
        response_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_parameters: Optional[Dict[str, Any]] = None,
        response_time_ms: Optional[float] = None,
        tokens_processed: Optional[int] = None,
        error_message: Optional[str] = None,
        retrieval_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build log entry based on configured log level."""

        # Base entry (MINIMAL level)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query_text,

            "response": (
                response_text[:500]
                if self.log_level == LogLevel.MINIMAL
                else response_text
            ),

        }

        # Add standard fields (STANDARD and VERBOSE levels)
        if self.log_level in [LogLevel.STANDARD, LogLevel.VERBOSE]:
            entry.update(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "response_time_ms": response_time_ms,
                    "tokens_processed": tokens_processed,
                    "success": error_message is None,
                    "error": error_message,
                }
            )

            # Add basic query parameters
            if query_parameters:
                entry["query_params"] = {
                    "mode": query_parameters.get("mode"),
                    "top_k": query_parameters.get("top_k"),
                    "response_type": query_parameters.get("response_type"),
                }

        # Add detailed fields (VERBOSE level only)
        if self.log_level == LogLevel.VERBOSE:
            if query_parameters:
                entry["query_params"] = query_parameters  # Full parameters

            if retrieval_details:
                entry["retrieval_details"] = retrieval_details

        return entry

    async def _check_rotation(self):
        """Check if log rotation is needed and perform if necessary."""
        if not self.log_file_path.exists():
            return

        file_size = self.log_file_path.stat().st_size
        if file_size >= self.max_file_size_bytes:
            await self._rotate_logs()

    async def _rotate_logs(self):
        """Rotate log files and optionally archive old ones."""
        lightrag_logger.info(f"Rotating query logs: {self.log_file_path}")

        # Shift existing backup files
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = self.log_file_path.with_suffix(f".{i}.log")
            new_backup = self.log_file_path.with_suffix(f".{i + 1}.log")
            if old_backup.exists():
                if i == self.backup_count - 1 and self.archive_dir:
                    # Archive the oldest backup
                    await self._archive_log(old_backup)
                else:
                    old_backup.rename(new_backup)

        # Rename current log to .1.log
        if self.log_file_path.exists():
            self.log_file_path.rename(self.log_file_path.with_suffix(".1.log"))

        # Clean up old archives if retention is set
        if self.archive_dir and self.retention_days:
            await self._cleanup_old_archives()

    async def _archive_log(self, log_path: Path):
        """Compress and archive a log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"query_log_{timestamp}.log.gz"
        archive_path = self.archive_dir / archive_name

        try:
            with open(log_path, "rb") as f_in:
                with gzip.open(archive_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            log_path.unlink()  # Remove original file
            lightrag_logger.info(f"Archived log to: {archive_path}")

        except Exception as e:
            lightrag_logger.error(f"Failed to archive log: {e}")

    async def _cleanup_old_archives(self):
        """Remove archives older than retention_days."""
        if not self.archive_dir or not self.retention_days:
            return

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for archive_file in self.archive_dir.glob("query_log_*.log.gz"):
            try:
                # Extract timestamp from filename
                timestamp_str = archive_file.stem.split("_", 2)[2].split(".")[0]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if file_date < cutoff_date:
                    archive_file.unlink()
                    lightrag_logger.info(f"Removed old archive: {archive_file}")

            except Exception as e:
                lightrag_logger.warning(f"Error processing archive {archive_file}: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get query logger statistics."""
        async with self._lock:
            stats = self.stats.copy()

            # Add current file info
            if self.log_file_path.exists():
                stats["current_file_size_mb"] = self.log_file_path.stat().st_size / (
                    1024 * 1024
                )
                stats["current_file_path"] = str(self.log_file_path)

            # Count backup files
            backup_count = sum(
                1
                for i in range(1, self.backup_count + 1)
                if self.log_file_path.with_suffix(f".{i}.log").exists()
            )
            stats["backup_files_count"] = backup_count

            # Count archives
            if self.archive_dir and self.archive_dir.exists():
                stats["archived_files_count"] = len(
                    list(self.archive_dir.glob("query_log_*.log.gz"))
                )

            return stats

    async def search_queries(
        self,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_errors_only: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search through query logs with various filters.

        Args:
            search_term: Text to search for in queries/responses
            user_id: Filter by user ID
            session_id: Filter by session ID
            start_date: Filter by start date
            end_date: Filter by end date
            include_errors_only: Only return failed queries
            limit: Maximum number of results

        Returns:
            List of matching log entries
        """
        results = []
        files_to_search = [self.log_file_path]

        # Include backup files
        for i in range(1, self.backup_count + 1):
            backup_file = self.log_file_path.with_suffix(f".{i}.log")
            if backup_file.exists():
                files_to_search.append(backup_file)

        for file_path in files_to_search:
            if not file_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        try:
                            entry = json.loads(line.strip())

                            # Apply filters
                            if not self._matches_filters(
                                entry,
                                search_term,
                                user_id,
                                session_id,
                                start_date,
                                end_date,
                                include_errors_only,
                            ):
                                continue

                            results.append(entry)

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                lightrag_logger.error(f"Error searching log file {file_path}: {e}")

        return results

    def _matches_filters(
        self,
        entry: Dict[str, Any],
        search_term: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        include_errors_only: bool,
    ) -> bool:
        """Check if a log entry matches all specified filters."""

        # Check error filter
        if include_errors_only and entry.get("success", True):
            return False

        # Check user/session filters
        if user_id and entry.get("user_id") != user_id:
            return False
        if session_id and entry.get("session_id") != session_id:
            return False

        # Check date filters
        if start_date or end_date:
            entry_date = datetime.fromisoformat(entry["timestamp"])
            if start_date and entry_date < start_date:
                return False
            if end_date and entry_date > end_date:
                return False

        # Check search term
        if search_term:
            search_term_lower = search_term.lower()
            searchable_text = (
                entry.get("query", "").lower() + " " + entry.get("response", "").lower()
            )
            if search_term_lower not in searchable_text:
                return False

        return True


# Singleton instance management
_query_logger_instance: Optional[QueryLogger] = None
_query_logger_lock = asyncio.Lock()


async def get_query_logger(
    log_file_path: str = "lightrag_queries.log",
    max_file_size_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    log_level: LogLevel = LogLevel.STANDARD,
    archive_dir: Optional[str] = None,
    retention_days: Optional[int] = None,
) -> QueryLogger:
    """
    Get or create a singleton QueryLogger instance.

    This ensures only one logger instance exists per configuration.
    """
    global _query_logger_instance

    async with _query_logger_lock:
        if _query_logger_instance is None:
            _query_logger_instance = QueryLogger(
                log_file_path=log_file_path,
                max_file_size_bytes=max_file_size_bytes,
                backup_count=backup_count,
                log_level=log_level,
                archive_dir=archive_dir,
                retention_days=retention_days,
            )

        return _query_logger_instance
