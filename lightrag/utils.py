from __future__ import annotations
import weakref

import sys

import asyncio
import contextvars
import html
import csv
import inspect
import json
import logging
import logging.handlers
import math
import os
import re
import time
import uuid
import threading
import warnings
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from hashlib import md5
from pathlib import Path
from typing import (
    Any,
    Protocol,
    Callable,
    TYPE_CHECKING,
    List,
    Optional,
    Iterable,
    Sequence,
    Collection,
)
import numpy as np
from dotenv import load_dotenv
import json_repair

from lightrag.exceptions import ChunkBlockMatchError, EmptyTruncatedResponseError
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_PROCESSING_PRIORITY,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    VALID_SOURCE_IDS_LIMIT_METHODS,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
    PARSED_DIR_NAME,
    DEFAULT_GLOBAL_SLOT_POLL_MIN,
    DEFAULT_GLOBAL_SLOT_POLL_DEFERRED_MAX,
    DEFAULT_GLOBAL_SLOT_DRAIN_LIMIT,
    DEFAULT_ZOMBIE_COMPACT_THRESHOLD,
    DEFAULT_COMPACT_BATCH_LIMIT,
    DEFAULT_QUEUE_STATS_MIN_PUBLISH_INTERVAL,
)

# Precompile regex pattern for JSON sanitization (module-level, compiled once)
_SURROGATE_PATTERN = re.compile(r"[\uD800-\uDFFF\uFFFE\uFFFF]")
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that gracefully handles closed streams during shutdown.

    This handler prevents "ValueError: I/O operation on closed file" errors
    that can occur when pytest or other test frameworks close stdout/stderr
    before Python's logging cleanup runs.
    """

    def flush(self):
        """Flush the stream, ignoring errors if the stream is closed."""
        try:
            super().flush()
        except (ValueError, OSError):
            # Stream is closed or otherwise unavailable, silently ignore
            pass

    def close(self):
        """Close the handler, ignoring errors if the stream is already closed."""
        try:
            super().close()
        except (ValueError, OSError):
            # Stream is closed or otherwise unavailable, silently ignore
            pass


# Initialize logger with basic configuration
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root logger
logger.setLevel(logging.INFO)

# Add console handler if no handlers exist
if not logger.handlers:
    console_handler = SafeStreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


def _patch_ascii_colors_console_handler() -> None:
    """Prevent ascii_colors from printing flush errors during interpreter exit."""

    try:
        from ascii_colors import ConsoleHandler
    except ImportError:
        return

    if getattr(ConsoleHandler, "_lightrag_patched", False):
        return

    original_handle_error = ConsoleHandler.handle_error

    def _safe_handle_error(self, message: str) -> None:  # type: ignore[override]
        exc_type, _, _ = sys.exc_info()
        if exc_type in (ValueError, OSError) and "close" in message.lower():
            return
        original_handle_error(self, message)

    ConsoleHandler.handle_error = _safe_handle_error  # type: ignore[assignment]
    ConsoleHandler._lightrag_patched = True  # type: ignore[attr-defined]


_patch_ascii_colors_console_handler()


# Global import for pypinyin with startup-time logging
try:
    import pypinyin

    _PYPINYIN_AVAILABLE = True
    # logger.info("pypinyin loaded successfully for Chinese pinyin sorting")
except ImportError:
    pypinyin = None
    _PYPINYIN_AVAILABLE = False
    logger.warning(
        "pypinyin is not installed. Chinese pinyin sorting will use simple string sorting."
    )


async def safe_vdb_operation_with_exception(
    operation: Callable,
    operation_name: str,
    entity_name: str = "",
    max_retries: int = 3,
    retry_delay: float = 0.2,
    logger_func: Optional[Callable] = None,
    timeout_seconds: float | None = None,
    log_start: bool = False,
    success_log_threshold_seconds: float = 10.0,
) -> None:
    """
    Safely execute vector database operations with retry mechanism and exception handling.

    This function ensures that VDB operations are executed with proper error handling
    and retry logic. If all retries fail, it raises an exception to maintain data consistency.

    Args:
        operation: The async operation to execute
        operation_name: Operation name for logging purposes
        entity_name: Entity name for logging purposes
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        logger_func: Logger function to use for error messages
        timeout_seconds: Optional timeout for a single operation attempt
        log_start: Whether to emit start/success logs for each attempt
        success_log_threshold_seconds: Log successful attempts when duration exceeds this threshold

    Raises:
        Exception: When operation fails after all retry attempts
    """
    log_func = logger_func or logger.warning

    for attempt in range(max_retries):
        start_ts = time.perf_counter()
        attempt_label = f"{attempt + 1}/{max_retries}"
        try:
            if log_start:
                logger.info(
                    "VDB %s start for %s (attempt %s, timeout=%s)",
                    operation_name,
                    entity_name or "<unknown>",
                    attempt_label,
                    f"{timeout_seconds:.1f}s"
                    if timeout_seconds is not None
                    else "none",
                )

            if timeout_seconds is not None and timeout_seconds > 0:
                await asyncio.wait_for(operation(), timeout=timeout_seconds)
            else:
                await operation()

            elapsed = time.perf_counter() - start_ts
            if log_start or elapsed >= success_log_threshold_seconds:
                logger.info(
                    "VDB %s success for %s in %.2fs (attempt %s)",
                    operation_name,
                    entity_name or "<unknown>",
                    elapsed,
                    attempt_label,
                )
            return  # Success, return immediately
        except asyncio.TimeoutError as e:
            elapsed = time.perf_counter() - start_ts
            timeout_msg = (
                f"VDB {operation_name} timeout for {entity_name or '<unknown>'} "
                f"after {elapsed:.2f}s (attempt {attempt_label}, timeout={timeout_seconds}s)"
            )
            if attempt >= max_retries - 1:
                log_func(timeout_msg)
                raise TimeoutError(timeout_msg) from e
            log_func(f"{timeout_msg}, retrying...")
            if retry_delay > 0:
                await asyncio.sleep(retry_delay)
        except Exception as e:
            elapsed = time.perf_counter() - start_ts
            if attempt >= max_retries - 1:
                error_msg = (
                    f"VDB {operation_name} failed for {entity_name or '<unknown>'} "
                    f"after {max_retries} attempts in {elapsed:.2f}s: {e}"
                )
                log_func(error_msg)
                raise Exception(error_msg) from e
            else:
                log_func(
                    f"VDB {operation_name} attempt {attempt + 1} failed for "
                    f"{entity_name or '<unknown>'} after {elapsed:.2f}s: {e}, retrying..."
                )
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay)


def parse_optional_float(raw: str | None) -> float | None:
    """Decode env strings (or any text) into ``float | None``.

    Empty string and the literal ``"None"`` (case-insensitive) collapse
    to ``None`` so users can leave a knob un-set in ``.env`` and have
    the consuming code fall back to its own default.  Any other
    non-numeric value raises :class:`ValueError` so misconfigured envs
    fail loudly at parse time rather than silently downstream.
    Non-finite values (``nan`` / ``inf``) are also rejected: they parse as
    floats but corrupt semantic-chunker thresholds.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped or stripped.lower() == "none":
        return None
    value = float(stripped)
    if not math.isfinite(value):
        raise ValueError(f"expected a finite float, got {raw!r}")
    return value


def validate_file_path_security(file_path_str: str, base_dir: Path) -> Optional[Path]:
    """
    Validate file path security to prevent Path Traversal attacks.

    Args:
        file_path_str: The file path string to validate
        base_dir: The base directory that the file must be within

    Returns:
        Path: Safe file path (resolved, contained to ``base_dir``) if valid;
            None if unsafe, malformed, or unresolvable. Does NOT check
            existence — a returned path is guaranteed inside ``base_dir`` but
            may not exist; the caller decides what "inside but absent" means.
    """
    if not file_path_str or not file_path_str.strip():
        return None

    try:
        # Clean the file path string
        clean_path_str = file_path_str.strip()

        # Check for obvious path traversal patterns before processing
        # This catches both Unix (..) and Windows (..\) style traversals
        if ".." in clean_path_str:
            # Additional check for Windows-style backslash traversal
            if (
                "\\..\\" in clean_path_str
                or clean_path_str.startswith("..\\")
                or clean_path_str.endswith("\\..")
            ):
                return None

        # Normalize path separators (convert backslashes to forward slashes)
        # This helps handle Windows-style paths on Unix systems
        normalized_path = clean_path_str.replace("\\", "/")

        # Create path object and resolve it (handles symlinks and relative paths)
        candidate_path = (base_dir / normalized_path).resolve()
        base_dir_resolved = base_dir.resolve()

        # Check if the resolved path is within the base directory
        if not candidate_path.is_relative_to(base_dir_resolved):
            return None

        return candidate_path

    except Exception as e:
        logger.warning(f"Invalid file path detected: {file_path_str} - {str(e)}")
        return None


def get_env_value(
    env_key: str, default: any, value_type: type = str, special_none: bool = False
) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to
        special_none (bool): If True, return None when value is "None"

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    # Handle special case for "None" string
    if special_none and value == "None":
        return None

    if value_type is bool:
        return value.strip().lower() in ("true", "1", "yes", "t", "on")

    # Handle list type with JSON parsing
    if value_type is list:
        try:
            import json

            parsed_value = json.loads(value)
            # Ensure the parsed value is actually a list
            if isinstance(parsed_value, list):
                return parsed_value
            else:
                logger.warning(
                    f"Environment variable {env_key} is not a valid JSON list, using default"
                )
                return default
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse {env_key} as JSON list: {e}, using default"
            )
            return default

    try:
        converted = value_type(value)
    except (ValueError, TypeError):
        return default

    # Reject non-finite floats so env-backed thresholds cannot become NaN/Inf.
    if (
        value_type is float
        and isinstance(converted, float)
        and not math.isfinite(converted)
    ):
        logger.warning(
            "Environment variable %s=%r is not a finite float, using default",
            env_key,
            value,
        )
        return default
    return converted


# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from lightrag.base import BaseKVStorage, BaseVectorStorage, QueryParam

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"
PERFORMANCE_TIMING_LOGS = (
    os.getenv("LIGHTRAG_PERFORMANCE_TIMING_LOGS", "false").lower() == "true"
)


def verbose_debug(msg: str, *args, **kwargs):
    """Function for outputting detailed debug information.
    When VERBOSE_DEBUG=True, outputs the complete message.
    When VERBOSE_DEBUG=False, outputs only the first 50 characters.

    Args:
        msg: The message format string
        *args: Arguments to be formatted into the message
        **kwargs: Keyword arguments passed to logger.debug()
    """
    if VERBOSE_DEBUG:
        logger.debug(msg, *args, **kwargs)
    else:
        # Format the message with args first
        if args:
            formatted_msg = msg % args
        else:
            formatted_msg = msg
        # Then truncate the formatted message
        truncated_msg = (
            formatted_msg[:150] + "..." if len(formatted_msg) > 150 else formatted_msg
        )
        # Remove consecutive newlines
        truncated_msg = re.sub(r"\n+", "\n", truncated_msg)
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    """Enable or disable verbose debug output"""
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


def performance_timing_log(msg: str, *args, **kwargs):
    """Emit targeted performance timing logs only when explicitly enabled."""
    if PERFORMANCE_TIMING_LOGS:
        logger.info(msg, *args, **kwargs)


def safe_log_value(value: str, max_length: int = 200) -> str:
    """Make a caller-supplied value safe to embed in a log line.

    Replaces non-printable characters -- notably CR/LF, which could otherwise
    be used to forge extra log lines (log injection) -- with '?' and truncates
    over-long values. Only the *logged* form is sanitized; the caller keeps the
    original value for its own logic.

    Used wherever untrusted input reaches the log: rate-limit keys built from a
    submitted username, and operator-supplied identifiers in audited admin
    actions (source-conflict repair).
    """
    sanitized = "".join(ch if ch.isprintable() else "?" for ch in value)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "…(truncated)"
    return sanitized


statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""

    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = [
            "/documents",
            "/documents/paginated",
            "/health",
            "/webui/",
            "/documents/pipeline_status",
        ]
        # self.filtered_paths = ["/health", "/webui/"]

    def filter(self, record):
        try:
            # Check if record has the required attributes for an access log
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            # Extract method, path and status from the record args
            method = record.args[1]
            path = record.args[2]
            status = record.args[4]

            # Filter out successful GET/POST requests to filtered paths
            if (
                (method == "GET" or method == "POST")
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True
        except Exception:
            # In case of any error, let the message through
            return True


def setup_logger(
    logger_name: str,
    level: str = "INFO",
    add_filter: bool = False,
    log_file_path: str | None = None,
    enable_file_logging: bool = True,
):
    """Set up a logger with console and optionally file handlers

    Args:
        logger_name: Name of the logger to set up
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        add_filter: Whether to add LightragPathFilter to the logger
        log_file_path: Path to the log file. If None and file logging is enabled, defaults to lightrag.log in LOG_DIR or cwd
        enable_file_logging: Whether to enable logging to a file (defaults to True)
    """
    # Configure formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(level)
    logger_instance.handlers = []  # Clear existing handlers
    logger_instance.propagate = False

    # Add console handler with safe stream handling
    console_handler = SafeStreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    logger_instance.addHandler(console_handler)

    # Add file handler by default unless explicitly disabled
    if enable_file_logging:
        # Get log file path
        if log_file_path is None:
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Get log file max size and backup count from environment variables
        log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
        log_backup_count = get_env_value(
            "LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int
        )

        try:
            # Add file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file_path,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(level)
            logger_instance.addHandler(file_handler)
        except PermissionError as e:
            logger.warning(f"Could not create log file at {log_file_path}: {str(e)}")
            logger.warning("Continuing with console logging only")

    # Add path filter if requested
    if add_filter:
        path_filter = LightragPathFilter()
        logger_instance.addFilter(path_filter)


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


@dataclass
class TaskState:
    """Task state tracking for priority queue management"""

    future: asyncio.Future
    start_time: float
    execution_start_time: float = None
    worker_started: bool = False
    cancellation_requested: bool = False
    cleanup_done: bool = False


@dataclass
class EmbeddingFunc:
    """Embedding function wrapper with dimension validation

    This class wraps an embedding function to ensure that the output embeddings have the correct dimension.
    If wrapped multiple times, the inner wrappers will be automatically unwrapped to prevent
    configuration conflicts where inner wrapper settings would override outer wrapper settings.

    Using functools.partial for parameter binding:
        A common pattern is to use functools.partial to pre-bind model and host parameters
        to an embedding function. When the base embedding function is already decorated with
        @wrap_embedding_func_with_attrs (e.g., ollama_embed), use `.func` to access the
        original unwrapped function to avoid double wrapping:

        Example:
            from functools import partial

            # ❌ Wrong - causes double wrapping (inner EmbeddingFunc still executes)
            func=partial(ollama_embed, embed_model="bge-m3:latest", host="http://localhost:11434")

            # ✅ Correct - access the unwrapped function via .func
            func=partial(ollama_embed.func, embed_model="bge-m3:latest", host="http://localhost:11434")

    Context-aware embedding:
        The wrapper supports passing a 'context' parameter to distinguish between query and document
        embeddings. This allows wrapped functions to apply different processing (e.g., prefixes,
        different models) based on the context:

        Example:
            embeddings = await embed_func(texts, context="document")  # For indexing
            embeddings = await embed_func([query], context="query")   # For search

    Args:
        embedding_dim: Expected dimension of the embeddings(For dimension checking and workspace data isolation in vector DB)
        func: The actual embedding function to wrap
        max_token_size: Enable embedding token limit checking for description summarization(Set embedding_token_limit in LightRAG)
        send_dimensions: Whether to inject embedding_dim argument to underlying function
        model_name: Model name for implementing workspace data isolation in vector DB
        supports_asymmetric: Whether the underlying function supports context parameter so it can be injected
    """

    embedding_dim: int
    func: callable
    max_token_size: int | None = None
    send_dimensions: bool = False
    model_name: str | None = (
        None  # Model name for implementing workspace data isolation in vector DB
    )
    supports_asymmetric: bool = (
        False  # Whether underlying function accepts context parameter
    )

    def __post_init__(self):
        """Unwrap nested EmbeddingFunc to prevent double wrapping issues.

        When an EmbeddingFunc wraps another EmbeddingFunc, the inner wrapper's
        __call__ preprocessing would override the outer wrapper's settings.
        This method detects and unwraps nested EmbeddingFunc instances to ensure
        that only the outermost wrapper's configuration is applied.
        """
        # Check if func is already an EmbeddingFunc instance and unwrap it
        max_unwrap_depth = 3  # Safety limit to prevent infinite loops
        unwrap_count = 0
        while isinstance(self.func, EmbeddingFunc):
            unwrap_count += 1
            if unwrap_count > max_unwrap_depth:
                raise ValueError(
                    f"EmbeddingFunc unwrap depth exceeded {max_unwrap_depth}. "
                    "Possible circular reference detected."
                )
            # Unwrap to get the original function
            self.func = self.func.func

        if unwrap_count > 0:
            logger.warning(
                f"Detected nested EmbeddingFunc wrapping (depth: {unwrap_count}), "
                "auto-unwrapped to prevent configuration conflicts. "
                "Consider using .func to access the unwrapped function directly."
            )

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Only inject embedding_dim when send_dimensions is True
        if self.send_dimensions:
            # Check if user provided embedding_dim parameter
            if "embedding_dim" in kwargs:
                user_provided_dim = kwargs["embedding_dim"]
                # If user's value differs from class attribute, output warning
                if (
                    user_provided_dim is not None
                    and user_provided_dim != self.embedding_dim
                ):
                    logger.warning(
                        f"Ignoring user-provided embedding_dim={user_provided_dim}, "
                        f"using declared embedding_dim={self.embedding_dim} from decorator"
                    )

            # Inject embedding_dim from decorator
            kwargs["embedding_dim"] = self.embedding_dim

        # Remove context parameter if underlying function does not support asymmetric embedding
        if "context" in kwargs and not self.supports_asymmetric:
            # Log when a user-provided context is ignored due to lack of support
            logger.debug(
                "Context parameter was provided but supports_asymmetric=False. The context value has been ignored."
            )
            kwargs.pop("context")

        # Check if underlying function supports max_token_size and inject if not provided
        if self.max_token_size is not None and "max_token_size" not in kwargs:
            sig = inspect.signature(self.func)
            if "max_token_size" in sig.parameters:
                kwargs["max_token_size"] = self.max_token_size

        # Call the actual embedding function
        result = await self.func(*args, **kwargs)

        # Validate embedding dimensions using total element count
        total_elements = result.size  # Total number of elements in the numpy array
        expected_dim = self.embedding_dim

        # Check if total elements can be evenly divided by embedding_dim
        if total_elements % expected_dim != 0:
            raise ValueError(
                f"Embedding dimension mismatch detected: "
                f"total elements ({total_elements}) cannot be evenly divided by "
                f"expected dimension ({expected_dim}). "
            )

        # Optional: Verify vector count matches input text count
        actual_vectors = total_elements // expected_dim
        if args and isinstance(args[0], (list, tuple)):
            expected_vectors = len(args[0])
            if actual_vectors != expected_vectors:
                raise ValueError(
                    f"Vector count mismatch: "
                    f"expected {expected_vectors} vectors but got {actual_vectors} vectors (from embedding result)."
                )

        return result


def compute_args_hash(*args: Any) -> str:
    """Compute a hash for the given arguments with safe Unicode handling.

    Args:
        *args: Arguments to hash
    Returns:
        str: Hash string

    Note:
        - **Single-argument calls preserve the original algorithm** (plain
          ``str(arg)``) so that ``compute_mdhash_id(content)`` keeps producing
          the same document IDs across upgrades. Existing persisted IDs stay valid.
        - **Two or more arguments** use length-prefixed encoding
          (``"{len}:{str(arg)}"`` joined without separators) so that adjacent
          field boundaries are unambiguously recoverable. This prevents the
          cache-key collisions that the delimiter-less join could produce
          (e.g. ``("abc","x")`` vs ``("ab","cx")`` both hashed to the same MD5).
          Length-prefixing is used instead of a sentinel character because
          query text and free-form fields can contain arbitrary characters,
          so any fixed delimiter could still be constructed to collide.
    """
    if len(args) <= 1:
        # Single-arg (or no-arg) path: keep the legacy behavior unchanged so
        # compute_mdhash_id(content) document IDs remain stable.
        args_str = "".join([str(arg) for arg in args])
    else:
        # Multi-arg path: length-prefix each field to make boundaries unique.
        # Format: "3:abc1:x2:10" — the leading length makes the field extent
        # unambiguous regardless of the field content.
        args_str = "".join(f"{len(s)}:{s}" for s in (str(arg) for arg in args))

    # Use 'replace' error handling to safely encode problematic Unicode characters
    # This replaces invalid characters with Unicode replacement character (U+FFFD)
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        # Handle surrogate characters and other encoding issues
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()


def _serialize_cache_variant(value: Any) -> str:
    """Serialize cache-affecting options to a stable string for hash inputs."""
    if value is None:
        return ""

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()

    if hasattr(value, "model_json_schema") and callable(value.model_json_schema):
        value = value.model_json_schema()

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        )
    except (TypeError, ValueError):
        return repr(value)


def get_llm_cache_identity(
    global_config: dict[str, Any] | None,
    role: str,
) -> dict[str, Any]:
    """Get the non-secret LLM identity used to partition LLM cache keys.

    Includes ``role``, ``binding``, ``model``, and ``host``. Deliberately excludes
    ``api_key`` and ``provider_options`` so cache keys remain non-secret and safe
    to persist.
    """
    config = global_config or {}
    identities = config.get("llm_cache_identities")
    if isinstance(identities, dict):
        identity = identities.get(role)
        if isinstance(identity, dict):
            return dict(identity)

    return {
        "role": role,
        "binding": None,
        "model": config.get("llm_model_name"),
        "host": None,
    }


def serialize_llm_cache_identity(identity: Any) -> str:
    """Serialize an LLM cache identity for inclusion in hash inputs."""
    return _serialize_cache_variant(identity)


def _validate_cached_response_format(response_format: Any | None) -> None:
    """Reject structured-output modes that the cache wrapper does not support."""
    if response_format is None:
        return

    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_object"
    ):
        return

    raise ValueError(
        "use_llm_func_with_cache only supports response_format={'type': 'json_object'}; "
        "json_schema and typed response_format values must not be passed through the cache wrapper."
    )


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + compute_args_hash(content)


def get_unique_filename_in_parsed(target_dir: Path, original_name: str) -> str:
    """Generate a unique filename in target_dir, adding numeric suffixes on conflict.

    Tries the original name first, then `{stem}_001{ext}` ... `{stem}_999{ext}`,
    falling back to a timestamp-suffixed name if all numeric slots are taken.
    """
    original_path = Path(original_name)
    base_name = original_path.stem
    extension = original_path.suffix

    if not (target_dir / original_name).exists():
        return original_name

    for i in range(1, 1000):
        new_name = f"{base_name}_{i:03d}{extension}"
        if not (target_dir / new_name).exists():
            return new_name

    return f"{base_name}_{int(time.time())}{extension}"


async def move_file_to_parsed_dir(
    file_path: Path,
    *,
    skip_if_already_parsed: bool = False,
) -> Path | None:
    """Move a processed source file into its sibling __parsed__ directory.

    Returns the new path on success, the input path if `skip_if_already_parsed`
    is set and the file already lives in a `__parsed__` directory, or None if
    the source no longer exists.
    """
    if not file_path.exists() or not file_path.is_file():
        return None
    if skip_if_already_parsed and file_path.parent.name == PARSED_DIR_NAME:
        return file_path

    parsed_dir = file_path.parent / PARSED_DIR_NAME
    await asyncio.to_thread(parsed_dir.mkdir, parents=True, exist_ok=True)

    unique_filename = get_unique_filename_in_parsed(parsed_dir, file_path.name)
    target_path = parsed_dir / unique_filename
    await asyncio.to_thread(file_path.rename, target_path)
    logger.debug(
        f"Moved file to parsed directory: {file_path.name} -> {unique_filename}"
    )
    return target_path


def make_relation_vdb_ids(src_entity: str, tgt_entity: str) -> list[str]:
    """Return candidate relation VDB IDs for an undirected edge.

    The normalized ID is returned first for all new writes. The reverse-order ID is
    kept as a compatibility fallback for historical custom-KG imports that hashed
    the relation using the original endpoint order.
    """
    normalized_src, normalized_tgt = sorted((src_entity, tgt_entity))
    relation_ids = [compute_mdhash_id(normalized_src + normalized_tgt, prefix="rel-")]
    reverse_relation_id = compute_mdhash_id(
        normalized_tgt + normalized_src, prefix="rel-"
    )
    if reverse_relation_id not in relation_ids:
        relation_ids.append(reverse_relation_id)
    return relation_ids


def generate_cache_key(mode: str, cache_type: str, hash_value: str) -> str:
    """Generate a flattened cache key in the format {mode}:{cache_type}:{hash}

    Args:
        mode: Cache mode (e.g., 'default', 'local', 'global')
        cache_type: Type of cache (e.g., 'extract', 'query', 'keywords')
        hash_value: Hash value from compute_args_hash

    Returns:
        str: Flattened cache key
    """
    return f"{mode}:{cache_type}:{hash_value}"


def parse_cache_key(cache_key: str) -> tuple[str, str, str] | None:
    """Parse a flattened cache key back into its components

    Args:
        cache_key: Flattened cache key in format {mode}:{cache_type}:{hash}

    Returns:
        tuple[str, str, str] | None: (mode, cache_type, hash) or None if invalid format
    """
    parts = cache_key.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None


# Custom exception classes
class QueueFullError(Exception):
    """Raised when the queue is full and the wait times out"""

    pass


class VectorStorageConsistencyError(Exception):
    """Raised when a vector storage write fails after the graph has already been updated.

    The knowledge graph (plus the text_chunks KV store) is the authoritative data
    source, so no data is lost — but the vector storage no longer mirrors the graph
    and query results may be incomplete until it is rebuilt. Stop the LightRAG
    server and run the offline rebuild tool (``lightrag-rebuild-vdb``) to restore
    consistency.
    """

    pass


class WorkerTimeoutError(Exception):
    """Worker-level timeout exception with specific timeout information"""

    def __init__(self, timeout_value: float, timeout_type: str = "execution"):
        self.timeout_value = timeout_value
        self.timeout_type = timeout_type
        super().__init__(f"Worker {timeout_type} timeout after {timeout_value}s")


class HealthCheckTimeoutError(Exception):
    """Health Check-level timeout exception"""

    def __init__(self, timeout_value: float, execution_duration: float):
        self.timeout_value = timeout_value
        self.execution_duration = execution_duration
        super().__init__(
            f"Task forcefully terminated due to execution timeout (>{timeout_value}s, actual: {execution_duration:.1f}s)"
        )


class TokenBudgetError(Exception):
    """Raised by :class:`Tokenizer`'s split/truncate contract when not even the
    first complete Unicode code point of a candidate fits within ``max_tokens``.

    Carries the fields needed to diagnose the failure without re-deriving them:
    the budget that was violated, the independently-encoded token count of that
    single code point under the same tokenizer, and a bounded preview of the
    offending text.
    """

    def __init__(self, max_tokens: int, code_point_token_count: int, preview: str):
        self.max_tokens = max_tokens
        self.code_point_token_count = code_point_token_count
        self.preview = preview
        super().__init__(
            f"Cannot fit content within max_tokens={max_tokens}: the first "
            f"complete Unicode code point alone encodes to "
            f"{code_point_token_count} token(s). Preview: {preview!r}"
        )


def priority_limit_async_func_call(
    max_size: int,
    llm_timeout: float = None,
    max_execution_timeout: float = None,
    max_task_duration: float = None,
    max_queue_size: int = 1000,
    cleanup_timeout: float = 2.0,
    queue_name: str = "limit_async",
    concurrency_group: str | None = None,
):
    """
    Enhanced priority-limited asynchronous function call decorator with robust timeout handling

    This decorator provides a comprehensive solution for managing concurrent LLM requests with:
    - Multi-layer timeout protection (LLM -> Worker -> Health Check -> User)
    - Task state tracking to prevent race conditions
    - Enhanced health check system with stuck task detection
    - Proper resource cleanup and error recovery
    - Optional cross-process global concurrency gating (gunicorn multi-worker)

    Args:
        max_size: Maximum number of concurrent calls
        max_queue_size: Maximum queue capacity to prevent memory overflow
        llm_timeout: LLM provider timeout (from global config), used to calculate other timeouts
        max_execution_timeout: Maximum time for worker to execute function (defaults to llm_timeout + 30s)
        max_task_duration: Maximum time before health check intervenes (defaults to llm_timeout + 60s)
        cleanup_timeout: Maximum time to wait for cleanup operations (defaults to 2.0s)
        queue_name: Optional queue name for logging identification (defaults to "limit_async")
        concurrency_group: Optional cross-process concurrency group name (e.g.
            "llm:extract", "embedding", "rerank"). When shared storage was
            initialized with a global limit for this group, workers acquire a
            cross-worker slot (lease with heartbeat self-healing) before
            executing, capping total in-flight calls across all gunicorn
            workers; the group's queue stats are also published for /health
            aggregation. With no global limit configured for the group
            (single-process / embedded usage) the slot gate is bypassed —
            execution behavior matches the original per-process decorator —
            but queue stats are still published to shared storage so the
            aggregated /health view works; in single-process mode that is a
            cheap local-dict write (no IPC, no slot acquisition). Only
            concurrency_group=None is fully self-contained: shared storage is
            never touched at all (no slot gate AND no stats publishing).

    Returns:
        Decorator function
    """

    def final_dec(func):
        # Ensure func is callable
        if not callable(func):
            raise TypeError(f"Expected a callable object, got {type(func)}")

        # Calculate timeout hierarchy if llm_timeout is provided (Dynamic Timeout Calculation)
        if llm_timeout is not None:
            nonlocal max_execution_timeout, max_task_duration
            if max_execution_timeout is None:
                max_execution_timeout = (
                    llm_timeout * 2
                )  # Reserved timeout buffer for low-level retry
            if max_task_duration is None:
                max_task_duration = (
                    llm_timeout * 2 + 15
                )  # Reserved timeout buffer for health check phase

        # The queue is created lazily in ensure_workers(): the default path
        # keeps the bounded queue, while global-limit mode needs an unbounded
        # physical queue (admission is enforced logically via live_queued so
        # cancelled-but-not-yet-drained tuples can never wedge the queue).
        queue: asyncio.PriorityQueue | None = None
        tasks = set()
        initialization_lock = asyncio.Lock()
        counter = 0
        shutdown_event = asyncio.Event()
        initialized = False
        accepting_new_tasks = True
        worker_health_check_task = None

        # Enhanced task state management
        task_states = {}  # task_id -> TaskState
        task_states_lock = asyncio.Lock()
        active_futures = weakref.WeakSet()
        reinit_count = 0
        submitted_total = 0
        completed_total = 0
        failed_total = 0
        cancelled_total = 0
        rejected_total = 0

        # --- Cross-worker global concurrency gate state (global-limit mode) ---
        # Tri-state: None until resolved on first ensure_workers() (which runs
        # after initialize_share_data() in every supported flow).
        use_global_limit: bool | None = None
        publish_stats = False
        shared = None  # lazily imported lightrag.kg.shared_storage module
        work_available = asyncio.Event()
        admission_cond = asyncio.Condition()
        # Logical queued count: live tasks waiting in the queue (excludes
        # running tasks and cancelled zombies) — same capacity semantics as
        # the bounded queue's maxsize in the default path.
        live_queued = 0
        held_leases: set[str] = set()
        pending_release: set[str] = set()
        global_slot_waits = 0
        zombie_compact_threshold = max(
            DEFAULT_ZOMBIE_COMPACT_THRESHOLD,
            max_queue_size if max_queue_size > 0 else 0,
        )
        # Slot pump machinery (global-limit mode): ONE coroutine per process
        # acquires global slots and hands (lease, task) pairs to executor
        # workers through dispatch_queue. executing counts tasks picked up
        # by workers; worker_free wakes the pump when one finishes.
        # NOTE: dispatch_queue deliberately never gets task_done()/join() —
        # the join()-based graceful drain tracks the PHYSICAL queue only (a
        # dispatched item's physical-queue task_done() is deferred to the
        # worker), and shutdown empties any undelivered dispatch entries with
        # a get_nowait() loop. So dispatch_queue.unfinished_tasks grows
        # unbounded by design; it is never read. Don't add a join() here
        # without also adding matching task_done() calls.
        dispatch_queue: asyncio.Queue | None = None
        pump_task: asyncio.Task | None = None
        executing = 0
        worker_free = asyncio.Event()
        last_publish_time = 0.0
        last_release_warn_time = 0.0
        last_renew_warn_time = 0.0

        def _resolve_mode() -> bool:
            """Resolve global-limit / stats-publishing mode from shared storage.

            Returns True when the resolution is final. Never imports or
            touches shared storage when concurrency_group is None
            (standalone decorator usage stays fully self-contained).
            """
            nonlocal use_global_limit, publish_stats, shared
            if use_global_limit is not None:
                return True
            if concurrency_group is None:
                use_global_limit = False
                publish_stats = False
                return True
            if shared is None:
                from lightrag.kg import shared_storage as shared_module

                shared = shared_module
            if not shared.is_share_data_initialized():
                return False  # not final yet — caller decides how to commit
            use_global_limit = shared.is_global_concurrency_limited(concurrency_group)
            publish_stats = True
            return True

        def _snapshot() -> dict:
            """Synchronous snapshot of local state for cross-worker publishing.

            Reads counters without locks: all mutations happen on the event
            loop between awaits, so a synchronous read is always consistent.
            """
            running = sum(
                1
                for task_state in task_states.values()
                if task_state.worker_started and not task_state.future.done()
            )
            physical_queued = queue.qsize() if queue is not None else 0
            return {
                "queue_name": queue_name,
                "max_async": max_size,
                "max_queue_size": max_queue_size,
                "queued": live_queued if use_global_limit else physical_queued,
                "physical_queued": physical_queued,
                "running": running,
                "in_flight": len(task_states),
                "worker_count": len([task for task in tasks if not task.done()]),
                "initialized": initialized,
                "submitted_total": submitted_total,
                "completed_total": completed_total,
                "failed_total": failed_total,
                "cancelled_total": cancelled_total,
                "rejected_total": rejected_total,
                "global_slot_waits": global_slot_waits,
                "pid": os.getpid(),
                "updated_at": time.time(),
            }

        async def _publish_stats(force: bool = False) -> None:
            """Best-effort, debounced publish of the local stats snapshot.

            Called from counter-update points (debounced to the min publish
            interval) and force-flushed by the 5s maintenance pass, which
            also propagates any counter change that happened between
            debounced publishes and keeps the snapshot ahead of the
            aggregation stale TTL.
            """
            nonlocal last_publish_time
            if not publish_stats:
                return
            now = time.time()
            if (
                not force
                and now - last_publish_time < DEFAULT_QUEUE_STATS_MIN_PUBLISH_INTERVAL
            ):
                return
            try:
                await shared.publish_queue_stats(queue_name, _snapshot())
                last_publish_time = now
            except Exception as e:
                logger.debug(f"{queue_name}: queue stats publish failed: {e}")

        async def _notify_admission() -> None:
            async with admission_cond:
                admission_cond.notify_all()

        async def _try_acquire_slot() -> tuple[str | None, bool]:
            """Non-blocking global slot acquisition (fail-closed on errors).

            Returns ``(lease_id, is_priority_waiter)``: on failure the
            second element reports whether this process is the
            longest-waiting live poller of the group, which drives the
            adaptive poll backoff below.
            """
            try:
                lease_id, is_priority = await shared.try_acquire_global_slot_tracked(
                    concurrency_group
                )
            except Exception as e:
                # try_acquire_global_slot_tracked is fail-closed internally;
                # this guard keeps the worker alive even if it ever raises.
                logger.debug(f"{queue_name}: global slot acquisition error: {e}")
                return None, False
            if lease_id is not None:
                held_leases.add(lease_id)
            return lease_id, is_priority

        async def _release_lease_safely(lease_id: str) -> None:
            """Release a global slot without raising (safe in finally blocks).

            A failed release is parked in pending_release: it is no longer
            renewed, the health check retries it, and even if every retry
            fails the heartbeat TTL guarantees any process eventually
            reclaims the slot — capacity never leaks permanently.
            """
            nonlocal last_release_warn_time
            held_leases.discard(lease_id)
            try:
                await shared.release_global_slot(concurrency_group, lease_id)
                pending_release.discard(lease_id)
            except asyncio.CancelledError:
                pending_release.add(lease_id)
                raise
            except Exception as e:
                pending_release.add(lease_id)
                now = time.time()
                if now - last_release_warn_time >= 30.0:
                    last_release_warn_time = now
                    logger.warning(
                        f"{queue_name}: failed to release global slot lease "
                        f"(queued for retry; heartbeat expiry guarantees "
                        f"reclamation): {e}"
                    )

        async def _compact_physical_queue() -> None:
            """Drain zombie tuples that accumulate while no slot is available.

            Without this, a long fail-closed period (shared storage errors)
            or externally saturated slots would let cancelled tasks pile up
            in the unbounded physical queue with no consumer. Bounded batches
            keep the event loop responsive; every popped tuple gets exactly
            one task_done() (live tuples are re-queued first, adding a fresh
            unfinished count) so queue.join() in shutdown never wedges.
            """
            nonlocal live_queued
            if queue is None or not use_global_limit:
                return
            if queue.qsize() - live_queued <= zombie_compact_threshold:
                return
            survivors = []
            scanned = 0
            notify_needed = False
            while scanned < DEFAULT_COMPACT_BATCH_LIMIT:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                scanned += 1
                task_id = item[2]
                is_zombie = False
                # Classify under task_states_lock so we serialize with the
                # wait_func timeout cleanup path (never judge by a stale
                # snapshot taken outside the lock).
                async with task_states_lock:
                    task_state = task_states.get(task_id)
                    if (
                        task_state is None
                        or task_state.cancellation_requested
                        or task_state.future.cancelled()
                        or task_state.future.done()
                    ):
                        is_zombie = True
                        if task_state is not None:
                            task_states.pop(task_id, None)
                            if not task_state.worker_started:
                                live_queued -= 1
                                notify_needed = True
                if is_zombie:
                    queue.task_done()
                else:
                    survivors.append(item)
            for item in survivors:
                queue.put_nowait(item)
                queue.task_done()
            if survivors:
                work_available.set()
            if notify_needed:
                await _notify_admission()

        async def _run_maintenance() -> None:
            """One heartbeat pass of cross-worker upkeep (never raises).

            Runs every health-check tick: lease renewal (correctness path —
            failures get a rate-limited WARNING, the suspect grace absorbs
            short outages), pending-release retries, lease reaping, zombie
            compaction, and a forced stats flush (which also keeps this
            worker's snapshot from going stale in the aggregation view).
            """
            nonlocal last_renew_warn_time
            if use_global_limit:
                try:
                    await shared.renew_global_slots(
                        concurrency_group, tuple(held_leases)
                    )
                except Exception as e:
                    now = time.time()
                    if now - last_renew_warn_time >= 30.0:
                        last_renew_warn_time = now
                        logger.warning(
                            f"{queue_name}: global slot lease renewal failed "
                            f"(leases may be reclaimed after the suspect "
                            f"grace if this persists): {e}"
                        )
                for lease_id in tuple(pending_release):
                    try:
                        await shared.release_global_slot(concurrency_group, lease_id)
                        pending_release.discard(lease_id)
                    except Exception as e:
                        logger.debug(
                            f"{queue_name}: pending lease release retry failed: {e}"
                        )
                        break  # shared area still unhealthy; retry next pass
                try:
                    await shared.reconcile_global_slots(concurrency_group)
                except Exception as e:
                    logger.debug(f"{queue_name}: global slot reconcile failed: {e}")
                try:
                    await _compact_physical_queue()
                except Exception as e:
                    logger.warning(f"{queue_name}: queue compaction failed: {e}")
            if publish_stats:
                await _publish_stats(force=True)

        async def worker():
            """Enhanced worker that processes tasks with proper timeout and state management"""
            try:
                while not shutdown_event.is_set():
                    try:
                        # Get task from queue with timeout for shutdown checking
                        try:
                            (
                                priority,
                                count,
                                task_id,
                                args,
                                kwargs,
                                ctx,
                            ) = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

                        # Get task state and mark worker as started
                        async with task_states_lock:
                            if task_id not in task_states:
                                queue.task_done()
                                continue
                            task_state = task_states[task_id]
                            task_state.worker_started = True
                            # Record execution start time when worker actually begins processing
                            task_state.execution_start_time = (
                                asyncio.get_running_loop().time()
                            )

                        # Check if task was cancelled before worker started
                        if (
                            task_state.cancellation_requested
                            or task_state.future.cancelled()
                        ):
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()
                            continue

                        try:
                            # Execute the call under the enqueuer's captured
                            # context, not the worker's creation-time context
                            # (asyncio.Task snapshots contextvars once, at
                            # create_task() time, and this worker task is
                            # long-lived — reused across many unrelated
                            # calls). ctx.run(asyncio.ensure_future, ...)
                            # schedules the coroutine as a fresh Task whose
                            # own context is copied from ctx (works on
                            # Python 3.10, unlike create_task's context=
                            # kwarg which needs 3.11+).
                            exec_task = ctx.run(
                                asyncio.ensure_future, func(*args, **kwargs)
                            )
                            if max_execution_timeout is not None:
                                result = await asyncio.wait_for(
                                    exec_task, timeout=max_execution_timeout
                                )
                            else:
                                result = await exec_task

                            # Set result if future is still valid
                            if not task_state.future.done():
                                task_state.future.set_result(result)

                        except asyncio.TimeoutError:
                            # Worker-level timeout (max_execution_timeout exceeded)
                            logger.warning(
                                f"{queue_name}: Worker timeout for task {task_id} after {max_execution_timeout}s"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(
                                    WorkerTimeoutError(
                                        max_execution_timeout, "execution"
                                    )
                                )
                        except asyncio.CancelledError:
                            # Task was cancelled during execution
                            if not task_state.future.done():
                                task_state.future.cancel()
                            logger.debug(
                                f"{queue_name}: Task {task_id} cancelled during execution"
                            )
                        except Exception as e:
                            # Function execution error
                            logger.error(
                                f"{queue_name}: Error in decorated function for task {task_id}: {str(e)}"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(e)
                        finally:
                            # Clean up task state
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()

                    except Exception as e:
                        # Critical error in worker loop
                        logger.error(
                            f"{queue_name}: Critical error in worker: {str(e)}"
                        )
                        await asyncio.sleep(0.1)
            finally:
                logger.debug(f"{queue_name}: Worker exiting")

        async def slot_pump():
            """Single per-process slot acquirer for global-limit mode.

            Slot-first, drain-second: the pump acquires a cross-process slot
            BEFORE consuming the local queue, so while slots are saturated
            tasks stay queued (cancellable, never misjudged as running) and
            local priority order is preserved — the queue head is committed
            only once a slot is held, so a later high-priority arrival can
            still overtake. Centralizing acquisition in ONE coroutine
            (instead of max_size polling workers) divides the cross-process
            poll/IPC rate by max_size, and a slot is requested only when
            there is BOTH physically queued work and an idle worker to run
            it immediately — the worker herd can no longer grab slots it
            cannot use (which inflated global_in_use and reset this
            process's waiter seniority on every no-op acquire). Residual
            churn: queued items may all turn out to be zombies after the
            slot is acquired (drained bounded, slot returned right away).
            """
            nonlocal live_queued, global_slot_waits
            poll_delay = DEFAULT_GLOBAL_SLOT_POLL_MIN
            try:
                while not shutdown_event.is_set():
                    try:
                        # Idle wait on an event instead of qsize polling. The
                        # clear-then-recheck ordering has no await in between,
                        # so a concurrent put+set can never be lost; the 1.0s
                        # timeout only preserves the shutdown check.
                        if queue.qsize() == 0:
                            work_available.clear()
                            if queue.qsize() == 0:
                                try:
                                    await asyncio.wait_for(
                                        work_available.wait(), timeout=1.0
                                    )
                                except asyncio.TimeoutError:
                                    pass
                                continue

                        # Never hold a slot no local worker could service
                        # immediately: undelivered dispatches plus running
                        # executions already saturate max_size.
                        if dispatch_queue.qsize() + executing >= max_size:
                            worker_free.clear()
                            if dispatch_queue.qsize() + executing >= max_size:
                                try:
                                    await asyncio.wait_for(
                                        worker_free.wait(), timeout=1.0
                                    )
                                except asyncio.TimeoutError:
                                    pass
                                continue

                        # Acquire a global slot before touching the queue —
                        # tasks must remain queued (and cancellable) while
                        # all slots are busy. Fail-closed errors land here
                        # too, as a None lease.
                        lease_id, is_priority_waiter = await _try_acquire_slot()
                        if lease_id is None:
                            global_slot_waits += 1
                            # Soft FIFO across processes: the longest-waiting
                            # live process keeps the fastest poll rate so it
                            # usually claims the next freed slot; everyone
                            # else backs off, bounded by the deferred cap so
                            # a freed slot is never left idle for long when
                            # the favored waiter is gone (promotion lag).
                            if is_priority_waiter:
                                poll_delay = DEFAULT_GLOBAL_SLOT_POLL_MIN
                            else:
                                poll_delay = min(
                                    poll_delay * 2,
                                    DEFAULT_GLOBAL_SLOT_POLL_DEFERRED_MAX,
                                )
                            await asyncio.sleep(poll_delay)
                            continue
                        poll_delay = DEFAULT_GLOBAL_SLOT_POLL_MIN

                        live_task = None
                        dispatched = False
                        try:
                            # Take the queue head, draining zombies (bounded
                            # by the drain limit so a zombie-heavy process
                            # doesn't hog a scarce slot for local cleanup).
                            zombies_drained = 0
                            notify_needed = False
                            while live_task is None:
                                try:
                                    item = queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    # All queued items were zombies (or
                                    # compaction holds them): return the
                                    # slot immediately.
                                    break
                                task_id, args, kwargs, ctx = (
                                    item[2],
                                    item[3],
                                    item[4],
                                    item[5],
                                )
                                is_zombie = False
                                async with task_states_lock:
                                    task_state = task_states.get(task_id)
                                    if (
                                        task_state is None
                                        or task_state.cancellation_requested
                                        or task_state.future.cancelled()
                                        or task_state.future.done()
                                    ):
                                        is_zombie = True
                                        if task_state is not None:
                                            task_states.pop(task_id, None)
                                            if not task_state.worker_started:
                                                live_queued -= 1
                                                notify_needed = True
                                    else:
                                        task_state.worker_started = True
                                        task_state.execution_start_time = (
                                            asyncio.get_running_loop().time()
                                        )
                                        live_queued -= 1
                                        notify_needed = True
                                        live_task = (
                                            task_id,
                                            task_state,
                                            args,
                                            kwargs,
                                            ctx,
                                        )
                                if is_zombie:
                                    # Never call the provider for a zombie.
                                    queue.task_done()
                                    zombies_drained += 1
                                    if (
                                        zombies_drained
                                        >= DEFAULT_GLOBAL_SLOT_DRAIN_LIMIT
                                    ):
                                        break
                            if live_task is not None:
                                # No suspension points between claiming the
                                # live task above and this put_nowait, so a
                                # claimed task is always dispatched (the
                                # admission check guaranteed a free worker).
                                dispatch_queue.put_nowait((lease_id, *live_task))
                                dispatched = True
                            if notify_needed:
                                await _notify_admission()
                            await _publish_stats()
                        finally:
                            if not dispatched:
                                await _release_lease_safely(lease_id)

                    except Exception as e:
                        logger.error(
                            f"{queue_name}: Critical error in slot pump: {str(e)}"
                        )
                        await asyncio.sleep(0.1)
            finally:
                logger.debug(f"{queue_name}: Slot pump exiting")

        async def limited_worker():
            """Executor worker for global-limit mode.

            Runs tasks handed over by the slot pump together with their
            already-held global slot; execution/timeout/exception semantics
            match the default worker. The lease travels with the task and is
            always released here (or by the shutdown drain for undelivered
            dispatch entries).
            """
            nonlocal executing
            try:
                while not shutdown_event.is_set():
                    try:
                        try:
                            (
                                lease_id,
                                task_id,
                                task_state,
                                args,
                                kwargs,
                                ctx,
                            ) = await asyncio.wait_for(
                                dispatch_queue.get(), timeout=1.0
                            )
                        except asyncio.TimeoutError:
                            continue

                        executing += 1
                        try:
                            # Re-check: the task may have been cancelled in
                            # the (tiny) window between dispatch and pickup.
                            if (
                                task_state.cancellation_requested
                                or task_state.future.cancelled()
                                or task_state.future.done()
                            ):
                                continue  # finally cleans up + returns slot

                            # Run under the enqueuer's captured context, not
                            # this long-lived worker's creation-time context
                            # — see the matching comment in worker().
                            exec_task = ctx.run(
                                asyncio.ensure_future, func(*args, **kwargs)
                            )
                            if max_execution_timeout is not None:
                                result = await asyncio.wait_for(
                                    exec_task,
                                    timeout=max_execution_timeout,
                                )
                            else:
                                result = await exec_task

                            if not task_state.future.done():
                                task_state.future.set_result(result)

                        except asyncio.TimeoutError:
                            logger.warning(
                                f"{queue_name}: Worker timeout for task {task_id} after {max_execution_timeout}s"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(
                                    WorkerTimeoutError(
                                        max_execution_timeout, "execution"
                                    )
                                )
                        except asyncio.CancelledError:
                            if not task_state.future.done():
                                task_state.future.cancel()
                            logger.debug(
                                f"{queue_name}: Task {task_id} cancelled during execution"
                            )
                        except Exception as e:
                            logger.error(
                                f"{queue_name}: Error in decorated function for task {task_id}: {str(e)}"
                            )
                            if not task_state.future.done():
                                task_state.future.set_exception(e)
                        finally:
                            executing -= 1
                            worker_free.set()
                            async with task_states_lock:
                                task_states.pop(task_id, None)
                            queue.task_done()
                            await _release_lease_safely(lease_id)
                            await _publish_stats()

                    except Exception as e:
                        logger.error(
                            f"{queue_name}: Critical error in worker: {str(e)}"
                        )
                        await asyncio.sleep(0.1)
            finally:
                logger.debug(f"{queue_name}: Worker exiting")

        def _create_worker_task() -> asyncio.Task:
            return asyncio.create_task(
                limited_worker() if use_global_limit else worker()
            )

        async def enhanced_health_check():
            """Enhanced health check with stuck task detection and recovery"""
            nonlocal initialized, pump_task
            try:
                while not shutdown_event.is_set():
                    await asyncio.sleep(5)  # Check every 5 seconds

                    current_time = asyncio.get_running_loop().time()

                    # Detect and handle stuck tasks based on execution start time
                    if max_task_duration is not None:
                        stuck_tasks = []
                        async with task_states_lock:
                            for task_id, task_state in list(task_states.items()):
                                # Only check tasks that have started execution
                                if (
                                    task_state.worker_started
                                    and task_state.execution_start_time is not None
                                    and current_time - task_state.execution_start_time
                                    > max_task_duration
                                ):
                                    stuck_tasks.append(
                                        (
                                            task_id,
                                            current_time
                                            - task_state.execution_start_time,
                                        )
                                    )

                        # Force cleanup of stuck tasks
                        for task_id, execution_duration in stuck_tasks:
                            logger.warning(
                                f"{queue_name}: Detected stuck task {task_id} (execution time: {execution_duration:.1f}s), forcing cleanup"
                            )
                            async with task_states_lock:
                                if task_id in task_states:
                                    task_state = task_states[task_id]
                                    if not task_state.future.done():
                                        task_state.future.set_exception(
                                            HealthCheckTimeoutError(
                                                max_task_duration, execution_duration
                                            )
                                        )
                                    task_states.pop(task_id, None)

                    # Worker recovery logic
                    current_tasks = set(tasks)
                    done_tasks = {t for t in current_tasks if t.done()}
                    tasks.difference_update(done_tasks)

                    active_tasks_count = len(tasks)
                    workers_needed = max_size - active_tasks_count

                    if workers_needed > 0:
                        logger.info(
                            f"{queue_name}: Creating {workers_needed} new workers"
                        )
                        new_tasks = set()
                        for _ in range(workers_needed):
                            task = _create_worker_task()
                            new_tasks.add(task)
                            task.add_done_callback(tasks.discard)
                        tasks.update(new_tasks)

                    # Pump recovery: without it no slot is ever acquired and
                    # the whole limited queue stalls.
                    if use_global_limit and (pump_task is None or pump_task.done()):
                        logger.warning(f"{queue_name}: Recreating dead slot pump")
                        pump_task = asyncio.create_task(slot_pump())

                    # Cross-worker upkeep: lease heartbeat / reaping, zombie
                    # compaction, stats flush. Internally best-effort — each
                    # step isolates its own failures so the health check
                    # loop never exits because of shared-storage errors.
                    await _run_maintenance()

            except Exception as e:
                logger.error(f"{queue_name}: Error in enhanced health check: {str(e)}")
            finally:
                logger.debug(f"{queue_name}: Enhanced health check task exiting")
                initialized = False

        async def ensure_workers():
            """Ensure worker system is initialized with enhanced error handling"""
            nonlocal initialized, worker_health_check_task, tasks, reinit_count
            nonlocal queue, use_global_limit, dispatch_queue, pump_task

            if initialized:
                return

            async with initialization_lock:
                if initialized:
                    return

                # Resolve the concurrency mode once (cached for the lifetime
                # of the wrapper) and lazily create the matching queue. When
                # shared storage is not initialized at this point (standalone
                # usage), commit to the default unlimited path.
                if use_global_limit is None and not _resolve_mode():
                    use_global_limit = False
                if queue is None:
                    if use_global_limit:
                        queue = asyncio.PriorityQueue()
                        dispatch_queue = asyncio.Queue()
                    else:
                        queue = asyncio.PriorityQueue(maxsize=max_queue_size)

                if reinit_count > 0:
                    reinit_count += 1
                    logger.warning(
                        f"{queue_name}: Reinitializing system (count: {reinit_count})"
                    )
                else:
                    reinit_count = 1

                # Clean up completed tasks
                current_tasks = set(tasks)
                done_tasks = {t for t in current_tasks if t.done()}
                tasks.difference_update(done_tasks)

                active_tasks_count = len(tasks)
                if active_tasks_count > 0 and reinit_count > 1:
                    logger.warning(
                        f"{queue_name}: {active_tasks_count} tasks still running during reinitialization"
                    )

                # Create worker tasks
                workers_needed = max_size - active_tasks_count
                for _ in range(workers_needed):
                    task = _create_worker_task()
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                # Start the slot pump (kept out of `tasks` so the worker
                # recovery count never mistakes it for an executor worker).
                if use_global_limit and (pump_task is None or pump_task.done()):
                    pump_task = asyncio.create_task(slot_pump())

                # Start enhanced health check
                worker_health_check_task = asyncio.create_task(enhanced_health_check())

                initialized = True
                # Log dynamic timeout configuration
                timeout_info = []
                if llm_timeout is not None:
                    timeout_info.append(f"Func: {llm_timeout}s")
                if max_execution_timeout is not None:
                    timeout_info.append(f"Worker: {max_execution_timeout}s")
                if max_task_duration is not None:
                    timeout_info.append(f"Health Check: {max_task_duration}s")

                timeout_str = (
                    f"(Timeouts: {', '.join(timeout_info)})" if timeout_info else ""
                )
                logger.info(
                    f"{queue_name}: {workers_needed} new workers initialized {timeout_str}"
                )

        async def get_queue_stats():
            """Return a best-effort snapshot of queue and worker state."""
            async with task_states_lock:
                running = sum(
                    1
                    for task_state in task_states.values()
                    if task_state.worker_started and not task_state.future.done()
                )
                in_flight = len(task_states)

            active_workers = len([task for task in tasks if not task.done()])
            physical_queued = queue.qsize() if queue is not None else 0
            stats = {
                "queue_name": queue_name,
                "max_async": max_size,
                "max_queue_size": max_queue_size,
                # Global-limit mode reports the logical queued count (live
                # tasks only — cancelled zombies still physically present in
                # the unbounded queue are excluded).
                "queued": live_queued if use_global_limit else physical_queued,
                "running": running,
                "in_flight": in_flight,
                "worker_count": active_workers,
                "initialized": initialized,
                "submitted_total": submitted_total,
                "completed_total": completed_total,
                "failed_total": failed_total,
                "cancelled_total": cancelled_total,
                "rejected_total": rejected_total,
            }
            if use_global_limit:
                stats["physical_queued"] = physical_queued
                stats["global_slot_waits"] = global_slot_waits
            return stats

        async def get_aggregated_queue_stats():
            """Local stats merged with every worker process's published snapshot.

            Publishes this process's fresh snapshot first, then sums the flat
            counter fields across all live snapshots (schema-compatible with
            get_queue_stats so /health consumers and the webui need no
            changes), adding ``reporting_workers`` / ``per_worker`` and — in
            global-limit mode — ``global_limit`` / ``global_in_use``. Any
            shared-storage failure falls back to the local snapshot.
            """
            local = await get_queue_stats()
            if not _resolve_mode() or not publish_stats:
                return local
            try:
                await shared.publish_queue_stats(queue_name, _snapshot())
                aggregated = await shared.aggregate_queue_stats(queue_name)
                result = dict(local)
                for field_name in shared.QUEUE_STATS_SUM_FIELDS:
                    if field_name in aggregated:
                        result[field_name] = aggregated[field_name]
                result["reporting_workers"] = aggregated["reporting_workers"]
                result["per_worker"] = aggregated["per_worker"]
                if use_global_limit:
                    result["global_limit"] = shared.get_global_concurrency_limit(
                        concurrency_group
                    )
                    result["global_in_use"] = await shared.global_concurrency_in_use(
                        concurrency_group
                    )
                    waiters = await shared.global_slot_waiters(concurrency_group)
                    result["global_waiting_workers"] = len(waiters)
                    result["global_longest_wait"] = (
                        round(waiters[0]["waited"], 3) if waiters else 0.0
                    )
                return result
            except Exception as e:
                logger.debug(
                    f"{queue_name}: queue stats aggregation failed, "
                    f"falling back to local snapshot: {e}"
                )
                return local

        async def shutdown(graceful: bool = True, timeout: float | None = None):
            """Shut down workers and cleanup resources.

            Graceful mode stops new submissions and drains queued/running
            work; if the drain exceeds ``timeout`` (defaulting to
            ``max_task_duration`` or 30s), it falls through to forced
            cancellation so shutdown never blocks indefinitely.
            """
            nonlocal accepting_new_tasks, initialized, worker_health_check_task
            nonlocal pump_task
            logger.info(f"{queue_name}: Shutting down priority queue workers")

            if use_global_limit:
                # Stop accepting and wake admission waiters inside the same
                # Condition critical section: a request sleeping on admission
                # (no _queue_timeout) must observe the flag flip and raise
                # the shutdown rejection instead of sleeping forever.
                async with admission_cond:
                    accepting_new_tasks = False
                    admission_cond.notify_all()
            else:
                accepting_new_tasks = False

            drain_timed_out = False
            if graceful and queue is not None:
                effective_timeout = timeout
                if effective_timeout is None:
                    effective_timeout = (
                        max_task_duration if max_task_duration is not None else 30.0
                    )
                try:
                    await asyncio.wait_for(queue.join(), timeout=effective_timeout)
                except asyncio.TimeoutError:
                    drain_timed_out = True
                    logger.warning(
                        f"{queue_name}: Graceful drain timed out after "
                        f"{effective_timeout}s; cancelling pending work"
                    )

            if not graceful or drain_timed_out:
                # Cancel all active futures
                for future in list(active_futures):
                    if not future.done():
                        future.cancel()

                # Cancel all pending tasks
                async with task_states_lock:
                    for task_id, task_state in list(task_states.items()):
                        if not task_state.future.done():
                            task_state.future.cancel()
                    task_states.clear()

                while queue is not None:
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        break

            shutdown_event.set()

            # Cancel the slot pump first so no new dispatch entries appear
            # while workers drain below.
            if pump_task is not None and not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
            pump_task = None

            # Cancel worker tasks
            for task in list(tasks):
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Drain undelivered dispatch entries: each one was already
            # popped from the physical queue and carries a held lease.
            while dispatch_queue is not None:
                try:
                    (
                        lease_id,
                        task_id,
                        task_state,
                        _args,
                        _kwargs,
                        _ctx,
                    ) = dispatch_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not task_state.future.done():
                    task_state.future.cancel()
                async with task_states_lock:
                    task_states.pop(task_id, None)
                queue.task_done()
                await _release_lease_safely(lease_id)

            # Cancel health check task
            if worker_health_check_task and not worker_health_check_task.done():
                worker_health_check_task.cancel()
                try:
                    await worker_health_check_task
                except asyncio.CancelledError:
                    pass
            worker_health_check_task = None
            initialized = False

            # Return any global slots still held (worker cancellation may
            # have interrupted a release) and retract our published stats.
            # Best-effort: heartbeat expiry reclaims anything left behind.
            if use_global_limit:
                for lease_id in list(held_leases | pending_release):
                    held_leases.discard(lease_id)
                    pending_release.discard(lease_id)
                    try:
                        await shared.release_global_slot(concurrency_group, lease_id)
                    except Exception:
                        pass
                # Our workers stop polling now: drop the waiter record so
                # this process never lingers in the longest-waiter seat
                # (the stale TTL covers crashes where this never runs).
                try:
                    await shared.clear_slot_waiter(concurrency_group)
                except Exception:
                    pass
            if publish_stats:
                try:
                    await shared.unpublish_queue_stats(queue_name)
                except Exception:
                    pass

            logger.info(f"{queue_name}: Priority queue workers shutdown complete")

        async def _limited_wait(args, kwargs, _priority, _timeout, _queue_timeout):
            """wait_func body for global-limit mode (logical admission).

            Admission reserves logical capacity (live_queued) BEFORE the
            task state is registered, with the same semantics as the bounded
            queue in the default path: only live queued tasks count toward
            max_queue_size (running tasks and cancelled zombies do not),
            _queue_timeout bounds the wait with QueueFullError, and
            max_queue_size <= 0 means unlimited admission. The reservation
            is released exactly once — by the worker when the task turns
            running (worker_started flip), or by the cleanup below when the
            task dies while still queued.
            """
            nonlocal counter, submitted_total, completed_total, cancelled_total
            nonlocal failed_total, rejected_total, live_queued

            task_id = (
                f"{id(asyncio.current_task())}_{asyncio.get_running_loop().time()}"
            )
            future = asyncio.Future()
            task_state = TaskState(
                future=future, start_time=asyncio.get_running_loop().time()
            )

            def _admission_open() -> bool:
                return live_queued < max_queue_size or not accepting_new_tasks

            # --- Admission: reserve capacity before registering ---
            async with admission_cond:
                if not accepting_new_tasks:
                    rejected_total += 1
                    raise RuntimeError(f"{queue_name}: Queue is shutting down")
                if max_queue_size > 0 and live_queued >= max_queue_size:
                    try:
                        if _queue_timeout is not None:
                            await asyncio.wait_for(
                                admission_cond.wait_for(_admission_open),
                                timeout=_queue_timeout,
                            )
                        else:
                            await admission_cond.wait_for(_admission_open)
                    except asyncio.TimeoutError:
                        raise QueueFullError(
                            f"{queue_name}: Queue full, timeout after {_queue_timeout} seconds"
                        )
                    if not accepting_new_tasks:
                        # Woken by shutdown's notify_all.
                        rejected_total += 1
                        raise RuntimeError(f"{queue_name}: Queue is shutting down")
                live_queued += 1

            # Reservation window: until the task state is registered, any
            # exception/cancellation must hand the reservation back or this
            # slot of logical capacity would be occupied forever.
            try:
                async with task_states_lock:
                    task_states[task_id] = task_state
            except BaseException:
                async with admission_cond:
                    live_queued -= 1
                    admission_cond.notify_all()
                raise
            # From here the reservation belongs to the exactly-once rule
            # (worker_started transfer, or the finally cleanup below).

            try:
                active_futures.add(future)

                # Get counter for FIFO ordering
                async with initialization_lock:
                    current_count = counter
                    counter += 1

                # Capture the enqueuer's context (e.g. an active OpenTelemetry
                # span) so the worker executes func() under it instead of the
                # long-lived worker task's own creation-time context. It rides
                # as the 6th tuple element; keep the unpack sites in sync
                # (worker / slot_pump / limited_worker / shutdown drain).
                ctx = contextvars.copy_context()

                # Unbounded physical queue: put_nowait never blocks, and the
                # (priority, count, ...) tuple keeps heap ordering intact.
                queue.put_nowait((_priority, current_count, task_id, args, kwargs, ctx))
                submitted_total += 1
                work_available.set()
                await _publish_stats()

                # Wait for result with the same semantics as the default path
                try:
                    if _timeout is not None:
                        result = await asyncio.wait_for(future, _timeout)
                    else:
                        result = await future
                    completed_total += 1
                    await _publish_stats()
                    return result
                except asyncio.TimeoutError:
                    # User-level timeout: the task may still be queued (e.g.
                    # waiting for a global slot) — mark it cancelled so no
                    # worker ever calls the provider for it.
                    async with task_states_lock:
                        if task_id in task_states:
                            task_states[task_id].cancellation_requested = True

                    if not future.done():
                        future.cancel()

                    cleanup_start = asyncio.get_running_loop().time()
                    while (
                        task_id in task_states
                        and asyncio.get_running_loop().time() - cleanup_start
                        < cleanup_timeout
                    ):
                        await asyncio.sleep(0.1)

                    cancelled_total += 1
                    raise TimeoutError(
                        f"{queue_name}: User timeout after {_timeout} seconds"
                    )
                except WorkerTimeoutError as e:
                    failed_total += 1
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except HealthCheckTimeoutError as e:
                    failed_total += 1
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except asyncio.CancelledError:
                    cancelled_total += 1
                    raise
                except Exception:
                    failed_total += 1
                    raise

            finally:
                active_futures.discard(future)
                notify_needed = False
                async with task_states_lock:
                    popped = task_states.pop(task_id, None)
                    if popped is not None and not popped.worker_started:
                        # Died while still queued: release the reservation
                        # here — the worker never will (exactly-once).
                        live_queued -= 1
                        notify_needed = True
                if notify_needed:
                    async with admission_cond:
                        admission_cond.notify_all()

        @wraps(func)
        async def wait_func(
            *args,
            _priority=DEFAULT_PROCESSING_PRIORITY,
            _timeout=None,
            _queue_timeout=None,
            **kwargs,
        ):
            """
            Execute function with enhanced priority-based concurrency control and timeout handling

            Args:
                *args: Positional arguments passed to the function
                _priority: Call priority (lower values have higher priority)
                _timeout: Maximum time to wait for completion (in seconds, none means determinded by max_execution_timeout of the queue)
                _queue_timeout: Maximum time to wait for entering the queue (in seconds)
                **kwargs: Keyword arguments passed to the function

            Returns:
                The result of the function call

            Raises:
                TimeoutError: If the function call times out at any level
                QueueFullError: If the queue is full and waiting times out
                Any exception raised by the decorated function
            """
            nonlocal submitted_total, completed_total, cancelled_total, failed_total
            nonlocal rejected_total
            if not accepting_new_tasks:
                rejected_total += 1
                raise RuntimeError(f"{queue_name}: Queue is shutting down")

            await ensure_workers()

            if use_global_limit:
                return await _limited_wait(
                    args, kwargs, _priority, _timeout, _queue_timeout
                )

            # Generate unique task ID
            task_id = (
                f"{id(asyncio.current_task())}_{asyncio.get_running_loop().time()}"
            )
            future = asyncio.Future()

            # Create task state
            task_state = TaskState(
                future=future, start_time=asyncio.get_running_loop().time()
            )

            try:
                # Register task state
                async with task_states_lock:
                    task_states[task_id] = task_state

                active_futures.add(future)

                # Get counter for FIFO ordering
                nonlocal counter
                async with initialization_lock:
                    current_count = counter
                    counter += 1

                # Capture the enqueuer's context (e.g. an active
                # OpenTelemetry span) so the worker executes func() under
                # it instead of the long-lived worker task's own
                # creation-time context. It rides as the 6th tuple element;
                # keep the unpack sites in sync (worker / slot_pump /
                # limited_worker / shutdown drain).
                ctx = contextvars.copy_context()

                # Queue the task with timeout handling
                try:
                    if not accepting_new_tasks:
                        rejected_total += 1
                        raise RuntimeError(f"{queue_name}: Queue is shutting down")
                    if _queue_timeout is not None:
                        await asyncio.wait_for(
                            queue.put(
                                (_priority, current_count, task_id, args, kwargs, ctx)
                            ),
                            timeout=_queue_timeout,
                        )
                    else:
                        await queue.put(
                            (_priority, current_count, task_id, args, kwargs, ctx)
                        )
                    submitted_total += 1
                    await _publish_stats()
                except asyncio.TimeoutError:
                    raise QueueFullError(
                        f"{queue_name}: Queue full, timeout after {_queue_timeout} seconds"
                    )
                except Exception as e:
                    # Clean up on queue error
                    if not future.done():
                        future.set_exception(e)
                    raise

                # Wait for result with timeout handling
                try:
                    if _timeout is not None:
                        result = await asyncio.wait_for(future, _timeout)
                    else:
                        result = await future
                    completed_total += 1
                    await _publish_stats()
                    return result
                except asyncio.TimeoutError:
                    # This is user-level timeout (asyncio.wait_for caused)
                    # Mark cancellation request
                    async with task_states_lock:
                        if task_id in task_states:
                            task_states[task_id].cancellation_requested = True

                    # Cancel future
                    if not future.done():
                        future.cancel()

                    # Wait for worker cleanup with timeout
                    cleanup_start = asyncio.get_running_loop().time()
                    while (
                        task_id in task_states
                        and asyncio.get_running_loop().time() - cleanup_start
                        < cleanup_timeout
                    ):
                        await asyncio.sleep(0.1)

                    cancelled_total += 1
                    raise TimeoutError(
                        f"{queue_name}: User timeout after {_timeout} seconds"
                    )
                except WorkerTimeoutError as e:
                    # This is Worker-level timeout, directly propagate exception information
                    failed_total += 1
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except HealthCheckTimeoutError as e:
                    # This is Health Check-level timeout, directly propagate exception information
                    failed_total += 1
                    raise TimeoutError(f"{queue_name}: {str(e)}")
                except asyncio.CancelledError:
                    cancelled_total += 1
                    raise
                except Exception:
                    failed_total += 1
                    raise

            finally:
                # Ensure cleanup
                active_futures.discard(future)
                async with task_states_lock:
                    task_states.pop(task_id, None)

        # Add shutdown method to decorated function
        wait_func.shutdown = shutdown
        wait_func.get_queue_stats = get_queue_stats
        wait_func.get_aggregated_queue_stats = get_aggregated_queue_stats
        # One upkeep pass (lease renewal / pending releases / reaping /
        # compaction / stats flush). The health check runs it every 5s;
        # exposed for tests and operational tooling.
        wait_func.run_maintenance = _run_maintenance

        return wait_func

    return final_dec


def wrap_embedding_func_with_attrs(**kwargs):
    """Decorator to add embedding dimension and token limit attributes to embedding functions.

    This decorator wraps an async embedding function and returns an EmbeddingFunc instance
    that automatically handles dimension parameter injection and attribute management.

    WARNING: DO NOT apply this decorator to wrapper functions that call other
    decorated embedding functions. This will cause double decoration and parameter
    injection conflicts.

    Correct usage patterns:

    1. Direct decoration:
        ```python
        @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192, model_name="my_embedding_model")
        async def my_embed(texts, embedding_dim=None):
            # Direct implementation
            return embeddings
        ```
    2. Double decoration:
        ```python
        @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192, model_name="my_embedding_model")
        @retry(...)
        async def my_embed(texts, ...):
            # Base implementation
            pass

        @wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=4096, model_name="another_embedding_model")
        # Note: No @retry here!
        async def my_new_embed(texts, ...):
            # CRITICAL: Call .func to access unwrapped function
            return await my_embed.func(texts, ...)  # ✅ Correct
            # return await my_embed(texts, ...)     # ❌ Wrong - double decoration!
        ```
    3. Context-aware decoration:
        ```python
        @wrap_embedding_func_with_attrs(
            embedding_dim=1536,
            model_name="my_embedding_model",
            supports_asymmetric=True
        )
        async def my_embed(texts, context="document"):
            # Apply different prefixes based on context
            if context == "query":
                texts = ["search_query: " + t for t in texts]
            elif context == "document":
                texts = ["search_document: " + t for t in texts]
            return embeddings
        ```

    The decorated function becomes an EmbeddingFunc instance with:
    - embedding_dim: The embedding dimension
    - max_token_size: Maximum token limit (optional)
    - model_name: Model name (optional)
    - supports_asymmetric: Whether context parameter is supported (optional)
    - func: The original unwrapped function (access via .func)
    - __call__: Wrapper that injects embedding_dim parameter and context

    Args:
        embedding_dim: The dimension of embedding vectors
        max_token_size: Maximum number of tokens (optional)
        send_dimensions: Whether to pass embedding_dim as a keyword argument (for models with configurable embedding dimensions).
        supports_asymmetric: Whether the function supports context parameter (optional).
            If omitted, this is auto-detected from the wrapped function's signature
            (set to True iff the function accepts a ``context`` parameter).

    Returns:
        A decorator that wraps the function as an EmbeddingFunc instance
    """

    def final_dec(func) -> EmbeddingFunc:
        embedding_kwargs = dict(kwargs)
        # Auto-detect supports_asymmetric from the wrapped function's signature
        # if the caller did not declare it explicitly. Without this, any user or
        # third-party embed function that accepts a `context` parameter but
        # forgets to set ``supports_asymmetric=True`` would have its `context`
        # silently dropped by ``EmbeddingFunc.__call__``, defeating the
        # task-aware embedding feature.
        if "supports_asymmetric" not in embedding_kwargs:
            try:
                sig = inspect.signature(func)
                embedding_kwargs["supports_asymmetric"] = "context" in sig.parameters
            except (TypeError, ValueError):
                # inspect.signature can fail for builtins; fall back to False.
                embedding_kwargs["supports_asymmetric"] = False
        new_func = EmbeddingFunc(**embedding_kwargs, func=func)
        return new_func

    return final_dec


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8-sig") as f:
        content = f.read()
    # Empty/whitespace existing file: same contract as missing (callers use or {}).
    if not content.strip():
        logger.warning("Empty JSON file treated as missing: %s", file_name)
        return None
    return json.loads(content)


def _sanitize_string_for_json(text: str) -> str:
    """Remove characters that cannot be encoded in UTF-8 for JSON serialization.

    Uses regex for optimal performance with zero-copy optimization for clean strings.
    Fast detection path for clean strings (99% of cases) with efficient removal for dirty strings.

    Args:
        text: String to sanitize

    Returns:
        Original string if clean (zero-copy), sanitized string if dirty
    """
    if not text:
        return text

    # Fast path: Check if sanitization is needed using C-level regex search
    if not _SURROGATE_PATTERN.search(text):
        return text  # Zero-copy for clean strings - most common case

    # Slow path: Remove problematic characters using C-level regex substitution
    return _SURROGATE_PATTERN.sub("", text)


class SanitizingJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that sanitizes data during serialization.

    This encoder cleans strings during the encoding process without creating
    a full copy of the data structure, making it memory-efficient for large datasets.
    """

    def encode(self, o):
        """Override encode method to handle simple string cases"""
        if isinstance(o, str):
            return json.encoder.encode_basestring(_sanitize_string_for_json(o))
        return super().encode(o)

    def iterencode(self, o, _one_shot=False):
        """
        Override iterencode to sanitize strings during serialization.
        This is the core method that handles complex nested structures.
        """
        # Preprocess: sanitize all strings in the object
        sanitized = self._sanitize_for_encoding(o)

        # Call parent's iterencode with sanitized data
        for chunk in super().iterencode(sanitized, _one_shot):
            yield chunk

    def _sanitize_for_encoding(self, obj):
        """
        Recursively sanitize strings in an object.
        Creates new objects only when necessary to avoid deep copies.

        Args:
            obj: Object to sanitize

        Returns:
            Sanitized object with cleaned strings
        """
        if isinstance(obj, str):
            return _sanitize_string_for_json(obj)

        elif isinstance(obj, dict):
            # Create new dict with sanitized keys and values
            new_dict = {}
            for k, v in obj.items():
                clean_k = _sanitize_string_for_json(k) if isinstance(k, str) else k
                clean_v = self._sanitize_for_encoding(v)
                new_dict[clean_k] = clean_v
            return new_dict

        elif isinstance(obj, (list, tuple)):
            # Sanitize list/tuple elements
            cleaned = [self._sanitize_for_encoding(item) for item in obj]
            return type(obj)(cleaned) if isinstance(obj, tuple) else cleaned

        else:
            # Numbers, booleans, None, etc. remain unchanged
            return obj


def write_json(json_obj, file_name):
    """
    Write JSON data to file with optimized sanitization strategy.

    This function uses a two-stage approach:
    1. Fast path: Try direct serialization (works for clean data ~99% of time)
    2. Slow path: Use custom encoder that sanitizes during serialization

    The custom encoder approach avoids creating a deep copy of the data,
    making it memory-efficient. When sanitization occurs, the caller should
    reload the cleaned data from the file to update shared memory.

    Writes are atomic: both the fast path and the sanitizing fallback land
    in the same per-writer tmp sibling, and only the final ``os.replace``
    publishes the file. A crash mid-write leaves the prior snapshot intact.

    Args:
        json_obj: Object to serialize (may be a shallow copy from shared memory)
        file_name: Output file path

    Returns:
        bool: True if sanitization was applied (caller should reload data),
              False if direct write succeeded (no reload needed)
    """
    from lightrag.file_atomic import atomic_write

    sanitized = False

    def _do_write(tmp_path: str) -> None:
        nonlocal sanitized
        try:
            # Strategy 1: Fast path - try direct serialization.
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2, ensure_ascii=False)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            logger.debug(f"Direct JSON write failed, using sanitizing encoder: {e}")
            # Strategy 2: Use sanitizing encoder (zero-copy). Reusing the
            # same tmp path keeps the operation single-rename even on the
            # slow path.
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    json_obj,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    cls=SanitizingJSONEncoder,
                )
            sanitized = True

    atomic_write(file_name, _do_write)

    if sanitized:
        logger.info(f"JSON sanitization applied during write: {file_name}")
    return sanitized


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """A character-offset span into an original string, plus its own token count.

    ``content[start:end]`` is always a verbatim substring of the string that
    produced this span — never a decoded reconstruction — and independently
    re-encoding it under the same :class:`Tokenizer` yields exactly
    ``token_count`` tokens, which is guaranteed to be ``<= max_tokens`` for
    whatever budget produced the span. See the ``Tokenizer.split_by_token_limit``
    / ``truncate_by_token_limit`` docstrings for the full contract.
    """

    start: int
    end: int
    token_count: int


class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.

    Implementations MUST be safe to call concurrently from multiple OS threads —
    LightRAG runs token counting in worker threads to keep it off the asyncio
    event loop, and does not serialize calls on the implementation's behalf. See
    :class:`Tokenizer` for why serializing here would be the wrong layer.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens.

        May be called concurrently from multiple threads.
        """
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string.

        May be called concurrently from multiple threads.
        """
        ...


# ---------------------------------------------------------------------------
# Bounded submission to a thread pool
# ---------------------------------------------------------------------------
#
# Moving CPU-bound work off the event loop frees the loop to keep accepting
# requests, which means more work can be in flight at once than before. A
# ``ThreadPoolExecutor``'s wait queue is unbounded, so submissions need their own
# ceiling.
#
# The subtle part is WHO holds the permit. Writing ``async with sem: await
# run_in_executor(...)`` is wrong: cancelling the awaiting coroutine returns the
# permit immediately, but the thread-pool task is not cancelled — an already
# running ``concurrent.futures.Future.cancel()`` just returns False and the thread
# runs to completion. A client that repeatedly submits and cancels would hand back
# permits it is still consuming and rebuild an unbounded queue. The permit
# therefore belongs to the executor future and is released from its done callback,
# so permit occupancy tracks actual thread occupancy exactly.


def _consume_future_exception(fut: "asyncio.Future") -> None:
    """Mark a shielded future's exception as retrieved.

    When ``shield`` is cancelled nobody awaits the inner future any more, so an
    exception raised by the thread would surface as an "exception was never
    retrieved" traceback at GC time. Retrieving it here only clears that log
    flag; a caller that is still awaiting still sees the exception.
    """
    if not fut.cancelled():
        fut.exception()


async def bounded_submit(
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``fn`` in ``executor`` under a permit owned by the executor future.

    Args:
        executor: destination pool.
        semaphore: submission ceiling for this pool, from :func:`get_loop_semaphore`.
        fn: synchronous callable to run in the pool.
        *args, **kwargs: forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.

    Cancelling the caller does not cancel the thread: the work runs to completion
    and only then is the permit returned. That is deliberate — see the module
    comment above.
    """
    await semaphore.acquire()
    try:
        concurrent_future: ConcurrentFuture = executor.submit(
            contextvars.copy_context().run, partial(fn, *args, **kwargs)
        )
    except BaseException:
        semaphore.release()
        raise

    loop = asyncio.get_running_loop()

    def _release(_done: ConcurrentFuture) -> None:
        # Runs on the worker thread, so the release has to be posted back to the
        # loop that owns the semaphore. A closed loop means the semaphore died
        # with it and there is nothing left to release.
        try:
            loop.call_soon_threadsafe(semaphore.release)
        except RuntimeError:
            pass

    concurrent_future.add_done_callback(_release)

    async_future = asyncio.wrap_future(concurrent_future)
    async_future.add_done_callback(_consume_future_exception)
    return await asyncio.shield(async_future)


# Semaphores are per event loop, executors are per process. ``asyncio.Semaphore``
# binds to the first loop that has to wait on it and raises from any other loop
# (``asyncio.mixins._LoopBoundMixin``), and the test suite runs one fresh loop per
# case, so a module-level singleton would fail. Storing it ON the loop makes its
# lifetime exactly the loop's; a container keyed by the loop would not, because
# once contended the semaphore holds a strong reference back to its loop and a
# ``WeakKeyDictionary`` entry whose value references its key never expires.
_LOOP_SEMAPHORE_ATTR_PREFIX = "_lightrag_submit_semaphore_"
_LOOP_SEMAPHORE_FALLBACK: dict[int, tuple[Any, dict[str, asyncio.Semaphore]]] = {}


def _fallback_semaphore(loop: Any, name: str, capacity: int) -> asyncio.Semaphore:
    """Per-loop semaphore for loops that reject attribute assignment.

    The entry keeps a strong reference to the loop on purpose. Deriving the loop
    from the semaphore does not work: ``Semaphore.acquire()`` only binds ``_loop``
    when it actually has to wait, so an uncontended semaphore never learns which
    loop it belongs to, leaving nothing to test ``is_closed()`` on and no way to
    detect ``id()`` reuse. Closed loops are swept on the next call, so a closed
    loop outlives itself by at most one lookup.
    """
    for stale_key, (stale_loop, _) in list(_LOOP_SEMAPHORE_FALLBACK.items()):
        if stale_loop.is_closed():
            _LOOP_SEMAPHORE_FALLBACK.pop(stale_key, None)
    key = id(loop)
    entry = _LOOP_SEMAPHORE_FALLBACK.get(key)
    if entry is None or entry[0] is not loop:
        entry = (loop, {})
        _LOOP_SEMAPHORE_FALLBACK[key] = entry
    semaphores = entry[1]
    semaphore = semaphores.get(name)
    if semaphore is None:
        semaphore = asyncio.Semaphore(capacity)
        semaphores[name] = semaphore
    return semaphore


def get_loop_semaphore(name: str, capacity: int) -> asyncio.Semaphore:
    """Return the running loop's semaphore called ``name``, creating it once.

    ``capacity`` is only used on creation. It is intentionally a fixed constant
    rather than a per-``LightRAG`` setting: this helper is module level and is
    called from places that hold no instance, and several instances sharing a loop
    would otherwise each build their own semaphore and lose the global ceiling.
    """
    loop = asyncio.get_running_loop()
    attr = _LOOP_SEMAPHORE_ATTR_PREFIX + name
    semaphore = getattr(loop, attr, None)
    if semaphore is not None:
        return semaphore
    semaphore = asyncio.Semaphore(capacity)
    try:
        setattr(loop, attr, semaphore)
    except (AttributeError, TypeError):
        return _fallback_semaphore(loop, name, capacity)
    return semaphore


# ---------------------------------------------------------------------------
# Chunking off the event loop
# ---------------------------------------------------------------------------
#
# Chunking is CPU-bound in the document the caller supplied and, for the
# recursive strategy, in the separator cascade supplied with it. Run inline in an
# ``async def`` — which is what every branch except V did — it holds the only
# thread serving HTTP for its whole duration, so one 414 KiB document made an
# unrelated ``GET /health`` take 63 seconds (GHSA-26pm-px5v-8c4w).
#
# One worker. Chunking is serialized by the event loop as things stand — while
# one document chunks, no other document's coroutine can run — so a single worker
# preserves that concurrency exactly while freeing the loop, rather than
# introducing parallelism this change has no reason to introduce.
#
# Invariants:
#   * code running in here calls the SYNCHRONOUS tokenizer API. Awaiting an async
#     helper from a worker thread would park it on the loop while the loop waits
#     for the thread.
#   * no nested submissions. One worker plus a held permit means a task that
#     submits again deadlocks. Where a chunker calls another chunker (V's
#     post-processing calls R), it calls it DIRECTLY — it is already in the pool.

_CHUNKING_EXECUTOR: Optional[ThreadPoolExecutor] = None
_CHUNKING_EXECUTOR_GUARD = threading.Lock()


def get_chunking_executor() -> ThreadPoolExecutor:
    """The process-wide single-worker pool used for chunking."""
    global _CHUNKING_EXECUTOR
    if _CHUNKING_EXECUTOR is None:
        with _CHUNKING_EXECUTOR_GUARD:
            if _CHUNKING_EXECUTOR is None:
                _CHUNKING_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="lightrag-chunking"
                )
    return _CHUNKING_EXECUTOR


async def run_in_chunking_executor(fn: Callable[..., Any], *args: Any, **kwargs) -> Any:
    """Run a synchronous, chunking-bound callable off the event loop.

    Bounded like any other submission: the submitters include public SDK entry
    points (``ainsert_custom_kg``, ``ainsert_custom_chunks``) gated by neither
    ``max_parallel_insert`` nor the pipeline's busy flag, so "the pipeline limits
    concurrency" is not a ceiling that actually holds here.
    """
    from lightrag.constants import CHUNKING_SUBMIT_LIMIT

    semaphore = get_loop_semaphore("chunking", CHUNKING_SUBMIT_LIMIT)
    return await bounded_submit(
        get_chunking_executor(), semaphore, partial(fn, *args, **kwargs)
    )


class Tokenizer:
    """
    A wrapper around a tokenizer to provide a consistent interface for encoding and decoding.

    Thread-safety contract
    ----------------------
    ``encode`` and ``decode`` MAY be called concurrently from several OS threads:
    token counting is CPU-bound and runs in worker threads so it does not block
    the asyncio event loop. **An injected tokenizer is responsible for its own
    thread safety** — through immutable state, an internal lock, thread-local
    state, or whatever its implementation requires. LightRAG does not serialize
    it, and adding a lock here would be actively harmful: the event loop would
    then wait on a worker thread holding it, recreating the very freeze that
    moving this work off the loop exists to remove.

    The built-in :class:`TiktokenTokenizer` satisfies this: ``tiktoken``'s own
    ``encode_batch`` / ``decode_batch`` fan ``Encoding.encode`` / ``.decode`` of a
    single instance across a ``ThreadPoolExecutor``, so concurrent use is a
    supported capability of the library rather than an assumption made here. The
    dependency floor in ``pyproject.toml`` is pinned to the oldest release for
    which that holds.

    Note that sharing is the norm, not the exception: ``tiktoken`` caches
    encodings in a process-wide registry, and ``Encoding.__setstate__`` rebinds a
    copy's ``__dict__`` to the registered instance's — so every
    ``TiktokenTokenizer`` in the process, including ones produced by
    ``copy.deepcopy``, funnels into a single ``CoreBPE``. An injected tokenizer
    that is not thread-safe cannot be made safe by copying it.

    Deep-copy requirement
    ---------------------
    An injected tokenizer must additionally survive ``copy.deepcopy``: ``LightRAG``
    is a dataclass and ``_build_global_config`` runs ``dataclasses.asdict`` over
    it, which deep-copies non-dataclass fields. An implementation that achieves
    thread safety with an internal ``threading.Lock`` would otherwise fail at
    construction with ``TypeError: cannot pickle '_thread.lock' object``. Such an
    implementation should define ``__deepcopy__`` returning ``self`` — correct
    precisely because being thread-safe is what makes it shareable.

    ``_build_global_config`` then restores this object over the copy ``asdict``
    made, so every consumer reading ``global_config["tokenizer"]`` holds the same
    instance as ``LightRAG.tokenizer``. That is an identity guarantee, not a
    saving: ``asdict`` copies the field before the restore can intervene, which is
    exactly why the deep-copy requirement above still applies.
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface,
                which must be safe to call concurrently from multiple threads.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        May be called concurrently from multiple threads; see the class docstring.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        try:
            return self.tokenizer.encode(content)
        except ValueError as e:
            # tiktoken (and some other tokenizers) raise ValueError when the
            # content contains literal special-token strings such as
            # "<|endoftext|>", because by default disallowed_special is the
            # full set of special tokens. This crashes document indexing on
            # any user content that happens to contain those strings — common
            # in documentation, notes, or model output captured in source
            # corpora. Retry with disallowed_special=() so the tokens are
            # encoded as ordinary text. Tokenizers that don't accept the
            # kwarg fall through and re-raise the original error.
            if "special token" not in str(e):
                raise
            try:
                return self.tokenizer.encode(content, disallowed_special=())
            except TypeError:
                raise e

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        May be called concurrently from multiple threads; see the class docstring.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)

    # -----------------------------------------------------------------------
    # Safe split/truncate contract
    # -----------------------------------------------------------------------
    #
    # These default implementations rely ONLY on ``encode()`` — never on
    # ``decode()`` — because ``decode(tokens[:k])`` is not guaranteed to be a
    # character-exact prefix of the original string for an arbitrary
    # third-party tokenizer (only tiktoken's ``decode_with_offsets`` gives a
    # reliable token-to-character mapping; see ``TiktokenTokenizer`` below for
    # the optimized override). Every candidate here is instead a literal
    # character-offset slice of the input, so there is no alignment risk, at
    # the cost of a handful of extra ``encode()`` calls per span compared to
    # the tiktoken fast path.
    #
    # BPE token count is not monotonic in character-prefix length, so this is
    # a bounded, STRICTLY ONE-DIRECTIONAL retreat: candidate length only ever
    # decreases (by exponential character steps: 1, 2, 4, 8, ...), never
    # re-expands. That rules out oscillation and bounds the number of
    # re-encodes to O(log max_tokens) regardless of how good the initial
    # char/token ratio estimate turns out to be for a given region of text.
    #
    # No cross-call state: nothing here is written back onto ``self``. An
    # injected tokenizer must remain safe under concurrent calls from
    # multiple threads and under ``copy.deepcopy`` (see the class docstring),
    # and any mutable "observed retreat" state would violate both.

    def _bounded_prefix_span(
        self, content: str, start: int, max_tokens: int, chars_per_token: float
    ) -> TokenSpan:
        """Find a safe ``TokenSpan`` starting at character offset ``start``.

        ``chars_per_token`` is a rough estimate (chars per token, from a prior
        whole-content encode) used only to pick a starting candidate length;
        the result is always independently verified by encoding the exact
        candidate substring.
        """
        remaining_len = len(content) - start
        candidate_len = min(remaining_len, max(1, int(max_tokens * chars_per_token)))
        step = 1
        while True:
            candidate = content[start : start + candidate_len]
            count = len(self.encode(candidate))
            if count <= max_tokens:
                return TokenSpan(start, start + candidate_len, count)
            if candidate_len <= 1:
                break
            candidate_len = max(1, candidate_len - step)
            step *= 2

        first_cp = content[start : start + 1]
        first_cp_tokens = len(self.encode(first_cp))
        if first_cp_tokens > max_tokens:
            raise TokenBudgetError(
                max_tokens, first_cp_tokens, content[start : start + 80]
            )
        return TokenSpan(start, start + 1, first_cp_tokens)

    def truncate_by_token_limit(self, content: str, max_tokens: int) -> TokenSpan:
        """Return the longest safe prefix of ``content`` that fits ``max_tokens``.

        The result always starts at character offset 0. If the whole string
        already fits, the returned span covers it entirely. Raises
        ``ValueError`` for a non-positive budget, and ``TokenBudgetError`` if
        not even the first complete Unicode code point fits.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not content:
            return TokenSpan(0, 0, 0)

        full_tokens = len(self.encode(content))
        if full_tokens <= max_tokens:
            return TokenSpan(0, len(content), full_tokens)

        chars_per_token = len(content) / full_tokens
        return self._bounded_prefix_span(content, 0, max_tokens, chars_per_token)

    def split_by_token_limit(
        self, content: str, max_tokens: int, overlap_tokens: int = 0
    ) -> List[TokenSpan]:
        """Split ``content`` into safe, contiguous, optionally-overlapping spans.

        Covers the whole string with no gaps and real forward progress: every
        non-final span strictly extends the covered range (see the module
        contract notes above the class). ``overlap_tokens`` is a best-effort
        target, not a guarantee of the mathematically closest boundary — BPE
        token count is not monotonic in character length, so this generic
        implementation only guarantees a safe, progress-making boundary near
        the target, retreating the target itself (by the same exponential
        steps) whenever the estimated overlap would fail to make progress.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if overlap_tokens < 0:
            raise ValueError(
                f"overlap_tokens must be non-negative, got {overlap_tokens}"
            )
        if not content:
            return []

        total_tokens = len(self.encode(content))
        if total_tokens <= max_tokens:
            return [TokenSpan(0, len(content), total_tokens)]

        # One global char/token ratio estimate, reused as a local performance
        # hint for every window and overlap guess in this call only — never
        # persisted on `self` (see the note above).
        chars_per_token = len(content) / total_tokens

        spans: List[TokenSpan] = []
        covered_end = 0
        total_len = len(content)

        while covered_end < total_len:
            if covered_end > 0:
                # Clamp to what the previous window can actually lend before
                # even trying: an overlap target that dwarfs the previous
                # window's own token count would otherwise retreat from that
                # huge starting point step by step, wasting retries on values
                # that could never have worked.
                target_overlap = min(overlap_tokens, max(0, spans[-1].token_count - 1))
            else:
                target_overlap = 0
            step = 1
            span: Optional[TokenSpan] = None
            while True:
                overlap_chars = (
                    int(round(target_overlap * chars_per_token))
                    if target_overlap > 0
                    else 0
                )
                start = max(0, covered_end - overlap_chars)
                try:
                    candidate_span = self._bounded_prefix_span(
                        content, start, max_tokens, chars_per_token
                    )
                except TokenBudgetError:
                    if target_overlap == 0:
                        raise
                    candidate_span = None
                if candidate_span is not None and (
                    candidate_span.end > covered_end or target_overlap == 0
                ):
                    span = candidate_span
                    break
                target_overlap = max(0, target_overlap - step)
                step *= 2
            spans.append(span)
            covered_end = span.end

        return spans


class TiktokenTokenizer(Tokenizer):
    """
    A Tokenizer implementation using the tiktoken library.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the TiktokenTokenizer with a specified model name.

        Args:
            model_name: The model name for the tiktoken tokenizer to use.  Defaults to "gpt-4o-mini".

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If the model_name is invalid.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` or define custom `tokenizer_func`."
            )

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.")

    # -----------------------------------------------------------------------
    # Optimized split/truncate: decode_with_offsets fast path
    # -----------------------------------------------------------------------
    #
    # `tiktoken.Encoding.decode_with_offsets` (available since the repo's
    # declared `tiktoken>=0.7.0` floor) maps each token index to the character
    # offset where it starts, so a candidate "keep the first k tokens" prefix
    # can be located in O(1) instead of guessing a char/token ratio. The
    # candidate substring is still always independently re-encoded via
    # `self.encode` (never `self.tokenizer.encode` directly, to preserve the
    # base class's disallowed_special=() fallback) before being trusted,
    # because BPE re-tokenization of a truncated substring is not guaranteed
    # to reproduce the same token count.
    #
    # Retreat here happens in TOKEN space (1, 2, 4, 8, ... tokens) rather than
    # character space, which is both more precise and, since consecutive token
    # counts can map to the same character offset when a multi-byte UTF-8
    # character's bytes are split across tokens, deduplicated so the same
    # substring is never re-encoded twice in a row.

    def _prefix_span_via_boundary(
        self,
        content: str,
        start_char: int,
        start_token: int,
        max_tokens: int,
        total_tokens: int,
        boundary: Callable[[int], int],
    ) -> tuple[TokenSpan, int]:
        """Find a safe span starting at ``start_char``/``start_token``.

        ``boundary(k)`` returns the character offset where absolute token
        index ``k`` starts (``boundary(total_tokens)`` is ``len(content)``).
        Returns ``(span, end_token)`` — the absolute token index the span's
        ``end`` corresponds to, so callers can resume from it without
        re-deriving it via a char-offset search.
        """
        remaining_tokens = total_tokens - start_token
        k = min(remaining_tokens, max_tokens)
        step = 1
        last_end: Optional[int] = None
        while True:
            end = boundary(start_token + k)
            if end != last_end:
                # A token boundary can coincide with start_char (a
                # continuation-byte-only token reports no character advance —
                # see decode_with_offsets' docstring); an empty candidate
                # would trivially satisfy count <= max_tokens without
                # covering anything, so it must never be accepted here.
                if end > start_char:
                    candidate = content[start_char:end]
                    count = len(self.encode(candidate))
                    if count <= max_tokens:
                        return TokenSpan(start_char, end, count), start_token + k
                last_end = end
            if k <= 1:
                break
            k = max(1, k - step)
            step *= 2

        # Floor: fall back to a direct check of the first complete Unicode
        # code point, bypassing token-boundary ambiguity entirely (a single
        # code point's bytes can themselves be split across tokens).
        end_char = start_char + 1
        first_cp = content[start_char:end_char]
        first_cp_tokens = len(self.encode(first_cp))
        if first_cp_tokens > max_tokens:
            raise TokenBudgetError(
                max_tokens, first_cp_tokens, content[start_char : start_char + 80]
            )
        end_token = start_token
        while end_token < total_tokens and boundary(end_token) < end_char:
            end_token += 1
        return TokenSpan(start_char, end_char, first_cp_tokens), end_token

    def truncate_by_token_limit(self, content: str, max_tokens: int) -> TokenSpan:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if not content:
            return TokenSpan(0, 0, 0)

        tokens = self.encode(content)
        total_tokens = len(tokens)
        if total_tokens <= max_tokens:
            return TokenSpan(0, len(content), total_tokens)

        try:
            decoded_text, offsets = self.tokenizer.decode_with_offsets(tokens)
        except Exception:
            return super().truncate_by_token_limit(content, max_tokens)
        if decoded_text != content:
            # Lossy roundtrip (shouldn't normally happen once special tokens
            # are handled via self.encode's fallback) — fall back to the
            # generic, alignment-safe implementation rather than risk an
            # off-by-however-many char offset.
            return super().truncate_by_token_limit(content, max_tokens)

        def boundary(k: int) -> int:
            return offsets[k] if k < total_tokens else len(content)

        span, _ = self._prefix_span_via_boundary(
            content, 0, 0, max_tokens, total_tokens, boundary
        )
        return span

    def split_by_token_limit(
        self, content: str, max_tokens: int, overlap_tokens: int = 0
    ) -> List[TokenSpan]:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if overlap_tokens < 0:
            raise ValueError(
                f"overlap_tokens must be non-negative, got {overlap_tokens}"
            )
        if not content:
            return []

        tokens = self.encode(content)
        total_tokens = len(tokens)
        if total_tokens <= max_tokens:
            return [TokenSpan(0, len(content), total_tokens)]

        try:
            decoded_text, offsets = self.tokenizer.decode_with_offsets(tokens)
        except Exception:
            return super().split_by_token_limit(content, max_tokens, overlap_tokens)
        if decoded_text != content:
            return super().split_by_token_limit(content, max_tokens, overlap_tokens)

        def boundary(k: int) -> int:
            return offsets[k] if k < total_tokens else len(content)

        spans: List[TokenSpan] = []
        covered_end_char = 0
        covered_end_token = 0
        total_len = len(content)

        while covered_end_char < total_len:
            if covered_end_token > 0:
                # Clamp before trying: an overlap target that dwarfs the
                # previous window's own token count would otherwise retreat
                # from that huge starting point step by step, wasting
                # retries on values that could never have worked.
                target_overlap = min(overlap_tokens, max(0, spans[-1].token_count - 1))
            else:
                target_overlap = 0
            step = 1
            accepted: Optional[tuple[TokenSpan, int]] = None
            while True:
                start_token = max(0, covered_end_token - target_overlap)
                start_char = boundary(start_token)
                try:
                    candidate = self._prefix_span_via_boundary(
                        content,
                        start_char,
                        start_token,
                        max_tokens,
                        total_tokens,
                        boundary,
                    )
                except TokenBudgetError:
                    if target_overlap == 0:
                        raise
                    candidate = None
                if candidate is not None and (
                    candidate[0].end > covered_end_char or target_overlap == 0
                ):
                    accepted = candidate
                    break
                target_overlap = max(0, target_overlap - step)
                step *= 2
            span, covered_end_token = accepted
            spans.append(span)
            covered_end_char = span.end

        return spans


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# ---------------------------------------------------------------------------
# Token counting off the event loop
# ---------------------------------------------------------------------------
#
# Tokenizing is CPU-bound and roughly linear in the input the caller chose —
# about half a second per MiB — so doing it inline in an ``async def`` stops the
# process from answering anything at all for the duration, /health included
# (GHSA-r8jh-295g-vv42). These helpers move it to a thread.
#
# One worker, deliberately. Query-side tokenizing is serialized by the event loop
# today, so a single worker preserves that concurrency exactly rather than
# introducing parallelism this change has no reason to introduce. Thread-safety
# is NOT what the single worker is for — that is the injected tokenizer's own
# responsibility (see ``Tokenizer``), and it does not depend on how many pools
# exist.
#
# Invariant: code running inside this executor must call the SYNCHRONOUS
# tokenizer API. Awaiting these helpers from a worker thread would park that
# thread on the loop while the loop waits for the thread.

_TOKENIZER_EXECUTOR: Optional[ThreadPoolExecutor] = None
_TOKENIZER_EXECUTOR_GUARD = threading.Lock()


def get_tokenizer_executor() -> ThreadPoolExecutor:
    """The process-wide single-worker pool used for token counting."""
    global _TOKENIZER_EXECUTOR
    if _TOKENIZER_EXECUTOR is None:
        with _TOKENIZER_EXECUTOR_GUARD:
            if _TOKENIZER_EXECUTOR is None:
                _TOKENIZER_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="lightrag-tokenizer"
                )
    return _TOKENIZER_EXECUTOR


async def run_in_tokenizer_executor(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a synchronous, tokenizer-bound callable off the event loop.

    Callers that already have a whole loop of encodes should hand over the loop,
    not each encode: submissions must stay O(1) per request or the executor's
    queue depth scales with the work instead of with the concurrency.
    """
    from lightrag.constants import TOKENIZER_SUBMIT_LIMIT

    semaphore = get_loop_semaphore("tokenizer", TOKENIZER_SUBMIT_LIMIT)
    return await bounded_submit(get_tokenizer_executor(), semaphore, fn, *args)


async def aencode(tokenizer: Tokenizer, content: str) -> List[int]:
    """Encode ``content`` without occupying the event loop."""
    return await run_in_tokenizer_executor(tokenizer.encode, content)


async def acount_tokens(tokenizer: Tokenizer, content: str) -> int:
    """Token count of ``content``, computed off the event loop."""
    return await run_in_tokenizer_executor(_count_tokens_sync, tokenizer, content)


def _count_tokens_sync(tokenizer: Tokenizer, content: str) -> int:
    return len(tokenizer.encode(content))


def _rendered_prefix_item_count(
    rendered: list[str], separator: str, safe_end: int
) -> int:
    """Largest ``k`` such that ``separator.join(rendered[:k])`` fits in ``safe_end`` chars."""
    cumulative = 0
    sep_len = len(separator)
    for i, text in enumerate(rendered):
        cumulative += (sep_len if i > 0 else 0) + len(text)
        if cumulative > safe_end:
            return i
    return len(rendered)


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    separator: str,
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[Any]:
    """Truncate a list of data by token size, keeping only whole items.

    Counts the real serialized text — every item's ``key(item)`` joined by
    ``separator`` — so the separator's own tokens are part of the budget
    (the previous per-item-only count silently missed them; see #3559).
    Never partially truncates an item: the result is always "keep the first
    K complete items, drop the rest", never a half-rendered item.

    ``key``/``separator`` must match exactly what the caller will actually
    render downstream — a mismatch (e.g. truncating on a fuller dict than the
    one that ends up serialized) reintroduces the same class of undercount.
    """
    if max_token_size <= 0 or not list_data:
        return []

    rendered = [key(data) for data in list_data]
    full_text = separator.join(rendered)
    try:
        safe_span = tokenizer.truncate_by_token_limit(full_text, max_token_size)
    except TokenBudgetError:
        return []

    k = _rendered_prefix_item_count(rendered, separator, safe_span.end)

    # BPE token count is not monotonic in text length, so the safe prefix
    # above is not proof that the first k items' OWN serialization (which is
    # shorter, since it excludes whatever partial item/separator was cut off)
    # is itself safe — independently re-verify and shrink k if needed.
    while k > 0:
        candidate_text = separator.join(rendered[:k])
        if len(tokenizer.encode(candidate_text)) <= max_token_size:
            break
        k -= 1

    return list_data[:k]


async def atruncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    separator: str,
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[Any]:
    """Async :func:`truncate_list_by_token_size`.

    The WHOLE loop is one submission, never one per element. Submitting per
    element would make the executor's queue depth scale with the list length
    times the number of in-flight requests, turning a bounded queue into an
    amplifier — and this loop routinely runs over every retrieved chunk, which is
    the largest single block of synchronous tokenizing on the query path.
    """
    return await run_in_tokenizer_executor(
        truncate_list_by_token_size,
        list_data,
        key,
        separator,
        max_token_size,
        tokenizer,
    )


def normalize_string_list(raw_values: Any, context: str = "") -> list[str]:
    """Return a list of non-empty strings from raw_values.

    Non-string elements are dropped and logged as warnings. If raw_values is
    not a list, an empty list is returned.
    """
    if not isinstance(raw_values, list):
        return []
    result = []
    for i, value in enumerate(raw_values):
        if isinstance(value, str) and value:
            result.append(value)
        else:
            logger.warning(
                "Non-string element dropped from list%s at index %d: %r",
                f" ({context})" if context else "",
                i,
                value,
            )
    return result


def split_text_by_token_limit(
    text: str, tokenizer: Tokenizer, max_tokens: int
) -> list[str]:
    """Deprecated: use ``tokenizer.split_by_token_limit`` directly instead.

    Thin compatibility wrapper around the safe ``TokenSpan``-based split
    contract (zero overlap, no sentence-boundary preservation — overlap is
    what now carries the "don't lose context at a cut" job the old
    sentence-first packer used to do; see
    ``enforce_chunk_token_limit_before_embedding`` for the overlap-bearing
    caller). Stays permissive on a non-positive ``max_tokens`` (returns ``[]``
    rather than raising) to match this function's pre-existing contract for
    any remaining direct callers.
    """
    if not text or max_tokens <= 0:
        return []
    spans = tokenizer.split_by_token_limit(text, max_tokens, overlap_tokens=0)
    return [text[span.start : span.end] for span in spans]


def _parent_to_source_projection(
    source_content: str, parent_start: int, parent_end: int, parent_content: str
) -> list[int] | None:
    """Monotonic char-offset map from ``parent_content`` positions to ``source_content``.

    Built once per parent chunk and reused for every hard-split child it
    produces (an ``O(parent length)`` build shared across ``N`` children,
    instead of redone per child). ``proj[i]`` for ``0 <= i <= len(parent_content)``
    is the absolute ``source_content`` offset corresponding to position ``i``
    in ``parent_content``.

    Needed because a parent's own ``content`` is not always a byte-verbatim
    slice of the document text it was extracted from — e.g. the V strategy's
    ``SemanticChunker`` rejoins sentences with a single space, which can
    differ from the original whitespace run by more or fewer characters.
    Returns ``None`` if the two cannot be reconciled even after removing all
    whitespace (content diverged for some other reason, e.g. a summarizing
    rewrite) — the caller must not guess in that case.
    """
    source_slice = source_content[parent_start:parent_end]
    if source_slice == parent_content:
        return list(range(parent_start, parent_end + 1))

    src_non_ws_positions = [i for i, ch in enumerate(source_slice) if not ch.isspace()]
    parent_non_ws_positions = [
        i for i, ch in enumerate(parent_content) if not ch.isspace()
    ]
    if len(src_non_ws_positions) != len(parent_non_ws_positions) or [
        source_slice[i] for i in src_non_ws_positions
    ] != [parent_content[i] for i in parent_non_ws_positions]:
        return None

    proj: list[Optional[int]] = [None] * (len(parent_content) + 1)
    for p_idx, s_idx in zip(parent_non_ws_positions, src_non_ws_positions):
        proj[p_idx] = parent_start + s_idx

    # Whitespace runs (and the trailing end position) inherit the offset of
    # the next known non-whitespace position, so every index resolves to a
    # definite, still-monotonic offset.
    next_val = parent_end
    for i in range(len(proj) - 1, -1, -1):
        if proj[i] is None:
            proj[i] = next_val
        else:
            next_val = proj[i]
    return proj  # type: ignore[return-value]


def _map_child_span(
    local_start: int,
    local_end: int,
    parent_start: int,
    parent_end: int,
    projection: list[int] | None,
) -> dict[str, int] | None:
    """Map a child's ``[local_start, local_end)`` offset within the parent's own
    ``content`` to an absolute ``{start, end}`` span into the document text.

    ``projection`` is ``None`` for the direct-arithmetic fast path (parent
    ``content`` is a verbatim slice of the document, so ``parent_start +``
    is exact); otherwise it is the per-parent map from
    :func:`_parent_to_source_projection`.
    """
    if projection is None:
        abs_start, abs_end = parent_start + local_start, parent_start + local_end
    else:
        abs_start, abs_end = projection[local_start], projection[local_end]
    if abs_start < parent_start or abs_end > parent_end or abs_end <= abs_start:
        return None
    return {"start": abs_start, "end": abs_end}


def enforce_chunk_token_limit_before_embedding(
    chunking_result: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int = 0,
    source_content: str | None = None,
) -> list[dict[str, Any]]:
    """Hard fallback split before embedding while preserving heading hierarchy.

    Splits any chunk still over ``max_tokens`` via the safe
    ``Tokenizer.split_by_token_limit`` contract (see ``lightrag.utils.Tokenizer``),
    carrying ``overlap_tokens`` of context between consecutive pieces. This no
    longer preserves sentence/paragraph boundaries — overlap is what now
    carries the "don't lose context at a cut" job that used to be the
    sentence-first packer's.

    ``source_content`` is the merged document text the chunker itself
    received (the same string :mod:`lightrag.sidecar.backfill` reconstructs
    from ``blocks.jsonl``) — needed to correctly relocate a hard-split
    child's ``_source_span`` when a parent chunk's ``content`` is not a
    byte-verbatim slice of it (see :func:`_parent_to_source_projection`).
    Pass ``None`` only when the caller genuinely has no such text (e.g.
    ``ainsert_custom_chunks``); a parent that would need the projection in
    that case just loses its children's ``_source_span`` instead of raising,
    matching this function's pre-existing behavior for callers with no
    provenance concept at all. When ``source_content`` IS supplied but a
    parent's content has diverged from it beyond whitespace, the projection
    cannot be trusted at all, and this raises ``ChunkBlockMatchError`` rather
    than silently emitting a wrong span — the same failure sidecar backfill
    itself would eventually surface, just earlier and with better context.

    Known limitation -- ``sidecar`` is not re-scoped for pre-sidecar chunks:
    this function runs BEFORE :func:`lightrag.sidecar.backfill.backfill_chunk_sidecars`
    in the pipeline (see ``pipeline.py``'s call order around the "Final hard
    guard before embedding" comment). For the F/R/V strategies that is exactly
    right: at this point they carry only ``_source_span`` (no ``sidecar`` yet),
    this function recomputes a correctly-shrunk ``_source_span`` per child, and
    backfill then derives each child's own ``sidecar`` from that shrunk span
    afterward. But the P (paragraph_semantic) strategy attaches its ``sidecar``
    field during chunking itself, upstream of this function, and does not
    route through backfill at all. If a P chunk is large enough to still need
    a hard split here, every resulting child is a shallow copy of the parent
    (see the loop below) and so inherits the *exact same* ``sidecar`` value
    unmodified -- all children end up pointing at the whole parent's block
    range instead of each getting a sidecar scoped to its own, smaller slice.
    ``_source_span`` itself never reaches storage either way (stripped in
    ``build_chunks_dict_from_chunking_result``); this note is only about the
    accuracy of the ``sidecar`` field that IS persisted to ``text_chunks``.
    Follow-up if this needs fixing: shrink/re-derive ``sidecar`` per child for
    the P case too, analogous to what backfill already does for F/R/V.
    """
    if max_tokens <= 0:
        return list(chunking_result)

    normalized: list[dict[str, Any]] = []

    for dp in chunking_result:
        if not isinstance(dp, dict):
            continue

        content = dp.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue

        # A single call does both the "is this already within budget" check
        # AND the split: split_by_token_limit's own fast path returns one
        # span covering the whole content when it already fits, so there is
        # no separate pre-check encode here to duplicate the full-content
        # encode it does internally regardless. TokenBudgetError (content
        # cannot be safely split at all — not even its first Unicode code
        # point fits max_tokens) is intentionally NOT caught: swallowing it
        # here would mean silently re-emitting the original, still-oversized
        # chunk instead of failing the document — exactly the unsafe
        # fallback this contract exists to remove.
        spans = tokenizer.split_by_token_limit(
            content, max_tokens, overlap_tokens=overlap_tokens
        )

        if len(spans) == 1:
            ndp = dict(dp)
            ndp["tokens"] = spans[0].token_count
            normalized.append(ndp)
            continue

        if overlap_tokens > 0 and any(
            spans[i].start >= spans[i - 1].end for i in range(1, len(spans))
        ):
            logger.warning(
                "Requested embedding_chunk_overlap_token_size=%d could not be "
                "honored for at least one window of chunk %r (retreated to 0 "
                "to preserve forward progress)",
                overlap_tokens,
                dp.get("chunk_id", dp.get("chunk_order_index")),
            )

        base_chunk_id = dp.get("chunk_id")
        parent_span = dp.get("_source_span")

        # Built once per parent, reused for every child below — never redone
        # per child (that would be O(children x parent length) instead of
        # O(parent length)).
        projection: list[int] | None = None
        parent_start = parent_end = None
        if isinstance(parent_span, dict):
            try:
                parent_start = int(parent_span["start"])
                parent_end = int(parent_span["end"])
            except (KeyError, TypeError, ValueError):
                parent_start = parent_end = None
            if parent_start is not None and (
                parent_start < 0 or parent_end < parent_start
            ):
                parent_start = parent_end = None

            if parent_start is not None and parent_end - parent_start != len(content):
                if source_content:
                    projection = _parent_to_source_projection(
                        source_content, parent_start, parent_end, content
                    )
                    if projection is None:
                        raise ChunkBlockMatchError(
                            chunk_order_index=int(dp.get("chunk_order_index", -1)),
                            chunk_preview=content,
                            blocks_path=None,
                        )
                else:
                    # No provenance text available: children below fall back
                    # to dropping _source_span rather than emit a naive,
                    # possibly-wrong offset (parent_start is cleared so
                    # _map_child_span is never reached with an untrustworthy
                    # direct-arithmetic assumption).
                    parent_start = parent_end = None

        total_parts = len(spans)
        for i, span in enumerate(spans, 1):
            piece = content[span.start : span.end]
            new_dp = dict(dp)
            new_dp["content"] = piece
            new_dp["tokens"] = span.token_count

            # Shallow-copy preserves the nested heading dict and sidecar
            # block from the source chunk; only the payload (content/tokens
            # /chunk_id) is rewritten per split slice. For a P-strategy
            # parent that already carries a "sidecar" field, this means
            # every child gets an unmodified copy of the PARENT's sidecar,
            # not one re-scoped to its own smaller slice -- see the "Known
            # limitation" paragraph in this function's docstring.
            if isinstance(base_chunk_id, str) and base_chunk_id.strip():
                new_dp["chunk_id"] = f"{base_chunk_id}-s{i:02d}"

            child_span = None
            if parent_start is not None:
                child_span = _map_child_span(
                    span.start, span.end, parent_start, parent_end, projection
                )
            if child_span is not None:
                new_dp["_source_span"] = child_span
            elif "_source_span" in new_dp:
                new_dp.pop("_source_span", None)

            new_dp["split_type"] = "hard_fallback"
            new_dp["split_part"] = i
            new_dp["split_total"] = total_parts
            normalized.append(new_dp)

    # Rebuild order index to keep continuity after splitting.
    for idx, item in enumerate(normalized):
        item["chunk_order_index"] = idx
    return normalized


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    denom = norm1 * norm2
    # Zero or non-finite vectors are orthogonal to everything in ranking use; avoid NaN/Inf.
    if denom == 0 or not np.isfinite(denom) or not np.isfinite(dot_product):
        return 0.0
    return float(dot_product / denom)


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type="unknown",
) -> tuple[str, int] | None:
    """Generic cache handling function with flattened cache keys

    Returns:
        tuple[str, int] | None: (content, create_time) if cache hit, None if cache miss
    """
    if hashing_kv is None:
        return None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(mode, cache_type, args_hash)
    cache_entry = await hashing_kv.get_by_id(flattened_key)
    if cache_entry:
        logger.debug(f"Flattened cache hit(key:{flattened_key})")
        content = cache_entry["return"]
        timestamp = cache_entry.get("create_time", 0)
        return content, timestamp

    logger.debug(f"Cache missed(mode:{mode} type:{cache_type})")
    return None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    mode: str = "default"
    cache_type: str = "query"
    chunk_id: str | None = None
    queryparam: dict | None = None


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache using flattened key structure.

    Args:
        hashing_kv: The key-value storage for caching
        cache_data: The cache data to save
    """
    # Skip if storage is None or content is a streaming response
    if hashing_kv is None or not cache_data.content:
        return

    # If content is a streaming response, don't cache it
    if hasattr(cache_data.content, "__aiter__"):
        logger.debug("Streaming response detected, skipping cache")
        return

    # Use flattened cache key format: {mode}:{cache_type}:{hash}
    flattened_key = generate_cache_key(
        cache_data.mode, cache_data.cache_type, cache_data.args_hash
    )

    # Check if we already have identical content cached
    existing_cache = await hashing_kv.get_by_id(flattened_key)
    if existing_cache:
        existing_content = existing_cache.get("return")
        if existing_content == cache_data.content:
            logger.warning(
                f"Cache duplication detected for {flattened_key}, skipping update"
            )
            return

    # Create cache entry with flattened structure
    cache_entry = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "chunk_id": cache_data.chunk_id if cache_data.chunk_id is not None else None,
        "original_prompt": cache_data.prompt,
        "queryparam": cache_data.queryparam
        if cache_data.queryparam is not None
        else None,
    }

    logger.info(f" == LLM cache == saving: {flattened_key}")

    # Save using flattened key
    await hashing_kv.upsert({flattened_key: cache_entry})


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def exists_func(obj, func_name: str) -> bool:
    """Check if a function exists in an object or not.
    :param obj:
    :param func_name:
    :return: True / False
    """
    if callable(getattr(obj, func_name, None)):
        return True
    else:
        return False


async def _cooperative_yield(iteration: int, every: int = 64) -> None:
    """Periodically yield control to the event loop during CPU-heavy async loops.

    Call inside long synchronous-style loops to prevent event loop starvation
    in single-worker deployments. Yields every `every` iterations.
    """
    if iteration > 0 and iteration % every == 0:
        await asyncio.sleep(0)


async def wait_tasks_with_drain(
    tasks: list[asyncio.Task],
    *,
    context: str = "",
    task_labels: dict[asyncio.Task, str] | None = None,
) -> list[Any]:
    """Await *tasks*, guaranteeing none is left running when this returns/raises.

    Concurrent multi-store writers (entity/relation merge, rebuild) must never
    leave a sibling task writing in the background after a failure — a failed
    ``gather``/``wait`` does not by itself imply the other write tasks stopped
    (issue #3400, "incomplete async failure coordination").

    Behavior:
      - All tasks succeed: returns their results (completion order).
      - Any task fails: cancels every still-pending sibling, awaits (drains)
        them ALL so no background write survives, logs every additional
        exception, then re-raises the FIRST exception observed.
      - This coroutine itself is cancelled: cancels and drains all tasks
        before propagating ``CancelledError``.

    Args:
        tasks: ``asyncio.Task`` objects (already scheduled).
        context: Optional short label used in log messages.
        task_labels: Optional per-task display labels for pending-task logging.
    """
    if not tasks:
        return []

    ctx = f" [{context}]" if context else ""
    results: list[Any] = []
    first_exception: BaseException | None = None

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        for i, task in enumerate(done, start=1):
            try:
                results.append(task.result())
            except BaseException as e:
                if first_exception is None:
                    first_exception = e
                elif not isinstance(e, asyncio.CancelledError):
                    logger.error(f"Additional task failure{ctx}: {e}")
            # NOTE: an await point — external cancellation delivered here is
            # handled by the enclosing except so still-pending siblings never
            # keep writing in the background.
            await _cooperative_yield(i, every=32)

        if pending:
            if task_labels:
                pending_labels = [
                    task_labels.get(task, "<unknown>") for task in pending
                ]
                preview = ", ".join(pending_labels[:10])
                if len(pending_labels) > 10:
                    preview += f", ... (+{len(pending_labels) - 10} more)"
                logger.warning(f"Draining pending tasks{ctx}: {preview or '<none>'}")
            for task in pending:
                task.cancel()
            pending_results = await asyncio.gather(*pending, return_exceptions=True)
            for result in pending_results:
                if isinstance(result, BaseException):
                    if first_exception is None:
                        first_exception = result
                    elif not isinstance(result, asyncio.CancelledError):
                        logger.error(f"Additional task failure{ctx}: {result}")
                else:
                    results.append(result)
    except asyncio.CancelledError:
        # External cancellation of THIS waiter, delivered at ANY await point
        # above — asyncio.wait itself, a cooperative yield while collecting
        # done results (with siblings still pending after FIRST_EXCEPTION),
        # or the drain gather. Cancel and drain everything before
        # propagating so no task keeps writing in the background. Per-task
        # CancelledError stored in a task's own result is caught by the
        # inner handler and does NOT take this path.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    if first_exception is not None:
        raise first_exception

    return results


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    Reuses the loop running on (or installed on) the current thread so that
    repeated synchronous calls share a single loop; if none exists or it is
    closed, creates a new one and installs it as the current loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    # Reuse a loop actively running on this thread.
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        pass

    # Reuse a loop already installed on this thread, but never let
    # asyncio.get_event_loop() lazily auto-create one — on Python 3.12+ that
    # emits a DeprecationWarning. Promote that warning to an error so the
    # "no current loop" case falls through to explicit creation below, while a
    # genuinely installed (open) loop is still returned and reused.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        try:
            current_loop = asyncio.get_event_loop()
            if not current_loop.is_closed():
                return current_loop
        except (RuntimeError, DeprecationWarning):
            pass

    # No usable loop on this thread — create one and install it.
    logger.info("Creating a new event loop in main thread.")
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    return new_loop


async def aexport_data(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    output_path: str,
    file_format: str = "csv",
    include_vector_data: bool = False,
) -> None:
    """
    Asynchronously exports all entities, relations, and relationships to various formats.

    Args:
        chunk_entity_relation_graph: Graph storage instance for entities and relations
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        output_path: The path to the output file (including extension).
        file_format: Output format - "csv", "excel", "md", "txt".
            - csv: Comma-separated values file
            - excel: Microsoft Excel file with multiple sheets
            - md: Markdown tables
            - txt: Plain text formatted output
        include_vector_data: Whether to include data from the vector database.
    """
    # Collect data
    entities_data = []
    relations_data = []
    relationships_data = []

    # --- Entities ---
    all_entities = await chunk_entity_relation_graph.get_all_labels()
    for entity_name in all_entities:
        # Get entity information from graph
        node_data = await chunk_entity_relation_graph.get_node(entity_name)
        source_id = node_data.get("source_id") if node_data else None

        entity_info = {
            "graph_data": node_data,
            "source_id": source_id,
        }

        # Optional: Get vector database information
        if include_vector_data:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            vector_data = await entities_vdb.get_by_id(entity_id)
            entity_info["vector_data"] = vector_data

        entity_row = {
            "entity_name": entity_name,
            "source_id": source_id,
            "graph_data": str(
                entity_info["graph_data"]
            ),  # Convert to string to ensure compatibility
        }
        if include_vector_data and "vector_data" in entity_info:
            entity_row["vector_data"] = str(entity_info["vector_data"])
        entities_data.append(entity_row)

    # --- Relations ---
    for src_entity in all_entities:
        for tgt_entity in all_entities:
            if src_entity == tgt_entity:
                continue

            edge_exists = await chunk_entity_relation_graph.has_edge(
                src_entity, tgt_entity
            )
            if edge_exists:
                # Get edge information from graph
                edge_data = await chunk_entity_relation_graph.get_edge(
                    src_entity, tgt_entity
                )
                source_id = edge_data.get("source_id") if edge_data else None

                relation_info = {
                    "graph_data": edge_data,
                    "source_id": source_id,
                }

                # Optional: Get vector database information
                if include_vector_data:
                    vector_data = None
                    for rel_id in make_relation_vdb_ids(src_entity, tgt_entity):
                        vector_data = await relationships_vdb.get_by_id(rel_id)
                        if vector_data is not None:
                            break
                    relation_info["vector_data"] = vector_data

                relation_row = {
                    "src_entity": src_entity,
                    "tgt_entity": tgt_entity,
                    "source_id": relation_info["source_id"],
                    "graph_data": str(relation_info["graph_data"]),  # Convert to string
                }
                if include_vector_data and "vector_data" in relation_info:
                    relation_row["vector_data"] = str(relation_info["vector_data"])
                relations_data.append(relation_row)

    # --- Relationships (from VectorDB) ---
    all_relationships = await relationships_vdb.client_storage
    for rel in all_relationships["data"]:
        relationships_data.append(
            {
                "relationship_id": rel["__id__"],
                "data": str(rel),  # Convert to string for compatibility
            }
        )

    # Export based on format
    if file_format == "csv":
        # CSV export
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            # Entities
            if entities_data:
                csvfile.write("# ENTITIES\n")
                writer = csv.DictWriter(csvfile, fieldnames=entities_data[0].keys())
                writer.writeheader()
                writer.writerows(entities_data)
                csvfile.write("\n\n")

            # Relations
            if relations_data:
                csvfile.write("# RELATIONS\n")
                writer = csv.DictWriter(csvfile, fieldnames=relations_data[0].keys())
                writer.writeheader()
                writer.writerows(relations_data)
                csvfile.write("\n\n")

            # Relationships
            if relationships_data:
                csvfile.write("# RELATIONSHIPS\n")
                writer = csv.DictWriter(
                    csvfile, fieldnames=relationships_data[0].keys()
                )
                writer.writeheader()
                writer.writerows(relationships_data)

    elif file_format == "excel":
        # Excel export
        import pandas as pd

        entities_df = pd.DataFrame(entities_data) if entities_data else pd.DataFrame()
        relations_df = (
            pd.DataFrame(relations_data) if relations_data else pd.DataFrame()
        )
        relationships_df = (
            pd.DataFrame(relationships_data) if relationships_data else pd.DataFrame()
        )

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            if not entities_df.empty:
                entities_df.to_excel(writer, sheet_name="Entities", index=False)
            if not relations_df.empty:
                relations_df.to_excel(writer, sheet_name="Relations", index=False)
            if not relationships_df.empty:
                relationships_df.to_excel(
                    writer, sheet_name="Relationships", index=False
                )

    elif file_format == "md":
        # Markdown export
        with open(output_path, "w", encoding="utf-8") as mdfile:
            mdfile.write("# LightRAG Data Export\n\n")

            # Entities
            mdfile.write("## Entities\n\n")
            if entities_data:
                # Write header
                mdfile.write("| " + " | ".join(entities_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(entities_data[0].keys())) + " |\n"
                )

                # Write rows
                for entity in entities_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in entity.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No entity data available*\n\n")

            # Relations
            mdfile.write("## Relations\n\n")
            if relations_data:
                # Write header
                mdfile.write("| " + " | ".join(relations_data[0].keys()) + " |\n")
                mdfile.write(
                    "| " + " | ".join(["---"] * len(relations_data[0].keys())) + " |\n"
                )

                # Write rows
                for relation in relations_data:
                    mdfile.write(
                        "| " + " | ".join(str(v) for v in relation.values()) + " |\n"
                    )
                mdfile.write("\n\n")
            else:
                mdfile.write("*No relation data available*\n\n")

            # Relationships
            mdfile.write("## Relationships\n\n")
            if relationships_data:
                # Write header
                mdfile.write("| " + " | ".join(relationships_data[0].keys()) + " |\n")
                mdfile.write(
                    "| "
                    + " | ".join(["---"] * len(relationships_data[0].keys()))
                    + " |\n"
                )

                # Write rows
                for relationship in relationships_data:
                    mdfile.write(
                        "| "
                        + " | ".join(str(v) for v in relationship.values())
                        + " |\n"
                    )
            else:
                mdfile.write("*No relationship data available*\n\n")

    elif file_format == "txt":
        # Plain text export
        with open(output_path, "w", encoding="utf-8") as txtfile:
            txtfile.write("LIGHTRAG DATA EXPORT\n")
            txtfile.write("=" * 80 + "\n\n")

            # Entities
            txtfile.write("ENTITIES\n")
            txtfile.write("-" * 80 + "\n")
            if entities_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(e[k])) for e in entities_data))
                    for k in entities_data[0]
                }
                header = "  ".join(k.ljust(col_widths[k]) for k in entities_data[0])
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for entity in entities_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in entity.items()
                    )
                    txtfile.write(row + "\n")
                txtfile.write("\n\n")
            else:
                txtfile.write("No entity data available\n\n")

            # Relations
            txtfile.write("RELATIONS\n")
            txtfile.write("-" * 80 + "\n")
            if relations_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(r[k])) for r in relations_data))
                    for k in relations_data[0]
                }
                header = "  ".join(k.ljust(col_widths[k]) for k in relations_data[0])
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for relation in relations_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in relation.items()
                    )
                    txtfile.write(row + "\n")
                txtfile.write("\n\n")
            else:
                txtfile.write("No relation data available\n\n")

            # Relationships
            txtfile.write("RELATIONSHIPS\n")
            txtfile.write("-" * 80 + "\n")
            if relationships_data:
                # Create fixed width columns
                col_widths = {
                    k: max(len(k), max(len(str(r[k])) for r in relationships_data))
                    for k in relationships_data[0]
                }
                header = "  ".join(
                    k.ljust(col_widths[k]) for k in relationships_data[0]
                )
                txtfile.write(header + "\n")
                txtfile.write("-" * len(header) + "\n")

                # Write rows
                for relationship in relationships_data:
                    row = "  ".join(
                        str(v).ljust(col_widths[k]) for k, v in relationship.items()
                    )
                    txtfile.write(row + "\n")
            else:
                txtfile.write("No relationship data available\n\n")

    else:
        raise ValueError(
            f"Unsupported file format: {file_format}. Choose from: csv, excel, md, txt"
        )
    if file_format is not None:
        print(f"Data exported to: {output_path} with format: {file_format}")
    else:
        print("Data displayed as table format")


def export_data(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    output_path: str,
    file_format: str = "csv",
    include_vector_data: bool = False,
) -> None:
    """
    Synchronously exports all entities, relations, and relationships to various formats.

    Args:
        chunk_entity_relation_graph: Graph storage instance for entities and relations
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        output_path: The path to the output file (including extension).
        file_format: Output format - "csv", "excel", "md", "txt".
            - csv: Comma-separated values file
            - excel: Microsoft Excel file with multiple sheets
            - md: Markdown tables
            - txt: Plain text formatted output
        include_vector_data: Whether to include data from the vector database.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        aexport_data(
            chunk_entity_relation_graph,
            entities_vdb,
            relationships_vdb,
            output_path,
            file_format,
            include_vector_data,
        )
    )


def lazy_external_import(module_name: str, class_name: str) -> Callable[..., Any]:
    """Lazily import a class from an external module based on the package of the caller."""
    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args: Any, **kwargs: Any):
        import importlib

        module = importlib.import_module(module_name, package=package)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


async def update_chunk_cache_list(
    chunk_id: str,
    text_chunks_storage: "BaseKVStorage",
    cache_keys: list[str],
    cache_scenario: str = "batch_update",
) -> None:
    """Update chunk's llm_cache_list with the given cache keys

    Args:
        chunk_id: Chunk identifier
        text_chunks_storage: Text chunks storage instance
        cache_keys: List of cache keys to add to the list
        cache_scenario: Description of the cache scenario for logging
    """
    if not cache_keys:
        return

    try:
        chunk_data = await text_chunks_storage.get_by_id(chunk_id)
        if chunk_data:
            # Ensure llm_cache_list exists
            if "llm_cache_list" not in chunk_data:
                chunk_data["llm_cache_list"] = []

            # Add cache keys to the list if not already present
            existing_keys = set(chunk_data["llm_cache_list"])
            new_keys = [key for key in cache_keys if key not in existing_keys]

            if new_keys:
                chunk_data["llm_cache_list"].extend(new_keys)

                # Update the chunk in storage
                await text_chunks_storage.upsert({chunk_id: chunk_data})
                logger.debug(
                    f"Updated chunk {chunk_id} with {len(new_keys)} cache keys ({cache_scenario})"
                )
    except Exception as e:
        logger.warning(
            f"Failed to update chunk {chunk_id} with cache references on {cache_scenario}: {e}"
        )


class TruncatedResponse(str):
    """An LLM response that was cut off by the model's output token limit.

    Behaves exactly like ``str`` so every downstream consumer — tolerant JSON
    parsing, text sanitization, entity/relation extraction — keeps working on
    the partial content unchanged. The only added meaning is a signal to the
    LLM cache layer: a truncated response must NOT be persisted, because a
    cached partial payload (e.g. cut-off extraction JSON) would be replayed on
    every later run, even when a larger token budget would have produced the
    complete output. See ``is_truncated_response`` and the cache-write guards
    in ``use_llm_func_with_cache`` and ``lightrag.operate``.
    """

    __slots__ = ()


def is_truncated_response(value: Any) -> bool:
    """Return True if ``value`` is an LLM response flagged as token-limit truncated.

    Safe for any input: non-string values (e.g. streaming async iterators) and
    plain ``str`` responses from providers that do not emit the marker return
    False, so callers degrade to the pre-existing "cache everything" behavior.
    """
    return isinstance(value, TruncatedResponse)


def format_response_diagnostics(**fields: Any) -> str:
    """Render provider response diagnostics as ``key=value`` pairs.

    ``None`` becomes ``n/a`` so a field the provider did not report is visibly
    absent rather than looking like a zero.
    """
    return ", ".join(
        f"{key}={'n/a' if value is None else value}" for key, value in fields.items()
    )


def empty_length_truncated_hint(
    budget_hint: str, *, reasoning_consumed_budget: bool = False
) -> str:
    """Explain an EMPTY response whose finish reason is the output token limit.

    Shared by every binding so the four providers describe the same failure
    identically. This is the structurally-broken case, not "ran a bit long":
    generation stopped before producing a single content token, so there is
    nothing to salvage and nothing to cache — the caller raises rather than
    returning "" and letting the document be indexed as an empty graph
    (issue #3601 gap 4).

    ``budget_hint`` names the provider's own output-budget knob, since that is
    the actionable part and only the binding knows it.
    """
    cause = "generation hit the token limit before emitting any content"
    if reasoning_consumed_budget:
        cause += " (budget consumed by reasoning)"
    return f"{cause}; {budget_hint}"


# doc_status.metadata key holding the per-document truncation summary.
LLM_TRUNCATION_METADATA_KEY = "llm_truncation"

# Cap on the number of affected subjects (chunk ids / entity names) echoed into
# that metadata entry. The doc_status row is serialized into every documents
# listing response, so the summary must stay O(1) in document size; the exact
# per-chunk detail lives in the server log, which is unbounded by design.
TRUNCATION_METADATA_SAMPLE_LIMIT = 10


class TokenLimitTruncationTally:
    """Accumulator for token-limit truncation events over one scope.

    A document whose output budget is too small does not truncate once, it
    truncates on EVERY chunk: publishing one ``pipeline_status`` line per event
    would push the rest of the run out of the bounded ``history_messages`` ring
    and still leave the operator counting lines. So each producer stage keeps a
    tally, publishes its FIRST event immediately (the condition must not stay
    hidden until a long run ends) plus ONE aggregated line when it finishes,
    and folds its counts into the document-scoped tally the pipeline stamps
    into ``doc_status.metadata`` — the durable, machine-readable record that
    outlives the status ring and reaches the API.

    Not thread-safe, and does not need to be: every mutation is a plain,
    await-free update made from tasks on a single event loop.
    """

    __slots__ = ("_events", "_stages", "_subjects")

    def __init__(self) -> None:
        self._events = 0
        self._stages: dict[str, int] = {}
        # Ordered set: preserves first-seen order for the metadata sample while
        # de-duplicating a subject that truncated at more than one stage.
        self._subjects: dict[str, None] = {}

    def __bool__(self) -> bool:
        return self._events > 0

    @property
    def events(self) -> int:
        """Total truncated responses, counting a subject once per stage."""
        return self._events

    @property
    def affected(self) -> int:
        """Distinct subjects (chunk ids / entity names) with >= 1 truncation."""
        return len(self._subjects)

    def record(self, stage: str, subject: str) -> bool:
        """Record one truncated response; True when it is this tally's first."""
        first = self._events == 0
        self._events += 1
        self._stages[stage] = self._stages.get(stage, 0) + 1
        if subject:
            self._subjects.setdefault(subject, None)
        return first

    def absorb(self, other: "TokenLimitTruncationTally | None") -> None:
        """Fold a stage-scoped tally into this (document-scoped) one."""
        if not other:
            return
        self._events += other._events
        for stage, count in other._stages.items():
            self._stages[stage] = self._stages.get(stage, 0) + count
        for subject in other._subjects:
            self._subjects.setdefault(subject, None)

    def stage_breakdown(self) -> str:
        """Human-readable per-stage counts, e.g. ``initial: 12, gleaning: 3``."""
        return ", ".join(f"{stage}: {count}" for stage, count in self._stages.items())

    def as_metadata(self) -> dict[str, Any] | None:
        """Summary payload for ``doc_status.metadata``; None when nothing hit."""
        if not self._events:
            return None
        subjects = list(self._subjects)
        payload: dict[str, Any] = {
            "events": self._events,
            "affected": len(subjects),
            "stages": dict(self._stages),
            "samples": subjects[:TRUNCATION_METADATA_SAMPLE_LIMIT],
        }
        omitted = len(subjects) - TRUNCATION_METADATA_SAMPLE_LIMIT
        if omitted > 0:
            payload["samples_omitted"] = omitted
        return payload

    def as_metadata_extra(self) -> dict[str, Any]:
        """``metadata_extra`` fragment for a doc_status transition upsert.

        Empty on a clean run, so the key is simply absent rather than persisting
        a zeroed record. Because ``LLM_TRUNCATION_METADATA_KEY`` is deliberately
        NOT in ``_DOC_STATUS_METADATA_CARRY_OVER_KEYS``, that absence also
        CLEARS a previous attempt's summary instead of resurrecting it — a
        re-run with a larger token budget must not keep reporting the old
        truncation.
        """
        payload = self.as_metadata()
        return {LLM_TRUNCATION_METADATA_KEY: payload} if payload else {}


def merge_truncation_metadata(
    base: dict[str, Any] | None, extra: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Combine two persisted ``llm_truncation`` payloads into one.

    Exists for paths that ADD to a surviving record instead of replacing it:
    the custom-chunk patch (its truncations must not overwrite the base run's
    record, nor a truncated base be blanked by a clean patch), the same
    operation's resumed attempts (a failed attempt's partial graph writes stay,
    so its record must survive a clean resume), and the analyze→process
    hand-off. The pipeline's whole-document reprocess never needs this —
    there replace-or-clear is the correct semantics.

    Merging live tallies would be exact, but only the persisted payloads
    survive (the base run's tally object is long gone), and those are lossy:
    ``samples`` is capped at :data:`TRUNCATION_METADATA_SAMPLE_LIMIT`, so
    subjects beyond the cap cannot be de-duplicated. ``events`` and ``stages``
    are exact sums regardless. ``affected`` is exact whenever both sample
    lists are complete (no ``samples_omitted``); otherwise it deduplicates
    what the samples do show and over-counts a subject that truncated on both
    sides but is visible in neither. Base-vs-patch merges rarely collide
    (different chunk id schemes; only repeat ``summary`` subjects overlap),
    but attempt-over-attempt merges re-run the SAME chunk ids, so collision is
    the normal case there — still exact up to the sample cap, over-counted
    past it.
    """
    if not base:
        return dict(extra) if extra else None
    if not extra:
        return dict(base)

    base_samples = [s for s in (base.get("samples") or []) if isinstance(s, str)]
    extra_samples = [s for s in (extra.get("samples") or []) if isinstance(s, str)]
    # Ordered union, base first — mirrors the tally's first-seen ordering.
    union_samples = list(dict.fromkeys(base_samples + extra_samples))

    both_complete = not base.get("samples_omitted") and not extra.get("samples_omitted")
    if both_complete:
        affected = len(union_samples)
    else:
        overlap = len(set(base_samples) & set(extra_samples))
        affected = int(base.get("affected") or 0) + int(extra.get("affected") or 0)
        affected -= overlap

    stages: dict[str, int] = dict(base.get("stages") or {})
    for stage, count in (extra.get("stages") or {}).items():
        stages[stage] = stages.get(stage, 0) + int(count)

    merged: dict[str, Any] = {
        "events": int(base.get("events") or 0) + int(extra.get("events") or 0),
        "affected": affected,
        "stages": stages,
        "samples": union_samples[:TRUNCATION_METADATA_SAMPLE_LIMIT],
    }
    omitted = affected - len(merged["samples"])
    if omitted > 0:
        merged["samples_omitted"] = omitted
    return merged


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their content from the text.

    Preserves the :class:`TruncatedResponse` marker so downstream consumers
    can still distinguish partial model output after sanitization.

    Handles two cases:
    1. Complete <think>...</think> blocks anywhere in the text.
    2. Orphaned </think> at the very start (e.g., from streaming that begins
       mid-think-block), removing everything before and including it.
    """
    was_truncated = is_truncated_response(text)

    # First, remove orphaned </think> prefix (content before first </think>
    # when there is no preceding <think> tag)
    text = re.sub(r"^((?!<think>).)*?</think>", "", text, flags=re.DOTALL)
    # Then remove all complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = text.strip()
    return TruncatedResponse(cleaned) if was_truncated else cleaned


def _reject_empty_truncated_response(
    cleaned: str, *, raw_len: int, cache_type: str, chunk_id: str | None
) -> None:
    """Fail a truncated response that think-tag removal left with nothing.

    The provider bindings each escalate their own empty + token-limit response,
    but they cannot see this one: ``<think>reasoning</think>`` with no answer
    after it is a NON-empty payload, so it passes every binding's check and
    only becomes visibly empty here. This function is the shared throat that
    every KG-building LLM call passes through, so the rule lands once for all
    four providers.

    Parse-stage callers (``cache_type="smartheading"``) already treat an empty
    answer as an error — ``_parse_llm_json`` raises ``TitleBlockLLMError`` on
    it — so this only replaces "answer carries no JSON object" with the actual
    cause and its remedy.
    """
    if not is_truncated_response(cleaned) or cleaned.strip():
        return
    diagnostics = format_response_diagnostics(
        chunk_id=chunk_id,
        # Everything the model produced was reasoning, by construction: the
        # payload was non-empty before think-tag removal and empty after.
        reasoning_content_len=raw_len,
    )
    hint = empty_length_truncated_hint(
        "consider raising the LLM's output token limit or disabling thinking "
        "mode for this role",
        reasoning_consumed_budget=True,
    )
    message = (
        f"Received empty {cache_type} content after think-tag removal "
        f"({diagnostics}): {hint}"
    )
    logger.error(message)
    raise EmptyTruncatedResponseError(message)


async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    system_prompt: str | None = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
    response_format: Any | None = None,
    entity_extraction: bool = False,
    llm_cache_identity: Any | None = None,
) -> tuple[str, int]:
    """Call LLM function with cache support and text sanitization

    If cache is available and enabled (determined by handle_cache based on mode),
    retrieve result from cache; otherwise call LLM function and save result to cache.

    This function applies text sanitization to prevent UTF-8 encoding errors for all LLM providers.

    Args:
        input_text: Input text to send to LLM
        use_llm_func: LLM function with higher priority
        llm_response_cache: Cache storage instance
        max_tokens: Maximum tokens for generation
        history_messages: History messages list
        cache_type: Type of cache
        chunk_id: Chunk identifier to store in cache
        text_chunks_storage: Text chunks storage to update llm_cache_list
        cache_keys_collector: Optional list to collect cache keys for batch processing
        response_format: Structured output control forwarded to the LLM provider.
            Providers translate this to their native structured-output surface
            (OpenAI response_format, Ollama format, Gemini response_mime_type/schema).
            ``{"type": "json_object"}`` requests JSON output; typed/schema payloads
            trigger schema-constrained output where supported; ``None`` leaves
            output unconstrained. Providers that do not support structured output
            safely strip this argument.
        entity_extraction: Deprecated. When True and ``response_format`` is not
            provided, maps to ``{"type": "json_object"}``. Prefer passing
            ``response_format`` directly.
        llm_cache_identity: Non-secret model/provider identity used to partition
            cache entries across role model, binding, or host changes.

    Returns:
        tuple[str, int]: (LLM response text, timestamp)
            - For cache hits: (content, cache_create_time)
            - For cache misses: (content, current_timestamp)
    """
    if entity_extraction and response_format is None:
        warnings.warn(
            "use_llm_func_with_cache(entity_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        response_format = {"type": "json_object"}
    _validate_cached_response_format(response_format)
    # Sanitize input text to prevent UTF-8 encoding errors for all LLM providers
    safe_user_prompt = sanitize_text_for_encoding(user_prompt)
    safe_system_prompt = (
        sanitize_text_for_encoding(system_prompt) if system_prompt else None
    )

    # Sanitize history messages if provided
    safe_history_messages = None
    if history_messages:
        safe_history_messages = []
        for i, msg in enumerate(history_messages):
            safe_msg = msg.copy()
            if "content" in safe_msg:
                safe_msg["content"] = sanitize_text_for_encoding(safe_msg["content"])
            safe_history_messages.append(safe_msg)
        history = json.dumps(safe_history_messages, ensure_ascii=False)
    else:
        history = None

    if llm_response_cache:
        prompt_parts = []
        if safe_user_prompt:
            prompt_parts.append(safe_user_prompt)
        if safe_system_prompt:
            prompt_parts.append(safe_system_prompt)
        if history:
            prompt_parts.append(history)
        _prompt = "\n".join(prompt_parts)

        response_format_key = _serialize_cache_variant(response_format)
        llm_identity_key = serialize_llm_cache_identity(llm_cache_identity)
        arg_hash = compute_args_hash(
            _prompt,
            "\n<response_format>\n",
            response_format_key,
            "\n<llm_identity>\n",
            llm_identity_key,
        )
        # Generate cache key for this LLM call
        cache_key = generate_cache_key("default", cache_type, arg_hash)

        cached_result = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
        )
        if cached_result:
            content, timestamp = cached_result
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

            return content, timestamp
        statistic_data["llm_call"] += 1

        # Call LLM with sanitized input
        kwargs = {}
        if safe_history_messages:
            kwargs["history_messages"] = safe_history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        res: str = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )

        # ``remove_think_tags`` re-wraps its result, so the marker survives
        # sanitization and both this cache guard and the caller (which reports
        # the truncation to pipeline status) read the same flag off ``res``.
        raw_len = len(res)
        res = remove_think_tags(res)
        _reject_empty_truncated_response(
            res, raw_len=raw_len, cache_type=cache_type, chunk_id=chunk_id
        )
        res_truncated = is_truncated_response(res)

        # Generate timestamp for cache miss (LLM call completion time)
        current_timestamp = int(time.time())

        if llm_response_cache.global_config.get("enable_llm_cache_for_entity_extract"):
            if res_truncated:
                # Do not persist truncated extraction output: a cached partial
                # payload would be replayed on every later run, even when a
                # larger token budget would have completed the extraction.
                logger.warning(
                    f"Skipping LLM cache write for truncated {cache_type} response "
                    f"(finish_reason=length, chunk_id={chunk_id})"
                )
            else:
                await save_to_cache(
                    llm_response_cache,
                    CacheData(
                        args_hash=arg_hash,
                        content=res,
                        prompt=_prompt,
                        cache_type=cache_type,
                        chunk_id=chunk_id,
                    ),
                )

                # Add cache key to collector if provided
                if cache_keys_collector is not None:
                    cache_keys_collector.append(cache_key)

        return res, current_timestamp

    # When cache is disabled, directly call LLM with sanitized input
    kwargs = {}
    if safe_history_messages:
        kwargs["history_messages"] = safe_history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        res = await use_llm_func(
            safe_user_prompt, system_prompt=safe_system_prompt, **kwargs
        )
    except Exception as e:
        # Add [LLM func] prefix to error message
        error_msg = f"[LLM func] {str(e)}"
        # Re-raise with the same exception type but modified message
        raise type(e)(error_msg) from e

    # Generate timestamp for non-cached LLM call
    current_timestamp = int(time.time())
    raw_len = len(res)
    cleaned = remove_think_tags(res)
    _reject_empty_truncated_response(
        cleaned, raw_len=raw_len, cache_type=cache_type, chunk_id=chunk_id
    )
    return cleaned, current_timestamp


def get_content_summary(content: str, max_length: int = 250) -> str:
    """Get summary of document content

    Args:
        content: Original document content
        max_length: Maximum length of summary

    Returns:
        Truncated content with ellipsis if needed
    """
    content = content.strip()
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


def sanitize_and_normalize_extracted_text(
    input_text: str, remove_inner_quotes=False
) -> str:
    """Santitize and normalize extracted text
    Args:
        input_text: text string to be processed
        is_name: whether the input text is a entity or relation name

    Returns:
        Santitized and normalized text string
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(
            safe_input_text, remove_inner_quotes=remove_inner_quotes
        )
        return normalized_text
    return ""


def normalize_entity_name(input_text: str) -> str:
    """Normalize an entity identifier using the extraction naming contract."""
    return sanitize_and_normalize_extracted_text(input_text, remove_inner_quotes=True)


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    - Clean HTML tags (paragraph and line break tags)
    - Convert Chinese symbols to English symbols
    - Remove spaces between Chinese characters
    - Remove spaces between Chinese characters and English letters/numbers
    - Preserve spaces within English text and numbers
    - Replace Chinese parentheses with English parentheses
    - Replace Chinese dash with English dash
    - Remove English quotation marks from the beginning and end of the text
    - Remove English quotation marks in and around chinese
    - Remove Chinese quotation marks
    - Filter out short numeric-only text (length < 3 and only digits/dots)
    - remove_inner_quotes = True
        remove Chinese quotes
        remove English quotes in and around chinese
        Convert non-breaking spaces to regular spaces
        Convert narrow non-breaking spaces after non-digits to regular spaces

    Args:
        name: Entity name to normalize
        is_entity: Whether this is an entity name (affects quote handling)

    Returns:
        Normalized entity name
    """
    # Clean HTML tags - remove paragraph and line break tags
    name = re.sub(r"</p\s*>|<p\s*>|<p/>", "", name, flags=re.IGNORECASE)
    name = re.sub(r"</br\s*>|<br\s*>|<br/>", "", name, flags=re.IGNORECASE)

    # Chinese full-width letters to half-width (A-Z, a-z)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # Chinese full-width numbers to half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols to half-width
    name = name.replace("－", "-")  # Chinese minus
    name = name.replace("＋", "+")  # Chinese plus
    name = name.replace("／", "/")  # Chinese slash
    name = name.replace("＊", "*")  # Chinese asterisk

    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash (additional patterns)
    name = name.replace("—", "-").replace("－", "-")

    # Chinese full-width space to regular space (after other replacements)
    name = name.replace("　", " ")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # Remove outer quotes
    if len(name) >= 2:
        # Handle double quotes
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:  # No double quotes inside
                name = inner_content

        # Handle single quotes
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:  # No single quotes inside
                name = inner_content

        # Handle Chinese-style double quotes
        if name.startswith("“") and name.endswith("”"):
            inner_content = name[1:-1]
            if "“" not in inner_content and "”" not in inner_content:
                name = inner_content
        if name.startswith("‘") and name.endswith("’"):
            inner_content = name[1:-1]
            if "‘" not in inner_content and "’" not in inner_content:
                name = inner_content

        # Handle Chinese-style book title mark
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # Remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # Remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        # Convert non-breaking space to regular space
        name = name.replace("\u00a0", " ")
        # Convert narrow non-breaking space to regular space when after non-digits
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    # Remove spaces from the beginning and end of the text
    name = name.strip()

    # Filter out pure numeric content with length < 3
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text):
        """
        Check if the string consists only of dots and digits, with at least one dot
        Filter cases include: 1.2.3, 12.3, .123, 123., 12.3., .1.23 etc.
        """
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        # Filter out mixed numeric and dot content with length < 6, requiring at least one dot
        return ""

    return name


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding by removing or replacing problematic characters.

    This function handles:
    - Surrogate characters (the main cause of encoding errors)
    - Other invalid Unicode sequences
    - Control characters that might cause issues
    - Unescape HTML escapes
    - Remove control characters
    - Whitespace trimming

    Args:
        text: Input text to sanitize
        replacement_char: Character to use for replacing invalid sequences

    Returns:
        Sanitized text that can be safely encoded as UTF-8
    """
    if not text:
        return text

    # First, strip whitespace
    text = text.strip()

    # Early return if text is empty after basic cleaning
    if not text:
        return text

    # 1. html.unescape first to catch entities that might become surrogates or control chars
    text = html.unescape(text)

    # 2. Use pre-compiled regex to clean surrogates and non-characters in one pass
    # This replaces the slow manual loop and initial .encode() check
    text = _SURROGATE_PATTERN.sub(replacement_char, text)

    # 3. Remove control characters but preserve common whitespace (\t, \n, \r)
    text = _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)

    return text.strip()


def strip_control_characters(text: str, replacement_char: str = "") -> str:
    """Remove control/separator chars and surrogates while preserving text.

    Strips the same character classes as :func:`sanitize_text_for_encoding`
    (surrogates via ``_SURROGATE_PATTERN`` and control chars via
    ``_CONTROL_CHAR_PATTERN_ALL`` — including the C0 separators ``\\x1c``-``\\x1f``
    FS/GS/RS/US, while keeping ``\\t``/``\\n``/``\\r``) but deliberately does
    *not* ``html.unescape`` or ``.strip()`` the result.

    This makes it safe for text that carries intentional markup (e.g. sidecar
    block content with ``<table>``/``<drawing>``/``<equation>`` tags, where
    unescaping ``&lt;`` would corrupt the markup) or significant leading/
    trailing whitespace. For control-char-free input it returns the string
    unchanged, so it does not perturb existing content hashes or snapshots.
    """
    if not text:
        return text
    text = _SURROGATE_PATTERN.sub(replacement_char, text)
    return _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, text)


# LLMs emitting LaTeX inside JSON strings routinely under-escape backslashes:
# "\frac" is *valid* JSON meaning form feed + "rac", so JSON parsers
# (including json_repair) silently decode it and the LaTeX command is
# destroyed. Form feed (\x0c) and backspace (\x08) followed by a letter have
# no legitimate use in LLM-generated prose, so restoring the backslash is
# unconditionally safe. The other three decodable escapes (\t, \n, \r) map to
# legitimate whitespace and cannot be restored without guessing; they are only
# *detected* (see _WS_LATEX_SUSPECT_PATTERN) so real-world frequency can be
# observed before deciding on heuristic restoration.
_FORMFEED_LATEX_PATTERN = re.compile(r"\x0c(?=[A-Za-z])")
_BACKSPACE_LATEX_PATTERN = re.compile(r"\x08(?=[A-Za-z])")
# Whitespace + residue spelling that completes a common LaTeX command whose
# remainder collides with no English word ("eq"/"o"/"exists" are deliberately
# absent: "eq." abbreviations, the word "o"/"exists" would false-positive).
_WS_LATEX_SUSPECT_PATTERN = re.compile(
    r"\t(?=(?:au|heta|imes|ext|ilde|herefore|riangle)\b)"
    r"|\r(?=(?:ho|ight|angle|ceil)\b)"
    r"|\n(?=(?:abla|otin)\b)"
)


def repair_vlm_json_escape_damage(text: str, *, context: str = "") -> str:
    """Restore LaTeX backslashes destroyed by JSON escape decoding.

    Applied to string values parsed out of VLM/LLM JSON responses, where an
    un-doubled LaTeX command like ``"\\frac"`` arrives as ``\\x0c`` + ``rac``.
    Only the two zero-risk cases are repaired:

    - form feed + letter  -> ``\\f`` + letter (``\\frac``, ``\\forall``, ...)
    - backspace + letter  -> ``\\b`` + letter (``\\beta``, ``\\bar``, ...)

    Isolated control characters (not followed by a letter) are left alone for
    downstream sanitization to drop. Whitespace-class damage (``\\tau`` ->
    tab + ``au`` etc.) is ambiguous with legitimate whitespace and is only
    logged at WARNING level, never rewritten.

    Args:
        text: Parsed string value to repair.
        context: Optional label (e.g. ``"table/t1.description"``) included in
            the detection log line.
    """
    if not text:
        return text

    repaired = _FORMFEED_LATEX_PATTERN.sub(r"\\f", text)
    repaired = _BACKSPACE_LATEX_PATTERN.sub(r"\\b", repaired)
    if repaired != text:
        logger.warning(
            "Repaired LaTeX escape damage (\\f/\\b decoded by JSON parser)%s",
            f" in {context}" if context else "",
        )

    suspect = _WS_LATEX_SUSPECT_PATTERN.search(repaired)
    if suspect:
        snippet = repaired[max(0, suspect.start() - 30) : suspect.start() + 30]
        logger.warning(
            "Suspected whitespace-class LaTeX escape damage%s (not auto-repaired): %r",
            f" in {context}" if context else "",
            snippet,
        )

    return repaired


def repair_vlm_json_escape_damage_nested(obj: Any, *, context: str = "") -> Any:
    """Apply :func:`repair_vlm_json_escape_damage` to every string inside a
    parsed JSON structure (dicts / lists nested arbitrarily).

    Used on the output of ``json_repair.loads`` for LLM responses that may
    quote LaTeX — multimodal analysis objects and entity-extraction results
    (``{"entities": [{...}], "relationships": [{...}]}``). Non-string leaves
    are returned untouched.
    """
    if isinstance(obj, str):
        return repair_vlm_json_escape_damage(obj, context=context)
    if isinstance(obj, dict):
        return {
            key: repair_vlm_json_escape_damage_nested(
                value, context=f"{context}.{key}" if context else str(key)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            repair_vlm_json_escape_damage_nested(item, context=context) for item in obj
        ]
    return obj


_CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL
)


def strip_markdown_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence (```json ... ``` or ``` ... ```).

    Why: LLM training priors strongly associate "JSON output" with fenced code
    blocks, so providers routinely wrap responses despite explicit instructions
    to the contrary. Stripping here avoids relying on ``json_repair`` and the
    noisy warning it emits. The ``\\n?`` make the interior newlines optional so
    single-line fences (```` ```json {...}``` ````) are handled too.
    """

    match = _CODE_FENCE_PATTERN.match(text)
    return match.group(1) if match else text


def _first_structural_opener(text: str) -> tuple[str | None, int]:
    """Return the first ``[`` or ``{`` that sits outside a double-quoted string.

    Only ``"`` is treated as a string delimiter: prose apostrophes (``Here's``)
    are far too common to treat ``'`` as a quote, and leading prose that wraps a
    ``[`` inside single quotes is rare enough to accept as graceful degradation.
    The result drives the top-level array-vs-object decision — a leading ``[``
    means a top-level array (rejected by the caller), a leading ``{`` marks where
    object recovery starts.
    """
    quote_open = False
    escaped = False
    for index, char in enumerate(text):
        if quote_open:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote_open = False
            continue
        if char == '"':
            quote_open = True
        elif char in "[{":
            return char, index
    return None, -1


def _first_balanced_object_slice(text: str) -> str:
    """Return the first brace-balanced ``{...}`` slice; ``text`` starts at ``{``.

    ``"`` always opens a string. ``'`` opens a string only in a value/key
    position — right after ``{``, ``[``, ``,`` or ``:`` (spaces skipped) — so a
    bare apostrophe inside an unquoted token (``O'Reilly``, ``it's``) is not
    mistaken for a string start. Without that guard the scan would run past the
    object's closing ``}`` and json_repair would fold trailing prose into the
    last value — a silent, schema-valid but corrupted result. Single quotes are
    tracked at all because weaker models emit single-quoted JSON strings that can
    legitimately contain stray braces. Falls back to the whole string when no
    matching close brace exists so an incomplete object stays repairable.
    """
    depth = 0
    quote: str | None = None
    escaped = False
    previous_non_space = ""
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char == '"' or (char == "'" and previous_non_space in "{[,:"):
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[: index + 1]
        if not char.isspace():
            previous_non_space = char
    return text


def tolerant_load_json_dict(text: str) -> dict[str, Any]:
    """Recover a single JSON object from noisy LLM/VLM text.

    Returns the first genuine JSON *object*, or ``{}`` when the text does not
    yield exactly one object. ``{}`` is the signal callers act on: multimodal
    analysis retries once, entity extraction falls back to delimiter parsing.

    Recovered — returns the object (the malformed shapes weak models emit):

    * a clean object, optionally inside a markdown ``json`` code fence, with or
      without interior newlines: ``{"a": 1}``
    * leading prose before the object — a label, a ``#`` or ``//`` comment
      lead-in, a URL, or an apostrophe: ``Here is the result: {"a":1}``,
      ``Result #1: {"a":1}``, ``Source: http://x//y#z {"a":1}``,
      ``Here's the note: {"a":1}``
    * trailing prose after the object, even when it contains braces — the bug
      this helper exists for, where a greedy ``{...}`` slice over-extends and
      drops everything: ``{"facts":[...]} trailing {brace}``
    * object-level slips ``json_repair`` fixes — trailing comma, single quotes,
      unquoted keys, truncation: ``{"a":1,}``, ``{'a':1}``, ``{a:1}``, ``{"a":1``
    * a genuine (possibly malformed) object followed by a bracket citation:
      ``{"a":1,} [1]``

    Rejected — returns ``{}`` so callers retry / fall back rather than accept a
    non-object:

    * any top-level array, so one element is never mistaken for the whole
      answer — including prose-prefixed, truncated, single-quoted, and commented
      arrays: ``[{"a":1},{"b":2}]``, ``Here is the result: [{...}]``,
      ``['note', {...}]``, ``[/* ] */ {...}]``
    * a top-level array reached past a broken array shell, an unclosed quote, or
      pseudo-object prose — never scavenged for an inner object: ``[}{"a":1}]``,
      ``"oops [}{"a":1}]``, ``{note} [{"a":1}]``
    * an object behind bracketed prose — indistinguishable from a real array
      without heuristics and not seen in practice: ``analysis: [draft] {"a":1}``

    Two edge rules worth knowing when extending this:

    * the top-level shape is decided by the first structural opener outside a
      double-quoted string; ``json_repair`` runs only on the first balanced
      ``{...}`` slice, never the whole string (it would scavenge a dict across
      structural boundaries and defeat array rejection).
    * a single quote is a string delimiter only in a value/key position, so a
      bare apostrophe in an unquoted token (``O'Reilly``, ``it's``) does not
      swallow the object's closing brace.

    Not handled here: LaTeX-escape damage in string values — callers apply
    ``repair_vlm_json_escape_damage_nested`` themselves, keeping that choke
    point out of this helper.
    """
    if not text:
        return {}
    candidate = strip_markdown_code_fence(text).strip()

    # Decide the top-level shape from the raw structure FIRST: the first opener
    # outside a double-quoted string. json_repair is never run over the whole
    # candidate — it scavenges a dict across structural boundaries (out of a
    # broken array shell '[}{...}]', or past pseudo-object prose '{note} [..]'),
    # which would bypass the top-level-array rejection contract.
    opener, index = _first_structural_opener(candidate)
    if opener != "{":
        # '[' is a top-level array. None means the structure is untrusted — e.g.
        # an unclosed double quote swallows the openers ('"oops [}{"a":1}]').
        # Reject either way so callers retry (multimodal) / fall back (entity).
        return {}
    suffix = candidate[index:]

    # 1) Strict decode of the first object from the opener. A clean object,
    #    optionally followed by trailing prose, is the answer — raw_decode stops
    #    at the end of the first value, so "{...} trailing {brace}" is handled.
    try:
        obj, _end = json.JSONDecoder().raw_decode(suffix)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) The opener did not start a clean object: a malformed leading object
    #    (trailing comma / single quotes / unquoted keys), possibly followed by
    #    trailing text such as a "[1]" citation, must still be recovered. Repair
    #    the first balanced slice and accept it only if it is itself an object.
    #    json_repair returns a *list* (not a dict) for pseudo-object prose like
    #    '{note}', so a real payload of the form '{note} [ {...} ]' falls through
    #    to {} here — a trailing top-level array is never scavenged for an
    #    element (repairing only the slice, never the whole candidate, is what
    #    keeps the array out of reach).
    slice_ = _first_balanced_object_slice(suffix)
    try:
        repaired = json_repair.loads(slice_)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    return {}


def check_storage_env_vars(storage_name: str) -> None:
    """Check if all required environment variables for storage implementation exist

    Args:
        storage_name: Storage implementation name

    Raises:
        ValueError: If required environment variables are missing
    """
    from lightrag.kg import STORAGE_ENV_REQUIREMENTS

    required_vars = STORAGE_ENV_REQUIREMENTS.get(storage_name, [])
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        raise ValueError(
            f"Storage implementation '{storage_name}' requires the following "
            f"environment variables: {', '.join(missing_vars)}"
        )


def pick_by_weighted_polling(
    entities_or_relations: list[dict],
    max_related_chunks: int,
    min_related_chunks: int = 1,
) -> list[str]:
    """
    Linear gradient weighted polling algorithm for text chunk selection.

    This algorithm ensures that entities/relations with higher importance get more text chunks,
    forming a linear decreasing allocation pattern.

    Args:
        entities_or_relations: List of entities or relations sorted by importance (high to low)
        max_related_chunks: Expected number of text chunks for the highest importance entity/relation
        min_related_chunks: Expected number of text chunks for the lowest importance entity/relation

    Returns:
        List of selected text chunk IDs
    """
    if not entities_or_relations:
        return []
    if max_related_chunks <= 0:
        return []

    n = len(entities_or_relations)
    if n == 1:
        # Only one entity/relation, return its first max_related_chunks text chunks
        entity_chunks = entities_or_relations[0].get("sorted_chunks", [])
        return entity_chunks[:max_related_chunks]

    # Calculate expected text chunk count for each position (linear decrease)
    expected_counts = []
    for i in range(n):
        # Linear interpolation: from max_related_chunks to min_related_chunks
        ratio = i / (n - 1) if n > 1 else 0
        expected = max_related_chunks - ratio * (
            max_related_chunks - min_related_chunks
        )
        expected_counts.append(int(round(expected)))

    # First round allocation: allocate by expected values
    selected_chunks = []
    used_counts = []  # Track number of chunks used by each entity
    total_remaining = 0  # Accumulate remaining quotas

    for i, entity_rel in enumerate(entities_or_relations):
        entity_chunks = entity_rel.get("sorted_chunks", [])
        expected = expected_counts[i]

        # Actual allocatable count
        actual = min(expected, len(entity_chunks))
        selected_chunks.extend(entity_chunks[:actual])
        used_counts.append(actual)

        # Accumulate remaining quota
        remaining = expected - actual
        if remaining > 0:
            total_remaining += remaining

    # Second round allocation: multi-round scanning to allocate remaining quotas
    for _ in range(total_remaining):
        allocated = False

        # Scan entities one by one, allocate one chunk when finding unused chunks
        for i, entity_rel in enumerate(entities_or_relations):
            entity_chunks = entity_rel.get("sorted_chunks", [])

            # Check if there are still unused chunks
            if used_counts[i] < len(entity_chunks):
                # Allocate one chunk
                selected_chunks.append(entity_chunks[used_counts[i]])
                used_counts[i] += 1
                allocated = True
                break

        # If no chunks were allocated in this round, all entities are exhausted
        if not allocated:
            break

    return selected_chunks


async def pick_by_vector_similarity(
    query: str,
    text_chunks_storage: "BaseKVStorage",
    chunks_vdb: "BaseVectorStorage",
    num_of_chunks: int,
    entity_info: list[dict[str, Any]],
    embedding_func: callable,
    query_embedding=None,
) -> list[str]:
    """
    Vector similarity-based text chunk selection algorithm.

    This algorithm selects text chunks based on cosine similarity between
    the query embedding and text chunk embeddings.

    Args:
        query: User's original query string
        text_chunks_storage: Text chunks storage instance
        chunks_vdb: Vector database storage for chunks
        num_of_chunks: Number of chunks to select
        entity_info: List of entity information containing chunk IDs
        embedding_func: Embedding function to compute query embedding

    Returns:
        List of selected text chunk IDs sorted by similarity (highest first)
    """
    logger.debug(
        f"Vector similarity chunk selection: num_of_chunks={num_of_chunks}, entity_info_count={len(entity_info) if entity_info else 0}"
    )

    if not entity_info or num_of_chunks <= 0:
        return []

    # Collect all unique chunk IDs from entity info
    all_chunk_ids = set()
    for i, entity in enumerate(entity_info):
        chunk_ids = entity.get("sorted_chunks", [])
        all_chunk_ids.update(chunk_ids)

    if not all_chunk_ids:
        logger.warning(
            "Vector similarity chunk selection:  no chunk IDs found in entity_info"
        )
        return []

    logger.debug(
        f"Vector similarity chunk selection: {len(all_chunk_ids)} unique chunk IDs collected"
    )

    all_chunk_ids = list(all_chunk_ids)

    try:
        # Use pre-computed query embedding if provided, otherwise compute it
        if query_embedding is None:
            query_embedding = await embedding_func([query], context="query")
            query_embedding = query_embedding[
                0
            ]  # Extract first embedding from batch result
            logger.debug(
                "Computed query embedding for vector similarity chunk selection"
            )
        else:
            logger.debug(
                "Using pre-computed query embedding for vector similarity chunk selection"
            )

        # Get chunk embeddings from vector database
        chunk_vectors = await chunks_vdb.get_vectors_by_ids(all_chunk_ids)
        logger.debug(
            f"Vector similarity chunk selection: {len(chunk_vectors)} chunk vectors Retrieved"
        )

        if not chunk_vectors:
            logger.warning(
                "Vector similarity chunk selection: no vectors retrieved from chunks_vdb"
            )
            return []

        if len(chunk_vectors) != len(all_chunk_ids):
            # A referenced chunk with no stored vector signals an inconsistency
            # between the graph/text stores and the vector store. Keep the
            # existing safety behavior: abort vector ranking so callers can fall
            # back to the WEIGHT method, but make the mismatch easier to observe
            # and repair by logging counts and a bounded sample of missing IDs.
            expected = len(all_chunk_ids)
            retrieved = len(chunk_vectors)
            missing_ids = [
                chunk_id for chunk_id in all_chunk_ids if chunk_id not in chunk_vectors
            ]
            sample = missing_ids[:5]
            logger.warning(
                "Vector similarity chunk selection: data inconsistency detected "
                "(expected %s, retrieved %s, missing %s). Falling back to WEIGHT method. "
                "Missing chunk IDs (sample): %s. "
                "Vector/text storages may be out of sync; consider re-embedding or repairing.",
                expected,
                retrieved,
                expected - retrieved,
                sample,
            )
            return []

        # Calculate cosine similarities
        similarities = []
        valid_vectors = 0
        for chunk_id in all_chunk_ids:
            if chunk_id in chunk_vectors:
                chunk_embedding = chunk_vectors[chunk_id]
                try:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(query_embedding, chunk_embedding)
                    similarities.append((chunk_id, similarity))
                    valid_vectors += 1
                except Exception as e:
                    logger.warning(
                        f"Vector similarity chunk selection: failed to calculate similarity for chunk {chunk_id}: {e}"
                    )
            else:
                logger.warning(
                    f"Vector similarity chunk selection:  no vector found for chunk {chunk_id}"
                )

        # Sort by similarity (highest first) and select top num_of_chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        selected_chunks = [chunk_id for chunk_id, _ in similarities[:num_of_chunks]]

        logger.debug(
            f"Vector similarity chunk selection: {len(selected_chunks)} chunks from {len(all_chunk_ids)} candidates"
        )

        return selected_chunks

    except Exception as e:
        logger.error(f"[VECTOR_SIMILARITY] Error in vector similarity sorting: {e}")
        import traceback

        logger.error(f"[VECTOR_SIMILARITY] Traceback: {traceback.format_exc()}")
        # Fallback to simple truncation
        logger.debug("[VECTOR_SIMILARITY] Falling back to simple truncation")
        return all_chunk_ids[:num_of_chunks]


class TokenTracker:
    """Track token usage for LLM calls."""

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self)

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def add_usage(self, token_counts):
        """Add token usage from one LLM call.

        Args:
            token_counts: A dictionary containing prompt_tokens, completion_tokens, total_tokens
        """
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)

        # If total_tokens is provided, use it directly; otherwise calculate the sum
        if "total_tokens" in token_counts:
            self.total_tokens += token_counts["total_tokens"]
        else:
            self.total_tokens += token_counts.get(
                "prompt_tokens", 0
            ) + token_counts.get("completion_tokens", 0)

        self.call_count += 1

    def get_usage(self):
        """Get current usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }

    def __str__(self):
        usage = self.get_usage()
        return (
            f"LLM call count: {usage['call_count']}, "
            f"Prompt tokens: {usage['prompt_tokens']}, "
            f"Completion tokens: {usage['completion_tokens']}, "
            f"Total tokens: {usage['total_tokens']}"
        )


async def apply_rerank_if_enabled(
    query: str,
    retrieved_docs: list[dict],
    global_config: dict,
    enable_rerank: bool = True,
    top_n: int = None,
) -> list[dict]:
    """
    Apply reranking to retrieved documents if rerank is enabled.

    Args:
        query: The search query
        retrieved_docs: List of retrieved documents
        global_config: Global configuration containing rerank settings
        enable_rerank: Whether to enable reranking from query parameter
        top_n: Number of top documents to return after reranking

    Returns:
        Reranked documents if rerank is enabled, otherwise original documents
    """
    if not enable_rerank or not retrieved_docs:
        return retrieved_docs

    rerank_func = global_config.get("rerank_model_func")
    if not rerank_func:
        logger.warning(
            "Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters."
        )
        return retrieved_docs

    try:
        # Extract document content for reranking
        document_texts = []
        for doc in retrieved_docs:
            # Try multiple possible content fields
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("chunk_content")
                or doc.get("document")
                or str(doc)
            )
            document_texts.append(content)

        # Call the new rerank function that returns index-based results
        rerank_results = await rerank_func(
            query=query,
            documents=document_texts,
            top_n=top_n,
        )

        # Process rerank results based on return format
        if rerank_results and len(rerank_results) > 0:
            first_object = next(
                (result for result in rerank_results if isinstance(result, dict)),
                None,
            )
            # Check if results are in the new index-based format
            if first_object is not None and "index" in first_object:
                # New format: [{"index": 0, "relevance_score": 0.85}, ...]
                reranked_docs = []
                for result in rerank_results:
                    normalized_result, _ = normalize_rerank_result(
                        result, len(retrieved_docs)
                    )
                    if normalized_result is None:
                        continue

                    index = normalized_result["index"]
                    relevance_score = normalized_result["relevance_score"]

                    # Get original document and add rerank score
                    doc = retrieved_docs[index].copy()
                    doc["rerank_score"] = relevance_score
                    reranked_docs.append(doc)

                if not reranked_docs:
                    logger.warning(
                        "Rerank returned no usable results, using original chunks"
                    )
                    return retrieved_docs

                logger.info(
                    f"Successfully reranked: {len(reranked_docs)} chunks from {len(retrieved_docs)} original chunks"
                )
                return reranked_docs
            else:
                # Legacy format: assume it's already reranked documents
                logger.info(f"Using legacy rerank format: {len(rerank_results)} chunks")
                return rerank_results[:top_n] if top_n else rerank_results
        else:
            logger.warning("Rerank returned empty results, using original chunks")
            return retrieved_docs

    except Exception as e:
        logger.error(f"Error during reranking: {e}, using original chunks")
        return retrieved_docs


def normalize_rerank_result(
    result: Any, max_index: int
) -> tuple[dict[str, int | float] | None, str | None]:
    """Validate one provider result and return the public rerank shape.

    Providers, compatible proxies, and custom rerank functions can return mixed
    result lists. Centralizing validation keeps provider adapters, chunk score
    aggregation, and the final query boundary consistent.
    """
    if not isinstance(result, dict):
        return None, "not an object"

    index = result.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return None, "invalid index"
    if not 0 <= index < max_index:
        return None, "index out of range"

    score_value = result.get("relevance_score")
    if isinstance(score_value, bool):
        return None, "invalid relevance score"

    try:
        score = float(score_value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid relevance score"
    if not math.isfinite(score):
        return None, "non-finite relevance score"

    return {"index": index, "relevance_score": score}, None


async def process_chunks_unified(
    query: str,
    unique_chunks: list[dict],
    query_param: "QueryParam",
    global_config: dict,
    source_type: str = "mixed",
    chunk_token_limit: int = None,  # Add parameter for dynamic token limit
    progress_callback=None,
) -> list[dict]:
    """
    Unified processing for text chunks: deduplication, chunk_top_k limiting, reranking, and token truncation.

    Args:
        query: Search query for reranking
        chunks: List of text chunks to process
        query_param: Query parameters containing configuration
        global_config: Global configuration dictionary
        source_type: Source type for logging ("vector", "entity", "relationship", "mixed")
        chunk_token_limit: Dynamic token limit for chunks (if None, uses default)

    Returns:
        Processed and filtered list of text chunks
    """
    if not unique_chunks:
        return []

    origin_count = len(unique_chunks)

    # 1. Apply reranking if enabled and query is provided
    if query_param.enable_rerank and query and unique_chunks:
        if progress_callback:
            await progress_callback("reranking")
        rerank_top_k = query_param.chunk_top_k or len(unique_chunks)
        unique_chunks = await apply_rerank_if_enabled(
            query=query,
            retrieved_docs=unique_chunks,
            global_config=global_config,
            enable_rerank=query_param.enable_rerank,
            top_n=rerank_top_k,
        )

    # 2. Filter by minimum rerank score if reranking is enabled
    if query_param.enable_rerank and unique_chunks:
        min_rerank_score = global_config.get("min_rerank_score", 0.5)
        if min_rerank_score > 0.0:
            original_count = len(unique_chunks)

            # Filter chunks with score below threshold
            filtered_chunks = []
            for chunk in unique_chunks:
                rerank_score = chunk.get(
                    "rerank_score", 1.0
                )  # Default to 1.0 if no score
                if rerank_score >= min_rerank_score:
                    filtered_chunks.append(chunk)

            unique_chunks = filtered_chunks
            filtered_count = original_count - len(unique_chunks)

            if filtered_count > 0:
                logger.info(
                    f"Rerank filtering: {len(unique_chunks)} chunks remained (min rerank score: {min_rerank_score})"
                )
            if not unique_chunks:
                return []

    # 3. Apply chunk_top_k limiting if specified
    if query_param.chunk_top_k is not None and query_param.chunk_top_k > 0:
        if len(unique_chunks) > query_param.chunk_top_k:
            unique_chunks = unique_chunks[: query_param.chunk_top_k]
        logger.debug(
            f"Kept chunk_top-k: {len(unique_chunks)} chunks (deduplicated original: {origin_count})"
        )

    # 4. Token-based final truncation
    tokenizer = global_config.get("tokenizer")
    if tokenizer and unique_chunks:
        # Set default chunk_token_limit if not provided
        if chunk_token_limit is None:
            # Get default from query_param or global_config
            chunk_token_limit = getattr(
                query_param,
                "max_total_tokens",
                global_config.get("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS),
            )

        original_count = len(unique_chunks)

        unique_chunks = await run_in_tokenizer_executor(
            _truncate_chunks_for_unified_context,
            unique_chunks,
            chunk_token_limit,
            tokenizer,
        )

        logger.debug(
            f"Token truncation: {len(unique_chunks)} chunks from {original_count} "
            f"(chunk available tokens: {chunk_token_limit}, source: {source_type})"
        )

    # 5. add id field to each chunk
    final_chunks = []
    for i, chunk in enumerate(unique_chunks):
        chunk_with_id = chunk.copy()
        chunk_with_id["id"] = f"DC{i + 1}"
        final_chunks.append(chunk_with_id)

    return final_chunks


def normalize_source_ids_limit_method(method: str | None) -> str:
    """Normalize the source ID limiting strategy and fall back to default when invalid."""

    if not method:
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    normalized = method.upper()
    if normalized not in VALID_SOURCE_IDS_LIMIT_METHODS:
        logger.warning(
            "Unknown SOURCE_IDS_LIMIT_METHOD '%s', falling back to %s",
            method,
            DEFAULT_SOURCE_IDS_LIMIT_METHOD,
        )
        return DEFAULT_SOURCE_IDS_LIMIT_METHOD

    return normalized


def merge_source_ids(
    existing_ids: Iterable[str] | None, new_ids: Iterable[str] | None
) -> list[str]:
    """Merge two iterables of source IDs into one flat, ordered, deduplicated list.

    Every element is split on ``GRAPH_FIELD_SEP``, stripped, and deduplicated by
    first-seen order. The split is what makes this the normalization boundary
    for chunk-id lists: an element that is itself a joined string
    (``"chunk-a<SEP>chunk-b"``) would otherwise be stored as a single id, and
    such an id matches no key in ``text_chunks``. It would then corrupt the
    ``chunk_ids``/``count`` rows of ``entity_chunks_storage`` /
    ``relation_chunks_storage``, miscount ``apply_source_ids_limit``'s
    truncation (two chunks counted as one), and miss every downstream
    ``get_by_ids`` lookup.

    No in-tree producer passes a joined element today — every caller splits its
    graph ``source_id`` first, and extraction emits one ``chunk_key`` per record
    — so the split is a guard against a future upstream regression rather than a
    fix for a live path. A joined element arriving here therefore means an
    upstream bug and is logged at debug level so the silent repair does not hide
    the root cause.

    Despite the name, this also merges the sibling union fields that share the
    same ``GRAPH_FIELD_SEP`` encoding: ``file_path`` and ``description`` in the
    edge-dedup merges of ``mongo_impl`` / ``opensearch_impl``. Stripping applies
    to those fragments too, so ``" foo"`` and ``"foo"`` collapse into one entry.
    """

    merged: list[str] = []
    seen: set[str] = set()

    for sequence in (existing_ids, new_ids):
        if not sequence:
            continue
        for source_id in sequence:
            if not source_id:
                continue
            if not isinstance(source_id, str):
                # Coerce rather than skip: dropping the value would silently
                # lose provenance, which is the failure mode this function
                # exists to prevent. The warning surfaces the type bug.
                logger.warning(
                    f"merge_source_ids received a non-string id "
                    f"{source_id!r} ({type(source_id).__name__}); coercing to str"
                )
                source_id = str(source_id)
            if GRAPH_FIELD_SEP in source_id:
                logger.debug(
                    f"merge_source_ids splitting a GRAPH_FIELD_SEP-joined value "
                    f"{source_id!r}; callers are expected to pass individual ids"
                )
            # split_string_by_multi_markers already strips each fragment and
            # drops the empty ones, so a whitespace-only element yields nothing.
            for sid in split_string_by_multi_markers(source_id, [GRAPH_FIELD_SEP]):
                if sid not in seen:
                    seen.add(sid)
                    merged.append(sid)

    return merged


def apply_source_ids_limit(
    source_ids: Sequence[str],
    limit: int,
    method: str,
    *,
    identifier: str | None = None,
) -> list[str]:
    """Apply a limit strategy to a sequence of source IDs."""

    if limit <= 0:
        return []

    source_ids_list = list(source_ids)
    if len(source_ids_list) <= limit:
        return source_ids_list

    normalized_method = normalize_source_ids_limit_method(method)

    if normalized_method == SOURCE_IDS_LIMIT_METHOD_FIFO:
        truncated = source_ids_list[-limit:]
    else:  # IGNORE_NEW
        truncated = source_ids_list[:limit]

    if identifier and len(truncated) < len(source_ids_list):
        logger.debug(
            "Source_id truncated: %s | %s keeping %s of %s entries",
            identifier,
            normalized_method,
            len(truncated),
            len(source_ids_list),
        )

    return truncated


def compute_incremental_chunk_ids(
    existing_full_chunk_ids: list[str],
    old_chunk_ids: list[str],
    new_chunk_ids: list[str],
) -> list[str]:
    """
    Compute incrementally updated chunk IDs based on changes.

    This function applies delta changes (additions and removals) to an existing
    list of chunk IDs while maintaining order and ensuring deduplication.
    Delta additions from new_chunk_ids are placed at the end. Empty IDs are
    dropped from both inputs.

    Authority model:
        ``existing_full_chunk_ids`` — the entity/relation chunk-tracking row — is
        AUTHORITATIVE. A graph node's ``source_id`` is only a truncated view of it
        (see ``apply_source_ids_limit``), and it may legitimately retain STALE chunk
        IDs that tracking has already pruned: the purge path reads tracking first and
        falls back to ``source_id`` only when the tracking row is absent, and its
        ``graph_references_deleted_chunks`` branch exists precisely to handle a graph
        that still references chunks tracking has dropped.

        Consequently an ID present in BOTH ``old_chunk_ids`` and ``new_chunk_ids`` but
        absent from ``existing_full_chunk_ids`` is treated as intentionally pruned and
        is NOT restored — only genuine additions (``new - old``) are appended. Widening
        Step 2 to append every entry of ``new_chunk_ids`` would resurrect stale
        attribution into the authoritative store, which a later purge would then use to
        rebuild or retain KG objects sourced from chunks that no longer exist.
        Repairing genuinely missing attribution is the job of
        ``audit_kg_integrity(..., apply=True)``, never of this function.

    Args:
        existing_full_chunk_ids: Complete list of existing chunk IDs from storage
        old_chunk_ids: Previous chunk IDs from source_id (chunks being replaced)
        new_chunk_ids: New chunk IDs from updated source_id (chunks being added)

    Returns:
        Updated list of chunk IDs with deduplication

    Example:
        >>> existing = ['chunk-1', 'chunk-2', 'chunk-3']
        >>> old = ['chunk-1', 'chunk-2']
        >>> new = ['chunk-2', 'chunk-4']
        >>> compute_incremental_chunk_ids(existing, old, new)
        ['chunk-2', 'chunk-3', 'chunk-4']
    """
    # Calculate changes
    chunks_to_remove = set(old_chunk_ids) - set(new_chunk_ids)
    chunks_to_add = set(new_chunk_ids) - set(old_chunk_ids)

    # Step 1: Remove chunks that are no longer needed
    updated_chunk_ids = [
        cid for cid in existing_full_chunk_ids if cid and cid not in chunks_to_remove
    ]
    seen = set(updated_chunk_ids)

    # Step 2: Append genuine additions only (preserving order from new_chunk_ids).
    # The `seen` check is not redundant with `chunks_to_add`: tracking may already
    # hold an ID that is in new_chunk_ids but not in old_chunk_ids, because
    # source_id is a truncated view of the tracking row.
    for cid in new_chunk_ids:
        if cid and cid in chunks_to_add and cid not in seen:
            seen.add(cid)
            updated_chunk_ids.append(cid)

    return updated_chunk_ids


def subtract_source_ids(
    source_ids: Iterable[str],
    ids_to_remove: Collection[str],
) -> list[str]:
    """Remove a collection of IDs from an ordered iterable while preserving order."""

    removal_set = set(ids_to_remove)
    if not removal_set:
        return [source_id for source_id in source_ids if source_id]

    return [
        source_id
        for source_id in source_ids
        if source_id and source_id not in removal_set
    ]


def make_relation_chunk_key(src: str, tgt: str) -> str:
    """Create a deterministic storage key for relation chunk tracking."""

    return GRAPH_FIELD_SEP.join(sorted((src, tgt)))


def parse_relation_chunk_key(key: str) -> tuple[str, str]:
    """Parse a relation chunk storage key back into its entity pair."""

    parts = key.split(GRAPH_FIELD_SEP)
    if len(parts) != 2:
        raise ValueError(f"Invalid relation chunk key: {key}")
    return parts[0], parts[1]


def generate_track_id(prefix: str = "upload") -> str:
    """Generate a unique tracking ID with timestamp and UUID

    Args:
        prefix: Prefix for the track ID (e.g., 'upload', 'insert')

    Returns:
        str: Unique tracking ID in format: {prefix}_{timestamp}_{uuid}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    return f"{prefix}_{timestamp}_{unique_id}"


def get_pinyin_sort_key(text: str) -> str:
    """Generate sort key for Chinese pinyin sorting

    This function uses pypinyin for true Chinese pinyin sorting.
    If pypinyin is not available, it falls back to simple lowercase string sorting.

    Args:
        text: Text to generate sort key for

    Returns:
        str: Sort key that can be used for comparison and sorting
    """
    if not text:
        return ""

    if _PYPINYIN_AVAILABLE:
        try:
            # Convert Chinese characters to pinyin, keep non-Chinese as-is
            pinyin_list = pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL)
            return "".join(pinyin_list).lower()
        except Exception:
            # Silently fall back to simple string sorting on any error
            return text.lower()
    else:
        # pypinyin not available, use simple string sorting
        return text.lower()


def fix_tuple_delimiter_corruption(
    record: str, delimiter_core: str, tuple_delimiter: str
) -> str:
    """
    Fix various forms of tuple_delimiter corruption from LLM output.

    This function handles missing or replaced characters around the core delimiter.
    It fixes common corruption patterns where the LLM output doesn't match the expected
    tuple_delimiter format.

    Args:
        record: The text record to fix
        delimiter_core: The core delimiter (e.g., "S" from "<|#|>")
        tuple_delimiter: The complete tuple delimiter (e.g., "<|#|>")

    Returns:
        The corrected record with proper tuple_delimiter format
    """
    if not record or not delimiter_core or not tuple_delimiter:
        return record

    # Escape the delimiter core for regex use
    escaped_delimiter_core = re.escape(delimiter_core)

    # Fix: <|##|> -> <|#|>, <|#||#|> -> <|#|>, <|#|||#|> -> <|#|>
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|\#|> -> <|#|>
    record = re.sub(
        rf"<\|\\{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|> -> <|#|>, <||> -> <|#|> (glued only; keep free-text "a <|> b")
    record = re.sub(
        r"(?<=\S)<\|+>(?=\S)",
        tuple_delimiter,
        record,
    )

    # Fix: <X|#|> -> <|#|>, <|#|Y> -> <|#|>, <X|#|Y> -> <|#|>, <||#||> -> <|#|> (one extra characters outside pipes)
    record = re.sub(
        rf"<.?\|{escaped_delimiter_core}\|.?>",
        tuple_delimiter,
        record,
    )

    # Fix: <#>, <#|>, <|#> -> <|#|> (missing one or both pipes)
    record = re.sub(
        rf"<\|?{escaped_delimiter_core}\|?>",
        tuple_delimiter,
        record,
    )

    # Fix: <X#|> -> <|#|>, <|#X> -> <|#|> (one pipe is replaced by other character)
    record = re.sub(
        rf"<[^|]{escaped_delimiter_core}\|>|<\|{escaped_delimiter_core}[^|]>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#| -> <|#|>, <|#|| -> <|#|> (missing closing >)
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|+(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix <|#: -> <|#|> (missing closing >)
    record = re.sub(
        rf"<\|{escaped_delimiter_core}:(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: <||#> -> <|#|> (double pipe at start, missing pipe at end)
    record = re.sub(
        rf"<\|+{escaped_delimiter_core}>",
        tuple_delimiter,
        record,
    )

    # Fix: <|| -> <|#|> (glued only; keep free-text/code "x <|| y")
    # Anchor the left side only: "<||" is an unterminated separator whose right
    # side is the next field's raw content, so a right-hand \S anchor would add
    # nothing while risking a boundary miss. The left \S is what distinguishes a
    # glued corruption (name<||type) from a spaced free-text mention (x <|| y).
    record = re.sub(
        r"(?<=\S)<\|\|(?!>)",
        tuple_delimiter,
        record,
    )

    # Fix: |#|> -> <|#|> (missing opening <)
    record = re.sub(
        rf"(?<!<)\|{escaped_delimiter_core}\|>",
        tuple_delimiter,
        record,
    )

    # Fix: <|#|>| -> <|#|>  ( this is a fix for: <|#|| -> <|#|> )
    record = re.sub(
        rf"<\|{escaped_delimiter_core}\|>\|",
        tuple_delimiter,
        record,
    )

    # Fix: ||#|| -> <|#|> (double pipes on both sides without angle brackets)
    record = re.sub(
        rf"\|\|{escaped_delimiter_core}\|\|",
        tuple_delimiter,
        record,
    )

    return record


def create_prefixed_exception(original_exception: Exception, prefix: str) -> Exception:
    """
    Safely create a prefixed exception that adapts to all error types.

    Args:
        original_exception: The original exception.
        prefix: The prefix to add.

    Returns:
        A new exception with the prefix, maintaining the original exception type if possible.
    """
    try:
        # Method 1: Try to reconstruct using original arguments.
        if hasattr(original_exception, "args") and original_exception.args:
            args = list(original_exception.args)
            # Find the first string argument and prefix it. This is safer for
            # exceptions like OSError where the first arg is an integer (errno).
            found_str = False
            for i, arg in enumerate(args):
                if isinstance(arg, str):
                    args[i] = f"{prefix}: {arg}"
                    found_str = True
                    break

            # If no string argument is found, prefix the first argument's string representation.
            if not found_str:
                args[0] = f"{prefix}: {args[0]}"

            return type(original_exception)(*args)
        else:
            # Method 2: If no args, try single parameter construction.
            return type(original_exception)(f"{prefix}: {str(original_exception)}")
    except Exception:
        # Method 3: If reconstruction fails for any reason, wrap it in a
        # RuntimeError preserving the original type name and message. This is a
        # defensive catch-all: most known failures already surface as TypeError
        # (e.g. json.JSONDecodeError needs (msg, doc, pos) and
        # openai.APIStatusError/BadRequestError need keyword-only
        # (response, body), so rebuilding from args alone raises TypeError), but
        # an exotic constructor could raise something else (KeyError, a
        # validation error, ...). Catching `Exception` guarantees this helper
        # never raises while prefixing — `KeyboardInterrupt`/`SystemExit` are
        # BaseException and still propagate. The original exception and its full
        # traceback are preserved by the caller's `raise ... from original`.
        return RuntimeError(
            f"{prefix}: {type(original_exception).__name__}: {str(original_exception)}"
        )


def convert_to_user_format(
    entities_context: list[dict],
    relations_context: list[dict],
    chunks: list[dict],
    references: list[dict],
    query_mode: str,
    entity_id_to_original: dict = None,
    relation_id_to_original: dict = None,
) -> dict[str, Any]:
    """Convert internal data format to user-friendly format using original database data"""

    # Convert entities format using original data when available
    formatted_entities = []
    for entity in entities_context:
        entity_name = entity.get("entity", "")

        # Try to get original data first
        original_entity = None
        if entity_id_to_original and entity_name in entity_id_to_original:
            original_entity = entity_id_to_original[entity_name]

        if original_entity:
            # Use original database data
            formatted_entities.append(
                {
                    "entity_name": original_entity.get("entity_name", entity_name),
                    "entity_type": original_entity.get("entity_type", "UNKNOWN"),
                    "description": original_entity.get("description", ""),
                    "source_id": original_entity.get("source_id", ""),
                    "file_path": original_entity.get("file_path", "unknown_source"),
                    "created_at": original_entity.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_entities.append(
                {
                    "entity_name": entity_name,
                    "entity_type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", "unknown_source"),
                    "created_at": entity.get("created_at", ""),
                }
            )

    # Convert relationships format using original data when available
    formatted_relationships = []
    for relation in relations_context:
        entity1 = relation.get("entity1", "")
        entity2 = relation.get("entity2", "")
        relation_key = (entity1, entity2)

        # Try to get original data first
        original_relation = None
        if relation_id_to_original and relation_key in relation_id_to_original:
            original_relation = relation_id_to_original[relation_key]

        if original_relation:
            # Use original database data
            formatted_relationships.append(
                {
                    "src_id": original_relation.get("src_id", entity1),
                    "tgt_id": original_relation.get("tgt_id", entity2),
                    "description": original_relation.get("description", ""),
                    "keywords": original_relation.get("keywords", ""),
                    "weight": original_relation.get("weight", 1.0),
                    "source_id": original_relation.get("source_id", ""),
                    "file_path": original_relation.get("file_path", "unknown_source"),
                    "created_at": original_relation.get("created_at", ""),
                }
            )
        else:
            # Fallback to LLM context data (for backward compatibility)
            formatted_relationships.append(
                {
                    "src_id": entity1,
                    "tgt_id": entity2,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": relation.get("weight", 1.0),
                    "source_id": relation.get("source_id", ""),
                    "file_path": relation.get("file_path", "unknown_source"),
                    "created_at": relation.get("created_at", ""),
                }
            )

    # Convert chunks format (chunks already contain complete data)
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "reference_id": chunk.get("reference_id", ""),
            "content": chunk.get("content", ""),
            "file_path": chunk.get("file_path", "unknown_source"),
            "chunk_id": chunk.get("chunk_id", ""),
        }
        formatted_chunks.append(chunk_data)

    logger.debug(
        f"[convert_to_user_format] Formatted {len(formatted_chunks)}/{len(chunks)} chunks"
    )

    # Build basic metadata (metadata details will be added by calling functions)
    metadata = {
        "query_mode": query_mode,
        "keywords": {
            "high_level": [],
            "low_level": [],
        },  # Placeholder, will be set by calling functions
    }

    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": formatted_entities,
            "relationships": formatted_relationships,
            "chunks": formatted_chunks,
            "references": references,
        },
        "metadata": metadata,
    }


def generate_reference_list_from_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Generate reference list from chunks, prioritizing by occurrence frequency.

    This function extracts file_paths from chunks, counts their occurrences,
    sorts by frequency and first appearance order, creates reference_id mappings,
    and builds a reference_list structure.

    Args:
        chunks: List of chunk dictionaries with file_path information

    Returns:
        tuple: (reference_list, updated_chunks_with_reference_ids)
            - reference_list: List of dicts with reference_id and file_path
            - updated_chunks_with_reference_ids: Original chunks with reference_id field added
    """
    if not chunks:
        return [], []

    # 1. Extract all valid file_paths and count their occurrences
    file_path_counts = {}
    for chunk in chunks:
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source":
            file_path_counts[file_path] = file_path_counts.get(file_path, 0) + 1

    # 2. Sort file paths by frequency (descending), then by first appearance order
    # Create a list of (file_path, count, first_index) tuples
    file_path_with_indices = []
    seen_paths = set()
    for i, chunk in enumerate(chunks):
        file_path = chunk.get("file_path", "")
        if file_path and file_path != "unknown_source" and file_path not in seen_paths:
            file_path_with_indices.append((file_path, file_path_counts[file_path], i))
            seen_paths.add(file_path)

    # Sort by count (descending), then by first appearance index (ascending)
    sorted_file_paths = sorted(file_path_with_indices, key=lambda x: (-x[1], x[2]))
    unique_file_paths = [item[0] for item in sorted_file_paths]

    # 3. Create mapping from file_path to reference_id (prioritized by frequency)
    file_path_to_ref_id = {}
    for i, file_path in enumerate(unique_file_paths):
        file_path_to_ref_id[file_path] = str(i + 1)

    # 4. Add reference_id field to each chunk
    updated_chunks = []
    for chunk in chunks:
        chunk_copy = chunk.copy()
        file_path = chunk_copy.get("file_path", "")
        if file_path and file_path != "unknown_source":
            chunk_copy["reference_id"] = file_path_to_ref_id[file_path]
        else:
            chunk_copy["reference_id"] = ""
        updated_chunks.append(chunk_copy)

    # 5. Build reference_list
    reference_list = []
    for i, file_path in enumerate(unique_file_paths):
        reference_list.append({"reference_id": str(i + 1), "file_path": file_path})

    return reference_list, updated_chunks


def render_chunks_context_text(chunks_with_reference_ids: list[dict]) -> str:
    """Render the exact chunk-context text sent to the LLM.

    ``chunks_with_reference_ids`` must already carry ``reference_id`` — the
    second return value of :func:`generate_reference_list_from_chunks`. This
    is the single place that projects a chunk down to
    ``{reference_id, content, content_headings?}`` and serializes it, one JSON
    object per line, so that any token-budget check done against this exact
    call sequence matches what callers go on to send downstream verbatim.
    """
    chunks_context = []
    for chunk in chunks_with_reference_ids:
        entry = {
            "reference_id": chunk["reference_id"],
            "content": chunk["content"],
        }
        if chunk.get("content_headings"):
            entry["content_headings"] = chunk["content_headings"]
        chunks_context.append(entry)
    return "\n".join(
        json.dumps(text_unit, ensure_ascii=False) for text_unit in chunks_context
    )


def _truncate_chunks_for_unified_context(
    chunks: list[dict], max_token_size: int, tokenizer: "Tokenizer"
) -> list[dict]:
    """Two-stage truncation used by :func:`process_chunks_unified`.

    Counting a chunk list's tokens against the chunk dicts themselves (as the
    single-stage version used to) undercounts: what actually reaches the LLM
    is the ``{reference_id, content, content_headings?}`` projection built by
    :func:`generate_reference_list_from_chunks` /
    :func:`render_chunks_context_text`, and ``reference_id`` itself is
    recomputed from each survivor's ``file_path`` frequency — which changes
    with the exact set of chunks kept, not just by a token or two.

    Stage 1 approximates a safe count K from ``{content, content_headings}``
    alone (``reference_id`` isn't assigned yet, and can't be until the
    survivor set is known — a chicken-and-egg the real renderer resolves by
    running after truncation, not before). Stage 2 re-renders that exact
    candidate list through the real renderer and independently re-verifies
    (shrinking K if needed), so the K this function returns is guaranteed safe
    under the SAME rendering the caller will perform afterward on the same
    list. Both stages run in this one synchronous call — it must always be
    submitted as a single ``run_in_tokenizer_executor`` job, never split
    across two round-trips through the event loop.
    """
    if max_token_size <= 0 or not chunks:
        return []

    def _approx_key(chunk: dict) -> str:
        payload = {"content": chunk.get("content")}
        if chunk.get("content_headings"):
            payload["content_headings"] = chunk["content_headings"]
        return json.dumps(payload, ensure_ascii=False)

    approx = truncate_list_by_token_size(
        chunks,
        key=_approx_key,
        separator="\n",
        max_token_size=max_token_size,
        tokenizer=tokenizer,
    )

    k = len(approx)
    while k > 0:
        _, rendered_chunks = generate_reference_list_from_chunks(approx[:k])
        text = render_chunks_context_text(rendered_chunks)
        if len(tokenizer.encode(text)) <= max_token_size:
            break
        k -= 1

    return approx[:k]


def validate_workspace(workspace: str) -> str:
    """Validate a workspace name used to build per-workspace directories.

    File-based storages place their data in a subdirectory named after the
    workspace under ``working_dir`` (``os.path.join(working_dir, workspace)``).
    To prevent path traversal, the workspace must be a single path component:
    it may not contain a path separator nor be a relative path reference.

    Unlike a sanitizing approach, this validator does not rewrite the name.
    Legitimate names containing dots (e.g. ``"v1.0"``) are accepted unchanged,
    while unsafe names are rejected so the caller fails fast instead of
    silently reading or writing outside the intended directory.

    Args:
        workspace: Workspace name from configuration or environment variables.

    Returns:
        The workspace name unchanged when it is valid.

    Raises:
        ValueError: If the workspace contains ``/`` or ``\\``, or is ``"."`` or
            ``".."``.

    Examples:
        >>> validate_workspace("my_workspace")
        'my_workspace'
        >>> validate_workspace("v1.0")
        'v1.0'
        >>> validate_workspace("../../../etc")
        Traceback (most recent call last):
            ...
        ValueError: Invalid workspace name '../../../etc': must not contain path separators ('/', '\\') or be a relative path reference ('.', '..')
    """
    if "/" in workspace or "\\" in workspace or workspace in (".", ".."):
        raise ValueError(
            f"Invalid workspace name {workspace!r}: must not contain path "
            "separators ('/', '\\') or be a relative path reference ('.', '..')"
        )
    return workspace


# Complement of the XML 1.0 ``Char`` production. GraphML is XML, so a string
# holding any other code point cannot be serialized at all -- tab, newline and
# carriage return are the only C0 controls XML admits, and descriptions
# legitimately carry those.
_XML_INCOMPATIBLE_CHAR_PATTERN = re.compile(
    r"[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]"
)

# Attribute names a backend would read as something other than a name. Kept to
# exactly that: MongoGraphStorage passes names straight into a ``$set``
# document, where a dot is a *path* separator (``source_ids.0`` rewrites an
# element of the chunk-attribution array rather than creating a field) and a
# leading ``$`` is an update operator.
#
# Deliberately NOT "must be an identifier". Names like ``display-name`` or
# ``has space`` carry no interpretation hazard on any backend, and the manual
# edit API accepted them before the field allowlist landed -- so the document
# backends that persisted them hold such names today. Refusing them here would
# make those entities unmodifiable, because every rewrite path (entity edit,
# rename, merge, extraction rebuild) spreads the *stored* attributes back into
# the upsert payload.
_GRAPH_ATTRIBUTE_NAME_DISALLOWED_SUBSTRING = "."
_GRAPH_ATTRIBUTE_NAME_DISALLOWED_PREFIX = "$"

# Integer attributes must fit in a signed 64-bit integer. Verified against the
# installed driver rather than assumed: neo4j's packstream Packer raises
# ``OverflowError("Integer ... out of range")`` outside ``[-2**63, 2**63)``
# (``neo4j/_codec/packstream/v1/__init__.py``), and GraphML declares an int
# attribute as ``attr.type="long"``, which is ``xsd:long`` -- also int64. So a
# larger int is outside the intersection this module defines, even though
# networkx will happily write it and PostgreSQL's arbitrary-precision ``jsonb``
# numeric will happily store it.
_GRAPH_ATTRIBUTE_INT_MIN = -(2**63)
_GRAPH_ATTRIBUTE_INT_MAX = 2**63 - 1


def _graphml_encodable_types() -> frozenset[type]:
    """The exact types networkx's GraphML writer knows how to encode.

    Asked of networkx rather than enumerated here, because ``add_data`` resolves
    a value by looking its type up in ``self.xml_type`` -- so that table *is* the
    definition, and reproducing it by hand would drift. Sourcing it also makes
    the guard's justification true by construction: every type this module
    refuses is a type the write would have failed on.

    An ``isinstance`` shortcut over ``np.integer`` / ``np.floating`` would be
    wrong, not merely loose: ``np.longdouble`` is an ``np.floating`` and is *not*
    in the table, so it would be accepted here and rejected at write time --
    the poisoning this exists to prevent.

    The four builtins are unioned in as a floor, and a failure to read the table
    degrades to exactly that floor. ``networkx`` carries no version bound in
    pyproject.toml, so its internals could move; degrading to the floor is
    stricter, never looser, so it cannot open the hole it guards.
    """
    builtin_scalars = frozenset({str, int, float, bool})
    try:
        from networkx.readwrite.graphml import GraphML

        probe = GraphML()
        probe.construct_types()
        return frozenset(probe.xml_type) | builtin_scalars
    except Exception:  # networkx internals moved: fall back to the strict floor
        return builtin_scalars


_GRAPHML_ENCODABLE_TYPES = _graphml_encodable_types()

# The intersection every registered backend can carry. Narrower than
# ``_GRAPHML_ENCODABLE_TYPES``, which also holds the numpy scalar types GraphML
# writes: those are not portable, and not marginally so -- ``json.dumps``
# (PGTableGraphStorage's jsonb column) and ``bson`` (MongoGraphStorage) both
# refuse ``np.float32`` / ``np.int64`` / ``np.uint32`` / ``np.bool_`` outright.
_PORTABLE_SCALAR_TYPES = frozenset({str, int, float, bool})


def xml_attribute_name_rejection(name: Any) -> str | None:
    """Explain why *name* cannot be used as an XML-serialized attribute name.

    GraphML writes attribute names into the XML ``attr.name`` field, so a name is
    subject to the same character rule as a value -- a claim checked against
    networkx rather than assumed: a name holding a control character, a lone
    surrogate or U+FFFE makes ``write_graphml`` raise, which on
    ``NetworkXStorage`` is the instance-wide persistence outage the storage guard
    exists to prevent.

    Everything else is allowed, including names a *portable* rule would refuse
    (``a.b``, ``$set``, ``display-name``, ``has space``, ``""``). Two reasons:
    GraphML round-trips all of those, and -- the load-bearing one -- a name this
    function rejects can never have been persisted in an existing GraphML file,
    because the write that would have stored it failed. That is what makes the
    check safe to apply to a rewrite payload, which spreads a fetched object's
    stored names back in.

    Args:
        name: Candidate attribute name.

    Returns:
        A reason fragment, or ``None`` when XML can carry the name.
    """
    if not isinstance(name, str):
        # networkx raises TypeError("keywords must be strings") before mutating,
        # so this cannot poison the graph -- it is folded in so the caller gets a
        # validation error rather than an unhandled TypeError.
        return f"must be a string, got {type(name).__name__}"
    match = _XML_INCOMPATIBLE_CHAR_PATTERN.search(name)
    if match:
        return (
            "must not contain the character "
            f"U+{ord(match.group()):04X}, which XML cannot encode"
        )
    return None


def xml_attribute_value_rejection(value: Any) -> str | None:
    """Explain why *value* cannot be encoded in XML at all.

    The narrower of the two value rules, and the one a GraphML-backed store
    should enforce: it rejects only what cannot be *serialized*, not what is
    merely unportable. ``NaN``, ``inf`` and an integer past int64 all round-trip
    through GraphML unchanged, so they are accepted here even though
    ``graph_attribute_value_rejection`` refuses them.

    "Serialized" is the operative word, and it is not the same as "is a scalar":
    an integer with more digits than ``sys.get_int_max_str_digits()`` cannot be
    rendered as text at all, so GraphML cannot hold it however ordinary it looks.

    That difference is deliberate, and it is what keeps a backend guard from
    stranding data. A workspace can already hold such a value -- the manual edit
    API accepted anything before its field allowlist landed -- and every rewrite
    path (entity edit, extraction rebuild) spreads a fetched object's stored
    attributes back into the upsert payload. Enforcing the *portable* rule at the
    storage layer would therefore make those objects unmodifiable, while
    enforcing it where caller input enters costs nothing.

    Args:
        value: Candidate attribute value.

    Returns:
        A reason fragment suitable for appending to ``"attribute 'x' "``, or
        ``None`` when XML can carry the value.
    """
    # Exact type, not ``isinstance``. networkx's GraphML writer resolves a value
    # by looking its type up in a dict (``add_data``: ``if element_type not in
    # self.xml_type``), so a *subclass* of a supported scalar is rejected at write
    # time -- ``enum.StrEnum``, ``enum.IntEnum``, ``np.str_``, ``Decimal`` and any
    # user ``class X(str)`` all satisfy ``isinstance`` and all make
    # ``write_graphml`` raise. Read from the installed networkx, not assumed.
    #
    # The set comes from that same table (see ``_graphml_encodable_types``), so
    # it includes the numpy scalar types GraphML writes -- and so that every type
    # refused here is one the write would genuinely have failed on.
    value_type = type(value)
    if value_type not in _GRAPHML_ENCODABLE_TYPES:
        if isinstance(value, (str, int, float, bool)):
            # A subclass: worth saying so, because the caller reasonably expects
            # a `str` subclass to be a string.
            return (
                f"must be a type GraphML can encode, got {value_type.__name__} "
                "(GraphML resolves value types exactly, so a subclass of a "
                "supported scalar is still refused)"
            )
        return f"must be a string, number or boolean, got {value_type.__name__}"
    if value_type is not str and value_type is not int:
        # bool, float and the numpy numeric scalars need nothing further: the
        # only remaining checks are the digit limit (int) and the character rule
        # (str). A numpy int is width-bounded, so it cannot hit the digit limit.
        return None
    if value_type is int:
        # GraphML stores values as text, and networkx stringifies them on write.
        # CPython refuses to stringify an integer with more digits than
        # ``sys.get_int_max_str_digits()`` (4300 by default since 3.11, and
        # settable at runtime), so a large enough int makes ``write_graphml``
        # raise even though the int itself is a perfectly ordinary scalar.
        # Attempt the same conversion the writer will make rather than comparing
        # against the limit, so this cannot drift from it if the limit is changed.
        try:
            str(value)
        except ValueError:
            # `get_int_max_str_digits` arrived in 3.11 and was backported to
            # 3.10.7, so it can be missing on the `requires-python = ">=3.10"`
            # floor. It is unreachable here when missing -- no limit means
            # `str()` cannot raise -- but read it defensively rather than relying
            # on that.
            read_limit = getattr(sys, "get_int_max_str_digits", None)
            limit = f" (limit {read_limit()} digits)" if read_limit else ""
            return f"has more digits than Python will render as text{limit}"
        return None
    match = _XML_INCOMPATIBLE_CHAR_PATTERN.search(value)
    if match:
        return (
            "must not contain the character "
            f"U+{ord(match.group()):04X}, which XML cannot encode"
        )
    return None


def graph_attribute_value_rejection(value: Any) -> str | None:
    """Explain why *value* cannot be stored as a graph node/edge attribute.

    The wider of the two value rules: the intersection of what *every*
    registered backend can carry, meant for the point where caller input enters
    (see ``xml_attribute_value_rejection`` for the narrower serialization rule a
    storage backend should enforce instead, and why the two differ).

    The rules are the intersection of what the seven registered
    ``GRAPH_STORAGE`` backends can carry, so the same payload behaves the same
    way on all of them:

    * ``bool`` -- accepted everywhere.
    * ``int`` -- must fit in int64. A larger int is written happily by networkx
      and stored happily by PostgreSQL's arbitrary-precision ``jsonb`` numeric,
      but the Neo4j driver refuses to pack it and GraphML mislabels it as
      ``xsd:long``. See ``_GRAPH_ATTRIBUTE_INT_MIN``.
    * ``float`` -- must be finite. ``NaN`` / ``inf`` survive GraphML but
      ``json.dumps`` renders them as bare ``NaN`` / ``Infinity``, which is not
      valid JSON and is rejected by the ``jsonb`` column PGTableGraphStorage
      writes to.
    * ``str`` -- must be XML-compatible, because GraphML cannot encode the
      other code points at all (see ``_XML_INCOMPATIBLE_CHAR_PATTERN``).
    * anything else (``dict``, ``list``, ``None``, ``bytes``, ...) -- rejected.
      ``None`` is called out because it is the one non-scalar that reads as
      harmless: GraphML refuses it outright, while on the Cypher backends
      ``SET n += {k: null}`` silently *deletes* the property.

    Args:
        value: Candidate attribute value.

    Returns:
        A reason fragment suitable for appending to ``"attribute 'x' "``, or
        ``None`` when the value is storable.
    """
    rejection = xml_attribute_value_rejection(value)
    if rejection is not None:
        return rejection
    # The XML rule accepts what GraphML can write, which includes numpy scalars;
    # portability is narrower. ``json.dumps`` (PGTableGraphStorage's jsonb
    # column) and ``bson`` (MongoGraphStorage) both refuse ``np.float32`` /
    # ``np.int64`` / ``np.uint32`` / ``np.bool_``, so a numpy value that persists
    # fine on NetworkX cannot cross to another backend.
    value_type = type(value)
    if value_type not in _PORTABLE_SCALAR_TYPES:
        return (
            f"must be a str, int, float or bool to be portable across graph "
            f"backends, got {value_type.__name__}"
        )
    # The type is pinned exactly now, so ``is int`` rather than ``isinstance`` --
    # a bool cannot reach the range check.
    if value_type is int:
        if not _GRAPH_ATTRIBUTE_INT_MIN <= value <= _GRAPH_ATTRIBUTE_INT_MAX:
            return (
                "must be a 64-bit integer "
                f"({_GRAPH_ATTRIBUTE_INT_MIN} to {_GRAPH_ATTRIBUTE_INT_MAX})"
            )
        return None
    if value_type is float and not math.isfinite(value):
        return "must be a finite number"
    return None


def validate_graph_attributes(attributes: dict[str, Any], *, context: str) -> None:
    """Reject a node/edge attribute mapping no graph backend could store.

    Args:
        attributes: Attribute mapping about to be written to graph storage.
        context: Prefix identifying the object being written, used in the error
            message (e.g. ``"entity 'Tesla'"``).

    Raises:
        ValueError: On the first unusable attribute name or value.
    """
    # The portable contract is the intersection, so it owns both name rules: a
    # name XML cannot encode is unstorable on NetworkX, and a name MongoDB would
    # read as a field path is unstorable there. Neither backend needs the other's
    # rule (see each rejection helper), but caller input has to satisfy both.
    validate_xml_attributes(attributes, context=context)
    validate_interpreted_attribute_names(attributes, context=context)
    validate_graph_attribute_values(attributes, context=context)


def validate_interpreted_attribute_names(
    attributes: dict[str, Any], *, context: str
) -> None:
    """Reject attribute names a backend would read as something other than a name.

    The *interpretation* rule, not the serialization one: see
    ``xml_attribute_name_rejection`` for the character rule a GraphML-backed
    store needs. A backend should enforce the hazard it actually has, and the two
    hazards live in different backends:

    * Names are dangerous only where they are *interpreted*. MongoDB passes them
      into a ``$set`` document, so a dot is a path (``source_ids.0`` rewrites an
      element of the chunk-attribution array instead of creating a field) and a
      leading ``$`` is an update operator.
    * Names are inert in GraphML, which is why NetworkXStorage validates values
      only. Enforcing names there would additionally reject re-ingesting a node
      whose *stored* attributes predate this validation, which is a migration
      failure with no safety benefit.

    The rule is the interpretation hazard and nothing beyond it -- see
    ``_GRAPH_ATTRIBUTE_NAME_DISALLOWED_SUBSTRING`` for why "must be an
    identifier" would be the wrong rule. A name that a backend *stores* rather
    than interprets is legal here even if it is unusual, because rewrite paths
    feed stored attributes back into the upsert payload and a stricter rule
    would strand existing data.

    Args:
        attributes: Attribute mapping about to be written to graph storage.
        context: Prefix identifying the object being written.

    Raises:
        ValueError: On the first unusable attribute name.
    """
    for key in attributes:
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{context}: invalid attribute name {key!r}: must be a non-empty string"
            )
        if _GRAPH_ATTRIBUTE_NAME_DISALLOWED_SUBSTRING in key:
            raise ValueError(
                f"{context}: invalid attribute name {key!r}: must not contain "
                f"{_GRAPH_ATTRIBUTE_NAME_DISALLOWED_SUBSTRING!r}, which is read "
                "as a field path rather than part of the name"
            )
        if key.startswith(_GRAPH_ATTRIBUTE_NAME_DISALLOWED_PREFIX):
            raise ValueError(
                f"{context}: invalid attribute name {key!r}: must not start with "
                f"{_GRAPH_ATTRIBUTE_NAME_DISALLOWED_PREFIX!r}, which is read as "
                "an update operator"
            )


def validate_xml_attributes(attributes: dict[str, Any], *, context: str) -> None:
    """Reject attribute names or values XML cannot encode.

    For a GraphML-backed store. Names are checked as well as values because
    GraphML serializes them into the XML ``attr.name`` field, so an unencodable
    name breaks the same write. See ``xml_attribute_name_rejection`` and
    ``xml_attribute_value_rejection`` for why both rules are narrower than the
    portable contract.

    Args:
        attributes: Attribute mapping about to be written to graph storage.
        context: Prefix identifying the object being written.

    Raises:
        ValueError: On the first name or value XML cannot carry.
    """
    for key, value in attributes.items():
        rejection = xml_attribute_name_rejection(key)
        if rejection is not None:
            raise ValueError(f"{context}: attribute name {key!r} {rejection}")
        rejection = xml_attribute_value_rejection(value)
        if rejection is not None:
            raise ValueError(f"{context}: attribute {key!r} {rejection}")


def validate_graph_attribute_values(
    attributes: dict[str, Any], *, context: str
) -> None:
    """Reject attribute values no graph backend can store.

    See ``graph_attribute_value_rejection`` for the rules and
    ``validate_interpreted_attribute_names`` for why the halves are separate.

    Args:
        attributes: Attribute mapping about to be written to graph storage.
        context: Prefix identifying the object being written.

    Raises:
        ValueError: On the first unstorable value.
    """
    for key, value in attributes.items():
        rejection = graph_attribute_value_rejection(value)
        if rejection is not None:
            raise ValueError(f"{context}: attribute {key!r} {rejection}")
