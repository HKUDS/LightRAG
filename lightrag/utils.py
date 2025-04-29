from __future__ import annotations
import weakref

import asyncio
import html
import csv
import json
import logging
import logging.handlers
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Protocol, Callable, TYPE_CHECKING, List
import xml.etree.ElementTree as ET
import numpy as np
from lightrag.prompt import PROMPTS
from dotenv import load_dotenv

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from lightrag.base import BaseKVStorage

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"


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
            formatted_msg[:100] + "..." if len(formatted_msg) > 100 else formatted_msg
        )
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    """Enable or disable verbose debug output"""
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}

# Initialize logger
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root loggger
# Let the main application configure the handlers
logger.setLevel(logging.INFO)

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""

    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = [
            "/documents",
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

            # Filter out successful GET requests to filtered paths
            if (
                method == "GET"
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

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(level)
    logger_instance.addHandler(console_handler)

    # Add file handler by default unless explicitly disabled
    if enable_file_logging:
        # Get log file path
        if log_file_path is None:
            log_dir = os.getenv("LOG_DIR", os.getcwd())
            log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag.log"))

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Get log file max size and backup count from environment variables
        log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
        log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

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
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    # concurrent_limit: int = 16

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def locate_json_string_body_from_string(content: str) -> str | None:
    """Locate the JSON string body from a string"""
    try:
        maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
        if maybe_json_str is not None:
            maybe_json_str = maybe_json_str.group(0)
            maybe_json_str = maybe_json_str.replace("\\n", "")
            maybe_json_str = maybe_json_str.replace("\n", "")
            maybe_json_str = maybe_json_str.replace("'", '"')
            # json.loads(maybe_json_str) # don't check here, cannot validate schema after all
            return maybe_json_str
    except Exception:
        pass
        # try:
        #     content = (
        #         content.replace(kw_prompt[:-1], "")
        #         .replace("user", "")
        #         .replace("model", "")
        #         .strip()
        #     )
        #     maybe_json_str = "{" + content.split("{")[1].split("}")[0] + "}"
        #     json.loads(maybe_json_str)

        return None


def convert_response_to_json(response: str) -> dict[str, Any]:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args: Any, cache_type: str | None = None) -> str:
    """Compute a hash for the given arguments.
    Args:
        *args: Arguments to hash
        cache_type: Type of cache (e.g., 'keywords', 'query', 'extract')
    Returns:
        str: Hash string
    """
    import hashlib

    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])
    if cache_type:
        args_str = f"{cache_type}:{args_str}"

    # Compute MD5 hash
    return hashlib.md5(args_str.encode()).hexdigest()


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute a unique ID for a given content string.

    The ID is a combination of the given prefix and the MD5 hash of the content string.
    """
    return prefix + md5(content.encode()).hexdigest()


# Custom exception class
class QueueFullError(Exception):
    """Raised when the queue is full and the wait times out"""

    pass


def priority_limit_async_func_call(max_size: int, max_queue_size: int = 1000):
    """
    Enhanced priority-limited asynchronous function call decorator

    Args:
        max_size: Maximum number of concurrent calls
        max_queue_size: Maximum queue capacity to prevent memory overflow
    Returns:
        Decorator function
    """

    def final_decro(func):
        queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        tasks = set()
        initialization_lock = asyncio.Lock()
        counter = 0
        shutdown_event = asyncio.Event()
        initialized = False  # Global initialization flag
        worker_health_check_task = None

        # Track active future objects for cleanup
        active_futures = weakref.WeakSet()
        reinit_count = 0  # Reinitialization counter to track system health

        # Worker function to process tasks in the queue
        async def worker():
            """Worker that processes tasks in the priority queue"""
            try:
                while not shutdown_event.is_set():
                    try:
                        # Use timeout to get tasks, allowing periodic checking of shutdown signal
                        try:
                            (
                                priority,
                                count,
                                future,
                                args,
                                kwargs,
                            ) = await asyncio.wait_for(queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            # Timeout is just to check shutdown signal, continue to next iteration
                            continue

                        # If future is cancelled, skip execution
                        if future.cancelled():
                            queue.task_done()
                            continue

                        try:
                            # Execute function
                            result = await func(*args, **kwargs)
                            # If future is not done, set the result
                            if not future.done():
                                future.set_result(result)
                        except asyncio.CancelledError:
                            if not future.done():
                                future.cancel()
                            logger.debug("limit_async: Task cancelled during execution")
                        except Exception as e:
                            logger.error(
                                f"limit_async: Error in decorated function: {str(e)}"
                            )
                            if not future.done():
                                future.set_exception(e)
                        finally:
                            queue.task_done()
                    except Exception as e:
                        # Catch all exceptions in worker loop to prevent worker termination
                        logger.error(f"limit_async: Critical error in worker: {str(e)}")
                        await asyncio.sleep(0.1)  # Prevent high CPU usage
            finally:
                logger.warning("limit_async: Worker exiting")

        async def health_check():
            """Periodically check worker health status and recover"""
            nonlocal initialized
            try:
                while not shutdown_event.is_set():
                    await asyncio.sleep(5)  # Check every 5 seconds

                    # No longer acquire lock, directly operate on task set
                    # Use a copy of the task set to avoid concurrent modification
                    current_tasks = set(tasks)
                    done_tasks = {t for t in current_tasks if t.done()}
                    tasks.difference_update(done_tasks)

                    # Calculate active tasks count
                    active_tasks_count = len(tasks)
                    workers_needed = max_size - active_tasks_count

                    if workers_needed > 0:
                        logger.info(
                            f"limit_async: Creating {workers_needed} new workers"
                        )
                        new_tasks = set()
                        for _ in range(workers_needed):
                            task = asyncio.create_task(worker())
                            new_tasks.add(task)
                            task.add_done_callback(tasks.discard)
                        # Update task set in one operation
                        tasks.update(new_tasks)
            except Exception as e:
                logger.error(f"limit_async: Error in health check: {str(e)}")
            finally:
                logger.warning("limit_async: Health check task exiting")
                initialized = False

        async def ensure_workers():
            """Ensure worker threads and health check system are available

            This function checks if the worker system is already initialized.
            If not, it performs a one-time initialization of all worker threads
            and starts the health check system.
            """
            nonlocal initialized, worker_health_check_task, tasks, reinit_count

            if initialized:
                return

            async with initialization_lock:
                if initialized:
                    return

                # Increment reinitialization counter if this is not the first initialization
                if reinit_count > 0:
                    reinit_count += 1
                    logger.warning(
                        f"limit_async: Reinitializing needed (count: {reinit_count})"
                    )
                else:
                    reinit_count = 1  # First initialization

                # Check for completed tasks and remove them from the task set
                current_tasks = set(tasks)
                done_tasks = {t for t in current_tasks if t.done()}
                tasks.difference_update(done_tasks)

                # Log active tasks count during reinitialization
                active_tasks_count = len(tasks)
                if active_tasks_count > 0 and reinit_count > 1:
                    logger.warning(
                        f"limit_async: {active_tasks_count} tasks still running during reinitialization"
                    )

                # Create initial worker tasks, only adding the number needed
                workers_needed = max_size - active_tasks_count
                for _ in range(workers_needed):
                    task = asyncio.create_task(worker())
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

                # Start health check
                worker_health_check_task = asyncio.create_task(health_check())

                initialized = True
                logger.info(f"limit_async: {workers_needed} new workers initialized")

        async def shutdown():
            """Gracefully shut down all workers and the queue"""
            logger.info("limit_async: Shutting down priority queue workers")

            # Set the shutdown event
            shutdown_event.set()

            # Cancel all active futures
            for future in list(active_futures):
                if not future.done():
                    future.cancel()

            # Wait for the queue to empty
            try:
                await asyncio.wait_for(queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "limit_async: Timeout waiting for queue to empty during shutdown"
                )

            # Cancel all worker tasks
            for task in list(tasks):
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # Cancel the health check task
            if worker_health_check_task and not worker_health_check_task.done():
                worker_health_check_task.cancel()
                try:
                    await worker_health_check_task
                except asyncio.CancelledError:
                    pass

            logger.info("limit_async: Priority queue workers shutdown complete")

        @wraps(func)
        async def wait_func(
            *args, _priority=10, _timeout=None, _queue_timeout=None, **kwargs
        ):
            """
            Execute the function with priority-based concurrency control
            Args:
                *args: Positional arguments passed to the function
                _priority: Call priority (lower values have higher priority)
                _timeout: Maximum time to wait for function completion (in seconds)
                _queue_timeout: Maximum time to wait for entering the queue (in seconds)
                **kwargs: Keyword arguments passed to the function
            Returns:
                The result of the function call
            Raises:
                TimeoutError: If the function call times out
                QueueFullError: If the queue is full and waiting times out
                Any exception raised by the decorated function
            """
            # Ensure worker system is initialized
            await ensure_workers()

            # Create a future for the result
            future = asyncio.Future()
            active_futures.add(future)

            nonlocal counter
            async with initialization_lock:
                current_count = counter  # Use local variable to avoid race conditions
                counter += 1

            # Try to put the task into the queue, supporting timeout
            try:
                if _queue_timeout is not None:
                    # Use timeout to wait for queue space
                    try:
                        await asyncio.wait_for(
                            # current_count is used to ensure FIFO order
                            queue.put((_priority, current_count, future, args, kwargs)),
                            timeout=_queue_timeout,
                        )
                    except asyncio.TimeoutError:
                        raise QueueFullError(
                            f"Queue full, timeout after {_queue_timeout} seconds"
                        )
                else:
                    # No timeout, may wait indefinitely
                    # current_count is used to ensure FIFO order
                    await queue.put((_priority, current_count, future, args, kwargs))
            except Exception as e:
                # Clean up the future
                if not future.done():
                    future.set_exception(e)
                active_futures.discard(future)
                raise

            try:
                # Wait for the result, optional timeout
                if _timeout is not None:
                    try:
                        return await asyncio.wait_for(future, _timeout)
                    except asyncio.TimeoutError:
                        # Cancel the future
                        if not future.done():
                            future.cancel()
                        raise TimeoutError(
                            f"limit_async: Task timed out after {_timeout} seconds"
                        )
                else:
                    # Wait for the result without timeout
                    return await future
            finally:
                # Clean up the future reference
                active_futures.discard(future)

        # Add the shutdown method to the decorated function
        wait_func.shutdown = shutdown

        return wait_func

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


class TokenizerInterface(Protocol):
    """
    Defines the interface for a tokenizer, requiring encode and decode methods.
    """

    def encode(self, content: str) -> List[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """
    A wrapper around a tokenizer to provide a consistent interface for encoding and decoding.
    """

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> List[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        return self.tokenizer.encode(content)

    def decode(self, tokens: List[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)


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


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_json(data: list[list[str]]) -> list[dict[str, str]]:
    if not data or len(data) <= 1:
        return []

    header = data[0]
    result = []

    for row in data[1:]:
        if len(row) >= 2:
            item = {}
            for i, field_name in enumerate(header):
                if i < len(row):
                    item[field_name] = str(row[i])
                else:
                    item[field_name] = ""
            result.append(item)

    return result


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(
    hl_context: list[dict[str, str]], ll_context: list[dict[str, str]]
):
    seen_content = {}
    combined_data = []

    for item in hl_context + ll_context:
        content_dict = {k: v for k, v in item.items() if k != "id"}
        content_key = tuple(sorted(content_dict.items()))
        if content_key not in seen_content:
            seen_content[content_key] = item
            combined_data.append(item)

    for i, item in enumerate(combined_data):
        item["id"] = str(i)

    return combined_data


async def get_best_cached_response(
    hashing_kv,
    current_embedding,
    similarity_threshold=0.95,
    mode="default",
    use_llm_check=False,
    llm_func=None,
    original_prompt=None,
    cache_type=None,
) -> str | None:
    logger.debug(
        f"get_best_cached_response:  mode={mode} cache_type={cache_type} use_llm_check={use_llm_check}"
    )
    mode_cache = await hashing_kv.get_by_id(mode)
    if not mode_cache:
        return None

    best_similarity = -1
    best_response = None
    best_prompt = None
    best_cache_id = None

    # Only iterate through cache entries for this mode
    for cache_id, cache_data in mode_cache.items():
        # Skip if cache_type doesn't match
        if cache_type and cache_data.get("cache_type") != cache_type:
            continue

        # Check if cache data is valid
        if cache_data["embedding"] is None:
            continue

        try:
            # Safely convert cached embedding
            cached_quantized = np.frombuffer(
                bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
            ).reshape(cache_data["embedding_shape"])

            # Ensure min_val and max_val are valid float values
            embedding_min = cache_data.get("embedding_min")
            embedding_max = cache_data.get("embedding_max")

            if (
                embedding_min is None
                or embedding_max is None
                or embedding_min >= embedding_max
            ):
                logger.warning(
                    f"Invalid embedding min/max values: min={embedding_min}, max={embedding_max}"
                )
                continue

            cached_embedding = dequantize_embedding(
                cached_quantized,
                embedding_min,
                embedding_max,
            )
        except Exception as e:
            logger.warning(f"Error processing cached embedding: {str(e)}")
            continue

        similarity = cosine_similarity(current_embedding, cached_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_response = cache_data["return"]
            best_prompt = cache_data["original_prompt"]
            best_cache_id = cache_id

    if best_similarity > similarity_threshold:
        # If LLM check is enabled and all required parameters are provided
        if (
            use_llm_check
            and llm_func
            and original_prompt
            and best_prompt
            and best_response is not None
        ):
            compare_prompt = PROMPTS["similarity_check"].format(
                original_prompt=original_prompt, cached_prompt=best_prompt
            )

            try:
                llm_result = await llm_func(compare_prompt)
                llm_result = llm_result.strip()
                llm_similarity = float(llm_result)

                # Replace vector similarity with LLM similarity score
                best_similarity = llm_similarity
                if best_similarity < similarity_threshold:
                    log_data = {
                        "event": "cache_rejected_by_llm",
                        "type": cache_type,
                        "mode": mode,
                        "original_question": original_prompt[:100] + "..."
                        if len(original_prompt) > 100
                        else original_prompt,
                        "cached_question": best_prompt[:100] + "..."
                        if len(best_prompt) > 100
                        else best_prompt,
                        "similarity_score": round(best_similarity, 4),
                        "threshold": similarity_threshold,
                    }
                    logger.debug(json.dumps(log_data, ensure_ascii=False))
                    logger.info(f"Cache rejected by LLM(mode:{mode} tpye:{cache_type})")
                    return None
            except Exception as e:  # Catch all possible exceptions
                logger.warning(f"LLM similarity check failed: {e}")
                return None  # Return None directly when LLM check fails

        prompt_display = (
            best_prompt[:50] + "..." if len(best_prompt) > 50 else best_prompt
        )
        log_data = {
            "event": "cache_hit",
            "type": cache_type,
            "mode": mode,
            "similarity": round(best_similarity, 4),
            "cache_id": best_cache_id,
            "original_prompt": prompt_display,
        }
        logger.debug(json.dumps(log_data, ensure_ascii=False))
        return best_response
    return None


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def quantize_embedding(embedding: np.ndarray | list[float], bits: int = 8) -> tuple:
    """Quantize embedding to specified bits"""
    # Convert list to numpy array if needed
    if isinstance(embedding, list):
        embedding = np.array(embedding)

    # Calculate min/max values for reconstruction
    min_val = embedding.min()
    max_val = embedding.max()

    if min_val == max_val:
        # handle constant vector
        quantized = np.zeros_like(embedding, dtype=np.uint8)
        return quantized, min_val, max_val

    # Quantize to 0-255 range
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    if min_val == max_val:
        # handle constant vector
        return np.full_like(quantized, min_val, dtype=np.float32)

    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type=None,
):
    """Generic cache handling function"""
    if hashing_kv is None:
        return None, None, None, None

    if mode != "default":  # handle cache for all type of query
        if not hashing_kv.global_config.get("enable_llm_cache"):
            return None, None, None, None
    else:  # handle cache for entity extraction
        if not hashing_kv.global_config.get("enable_llm_cache_for_entity_extract"):
            return None, None, None, None

    if exists_func(hashing_kv, "get_by_mode_and_id"):
        mode_cache = await hashing_kv.get_by_mode_and_id(mode, args_hash) or {}
    else:
        mode_cache = await hashing_kv.get_by_id(mode) or {}
    if args_hash in mode_cache:
        logger.debug(f"Non-embedding cached hit(mode:{mode} type:{cache_type})")
        return mode_cache[args_hash]["return"], None, None, None

    logger.debug(f"Non-embedding cached missed(mode:{mode} type:{cache_type})")
    return None, None, None, None


@dataclass
class CacheData:
    args_hash: str
    content: str
    prompt: str
    quantized: np.ndarray | None = None
    min_val: float | None = None
    max_val: float | None = None
    mode: str = "default"
    cache_type: str = "query"


async def save_to_cache(hashing_kv, cache_data: CacheData):
    """Save data to cache, with improved handling for streaming responses and duplicate content.

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

    # Get existing cache data
    if exists_func(hashing_kv, "get_by_mode_and_id"):
        mode_cache = (
            await hashing_kv.get_by_mode_and_id(cache_data.mode, cache_data.args_hash)
            or {}
        )
    else:
        mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}

    # Check if we already have identical content cached
    if cache_data.args_hash in mode_cache:
        existing_content = mode_cache[cache_data.args_hash].get("return")
        if existing_content == cache_data.content:
            logger.info(
                f"Cache content unchanged for {cache_data.args_hash}, skipping update"
            )
            return

    # Update cache with new content
    mode_cache[cache_data.args_hash] = {
        "return": cache_data.content,
        "cache_type": cache_data.cache_type,
        "embedding": cache_data.quantized.tobytes().hex()
        if cache_data.quantized is not None
        else None,
        "embedding_shape": cache_data.quantized.shape
        if cache_data.quantized is not None
        else None,
        "embedding_min": cache_data.min_val,
        "embedding_max": cache_data.max_val,
        "original_prompt": cache_data.prompt,
    }

    logger.info(f" == LLM cache == saving {cache_data.mode}: {cache_data.args_hash}")

    # Only upsert if there's actual new content
    await hashing_kv.upsert({cache_data.mode: mode_cache})


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


def get_conversation_turns(
    conversation_history: list[dict[str, Any]], num_turns: int
) -> str:
    """
    Process conversation history to get the specified number of complete turns.

    Args:
        conversation_history: List of conversation messages in chronological order
        num_turns: Number of complete turns to include

    Returns:
        Formatted string of the conversation history
    """
    # Check if num_turns is valid
    if num_turns <= 0:
        return ""

    # Group messages into turns
    turns: list[list[dict[str, Any]]] = []
    messages: list[dict[str, Any]] = []

    # First, filter out keyword extraction messages
    for msg in conversation_history:
        if msg["role"] == "assistant" and (
            msg["content"].startswith('{ "high_level_keywords"')
            or msg["content"].startswith("{'high_level_keywords'")
        ):
            continue
        messages.append(msg)

    # Then process messages in chronological order
    i = 0
    while i < len(messages) - 1:
        msg1 = messages[i]
        msg2 = messages[i + 1]

        # Check if we have a user-assistant or assistant-user pair
        if (msg1["role"] == "user" and msg2["role"] == "assistant") or (
            msg1["role"] == "assistant" and msg2["role"] == "user"
        ):
            # Always put user message first in the turn
            if msg1["role"] == "assistant":
                turn = [msg2, msg1]  # user, assistant
            else:
                turn = [msg1, msg2]  # user, assistant
            turns.append(turn)
        i += 2

    # Keep only the most recent num_turns
    if len(turns) > num_turns:
        turns = turns[-num_turns:]

    # Format the turns into a string
    formatted_turns: list[str] = []
    for turn in turns:
        formatted_turns.extend(
            [f"user: {turn[0]['content']}", f"assistant: {turn[1]['content']}"]
        )

    return "\n".join(formatted_turns)


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
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
                    rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-")
                    vector_data = await relationships_vdb.get_by_id(rel_id)
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
            f"Unsupported file format: {file_format}. "
            f"Choose from: csv, excel, md, txt"
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


async def use_llm_func_with_cache(
    input_text: str,
    use_llm_func: callable,
    llm_response_cache: "BaseKVStorage | None" = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
) -> str:
    """Call LLM function with cache support

    If cache is available and enabled (determined by handle_cache based on mode),
    retrieve result from cache; otherwise call LLM function and save result to cache.

    Args:
        input_text: Input text to send to LLM
        use_llm_func: LLM function with higher priority
        llm_response_cache: Cache storage instance
        max_tokens: Maximum tokens for generation
        history_messages: History messages list
        cache_type: Type of cache

    Returns:
        LLM response text
    """
    if llm_response_cache:
        if history_messages:
            history = json.dumps(history_messages, ensure_ascii=False)
            _prompt = history + "\n" + input_text
        else:
            _prompt = input_text

        arg_hash = compute_args_hash(_prompt)
        cached_return, _1, _2, _3 = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
        )
        if cached_return:
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1
            return cached_return
        statistic_data["llm_call"] += 1

        # Call LLM
        kwargs = {}
        if history_messages:
            kwargs["history_messages"] = history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        res: str = await use_llm_func(input_text, **kwargs)

        if llm_response_cache.global_config.get("enable_llm_cache_for_entity_extract"):
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type=cache_type,
                ),
            )

        return res

    # When cache is disabled, directly call LLM
    kwargs = {}
    if history_messages:
        kwargs["history_messages"] = history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.info(f"Call LLM function with query text lenght: {len(input_text)}")
    return await use_llm_func(input_text, **kwargs)


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


def normalize_extracted_info(name: str, is_entity=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    1. Remove spaces between Chinese characters
    2. Remove spaces between Chinese characters and English letters/numbers
    3. Preserve spaces within English text and numbers
    4. Replace Chinese parentheses with English parentheses
    5. Replace Chinese dash with English dash

    Args:
        name: Entity name to normalize

    Returns:
        Normalized entity name
    """
    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash
    name = name.replace("—", "-").replace("－", "-")

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

    # Remove English quotation marks from the beginning and end
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    if len(name) >= 2 and name.startswith("'") and name.endswith("'"):
        name = name[1:-1]

    if is_entity:
        # remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)

    return name


def clean_text(text: str) -> str:
    """Clean text by removing null bytes (0x00) and whitespace

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    return text.strip().replace("\x00", "")


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
