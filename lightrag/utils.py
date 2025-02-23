from __future__ import annotations

import asyncio
import html
import io
import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Callable
import xml.etree.ElementTree as ET
import numpy as np
import tiktoken
from lightrag.prompt import PROMPTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


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
            formatted_msg[:50] + "..." if len(formatted_msg) > 50 else formatted_msg
        )
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    """Enable or disable verbose debug output"""
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


class UnlimitedSemaphore:
    """A context manager that allows unlimited access."""

    async def __aenter__(self):
        pass

    async def __aexit__(self, exc_type, exc, tb):
        pass


ENCODER = None

statistic_data = {"llm_call": 0, "llm_cache": 0, "embed_call": 0}

logger = logging.getLogger("lightrag")

# Set httpx logging level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)


def set_logger(log_file: str, level: int = logging.DEBUG):
    """Set up file logging with the specified level.

    Args:
        log_file: Path to the log file
        level: Logging level (e.g. logging.DEBUG, logging.INFO)
    """
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


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


def limit_async_func_call(max_size: int):
    """Add restriction of maximum concurrent async calls using asyncio.Semaphore"""

    def final_decro(func):
        sem = asyncio.Semaphore(max_size)

        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with sem:
                result = await func(*args, **kwargs)
                return result

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


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
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
    list_data: list[Any], key: Callable[[Any], str], max_token_size: int
) -> list[int]:
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(
        output,
        quoting=csv.QUOTE_ALL,  # Quote all fields
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
        lineterminator="\n",  # Explicit line terminator
    )
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> list[list[str]]:
    # Clean the string by removing NUL characters
    cleaned_string = csv_string.replace("\0", "")

    output = io.StringIO(cleaned_string)
    reader = csv.reader(
        output,
        quoting=csv.QUOTE_ALL,  # Match the writer configuration
        escapechar="\\",  # Use backslash as escape character
        quotechar='"',  # Use double quotes
    )

    try:
        return [row for row in reader]
    except csv.Error as e:
        raise ValueError(f"Failed to parse CSV string: {str(e)}")
    finally:
        output.close()


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


def process_combine_contexts(hl: str, ll: str):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources = []
    seen = set()

    for item in list_hl + list_ll:
        if item and item not in seen:
            combined_sources.append(item)
            seen.add(item)

    combined_sources_result = [",\t".join(header)]

    for i, item in enumerate(combined_sources, start=1):
        combined_sources_result.append(f"{i},\t{item}")

    combined_sources_result = "\n".join(combined_sources_result)

    return combined_sources_result


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

        if cache_data["embedding"] is None:
            continue

        # Convert cached embedding list to ndarray
        cached_quantized = np.frombuffer(
            bytes.fromhex(cache_data["embedding"]), dtype=np.uint8
        ).reshape(cache_data["embedding_shape"])
        cached_embedding = dequantize_embedding(
            cached_quantized,
            cache_data["embedding_min"],
            cache_data["embedding_max"],
        )

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

    # Quantize to 0-255 range
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((embedding - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize_embedding(
    quantized: np.ndarray, min_val: float, max_val: float, bits=8
) -> np.ndarray:
    """Restore quantized embedding"""
    scale = (max_val - min_val) / (2**bits - 1)
    return (quantized * scale + min_val).astype(np.float32)


async def handle_cache(
    hashing_kv,
    args_hash,
    prompt,
    mode="default",
    cache_type=None,
    force_llm_cache=False,
):
    """Generic cache handling function"""
    if hashing_kv is None or not (
        force_llm_cache or hashing_kv.global_config.get("enable_llm_cache")
    ):
        return None, None, None, None

    if mode != "default":
        # Get embedding cache configuration
        embedding_cache_config = hashing_kv.global_config.get(
            "embedding_cache_config",
            {"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False},
        )
        is_embedding_cache_enabled = embedding_cache_config["enabled"]
        use_llm_check = embedding_cache_config.get("use_llm_check", False)

        quantized = min_val = max_val = None
        if is_embedding_cache_enabled:
            # Use embedding cache
            current_embedding = await hashing_kv.embedding_func([prompt])
            llm_model_func = hashing_kv.global_config.get("llm_model_func")
            quantized, min_val, max_val = quantize_embedding(current_embedding[0])
            best_cached_response = await get_best_cached_response(
                hashing_kv,
                current_embedding[0],
                similarity_threshold=embedding_cache_config["similarity_threshold"],
                mode=mode,
                use_llm_check=use_llm_check,
                llm_func=llm_model_func if use_llm_check else None,
                original_prompt=prompt,
                cache_type=cache_type,
            )
            if best_cached_response is not None:
                logger.info(f"Embedding cached hit(mode:{mode} type:{cache_type})")
                return best_cached_response, None, None, None
            else:
                # if caching keyword embedding is enabled, return the quantized embedding for saving it latter
                logger.info(f"Embedding cached missed(mode:{mode} type:{cache_type})")
                return None, quantized, min_val, max_val

    # For default mode or is_embedding_cache_enabled is False, use regular cache
    # default mode is for extract_entities or naive query
    if exists_func(hashing_kv, "get_by_mode_and_id"):
        mode_cache = await hashing_kv.get_by_mode_and_id(mode, args_hash) or {}
    else:
        mode_cache = await hashing_kv.get_by_id(mode) or {}
    if args_hash in mode_cache:
        logger.info(f"Non-embedding cached hit(mode:{mode} type:{cache_type})")
        return mode_cache[args_hash]["return"], None, None, None

    logger.info(f"Non-embedding cached missed(mode:{mode} type:{cache_type})")
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
    if hashing_kv is None or hasattr(cache_data.content, "__aiter__"):
        return

    if exists_func(hashing_kv, "get_by_mode_and_id"):
        mode_cache = (
            await hashing_kv.get_by_mode_and_id(cache_data.mode, cache_data.args_hash)
            or {}
        )
    else:
        mode_cache = await hashing_kv.get_by_id(cache_data.mode) or {}

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
