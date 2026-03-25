"""Tests for entity extraction gleaning token limit guard."""

from dataclasses import dataclass, field
from itertools import count
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config(
    max_extract_input_tokens: int = 20480,
    entity_extract_max_gleaning: int = 1,
) -> dict:
    """Build a minimal global_config dict for extract_entities."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": AsyncMock(return_value=""),
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "addon_params": {},
        "tokenizer": tokenizer,
        "max_extract_input_tokens": max_extract_input_tokens,
        "llm_model_max_async": 1,
    }


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)


@dataclass
class _DummyLLMCache:
    global_config: dict = field(
        default_factory=lambda: {"enable_llm_cache_for_entity_extract": True}
    )
    records: dict[str, dict] = field(default_factory=dict)
    _clock: count = field(default_factory=lambda: count(1))

    async def get_by_id(self, key: str):
        return self.records.get(key)

    async def get_by_ids(self, keys: list[str]):
        return [self.records.get(key) for key in keys]

    async def upsert(self, data: dict[str, dict]):
        for key, value in data.items():
            cache_entry = dict(value)
            cache_entry.setdefault("create_time", next(self._clock))
            self.records[key] = cache_entry


@dataclass
class _DummyTextChunksStorage:
    chunks: dict[str, dict] = field(default_factory=dict)

    async def get_by_id(self, key: str):
        return self.chunks.get(key)

    async def get_by_ids(self, keys: list[str]):
        return [self.chunks.get(key) for key in keys]


class _NoopAsyncContext:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_chunks(content: str = "Test content.") -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_skipped_when_tokens_exceed_limit():
    """Gleaning should be skipped when estimated tokens exceed max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    # Use a very small token limit so the gleaning context will exceed it
    global_config = _make_global_config(
        max_extract_input_tokens=10,
        entity_extract_max_gleaning=1,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger") as mock_logger:
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called exactly once (initial extraction only, no gleaning)
    assert llm_func.await_count == 1
    # Warning should be logged about skipping gleaning
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "Gleaning stopped" in warning_msg
    assert "exceeded limit" in warning_msg


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_proceeds_when_tokens_within_limit():
    """Gleaning should proceed when estimated tokens are within max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    # Use a very large token limit so gleaning will proceed
    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=1,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called twice (initial extraction + gleaning)
    assert llm_func.await_count == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_gleaning_when_max_gleaning_zero():
    """No gleaning when entity_extract_max_gleaning is 0, regardless of token limit."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=0,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called exactly once (initial extraction only)
    assert llm_func.await_count == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_uses_instance_prompt_config_templates():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=1,
    )
    global_config["prompt_config"] = {
        "entity_extraction": {
            "system_prompt": "SYS {tuple_delimiter} {completion_delimiter} {examples}",
            "user_prompt": "USR {input_text}",
            "continue_prompt": "CONT {tuple_delimiter} {completion_delimiter}",
            "examples": ["EXAMPLE-LINE"],
        }
    }

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(content="Prompt body."),
            global_config=global_config,
        )

    assert llm_func.await_count == 2
    first_call = llm_func.await_args_list[0]
    second_call = llm_func.await_args_list[1]
    assert "USR Prompt body." in first_call.args[0]
    assert "SYS" in first_call.kwargs["system_prompt"]
    assert "EXAMPLE-LINE" in first_call.kwargs["system_prompt"]
    assert "CONT" in second_call.args[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_cache_identity_changes_when_indexing_prompt_fingerprint_changes():
    from lightrag.operate import extract_entities

    config_a = _make_global_config(entity_extract_max_gleaning=0)
    config_b = _make_global_config(entity_extract_max_gleaning=0)

    for cfg in (config_a, config_b):
        cfg["llm_model_func"].return_value = _EXTRACTION_RESULT
        cfg["prompt_config"] = {
            "entity_extraction": {
                "system_prompt": "SYS {tuple_delimiter} {completion_delimiter}",
                "user_prompt": "USR {input_text}",
                "continue_prompt": "CONT {tuple_delimiter} {completion_delimiter}",
                "examples": ["EXAMPLE-A"],
            }
        }

    # Only examples differ; prompt templates above do not reference examples.
    # Cache identity should still change once prompt fingerprint is included.
    config_b["prompt_config"]["entity_extraction"]["examples"] = ["EXAMPLE-B"]

    cache_a = _DummyLLMCache()
    cache_b = _DummyLLMCache()

    await extract_entities(
        chunks=_make_chunks(content="Same input."),
        global_config=config_a,
        llm_response_cache=cache_a,
    )
    await extract_entities(
        chunks=_make_chunks(content="Same input."),
        global_config=config_b,
        llm_response_cache=cache_b,
    )

    key_a = next(iter(cache_a.records))
    key_b = next(iter(cache_b.records))
    assert key_a != key_b


@pytest.mark.offline
@pytest.mark.asyncio
async def test_summary_uses_instance_prompt_config_template():
    from lightrag.operate import _summarize_descriptions

    global_config = _make_global_config()
    global_config["summary_length_recommended"] = 64
    global_config["summary_context_size"] = 4096
    global_config["prompt_config"] = {
        "summary": {
            "summarize_entity_descriptions": (
                "SUM {description_type}:{description_name}\n{description_list}"
            )
        }
    }
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = "summary-output"

    with patch("lightrag.operate.logger"):
        result = await _summarize_descriptions(
            "entity",
            "DemoEntity",
            ["desc one", "desc two"],
            global_config,
        )

    assert result == "summary-output"
    llm_func.assert_awaited()
    assert "SUM entity:DemoEntity" in llm_func.await_args.args[0]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_summary_cache_identity_changes_with_summary_shared_prompt_fingerprint():
    from lightrag.operate import _summarize_descriptions

    config_a = _make_global_config()
    config_b = _make_global_config()
    for cfg in (config_a, config_b):
        cfg["summary_length_recommended"] = 64
        cfg["summary_context_size"] = 4096
        cfg["prompt_config"] = {
            "summary": {
                "summarize_entity_descriptions": (
                    "SUM {description_type}:{description_name}\n{description_list}"
                )
            },
            "shared": {
                "tuple_delimiter": "<|#|>",
                "completion_delimiter": "<|COMPLETE|>",
            },
        }
        cfg["llm_model_func"].return_value = "summary-output"

    # Prompt text is unchanged; only shared delimiter differs.
    # Summary cache identity should still change due to fingerprint coverage.
    config_b["prompt_config"]["shared"]["tuple_delimiter"] = "<|@@|>"

    cache_a = _DummyLLMCache()
    cache_b = _DummyLLMCache()

    await _summarize_descriptions(
        "entity", "DemoEntity", ["desc"], config_a, llm_response_cache=cache_a
    )
    await _summarize_descriptions(
        "entity", "DemoEntity", ["desc"], config_b, llm_response_cache=cache_b
    )

    key_a = next(iter(cache_a.records))
    key_b = next(iter(cache_b.records))
    assert key_a != key_b


@pytest.mark.offline
@pytest.mark.asyncio
async def test_indexing_prompt_change_warning_logged_once_per_global_config():
    import lightrag.operate as operate_module

    extract_entities = operate_module.extract_entities
    operate_module._INDEXING_PROMPT_WARNED_FINGERPRINTS.clear()

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=0,
    )
    global_config["prompt_config"] = {
        "entity_extraction": {
            "system_prompt": "SYS {tuple_delimiter} {completion_delimiter} {examples}",
            "user_prompt": "USR {input_text}",
            "continue_prompt": "CONT {tuple_delimiter} {completion_delimiter}",
            "examples": ["EXAMPLE-WARN"],
        }
    }
    global_config["llm_model_func"].return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger") as mock_logger:
        await extract_entities(_make_chunks(), global_config)
        await extract_entities(_make_chunks(), global_config)

    warning_messages = [
        call.args[0]
        for call in mock_logger.warning.call_args_list
        if call.args and isinstance(call.args[0], str)
    ]
    prompt_warnings = [
        msg for msg in warning_messages if "Indexing-time prompt_config has changed" in msg
    ]
    assert len(prompt_warnings) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_from_cached_extraction_accepts_custom_shared_delimiters():
    from lightrag.operate import _rebuild_from_extraction_result

    text_chunks_storage = _DummyTextChunksStorage(
        chunks={"chunk-001": {"file_path": "custom.txt"}}
    )

    custom_tuple_delimiter = "<|@@|>"
    custom_completion_delimiter = "<|DONE|>"
    extraction_result = (
        f"(entity{custom_tuple_delimiter}TEST_ENTITY{custom_tuple_delimiter}CONCEPT"
        f"{custom_tuple_delimiter}A test entity){custom_completion_delimiter}"
    )

    maybe_nodes, maybe_edges = await _rebuild_from_extraction_result(
        text_chunks_storage=text_chunks_storage,
        extraction_result=extraction_result,
        chunk_id="chunk-001",
        timestamp=1,
        tuple_delimiter=custom_tuple_delimiter,
        completion_delimiter=custom_completion_delimiter,
    )

    assert "TEST_ENTITY" in maybe_nodes
    assert maybe_edges == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_cache_stores_generation_delimiter_metadata():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(entity_extract_max_gleaning=0)
    global_config["prompt_config"] = {
        "shared": {
            "tuple_delimiter": "<|@@|>",
            "completion_delimiter": "<|DONE|>",
        },
        "entity_extraction": {
            "system_prompt": (
                "SYS {tuple_delimiter} {completion_delimiter} {examples}"
            ),
            "user_prompt": "USR {input_text}",
            "continue_prompt": "CONT {tuple_delimiter} {completion_delimiter}",
            "examples": ["entity<|@@|>A<|@@|>T<|@@|>D"],
        },
    }

    global_config["llm_model_func"].return_value = (
        "(entity<|@@|>TEST_ENTITY<|@@|>CONCEPT<|@@|>A test entity)<|DONE|>"
    )
    cache = _DummyLLMCache()

    await extract_entities(
        chunks=_make_chunks(content="Meta test"),
        global_config=global_config,
        llm_response_cache=cache,
    )

    saved_entry = next(iter(cache.records.values()))
    assert saved_entry["queryparam"]["tuple_delimiter"] == "<|@@|>"
    assert saved_entry["queryparam"]["completion_delimiter"] == "<|DONE|>"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_prefers_cached_delimiter_metadata_over_current_prompt_config():
    import lightrag.operate as operate_module

    old_tuple = "<|OLD|>"
    old_completion = "<|OLD_DONE|>"
    new_tuple = "<|NEW|>"
    new_completion = "<|NEW_DONE|>"

    cache_key = "default:extract:abc"
    llm_cache = _DummyLLMCache(
        records={
            cache_key: {
                "cache_type": "extract",
                "chunk_id": "chunk-001",
                "return": (
                    f"(entity{old_tuple}TEST_ENTITY{old_tuple}CONCEPT"
                    f"{old_tuple}desc){old_completion}"
                ),
                "queryparam": {
                    "tuple_delimiter": old_tuple,
                    "completion_delimiter": old_completion,
                },
                "create_time": 1,
            }
        }
    )
    text_chunks_storage = _DummyTextChunksStorage(
        chunks={"chunk-001": {"llm_cache_list": [cache_key], "file_path": "a.txt"}}
    )

    captured: list[tuple[str, str]] = []

    async def _fake_rebuild_from_result(*args, **kwargs):
        captured.append((kwargs["tuple_delimiter"], kwargs["completion_delimiter"]))
        return {}, {}

    with (
        patch.object(
            operate_module,
            "_rebuild_from_extraction_result",
            side_effect=_fake_rebuild_from_result,
        ),
        patch.object(operate_module, "_rebuild_single_entity", new=AsyncMock()),
        patch.object(operate_module, "_rebuild_single_relationship", new=AsyncMock()),
        patch.object(
            operate_module,
            "get_storage_keyed_lock",
            return_value=_NoopAsyncContext(),
        ),
    ):
        await operate_module.rebuild_knowledge_from_chunks(
            entities_to_rebuild={"TEST_ENTITY": ["chunk-001"]},
            relationships_to_rebuild={},
            knowledge_graph_inst=object(),
            entities_vdb=object(),
            relationships_vdb=object(),
            text_chunks_storage=text_chunks_storage,
            llm_response_cache=llm_cache,
            global_config={
                "llm_model_max_async": 1,
                "prompt_config": {
                    "shared": {
                        "tuple_delimiter": new_tuple,
                        "completion_delimiter": new_completion,
                    }
                },
            },
        )

    assert captured
    assert captured[0] == (old_tuple, old_completion)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_legacy_cache_without_metadata_falls_back_to_historical_defaults():
    import lightrag.operate as operate_module
    from lightrag.prompt import PROMPTS

    default_tuple = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    default_completion = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    cache_key = "default:extract:legacy"
    llm_cache = _DummyLLMCache(
        records={
            cache_key: {
                "cache_type": "extract",
                "chunk_id": "chunk-legacy",
                "return": (
                    f"(entity{default_tuple}LEGACY_ENTITY{default_tuple}CONCEPT"
                    f"{default_tuple}legacy desc){default_completion}"
                ),
                # legacy cache: no queryparam metadata
                "create_time": 1,
            }
        }
    )
    text_chunks_storage = _DummyTextChunksStorage(
        chunks={
            "chunk-legacy": {"llm_cache_list": [cache_key], "file_path": "legacy.txt"}
        }
    )

    captured: list[tuple[str, str]] = []

    async def _fake_rebuild_from_result(*args, **kwargs):
        captured.append((kwargs["tuple_delimiter"], kwargs["completion_delimiter"]))
        return {}, {}

    with (
        patch.object(
            operate_module,
            "_rebuild_from_extraction_result",
            side_effect=_fake_rebuild_from_result,
        ),
        patch.object(operate_module, "_rebuild_single_entity", new=AsyncMock()),
        patch.object(operate_module, "_rebuild_single_relationship", new=AsyncMock()),
        patch.object(
            operate_module,
            "get_storage_keyed_lock",
            return_value=_NoopAsyncContext(),
        ),
    ):
        await operate_module.rebuild_knowledge_from_chunks(
            entities_to_rebuild={"LEGACY_ENTITY": ["chunk-legacy"]},
            relationships_to_rebuild={},
            knowledge_graph_inst=object(),
            entities_vdb=object(),
            relationships_vdb=object(),
            text_chunks_storage=text_chunks_storage,
            llm_response_cache=llm_cache,
            global_config={
                "llm_model_max_async": 1,
                # upgraded instance uses new shared delimiter,
                # but legacy cache should still parse with historical defaults.
                "prompt_config": {
                    "shared": {
                        "tuple_delimiter": "<|NEW|>",
                        "completion_delimiter": "<|NEW_DONE|>",
                    }
                },
            },
        )

    assert captured
    assert captured[0] == (default_tuple, default_completion)
