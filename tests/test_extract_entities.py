"""Tests for entity extraction gleaning token limit guard."""

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
