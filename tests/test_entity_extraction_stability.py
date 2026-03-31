"""Tests for entity extraction stability after refactoring.

Covers:
- entity_types_guidance injected into prompts (text mode and JSON mode)
- custom entity_types_guidance via addon_params overrides default
- ENTITY_TYPES env var raises SystemExit at LightRAG init
- EntityExtractionResult Pydantic schema used in JSON mode (entity_extraction kwarg)
- Default entity type guidance constant is present and non-empty
"""

import json
import os
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
    addon_params: dict | None = None,
    use_json: bool = False,
    max_gleaning: int = 0,
) -> dict:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": AsyncMock(return_value=""),
        "entity_extract_max_gleaning": max_gleaning,
        "addon_params": addon_params if addon_params is not None else {},
        "tokenizer": tokenizer,
        "max_extract_input_tokens": 20480,
        "llm_model_max_async": 1,
        "entity_extraction_use_json": use_json,
    }


def _make_chunks(content: str = "Alice founded Acme Corp in 1990.") -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


# --- Minimal valid LLM responses ---

_TEXT_MODE_RESPONSE = (
    "entity<|#|>Alice<|#|>Person<|#|>Alice is the founder of Acme Corp."
    "\nentity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp is a company founded by Alice."
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp."
    "\n<|COMPLETE|>"
)

_JSON_MODE_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "entity_name": "Alice",
                "entity_type": "Person",
                "entity_description": "Alice is the founder of Acme Corp.",
            },
            {
                "entity_name": "Acme Corp",
                "entity_type": "Organization",
                "entity_description": "Acme Corp is a company founded by Alice.",
            },
        ],
        "relationships": [
            {
                "source_entity": "Alice",
                "target_entity": "Acme Corp",
                "relationship_keywords": "founded",
                "relationship_description": "Alice founded Acme Corp.",
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# 1. Default entity_types_guidance constant
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_default_entity_types_guidance_exists():
    """PROMPTS['default_entity_types_guidance'] must be a non-empty string."""
    from lightrag.prompt import PROMPTS

    guidance = PROMPTS["default_entity_types_guidance"]
    assert isinstance(guidance, str)
    assert len(guidance.strip()) > 0


@pytest.mark.offline
def test_default_entity_types_guidance_covers_all_types():
    """Default guidance must mention all 11 canonical entity types."""
    from lightrag.prompt import PROMPTS

    guidance = PROMPTS["default_entity_types_guidance"]
    expected_types = [
        "Person",
        "Creature",
        "Organization",
        "Location",
        "Event",
        "Concept",
        "Method",
        "Content",
        "Data",
        "Artifact",
        "NaturalObject",
    ]
    for t in expected_types:
        assert (
            t in guidance
        ), f"Expected entity type '{t}' missing from default_entity_types_guidance"


# ---------------------------------------------------------------------------
# 2. DEFAULT_ENTITY_TYPES is gone from constants
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_default_entity_types_removed_from_constants():
    """DEFAULT_ENTITY_TYPES must no longer exist in lightrag.constants."""
    import lightrag.constants as constants

    assert not hasattr(
        constants, "DEFAULT_ENTITY_TYPES"
    ), "DEFAULT_ENTITY_TYPES should have been removed from constants.py"


# ---------------------------------------------------------------------------
# 3. ENTITY_TYPES env var raises SystemExit
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_entity_types_env_var_raises_system_exit(tmp_path):
    """LightRAG.__post_init__ must raise SystemExit when ENTITY_TYPES env var is set."""
    from lightrag import LightRAG

    with patch.dict(os.environ, {"ENTITY_TYPES": '["Person"]'}):
        with pytest.raises(SystemExit) as exc_info:
            LightRAG(
                working_dir=str(tmp_path),
                llm_model_func=AsyncMock(),
                embedding_func=None,
            )
    assert "ENTITY_TYPES" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Text mode: entity_types_guidance injected into prompt
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_default_guidance_injected_into_prompt():
    """Default entity_types_guidance is passed to LLM system prompt in text mode."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    global_config = _make_global_config(use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # The system prompt passed to the LLM must contain the default guidance
    assert llm_func.await_count >= 1
    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert PROMPTS["default_entity_types_guidance"] in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_custom_guidance_overrides_default():
    """Custom entity_types_guidance in addon_params overrides default."""
    from lightrag.operate import extract_entities

    custom_guidance = "- Widget: A test widget type"
    global_config = _make_global_config(
        addon_params={"entity_types_guidance": custom_guidance},
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert custom_guidance in system_prompt


# ---------------------------------------------------------------------------
# 5. JSON mode: entity_types_guidance injected + entity_extraction kwarg set
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_default_guidance_injected_into_prompt():
    """Default entity_types_guidance is passed to LLM system prompt in JSON mode."""
    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    global_config = _make_global_config(use_json=True)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert llm_func.await_count >= 1
    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert PROMPTS["default_entity_types_guidance"] in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_entity_extraction_kwarg_passed():
    """JSON mode must pass entity_extraction=True to the LLM function."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=True)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert llm_func.await_count >= 1
    call_kwargs = llm_func.call_args_list[0][1]
    assert call_kwargs.get("entity_extraction") is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_custom_guidance_overrides_default():
    """Custom entity_types_guidance in addon_params overrides default in JSON mode."""
    from lightrag.operate import extract_entities

    custom_guidance = "- Gadget: A test gadget type"
    global_config = _make_global_config(
        addon_params={"entity_types_guidance": custom_guidance},
        use_json=True,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert custom_guidance in system_prompt


# ---------------------------------------------------------------------------
# 6. Text mode: entity_extraction kwarg NOT passed (only JSON mode uses it)
# ---------------------------------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_no_entity_extraction_kwarg():
    """Text mode must NOT pass entity_extraction=True to the LLM function."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    call_kwargs = llm_func.call_args_list[0][1]
    assert not call_kwargs.get("entity_extraction", False)
