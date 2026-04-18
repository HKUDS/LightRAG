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
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
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

_TEXT_MODE_MISPREFIXED_RELATION_RESPONSE = (
    "entity<|#|>Alice<|#|>Person<|#|>Alice is the founder of Acme Corp."
    "\nentity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp is a company founded by Alice."
    "\nentity<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp."
    "\n<|COMPLETE|>"
)

_TEXT_MODE_GLEANED_RELATION_RESPONSES = [
    _TEXT_MODE_MISPREFIXED_RELATION_RESPONSE,
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp.\n<|COMPLETE|>",
]

_TEXT_MODE_CROSS_PASS_RELATION_RESPONSES = [
    "entity<|#|>Alice<|#|>Person<|#|>Alice founded a company.\n<|COMPLETE|>",
    "entity<|#|>Acme Corp<|#|>Organization<|#|>Acme Corp was founded by Alice."
    "\nrelation<|#|>Alice<|#|>Acme Corp<|#|>founded<|#|>Alice founded Acme Corp.\n<|COMPLETE|>",
]

_JSON_MODE_RESPONSE = json.dumps(
    {
        "entities": [
            {
                "name": "Alice",
                "type": "Person",
                "description": "Alice is the founder of Acme Corp.",
            },
            {
                "name": "Acme Corp",
                "type": "Organization",
                "description": "Acme Corp is a company founded by Alice.",
            },
        ],
        "relationships": [
            {
                "source": "Alice",
                "target": "Acme Corp",
                "keywords": "founded",
                "description": "Alice founded Acme Corp.",
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


@pytest.mark.offline
def test_json_examples_define_all_relationship_endpoints_as_entities():
    """JSON examples must define every relationship endpoint in the entities list."""
    from lightrag.prompt import PROMPTS

    for example in PROMPTS["entity_extraction_json_examples"]:
        if "<Output>" in example:
            output = example.split("<Output>", 1)[1].strip()
        else:
            output = example.split("---Output---", 1)[1].strip()
        parsed = json.loads(output)
        entity_names = {
            entity["name"] for entity in parsed.get("entities", []) if entity
        }
        for relationship in parsed.get("relationships", []):
            assert relationship["source"] in entity_names
            assert relationship["target"] in entity_names


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
    assert "must start with `relation`, never `entity`" in system_prompt
    assert "After the last entity row, switch prefixes to `relation`" in system_prompt
    assert "Output at most 100 total rows" in system_prompt
    assert "Output at most 40 entity rows" in system_prompt


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


@pytest.mark.offline
def test_text_continue_prompt_requires_relation_prefix_for_corrections():
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["entity_continue_extraction_user_prompt"]
    assert (
        "Any corrected relationship row must be emitted with the literal `relation` prefix"
        in prompt
    )
    assert (
        "output at most {max_total_records} total rows and at most {max_entity_records} entity rows"
        in prompt
    )
    assert (
        "may reference entities that were already extracted correctly in the previous response"
        in prompt
    )
    assert (
        "whose source and target entities are both included in this response"
        not in prompt
    )


@pytest.mark.offline
def test_text_user_prompt_includes_quantity_limits():
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["entity_extraction_user_prompt"]
    assert (
        "output at most {max_total_records} total rows and at most {max_entity_records} entity rows"
        in prompt
    )
    assert (
        "If the row limit is reached, output `{completion_delimiter}` immediately"
        in prompt
    )


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
    assert "Output at most 100 total records" in system_prompt
    assert "Output at most 40 entity objects" in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_entity_extraction_kwarg_passed():
    """JSON mode must pass response_format={'type':'json_object'} to the LLM function."""
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
    assert call_kwargs.get("response_format") == {"type": "json_object"}
    assert call_kwargs.get("entity_extraction") is not True


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


@pytest.mark.offline
def test_json_user_prompt_includes_quantity_limits():
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["entity_extraction_json_user_prompt"]
    assert (
        "output at most {max_total_records} total records and at most {max_entity_records} entity objects"
        in prompt
    )
    assert (
        "Only output relationship objects whose `source` and `target` are both included"
        in prompt
    )


@pytest.mark.offline
def test_json_continue_prompt_includes_quantity_limits():
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["entity_continue_extraction_json_user_prompt"]
    assert (
        "output at most {max_total_records} total records and at most {max_entity_records} entity objects"
        in prompt
    )
    assert (
        "may reference entities already extracted correctly in the previous response"
        in prompt
    )


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


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_recovers_mis_prefixed_relationship_row():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_MISPREFIXED_RELATION_RESPONSE

    with patch("lightrag.operate.logger"):
        chunk_results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    entities, relationships = chunk_results[0]
    assert len(entities) == 2
    assert len(relationships) == 1
    assert next(iter(relationships.keys())) == ("Alice", "Acme Corp")


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_gleaned_relation_merges_cleanly_after_recovery():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False, max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.side_effect = _TEXT_MODE_GLEANED_RELATION_RESPONSES

    with patch("lightrag.operate.logger"):
        chunk_results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    entities, relationships = chunk_results[0]
    assert len(entities) == 2
    assert len(relationships) == 1
    relation_data = next(iter(relationships.values()))[0]
    assert relation_data["src_id"] == "Alice"
    assert relation_data["tgt_id"] == "Acme Corp"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_gleaned_relation_can_reference_prior_entity():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(use_json=False, max_gleaning=1)
    llm_func = global_config["llm_model_func"]
    llm_func.side_effect = _TEXT_MODE_CROSS_PASS_RELATION_RESPONSES

    with patch("lightrag.operate.logger"):
        chunk_results = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    entities, relationships = chunk_results[0]
    assert set(entities.keys()) == {"Alice", "Acme Corp"}
    assert len(relationships) == 1
    relation_data = next(iter(relationships.values()))[0]
    assert relation_data["src_id"] == "Alice"
    assert relation_data["tgt_id"] == "Acme Corp"
