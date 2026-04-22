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
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface


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
    prompt_profile: dict | None = None,
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
        "_entity_extraction_prompt_profile": prompt_profile,
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


def _require_yaml() -> None:
    pytest.importorskip("yaml")


def _write_prompt_profile(
    path: Path,
    *,
    guidance: str | None = None,
    text_examples: list[str] | None = None,
    json_examples: list[str] | None = None,
) -> None:
    lines: list[str] = []

    def _append_block(key: str, value: str) -> None:
        lines.append(f"{key}: |")
        for line in value.strip("\n").splitlines():
            lines.append(f"  {line}")

    def _append_examples(key: str, values: list[str]) -> None:
        lines.append(f"{key}:")
        for value in values:
            lines.append("  - |")
            for line in value.strip("\n").splitlines():
                lines.append(f"    {line}")

    if guidance is not None:
        _append_block("entity_types_guidance", guidance)
    if text_examples is not None:
        _append_examples("entity_extraction_examples", text_examples)
    if json_examples is not None:
        _append_examples("entity_extraction_json_examples", json_examples)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dummy_embedding_func() -> EmbeddingFunc:
    async def _embed(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    return EmbeddingFunc(embedding_dim=3, func=_embed)


def _patch_prompt_dir(path: Path):
    return patch("lightrag.prompt.get_entity_type_prompt_dir", return_value=path)


def _text_profile_example(label: str) -> str:
    return f"""---Entity Types---
- ExampleType: Test type

---Input Text---
```
{label}
```

---Output---
entity{{tuple_delimiter}}{label}{{tuple_delimiter}}ExampleType{{tuple_delimiter}}{label} description.
{{completion_delimiter}}"""


def _json_profile_example(label: str) -> str:
    return f"""---Entity Types---
- ExampleType: Test type

---Input Text---
```
{label}
```

---Output---
{{
  "entities": [
    {{"name": "{label}", "type": "ExampleType", "description": "{label} description."}}
  ],
  "relationships": []
}}"""


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


@pytest.mark.offline
def test_addon_params_default_includes_entity_type_prompt_file_env(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "entity_type_prompt.sample.yml",
        text_examples=[_text_profile_example("Env Default Example")],
    )

    with patch.dict(
        os.environ,
        {
            "SUMMARY_LANGUAGE": "English",
            "ENTITY_TYPE_PROMPT_FILE": "entity_type_prompt.sample.yml",
        },
    ):
        with _patch_prompt_dir(prompt_dir):
            rag = LightRAG(
                working_dir=str(tmp_path / "rag-default-env"),
                llm_model_func=AsyncMock(),
                embedding_func=_dummy_embedding_func(),
                entity_extraction_use_json=False,
            )

    assert (
        rag.addon_params["entity_type_prompt_file"] == "entity_type_prompt.sample.yml"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_text_mode_prompt_file_injects_examples_and_guidance():
    _require_yaml()

    from lightrag.operate import extract_entities

    guidance = "- ExampleType: Injected guidance"
    example_label = "Custom Text Example"
    prompt_profile = {
        "entity_types_guidance": guidance,
        "entity_extraction_examples": [_text_profile_example(example_label)],
        "entity_extraction_json_examples": [],
    }

    global_config = _make_global_config(
        prompt_profile=prompt_profile,
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert guidance in system_prompt
    assert example_label in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_mode_prompt_file_injects_examples_and_guidance():
    _require_yaml()

    from lightrag.operate import extract_entities

    guidance = "- ExampleType: Injected JSON guidance"
    example_label = "Custom Json Example"
    prompt_profile = {
        "entity_types_guidance": guidance,
        "entity_extraction_examples": [],
        "entity_extraction_json_examples": [_json_profile_example(example_label)],
    }

    global_config = _make_global_config(
        prompt_profile=prompt_profile,
        use_json=True,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _JSON_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert guidance in system_prompt
    assert example_label in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_prompt_file_guidance_falls_back_to_default_when_missing():
    _require_yaml()

    from lightrag.operate import extract_entities
    from lightrag.prompt import PROMPTS

    global_config = _make_global_config(
        prompt_profile={
            "entity_types_guidance": PROMPTS["default_entity_types_guidance"].rstrip(),
            "entity_extraction_examples": [
                _text_profile_example("Fallback Guidance Example")
            ],
            "entity_extraction_json_examples": [],
        },
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert PROMPTS["default_entity_types_guidance"] in system_prompt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cached_prompt_profile_supplies_merged_guidance():
    from lightrag.operate import extract_entities

    merged_guidance = "- ExampleType: Addon override"

    global_config = _make_global_config(
        prompt_profile={
            "entity_types_guidance": merged_guidance,
            "entity_extraction_examples": [_text_profile_example("Override Example")],
            "entity_extraction_json_examples": [],
        },
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch("lightrag.operate.logger"):
        await extract_entities(chunks=_make_chunks(), global_config=global_config)

    call_kwargs = llm_func.call_args_list[0][1]
    system_prompt = call_kwargs.get("system_prompt", "")
    assert merged_guidance in system_prompt


@pytest.mark.offline
def test_text_mode_prompt_file_can_omit_json_examples(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "text_only.yml",
        text_examples=[_text_profile_example("Text Only Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-text"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={"entity_type_prompt_file": "text_only.yml"},
        )

    assert rag.addon_params["entity_type_prompt_file"] == "text_only.yml"


@pytest.mark.offline
def test_json_mode_prompt_file_can_omit_text_examples(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "json_only.yml",
        json_examples=[_json_profile_example("Json Only Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-json"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=True,
            addon_params={"entity_type_prompt_file": "json_only.yml"},
        )

    assert rag.addon_params["entity_type_prompt_file"] == "json_only.yml"


@pytest.mark.offline
def test_text_mode_prompt_file_requires_text_examples(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "missing_text_examples.yml",
        json_examples=[_json_profile_example("Wrong Mode Only")],
    )

    with _patch_prompt_dir(prompt_dir):
        with pytest.raises(ValueError) as exc_info:
            LightRAG(
                working_dir=str(tmp_path / "rag-missing-text"),
                llm_model_func=AsyncMock(),
                embedding_func=None,
                entity_extraction_use_json=False,
                addon_params={"entity_type_prompt_file": "missing_text_examples.yml"},
            )

    assert "entity_extraction_examples" in str(exc_info.value)


@pytest.mark.offline
def test_json_mode_prompt_file_requires_json_examples(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "missing_json_examples.yml",
        text_examples=[_text_profile_example("Wrong Mode Only")],
    )

    with _patch_prompt_dir(prompt_dir):
        with pytest.raises(ValueError) as exc_info:
            LightRAG(
                working_dir=str(tmp_path / "rag-missing-json"),
                llm_model_func=AsyncMock(),
                embedding_func=None,
                entity_extraction_use_json=True,
                addon_params={"entity_type_prompt_file": "missing_json_examples.yml"},
            )

    assert "entity_extraction_json_examples" in str(exc_info.value)


@pytest.mark.offline
def test_prompt_file_rejects_directory_segments(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    with pytest.raises(ValueError) as exc_info:
        LightRAG(
            working_dir=str(tmp_path / "rag-bad-path"),
            llm_model_func=AsyncMock(),
            embedding_func=None,
            addon_params={"entity_type_prompt_file": "../outside.yml"},
        )

    assert "file name only" in str(exc_info.value)


@pytest.mark.offline
def test_prompt_file_rejects_absolute_paths(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    with pytest.raises(ValueError) as exc_info:
        LightRAG(
            working_dir=str(tmp_path / "rag-abs-path"),
            llm_model_func=AsyncMock(),
            embedding_func=None,
            addon_params={"entity_type_prompt_file": str(tmp_path / "abs.yml")},
        )

    assert "file name only" in str(exc_info.value)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_uses_cached_prompt_profile_without_reloading():
    from lightrag.operate import extract_entities

    cached_profile = {
        "entity_types_guidance": "- ExampleType: Cached guidance",
        "entity_extraction_examples": [_text_profile_example("Cached Text Example")],
        "entity_extraction_json_examples": [],
    }
    global_config = _make_global_config(use_json=False, prompt_profile=cached_profile)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _TEXT_MODE_RESPONSE

    with patch(
        "lightrag.operate.resolve_entity_extraction_prompt_profile",
        side_effect=AssertionError("should not resolve profile when cache exists"),
    ):
        with patch("lightrag.operate.logger"):
            await extract_entities(chunks=_make_chunks(), global_config=global_config)
            await extract_entities(chunks=_make_chunks(), global_config=global_config)

    system_prompt = llm_func.call_args_list[0][1].get("system_prompt", "")
    assert "Cached Text Example" in system_prompt
    assert "Cached guidance" in system_prompt


@pytest.mark.offline
def test_sample_prompt_file_matches_builtin_prompt_data():
    _require_yaml()

    from lightrag.prompt import (
        get_default_entity_extraction_prompt_profile,
        load_entity_extraction_prompt_profile,
    )

    sample_file = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "samples"
        / "entity_type_prompt.sample.yml"
    )

    loaded_profile = load_entity_extraction_prompt_profile(sample_file)
    assert loaded_profile == get_default_entity_extraction_prompt_profile()


@pytest.mark.offline
def test_prompt_dir_env_var_overrides_default(tmp_path, monkeypatch):
    _require_yaml()

    from lightrag.prompt import (
        get_entity_type_prompt_dir,
        resolve_entity_type_prompt_path,
    )

    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))
    expected_dir = (tmp_path / "entity_type").resolve()
    assert get_entity_type_prompt_dir() == expected_dir
    resolved = resolve_entity_type_prompt_path("custom.yml")
    assert resolved == expected_dir / "custom.yml"


@pytest.mark.offline
def test_prompt_dir_defaults_to_cwd_relative(tmp_path, monkeypatch):
    _require_yaml()

    from lightrag.prompt import get_entity_type_prompt_dir

    monkeypatch.delenv("PROMPT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert (
        get_entity_type_prompt_dir() == (tmp_path / "prompts" / "entity_type").resolve()
    )


@pytest.mark.offline
def test_prompt_file_rejects_unsupported_extension(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    with pytest.raises(ValueError, match="'.yml' or '.yaml'"):
        LightRAG(
            working_dir=str(tmp_path / "rag-bad-ext"),
            llm_model_func=AsyncMock(),
            embedding_func=None,
            addon_params={"entity_type_prompt_file": "profile.txt"},
        )


@pytest.mark.offline
def test_prompt_file_malformed_yaml_raises_valueerror(tmp_path):
    _require_yaml()

    from lightrag.prompt import load_entity_extraction_prompt_profile

    bad_file = tmp_path / "broken.yml"
    bad_file.write_text("entity_types_guidance: [unclosed", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML"):
        load_entity_extraction_prompt_profile(bad_file)


@pytest.mark.offline
def test_addon_guidance_overrides_file_profile(tmp_path):
    _require_yaml()

    from lightrag.prompt import resolve_entity_extraction_prompt_profile

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "profile.yml",
        guidance="- FileType: from file",
        text_examples=[_text_profile_example("Merged Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        profile = resolve_entity_extraction_prompt_profile(
            addon_params={
                "entity_type_prompt_file": "profile.yml",
                "entity_types_guidance": "- AddonType: from addon_params",
            },
            use_json=False,
        )

    assert profile["entity_types_guidance"] == "- AddonType: from addon_params"
    # File-provided examples must still be honored.
    assert any(
        "Merged Example" in example for example in profile["entity_extraction_examples"]
    )


@pytest.mark.offline
def test_explicit_addon_params_still_picks_up_env_defaults(tmp_path, monkeypatch):
    """Passing addon_params explicitly must not drop env-based defaults."""
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "from_env.yml",
        text_examples=[_text_profile_example("Env Example")],
    )

    monkeypatch.setenv("ENTITY_TYPE_PROMPT_FILE", "from_env.yml")

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-env-default"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={"language": "English"},
        )

    assert rag.addon_params["entity_type_prompt_file"] == "from_env.yml"


@pytest.mark.offline
def test_runtime_addon_params_item_update_refreshes_cached_values(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    prompt_dir = tmp_path / "entity_type"
    prompt_dir.mkdir()
    _write_prompt_profile(
        prompt_dir / "initial.yml",
        text_examples=[_text_profile_example("Initial Example")],
    )
    _write_prompt_profile(
        prompt_dir / "updated.yml",
        guidance="- UpdatedType: runtime update",
        text_examples=[_text_profile_example("Updated Example")],
    )

    with _patch_prompt_dir(prompt_dir):
        rag = LightRAG(
            working_dir=str(tmp_path / "rag-runtime-update"),
            llm_model_func=AsyncMock(),
            embedding_func=_dummy_embedding_func(),
            entity_extraction_use_json=False,
            addon_params={
                "entity_type_prompt_file": "initial.yml",
                "language": "English",
            },
        )
        rag.addon_params["entity_type_prompt_file"] = "updated.yml"
        rag.addon_params["language"] = "French"
        global_config = rag._build_global_config()

    assert global_config["addon_params"]["language"] == "French"
    assert global_config["_resolved_summary_language"] == "French"
    assert (
        global_config["_entity_extraction_prompt_profile"]["entity_types_guidance"]
        == "- UpdatedType: runtime update"
    )
    assert any(
        "Updated Example" in example
        for example in global_config["_entity_extraction_prompt_profile"][
            "entity_extraction_examples"
        ]
    )


@pytest.mark.offline
def test_runtime_addon_params_replacement_refreshes_cached_values(tmp_path):
    _require_yaml()

    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(tmp_path / "rag-runtime-replace"),
        llm_model_func=AsyncMock(),
        embedding_func=_dummy_embedding_func(),
        entity_extraction_use_json=False,
        addon_params={"language": "English"},
    )

    rag.addon_params = {
        "language": "German",
        "entity_types_guidance": "- ReplacementType: runtime replace",
    }
    global_config = rag._build_global_config()

    assert global_config["addon_params"]["language"] == "German"
    assert global_config["_resolved_summary_language"] == "German"
    assert (
        global_config["_entity_extraction_prompt_profile"]["entity_types_guidance"]
        == "- ReplacementType: runtime replace"
    )
