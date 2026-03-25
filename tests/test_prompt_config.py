import pytest

from lightrag.prompt import (
    get_default_prompt_config,
    get_prompt_fingerprint,
    merge_prompt_config,
    validate_prompt_config,
)


def test_get_default_prompt_config_contains_query_keyword_and_indexing_families():
    config = get_default_prompt_config()
    assert set(config["shared"].keys()) == {"tuple_delimiter", "completion_delimiter"}
    assert config["shared"]["tuple_delimiter"]
    assert config["shared"]["completion_delimiter"]
    assert config["query"]["rag_response"]
    assert config["keywords"]["keywords_extraction_examples"]
    assert config["entity_extraction"]["system_prompt"]


def test_default_prompt_config_can_pass_validation():
    config = get_default_prompt_config()
    result = validate_prompt_config(config)
    assert result.errors == []


def test_validate_prompt_config_rejects_missing_strict_placeholders():
    with pytest.raises(ValueError):
        validate_prompt_config(
            {"query": {"rag_response": "Answer plainly."}},
            allowed_families={"query"},
        )


def test_validate_prompt_config_warns_for_missing_recommended_placeholders():
    result = validate_prompt_config(
        {"query": {"rag_response": "{context_data}"}},
        allowed_families={"query"},
    )
    assert "user_prompt" in result.warnings[0]


def test_prompt_fingerprint_changes_when_effective_query_prompt_changes():
    left = get_prompt_fingerprint({"query": {"rag_response": "A {context_data}"}})
    right = get_prompt_fingerprint({"query": {"rag_response": "B {context_data}"}})
    assert left != right


def test_validate_prompt_config_rejects_unknown_family_and_key():
    with pytest.raises(ValueError):
        validate_prompt_config({"unknown": {"x": "y"}})

    with pytest.raises(ValueError):
        validate_prompt_config({"query": {"unknown_key": "value"}})


def test_merge_prompt_config_rejects_unknown_family_and_key():
    base = get_default_prompt_config()
    with pytest.raises(ValueError):
        merge_prompt_config(base, {"unknown": {"x": "y"}})

    with pytest.raises(ValueError):
        merge_prompt_config(base, {"query": {"unknown_key": "value"}})


def test_validate_prompt_config_rejects_invalid_list_field_types():
    with pytest.raises(ValueError):
        validate_prompt_config(
            {"keywords": {"keywords_extraction_examples": "not-a-list"}},
            allowed_families={"keywords"},
        )

    with pytest.raises(ValueError):
        validate_prompt_config(
            {"entity_extraction": {"examples": ["ok", 1]}},
            allowed_families={"entity_extraction"},
        )


def test_validate_prompt_config_accepts_kg_query_context_with_any_one_context_field():
    validate_prompt_config(
        {"query": {"kg_query_context": "{entities_str}"}},
        allowed_families={"query"},
    )
    validate_prompt_config(
        {"query": {"kg_query_context": "{reference_list_str}"}},
        allowed_families={"query"},
    )

    with pytest.raises(ValueError):
        validate_prompt_config(
            {"query": {"kg_query_context": "no placeholders"}},
            allowed_families={"query"},
        )


def test_validate_prompt_config_uses_reviewer_required_recommended_split():
    validate_prompt_config(
        {"summary": {"summarize_entity_descriptions": "{description_list}"}},
        allowed_families={"summary"},
    )


def test_validate_prompt_config_allows_entity_system_prompt_without_entity_types():
    validate_prompt_config(
        {
            "entity_extraction": {
                "system_prompt": "{tuple_delimiter} {completion_delimiter}"
            }
        },
        allowed_families={"entity_extraction"},
    )


def test_validate_prompt_config_requires_entity_user_prompt_input_text():
    with pytest.raises(ValueError):
        validate_prompt_config(
            {"entity_extraction": {"user_prompt": "{completion_delimiter}"}},
            allowed_families={"entity_extraction"},
        )


def test_validate_prompt_config_rejects_empty_shared_delimiters():
    with pytest.raises(ValueError):
        validate_prompt_config({"shared": {"tuple_delimiter": ""}})

    with pytest.raises(ValueError):
        validate_prompt_config({"shared": {"completion_delimiter": "   "}})


def test_validate_prompt_config_warns_missing_recommended_context_fields_for_kg_query_context():
    result = validate_prompt_config(
        {"query": {"kg_query_context": "{entities_str}"}},
        allowed_families={"query"},
    )
    warning_text = " ".join(result.warnings)
    assert "relations_str" in warning_text
    assert "text_chunks_str" in warning_text
    assert "reference_list_str" in warning_text


def test_validate_prompt_config_rejects_non_dict_top_level_input():
    with pytest.raises(ValueError):
        validate_prompt_config(["not", "a", "dict"])  # type: ignore[arg-type]


def test_merge_prompt_config_accepts_partial_base_dict():
    merged = merge_prompt_config({}, {"query": {"rag_response": "{context_data}"}})
    assert merged["query"]["rag_response"] == "{context_data}"


def test_merge_prompt_config_allows_none_override():
    base = get_default_prompt_config()
    merged = merge_prompt_config(base, None)
    assert merged == base


@pytest.mark.parametrize("bad_override", [[], "", 0, False])
def test_merge_prompt_config_rejects_falsy_non_none_override(bad_override):
    base = get_default_prompt_config()
    with pytest.raises(ValueError):
        merge_prompt_config(base, bad_override)  # type: ignore[arg-type]


def test_validate_prompt_config_can_return_errors_without_raising():
    result = validate_prompt_config(
        {"query": {"rag_response": "plain text"}},
        allowed_families={"query"},
        raise_on_error=False,
    )
    assert result.errors


def test_validate_prompt_config_malformed_template_returns_errors_without_raising():
    result = validate_prompt_config(
        {"query": {"rag_response": "{context_data"}},
        allowed_families={"query"},
        raise_on_error=False,
    )
    assert result.errors
    assert "invalid format string" in " ".join(result.errors).lower()


def test_validate_prompt_config_rejects_unknown_placeholders():
    with pytest.raises(ValueError):
        validate_prompt_config(
            {"query": {"rag_response": "{context_data} {bogus}"}},
            allowed_families={"query"},
        )
