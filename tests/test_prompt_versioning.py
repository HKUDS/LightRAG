import pytest

from lightrag.prompt_versions import (
    PROMPT_VERSION_GROUPS,
    build_localized_seed_versions,
    project_group_prompt_config,
    validate_prompt_group_payload,
)


def test_build_seed_versions_returns_indexing_and_retrieval_groups():
    seeds = build_localized_seed_versions("zh")

    assert set(seeds) == {"indexing", "retrieval"}
    assert seeds["indexing"]["version_name"].startswith("indexing")
    assert seeds["indexing"]["group_type"] == "indexing"
    assert seeds["retrieval"]["group_type"] == "retrieval"


def test_validate_indexing_payload_accepts_entity_types_and_summary_language():
    validate_prompt_group_payload(
        "indexing",
        {
            "entity_types": ["Person", "Organization"],
            "summary_language": "Chinese",
            "shared": {"tuple_delimiter": "<|#|>"},
        },
    )


def test_validate_indexing_payload_rejects_empty_entity_types():
    with pytest.raises(ValueError):
        validate_prompt_group_payload(
            "indexing",
            {
                "entity_types": [],
                "summary_language": "Chinese",
            },
        )


def test_validate_retrieval_payload_rejects_entity_extraction_family():
    with pytest.raises(ValueError):
        validate_prompt_group_payload(
            "retrieval",
            {"entity_extraction": {"system_prompt": "bad"}},
        )


def test_project_group_prompt_config_only_keeps_allowed_families():
    projected = project_group_prompt_config(
        "retrieval",
        {
            "query": {"rag_response": "{context_data}"},
            "keywords": {"keywords_extraction": "{query}"},
            "entity_extraction": {"system_prompt": "{tuple_delimiter}"},
        },
    )

    assert projected == {
        "query": {"rag_response": "{context_data}"},
        "keywords": {"keywords_extraction": "{query}"},
    }
    assert PROMPT_VERSION_GROUPS["retrieval"]["families"] == {"query", "keywords"}
