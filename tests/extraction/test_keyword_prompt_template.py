import pytest

from lightrag.prompt import PROMPTS


@pytest.mark.offline
def test_keywords_extraction_prompt_template_formats_with_literal_json_braces():
    rendered = PROMPTS["keywords_extraction"].format(
        query="hello",
        examples="example",
        language="en",
    )

    assert "first character of your response must be `{`" in rendered
    assert '{"high_level_keywords": [], "low_level_keywords": []}' in rendered


@pytest.mark.offline
def test_keywords_extraction_examples_are_format_only():
    examples = "\n".join(PROMPTS["keywords_extraction_examples"])

    forbidden_fragments = [
        "Query:",
        "international trade",
        "deforestation",
        "education",
        "Tariffs",
        "Biodiversity",
        "Job training",
    ]

    for fragment in forbidden_fragments:
        assert fragment not in examples

    assert '"high_level_keywords": ["<high_level_keyword>"]' in examples
    assert '"low_level_keywords": ["<low_level_keyword>"]' in examples


@pytest.mark.offline
def test_keywords_extraction_prompt_labels_template_as_not_source_text():
    prompt = PROMPTS["keywords_extraction"]

    assert "---Output Format Template---" in prompt
    assert "---Examples---" not in prompt
    assert "Apple Inc." not in prompt
    assert "output JSON format template only" in prompt
    assert "not source text" in prompt
    assert "must never be used as keyword extraction content" in prompt
    assert (
        "derived only from the `User Query` in the `---Real Data---` section" in prompt
    )


@pytest.mark.offline
def test_keywords_extraction_prompt_keeps_single_real_user_query_section():
    rendered = PROMPTS["keywords_extraction"].format(
        query="How did LightRAG improve retrieval?",
        examples="\n".join(PROMPTS["keywords_extraction_examples"]),
        language="English",
    )

    assert rendered.count("User Query:") == 1
    assert "User Query: How did LightRAG improve retrieval?" in rendered
    assert "User Query:" not in "\n".join(PROMPTS["keywords_extraction_examples"])
