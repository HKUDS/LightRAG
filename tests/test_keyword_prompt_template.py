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
