"""A caller-supplied system prompt must render in every query mode."""

from lightrag.prompt import PROMPTS


def _rendered(template: str) -> str:
    # The keys operate.py fills for the KG modes and for the naive mode.
    return template.format(
        response_type="Multiple Paragraphs",
        user_prompt="who?",
        context_data="KG CONTEXT",
        content_data="CHUNK CONTENT",
    )


def test_the_kg_mode_template_renders_with_the_naive_keys_too():
    assert "KG CONTEXT" in _rendered(PROMPTS["rag_response"])


def test_the_naive_mode_template_renders_with_the_kg_keys_too():
    assert "CHUNK CONTENT" in _rendered(PROMPTS["naive_rag_response"])


def test_a_caller_template_written_for_one_mode_renders_in_the_other():
    # A template that only knows {context_data} is what a KG-mode caller passes;
    # operate.py fills both spellings in every mode, so it must not raise KeyError.
    kg_only = "Context: {context_data}\nQ: {user_prompt}"
    assert "KG CONTEXT" in _rendered(kg_only)

    naive_only = "Chunks: {content_data}\nQ: {user_prompt}"
    assert "CHUNK CONTENT" in _rendered(naive_only)
