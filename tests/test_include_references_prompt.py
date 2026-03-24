"""Tests that include_references=False removes reference instructions from LLM prompts."""

import pytest

from lightrag.prompt import PROMPTS

pytestmark = pytest.mark.offline


class TestRagResponsePromptVariants:
    """Verify rag_response and rag_response_no_ref prompt templates."""

    def test_rag_response_contains_reference_instructions(self):
        prompt = PROMPTS["rag_response"]
        assert "References Section Format" in prompt
        assert "reference_id" in prompt
        assert "Reference Document List" in prompt

    def test_rag_response_no_ref_omits_reference_instructions(self):
        prompt = PROMPTS["rag_response_no_ref"]
        assert "References Section Format" not in prompt
        assert "reference_id" not in prompt
        assert "Reference Document List" not in prompt

    def test_rag_response_no_ref_has_required_placeholders(self):
        prompt = PROMPTS["rag_response_no_ref"]
        assert "{response_type}" in prompt
        assert "{user_prompt}" in prompt
        assert "{context_data}" in prompt


class TestNaiveRagResponsePromptVariants:
    """Verify naive_rag_response and naive_rag_response_no_ref prompt templates."""

    def test_naive_rag_response_contains_reference_instructions(self):
        prompt = PROMPTS["naive_rag_response"]
        assert "References Section Format" in prompt
        assert "reference_id" in prompt

    def test_naive_rag_response_no_ref_omits_reference_instructions(self):
        prompt = PROMPTS["naive_rag_response_no_ref"]
        assert "References Section Format" not in prompt
        assert "reference_id" not in prompt

    def test_naive_rag_response_no_ref_has_required_placeholders(self):
        prompt = PROMPTS["naive_rag_response_no_ref"]
        assert "{response_type}" in prompt
        assert "{user_prompt}" in prompt
        assert "{content_data}" in prompt


class TestContextTemplateVariants:
    """Verify context templates with and without reference sections."""

    def test_kg_context_contains_reference_list(self):
        template = PROMPTS["kg_query_context"]
        assert "Reference Document List" in template
        assert "{reference_list_str}" in template

    def test_kg_context_no_ref_omits_reference_list(self):
        template = PROMPTS["kg_query_context_no_ref"]
        assert "Reference Document List" not in template
        assert "{reference_list_str}" not in template

    def test_naive_context_contains_reference_list(self):
        template = PROMPTS["naive_query_context"]
        assert "Reference Document List" in template
        assert "{reference_list_str}" in template

    def test_naive_context_no_ref_omits_reference_list(self):
        template = PROMPTS["naive_query_context_no_ref"]
        assert "Reference Document List" not in template
        assert "{reference_list_str}" not in template
