"""Regression tests: every merge-stage description outcome must be
sanitized before it lands on graph nodes/edges.

Extraction-time descriptions are cleaned by
``sanitize_and_normalize_extracted_text``, but merge-stage descriptions
can bypass that: LLM re-summaries, descriptions read back from existing
graph nodes, and multimodal entity descriptions injected straight from
chunk content. Any of them carrying control characters / surrogates
(e.g. ``\\frac`` decoded as ``\\x0c`` + ``rac`` from unescaped LaTeX in
VLM JSON) would break GraphML (XML) serialization on the next NetworkX
flush with::

    ValueError: All strings must be XML compatible: Unicode or ASCII,
    no NULL bytes or control characters

``_handle_entity_relation_summary`` (single / join outcomes) and
``_summarize_descriptions`` (LLM outcome) now sanitize at every exit.
"""

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config(summary_return: str) -> dict:
    """Minimal global_config for ``_summarize_descriptions`` whose ``extract``
    role LLM returns ``summary_return`` verbatim."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    extract_func = AsyncMock(return_value=summary_return)
    return {
        "role_llm_funcs": {"extract": extract_func},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        "summary_context_size": 10_000,
        "tokenizer": tokenizer,
    }


# A description that the LLM "echoed" from a dirty source: NULL byte, bell,
# unit-separator and DEL are all illegal in XML 1.0; \t and \n are legal and
# must be preserved.
_DIRTY_SUMMARY = "Clean text\x00with\x07control\x1fchars\x7f\tand\nnewlines"
_EXPECTED_CLEAN = "Clean textwithcontrolchars\tand\nnewlines"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_summarize_descriptions_strips_control_chars():
    """The LLM summary path must remove XML-incompatible control characters
    while preserving legal whitespace."""
    from lightrag.operate import _summarize_descriptions

    global_config = _make_global_config(_DIRTY_SUMMARY)

    summary = await _summarize_descriptions(
        "Entity",
        "TEST_ENTITY",
        ["first description", "second description"],
        global_config,
        llm_response_cache=None,
    )

    assert summary == _EXPECTED_CLEAN
    assert "\x00" not in summary
    assert "\x07" not in summary
    assert "\x1f" not in summary
    assert "\x7f" not in summary


@pytest.mark.offline
@pytest.mark.asyncio
async def test_single_description_early_return_is_sanitized():
    """The ``len(description_list) == 1`` early return skips the LLM
    entirely — this is exactly the path a dirty multimodal entity
    description (chunk content reused verbatim) takes on first insert."""
    from lightrag.operate import _handle_entity_relation_summary

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        [_DIRTY_SUMMARY],
        "<SEP>",
        # Early return happens before any config key is read.
        global_config={},
        llm_response_cache=None,
    )

    assert llm_was_used is False
    assert description == _EXPECTED_CLEAN


@pytest.mark.offline
@pytest.mark.asyncio
async def test_join_without_llm_is_sanitized():
    """The no-LLM join outcome must also sanitize: descriptions read back
    from pre-existing (dirty) graph nodes re-enter the merge here."""
    from lightrag.operate import _handle_entity_relation_summary

    tokenizer = Tokenizer("dummy", DummyTokenizer())
    global_config = {
        "tokenizer": tokenizer,
        "summary_context_size": 100_000,
        "summary_max_tokens": 100_000,
        "force_llm_summary_on_merge": 10,
    }

    description, llm_was_used = await _handle_entity_relation_summary(
        "Entity",
        "TEST_ENTITY",
        [_DIRTY_SUMMARY, "clean second description"],
        "<SEP>",
        global_config,
        llm_response_cache=None,
    )

    assert llm_was_used is False
    assert description == f"{_EXPECTED_CLEAN}<SEP>clean second description"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_summarized_description_is_graphml_writable(tmp_path):
    """End-to-end guard: a node whose description came from the summary path
    must serialize to GraphML without the original ValueError."""
    from lightrag.operate import _summarize_descriptions

    global_config = _make_global_config(_DIRTY_SUMMARY)

    description = await _summarize_descriptions(
        "Entity",
        "TEST_ENTITY",
        ["first description", "second description"],
        global_config,
        llm_response_cache=None,
    )

    graph = nx.Graph()
    graph.add_node("TEST_ENTITY", description=description)

    # Pre-fix, write_graphml raised:
    # "All strings must be XML compatible: ... no NULL bytes or control characters"
    out = tmp_path / "graph.graphml"
    nx.write_graphml(graph, out)

    reloaded = nx.read_graphml(out)
    assert reloaded.nodes["TEST_ENTITY"]["description"] == _EXPECTED_CLEAN
