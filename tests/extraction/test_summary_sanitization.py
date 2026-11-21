"""Regression tests: every merge-stage description outcome must be
sanitized before it lands on graph nodes/edges, and must never be empty.

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

An LLM summary replaces every fragment it merges, so one that is empty after
think-tag removal and sanitization must fail the merge instead of erasing
the stored description.
"""

from unittest.mock import AsyncMock

import networkx as nx
import pytest

from lightrag.exceptions import EmptySummaryResponseError
from lightrag.operate import _handle_entity_relation_summary, _summarize_descriptions
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _FakeCache:
    """LLM response cache double that keeps rows in a dict."""

    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


def _make_global_config(*summary_returns: str) -> dict:
    """Minimal global_config for ``_summarize_descriptions`` whose ``extract``
    role LLM returns ``summary_returns`` verbatim, one per call."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    extract_func = AsyncMock(side_effect=list(summary_returns))
    return {
        "role_llm_funcs": {"extract": extract_func},
        "addon_params": {},
        "_resolved_summary_language": "English",
        "summary_length_recommended": 100,
        "summary_context_size": 10_000,
        "tokenizer": tokenizer,
    }


async def _summarize(global_config: dict, cache=None) -> str:
    return await _summarize_descriptions(
        "Entity",
        "TEST_ENTITY",
        ["first description", "second description"],
        global_config,
        llm_response_cache=cache,
    )


# A description that the LLM "echoed" from a dirty source: NULL byte, bell,
# unit-separator and DEL are all illegal in XML 1.0; \t and \n are legal and
# must be preserved.
_DIRTY_SUMMARY = "Clean text\x00with\x07control\x1fchars\x7f\tand\nnewlines"
_EXPECTED_CLEAN = "Clean textwithcontrolchars\tand\nnewlines"


@pytest.mark.asyncio
async def test_summarized_description_is_clean_and_graphml_writable(tmp_path):
    """The LLM summary path must remove XML-incompatible control characters
    while preserving legal whitespace, so the node serializes to GraphML."""
    description = await _summarize(_make_global_config(_DIRTY_SUMMARY))

    assert description == _EXPECTED_CLEAN

    graph = nx.Graph()
    graph.add_node("TEST_ENTITY", description=description)
    # Pre-fix, write_graphml raised:
    # "All strings must be XML compatible: ... no NULL bytes or control characters"
    out = tmp_path / "graph.graphml"
    nx.write_graphml(graph, out)
    assert nx.read_graphml(out).nodes["TEST_ENTITY"]["description"] == _EXPECTED_CLEAN


@pytest.mark.asyncio
async def test_single_description_early_return_is_sanitized():
    """The ``len(description_list) == 1`` early return skips the LLM
    entirely — this is exactly the path a dirty multimodal entity
    description (chunk content reused verbatim) takes on first insert."""
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


@pytest.mark.asyncio
async def test_join_without_llm_is_sanitized():
    """The no-LLM join outcome must also sanitize: descriptions read back
    from pre-existing (dirty) graph nodes re-enter the merge here."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_answer, retry_reaches_llm",
    [
        pytest.param("", True, id="empty"),
        pytest.param("<think>reasoning, no answer</think>", True, id="think-only"),
        # Accepted residue: cached before sanitization empties it, so the
        # retry replays it from the cache and fails again.
        pytest.param("\x00\x07", False, id="control-chars-only"),
    ],
)
async def test_empty_summary_fails_and_retry_asks_the_llm_again(
    empty_answer, retry_reaches_llm
):
    """An empty summary raises instead of erasing the merged fragments. Unless
    it is only control characters, it is not cached, so the retry gets the
    LLM's next answer."""
    global_config = _make_global_config(empty_answer, "Recovered summary.")
    llm = global_config["role_llm_funcs"]["extract"]
    cache = _FakeCache()

    with pytest.raises(EmptySummaryResponseError, match="Entity:TEST_ENTITY"):
        await _summarize(global_config, cache)

    if retry_reaches_llm:
        assert await _summarize(global_config, cache) == "Recovered summary."
        assert llm.await_count == 2
    else:
        with pytest.raises(EmptySummaryResponseError):
            await _summarize(global_config, cache)
        assert llm.await_count == 1
