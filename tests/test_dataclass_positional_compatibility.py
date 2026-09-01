"""``QueryParam`` and ``LightRAG`` are plain dataclasses: field ORDER is public API.

Neither declares ``kw_only=True``, which means SDK callers may construct them
positionally. Inserting a new field in the middle silently rebinds every
positional argument after it -- the kind of break that produces no error, just
different behaviour. Two real examples, both caught in review on this feature:

- ``QueryParam``: a positional ``False`` meant for ``enable_rerank`` landed on
  the inserted field while reranking quietly kept its environment default.
- ``LightRAG``: inserting mid-class shifted 55 of 72 constructor parameters, so
  an int meant for ``entity_extract_max_gleaning`` bound to a string field and
  only failed much later, during prompt composition.

New fields therefore go at the END of these classes. These tests pin that rule
against the field order as it stood before ``user_prompt_prefix`` and
``disable_user_prompt_prefix`` were added.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam


pytestmark = pytest.mark.offline


# The positional prefix of the signature as it stood before
# disable_user_prompt_prefix existed. Do NOT reorder to match a future refactor
# -- if this list stops matching, the constructor signature changed and every
# positional caller downstream changed behaviour with it.
_LEGACY_POSITIONAL_ORDER = [
    "mode",
    "only_need_context",
    "only_need_prompt",
    "response_type",
    "stream",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "hl_keywords",
    "ll_keywords",
    "conversation_history",
    "user_prompt",
    "enable_rerank",
    "include_references",
]


def test_legacy_positional_field_order_is_unchanged():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert names[: len(_LEGACY_POSITIONAL_ORDER)] == _LEGACY_POSITIONAL_ORDER


def test_new_fields_are_appended_after_the_legacy_ones():
    names = [f.name for f in dataclasses.fields(QueryParam)]
    assert "disable_user_prompt_prefix" in names
    assert names.index("disable_user_prompt_prefix") >= len(_LEGACY_POSITIONAL_ORDER)


def test_positional_construction_still_binds_enable_rerank():
    """The concrete break: the 15th positional argument.

    Before the field was moved to the end, this ``False`` bound to
    ``disable_user_prompt_prefix`` and ``enable_rerank`` silently kept its
    ``RERANK_BY_DEFAULT`` value.
    """
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        "be concise",
        False,
    )
    assert param.user_prompt == "be concise"
    assert param.enable_rerank is False
    assert param.disable_user_prompt_prefix is False


def test_positional_construction_still_binds_include_references():
    """The 16th positional argument, shifted by the same insertion."""
    param = QueryParam(
        "mix",
        False,
        False,
        "Multiple Paragraphs",
        False,
        40,
        20,
        6000,
        8000,
        30000,
        [],
        [],
        [],
        None,
        True,
        True,
    )
    assert param.enable_rerank is True
    assert param.include_references is True
    assert param.disable_user_prompt_prefix is False


# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------

# The constructor's positional order before `user_prompt_prefix` was added.
# Appending a new field at the end means appending one line here too; that tax
# is deliberate, because any change to this list is a change to the public
# constructor API and should be looked at by a human.
_LEGACY_LIGHTRAG_INIT_ORDER = [
    "working_dir",
    "kv_storage",
    "vector_storage",
    "graph_storage",
    "doc_status_storage",
    "workspace",
    "log_level",
    "log_file_path",
    "top_k",
    "chunk_top_k",
    "max_entity_tokens",
    "max_relation_tokens",
    "max_total_tokens",
    "related_chunk_number",
    "kg_chunk_pick_method",
    "enable_content_headings",
    "entity_extract_max_gleaning",
    "entity_extract_max_records",
    "entity_extract_max_entities",
    "force_llm_summary_on_merge",
    "chunk_token_size",
    "chunk_overlap_token_size",
    "tokenizer",
    "tiktoken_model_name",
    "chunking_func",
    "embedding_func",
    "embedding_chunk_overlap_token_size",
    "embedding_batch_num",
    "embedding_func_max_async",
    "embedding_cache_config",
    "default_embedding_timeout",
    "llm_model_func",
    "role_llm_configs",
    "llm_model_name",
    "summary_max_tokens",
    "summary_context_size",
    "summary_length_recommended",
    "llm_model_max_async",
    "llm_model_kwargs",
    "default_llm_timeout",
    "entity_extraction_use_json",
    "rerank_model_func",
    "rerank_model_max_async",
    "default_rerank_timeout",
    "min_rerank_score",
    "vector_db_storage_cls_kwargs",
    "enable_llm_cache",
    "enable_llm_cache_for_entity_extract",
    "vlm_process_enable",
    "max_parallel_insert",
    "max_parallel_parse_native",
    "max_parallel_parse_mineru",
    "max_parallel_parse_docling",
    "max_parallel_analyze",
    "queue_size_parse",
    "queue_size_analyze",
    "queue_size_insert",
    "pipeline_scheduling_page_size",
    "pipeline_require_strict_storage_reads",
    "max_pending_documents",
    "max_graph_nodes",
    "max_source_ids_per_entity",
    "max_source_ids_per_relation",
    "source_ids_limit_method",
    "max_file_paths",
    "file_path_more_placeholder",
    "addon_params",
    "auto_manage_storages_states",
    "cosine_better_than_threshold",
    "ollama_server_infos",
    "_storages_status",
]


def _lightrag_init_params() -> list[str]:
    """Introspect the signature rather than constructing LightRAG.

    ``__post_init__`` does real storage setup, which would make a positional
    construction test slow and brittle. The signature is exactly -- and only --
    the contract at stake here.
    """
    return list(inspect.signature(LightRAG.__init__).parameters)[1:]  # drop self


def test_legacy_lightrag_init_order_is_unchanged():
    params = _lightrag_init_params()
    assert params[: len(_LEGACY_LIGHTRAG_INIT_ORDER)] == _LEGACY_LIGHTRAG_INIT_ORDER


def test_user_prompt_prefix_is_appended_after_every_legacy_field():
    """The rule itself: new fields come after the last pre-existing one."""
    params = _lightrag_init_params()
    assert params.index("user_prompt_prefix") > params.index("_storages_status")


def test_lightrag_positional_anchors_did_not_shift():
    """Spot-checks either end of the range the insertion used to move."""
    params = _lightrag_init_params()
    # First field displaced by the original insertion.
    assert params[16] == "entity_extract_max_gleaning"
    # Last pre-existing field; nothing may be inserted before it.
    assert params[len(_LEGACY_LIGHTRAG_INIT_ORDER) - 1] == "_storages_status"
