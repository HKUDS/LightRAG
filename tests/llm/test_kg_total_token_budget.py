"""The total KG budget constrains graph records and the rendered references."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import operate
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, TiktokenTokenizer, resolve_user_prompt

pytestmark = pytest.mark.offline

QUERY = "Who knows Alice?"
TEMPLATE = "Evidence:\n{context_data}\nInstructions:{user_prompt}"


class _Characters:
    def encode(self, content):
        return list(map(ord, content))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def _config(tokenizer=None):
    return {"tokenizer": tokenizer or Tokenizer("characters", _Characters())}


def _entity(name, description="short evidence"):
    return {
        "entity_name": name,
        "entity_type": "person",
        "description": description,
        "source_id": f"chunk-{name}",
    }


def _relation(source, target, description="short relationship"):
    return {
        "src_id": source,
        "tgt_id": target,
        "description": description,
        "source_id": f"chunk-{source}-{target}",
    }


def _search(entities, relations, chunks=None):
    return {
        "final_entities": entities,
        "final_relations": relations,
        "vector_chunks": chunks or [],
        "chunk_tracking": {chunk["chunk_id"]: {} for chunk in chunks or []},
        "query_embedding": None,
    }


def _prompt_tokens(context, param, config, template=TEMPLATE):
    prompt = template.format(
        context_data=context,
        response_type=param.response_type,
        user_prompt=resolve_user_prompt(
            param.user_prompt,
            config.get("user_prompt_prefix", ""),
            param.disable_user_prompt_prefix,
        ).slot,
    )
    tokenizer = config["tokenizer"]
    return len(tokenizer.encode(prompt)) + len(tokenizer.encode(QUERY))


async def _build(monkeypatch, search, param, config, template=TEMPLATE):
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    merge = AsyncMock(return_value=search["vector_chunks"])
    monkeypatch.setattr(operate, "_merge_all_chunks", merge)
    result = await operate._build_query_context(
        QUERY,
        "Alice",
        "knows",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        system_prompt=template,
    )
    return result, merge


@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
@pytest.mark.parametrize("kind", ["entity", "relation", "both"])
async def test_total_budget_filters_graph_before_chunk_selection(
    monkeypatch, mode, kind
):
    entities = [_entity("Alice"), _entity("Bob", "e" * 3000)]
    relations = [_relation("Alice", "Carol"), _relation("Bob", "Carol", "r" * 3000)]
    if kind == "entity":
        relations = []
    if kind == "relation":
        entities = []
    config = _config()
    param = QueryParam(
        mode=mode,
        max_total_tokens=1000,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        enable_rerank=False,
    )
    result, merge = await _build(
        monkeypatch, _search(entities, relations), param, config
    )
    assert result is not None
    assert _prompt_tokens(result.context, param, config) <= param.max_total_tokens
    selected = merge.call_args.kwargs
    assert selected["filtered_entities"] == entities[:1]
    assert selected["filtered_relations"] == relations[:1]
    data = result.raw_data["data"]
    assert [row["entity_name"] for row in data["entities"]] == [
        row["entity_name"] for row in selected["filtered_entities"]
    ]
    assert [(row["src_id"], row["tgt_id"]) for row in data["relationships"]] == [
        (row["src_id"], row["tgt_id"]) for row in selected["filtered_relations"]
    ]
    assert all(row["source_id"] for row in data["entities"] + data["relationships"])


async def test_oversized_entity_does_not_starve_fitting_relationship(monkeypatch):
    config = _config()
    param = QueryParam(max_total_tokens=1000, enable_rerank=False)
    relationship = _relation("Alice", "Bob")
    result, merge = await _build(
        monkeypatch,
        _search([_entity("Alice", "e" * 3000)], [relationship]),
        param,
        config,
    )
    assert result is not None
    assert merge.call_args.kwargs["filtered_entities"] == []
    assert merge.call_args.kwargs["filtered_relations"] == [relationship]
    assert _prompt_tokens(result.context, param, config) <= 1000


async def test_only_retained_graph_records_supply_chunks(monkeypatch):
    config = _config()
    param = QueryParam(mode="hybrid", max_total_tokens=1000, enable_rerank=False)
    entities = [_entity("Alice"), _entity("Bob", "e" * 3000)]
    monkeypatch.setattr(
        operate, "_perform_kg_search", AsyncMock(return_value=_search(entities, []))
    )

    async def related_chunks(records, *args, **kwargs):
        return [
            {
                "chunk_id": record["source_id"],
                "content": f"Evidence for {record['entity_name']}",
                "file_path": f"{record['entity_name']}.txt",
            }
            for record in records
        ]

    source_selection = AsyncMock(side_effect=related_chunks)
    monkeypatch.setattr(
        operate, "_find_related_text_unit_from_entities", source_selection
    )
    result = await operate._build_query_context(
        QUERY,
        "Alice",
        "knows",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        system_prompt=TEMPLATE,
    )
    assert result is not None
    assert source_selection.call_args.args[0] == entities[:1]
    assert [chunk["chunk_id"] for chunk in result.raw_data["data"]["chunks"]] == [
        "chunk-Alice"
    ]
    assert "Evidence for Bob" not in result.context
    assert _prompt_tokens(result.context, param, config) <= 1000


@pytest.mark.parametrize(
    "template", [TEMPLATE, "{context_data}{context_data}{user_prompt}"]
)
async def test_graph_only_exact_boundary_and_zero_capacity(template):
    config = _config()
    entities = [{"entity": "Alice", "type": "person", "description": "e" * 300}]
    param = QueryParam(max_total_tokens=10000, enable_rerank=False)
    context, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert raw["status"] == "success"
    required = _prompt_tokens(context, param, config, template)
    param.max_total_tokens = required
    same, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert same == context
    assert len(raw["data"]["entities"]) == 1
    param.max_total_tokens = required - 1
    empty, raw = await operate._build_context_str(
        entities, [], [], QUERY, param, config, system_prompt=template
    )
    assert empty == ""
    assert raw["metadata"]["token_budget_exceeded"] is True


async def test_long_reference_paths_are_charged_and_reference_ids_rebuilt():
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=1800, enable_rerank=False)
    chunks = [
        {"chunk_id": "first", "content": "first evidence", "file_path": "a.md"},
        {"chunk_id": "last", "content": "last evidence", "file_path": "z" * 2000},
    ]
    context, raw = await operate._build_context_str(
        [], [], chunks, QUERY, param, config, system_prompt=TEMPLATE
    )
    assert _prompt_tokens(context, param, config) <= param.max_total_tokens
    assert [chunk["chunk_id"] for chunk in raw["data"]["chunks"]] == ["first"]
    assert raw["data"]["references"] == [{"reference_id": "1", "file_path": "a.md"}]
    assert '"reference_id": "1"' in context
    assert "last evidence" not in context


@pytest.mark.parametrize("limit", [1, 650])
async def test_budget_exhaustion_never_calls_answer_model(monkeypatch, limit):
    config = _config()
    model = AsyncMock(return_value="answer")
    config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid",
        max_total_tokens=limit,
        enable_rerank=False,
        hl_keywords=["knows"],
        ll_keywords=["Alice"],
    )
    search = _search([_entity("Alice", "e" * 3000)], [])
    monkeypatch.setattr(operate, "_perform_kg_search", AsyncMock(return_value=search))
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    model.assert_not_awaited()


async def test_real_tokenizer_default_prompt_and_user_instructions(monkeypatch):
    config = _config(TiktokenTokenizer())
    config["user_prompt_prefix"] = "Use the evidence. " * 10
    param = QueryParam(
        mode="mix", max_total_tokens=1000, user_prompt="Be brief.", enable_rerank=False
    )
    result, merge = await _build(
        monkeypatch,
        _search([_entity("Alice"), _entity("Bob", "Coherent evidence. " * 1200)], []),
        param,
        config,
        template=PROMPTS["rag_response"],
    )
    assert result is not None
    assert (
        _prompt_tokens(result.context, param, config, PROMPTS["rag_response"]) <= 1000
    )
    assert len(merge.call_args.kwargs["filtered_entities"]) == 1
    assert len(result.raw_data["data"]["entities"]) == 1


async def test_repeated_context_placeholder_counts_rendered_chunks():
    config = _config()
    param = QueryParam(mode="mix", max_total_tokens=2300, enable_rerank=False)
    template = "{context_data}{context_data}"
    chunks = [
        {"chunk_id": str(i), "content": str(i) * 500, "file_path": f"{i}.txt"}
        for i in range(2)
    ]
    context, raw = await operate._build_context_str(
        [], [], chunks, QUERY, param, config, system_prompt=template
    )
    assert raw["status"] == "success"
    assert len(raw["data"]["chunks"]) == 1
    assert _prompt_tokens(context, param, config, template) <= 2300


async def test_mix_vector_only_budget_exhaustion_does_not_call_answer(monkeypatch):
    config = _config()
    model = AsyncMock(return_value="answer")
    config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="mix", max_total_tokens=650, hl_keywords=["knows"], enable_rerank=False
    )
    chunks = [{"chunk_id": "chunk", "content": "e" * 1000, "file_path": "e.txt"}]
    monkeypatch.setattr(
        operate, "_perform_kg_search", AsyncMock(return_value=_search([], [], chunks))
    )
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    model.assert_not_awaited()


async def test_live_prefix_overrides_storage_snapshot_for_budget_only(monkeypatch):
    storage_config = _config()
    storage_config["user_prompt_prefix"] = ""
    live_config = {**storage_config, "user_prompt_prefix": "P" * 2000}
    model = AsyncMock(return_value="answer")
    live_config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid", max_total_tokens=1000, hl_keywords=["knows"], enable_rerank=False
    )
    monkeypatch.setattr(
        operate,
        "_perform_kg_search",
        AsyncMock(return_value=_search([_entity("Alice")], [])),
    )
    source_selection = AsyncMock(return_value=[])
    monkeypatch.setattr(operate, "_merge_all_chunks", source_selection)
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=storage_config),
        param,
        live_config,
        system_prompt=TEMPLATE,
    )
    assert result is None
    source_selection.assert_not_awaited()
    model.assert_not_awaited()
    assert storage_config["user_prompt_prefix"] == ""


async def test_live_prefix_does_not_replace_chunk_selection_snapshot(monkeypatch):
    storage_config = _config()
    storage_config.update(user_prompt_prefix="", related_chunk_number=7)
    live_config = {
        **storage_config,
        "user_prompt_prefix": "Live instructions.",
        "related_chunk_number": 2,
    }
    model = AsyncMock(return_value="answer")
    live_config["role_llm_funcs"] = {"query": model}
    param = QueryParam(
        mode="hybrid", max_total_tokens=1000, hl_keywords=["knows"], enable_rerank=False
    )
    monkeypatch.setattr(
        operate,
        "_perform_kg_search",
        AsyncMock(return_value=_search([_entity("Alice")], [])),
    )
    source_selection = AsyncMock(return_value=[])
    monkeypatch.setattr(operate, "_merge_all_chunks", source_selection)
    result = await operate.kg_query(
        QUERY,
        None,
        None,
        None,
        SimpleNamespace(global_config=storage_config),
        param,
        live_config,
        system_prompt=TEMPLATE,
    )
    assert result is not None
    selected_config = source_selection.call_args.kwargs["text_chunks_db"].global_config
    assert selected_config is storage_config
    assert selected_config["related_chunk_number"] == 7
    prompt = model.call_args.kwargs["system_prompt"]
    assert "Live instructions." in prompt
    assert len(storage_config["tokenizer"].encode(prompt)) + len(QUERY) <= 1000
