"""The plugin operates on stored keys, with the same policy on every path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag import entity_name_normalizer as plugins
from lightrag import LightRAG
from lightrag import operate, utils_graph
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import normalize_entity_name
from tests.api.routes.test_graph_entity_name_normalization import (
    _Graph,
    _NoopLock,
    _VectorStorage,
)
from tests.pipeline.test_custom_kg_self_loop_rejection import _make_rag
from tests.extraction.test_entity_name_cap_final_space_pass import (
    _json_path,
    _stored_key,
    _text_path,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    plugins.get_entity_name_normalizer.cache_clear()
    monkeypatch.delenv("CUSTOM_ENTITY_NAME_NORMALIZER", raising=False)
    yield
    plugins.get_entity_name_normalizer.cache_clear()


def _entry(normalize=str.lower, name="acme", distribution="acme-normalizer"):
    return SimpleNamespace(
        name=name,
        value=f"{distribution}:normalize",
        dist=SimpleNamespace(name=distribution, version="1.0"),
        load=lambda: normalize,
    )


@pytest.fixture
def select(monkeypatch):
    def install(normalize=str.lower, entries=None, name="acme"):
        monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", name)
        monkeypatch.setattr(
            plugins, "entry_points", lambda **kw: entries or [_entry(normalize)]
        )
        return plugins.get_entity_name_normalizer()

    return install


def test_unset_preserves_builtin_behavior_without_discovering(monkeypatch):
    def discover(**kwargs):
        pytest.fail("unset must not discover plugins")

    monkeypatch.setattr(plugins, "entry_points", discover)
    assert plugins.get_entity_name_normalizer() is None
    assert normalize_entity_name('Apple "苹果"') == "Apple苹果"
    assert normalize_entity_name("A" * 300) == "A" * 300


def test_selects_once_and_normalizes_only_final_names(select, monkeypatch):
    selected = select()
    assert selected.name == "acme"
    assert normalize_entity_name("ACME") == "acme"
    assert normalize_entity_name("ACME", final_space_pass=False) == "ACME"
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "missing")
    assert plugins.get_entity_name_normalizer() is selected


def test_unknown_name_fails_sdk_construction_before_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("CUSTOM_ENTITY_NAME_NORMALIZER", "missing")
    monkeypatch.setattr(plugins, "entry_points", lambda **kw: [])
    working_dir = tmp_path / "not-created"
    with pytest.raises(ValueError, match="unknown normalizer 'missing'"):
        LightRAG(working_dir=str(working_dir))
    assert not working_dir.exists()


@pytest.mark.parametrize("name", ["pkg:normalize", "../plugin", "UPPER", " "])
def test_import_paths_and_invalid_names_are_not_selections(select, name):
    with pytest.raises(ValueError, match="CUSTOM_ENTITY_NAME_NORMALIZER"):
        select(name=name)


def test_duplicate_selection_names_both_distributions(select):
    with pytest.raises(ValueError, match="duplicate name 'acme'") as exc:
        select(entries=[_entry(distribution="one"), _entry(distribution="two")])
    assert "one@1.0" in str(exc.value) and "two@1.0" in str(exc.value)


def test_provider_load_failure_names_selection_and_origin(select):
    def broken():
        raise RuntimeError("missing dependency")

    ep = _entry()
    ep.load = broken
    with pytest.raises(
        ValueError, match="acme.*acme-normalizer@1.0.*missing dependency"
    ):
        select(entries=[ep])


def test_unselected_broken_provider_is_never_loaded(select):
    ep = _entry(name="other")

    def broken():
        pytest.fail("unselected implementation imported")

    ep.load = broken
    select(entries=[ep, _entry()])
    assert normalize_entity_name("ACME") == "acme"


def _raises(name):
    raise RuntimeError("provider failed")


@pytest.mark.parametrize(
    "normalize,match",
    [
        (None, "callable"),
        (lambda: "name", "positional argument"),
        (_raises, "provider failed"),
        (lambda name: None, "return str"),
        (lambda name: name + "x", "idempotent"),
        (lambda name: "x" * 257, "256 characters"),
        (lambda name: "苹" * 171, "512 UTF-8 bytes"),
    ],
)
def test_invalid_provider_fails_startup_probe(select, normalize, match):
    with pytest.raises(ValueError, match=f"acme.*{match}"):
        select(normalize)


async def _async_normalize(name):
    return name


def test_async_provider_rejected_at_startup(select):
    with pytest.raises(ValueError, match="acme.*synchronous"):
        select(_async_normalize)


def test_empty_result_keeps_input_and_logs(select, lightrag_log_records):
    select(lambda name: "")
    assert normalize_entity_name("ACME") == "ACME"
    assert any(
        "acme" in r.getMessage() and "empty name" in r.getMessage()
        for r in lightrag_log_records
    )


@pytest.mark.parametrize("prefix", ["A" * 255, "苹" * 170 + "AA"])
def test_plugin_runs_after_character_and_byte_caps_and_space_pass(select, prefix):
    # Stripping before the cap would let the differing tails enter the key.
    select(lambda name: name.strip().lower())
    assert _stored_key(prefix + " B") == _stored_key(prefix + ' "苹果"')
    assert _stored_key(prefix + " B") == prefix.lower()


def test_plugin_sees_the_final_space_pass(select):
    select(lambda name: "canonical" if name == "Apple苹果" else name)
    assert _stored_key('Apple "苹果"') == "canonical"
    assert normalize_entity_name('Apple "苹果"') == "canonical"


@pytest.mark.asyncio
@pytest.mark.parametrize("parse", [_text_path, _json_path], ids=["text", "json"])
async def test_entities_and_endpoints_share_keys_and_remove_new_self_loops(
    select, parse
):
    select()
    nodes, edges = await parse(
        entities=["ACME", "Acme", "OTHER"],
        relations=[("ACME", "acme"), ("ACME", "OTHER")],
    )
    assert set(nodes) == {"acme", "other"}
    assert len(nodes["acme"]) == 2
    assert set(edges) == {("acme", "other")}
    assert edges[("acme", "other")][0]["src_id"] == "acme"
    assert normalize_entity_name("ACME") == "acme"
    assert normalize_entity_name(normalize_entity_name("ACME")) == "acme"


@pytest.mark.asyncio
@pytest.mark.parametrize("parse", [_text_path, _json_path], ids=["text", "json"])
@pytest.mark.parametrize("record", ["entity", "relation"])
@pytest.mark.parametrize(
    "failure", ["raise", "type", "characters", "bytes", "idempotence"]
)
async def test_input_dependent_failure_aborts_instead_of_skipping_record(
    select, parse, record, failure
):
    def normalize(name):
        if name.startswith("BROKEN"):
            if failure == "raise":
                raise RuntimeError("input-dependent failure")
            if failure == "type":
                return None
            if failure == "characters":
                return "x" * 257
            if failure == "bytes":
                return "苹" * 171
            return name + "x"
        return name

    select(normalize)  # The startup probe succeeds.
    kwargs = (
        {"entities": ["BROKEN"]}
        if record == "entity"
        else {"relations": [("OK", "BROKEN")]}
    )
    with pytest.raises(plugins.EntityNameNormalizationError, match="acme"):
        await parse(**kwargs)


@pytest.mark.asyncio
async def test_graph_api_creates_and_resolves_plugin_keys(select, monkeypatch):
    select()
    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **kw: _NoopLock()
    )
    graph = _Graph()
    entities = _VectorStorage()
    relations = _VectorStorage()
    for name in ("ACME", "OTHER"):
        await utils_graph.acreate_entity(
            graph,
            entities,
            relations,
            name,
            {"description": "A company", "entity_type": "organization"},
        )
    await utils_graph.acreate_relation(
        graph, entities, relations, "Acme", "Other", {"description": "supplies"}
    )
    assert set(graph.nodes) == {"acme", "other"}
    assert set(graph.edges) == {("acme", "other")}


@pytest.mark.asyncio
async def test_custom_kg_normalizes_before_self_loop_validation(select):
    select()
    finalize_share_data()
    initialize_share_data(1)
    try:
        rag = _make_rag()
        with pytest.raises(ValueError, match="self-loop on 'acme'"):
            await rag.ainsert_custom_kg(
                {
                    "relationships": [
                        {
                            "src_id": "ACME",
                            "tgt_id": "Acme",
                            "description": "same company",
                            "keywords": "same",
                        }
                    ]
                }
            )
        rag.chunk_entity_relation_graph.upsert_edge.assert_not_called()
        rag.chunk_entity_relation_graph.upsert_node.assert_not_called()
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_rebuild_does_not_discard_a_failing_plugin_record(select):
    def normalize(name):
        if name == "BROKEN":
            raise RuntimeError("broken mapping")
        return name

    select(normalize)
    cache = MagicMock()
    cache.get_by_ids = AsyncMock(
        return_value=[
            {
                "chunk_id": "chunk-1",
                "cache_type": "extract",
                "return": "entity<|#|>BROKEN<|#|>organization<|#|>A company.\n<|COMPLETE|>",
                "create_time": 1,
            }
        ]
    )
    chunks = MagicMock()
    chunks.get_by_ids = AsyncMock(
        return_value=[
            {
                "llm_cache_list": ["cache-1"],
                "file_path": "doc.txt",
            }
        ]
    )
    with pytest.raises(
        plugins.EntityNameNormalizationError, match="acme.*broken mapping"
    ):
        await operate.rebuild_knowledge_from_chunks(
            {"BROKEN": ["chunk-1"]},
            {},
            MagicMock(),
            MagicMock(),
            MagicMock(),
            chunks,
            cache,
            {},
            pipeline_status={"history_messages": []},
        )
