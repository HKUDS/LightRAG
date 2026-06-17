import json
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _FakeKVStorage:
    def __init__(self, initial: dict | None = None) -> None:
        self.data = dict(initial or {})
        self.upserts: list[dict] = []
        self.deletes: list[list[str]] = []

    async def get_by_id(self, key: str):
        return self.data.get(key)

    async def upsert(self, data: dict) -> None:
        self.upserts.append(data)
        self.data.update(data)

    async def delete(self, ids: list[str]) -> None:
        self.deletes.append(ids)
        for item_id in ids:
            self.data.pop(item_id, None)


class _FakeGraphStorage:
    def __init__(
        self,
        *,
        nodes: dict[str, dict] | None = None,
        edges: dict[tuple[str, str], dict] | None = None,
    ) -> None:
        self.nodes = dict(nodes or {})
        self.edges = {
            tuple(sorted(edge_key)): dict(edge_data)
            for edge_key, edge_data in (edges or {}).items()
        }

    async def get_node(self, node_id: str):
        return self.nodes.get(node_id)

    async def has_node(self, node_id: str):
        return node_id in self.nodes

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        self.nodes[node_id] = dict(node_data)

    async def has_edge(self, src: str, tgt: str):
        return tuple(sorted((src, tgt))) in self.edges

    async def get_edge(self, src: str, tgt: str):
        return self.edges.get(tuple(sorted((src, tgt))))

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        self.edges[tuple(sorted((src, tgt)))] = dict(edge_data)


class _FakeVectorStorage:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.deletes: list[list[str]] = []

    async def upsert(self, data: dict) -> None:
        self.upserts.append(data)

    async def delete(self, ids: list[str]) -> None:
        self.deletes.append(ids)


def _make_global_config(
    *,
    addon_params: dict | None = None,
    use_json: bool = False,
) -> dict:
    extract_func = AsyncMock(return_value="")
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": extract_func,
        "role_llm_funcs": {
            "extract": extract_func,
            "keyword": extract_func,
            "query": extract_func,
            "vlm": extract_func,
        },
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": addon_params if addon_params is not None else {},
        "tokenizer": tokenizer,
        "llm_model_max_async": 1,
        "entity_extraction_use_json": use_json,
        "_entity_extraction_prompt_profile": None,
    }


def _make_merge_global_config() -> dict:
    return {
        **_make_global_config(),
        "summary_context_size": 1024,
        "summary_max_tokens": 1024,
        "force_llm_summary_on_merge": 999,
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": 10,
        "source_ids_limit_method": "KEEP",
        "max_file_paths": 10,
    }


def _make_chunks() -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": 64,
            "content": "甲型流感病毒导致流感，重症可并发ARDS，奥司他韦75 mg治疗。",
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
            "file_path": "guideline.md",
        }
    }


def _medical_text_response() -> str:
    return (
        "entity<|#|>流感<|#|>Disease<|#|>流感是急性呼吸道传染病。"
        "\nentity<|#|>ARDS<|#|>Complication<|#|>重症病例可出现ARDS。"
        "\nentity<|#|>甲型流感病毒<|#|>Pathogen<|#|>甲型流感病毒可导致流感。"
        "\nentity<|#|>75 mg<|#|>Dosage<|#|>奥司他韦常用剂量。"
        "\nrelation<|#|>甲型流感病毒<|#|>流感<|#|>导致<|#|>甲型流感病毒导致流感。"
        "\nrelation<|#|>流感<|#|>ARDS<|#|>并发<|#|>重症流感可并发ARDS。"
        "\nrelation<|#|>流感<|#|>75 mg<|#|>治疗<|#|>奥司他韦75 mg可用于治疗。"
        "\n<|COMPLETE|>"
    )


def _medical_json_response() -> str:
    return json.dumps(
        {
            "entities": [
                {
                    "name": "流感",
                    "type": "Disease",
                    "description": "流感是急性呼吸道传染病。",
                },
                {
                    "name": "ARDS",
                    "type": "Complication",
                    "description": "重症病例可出现ARDS。",
                },
                {
                    "name": "甲型流感病毒",
                    "type": "Pathogen",
                    "description": "甲型流感病毒可导致流感。",
                },
                {
                    "name": "75 mg",
                    "type": "Dosage",
                    "description": "奥司他韦常用剂量。",
                },
            ],
            "relationships": [
                {
                    "source": "甲型流感病毒",
                    "target": "流感",
                    "keywords": "导致",
                    "description": "甲型流感病毒导致流感。",
                },
                {
                    "source": "流感",
                    "target": "ARDS",
                    "keywords": "并发",
                    "description": "重症流感可并发ARDS。",
                },
            ],
        },
        ensure_ascii=False,
    )


@pytest.mark.offline
def test_addon_params_backfills_medical_profile_from_env_for_explicit_params():
    from lightrag.addon_params import default_addon_params, normalize_addon_params

    with patch.dict(
        "os.environ",
        {"MEDICAL_KG_PROFILE": "clinical_guideline_zh"},
    ):
        assert default_addon_params()["medical_kg_profile"] == "clinical_guideline_zh"
        assert (
            normalize_addon_params({})["medical_kg_profile"]
            == "clinical_guideline_zh"
        )
        assert (
            normalize_addon_params({"language": "English"})["medical_kg_profile"]
            == "clinical_guideline_zh"
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_default_profile_does_not_normalize_or_add_hierarchy():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(addon_params={}, use_json=False)
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _medical_text_response()

    with patch("lightrag.operate.logger"):
        [(nodes, edges)] = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert "流感" in nodes
    assert "流行性感冒" not in nodes
    assert "ARDS" in nodes
    assert "急性呼吸窘迫综合征（ARDS）" not in nodes
    assert "75 mg" in nodes
    assert ("甲型流感病毒", "流感病毒") not in edges
    assert all(
        edge.get("keywords") != "属于"
        for edge_list in edges.values()
        for edge in edge_list
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_clinical_profile_normalizes_text_and_adds_hierarchy():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        addon_params={"medical_kg_profile": "clinical_guideline_zh"},
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _medical_text_response()

    with patch("lightrag.operate.logger"):
        [(nodes, edges)] = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert "流感" not in nodes
    assert "流行性感冒" in nodes
    assert "ARDS" not in nodes
    assert "急性呼吸窘迫综合征（ARDS）" in nodes
    assert "75 mg" not in nodes
    assert ("甲型流感病毒", "流感病毒") in edges
    assert edges[("甲型流感病毒", "流感病毒")][0]["keywords"] == "属于"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_clinical_profile_name_is_case_insensitive():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        addon_params={"medical_kg_profile": "Clinical_Guideline_ZH"},
        use_json=False,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _medical_text_response()

    with patch("lightrag.operate.logger"):
        [(nodes, edges)] = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert "流感" not in nodes
    assert "流行性感冒" in nodes
    assert "75 mg" not in nodes
    assert ("甲型流感病毒", "流感病毒") in edges


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_clinical_profile_normalizes_json():
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        addon_params={"medical_kg_profile": "clinical_guideline_zh"},
        use_json=True,
    )
    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _medical_json_response()

    with patch("lightrag.operate.logger"):
        [(nodes, edges)] = await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert "流行性感冒" in nodes
    assert "急性呼吸窘迫综合征（ARDS）" in nodes
    assert "75 mg" not in nodes
    assert ("甲型流感病毒", "流感病毒") in edges


@pytest.mark.offline
def test_medical_profile_source_is_excluded_from_chunk_tracking_ids():
    from lightrag.utils import filter_chunk_tracking_source_ids

    assert filter_chunk_tracking_source_ids(
        [
            "chunk-001",
            "medical_kg_profile",
            "",
            "chunk-002",
        ]
    ) == ["chunk-001", "chunk-002"]


@pytest.mark.offline
def test_synthetic_profile_source_is_excluded_from_delete_fallback_sources():
    from lightrag.utils import filter_chunk_tracking_source_ids

    stored_sources = filter_chunk_tracking_source_ids([])
    graph_sources = filter_chunk_tracking_source_ids(["medical_kg_profile"])
    existing_sources = stored_sources or graph_sources

    assert existing_sources == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_nodes_deletes_stale_synthetic_only_entity_tracking():
    from lightrag.operate import _merge_nodes_then_upsert

    entity_chunks = _FakeKVStorage(
        {"流感病毒": {"chunk_ids": ["medical_kg_profile"], "count": 1}}
    )
    graph = _FakeGraphStorage()

    await _merge_nodes_then_upsert(
        "流感病毒",
        [
            {
                "entity_name": "流感病毒",
                "entity_type": "MedicalGroup",
                "description": "流感病毒",
                "source_id": "medical_kg_profile",
                "file_path": "medical_kg_profile",
                "timestamp": 0,
                "generated_by": "medical_kg_profile",
            }
        ],
        graph,
        None,
        _make_merge_global_config(),
        entity_chunks_storage=entity_chunks,
    )

    assert "流感病毒" not in entity_chunks.data
    assert entity_chunks.deletes == [["流感病毒"]]
    assert graph.nodes["流感病毒"]["source_id"] == "medical_kg_profile"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_merge_edges_deletes_stale_synthetic_only_relation_tracking():
    from lightrag.operate import _merge_edges_then_upsert
    from lightrag.utils import make_relation_chunk_key

    relation_key = make_relation_chunk_key("甲型流感病毒", "流感病毒")
    relation_chunks = _FakeKVStorage(
        {relation_key: {"chunk_ids": ["medical_kg_profile"], "count": 1}}
    )
    graph = _FakeGraphStorage(
        nodes={
            "甲型流感病毒": {"entity_id": "甲型流感病毒", "source_id": "chunk-001"},
            "流感病毒": {"entity_id": "流感病毒", "source_id": "chunk-001"},
        }
    )

    await _merge_edges_then_upsert(
        "甲型流感病毒",
        "流感病毒",
        [
            {
                "src_id": "甲型流感病毒",
                "tgt_id": "流感病毒",
                "weight": 0.0,
                "keywords": "属于",
                "description": "甲型流感病毒属于流感病毒。",
                "source_id": "medical_kg_profile",
                "file_path": "medical_kg_profile",
                "timestamp": 0,
                "generated_by": "medical_kg_profile",
            }
        ],
        graph,
        None,
        _FakeVectorStorage(),
        _make_merge_global_config(),
        relation_chunks_storage=relation_chunks,
        entity_chunks_storage=_FakeKVStorage(),
    )

    assert relation_key not in relation_chunks.data
    assert relation_chunks.deletes == [[relation_key]]
    assert graph.edges[tuple(sorted(("甲型流感病毒", "流感病毒")))]["source_id"] == (
        "medical_kg_profile"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rebuild_relationship_filters_synthetic_tracking_for_relation_and_new_nodes():
    from lightrag.operate import _rebuild_single_relationship
    from lightrag.utils import make_relation_chunk_key

    relation_key = make_relation_chunk_key("甲型流感病毒", "流感病毒")
    relation_chunks = _FakeKVStorage(
        {relation_key: {"chunk_ids": ["medical_kg_profile"], "count": 1}}
    )
    entity_chunks = _FakeKVStorage(
        {
            "甲型流感病毒": {"chunk_ids": ["medical_kg_profile"], "count": 1},
            "流感病毒": {"chunk_ids": ["medical_kg_profile"], "count": 1},
        }
    )
    graph = _FakeGraphStorage(
        edges={
            ("甲型流感病毒", "流感病毒"): {
                "source_id": "medical_kg_profile",
                "description": "甲型流感病毒属于流感病毒。",
                "keywords": "属于",
                "weight": 0.0,
                "file_path": "medical_kg_profile",
            }
        }
    )

    await _rebuild_single_relationship(
        graph,
        _FakeVectorStorage(),
        _FakeVectorStorage(),
        "甲型流感病毒",
        "流感病毒",
        ["medical_kg_profile"],
        {
            "medical_kg_profile": {
                ("甲型流感病毒", "流感病毒"): [
                    {
                        "description": "甲型流感病毒属于流感病毒。",
                        "keywords": "属于",
                        "weight": 0.0,
                        "file_path": "medical_kg_profile",
                    }
                ]
            }
        },
        None,
        _make_merge_global_config(),
        relation_chunks_storage=relation_chunks,
        entity_chunks_storage=entity_chunks,
    )

    assert relation_key not in relation_chunks.data
    assert relation_chunks.deletes == [[relation_key]]
    assert "甲型流感病毒" not in entity_chunks.data
    assert "流感病毒" not in entity_chunks.data
    assert sorted(item_id for ids in entity_chunks.deletes for item_id in ids) == [
        "流感病毒",
        "甲型流感病毒",
    ]
    assert graph.edges[tuple(sorted(("甲型流感病毒", "流感病毒")))]["source_id"] == (
        "medical_kg_profile"
    )
