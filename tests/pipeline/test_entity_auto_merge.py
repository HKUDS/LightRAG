"""Tests for opt-in entity auto-merge (issue #1323).

Design, per the maintainer's own comments on the issue: scope merge checks to
one document's newly-extracted entities against existing same-type entities
already in the graph (never a whole-corpus rescan), bound the LLM call count
with a strict top-N similarity candidate cap, and -- the gap the maintainer
flagged in the prior reference implementation (PR #2102) -- record an
explicit verdict for every candidate, including a "different entity" verdict
for a conflicting near-duplicate, rather than silently dropping it.

``auto_merge_document_entities`` runs after ``merge_nodes_and_edges`` has
already durably merged the document itself, so these tests exercise it as a
standalone post-step against an in-memory graph/vector-store double -- the
same fixture shape ``tests/extraction/test_merge_entities_relation_weight.py``
uses for the underlying ``amerge_entities`` primitive, extended with
``get_nodes_batch`` (needed for same-type candidate filtering) and a
configurable vector-store ``query`` (needed for candidate generation).
"""

from __future__ import annotations

import pytest

from lightrag import utils_graph
from lightrag.operate import (
    _choose_entity_merge_target,
    _get_entity_merge_candidates,
    _parse_entity_merge_verdicts_payload,
    auto_merge_document_entities,
)

pytestmark = pytest.mark.offline


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _patch_graph_lock(monkeypatch):
    # amerge_entities (lightrag/utils_graph.py) takes a real keyed lock via
    # get_storage_keyed_lock, which requires initialize_share_data() to have
    # run. These tests exercise auto_merge_document_entities as a standalone
    # unit against in-memory doubles, not the full shared-storage lifecycle,
    # so the lock itself is replaced with a no-op -- the same substitution
    # tests/api/routes/test_graph_attribute_validation.py already uses for
    # exactly this reason.
    monkeypatch.setattr(
        utils_graph, "get_storage_keyed_lock", lambda *a, **k: _NoopLock()
    )


class _FakeGraphStorage:
    def __init__(self, nodes: dict, edges: dict | None = None):
        self.nodes = nodes
        self.edges = edges or {}

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def get_nodes_batch(self, node_ids):
        return {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        return [
            (src, tgt) for (src, tgt) in self.edges if src == node_id or tgt == node_id
        ]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != node_id and tgt != node_id
        }

    async def index_done_callback(self):
        return True


class _FakeVectorStorage:
    """``query`` returns a caller-configured candidate list; everything else
    is an in-memory double sufficient for ``amerge_entities``'s own writes."""

    def __init__(self, query_results: dict[str, list[dict]] | None = None):
        self.global_config = {"workspace": "test"}
        self.query_results = query_results or {}
        self.upserts = []
        self.deletes = []

    async def query(self, query, top_k, query_embedding=None):
        return self.query_results.get(query, [])[:top_k]

    async def upsert(self, data):
        self.upserts.append(data)

    async def delete(self, ids):
        self.deletes.append(ids)

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


class _FakeKVStorage:
    def __init__(self, rows: dict | None = None):
        self.rows = rows or {}

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def upsert(self, data):
        self.rows.update(data)

    async def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)

    async def index_done_callback(self):
        return True


def _node(entity_type="ORG", description="A description", source_id="chunk-x"):
    return {
        "entity_id": "unused",
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
        "file_path": "doc.txt",
    }


def _global_config(*, enabled=True, extract_func, top_k=3, threshold=0.85):
    return {
        "enable_entity_auto_merge": enabled,
        "entity_merge_top_k": top_k,
        "entity_merge_similarity_threshold": threshold,
        "role_llm_funcs": {"extract": extract_func},
        "llm_cache_identities": {"extract": {"role": "extract"}},
        "tokenizer": None,
    }


# ---------------------------------------------------------------------------
# auto_merge_document_entities: end-to-end against the in-memory doubles
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_by_default_is_a_true_no_op():
    async def never_called(*_args, **_kwargs):
        raise AssertionError("must not call the LLM when the feature is off")

    graph = _FakeGraphStorage({"Acme Corp": _node()})
    entities_vdb = _FakeVectorStorage()
    cfg = _global_config(enabled=False, extract_func=never_called)

    result = await auto_merge_document_entities(
        entity_names={"Acme Corp"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )
    assert result == {"merged": [], "conflicts": []}


@pytest.mark.asyncio
async def test_confirmed_same_entity_merges_via_the_real_amerge_entities_primitive():
    async def aware_extract(prompt, **_kwargs):
        return (
            '{"verdicts": [{"candidate": "Acme Corporation", '
            '"decision": "same_entity", "reason": "same company, full name"}]}'
        )

    graph = _FakeGraphStorage(
        {
            "Acme Corp": _node(description="Acme Corp, a widget maker"),
            "Acme Corporation": _node(description="Acme Corporation, widgets"),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {"Acme Corp": [{"entity_name": "Acme Corporation", "distance": 0.95}]}
    )
    cfg = _global_config(extract_func=aware_extract)

    result = await auto_merge_document_entities(
        entity_names={"Acme Corp"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )

    assert len(result["merged"]) == 1
    assert result["conflicts"] == []
    # Exactly one of the two survives in the graph; the fake's upsert_node
    # overwrite means the merge actually ran through amerge_entities.
    surviving = [n for n in ("Acme Corp", "Acme Corporation") if n in graph.nodes]
    assert len(surviving) == 1


@pytest.mark.asyncio
async def test_different_entity_verdict_is_recorded_as_a_conflict_not_dropped():
    """The gap the maintainer flagged in PR #2102: a non-merge must be an
    explicit, auditable verdict, not silence."""

    async def strict_extract(prompt, **_kwargs):
        return (
            '{"verdicts": [{"candidate": "Apple fruit", '
            '"decision": "different_entity", "reason": "company vs fruit"}]}'
        )

    graph = _FakeGraphStorage(
        {
            "Apple Inc.": _node(description="Technology company"),
            "Apple fruit": _node(description="A fruit"),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {"Apple Inc.": [{"entity_name": "Apple fruit", "distance": 0.9}]}
    )
    cfg = _global_config(extract_func=strict_extract)

    result = await auto_merge_document_entities(
        entity_names={"Apple Inc."},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )

    assert result["merged"] == []
    assert result["conflicts"] == [
        {
            "subject": "Apple Inc.",
            "candidate": "Apple fruit",
            "reason": "company vs fruit",
        }
    ]
    assert "Apple Inc." in graph.nodes and "Apple fruit" in graph.nodes


@pytest.mark.asyncio
async def test_candidate_of_a_different_entity_type_is_never_sent_to_the_llm():
    async def never_called(*_args, **_kwargs):
        raise AssertionError(
            "a same-name-but-different-type candidate must be filtered before "
            "the LLM call, never rejected by it"
        )

    graph = _FakeGraphStorage(
        {
            "Washington": _node(entity_type="PERSON"),
            "Washington State": _node(entity_type="LOCATION"),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {"Washington": [{"entity_name": "Washington State", "distance": 0.9}]}
    )
    cfg = _global_config(extract_func=never_called)

    result = await auto_merge_document_entities(
        entity_names={"Washington"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )
    assert result == {"merged": [], "conflicts": []}


@pytest.mark.asyncio
async def test_malformed_llm_response_never_merges():
    async def broken_extract(prompt, **_kwargs):
        return "not json at all"

    graph = _FakeGraphStorage(
        {
            "Acme Corp": _node(),
            "Acme Corporation": _node(),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {"Acme Corp": [{"entity_name": "Acme Corporation", "distance": 0.95}]}
    )
    cfg = _global_config(extract_func=broken_extract)

    result = await auto_merge_document_entities(
        entity_names={"Acme Corp"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )
    assert result == {"merged": [], "conflicts": []}
    assert "Acme Corp" in graph.nodes and "Acme Corporation" in graph.nodes


@pytest.mark.asyncio
async def test_llm_call_raising_skips_only_that_entity_not_the_whole_document():
    """A transient failure (LLM timeout, rate limit, provider error) raised
    by the LLM call itself -- not just a malformed response -- must be
    caught per-entity. This runs after merge_nodes_and_edges has already
    durably committed the document, so letting it propagate would regress
    an already-successfully-ingested document to FAILED over an optional
    best-effort refinement step."""

    async def flaky_extract(prompt, **_kwargs):
        if "Acme Corp\n" in prompt:
            raise TimeoutError("simulated LLM timeout")
        return (
            '{"verdicts": [{"candidate": "Widgets Ltd", '
            '"decision": "same_entity", "reason": "same company"}]}'
        )

    graph = _FakeGraphStorage(
        {
            "Acme Corp": _node(description="Acme"),
            "Acme Corporation": _node(description="Acme"),
            "Widget Co": _node(description="Widgets"),
            "Widgets Ltd": _node(description="Widgets"),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {
            "Acme Corp": [{"entity_name": "Acme Corporation", "distance": 0.95}],
            "Widget Co": [{"entity_name": "Widgets Ltd", "distance": 0.95}],
        }
    )
    cfg = _global_config(extract_func=flaky_extract)

    # Must not raise -- the whole point of the fix.
    result = await auto_merge_document_entities(
        entity_names={"Acme Corp", "Widget Co"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )

    # Acme's verification raised and was skipped entirely; Widget's
    # independent merge still went through.
    assert result["merged"] in (
        [("Widgets Ltd", "Widget Co")],
        [("Widget Co", "Widgets Ltd")],
    )
    assert "Acme Corp" in graph.nodes and "Acme Corporation" in graph.nodes


@pytest.mark.asyncio
async def test_a_failed_single_merge_does_not_abort_the_rest_of_the_document():
    async def aware_extract(prompt, **_kwargs):
        if "Acme Corp\n" in prompt:
            return (
                '{"verdicts": [{"candidate": "Acme Corporation", '
                '"decision": "same_entity", "reason": "same company"}]}'
            )
        return (
            '{"verdicts": [{"candidate": "Widgets Ltd", '
            '"decision": "same_entity", "reason": "same company"}]}'
        )

    class _FlakyGraphStorage(_FakeGraphStorage):
        async def delete_node(self, node_id):
            if node_id == "Acme Corporation":
                raise RuntimeError("simulated storage failure")
            return await super().delete_node(node_id)

    graph = _FlakyGraphStorage(
        {
            "Acme Corp": _node(description="Acme"),
            "Acme Corporation": _node(description="Acme"),
            "Widget Co": _node(description="Widgets"),
            "Widgets Ltd": _node(description="Widgets"),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {
            "Acme Corp": [{"entity_name": "Acme Corporation", "distance": 0.95}],
            "Widget Co": [{"entity_name": "Widgets Ltd", "distance": 0.95}],
        }
    )
    cfg = _global_config(extract_func=aware_extract)

    result = await auto_merge_document_entities(
        entity_names={"Acme Corp", "Widget Co"},
        doc_chunk_ids={"chunk-x"},
        knowledge_graph_inst=graph,
        entities_vdb=entities_vdb,
        relationships_vdb=_FakeVectorStorage(),
        global_config=cfg,
    )

    # Acme's merge failed and was skipped; Widget's independent merge still
    # went through.
    assert result["merged"] == [("Widgets Ltd", "Widget Co")] or result["merged"] == [
        ("Widget Co", "Widgets Ltd")
    ]
    assert "Acme Corp" in graph.nodes and "Acme Corporation" in graph.nodes


# ---------------------------------------------------------------------------
# _choose_entity_merge_target
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_choose_target_prefers_the_entity_established_by_another_document():
    entity_chunks = _FakeKVStorage(
        {
            "New": {"chunk_ids": ["chunk-current"], "count": 1},
            "Old": {"chunk_ids": ["chunk-current", "chunk-from-other-doc"], "count": 2},
        }
    )
    target, source = await _choose_entity_merge_target(
        "New", "Old", {"chunk-current"}, entity_chunks
    )
    assert (target, source) == ("Old", "New")


@pytest.mark.asyncio
async def test_choose_target_falls_back_to_alphabetical_when_neither_is_established():
    entity_chunks = _FakeKVStorage(
        {
            "Zeta": {"chunk_ids": ["chunk-current"], "count": 1},
            "Alpha": {"chunk_ids": ["chunk-current"], "count": 1},
        }
    )
    target, source = await _choose_entity_merge_target(
        "Zeta", "Alpha", {"chunk-current"}, entity_chunks
    )
    assert (target, source) == ("Alpha", "Zeta")


# ---------------------------------------------------------------------------
# _get_entity_merge_candidates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_candidates_excludes_self_and_caps_at_top_k():
    graph = _FakeGraphStorage(
        {
            "Acme": _node(),
            "Acme2": _node(),
            "Acme3": _node(),
            "Acme4": _node(),
        }
    )
    entities_vdb = _FakeVectorStorage(
        {
            "Acme": [
                {"entity_name": "Acme", "distance": 1.0},  # self, excluded
                {"entity_name": "Acme2", "distance": 0.95},
                {"entity_name": "Acme3", "distance": 0.9},
                {"entity_name": "Acme4", "distance": 0.87},
            ]
        }
    )
    candidates = await _get_entity_merge_candidates(
        "Acme",
        _node(),
        entities_vdb,
        graph,
        top_k=2,
        similarity_threshold=0.85,
        excluded_names=set(),
    )
    assert [name for name, _ in candidates] == ["Acme2", "Acme3"]


@pytest.mark.asyncio
async def test_get_candidates_applies_the_distance_threshold_when_present():
    graph = _FakeGraphStorage({"Acme": _node(), "Unrelated": _node()})
    entities_vdb = _FakeVectorStorage(
        {"Acme": [{"entity_name": "Unrelated", "distance": 0.5}]}
    )
    candidates = await _get_entity_merge_candidates(
        "Acme",
        _node(),
        entities_vdb,
        graph,
        top_k=3,
        similarity_threshold=0.85,
        excluded_names=set(),
    )
    assert candidates == []


# ---------------------------------------------------------------------------
# _parse_entity_merge_verdicts_payload
# ---------------------------------------------------------------------------


def test_parse_verdicts_tolerates_a_markdown_fence():
    result = (
        "```json\n"
        '{"verdicts": [{"candidate": "X", "decision": "same_entity", "reason": "r"}]}'
        "\n```"
    )
    verdicts = _parse_entity_merge_verdicts_payload(result)
    assert verdicts == [{"candidate": "X", "decision": "same_entity", "reason": "r"}]


def test_parse_verdicts_returns_empty_on_missing_verdicts_key():
    assert _parse_entity_merge_verdicts_payload('{"other": []}') == []


def test_parse_verdicts_returns_empty_on_none():
    assert _parse_entity_merge_verdicts_payload(None) == []


def test_parse_verdicts_skips_malformed_entries_but_keeps_valid_ones():
    result = (
        '{"verdicts": [{"candidate": "X"}, '
        '{"candidate": "Y", "decision": "different_entity"}]}'
    )
    verdicts = _parse_entity_merge_verdicts_payload(result)
    assert verdicts == [
        {"candidate": "Y", "decision": "different_entity", "reason": ""}
    ]
