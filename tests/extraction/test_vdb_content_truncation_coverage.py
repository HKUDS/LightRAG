"""VDB content-truncation coverage across every entity/relation write path.

Design: "Tokenizer safe splitting and truncation refactor". A prior review of
this refactor found that several entity/relation VDB write paths built their
``content`` field directly, bypassing ``_truncate_vdb_content`` entirely.
These tests drive each affected function directly (in-memory graph/VDB
fakes, no DB or LLM) and assert:

* the final VDB payload's ``content`` is actually truncated to fit
  ``embedding_token_limit`` when the source is oversized;
* a truncation failure prevents that object's own graph mutation. The
  project-wide contract is: the graph is authoritative, and a stale/missing
  VDB entry is always repairable via the offline rebuild tool -- but
  ``TokenBudgetError`` (unlike a transient VDB I/O failure) is a
  deterministic, non-retryable content-shape problem that a rebuild would
  hit identically, so it is NOT treated like a normal recoverable VDB
  failure. Instead, ``_truncate_vdb_content`` is front-loaded immediately
  before each object's own graph mutation (verify, then mutate the graph,
  then perform the actual VDB I/O) -- so a truncation failure leaves that
  object entirely unwritten (both graph and VDB), while the actual VDB
  upsert/delete I/O call still happens strictly after the graph write, so a
  *transient* VDB I/O failure still leaves the normal recoverable
  (graph-updated, VDB-stale) window. This front-loading is per-object, not
  across a whole multi-object operation: in a batch with several
  entities/relationships, an earlier object that already validated and
  wrote successfully is NOT rolled back just because a later object in the
  same call fails. Paths that already wrapped post-graph-write VDB failures
  in ``VectorStorageConsistencyError`` before this refactor (the merge
  helpers in ``utils_graph.py``) keep that same wrapping, unchanged.
"""

from __future__ import annotations

import pytest

import lightrag.operate as operate
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import (
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    _rebuild_single_entity,
    _rebuild_single_relationship,
)
from lightrag.utils import (
    Tokenizer,
    TokenBudgetError,
    TokenizerInterface,
    VectorStorageConsistencyError,
)
from lightrag.utils_graph import (
    _edit_entity_impl,
    _merge_entities_impl,
    acreate_entity,
    acreate_relation,
    aedit_relation,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _shared_data():
    # acreate_entity/acreate_relation take get_storage_keyed_lock, which needs
    # the single-process shared-data registry (idempotent to re-initialize).
    initialize_share_data()


class _CharTokenizerImpl(TokenizerInterface):
    """One token per character — a real (non-hostile) Tokenizer."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class _AlwaysFailsTruncateTokenizer(Tokenizer):
    """Real Tokenizer whose truncate_by_token_limit always raises.

    Used to simulate "even the safe contract cannot fit this budget" without
    needing to construct a genuinely pathological multi-token-per-char input.
    encode()/decode() behave normally so unrelated logic (e.g. single-item
    description passthrough) is unaffected.
    """

    def __init__(self):
        super().__init__("hostile", _CharTokenizerImpl())

    def truncate_by_token_limit(self, content, max_tokens):
        raise TokenBudgetError(max_tokens, 999, content[:20])


class _FailsOnlyForRelationContentTokenizer(Tokenizer):
    """Truncates entity content fine, but always fails for relation content.

    Relation content is always built as ``f"{keywords}\\t{src}\\n{tgt}\\n{desc}"``
    (a literal tab), while entity content is ``f"{name}\\n{desc}"`` (no tab) --
    so a literal tab reliably discriminates the two without needing to know
    which VDB the caller is building for. Used to reproduce "entity payload
    truncates fine, only the relationship's own content fails" -- the exact
    scenario a full front-load must protect against for multi-object
    operations (custom KG, relationship rebuild, edge merge with a new
    endpoint), where per-object front-loading alone is not enough.
    """

    def __init__(self):
        super().__init__("relation-hostile", _CharTokenizerImpl())

    def truncate_by_token_limit(self, content, max_tokens):
        if "\t" in content:
            raise TokenBudgetError(max_tokens, 999, content[:20])
        return super().truncate_by_token_limit(content, max_tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizerImpl())


def _cfg(tokenizer, embedding_token_limit=None) -> dict:
    return {
        "tokenizer": tokenizer,
        "embedding_token_limit": embedding_token_limit,
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": operate.SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


class _MemGraph:
    """Minimal in-memory graph mirroring the real get/upsert/has/delete contract."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict = {}
        self.upsert_node_calls = 0
        self.upsert_edge_calls = 0

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.upsert_node_calls += 1
        self.nodes[name] = dict(node_data)

    async def has_edge(self, s, t):
        return (s, t) in self.edges or (t, s) in self.edges

    async def get_edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def upsert_edge(self, s, t, edge_data):
        self.upsert_edge_calls += 1
        self.edges[(s, t)] = dict(edge_data)

    async def get_node_edges(self, name):
        return [pair for pair in self.edges if name in pair]

    async def delete_node(self, name):
        self.nodes.pop(name, None)

    async def index_done_callback(self):
        return None

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict]]):
        for name, node_data in nodes:
            await self.upsert_node(name, node_data)

    async def has_nodes_batch(self, names: list[str]) -> set[str]:
        return {name for name in names if name in self.nodes}

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict]]):
        for s, t, edge_data in edges:
            await self.upsert_edge(s, t, edge_data)


class _MemVDB:
    """Minimal in-memory vector storage recording every upsert payload."""

    def __init__(self, global_config: dict):
        self.global_config = global_config
        self.records: dict[str, dict] = {}
        self.upsert_calls = 0

    async def upsert(self, payload: dict):
        self.upsert_calls += 1
        self.records.update(payload)

    async def delete(self, ids):
        for _id in ids:
            self.records.pop(_id, None)

    async def index_done_callback(self):
        return None


LONG_DESCRIPTION = "x" * 500


# --------------------------------------------------------------------------- #
# operate.py: rebuild path
# --------------------------------------------------------------------------- #


async def test_rebuild_single_entity_truncates_content_and_updates_graph():
    graph = _MemGraph()
    await graph.upsert_node(
        "ALICE",
        {
            "entity_id": "ALICE",
            "description": "short",
            "entity_type": "PERSON",
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_tok(), embedding_token_limit=20)
    vdb = _MemVDB(cfg)

    await _rebuild_single_entity(
        graph,
        vdb,
        "ALICE",
        ["c1"],
        chunk_entities={"c1": {"ALICE": [{"description": LONG_DESCRIPTION}]}},
        llm_response_cache=None,
        global_config=cfg,
    )
    record = vdb.records[operate.compute_mdhash_id("ALICE", prefix="ent-")]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20
    assert record["content"] != "ALICE\n" + LONG_DESCRIPTION


async def test_rebuild_single_entity_truncation_failure_leaves_graph_untouched():
    graph = _MemGraph()
    await graph.upsert_node(
        "ALICE",
        {
            "entity_id": "ALICE",
            "description": "short",
            "entity_type": "PERSON",
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    vdb = _MemVDB(cfg)
    node_before = dict(graph.nodes["ALICE"])

    with pytest.raises(TokenBudgetError):
        await _rebuild_single_entity(
            graph,
            vdb,
            "ALICE",
            ["c1"],
            chunk_entities={"c1": {"ALICE": [{"description": LONG_DESCRIPTION}]}},
            llm_response_cache=None,
            global_config=cfg,
        )

    # The truncation failure happened before the graph write: the node is
    # unchanged from before the call, and the VDB was never touched.
    assert graph.nodes["ALICE"] == node_before
    assert vdb.upsert_calls == 0


async def test_rebuild_single_relationship_truncates_own_content():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(
            name, {"entity_id": name, "description": name, "source_id": "c1"}
        )
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "short",
            "keywords": "k",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    degraded = await _rebuild_single_relationship(
        graph,
        relationships_vdb,
        entities_vdb,
        "A",
        "B",
        ["c1"],
        chunk_relationships={
            "c1": {("A", "B"): [{"description": LONG_DESCRIPTION, "keywords": "k"}]}
        },
        llm_response_cache=None,
        global_config=cfg,
    )
    assert degraded is False
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    record = relationships_vdb.records[rel_id]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_rebuild_single_relationship_truncation_failure_leaves_edge_untouched():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(
            name, {"entity_id": name, "description": name, "source_id": "c1"}
        )
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "short",
            "keywords": "k",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)
    edge_before = dict(graph.edges[("A", "B")])

    with pytest.raises(TokenBudgetError):
        await _rebuild_single_relationship(
            graph,
            relationships_vdb,
            entities_vdb,
            "A",
            "B",
            ["c1"],
            chunk_relationships={
                "c1": {("A", "B"): [{"description": LONG_DESCRIPTION, "keywords": "k"}]}
            },
            llm_response_cache=None,
            global_config=cfg,
        )

    assert graph.edges[("A", "B")] == edge_before
    assert relationships_vdb.upsert_calls == 0


async def test_rebuild_single_relationship_endpoint_created_when_relation_content_fails():
    """Multi-object case: the endpoint entity's own content truncates fine,
    but the relationship's own content cannot. Truncation is front-loaded
    per-object (not across the whole operation), so a missing endpoint is
    still created (graph + VDB) even though the relationship's own VDB
    payload later fails to validate -- only the edge itself is never
    written."""
    graph = _MemGraph()
    await graph.upsert_node(
        "A", {"entity_id": "A", "description": "short", "source_id": "c1"}
    )
    # The edge already exists (so the rebuild has something to rebuild from)
    # but "B" does not exist as a node yet -- _rebuild_single_relationship
    # must create it as a missing endpoint.
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "short",
            "keywords": "k",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_FailsOnlyForRelationContentTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _rebuild_single_relationship(
            graph,
            relationships_vdb,
            entities_vdb,
            "A",
            "B",
            ["c1"],
            chunk_relationships={
                "c1": {("A", "B"): [{"description": "short", "keywords": "k"}]}
            },
            llm_response_cache=None,
            global_config=cfg,
            structural_fallback=True,
        )

    assert "B" in graph.nodes
    assert entities_vdb.upsert_calls == 1
    assert graph.edges[("A", "B")]["description"] == "short"
    assert relationships_vdb.upsert_calls == 0


# --------------------------------------------------------------------------- #
# operate.py: merge (ingestion) path
# --------------------------------------------------------------------------- #


async def test_merge_nodes_then_upsert_truncates_content():
    graph = _MemGraph()
    cfg = _cfg(_tok(), embedding_token_limit=20)
    vdb = _MemVDB(cfg)

    node_data = await _merge_nodes_then_upsert(
        "ALICE",
        [
            {
                "entity_type": "PERSON",
                "description": LONG_DESCRIPTION,
                "source_id": "c1",
                "file_path": "f",
            }
        ],
        graph,
        vdb,
        cfg,
    )
    assert node_data["entity_id"] == "ALICE"
    record = vdb.records[operate.compute_mdhash_id("ALICE", prefix="ent-")]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_merge_nodes_then_upsert_truncation_failure_leaves_graph_untouched():
    graph = _MemGraph()
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _merge_nodes_then_upsert(
            "ALICE",
            [
                {
                    "entity_type": "PERSON",
                    "description": LONG_DESCRIPTION,
                    "source_id": "c1",
                    "file_path": "f",
                }
            ],
            graph,
            vdb,
            cfg,
        )

    assert "ALICE" not in graph.nodes
    assert vdb.upsert_calls == 0


async def test_merge_edges_then_upsert_truncates_relation_content():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await _merge_edges_then_upsert(
        "A",
        "B",
        [
            {
                "description": LONG_DESCRIPTION,
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
            }
        ],
        graph,
        relationships_vdb,
        entities_vdb,
        cfg,
    )
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    record = relationships_vdb.records[rel_id]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_merge_edges_then_upsert_truncation_failure_leaves_edge_untouched():
    graph = _MemGraph()
    # source_id already matches the relation's own source_id so neither
    # endpoint's source_id changes -- the endpoint-update branch (which
    # would itself hit _truncate_vdb_content and fail first) is skipped,
    # isolating the failure to the relationship's own content.
    for name in ("A", "B"):
        await graph.upsert_node(
            name, {"entity_id": name, "description": name, "source_id": "c1"}
        )
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _merge_edges_then_upsert(
            "A",
            "B",
            [
                {
                    "description": LONG_DESCRIPTION,
                    "keywords": "k",
                    "weight": 1.0,
                    "source_id": "c1",
                }
            ],
            graph,
            relationships_vdb,
            entities_vdb,
            cfg,
        )

    assert ("A", "B") not in graph.edges and ("B", "A") not in graph.edges
    assert relationships_vdb.upsert_calls == 0


async def test_merge_edges_then_upsert_new_endpoint_entity_content_truncated():
    """The new-endpoint-entity branch (added_entities path) also truncates."""
    graph = _MemGraph()
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await _merge_edges_then_upsert(
        "NEW_A",
        "NEW_B",
        [
            {
                "description": LONG_DESCRIPTION,
                "keywords": "k",
                "weight": 1.0,
                "source_id": "c1",
            }
        ],
        graph,
        relationships_vdb,
        entities_vdb,
        cfg,
    )
    for name in ("NEW_A", "NEW_B"):
        record = entities_vdb.records[operate.compute_mdhash_id(name, prefix="ent-")]
        assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_merge_edges_then_upsert_new_endpoints_created_when_relation_content_fails():
    """Multi-object case: both new endpoints' own content truncates fine, but
    the relationship's own content cannot. Truncation is front-loaded
    per-object (not across the whole operation), so both new endpoints
    (graph + VDB) are created even though the relationship's own VDB payload
    later fails to validate -- only the edge itself is never written."""
    graph = _MemGraph()
    cfg = _cfg(_FailsOnlyForRelationContentTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await _merge_edges_then_upsert(
            "NEW_A",
            "NEW_B",
            [
                {
                    "description": "short",
                    "keywords": "k",
                    "weight": 1.0,
                    "source_id": "c1",
                }
            ],
            graph,
            relationships_vdb,
            entities_vdb,
            cfg,
        )

    assert "NEW_A" in graph.nodes
    assert "NEW_B" in graph.nodes
    assert ("NEW_A", "NEW_B") not in graph.edges
    assert entities_vdb.upsert_calls == 2
    assert relationships_vdb.upsert_calls == 0


# --------------------------------------------------------------------------- #
# utils_graph.py: create / edit(rename) / merge
# --------------------------------------------------------------------------- #


async def test_acreate_entity_truncates_content():
    graph = _MemGraph()
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await acreate_entity(
        graph,
        entities_vdb,
        relationships_vdb,
        "ALICE",
        {"description": LONG_DESCRIPTION, "entity_type": "PERSON"},
    )
    record = entities_vdb.records[operate.compute_mdhash_id("ALICE", prefix="ent-")]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_acreate_entity_truncation_failure_leaves_graph_untouched():
    graph = _MemGraph()
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await acreate_entity(
            graph,
            entities_vdb,
            relationships_vdb,
            "ALICE",
            {"description": LONG_DESCRIPTION, "entity_type": "PERSON"},
        )

    assert "ALICE" not in graph.nodes
    assert entities_vdb.upsert_calls == 0


async def test_acreate_relation_truncates_content():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await acreate_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "A",
        "B",
        {"description": LONG_DESCRIPTION, "keywords": "k"},
    )
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    record = relationships_vdb.records[rel_id]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_acreate_relation_truncation_failure_leaves_graph_untouched():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(TokenBudgetError):
        await acreate_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "A",
            "B",
            {"description": LONG_DESCRIPTION, "keywords": "k"},
        )

    assert ("A", "B") not in graph.edges and ("B", "A") not in graph.edges
    assert relationships_vdb.upsert_calls == 0


async def test_edit_relation_truncates_content():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    await graph.upsert_edge(
        "A",
        "B",
        {
            "description": "short",
            "keywords": "k",
            "weight": 1.0,
            "source_id": "c1",
            "file_path": "f",
        },
    )
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await aedit_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        "A",
        "B",
        {"description": LONG_DESCRIPTION},
    )
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    record = relationships_vdb.records[rel_id]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_edit_relation_truncation_failure_leaves_edge_and_vdb_untouched():
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    original_edge = {
        "description": "short",
        "keywords": "k",
        "weight": 1.0,
        "source_id": "c1",
        "file_path": "f",
    }
    await graph.upsert_edge("A", "B", dict(original_edge))
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)
    # Pre-seed the VDB record the edit would otherwise delete, to prove the
    # delete never happened either.
    rel_id = operate.compute_mdhash_id("A" + "B", prefix="rel-")
    relationships_vdb.records[rel_id] = {"content": "pre-existing"}

    with pytest.raises(TokenBudgetError):
        await aedit_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "A",
            "B",
            {"description": LONG_DESCRIPTION},
        )

    assert graph.edges[("A", "B")] == original_edge
    assert relationships_vdb.upsert_calls == 0
    assert rel_id in relationships_vdb.records  # delete never happened either


async def test_edit_relation_vdb_delete_failure_still_leaves_graph_updated():
    """A *transient* VDB I/O failure (unlike a TokenBudgetError) must not
    prevent the graph write: the graph write happens before the VDB delete,
    so a delete failure still leaves the new relation content saved in the
    graph -- the recoverable (graph-updated, VDB-stale) window the offline
    rebuild tool is for."""
    graph = _MemGraph()
    for name in ("A", "B"):
        await graph.upsert_node(name, {"entity_id": name, "description": name})
    original_edge = {
        "description": "short",
        "keywords": "k",
        "weight": 1.0,
        "source_id": "c1",
        "file_path": "f",
    }
    await graph.upsert_edge("A", "B", dict(original_edge))
    cfg = _cfg(_tok(), embedding_token_limit=1000)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    async def _boom(_ids):
        raise RuntimeError("transient VDB backend error")

    relationships_vdb.delete = _boom

    with pytest.raises(RuntimeError, match="transient VDB backend error"):
        await aedit_relation(
            graph,
            entities_vdb,
            relationships_vdb,
            "A",
            "B",
            {"description": "updated description"},
        )

    # The graph write already happened before the failing VDB delete.
    assert graph.edges[("A", "B")]["description"] == "updated description"
    assert relationships_vdb.upsert_calls == 0


async def test_edit_entity_rename_truncates_content():
    graph = _MemGraph()
    await graph.upsert_node(
        "ALICE", {"entity_id": "ALICE", "description": "short", "source_id": "c1"}
    )
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await _edit_entity_impl(
        graph,
        entities_vdb,
        relationships_vdb,
        "ALICE",
        {"entity_name": "ALICIA", "description": LONG_DESCRIPTION},
    )
    record = entities_vdb.records[operate.compute_mdhash_id("ALICIA", prefix="ent-")]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_edit_entity_rename_truncation_failure_raises_consistency_error():
    """The rename cascade mutates the graph before the entity's own VDB
    content is built; a truncation failure there must not surface as a raw
    TokenBudgetError but as VectorStorageConsistencyError, since the graph
    was already updated by this point."""
    graph = _MemGraph()
    await graph.upsert_node(
        "ALICE", {"entity_id": "ALICE", "description": "short", "source_id": "c1"}
    )
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    with pytest.raises(VectorStorageConsistencyError):
        await _edit_entity_impl(
            graph,
            entities_vdb,
            relationships_vdb,
            "ALICE",
            {"entity_name": "ALICIA", "description": LONG_DESCRIPTION},
        )

    # The rename already happened in the graph -- this is the documented
    # inconsistency window the error message describes, not a bug.
    assert "ALICIA" in graph.nodes
    assert entities_vdb.upsert_calls == 0


async def test_merge_entities_impl_truncates_content_and_wraps_failure():
    graph = _MemGraph()
    await graph.upsert_node(
        "ALICE", {"entity_id": "ALICE", "description": "short", "source_id": "c1"}
    )
    cfg = _cfg(_tok(), embedding_token_limit=20)
    entities_vdb = _MemVDB(cfg)
    relationships_vdb = _MemVDB(cfg)

    await _merge_entities_impl(
        graph,
        entities_vdb,
        relationships_vdb,
        ["ALICE"],
        "ALICE_MERGED",
        target_entity_data={"description": LONG_DESCRIPTION},
    )
    assert "ALICE_MERGED" in graph.nodes
    record = entities_vdb.records[
        operate.compute_mdhash_id("ALICE_MERGED", prefix="ent-")
    ]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20

    # Failure path: the target node is already merged into the graph by this
    # point (step 5), so a truncation failure must be VectorStorageConsistencyError.
    graph2 = _MemGraph()
    await graph2.upsert_node(
        "BOB", {"entity_id": "BOB", "description": "short", "source_id": "c1"}
    )
    cfg2 = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    entities_vdb2 = _MemVDB(cfg2)
    relationships_vdb2 = _MemVDB(cfg2)
    with pytest.raises(VectorStorageConsistencyError):
        await _merge_entities_impl(
            graph2,
            entities_vdb2,
            relationships_vdb2,
            ["BOB"],
            "BOB_MERGED",
            target_entity_data={"description": LONG_DESCRIPTION},
        )
    assert "BOB_MERGED" in graph2.nodes
    assert entities_vdb2.upsert_calls == 0


# --------------------------------------------------------------------------- #
# lightrag.py: ainsert_custom_kg
# --------------------------------------------------------------------------- #


def _make_custom_kg_rag(global_config: dict):
    """A bare LightRAG.__new__ instance wired just enough for
    ainsert_custom_kg's graph/VDB write path — bypasses __init__/__post_init__
    entirely (real dataclass state isn't needed), so _build_global_config is
    replaced with a plain callable returning the fixed config under test."""
    from unittest.mock import AsyncMock
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = global_config["tokenizer"]
    rag._build_global_config = lambda: global_config
    rag.chunk_entity_relation_graph = _MemGraph()
    rag.entities_vdb = _MemVDB(global_config)
    rag.relationships_vdb = _MemVDB(global_config)
    rag._insert_done = AsyncMock(return_value=None)
    return rag


async def test_ainsert_custom_kg_truncates_entity_and_relation_content():
    cfg = _cfg(_tok(), embedding_token_limit=20)
    rag = _make_custom_kg_rag(cfg)

    await rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "PERSON",
                    "description": LONG_DESCRIPTION,
                    "source_id": "chunk-1",
                    "file_path": "f",
                }
            ],
            "relationships": [],
        }
    )

    record = rag.entities_vdb.records[operate.compute_mdhash_id("Alice", prefix="ent-")]
    assert len(cfg["tokenizer"].encode(record["content"])) <= 20


async def test_ainsert_custom_kg_relationship_endpoints_keep_real_entity_data():
    """A relationship endpoint that is ALSO one of the batch's own explicit
    entities must not be treated as a missing/placeholder node. entity_nodes
    is batch-upserted to the graph FIRST, before the relationship-endpoint
    discovery pass runs has_nodes_batch, so the discovery pass correctly
    sees Alice/Bob as already existing and never queues a placeholder
    {description: "UNKNOWN"} node that would clobber the real submitted
    description/entity_type."""
    cfg = _cfg(_tok(), embedding_token_limit=1000)
    rag = _make_custom_kg_rag(cfg)

    await rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "PERSON",
                    "description": "Alice is a scientist.",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "PERSON",
                    "description": "Bob is an engineer.",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
            ],
            "relationships": [
                {
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "Alice and Bob are colleagues.",
                    "keywords": "colleague",
                    "weight": 1.0,
                    "source_id": "chunk-1",
                    "file_path": "f",
                }
            ],
        }
    )

    alice = rag.chunk_entity_relation_graph.nodes["Alice"]
    bob = rag.chunk_entity_relation_graph.nodes["Bob"]
    assert alice["description"] == "Alice is a scientist."
    assert alice["entity_type"] == "PERSON"
    assert bob["description"] == "Bob is an engineer."
    assert bob["entity_type"] == "PERSON"
    assert ("Alice", "Bob") in rag.chunk_entity_relation_graph.edges


async def test_ainsert_custom_kg_truncation_failure_leaves_graph_untouched():
    cfg = _cfg(_AlwaysFailsTruncateTokenizer(), embedding_token_limit=20)
    rag = _make_custom_kg_rag(cfg)

    with pytest.raises(TokenBudgetError):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "PERSON",
                        "description": LONG_DESCRIPTION,
                        "source_id": "chunk-1",
                        "file_path": "f",
                    }
                ],
                "relationships": [],
            }
        )

    assert "Alice" not in rag.chunk_entity_relation_graph.nodes
    assert rag.entities_vdb.upsert_calls == 0


async def test_ainsert_custom_kg_entities_created_but_not_edge_when_relationship_content_fails():
    """Multi-object case: the entity payloads would truncate fine, but the
    relationship's own content cannot. Truncation is front-loaded per-object
    (entity VDB payload before the entity_nodes graph batch write, relation
    VDB payload before the missing-node/edge graph batch writes) rather than
    across the whole operation, so Alice and Bob still land in the graph --
    only the edge (and both VDB upserts) never happen."""
    cfg = _cfg(_FailsOnlyForRelationContentTokenizer(), embedding_token_limit=20)
    rag = _make_custom_kg_rag(cfg)

    with pytest.raises(TokenBudgetError):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "PERSON",
                        "description": "short",
                        "source_id": "chunk-1",
                        "file_path": "f",
                    },
                    {
                        "entity_name": "Bob",
                        "entity_type": "PERSON",
                        "description": "short",
                        "source_id": "chunk-1",
                        "file_path": "f",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "Alice",
                        "tgt_id": "Bob",
                        "description": "short",
                        "keywords": "k",
                        "weight": 1.0,
                        "source_id": "chunk-1",
                        "file_path": "f",
                    }
                ],
            }
        )

    assert "Alice" in rag.chunk_entity_relation_graph.nodes
    assert "Bob" in rag.chunk_entity_relation_graph.nodes
    assert ("Alice", "Bob") not in rag.chunk_entity_relation_graph.edges
    assert rag.entities_vdb.upsert_calls == 0
    assert rag.relationships_vdb.upsert_calls == 0


async def test_ainsert_custom_kg_empty_batch_never_builds_global_config():
    """The empty-batch fast path must not pay for _build_global_config at
    all -- regression guard for the fix that scoped the new truncation call
    to `if all_entities_data or all_relationships_data`."""

    def _boom():
        raise AssertionError("_build_global_config should not be called")

    cfg = _cfg(_tok(), embedding_token_limit=20)
    rag = _make_custom_kg_rag(cfg)
    rag._build_global_config = _boom

    await rag.ainsert_custom_kg({"chunks": [], "entities": [], "relationships": []})
