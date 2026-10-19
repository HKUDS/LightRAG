"""Business-layer normalization for manual relation mutations (edit/delete).

Mirrors ``test_graph_entity_name_normalization``'s shape: in-process graph
and vector fakes, ``get_storage_keyed_lock`` patched to a no-op so only the
spelling logic runs. The relations endpoints that previously locked the
caller's raw spelling and probed the edge under that spelling now resolve
each side the way ``aedit_entity`` / ``amerge_entities`` do, so a caller
that spelled an endpoint the way the manual create endpoint stored it --
the extraction-normalized form -- can both edit and delete it.
"""

from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.utils import compute_mdhash_id

pytestmark = pytest.mark.offline


class _NoopLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _Graph:
    """Minimal in-process graph storage sufficient for aedit_relation / adelete_by_relation."""

    def __init__(self, nodes=None, edges=None):
        # ``create_relation`` writes the canonical sorted pair; mirror that
        # here so the lookup direction does not matter to tests.
        self.nodes = deepcopy(nodes or {})
        self.edges = {
            tuple(sorted(key)): deepcopy(value)
            for key, value in (edges or {}).items()
        }
        # ``remove_edges`` is what adelete_by_relation uses to commit the
        # mutation; track the calls so tests can assert what was actually
        # removed (not what was queried).
        self.removed_edges = []

    async def has_node(self, entity_name):
        return entity_name in self.nodes

    async def has_edge(self, source, target):
        # The graph is undirected: the create endpoint writes a canonical
        # pair via sorted([src, tgt]); both endpoints must look it up under
        # the same orientation.
        canonical = tuple(sorted([source, target]))
        return canonical in self.edges

    async def get_edge(self, source, target):
        canonical = tuple(sorted([source, target]))
        edge = self.edges.get(canonical)
        return deepcopy(edge) if edge is not None else None

    async def upsert_edge(self, source, target, edge_data):
        canonical = tuple(sorted([source, target]))
        self.edges[canonical] = deepcopy(edge_data)

    async def remove_edges(self, edges):
        for source, target in edges:
            canonical = tuple(sorted([source, target]))
            self.removed_edges.append(canonical)
            self.edges.pop(canonical, None)

    async def index_done_callback(self):
        return None


class _VectorStorage:
    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}
        # Record the deletes so tests can see what id set the delete branch
        # submitted -- the bug was that the raw pair produced an id that
        # missed the VDB row the canonical pair had written.
        self.deleted_ids = []

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for record_id in ids:
            self.deleted_ids.append(record_id)
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


class _KVStorage:
    """Minimal KV storage for chunk tracking; ``get_by_id`` returns None for empty rows."""

    def __init__(self):
        self.global_config = {"workspace": ""}
        self.records = {}

    async def get_by_id(self, key):
        return self.records.get(key)

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        for record_id in ids:
            self.records.pop(record_id, None)

    async def index_done_callback(self):
        return None


_captured_lock_keys: list[list[str]] = []


def _capturing_lock(*args, **kwargs):
    """Lock stub that records the key set each call site passed to it.

    The race-safety claim is that the keyed lock covers both the caller-
    requested spelling and the extraction-normalized spelling for each
    endpoint, so a historical node cannot race its canonical mutation
    under a different spelling. Recording the keys lets tests assert
    that membership directly, instead of relying on the resolution path
    to leak the lock-set composition through observable side effects.
    """
    if args:
        keys = list(args[0])
    else:
        keys = list(kwargs.get("keys", []))
    _captured_lock_keys.append(keys)
    return _NoopLock()


@pytest.fixture(autouse=True)
def patch_graph_lock(monkeypatch):
    _captured_lock_keys.clear()
    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        _capturing_lock,
    )
    return _captured_lock_keys


# ---------- aedit_relation ----------


@pytest.mark.asyncio
async def test_edit_relation_resolves_normalized_endpoint_spelling():
    """Editing with the raw spelling the manual create endpoint rewrote must work."""
    norm_a = "Tesla(US)"
    norm_b = "SpaceX"
    raw_a = "Tesla（US）"
    graph = _Graph(edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}})
    entities_vdb = _VectorStorage()
    relationships_vdb = _VectorStorage()

    result = await utils_graph.aedit_relation(
        graph,
        entities_vdb,
        relationships_vdb,
        raw_a,
        norm_b,
        {"description": "new", "keywords": "k"},
    )

    # aedit_relation returns the get_relation_info envelope; description
    # lives under graph_data.
    assert result["src_entity"] == norm_a
    assert result["tgt_entity"] == norm_b
    assert result["graph_data"]["description"] == "new"
    # ``upsert_edge`` rewrites the edge under the canonical sorted pair with
    # the weight-floor applied (the default 1.0 on a no-evidence edit).
    canonical_key = tuple(sorted([norm_a, norm_b]))
    assert graph.edges[canonical_key]["description"] == "new"
    assert graph.edges[canonical_key]["keywords"] == "k"
    assert graph.edges[canonical_key]["source_id"] == "chunk-1"
    # Lock key set must cover both the caller-raw spelling and the
    # extraction-normalized spelling for each endpoint so a concurrent
    # rename cannot race the canonical mutation under a different
    # spelling.
    assert _captured_lock_keys, "expected get_storage_keyed_lock to be called"
    lock_keys = set(_captured_lock_keys[0])
    assert {raw_a, norm_a, norm_b}.issubset(lock_keys)


@pytest.mark.asyncio
async def test_edit_relation_prefers_exact_legacy_endpoint_key():
    """If both spellings exist, the raw legacy one wins for that endpoint."""
    legacy = "“Ａ 公 司”"
    partner = "SpaceX"
    graph = _Graph(
        nodes={
            legacy: {"entity_id": legacy, "description": "legacy"},
        },
        edges={
            (legacy, partner): {"description": "legacy edge", "source_id": "manual_creation"},
        },
    )

    result = await utils_graph.aedit_relation(
        graph,
        _VectorStorage(),
        _VectorStorage(),
        legacy,
        partner,
        {"description": "renamed"},
    )

    # aedit_relation returns the get_relation_info envelope; description
    # lives under graph_data.
    assert result["src_entity"] == legacy
    assert result["tgt_entity"] == partner
    assert result["graph_data"]["description"] == "renamed"
    # The legacy pair must be the one mutated, not a (norm, partner) row
    # that does not exist. The fixture sorts keys on init, so compare with
    # the canonical sorted pair.
    canonical_key = tuple(sorted([legacy, partner]))
    assert graph.edges[canonical_key]["description"] == "renamed"


@pytest.mark.asyncio
async def test_edit_relation_uses_canonical_storage_key_for_vdb():
    """The VDB id set the edit writes must match what the create wrote under the canonical pair."""
    norm_a = "Tesla(US)"
    norm_b = "SpaceX"
    raw_a = "Tesla（US）"
    graph = _Graph(edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}})
    relationships_vdb = _VectorStorage()
    # `acreate_relation` writes the VDB row under the sorted pair, mirroring
    # `lightrag/utils_graph.py::acreate_relation`'s `vdb_src = min(src, tgt)`
    # convention; mirror that here so the test exercises the same shape.
    canonical_vdb_src, canonical_vdb_tgt = sorted([norm_a, norm_b])
    canonical_rel_id = compute_mdhash_id(
        canonical_vdb_src + canonical_vdb_tgt, prefix="rel-"
    )
    relationships_vdb.records[canonical_rel_id] = {"content": "old"}

    await utils_graph.aedit_relation(
        graph,
        _VectorStorage(),
        relationships_vdb,
        raw_a,
        norm_b,
        {"description": "new", "keywords": "k"},
    )

    # The delete must have targeted both permutations of the resolved
    # pair, so legacy ``rel-`` rows created under either orientation are
    # also swept.
    canonical_rel_id = compute_mdhash_id(
        canonical_vdb_src + canonical_vdb_tgt, prefix="rel-"
    )
    reverse_rel_id = compute_mdhash_id(
        canonical_vdb_tgt + canonical_vdb_src, prefix="rel-"
    )
    assert {canonical_rel_id, reverse_rel_id}.issubset(set(relationships_vdb.deleted_ids))
    # And the new VDB record must use the canonical pair, not the raw pair
    # nor the caller order.
    upserted_ids = set(relationships_vdb.records)
    assert compute_mdhash_id(
        canonical_vdb_src + canonical_vdb_tgt, prefix="rel-"
    ) in upserted_ids
    assert compute_mdhash_id(raw_a + norm_b, prefix="rel-") not in upserted_ids
    assert compute_mdhash_id(norm_a + norm_b, prefix="rel-") not in upserted_ids


@pytest.mark.asyncio
async def test_edit_relation_refuses_normalized_self_loop():
    """Two raw spellings that normalize to one identifier must refuse, not silently edit."""
    norm = "Tesla(US)"
    graph = _Graph(edges={(norm, norm): {"description": "self", "source_id": "chunk-1"}})

    with pytest.raises(ValueError, match="self-loop"):
        await utils_graph.aedit_relation(
            graph,
            _VectorStorage(),
            _VectorStorage(),
            "Tesla（US）",
            "Tesla(US)",
            {"description": "would-be self"},
        )


# ---------- adelete_by_relation ----------


@pytest.mark.asyncio
async def test_delete_relation_resolves_normalized_endpoint_spelling():
    """Deleting with the raw spelling the manual create endpoint rewrote must drop the edge."""
    norm_a = "Tesla(US)"
    norm_b = "SpaceX"
    raw_a = "Tesla（US）"
    graph = _Graph(edges={(norm_a, norm_b): {"description": "old", "source_id": "chunk-1"}})

    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        raw_a,
        norm_b,
    )

    assert result.status == "success"
    assert result.status_code == 200
    assert graph.edges == {}
    assert graph.removed_edges == [tuple(sorted([norm_a, norm_b]))]
    # Lock key set must cover both the caller-raw spelling and the
    # extraction-normalized spelling for each endpoint.
    assert _captured_lock_keys, "expected get_storage_keyed_lock to be called"
    lock_keys = set(_captured_lock_keys[0])
    assert {raw_a, norm_a, norm_b}.issubset(lock_keys)


@pytest.mark.asyncio
async def test_delete_relation_prefers_exact_legacy_endpoint_key():
    """The legacy endpoint key wins; the delete must hit the legacy edge, not a phantom canonical one."""
    legacy = "“Ａ 公 司”"
    partner = "SpaceX"
    graph = _Graph(
        nodes={
            legacy: {"entity_id": legacy, "description": "legacy"},
        },
        edges={
            (legacy, partner): {"description": "legacy edge", "source_id": "manual_creation"},
        },
    )

    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        legacy,
        partner,
    )

    assert result.status == "success"
    assert graph.removed_edges == [tuple(sorted([legacy, partner]))]
    assert graph.edges == {}


@pytest.mark.asyncio
async def test_delete_relation_returns_not_found_for_missing_edge_under_normalized_spelling():
    """The legacy key for the normalized-spelling path returns 404, matching the pre-fix message shape."""
    graph = _Graph(edges={})  # empty graph

    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        "Tesla（US）",
        "SpaceX",
    )

    assert result.status == "not_found"
    assert result.status_code == 404
    # The not_found message must carry the resolved pair, so the caller
    # learns the canonical spelling they would have to use. The order is
    # the resolution order (resolved source first, resolved target second),
    # which is the original caller's source-first / target-second framing.
    assert result.message == "Relation from 'Tesla(US)' to 'SpaceX' does not exist"


@pytest.mark.asyncio
async def test_delete_relation_refuses_normalized_self_loop():
    """Two raw spellings that normalize to one identifier must refuse, not silently delete.

    ``adelete_by_relation`` catches every exception and returns a
    ``DeletionResult(status="fail", status_code=500)``; the contract that
    distinguishes self-loop from a real graph error is the message text.
    """
    norm = "Tesla(US)"
    graph = _Graph(edges={(norm, norm): {"description": "self", "source_id": "chunk-1"}})

    result = await utils_graph.adelete_by_relation(
        graph,
        _VectorStorage(),
        "Tesla（US）",
        "Tesla(US)",
    )

    assert result.status == "fail"
    assert result.status_code == 500
    assert "self-loop" in result.message
    assert graph.edges == {(norm, norm): {"description": "self", "source_id": "chunk-1"}}
