"""Integration tests: the migration tool against the REAL NetworkXStorage.

Every write-path test so far ran authored fakes on both sides. This leg
runs the tool's enumerate -> check -> batch-write -> verify -> compensate
pipeline against actual storage code (NetworkXStorage is fully functional
offline: in-memory graph + GraphML file under a working dir), attacking
the fake-fidelity risk directly. The shipped pair allowlist stays
untouched; each test that needs the NetworkX -> NetworkX pair extends
ALLOWED_MIGRATION_PAIRS via monkeypatch, which doubles as the first
exercise of that extension seam.

What this leg DOES cover: real enumeration (get_all_nodes / get_all_edges
shapes as actually produced), real batch writes (including NetworkX's own
silent endpoint auto-creation in add_edge), real remove_* compensation
(including the absent-id no-op contract), instance isolation, and the
verification plumbing end to end.

What it does NOT cover, deliberately:

- The reciprocal / parallel directed-edge axes: NetworkX ``get_all_edges``
  returns each undirected edge exactly once (networkx_impl.py:727-740), so
  it cannot express AGE's directed reciprocal rows. Those stay synthesized
  in the unit tests in test_migrate_graph_storage.py.
- PGTable's entity_id injection requirement as a TARGET behavior: NetworkX
  imposes no such constraint, so the tool's entity_id gate is exercised
  here only from the source side (see the attribute-less node test).
- Real SQL / canonicalization semantics: covered by the mocked pgtable
  unit harness (tests/kg/pgtable_impl/) and ultimately the manual e2e run.
- Auto-created endpoints as part of the WRITTEN set: a legal NetworkX
  source cannot enumerate an edge whose endpoint it does not also
  enumerate as a node (add_edge creates the node), so on this pair the
  endpoint union never exceeds the migrated node ids. That axis stays with
  the step-2 fake tests.
"""

import pytest

import lightrag.tools.migrate_graph_storage as mgs
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc
from lightrag.tools.migrate_graph_storage import (
    MigrationDataError,
    MigrationPreconditionError,
    MigrationWriteError,
    canonicalize_edge,
    migrate_graph,
)

pytestmark = pytest.mark.offline

# Unicode, quote-bearing, and self-loop fixtures, mirroring the id shapes
# the pgtable harness exercises.
_NODES: dict[str, dict[str, str]] = {
    "alice": {"entity_id": "alice", "entity_type": "person", "description": "plain"},
    'He said "hi"': {"entity_id": 'He said "hi"', "entity_type": "quote"},
    "한국": {"entity_id": "한국", "entity_type": "국가", "description": "유니코드"},
    "loop": {"entity_id": "loop", "entity_type": "thing"},
}
_EDGES: list[tuple[str, str, dict[str, str]]] = [
    ("한국", "alice", {"weight": "1.0", "description": "unicode endpoint"}),
    ('He said "hi"', "alice", {"weight": "2.0", "source_id": "chunk-1"}),
    ("loop", "loop", {"weight": "3.0", "description": "self-loop"}),
]


@pytest.fixture
def allow_networkx_pair(monkeypatch):
    """Extend the pair allowlist for this test only — never the shipped one."""
    monkeypatch.setattr(
        mgs,
        "ALLOWED_MIGRATION_PAIRS",
        mgs.ALLOWED_MIGRATION_PAIRS | {("NetworkXStorage", "NetworkXStorage")},
    )


async def _never_embed(*_args, **_kwargs):
    raise RuntimeError("graph migration tests never embed")


async def _storage(tmp_path, namespace: str) -> NetworkXStorage:
    initialize_share_data()
    storage = NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        # Loud stub, matching the tool's _build_graph_storage (see the
        # rationale there): graph storages tolerate None at runtime but the
        # annotation does not promise it, so honor the declared contract
        # the way rebuild_vdb does.
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_never_embed),
    )
    await storage.initialize()
    return storage


async def _seeded_source(tmp_path) -> NetworkXStorage:
    source = await _storage(tmp_path, "migration_src")
    await source.upsert_nodes_batch(
        [(node_id, dict(props)) for node_id, props in _NODES.items()]
    )
    await source.upsert_edges_batch(
        [(src, tgt, dict(props)) for src, tgt, props in _EDGES]
    )
    return source


def _node_map(nodes: list[dict]) -> dict[str, dict]:
    return {n["id"]: {k: v for k, v in n.items() if k != "id"} for n in nodes}


def _canonical_edge_map(edges: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        canonicalize_edge(e["source"], e["target"]): {
            k: v for k, v in e.items() if k not in ("source", "target")
        }
        for e in edges
    }


class TestRealRoundTrip:
    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_round_trip_preserves_graph(self, tmp_path):
        source = await _seeded_source(tmp_path)
        target = await _storage(tmp_path, "migration_tgt")

        report = await migrate_graph(source, target)

        assert report.verified
        assert report.verification is not None and report.verification.ok
        assert report.node_count == len(_NODES)
        assert report.edge_count == len(_EDGES)
        # Target content equals source content, read back through the REAL
        # enumeration path on both sides.
        assert _node_map(await target.get_all_nodes()) == _node_map(
            await source.get_all_nodes()
        )
        assert _canonical_edge_map(await target.get_all_edges()) == (
            _canonical_edge_map(await source.get_all_edges())
        )
        # Instance isolation: the source slice is intact after migration.
        assert set(_node_map(await source.get_all_nodes())) == set(_NODES)

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_non_empty_real_target_rejected(self, tmp_path):
        source = await _seeded_source(tmp_path)
        target = await _storage(tmp_path, "migration_tgt")
        await target.upsert_nodes_batch([("pre", {"entity_id": "pre"})])
        with pytest.raises(MigrationPreconditionError):
            await migrate_graph(source, target)
        assert set(_node_map(await target.get_all_nodes())) == {"pre"}

    async def test_shipped_allowlist_still_rejects_networkx(self, tmp_path):
        # No fixture: the SHIPPED allowlist must reject this pair — the
        # extension above is test-scoped, not a loosening of the tool.
        source = await _seeded_source(tmp_path)
        target = await _storage(tmp_path, "migration_tgt")
        with pytest.raises(MigrationPreconditionError):
            await migrate_graph(source, target)

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_attributeless_source_node_fails_closed(self, tmp_path):
        # Real-backend wrinkle the fakes never showed: NetworkX add_edge
        # silently creates an attribute-less node for an unknown endpoint,
        # and get_all_nodes then enumerates it with NO entity_id. The tool
        # must fail closed on it before any write.
        source = await _storage(tmp_path, "migration_src")
        await source.upsert_nodes_batch([("a", {"entity_id": "a"})])
        await source.upsert_edges_batch([("a", "ghost", {"weight": "1"})])
        target = await _storage(tmp_path, "migration_tgt")
        with pytest.raises(MigrationDataError):
            await migrate_graph(source, target)
        assert await target.get_all_nodes() == []


class TestRealCompensation:
    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_edge_batch_failure_compensates_on_real_target(
        self, tmp_path, monkeypatch
    ):
        # AC-b against real storage: inject a failure after the first edge
        # lands, then let the REAL remove_edges / remove_nodes compensate.
        source = await _seeded_source(tmp_path)
        target = await _storage(tmp_path, "migration_tgt")
        # Data outside the written set: a sibling slice in the same working
        # dir. Compensation must not touch it.
        other = await _storage(tmp_path, "migration_other")
        await other.upsert_nodes_batch([("keep", {"entity_id": "keep"})])

        real_upsert_edges = target.upsert_edges_batch

        async def failing_upsert_edges(edges):
            await real_upsert_edges(edges[:1])
            raise RuntimeError("injected failure after first edge")

        removed_nodes: list[str] = []
        removed_edges: list[tuple[str, str]] = []
        real_remove_nodes = target.remove_nodes
        real_remove_edges = target.remove_edges

        async def recording_remove_nodes(nodes):
            removed_nodes.extend(nodes)
            await real_remove_nodes(nodes)

        async def recording_remove_edges(edges):
            removed_edges.extend(edges)
            await real_remove_edges(edges)

        monkeypatch.setattr(target, "upsert_edges_batch", failing_upsert_edges)
        monkeypatch.setattr(target, "remove_nodes", recording_remove_nodes)
        monkeypatch.setattr(target, "remove_edges", recording_remove_edges)

        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated
        # Exactly the derived written superset was passed to the real
        # remove_* — nothing more.
        assert removed_nodes == report.written_node_ids == sorted(_NODES)
        assert removed_edges == report.written_edge_keys
        # The real target slice is clean again; the sibling slice survived.
        assert await target.get_all_nodes() == []
        assert await target.get_all_edges() == []
        assert set(_node_map(await other.get_all_nodes())) == {"keep"}

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_node_batch_failure_absent_ids_are_noops(self, tmp_path, monkeypatch):
        # Failure INSIDE the node batch: most written ids and every edge key
        # were never applied, so compensation exercises the real backend's
        # absent-id no-op contract with the full superset.
        source = await _seeded_source(tmp_path)
        target = await _storage(tmp_path, "migration_tgt")

        real_upsert_nodes = target.upsert_nodes_batch

        async def failing_upsert_nodes(nodes):
            await real_upsert_nodes(nodes[:1])
            raise RuntimeError("injected failure after first node")

        monkeypatch.setattr(target, "upsert_nodes_batch", failing_upsert_nodes)

        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated
        assert report.compensation_error is None
        assert await target.get_all_nodes() == []
        assert await target.get_all_edges() == []
