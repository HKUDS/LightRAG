"""Tests for ``lightrag.tools.export_graph`` (the streaming graph exporter, issue #1937).

The format tests run the exporter against the real ``NetworkXStorage``; the
cursor, consistency and refusal tests use small fakes, because those are
behaviors a healthy NetworkX graph cannot produce on demand.
"""

import csv
import json

import networkx as nx
import pytest

from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.tools import export_graph as eg
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

_NODES = {
    "alice": {
        "entity_id": "alice",
        "entity_type": "person",
        "description": "plain, with a comma",
        "source_id": "chunk-1",
        "rank_hint": "7",
    },
    'He said "hi"': {"entity_id": 'He said "hi"', "entity_type": "quote"},
    "한국": {"entity_id": "한국", "entity_type": "국가", "description": "유니코드"},
    "isolated": {"entity_id": "isolated", "entity_type": "thing"},
}
_EDGES = [
    ("한국", "alice", {"weight": "1.5", "description": "unicode endpoint"}),
    ('He said "hi"', "alice", {"weight": "2.0", "keywords": "k1,k2"}),
]


async def _never_embed(*_args, **_kwargs):
    raise RuntimeError("export tests never embed")


async def _networkx(tmp_path, nodes=_NODES, edges=_EDGES) -> NetworkXStorage:
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace="ws",
        global_config={"working_dir": str(tmp_path / "rag")},
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_never_embed),
    )
    await storage.initialize()
    await storage.upsert_nodes_batch([(k, dict(v)) for k, v in nodes.items()])
    await storage.upsert_edges_batch([(s, t, dict(p)) for s, t, p in edges])
    return storage


def _pair(a, b):
    return tuple(sorted((a, b)))


class FakeGraph:
    """Minimal bounded-iteration graph with hooks for the failure cases."""

    def __init__(self, labels, nodes, edges, *, fail_edges=False):
        self.workspace = "fake"
        self._labels = labels
        self._nodes = nodes
        self._edges = edges
        self._fail_edges = fail_edges
        self.label_cursor_open = False
        self.reads_with_cursor_open = 0

    async def iter_labels(self, batch_size):
        self.label_cursor_open = True
        try:
            for i in range(0, len(self._labels), batch_size):
                yield self._labels[i : i + batch_size]
        finally:
            self.label_cursor_open = False

    async def get_nodes_batch(self, node_ids):
        if self.label_cursor_open:
            # A pool of one: the open label cursor holds the only connection.
            self.reads_with_cursor_open += 1
            raise TimeoutError("no free connection while the label cursor is open")
        return {n: dict(self._nodes[n]) for n in node_ids if n in self._nodes}

    async def iter_edges(self, batch_size):
        if self._fail_edges:
            raise RuntimeError("edge read failed")
        for i in range(0, len(self._edges), batch_size):
            yield [dict(e) for e in self._edges[i : i + batch_size]]


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def test_jsonl_keeps_every_property(tmp_path):
    storage = await _networkx(tmp_path)
    out = tmp_path / "out"

    # batch_size 1 drives the multi-batch path of every phase.
    report = await eg.export_graph(storage, str(out), batch_size=1)

    lines = _read_jsonl(out / "graph.jsonl")
    header, body = lines[0], lines[1:]
    assert header["type"] == "header"
    assert (header["format"], header["version"]) == ("lightrag-graph", 1)
    assert header["workspace"] == "ws"
    nodes = {r["id"]: r["properties"] for r in body if r["type"] == "node"}
    edges = {
        _pair(r["source"], r["target"]): r["properties"]
        for r in body
        if r["type"] == "edge"
    }
    assert nodes == _NODES
    assert edges == {_pair(s, t): p for s, t, p in _EDGES}
    assert (report.nodes, report.edges, report.consistent) == (4, 2, True)
    assert sorted(p.name for p in out.iterdir()) == ["graph.jsonl"]


async def test_csv_fixed_columns_and_extra(tmp_path):
    nodes = dict(_NODES)
    # A stored entity_id that differs from the label must survive in extra.
    nodes["renamed"] = {"entity_id": "old-name", "entity_type": "thing"}
    storage = await _networkx(tmp_path, nodes=nodes)
    out = tmp_path / "out"

    await eg.export_graph(storage, str(out), fmt="csv", batch_size=2)

    with open(out / "nodes.csv", newline="", encoding="utf-8") as f:
        node_rows = {r["entity_id"]: r for r in csv.DictReader(f)}
    with open(out / "edges.csv", newline="", encoding="utf-8") as f:
        edge_rows = list(csv.DictReader(f))
    assert list(node_rows["alice"]) == [*eg.NODE_FIELDS, "extra"]
    assert node_rows["alice"]["description"] == "plain, with a comma"
    assert json.loads(node_rows["alice"]["extra"]) == {"rank_hint": "7"}
    assert node_rows["isolated"]["extra"] == ""
    assert json.loads(node_rows["renamed"]["extra"]) == {"entity_id": "old-name"}
    assert {_pair(r["source"], r["target"]): r["keywords"] for r in edge_rows} == {
        _pair("한국", "alice"): "",
        _pair('He said "hi"', "alice"): "k1,k2",
    }


async def test_graphml_reads_back_and_reports_replaced_chars(tmp_path):
    # A fake, because NetworkXStorage refuses such characters on write while
    # Neo4j, for one, stores them.
    nodes = dict(_NODES)
    nodes["ctrl"] = {"entity_id": "ctrl", "description": "bell\x07here"}
    edges = [{"source": s, "target": t, **p} for s, t, p in _EDGES]
    fake = FakeGraph(list(nodes), nodes, edges)
    out = tmp_path / "out"

    report = await eg.export_graph(fake, str(out), fmt="graphml")

    graph = nx.read_graphml(out / "graph.graphml")
    assert set(graph.nodes) == set(nodes)
    assert graph.nodes["alice"]["entity_type"] == "person"
    assert json.loads(graph.nodes["alice"]["extra"]) == {"rank_hint": "7"}
    assert graph.nodes["ctrl"]["description"] == "bell\ufffdhere"
    assert graph.edges["한국", "alice"]["weight"] == 1.5
    assert report.replaced_xml_chars == 1


async def test_nodes_are_read_after_the_label_cursor_closes(tmp_path):
    labels = [f"n{i}" for i in range(5)]
    fake = FakeGraph(labels, {n: {"entity_id": n} for n in labels}, [])

    report = await eg.export_graph(fake, str(tmp_path / "out"), batch_size=2)

    assert fake.reads_with_cursor_open == 0
    assert report.nodes == 5


async def test_concurrent_changes_are_reported(tmp_path):
    fake = FakeGraph(
        ["a", "b", "gone", "a"],  # "a" listed twice; "gone" deleted mid-export
        {"a": {"entity_id": "a"}, "b": {"entity_id": "b"}},
        [
            {"source": "a", "target": "b", "weight": 1.0},
            {"source": "b", "target": "a", "weight": 2.0},  # reciprocal row
            {"source": "a", "target": "new", "weight": 1.0},  # endpoint added later
        ],
    )

    report = await eg.export_graph(fake, str(tmp_path / "out"))

    assert report.as_dict() | {"files": []} == {
        "graph_storage": "FakeGraph",
        "workspace": "fake",
        "format": "jsonl",
        "files": [],
        "nodes": 2,
        "edges": 3,
        "consistent": False,
        "duplicate_labels_skipped": 1,
        "duplicate_edge_pairs_kept": 1,
        "vanished_nodes": 1,
        "vanished_node_examples": ["gone"],
        "dangling_edges": 1,
        "dangling_edge_examples": [["a", "new"]],
    }


async def test_backend_without_iteration_is_refused(tmp_path):
    class NoIteration(FakeGraph):
        async def iter_labels(self, batch_size):
            raise StorageCapabilityError("NoIteration does not support it")
            yield []

    out = tmp_path / "out"
    with pytest.raises(eg.ExportError, match="does not fall back"):
        await eg.export_graph(NoIteration([], {}, []), str(out))
    assert list(out.iterdir()) == []


async def test_failed_export_keeps_the_previous_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "graph.jsonl").write_text("previous\n", encoding="utf-8")
    fake = FakeGraph(["a"], {"a": {"entity_id": "a"}}, [], fail_edges=True)

    with pytest.raises(eg.ExportError, match="already exists"):
        await eg.export_graph(fake, str(out))
    with pytest.raises(RuntimeError, match="edge read failed"):
        await eg.export_graph(fake, str(out), overwrite=True)

    assert sorted(p.name for p in out.iterdir()) == ["graph.jsonl"]
    assert (out / "graph.jsonl").read_text(encoding="utf-8") == "previous\n"


async def test_cli_exports_the_configured_networkx_graph(tmp_path, monkeypatch, capsys):
    storage = await _networkx(tmp_path)
    await storage.index_done_callback()  # publish the GraphML file the CLI loads
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    out = tmp_path / "cli-out"

    await eg._async_main(
        eg_args := _args(output=str(out), workspace="ws", format="csv")
    )
    first = json.loads(capsys.readouterr().out)
    code = await eg._async_main(eg_args)

    assert (first["nodes"], first["edges"], first["workspace"]) == (4, 2, "ws")
    assert sorted(p.name for p in out.iterdir()) == ["edges.csv", "nodes.csv"]
    assert code == 2
    assert "already exists" in capsys.readouterr().out


def _args(**overrides):
    import argparse

    values = dict(
        format="jsonl",
        output="graph_export",
        workspace=None,
        graph_storage=None,
        batch_size=eg.DEFAULT_BATCH_SIZE,
        overwrite=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)
