"""Offline contract tests for the DuckDB cold-query job and recall harness.

Cases that return hits drive the real ``duckdb`` CLI (the shipped surface), so
this file costs a few seconds; the subprocess is the behavior under test, not a
setup shortcut. The stale-tag case proves a tag behind the latest version is refused
before any scan can answer from newer rows, and every fail-loud case asserts
the specific error rather than "it did not return hits".
"""

import json
import shutil

import pytest

from lightrag.kg.lancedb_cold_query import (
    ColdQueryConfig,
    ColdQueryError,
    cold_search,
    main,
)
from lightrag.kg.lancedb_cold_recall import paired_bootstrap_recall, recall_at_k
from lightrag.kg.lancedb_impl import export_snapshot

from .conftest import CHUNKS, expected_vector

pytestmark = [
    pytest.mark.offline,
    pytest.mark.skipif(shutil.which("duckdb") is None, reason="duckdb CLI is required"),
]

_TAG = "snap"


def _config(dest, **overrides):
    return ColdQueryConfig(dest=str(dest), **{"tag": _TAG, **overrides})


@pytest.fixture
async def snapshot(tmp_path, seeded):
    rag, embedder = seeded
    dest = tmp_path / "lake"
    await export_snapshot(rag, str(dest), _TAG)
    return rag, embedder, dest


async def test_cold_search_matches_hot_top_k_with_cosine_distances(snapshot):
    rag, embedder, dest = snapshot
    vector = expected_vector(CHUNKS["chunk-1"]["content"])
    hot = await rag.chunks_vdb.query("ignored", top_k=10, query_embedding=vector)
    embedder.armed = True

    hits = cold_search(_config(dest), vector, top_k=10)

    assert embedder.export_calls == 0
    assert [hit.id for hit in hits] == ["chunk-1", "chunk-2", "chunk-3"]
    assert hits[0].distance == 0.0
    assert hits[2].distance == pytest.approx(1 - 0.9047618692287723, abs=1e-6)
    assert recall_at_k([hit.id for hit in hits], [row["id"] for row in hot], 3) == 1.0


async def test_tag_behind_latest_version_is_refused(tmp_path, seeded):
    rag, _ = seeded
    dest = tmp_path / "lake"
    await export_snapshot(rag, str(dest), "v1")
    extra = {
        "content": "gamma chunk",
        "full_doc_id": "doc-2",
        "chunk_order_index": 1,
        "tokens": 2,
        "file_path": "b.txt",
    }
    await rag.text_chunks.upsert({"chunk-4": extra})
    await rag.chunks_vdb.upsert(
        {
            "chunk-4": {
                "content": extra["content"],
                "full_doc_id": "doc-2",
                "file_path": "b.txt",
            }
        }
    )
    await rag.text_chunks.index_done_callback()
    await rag.chunks_vdb.index_done_callback()
    await export_snapshot(rag, str(dest), "v2")
    vector = expected_vector(CHUNKS["chunk-1"]["content"])

    with pytest.raises(ColdQueryError, match="behind the latest version"):
        cold_search(_config(dest, tag="v1"), vector, top_k=10)

    hits = cold_search(_config(dest, tag="v2"), vector, top_k=10)
    assert "chunk-4" in {hit.id for hit in hits}


async def test_wrong_query_dimension_fails_loud(snapshot):
    _, _, dest = snapshot
    with pytest.raises(ColdQueryError, match="dimension"):
        cold_search(_config(dest), [1.0, 2.0, 3.0], top_k=3)


async def test_missing_table_fails_loud(snapshot):
    _, _, dest = snapshot
    with pytest.raises(ColdQueryError, match="cannot open table 'missing'"):
        cold_search(_config(dest, table="missing"), [1.0, 0.25, 0.5, 0.75], top_k=3)


async def test_missing_tag_fails_loud(snapshot):
    _, _, dest = snapshot
    with pytest.raises(ColdQueryError, match="not found on table"):
        cold_search(
            _config(dest, tag="absent"),
            expected_vector(CHUNKS["chunk-1"]["content"]),
            3,
        )


async def test_l2_metric_orders_by_euclidean_distance(snapshot):
    _, _, dest = snapshot
    vector = expected_vector(CHUNKS["chunk-1"]["content"])

    hits = cold_search(_config(dest, metric="l2"), vector, top_k=10)

    assert [hit.id for hit in hits] == ["chunk-1", "chunk-2", "chunk-3"]
    assert hits[0].distance == 0.0
    assert hits[2].distance == pytest.approx(2.0)


async def test_missing_duckdb_binary_fails_loud(snapshot):
    _, _, dest = snapshot
    with pytest.raises(ColdQueryError, match="cannot run duckdb"):
        cold_search(
            _config(dest, duckdb_bin="/no/such/duckdb"),
            expected_vector(CHUNKS["chunk-1"]["content"]),
            top_k=3,
        )


@pytest.mark.parametrize(
    ("cold", "hot", "k", "expected"),
    [
        (["a", "b"], ["b", "a"], 2, 1.0),
        (["x", "y"], ["a", "b"], 2, 0.0),
        (["a"], ["a", "b"], 2, 0.5),
        (["a", "b", "c"], ["a", "b"], 10, 1.0),
        ([], [], 10, 0.0),
    ],
    ids=["full", "disjoint", "partial", "short-hot", "empty-hot"],
)
def test_recall_at_k(cold, hot, k, expected):
    assert recall_at_k(cold, hot, k) == expected


def test_paired_bootstrap_is_deterministic_and_bounds_the_mean():
    cold = [["a", "b"], ["a", "c"], ["x"]]
    hot = [["a", "b"], ["a", "b"], ["y"]]

    first = paired_bootstrap_recall(cold, hot, 2, samples=500, seed=11)
    second = paired_bootstrap_recall(cold, hot, 2, samples=500, seed=11)

    assert first == second
    assert first.per_query == (1.0, 0.5, 0.0)
    assert first.mean == pytest.approx(0.5)
    assert 0.0 <= first.ci_low <= first.mean <= first.ci_high <= 1.0
    single = paired_bootstrap_recall([["a"]], [["a"]], 1, samples=10)
    assert (single.ci_low, single.ci_high) == (1.0, 1.0)


@pytest.mark.parametrize("source", ["inline", "file"])
async def test_cli_emits_json_hits(snapshot, tmp_path, capsys, source):
    _, _, dest = snapshot
    vector = expected_vector(CHUNKS["chunk-3"]["content"])
    args = ["--dest", str(dest), "--tag", _TAG, "--top-k", "2"]
    if source == "file":
        path = tmp_path / "query.json"
        path.write_text(json.dumps(vector), encoding="utf-8")
        args += ["--query-vector-file", str(path)]
    else:
        args += ["--query-vector", json.dumps(vector)]

    main(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == _TAG
    assert payload["metric"] == "cosine"
    assert payload["top_k"] == 2
    assert isinstance(payload["latency_ms"], float)
    assert [hit["id"] for hit in payload["hits"]] == ["chunk-3", "chunk-1"]


def test_cli_refuses_two_query_vector_sources(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--dest",
                "unused",
                "--tag",
                _TAG,
                "--query-vector",
                "[1, 0.25, 0.5, 0.75]",
                "--query-vector-file",
                "unused.json",
            ]
        )
    assert exc_info.value.code == 1
    assert "exactly one" in capsys.readouterr().err
