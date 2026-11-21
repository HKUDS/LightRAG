"""Permanent offline mechanics smoke: ingest -> arms -> metrics -> report.

Tiny deterministic fixture on FILE backends with fake hash embeddings and the
corpus-aware fake LLM. SYNTHETIC infrastructure: this pins the plumbing, not
retrieval quality. The lexical arm, write-amp and backfill require PG and must
be reported as skipped here, never faked. The full PG mechanics run is a
manual-QA artifact of the harness.
"""

from __future__ import annotations

import json
import sys

import pytest

from lightrag.evaluation.hybrid_bench.cli import parse_args, run_bench

pytestmark = pytest.mark.offline

CORPUS = [
    {"id": "d1", "text": "光耦合器用于信号隔离，防止电气干扰。"},
    {"id": "d2", "text": "伺服电机控制关节运动，定位精度高。"},
    {"id": "d3", "text": "行星齿轮减速器传递大扭矩。"},
    {"id": "d4", "text": "锂电池为机器人平台提供能源。"},
]
QUERIES = [
    {"id": "q1", "text": "光耦合器如何隔离信号？"},
    {"id": "q2", "text": "伺服电机控制什么运动？"},
    {"id": "q3", "text": "行星齿轮传递什么？"},
]
QRELS = "q1\td1\t1\nq2\td2\t1\nq3\td3\t1\n"


def _write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in CORPUS) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in QUERIES) + "\n",
        encoding="utf-8",
    )
    (root / "qrels.tsv").write_text(QRELS, encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "domain": "terminology",
                "language": "zh",
                "source": "https://example.org/synthetic",
                "license": "test-fixture",
                "subset_protocol": "tiny mechanics fixture",
                "counts": {"corpus": 4, "queries": 3},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_file_backend_mechanics_smoke(tmp_path):
    dataset_dir = tmp_path / "tiny"
    _write_fixture(dataset_dir)
    output_dir = tmp_path / "out"
    args = parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--storage",
            "file",
            "--embedding-backend",
            "fake-hash",
            "--embedding-dim",
            "32",
            "--llm-backend",
            "fake",
            "--extraction-cap",
            "4",
            "--cosine-threshold",
            "0.1",
            "--working-dir",
            str(tmp_path / "working"),
        ]
    )
    results = await run_bench(args)

    assert results["synthetic"] is True
    assert "sentence_transformers" not in sys.modules
    assert (output_dir / "report.json").is_file()
    markdown = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC RUN" in markdown
    assert "Recovery: 0 round(s)" in markdown

    dataset = results["datasets"][0]
    assert dataset["ingest"]["chunk_count"] == 4
    assert dataset["ingest"]["extracted_doc_ids"] == ["d1", "d2", "d3", "d4"]
    assert dataset["ingest"]["recovery_rounds_used"] == 0
    assert dataset["ingest"]["recovered_doc_ids"] == []
    assert dataset["ingest"]["persistently_failed_doc_ids"] == []
    assert dataset["ingest"]["duplicate_aliased_doc_ids"] == []
    assert dataset["ingest"]["moderation_fallback_doc_ids"] == []
    # At least one extraction call per passage (gleaning may add more);
    # entity seeding extracts keywords once per query (run-cached); ablation
    # and latency still inject keywords, answers came from the answer-latency
    # protocol.
    assert dataset["fake_llm_calls"]["extraction"] >= 4
    assert dataset["fake_llm_calls"]["keywords"] == len(QUERIES)
    assert dataset["fake_llm_calls"]["answer"] > 0
    assert dataset["entity_seeding_protocol"] == "production-faithful-multi-keyword"
    assert dataset["entity_seeding_fallbacks"] == {}

    chunk_metrics = dataset["chunk_metrics"]
    assert set(chunk_metrics) == {"dense", "hybrid"}
    for arm in ("dense", "hybrid"):
        for metric in ("recall@10", "ndcg@10"):
            entry = chunk_metrics[arm][metric]
            assert entry["n_queries"] == 3
            assert 0.0 <= entry["mean"] <= 1.0
            lo, hi = entry["ci95"]
            assert lo <= entry["mean"] <= hi or lo <= hi
    assert dataset["entity_metrics"], "fake KG must yield derivable truth"
    for arm in ("dense", "hybrid"):
        entry = dataset["entity_metrics"][arm]["hit@10"]
        assert 0.0 <= entry["mean"] <= 1.0

    assert set(dataset["ablation"]) == {"entity_only", "chunk_only", "both"}
    assert "Ablation:" in markdown
    latency = dataset["latency"]
    for phase in ("retrieval", "answer"):
        assert latency[phase]["dense"]["p95"] > 0
        assert latency[phase]["hybrid"]["p95"] > 0
        assert latency[phase]["dense"]["query_slots"] >= 100
        assert latency[phase]["dense"]["rounds"] == 3
        for arm in ("dense", "hybrid"):
            samples = latency[phase][f"all_samples_{arm}"]
            assert len(samples) == latency[phase][arm]["query_slots"] * 3
            assert all(value > 0 for value in samples)

    assert "lexical-arm" in dataset["skipped"]
    assert "write-amp" in dataset["skipped"]
    assert "backfill" in dataset["skipped"]
    assert dataset["rerank"]["status"] == "not-run"

    statuses = {v["status"] for v in results["gates"]}
    assert statuses <= {"pass", "fail", "residue", "report", "not-run"}
    gate_ids = {v["gate_id"]: v["status"] for v in results["gates"]}
    assert gate_ids["write-amp/upsert-p95"] == "not-run"
    assert gate_ids["backfill/throughput"] == "not-run"
    assert gate_ids["ablation/recall@10"] in {"pass", "fail"}
