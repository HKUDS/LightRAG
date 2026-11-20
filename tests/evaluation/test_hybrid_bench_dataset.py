"""Normalized-dataset contract pins: happy path and fail-loud edge cases."""

from __future__ import annotations

import json

import pytest

from lightrag.evaluation.hybrid_bench.dataset import (
    DatasetError,
    discover_datasets,
    load_dataset,
)

pytestmark = pytest.mark.offline

_UNICODE_LINE_BREAKS = [
    pytest.param("\u0085", id="u0085-nel"),
    pytest.param("\u2028", id="u2028-ls"),
    pytest.param("\u2029", id="u2029-ps"),
]


def _write_dataset(root, corpus=None, queries=None, qrels=None, meta=None):
    root.mkdir(parents=True, exist_ok=True)
    corpus = (
        corpus
        if corpus is not None
        else [
            {"id": "d1", "text": "光耦合器隔离信号。"},
            {"id": "d2", "text": "Servo motors drive joints."},
        ]
    )
    queries = (
        queries
        if queries is not None
        else [
            {"id": "q1", "text": "光耦合器如何隔离信号？"},
            {"id": "q2", "text": "What drives robot joints?"},
        ]
    )
    qrels = qrels if qrels is not None else "q1\td1\t1\nq2\td2\t2\nq1\td2\t0\n"
    meta = (
        meta
        if meta is not None
        else {
            "domain": "terminology",
            "language": "zh",
            "source": "https://example.org",
            "license": "CC-BY-4.0",
            "subset_protocol": "full dev split",
            "counts": {"corpus": 2, "queries": 2},
        }
    )
    (root / "corpus.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in corpus) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in queries) + "\n",
        encoding="utf-8",
    )
    (root / "qrels.tsv").write_text(qrels, encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return root


def test_load_dataset_happy_path_and_zero_score_rows(tmp_path):
    dataset = load_dataset(_write_dataset(tmp_path / "ds"))
    assert dataset.name == "ds"
    assert dataset.corpus == {
        "d1": "光耦合器隔离信号。",
        "d2": "Servo motors drive joints.",
    }
    assert dataset.positives("q1") == {"d1": 1}
    assert dataset.positives("q2") == {"d2": 2}
    assert dataset.qrels["q1"]["d2"] == 0
    assert dataset.positive_corpus_ids() == {"d1", "d2"}
    assert dataset.meta.domain == "terminology"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: (r / "corpus.jsonl").unlink(), "missing required file"),
        (
            lambda r: (r / "corpus.jsonl").write_text("\n", encoding="utf-8"),
            "blank line",
        ),
        (
            lambda r: (r / "corpus.jsonl").write_text("{bad", encoding="utf-8"),
            "invalid JSON",
        ),
        (
            lambda r: (r / "corpus.jsonl").write_text("[1]", encoding="utf-8"),
            "expected a JSON object",
        ),
        (
            lambda r: (r / "corpus.jsonl").write_text(
                '{"id": "", "text": "x"}', encoding="utf-8"
            ),
            "non-empty string",
        ),
        (
            lambda r: (r / "corpus.jsonl").write_text(
                '{"id": "d1", "text": ""}', encoding="utf-8"
            ),
            "non-empty string",
        ),
        (
            lambda r: (r / "corpus.jsonl").write_text(
                '{"id": "d1", "text": "a"}\n{"id": "d1", "text": "b"}',
                encoding="utf-8",
            ),
            "duplicate id",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text("q1\td1\n", encoding="utf-8"),
            "3 tab-separated fields",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text("qX\td1\t1\n", encoding="utf-8"),
            "unknown query id",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text("q1\tdX\t1\n", encoding="utf-8"),
            "unknown corpus id",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text("q1\td1\thigh\n", encoding="utf-8"),
            "not an integer",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text("q1\td1\t-1\n", encoding="utf-8"),
            "negative score",
        ),
        (
            lambda r: (r / "qrels.tsv").write_text(
                "q1\td1\t1\nq1\td1\t2\n", encoding="utf-8"
            ),
            "duplicate qrels pair",
        ),
        (
            lambda r: (r / "meta.json").write_text(
                json.dumps({"domain": "other"}), encoding="utf-8"
            ),
            "domain must be one of",
        ),
        (
            lambda r: (r / "meta.json").write_text(
                json.dumps(
                    {
                        "domain": "general",
                        "language": "fr",
                        "source": "s",
                        "license": "l",
                        "subset_protocol": "p",
                        "counts": {},
                    }
                ),
                encoding="utf-8",
            ),
            "language must be one of",
        ),
        (
            lambda r: (r / "meta.json").write_text(
                json.dumps(
                    {
                        "domain": "general",
                        "language": "en",
                        "source": "s",
                        "license": "l",
                        "subset_protocol": "p",
                        "counts": [],
                    }
                ),
                encoding="utf-8",
            ),
            "counts must be an object",
        ),
    ],
    ids=[
        "missing-corpus",
        "blank-line",
        "bad-json",
        "non-object-row",
        "empty-id",
        "empty-text",
        "duplicate-id",
        "qrels-field-count",
        "qrels-unknown-query",
        "qrels-unknown-corpus",
        "qrels-non-int-score",
        "qrels-negative-score",
        "qrels-duplicate-pair",
        "meta-bad-domain",
        "meta-bad-language",
        "meta-bad-counts",
    ],
)
def test_malformed_dataset_fails_loud(tmp_path, mutate, message):
    root = _write_dataset(tmp_path / "ds")
    mutate(root)
    with pytest.raises(DatasetError, match=message):
        load_dataset(root)


def test_discover_datasets_single_and_parent(tmp_path):
    single = _write_dataset(tmp_path / "single")
    assert discover_datasets(single) == [single]
    parent = tmp_path / "parent"
    beta = _write_dataset(parent / "beta")
    alpha = _write_dataset(parent / "alpha")
    (parent / "not_a_dataset").mkdir()
    assert discover_datasets(parent) == [alpha, beta]


def test_discover_datasets_empty_root_fails_loud(tmp_path):
    with pytest.raises(DatasetError, match="no dataset directories"):
        discover_datasets(tmp_path)
    with pytest.raises(DatasetError, match="not a directory"):
        discover_datasets(tmp_path / "absent")


@pytest.mark.parametrize("table", ["corpus", "queries"])
@pytest.mark.parametrize("separator", _UNICODE_LINE_BREAKS)
def test_unicode_line_breaks_inside_text_load_as_one_record(tmp_path, table, separator):
    text = f"alpha{separator}beta{separator}gamma"
    rows = [{"id": "d1", "text": text}, {"id": "d2", "text": "plain"}]
    qrels = "q1\td1\t1\n" if table == "corpus" else "d1\td1\t1\n"
    dataset = load_dataset(
        _write_dataset(tmp_path / "ds", **{table: rows}, qrels=qrels)
    )
    assert getattr(dataset, table) == {"d1": text, "d2": "plain"}


@pytest.mark.parametrize("separator", _UNICODE_LINE_BREAKS)
def test_unicode_line_breaks_inside_qrels_ids_load_as_one_row(tmp_path, separator):
    query_id = f"q1{separator}x"
    corpus_id = f"d1{separator}x"
    dataset = load_dataset(
        _write_dataset(
            tmp_path / "ds",
            corpus=[{"id": corpus_id, "text": "a"}],
            queries=[{"id": query_id, "text": "b"}],
            qrels=f"{query_id}\t{corpus_id}\t1\n",
        )
    )
    assert dataset.qrels == {query_id: {corpus_id: 1}}


def test_interior_blank_line_rejected_with_line_number(tmp_path):
    root = _write_dataset(tmp_path / "ds")
    (root / "corpus.jsonl").write_text(
        '{"id": "d1", "text": "a"}\n\n{"id": "d2", "text": "b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError) as excinfo:
        load_dataset(root)
    assert "corpus.jsonl:2: blank line" in str(excinfo.value)


@pytest.mark.parametrize("trailing_newline", [True, False], ids=["with", "without"])
def test_single_trailing_newline_is_tolerated(tmp_path, trailing_newline):
    root = _write_dataset(tmp_path / "ds")
    payload = '{"id": "d1", "text": "a"}\n{"id": "d2", "text": "b"}'
    if trailing_newline:
        payload += "\n"
    (root / "corpus.jsonl").write_text(payload, encoding="utf-8")
    assert load_dataset(root).corpus == {"d1": "a", "d2": "b"}
