"""Loader for the fixed normalized benchmark dataset contract.

Contract, one directory per dataset:
- ``corpus.jsonl``: one JSON object per line, ``{"id": str, "text": str}``.
- ``queries.jsonl``: same shape as the corpus.
- ``qrels.tsv``: ``query_id<TAB>corpus_id<TAB>score`` per line; score is an
  integer, ``>= 1`` means relevant. Zero-score rows load but are NOT positive
  relevance; negative scores are rejected.
- ``meta.json``: ``domain`` in ``terminology``/``general``, ``language`` in
  ``zh``/``en``, plus ``source``, ``license``, ``subset_protocol`` strings and
  a ``counts`` object.

Rules:
- Every id is a non-empty string; duplicate ids, unknown qrels references,
  duplicate qrels pairs and malformed rows raise ``DatasetError`` naming the
  file and line. Nothing is silently dropped or repaired.
- Passage ids become LightRAG document ids at ingest, so qrels corpus ids map
  back through chunk ``full_doc_id`` attribution.
- A query with no positive qrels row keeps its place in the dataset; scoring
  reports it separately instead of inventing a zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class DatasetError(ValueError):
    """A normalized dataset violates the contract; the message names the spot."""


_DOMAINS = ("terminology", "general")
_LANGUAGES = ("zh", "en")


@dataclass(frozen=True, slots=True)
class DatasetMeta:
    domain: str
    language: str
    source: str
    license: str
    subset_protocol: str
    counts: dict


@dataclass(frozen=True, slots=True)
class BenchDataset:
    name: str
    meta: DatasetMeta
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def positives(self, query_id: str) -> dict[str, int]:
        """Relevance scores ``>= 1`` for one query; zero-score rows excluded."""
        return {
            corpus_id: score
            for corpus_id, score in self.qrels.get(query_id, {}).items()
            if score >= 1
        }

    def positive_corpus_ids(self) -> set[str]:
        """Union of every query's positive corpus ids, for the extraction cap."""
        result: set[str] = set()
        for query_id in self.queries:
            result.update(self.positives(query_id))
        return result


def _split_lines(text: str) -> list[str]:
    """Split file text into records on ``"\\n"`` only.

    ``str.splitlines`` also breaks on U+0085, U+2028 and U+2029, which are
    ordinary characters inside a JSON string value. Only a single trailing
    newline at EOF is dropped; every other blank line stays in place so the
    caller can reject it by line number.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def _load_jsonl(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise DatasetError(f"{path}: missing required file")
    records: dict[str, str] = {}
    for line_no, line in enumerate(
        _split_lines(path.read_text(encoding="utf-8")), start=1
    ):
        if not line.strip():
            raise DatasetError(f"{path}:{line_no}: blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"{path}:{line_no}: expected a JSON object")
        record_id = row.get("id")
        text = row.get("text")
        if not isinstance(record_id, str) or not record_id:
            raise DatasetError(f"{path}:{line_no}: id must be a non-empty string")
        if not isinstance(text, str) or not text:
            raise DatasetError(f"{path}:{line_no}: text must be a non-empty string")
        if record_id in records:
            raise DatasetError(f"{path}:{line_no}: duplicate id {record_id!r}")
        records[record_id] = text
    if not records:
        raise DatasetError(f"{path}: no records")
    return records


def _load_qrels(
    path: Path, queries: dict[str, str], corpus: dict[str, str]
) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise DatasetError(f"{path}: missing required file")
    qrels: dict[str, dict[str, int]] = {}
    for line_no, line in enumerate(
        _split_lines(path.read_text(encoding="utf-8")), start=1
    ):
        if not line.strip():
            raise DatasetError(f"{path}:{line_no}: blank line")
        fields = line.split("\t")
        if len(fields) != 3:
            raise DatasetError(
                f"{path}:{line_no}: expected 3 tab-separated fields, got {len(fields)}"
            )
        query_id, corpus_id, raw_score = fields
        if query_id not in queries:
            raise DatasetError(f"{path}:{line_no}: unknown query id {query_id!r}")
        if corpus_id not in corpus:
            raise DatasetError(f"{path}:{line_no}: unknown corpus id {corpus_id!r}")
        try:
            score = int(raw_score)
        except ValueError as exc:
            raise DatasetError(
                f"{path}:{line_no}: score {raw_score!r} is not an integer"
            ) from exc
        if score < 0:
            raise DatasetError(f"{path}:{line_no}: negative score {score}")
        per_query = qrels.setdefault(query_id, {})
        if corpus_id in per_query:
            raise DatasetError(
                f"{path}:{line_no}: duplicate qrels pair {query_id!r}/{corpus_id!r}"
            )
        per_query[corpus_id] = score
    return qrels


def _load_meta(path: Path) -> DatasetMeta:
    if not path.is_file():
        raise DatasetError(f"{path}: missing required file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"{path}: expected a JSON object")
    domain = payload.get("domain")
    if domain not in _DOMAINS:
        raise DatasetError(f"{path}: domain must be one of {_DOMAINS}")
    language = payload.get("language")
    if language not in _LANGUAGES:
        raise DatasetError(f"{path}: language must be one of {_LANGUAGES}")
    for key in ("source", "license", "subset_protocol"):
        if not isinstance(payload.get(key), str):
            raise DatasetError(f"{path}: {key} must be a string")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise DatasetError(f"{path}: counts must be an object")
    return DatasetMeta(
        domain=domain,
        language=language,
        source=payload["source"],
        license=payload["license"],
        subset_protocol=payload["subset_protocol"],
        counts=counts,
    )


def load_dataset(directory: Path) -> BenchDataset:
    """Load one normalized dataset directory; raises ``DatasetError`` on any
    contract violation, naming the file and line."""
    directory = Path(directory)
    corpus = _load_jsonl(directory / "corpus.jsonl")
    queries = _load_jsonl(directory / "queries.jsonl")
    qrels = _load_qrels(directory / "qrels.tsv", queries, corpus)
    meta = _load_meta(directory / "meta.json")
    return BenchDataset(
        name=directory.name,
        meta=meta,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )


def discover_datasets(root: Path) -> list[Path]:
    """Resolve the dataset root: a single dataset directory when it holds
    ``corpus.jsonl`` itself, otherwise every immediate subdirectory that does,
    in sorted order. Raises ``DatasetError`` when nothing qualifies."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"{root}: dataset root is not a directory")
    if (root / "corpus.jsonl").is_file():
        return [root]
    candidates = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "corpus.jsonl").is_file()
    )
    if not candidates:
        raise DatasetError(f"{root}: no dataset directories with corpus.jsonl")
    return candidates
