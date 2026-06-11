#!/usr/bin/env python3
"""Offline sanity check for the bundled LightRAG evaluation samples.

The check uses a small deterministic lexical ranker. It does not start
LightRAG, call the API server, compute embeddings, or call LLM/RAGAS services.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVAL_DIR / "sample_dataset.json"
DEFAULT_DOCS_DIR = EVAL_DIR / "sample_documents"
DEFAULT_ORACLE = EVAL_DIR / "sample_retrieval_oracle.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "what",
    "with",
}


@dataclass
class Document:
    name: str
    tokens: Counter[str]


@dataclass
class QueryResult:
    question: str
    expected: list[str]
    ranked: list[str]

    def recall_at(self, top_k: int) -> float:
        hits = set(self.expected) & set(self.ranked[:top_k])
        return len(hits) / len(self.expected)

    def reciprocal_rank(self) -> float:
        for rank, document in enumerate(self.ranked, start=1):
            if document in self.expected:
                return 1 / rank
        return 0.0


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def load_cases(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = payload.get("test_cases")
    if not isinstance(cases, list):
        raise ValueError(f"{dataset_path} must contain a test_cases list")
    return cases


def load_oracle(oracle_path: Path) -> dict[str, list[str]]:
    payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    entries = payload.get("oracle")
    if not isinstance(entries, list):
        raise ValueError(f"{oracle_path} must contain an oracle list")

    oracle: dict[str, list[str]] = {}
    for entry in entries:
        question = str(entry.get("question", "")).strip()
        expected = entry.get("expected_documents")
        if not question or not isinstance(expected, list) or not expected:
            raise ValueError("Each oracle entry needs question and expected_documents")
        oracle[question] = [str(document) for document in expected]
    return oracle


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        documents.append(
            Document(
                name=path.name,
                tokens=Counter(tokenize(path.read_text(encoding="utf-8"))),
            )
        )
    if not documents:
        raise ValueError(f"No markdown sample documents found in {docs_dir}")
    return documents


def inverse_document_frequency(documents: list[Document]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.tokens.keys())
    doc_count = len(documents)
    return {
        token: math.log((doc_count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


def score_query(
    query_tokens: list[str],
    document: Document,
    idf: dict[str, float],
) -> float:
    score = 0.0
    for token in query_tokens:
        if token in document.tokens:
            score += (1 + math.log(document.tokens[token])) * idf.get(token, 0.0)
    return score


def audit_samples(
    cases: list[dict[str, Any]],
    oracle: dict[str, list[str]],
    documents: list[Document],
) -> list[QueryResult]:
    idf = inverse_document_frequency(documents)
    results: list[QueryResult] = []

    for case in cases:
        question = str(case.get("question", "")).strip()
        if question not in oracle:
            raise ValueError(f"No oracle entry for question: {question}")

        query_tokens = tokenize(question)
        scored_documents = [
            (score_query(query_tokens, document, idf), document)
            for document in documents
        ]
        ranked = [
            document
            for score, document in sorted(
                scored_documents,
                key=lambda item: (-item[0], item[1].name),
            )
            if score > 0
        ]
        results.append(
            QueryResult(
                question=question,
                expected=oracle[question],
                ranked=[document.name for document in ranked],
            )
        )
    return results


def summarize(results: list[QueryResult], top_k: int) -> dict[str, Any]:
    if not results:
        raise ValueError("No query results to summarize")
    recalls = [result.recall_at(top_k) for result in results]
    reciprocal_ranks = [result.reciprocal_rank() for result in results]
    return {
        "queries": len(results),
        "top_k": top_k,
        "average_recall_at_k": sum(recalls) / len(recalls),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "full_recall_queries": sum(recall == 1.0 for recall in recalls),
        "no_hit_queries": sum(recall == 0.0 for recall in recalls),
    }


def print_report(results: list[QueryResult], top_k: int) -> None:
    summary = summarize(results, top_k)
    print("LightRAG sample retrieval check")
    print(f"Queries: {summary['queries']}")
    print(f"Top-k: {summary['top_k']}")
    print(f"Average recall@k: {summary['average_recall_at_k']:.3f}")
    print(f"Mean reciprocal rank: {summary['mean_reciprocal_rank']:.3f}")
    print(f"Full-recall queries: {summary['full_recall_queries']}/{summary['queries']}")
    print(f"No-hit queries: {summary['no_hit_queries']}")
    print()
    for index, result in enumerate(results, start=1):
        top_docs = ", ".join(result.ranked[:top_k])
        expected = ", ".join(result.expected)
        print(f"{index}. recall@{top_k}={result.recall_at(top_k):.3f}")
        print(f"   expected: {expected}")
        print(f"   top docs: {top_docs}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline retrieval check for LightRAG evaluation samples."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every sample query has full recall@k.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.top_k <= 0:
        print("--top-k must be positive", file=sys.stderr)
        return 2

    try:
        cases = load_cases(Path(args.dataset).expanduser())
        oracle = load_oracle(Path(args.oracle).expanduser())
        documents = load_documents(Path(args.docs_dir).expanduser())
        results = audit_samples(cases, oracle, documents)
        print_report(results, args.top_k)
        summary = summarize(results, args.top_k)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Sample retrieval check failed: {exc}", file=sys.stderr)
        return 2

    if args.strict and summary["full_recall_queries"] != summary["queries"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
