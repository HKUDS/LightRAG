import tempfile
import unittest
from pathlib import Path

from lightrag.evaluation.offline_retrieval_check import (
    audit_samples,
    load_cases,
    load_documents,
    load_oracle,
    summarize,
)


class OfflineRetrievalCheckTests(unittest.TestCase):
    def test_expected_document_ranks_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Alpha covers vector search and filtering.",
                encoding="utf-8",
            )
            (docs_dir / "beta.md").write_text(
                "Beta covers deployment and monitoring.",
                encoding="utf-8",
            )
            dataset = root / "dataset.json"
            dataset.write_text(
                '{"test_cases":[{"question":"Which file explains vector search?"}]}',
                encoding="utf-8",
            )
            oracle = root / "oracle.json"
            oracle.write_text(
                '{"oracle":[{"question":"Which file explains vector search?",'
                '"expected_documents":["alpha.md"]}]}',
                encoding="utf-8",
            )

            results = audit_samples(
                load_cases(dataset),
                load_oracle(oracle),
                load_documents(docs_dir),
            )
            summary = summarize(results, top_k=1)

        self.assertEqual(results[0].ranked[0], "alpha.md")
        self.assertEqual(summary["queries"], 1)
        self.assertEqual(summary["average_recall_at_k"], 1.0)

    def test_zero_score_documents_do_not_count_as_hits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Alpha covers deployment pipelines.",
                encoding="utf-8",
            )
            (docs_dir / "beta.md").write_text(
                "Beta covers monitoring dashboards.",
                encoding="utf-8",
            )
            dataset = root / "dataset.json"
            dataset.write_text(
                '{"test_cases":[{"question":"Which file explains vector search?"}]}',
                encoding="utf-8",
            )
            oracle = root / "oracle.json"
            oracle.write_text(
                '{"oracle":[{"question":"Which file explains vector search?",'
                '"expected_documents":["alpha.md"]}]}',
                encoding="utf-8",
            )

            results = audit_samples(
                load_cases(dataset),
                load_oracle(oracle),
                load_documents(docs_dir),
            )
            summary = summarize(results, top_k=1)

        self.assertEqual(results[0].ranked, [])
        self.assertEqual(summary["average_recall_at_k"], 0.0)
        self.assertEqual(summary["no_hit_queries"], 1)

    def test_sample_oracle_has_full_recall_at_two(self):
        results = audit_samples(
            load_cases(Path("lightrag/evaluation/sample_dataset.json")),
            load_oracle(Path("lightrag/evaluation/sample_retrieval_oracle.json")),
            load_documents(Path("lightrag/evaluation/sample_documents")),
        )
        summary = summarize(results, top_k=2)

        self.assertEqual(summary["queries"], 6)
        self.assertEqual(summary["full_recall_queries"], 6)
        self.assertEqual(summary["no_hit_queries"], 0)


if __name__ == "__main__":
    unittest.main()
