import tempfile
import unittest
from pathlib import Path

import pytest

from lightrag.evaluation.offline_retrieval_check import (
    QueryResult,
    audit_samples,
    load_cases,
    load_documents,
    load_oracle,
    summarize,
    tokenize,
)

pytestmark = pytest.mark.offline


class OfflineRetrievalCheckTests(unittest.TestCase):
    def test_recall_at_ignores_duplicate_expected_documents(self):
        # Oracle data can list the same expected document twice. The single
        # relevant document is retrieved, so recall should be 1.0, not
        # deflated by counting the duplicate in the denominator.
        result = QueryResult(
            question="dup",
            expected=["alpha.md", "alpha.md"],
            ranked=["alpha.md", "beta.md"],
        )

        self.assertEqual(result.recall_at(2), 1.0)

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

    def test_ascii_words_keep_their_stopword_and_length_rules(self):
        self.assertEqual(
            tokenize("What is a RAG system in 2026?"),
            ["rag", "system", "2026"],
        )

    def test_cjk_runs_become_overlapping_bigrams(self):
        self.assertEqual(tokenize("图检索"), ["图检", "检索"])
        self.assertEqual(
            tokenize("LightRAG支持图检索"),
            ["lightrag", "支持", "持图", "图检", "检索"],
        )

    def test_a_lone_cjk_character_is_kept(self):
        # One ideograph is often a whole word ("铜" is copper), so the rule
        # that drops single ASCII characters does not apply to it.
        self.assertEqual(tokenize("铜，价格"), ["铜", "价格"])

    def test_full_width_ascii_matches_its_ascii_form(self):
        self.assertEqual(tokenize("ＲＡＧ系统"), ["rag", "系统"])

    def test_half_width_katakana_matches_full_width(self):
        self.assertEqual(tokenize("ｶﾀｶﾅ"), tokenize("カタカナ"))
        self.assertEqual(tokenize("カタカナ"), ["カタ", "タカ", "カナ"])

    def test_katakana_middle_dot_splits_runs(self):
        self.assertEqual(
            tokenize("データ・ベース"),
            ["デー", "ータ", "ベー", "ース"],
        )

    def test_iteration_mark_stays_in_its_word(self):
        self.assertEqual(tokenize("人々"), ["人々"])

    def test_ideographs_outside_the_bmp_are_kept(self):
        # U+20000 and U+20001 are CJK Unified Ideographs Extension B.
        self.assertEqual(
            tokenize("\U00020000\U00020001图"),
            [
                "\U00020000\U00020001",
                "\U00020001图",
            ],
        )

    def test_chinese_question_ranks_the_matching_document_first(self):
        # Without CJK tokenization a Chinese question has no terms, every
        # document scores zero, and nothing is ranked at all.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "向量检索与过滤：按元数据过滤向量检索的结果。",
                encoding="utf-8",
            )
            (docs_dir / "beta.md").write_text(
                "部署与监控：服务部署、日志和告警。",
                encoding="utf-8",
            )
            dataset = root / "dataset.json"
            dataset.write_text(
                '{"test_cases":[{"question":"哪份文档讲向量检索？"}]}',
                encoding="utf-8",
            )
            oracle = root / "oracle.json"
            oracle.write_text(
                '{"oracle":[{"question":"哪份文档讲向量检索？",'
                '"expected_documents":["alpha.md"]}]}',
                encoding="utf-8",
            )

            results = audit_samples(
                load_cases(dataset),
                load_oracle(oracle),
                load_documents(docs_dir),
            )
            summary = summarize(results, top_k=1)

        self.assertEqual(results[0].ranked, ["alpha.md"])
        self.assertEqual(summary["average_recall_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
