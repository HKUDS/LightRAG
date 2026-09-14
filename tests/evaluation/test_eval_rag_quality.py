import builtins
import json

import pytest

from lightrag.evaluation.eval_rag_quality import RAGEvaluator

pytestmark = pytest.mark.offline


def test_load_test_dataset_reads_utf8_with_ascii_default(tmp_path, monkeypatch):
    expected = [{"question": "LightRAG 如何处理知识图谱？ 🌍"}]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps({"test_cases": expected}, ensure_ascii=False),
        encoding="utf-8",
    )

    original_open = builtins.open

    def open_with_ascii_default(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = "ascii"
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_ascii_default)

    evaluator = object.__new__(RAGEvaluator)
    evaluator.test_dataset_path = dataset_path

    assert evaluator._load_test_dataset() == expected
