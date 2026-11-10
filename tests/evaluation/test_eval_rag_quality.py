import asyncio
import builtins
import json

import pandas as pd
import pytest

from lightrag.evaluation import eval_rag_quality as module
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


NAN = float("nan")
_METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)


def _patch_ragas_stack(monkeypatch, fake_evaluate):
    """Patch every name the optional `ragas`/`tqdm` imports leave undefined.

    Without the `evaluation` extra installed (as in offline CI), the
    module-level try/except only sets Dataset/evaluate/LangchainLLMWrapper
    to None on ImportError — tqdm and the four metric classes stay
    undefined names, so evaluate_single_case raises NameError before ever
    reaching the logic under test. Patching only Dataset/evaluate is not
    enough to make these tests independent of the optional dependency.
    """

    class _FakeDataset:
        @staticmethod
        def from_dict(data):
            return data

    class _FakeMetric:
        """Stand-in for a RAGAS metric class: only ever instantiated."""

    class _FakeTqdm:
        """Stand-in for tqdm.auto.tqdm: only .close() is exercised."""

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    # Dataset/evaluate always exist (the except branch sets them to None),
    # but tqdm and the four metric classes are never assigned at all when
    # RAGAS_AVAILABLE is False — plain setattr would itself raise
    # AttributeError, so those need raising=False.
    monkeypatch.setattr(module, "Dataset", _FakeDataset)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "tqdm", _FakeTqdm, raising=False)
    monkeypatch.setattr(module, "Faithfulness", _FakeMetric, raising=False)
    monkeypatch.setattr(module, "AnswerRelevancy", _FakeMetric, raising=False)
    monkeypatch.setattr(module, "ContextRecall", _FakeMetric, raising=False)
    monkeypatch.setattr(module, "ContextPrecision", _FakeMetric, raising=False)


async def fake_generate_rag_response(question, client):
    return {"answer": "an answer", "contexts": ["a context"]}


async def _evaluate_one_case(monkeypatch, scores: dict[str, float]) -> dict:
    """Run evaluate_single_case with RAGAS stubbed to return ``scores``."""

    class _FakeResults:
        def to_pandas(self):
            return pd.DataFrame([{name: scores[name] for name in _METRIC_COLUMNS}])

    _patch_ragas_stack(monkeypatch, lambda **kwargs: _FakeResults())

    evaluator = object.__new__(RAGEvaluator)
    evaluator.eval_llm = None
    evaluator.eval_embeddings = None
    evaluator.generate_rag_response = fake_generate_rag_response

    position_pool = asyncio.Queue()
    position_pool.put_nowait(0)
    return await evaluator.evaluate_single_case(
        1,
        {"question": "q", "ground_truth": "gt"},
        asyncio.Semaphore(1),
        asyncio.Semaphore(1),
        None,
        {"completed": 0},
        position_pool,
        asyncio.Lock(),
    )


async def test_all_nan_metrics_is_a_failed_case_not_a_zero_score(monkeypatch):
    result = await _evaluate_one_case(monkeypatch, dict.fromkeys(_METRIC_COLUMNS, NAN))

    # Same shape as every other failed case, so the table, CSV and statistics
    # all treat it as an error instead of a successful 0.0.
    assert result["metrics"] == {}
    assert "NaN" in result["error"]


async def test_partially_nan_metrics_are_averaged_over_the_scored_ones(monkeypatch):
    scores = dict.fromkeys(_METRIC_COLUMNS, 0.5)
    scores["faithfulness"] = NAN

    result = await _evaluate_one_case(monkeypatch, scores)

    assert "error" not in result
    assert result["ragas_score"] == 0.5


def test_benchmark_stats_do_not_average_failed_cases_into_the_ragas_score():
    evaluator = object.__new__(RAGEvaluator)
    scored = {
        "metrics": {
            "faithfulness": 0.8,
            "answer_relevance": 0.8,
            "context_recall": 0.8,
            "context_precision": 0.8,
        },
        "ragas_score": 0.8,
    }
    failed = {"error": "boom", "metrics": {}, "ragas_score": 0}

    stats = evaluator._calculate_benchmark_stats([scored, failed])

    assert stats["successful_tests"] == 1
    assert stats["failed_tests"] == 1
    assert stats["success_rate"] == 50.0
    assert stats["average_metrics"]["ragas_score"] == 0.8
    assert stats["min_ragas_score"] == 0.8


def test_benchmark_stats_all_failed_still_has_the_success_path_keys():
    """Regression test: previously this branch omitted average_metrics,
    min_ragas_score and max_ragas_score, and run() reads them
    unconditionally — see test_run_completes_when_every_case_is_all_nan."""
    evaluator = object.__new__(RAGEvaluator)
    failed = {"error": "boom", "metrics": {}, "ragas_score": 0}

    stats = evaluator._calculate_benchmark_stats([failed])

    assert stats["successful_tests"] == 0
    assert stats["failed_tests"] == 1
    assert stats["average_metrics"] == {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "ragas_score": 0.0,
    }
    assert stats["min_ragas_score"] == 0
    assert stats["max_ragas_score"] == 0


async def test_run_completes_when_every_case_is_all_nan(tmp_path, monkeypatch):
    """Regression test for the run() KeyError when every case fails.

    Before the fix, an all-NaN RAGAS result made every case fail, which
    made _calculate_benchmark_stats omit average_metrics / min_ragas_score
    / max_ragas_score, and run() reads those keys unconditionally.
    """

    class _AllNanResults:
        def to_pandas(self):
            return pd.DataFrame([dict.fromkeys(_METRIC_COLUMNS, NAN)])

    _patch_ragas_stack(monkeypatch, lambda **kwargs: _AllNanResults())
    monkeypatch.setenv("EVAL_MAX_CONCURRENT", "1")

    evaluator = object.__new__(RAGEvaluator)
    evaluator.eval_llm = None
    evaluator.eval_embeddings = None
    evaluator.results_dir = tmp_path
    evaluator.test_cases = [{"question": "q", "ground_truth": "gt", "project": "p"}]
    evaluator.generate_rag_response = fake_generate_rag_response

    summary = await evaluator.run()

    stats = summary["benchmark_stats"]
    assert stats["successful_tests"] == 0
    assert stats["failed_tests"] == 1
    assert stats["average_metrics"]["ragas_score"] == 0.0
