"""Tests for ``normalize_rerank_result`` validation semantics.

``normalize_rerank_result`` is the single validation boundary between a rerank
provider's output and the chunk-scoring pipeline: ``apply_rerank_if_enabled``
feeds every result a provider emits — a hosted rerank model, a compatible proxy,
or a user-supplied ``rerank_model_func`` — through this helper before trusting
its ``index``/``relevance_score`` to reorder chunks. Two properties are therefore
load-bearing rather than cosmetic:

* **Malformed results are rejected, never coerced.** A result whose ``index`` or
  ``relevance_score`` cannot be trusted must be dropped and the chunk left in its
  original position; silently accepting it would surface as a wrong (or
  unexpectedly empty) ranking. The ``bool`` cases matter specifically because
  ``bool`` subclasses ``int`` and ``float(True)`` succeeds — both must be caught
  explicitly or a ``True`` index/score leaks through.
* **The reject reason string is stable.** Callers and future maintainers rely on
  ``"not an object"`` / ``"invalid index"`` / ``"index out of range"`` /
  ``"invalid relevance score"`` / ``"non-finite relevance score"`` as the single
  vocabulary for diagnosing a misbehaving provider.

The function had no direct tests; ``tests/llm/test_apply_rerank_result_validation.py``
exercises it only indirectly through the happy path and a single bool-index case.
"""

from __future__ import annotations

import pytest

from lightrag.utils import normalize_rerank_result

pytestmark = pytest.mark.offline


class TestAccepted:
    @pytest.mark.parametrize(
        ("result", "max_index", "expected_score"),
        [
            ({"index": 1, "relevance_score": 0.8}, 2, 0.8),
            # String scores are accepted and coerced to float.
            ({"index": 0, "relevance_score": "0.5"}, 3, 0.5),
            # Integral scores are accepted and normalized to float.
            ({"index": 2, "relevance_score": 3}, 4, 3.0),
            # The highest valid index is ``max_index - 1``.
            ({"index": 3, "relevance_score": 0.1}, 4, 0.1),
        ],
    )
    def test_valid_result_is_normalized(self, result, max_index, expected_score):
        normalized, error = normalize_rerank_result(result, max_index)
        assert error is None
        assert normalized == {
            "index": result["index"],
            "relevance_score": expected_score,
        }
        assert isinstance(normalized["relevance_score"], float)


class TestResultShape:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param(None, id="none"),
            pytest.param([], id="empty-list"),
            pytest.param("not a dict", id="string"),
            pytest.param(42, id="int"),
            pytest.param((0, 0.9), id="tuple"),
        ],
    )
    def test_non_dict_result_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (None, "not an object")


class TestIndexValidation:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({"relevance_score": 0.9}, id="missing-index"),
            pytest.param({"index": None, "relevance_score": 0.9}, id="none-index"),
            pytest.param({"index": "0", "relevance_score": 0.9}, id="string-index"),
            pytest.param({"index": 0.5, "relevance_score": 0.9}, id="float-index"),
            pytest.param({"index": [0], "relevance_score": 0.9}, id="list-index"),
        ],
    )
    def test_non_integer_index_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (None, "invalid index")

    def test_bool_index_is_rejected_even_though_bool_is_an_int_subclass(self):
        # ``isinstance(True, int)`` holds, so the bool case must be caught first.
        for index in (True, False):
            result = {"index": index, "relevance_score": 0.9}
            assert normalize_rerank_result(result, 2) == (None, "invalid index")

    @pytest.mark.parametrize(
        ("result", "max_index"),
        [
            pytest.param({"index": -1, "relevance_score": 0.9}, 2, id="negative"),
            pytest.param({"index": 2, "relevance_score": 0.9}, 2, id="at-max"),
            pytest.param({"index": 5, "relevance_score": 0.9}, 2, id="beyond-max"),
        ],
    )
    def test_out_of_range_index_is_rejected(self, result, max_index):
        assert normalize_rerank_result(result, max_index) == (
            None,
            "index out of range",
        )


class TestScoreValidation:
    @pytest.mark.parametrize(
        "result",
        [
            pytest.param({"index": 0}, id="missing-score"),
            pytest.param({"index": 0, "relevance_score": None}, id="none-score"),
            pytest.param(
                {"index": 0, "relevance_score": "not-a-score"}, id="string-score"
            ),
            pytest.param({"index": 0, "relevance_score": [0.5]}, id="list-score"),
            pytest.param(
                {"index": 0, "relevance_score": {"value": 0.5}}, id="dict-score"
            ),
        ],
    )
    def test_unconvertible_score_is_rejected(self, result):
        assert normalize_rerank_result(result, 2) == (
            None,
            "invalid relevance score",
        )

    def test_bool_score_is_rejected_even_though_bool_is_floatable(self):
        # ``float(True)`` would be ``1.0``, so the bool case must be caught first.
        result = {"index": 0, "relevance_score": True}
        assert normalize_rerank_result(result, 2) == (
            None,
            "invalid relevance score",
        )

    @pytest.mark.parametrize(
        "score",
        [
            pytest.param(float("nan"), id="nan"),
            pytest.param(float("inf"), id="positive-inf"),
            pytest.param(float("-inf"), id="negative-inf"),
        ],
    )
    def test_non_finite_score_is_rejected(self, score):
        result = {"index": 0, "relevance_score": score}
        assert normalize_rerank_result(result, 2) == (
            None,
            "non-finite relevance score",
        )
