"""``TokenTracker.add_usage`` with nullable and optional counts.

Provider SDKs report optional usage fields as ``None`` on successful calls; an
``int += None`` there would discard the generated answer. Each binding's own
tests cover what it passes in; this pins the tracker's own fallback and how
it sums the optional ``cached_prompt_tokens``.
"""

import pytest

from lightrag.utils import TokenTracker

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("reported_total", "expected_total"),
    [
        pytest.param(None, 2, id="none-total-is-summed"),
        pytest.param(0, 0, id="reported-zero-is-kept"),
    ],
)
def test_add_usage_treats_none_counts_as_missing(reported_total, expected_total):
    tracker = TokenTracker()

    tracker.add_usage(
        {"prompt_tokens": 2, "completion_tokens": None, "total_tokens": reported_total}
    )

    assert tracker.get_usage() == {
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "total_tokens": expected_total,
        "call_count": 1,
    }


def test_cached_prompt_tokens_sum_only_reported_calls():
    tracker = TokenTracker()
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}

    tracker.add_usage({**usage, "cached_prompt_tokens": 2})
    tracker.add_usage(usage)
    tracker.add_usage({**usage, "cached_prompt_tokens": 3})

    assert tracker.get_usage()["cached_prompt_tokens"] == 5
    assert str(tracker).endswith(", Cached prompt tokens: 5")
    tracker.reset()
    assert "cached_prompt_tokens" not in tracker.get_usage()
