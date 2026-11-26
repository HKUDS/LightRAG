"""``TokenTracker.add_usage`` with nullable and optional counts.

Provider SDKs report optional usage fields as ``None`` on successful calls; an
``int += None`` there would discard the generated answer. Each binding's own
tests cover what it passes in; this pins the tracker's own fallback and how
it sums the optional cache counts.
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


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("cached_prompt_tokens", "Cached prompt tokens"),
        ("cache_creation_tokens", "Cache write tokens"),
    ],
)
def test_optional_cache_counts_sum_only_reported_calls(field, label):
    tracker = TokenTracker()
    usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}

    tracker.add_usage({**usage, field: 2})
    tracker.add_usage(usage)
    tracker.add_usage({**usage, field: 3})

    assert tracker.get_usage()[field] == 5
    assert str(tracker).endswith(f", {label}: 5")
    tracker.reset()
    assert field not in tracker.get_usage()
