"""``log_without_raising`` must never propagate a sink failure (#3855).

Post-commit hooks call it for every diagnostic emitted after a durable
mutation landed. A raising log sink there would both misreport the completed
mutation as failed and skip the recovery bookkeeping that follows the log
line, so the sink failure is swallowed on purpose.
"""

import pytest

from lightrag.utils import log_without_raising

pytestmark = pytest.mark.offline


def test_emits_the_message_through_the_given_callable():
    emitted: list[str] = []

    log_without_raising(emitted.append, "hello")

    assert emitted == ["hello"]


def test_swallows_any_sink_failure():
    # One bare ``except Exception``: the exception class does not matter.
    error = RuntimeError("sink boom")

    def boom(message):
        raise error

    # No raise, no return value contract beyond None.
    assert log_without_raising(boom, "hello") is None
