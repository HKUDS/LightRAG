"""Unit tests for LoginRateLimiter (brute-force protection, CWE-307)."""

from lightrag.api.login_rate_limit import LoginRateLimiter


class _FakeClock:
    """Deterministic monotonic clock so window behavior is testable without sleep."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make(max_attempts=3, window_seconds=60.0):
    clock = _FakeClock()
    return LoginRateLimiter(max_attempts, window_seconds, clock=clock), clock


def test_allows_up_to_max_then_locks():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        assert limiter.retry_after(key) is None
        limiter.record_failure(key)

    retry_after = limiter.retry_after(key)
    assert retry_after is not None
    assert 0 < retry_after <= 60


def test_reset_clears_failures():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        limiter.record_failure(key)
    assert limiter.retry_after(key) is not None

    limiter.reset(key)
    assert limiter.retry_after(key) is None


def test_window_expiry_allows_again_and_prunes():
    limiter, clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        limiter.record_failure(key)
    assert limiter.retry_after(key) is not None

    clock.advance(61)  # every recorded failure ages out of the window
    assert limiter.retry_after(key) is None
    # The key is dropped once nothing is in-window (bounded memory).
    assert key not in limiter._failures


def test_retry_after_counts_down_as_window_passes():
    limiter, clock = _make(max_attempts=2, window_seconds=60)
    key = "1.2.3.4:admin"

    limiter.record_failure(key)  # t = 1000
    clock.advance(10)
    limiter.record_failure(key)  # t = 1010, now locked (oldest at 1000)

    retry_after = limiter.retry_after(key)  # 60 - (1010 - 1000) = 50
    assert 49 <= retry_after <= 50


def test_disabled_when_max_attempts_zero():
    limiter, _clock = _make(max_attempts=0, window_seconds=60)
    assert not limiter.enabled
    key = "1.2.3.4:admin"

    for _ in range(10):
        limiter.record_failure(key)
    assert limiter.retry_after(key) is None


def test_keys_are_independent():
    limiter, _clock = _make(max_attempts=2, window_seconds=60)
    admin = "1.2.3.4:admin"
    bob = "1.2.3.4:bob"

    for _ in range(2):
        limiter.record_failure(admin)

    assert limiter.retry_after(admin) is not None  # admin locked
    assert limiter.retry_after(bob) is None  # different key, own bucket


def test_tracked_keys_are_bounded_by_lru_eviction():
    """A flood of unique keys must not grow the map without bound; the oldest
    (least-recently-active) keys are evicted so memory stays capped.
    """
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=3
    )

    for i in range(10):
        limiter.record_failure(f"10.0.0.{i}:admin")

    assert len(limiter._failures) <= 3
    assert "10.0.0.9:admin" in limiter._failures  # most recent retained
    assert "10.0.0.0:admin" not in limiter._failures  # oldest evicted


def test_active_key_survives_a_flood_of_unique_keys():
    """A key that keeps failing stays most-recently-active and is not evicted by
    a flood of one-off keys, so a real brute-force target keeps its counter.
    """
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=3
    )
    target = "1.2.3.4:admin"

    for i in range(10):
        limiter.record_failure(f"flood:{i}")
        limiter.record_failure(target)  # touched every round -> stays hot

    assert len(limiter._failures) <= 3
    assert target in limiter._failures
