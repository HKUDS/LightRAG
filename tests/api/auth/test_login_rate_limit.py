"""Unit tests for LoginRateLimiter (brute-force protection, CWE-307)."""

from types import SimpleNamespace

import lightrag.api.login_rate_limit as lrl_module
from lightrag.api.login_rate_limit import LoginRateLimiter


def _capture_warnings(monkeypatch):
    """Redirect the module logger's WARNING output into a list for assertions."""
    messages: list[str] = []
    monkeypatch.setattr(lrl_module, "logger", SimpleNamespace(warning=messages.append))
    return messages


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


def test_tracked_keys_are_bounded():
    """A flood of unique keys must not grow the map beyond the cap."""
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=3
    )

    for i in range(100):
        limiter.record_failure(f"10.0.0.{i}:admin")

    assert len(limiter._failures) <= 3


def test_expired_keys_are_reclaimed_for_new_keys():
    """Once tracked keys age out of the window their slots are reclaimed, so new
    keys can be tracked again after a full-but-stale table self-heals.
    """
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=60, clock=clock, max_tracked_keys=2
    )
    limiter.record_failure("a")
    limiter.record_failure("b")  # table full
    assert len(limiter._failures) == 2

    clock.advance(61)  # "a" and "b" age out of the window
    limiter.record_failure("c")  # expired slot reclaimed -> "c" is tracked

    assert "c" in limiter._failures
    assert len(limiter._failures) <= 2


def test_active_lockout_survives_flood_of_unique_keys():
    """An in-window lockout must NOT be evicted to make room for new keys: an
    attacker could otherwise lock 'admin', then flood the same IP with unique
    fake usernames to push admin's record out and reset its counter.
    """
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=2, window_seconds=300, clock=clock, max_tracked_keys=3
    )
    admin = "1.2.3.4:admin"
    limiter.record_failure(admin)
    limiter.record_failure(admin)  # admin now locked, in-window
    assert limiter.retry_after(admin) is not None

    for i in range(100):  # flood unique usernames from the same IP
        limiter.record_failure(f"1.2.3.4:user{i}")

    assert limiter.retry_after(admin) is not None  # lockout preserved
    assert len(limiter._failures) <= 3  # still bounded


def test_lockout_emits_warning(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=2, window_seconds=300)
    key = "1.2.3.4:admin"

    limiter.record_failure(key)
    assert messages == []  # one failure -> not yet locked, no alert

    limiter.record_failure(key)  # trips the lockout
    assert len(messages) == 1
    assert "tripped" in messages[0]
    assert key in messages[0]


def test_lockout_alerts_are_throttled_with_suppressed_count(monkeypatch):
    """A burst of lockouts must not flood the log: only one WARNING per interval,
    with the suppressed count reported on the next emitted line.
    """
    messages = _capture_warnings(monkeypatch)
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=1, window_seconds=300, clock=clock, alert_interval_seconds=60
    )

    for i in range(5):  # each distinct key locks on its first failure
        limiter.record_failure(f"10.0.0.{i}:admin")
    assert len(messages) == 1  # first emitted; the other 4 suppressed

    clock.advance(61)  # interval elapsed
    limiter.record_failure("10.0.0.9:admin")
    assert len(messages) == 2
    assert "4 more since last alert" in messages[1]


def test_table_full_emits_throttled_warning(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=1
    )

    limiter.record_failure("a")  # fills the table, not locked (max_attempts=5)
    assert messages == []

    limiter.record_failure("b")  # table full of live keys -> refuse + warn
    assert len(messages) == 1
    assert "table is full" in messages[0]

    limiter.record_failure("c")  # still full, within interval -> suppressed
    assert len(messages) == 1
