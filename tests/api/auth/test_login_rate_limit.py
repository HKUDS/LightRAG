"""Unit tests for LoginRateLimiter (brute-force protection, CWE-307)."""

from types import SimpleNamespace

import lightrag.api.login_rate_limit as lrl_module
from lightrag.api.login_rate_limit import LoginRateLimiter, _safe_log_value


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


def _fail(limiter, key):
    """Mirror the route: reserve the attempt, then confirm it as a failure."""
    limiter.reserve_attempt(key)
    limiter.commit_failure(key)


def test_allows_up_to_max_then_locks():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        assert limiter.retry_after(key) is None
        limiter.reserve_attempt(key)

    retry_after = limiter.retry_after(key)
    assert retry_after is not None
    assert 0 < retry_after <= 60


def test_reset_clears_failures():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        limiter.reserve_attempt(key)
    assert limiter.retry_after(key) is not None

    limiter.reset(key)
    assert limiter.retry_after(key) is None


def test_window_expiry_allows_again_and_prunes():
    limiter, clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        limiter.reserve_attempt(key)
    assert limiter.retry_after(key) is not None

    clock.advance(61)  # every recorded attempt ages out of the window
    assert limiter.retry_after(key) is None
    # The key is dropped once nothing is in-window (bounded memory).
    assert key not in limiter._failures


def test_retry_after_counts_down_as_window_passes():
    limiter, clock = _make(max_attempts=2, window_seconds=60)
    key = "1.2.3.4:admin"

    limiter.reserve_attempt(key)  # t = 1000
    clock.advance(10)
    limiter.reserve_attempt(key)  # t = 1010, now locked (oldest at 1000)

    retry_after = limiter.retry_after(key)  # 60 - (1010 - 1000) = 50
    assert 49 <= retry_after <= 50


def test_disabled_when_max_attempts_zero():
    limiter, _clock = _make(max_attempts=0, window_seconds=60)
    assert not limiter.enabled
    key = "1.2.3.4:admin"

    for _ in range(10):
        limiter.reserve_attempt(key)
    assert limiter.retry_after(key) is None


def test_keys_are_independent():
    limiter, _clock = _make(max_attempts=2, window_seconds=60)
    admin = "1.2.3.4:admin"
    bob = "1.2.3.4:bob"

    for _ in range(2):
        limiter.reserve_attempt(admin)

    assert limiter.retry_after(admin) is not None  # admin locked
    assert limiter.retry_after(bob) is None  # different key, own bucket


def test_tracked_keys_are_bounded():
    """A flood of unique keys must not grow the map beyond the cap."""
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=3
    )

    for i in range(100):
        limiter.reserve_attempt(f"10.0.0.{i}:admin")

    assert len(limiter._failures) <= 3


def test_expired_keys_are_reclaimed_for_new_keys():
    """Once tracked keys age out of the window their slots are reclaimed, so new
    keys can be tracked again after a full-but-stale table self-heals.
    """
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=60, clock=clock, max_tracked_keys=2
    )
    limiter.reserve_attempt("a")
    limiter.reserve_attempt("b")  # table full
    assert len(limiter._failures) == 2

    clock.advance(61)  # "a" and "b" age out of the window
    limiter.reserve_attempt("c")  # expired slot reclaimed -> "c" is tracked

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
    limiter.reserve_attempt(admin)
    limiter.reserve_attempt(admin)  # admin now locked, in-window
    assert limiter.retry_after(admin) is not None

    for i in range(100):  # flood unique usernames from the same IP
        limiter.reserve_attempt(f"1.2.3.4:user{i}")

    assert limiter.retry_after(admin) is not None  # lockout preserved
    assert len(limiter._failures) <= 3  # still bounded


def test_reserve_does_not_alert_but_commit_does(monkeypatch):
    """The lockout alert must fire only after a *confirmed* failure, not at
    reservation time -- otherwise a correct password on the Nth attempt (which
    reserves, verifies OK, then resets) would emit a false lockout alert.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=2, window_seconds=300)
    key = "1.2.3.4:admin"

    # Four earlier failures would be reserved+committed; here simulate reaching
    # the limit purely by reservation and then succeeding.
    limiter.reserve_attempt(key)  # 1st reservation, no alert
    limiter.reserve_attempt(key)  # 2nd reservation reaches the limit, still no alert
    assert messages == []  # reservation alone never alerts

    limiter.reset(key)  # success clears everything
    assert messages == []
    assert limiter.retry_after(key) is None


def test_lockout_alert_emitted_on_confirmed_failure(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=2, window_seconds=300)
    key = "1.2.3.4:admin"

    _fail(limiter, key)  # confirmed failure #1 -> not yet locked
    assert messages == []

    _fail(limiter, key)  # confirmed failure #2 -> trips the lockout
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

    for i in range(5):  # each distinct key locks on its first confirmed failure
        _fail(limiter, f"10.0.0.{i}:admin")
    assert len(messages) == 1  # first emitted; the other 4 suppressed

    clock.advance(61)  # interval elapsed
    _fail(limiter, "10.0.0.9:admin")
    assert len(messages) == 2
    assert "4 more since last alert" in messages[1]


def test_table_full_emits_throttled_warning(monkeypatch):
    messages = _capture_warnings(monkeypatch)
    clock = _FakeClock()
    limiter = LoginRateLimiter(
        max_attempts=5, window_seconds=300, clock=clock, max_tracked_keys=1
    )

    limiter.reserve_attempt("a")  # fills the table, not locked (max_attempts=5)
    assert messages == []

    limiter.reserve_attempt("b")  # table full of live keys -> refuse + warn
    assert len(messages) == 1
    assert "table is full" in messages[0]

    limiter.reserve_attempt("c")  # still full, within interval -> suppressed
    assert len(messages) == 1


def test_lockout_alert_sanitizes_injected_username(monkeypatch):
    """An attacker-supplied username must not forge extra log lines (CRLF log
    injection) via the alert message.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=1, window_seconds=300)
    key = "127.0.0.1:admin\nCRITICAL forged second line"

    _fail(limiter, key)

    assert len(messages) == 1
    assert "\n" not in messages[0]  # newline neutralized -> no injected line
    assert "admin?CRITICAL" in messages[0]  # control char replaced with '?'


def test_safe_log_value_strips_control_chars_and_truncates():
    assert _safe_log_value("a\r\nb") == "a??b"

    long = _safe_log_value("x" * 500, max_length=100)
    assert long.startswith("x" * 100)
    assert "truncated" in long
