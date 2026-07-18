"""Unit tests for LoginRateLimiter (brute-force protection, CWE-307)."""

from types import SimpleNamespace

import lightrag.api.login_rate_limit as lrl_module
from lightrag.api.login_rate_limit import (
    _PENDING_RETRY_AFTER_SECONDS,
    LoginRateLimiter,
    _safe_log_value,
)


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


def _make(max_attempts=3, window_seconds=60.0, **kwargs):
    clock = _FakeClock()
    return LoginRateLimiter(max_attempts, window_seconds, clock=clock, **kwargs), clock


def _fail(limiter, key):
    """One full failed-attempt lifecycle: reserve -> confirm failure -> release."""
    limiter.reserve_attempt(key)
    limiter.commit_failure(key)
    limiter.release(key)


def _succeed(limiter, key):
    """One full successful-attempt lifecycle: reserve -> reset -> release."""
    limiter.reserve_attempt(key)
    limiter.reset(key)
    limiter.release(key)


def test_allows_up_to_max_then_locks():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        assert limiter.retry_after(key) is None
        _fail(limiter, key)

    retry_after = limiter.retry_after(key)
    assert retry_after is not None
    assert 0 < retry_after <= 60


def test_reset_clears_failures():
    limiter, _clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        _fail(limiter, key)
    assert limiter.retry_after(key) is not None

    limiter.reset(key)
    assert limiter.retry_after(key) is None


def test_window_expiry_allows_again_and_prunes():
    limiter, clock = _make(max_attempts=3, window_seconds=60)
    key = "1.2.3.4:admin"

    for _ in range(3):
        _fail(limiter, key)
    assert limiter.retry_after(key) is not None

    clock.advance(61)  # every confirmed failure ages out of the window
    assert limiter.retry_after(key) is None
    # The key is dropped once nothing is in-window (bounded memory).
    assert key not in limiter._entries


def test_retry_after_counts_down_as_window_passes():
    limiter, clock = _make(max_attempts=2, window_seconds=60)
    key = "1.2.3.4:admin"

    _fail(limiter, key)  # t = 1000
    clock.advance(10)
    _fail(limiter, key)  # t = 1010, now locked (oldest failure at 1000)

    retry_after = limiter.retry_after(key)  # 60 - (1010 - 1000) = 50
    assert 49 <= retry_after <= 50


def test_disabled_when_max_attempts_zero():
    limiter, _clock = _make(max_attempts=0, window_seconds=60)
    assert not limiter.enabled
    key = "1.2.3.4:admin"

    for _ in range(10):
        _fail(limiter, key)
    assert limiter.retry_after(key) is None


def test_keys_are_independent():
    limiter, _clock = _make(max_attempts=2, window_seconds=60)
    admin = "1.2.3.4:admin"
    bob = "1.2.3.4:bob"

    for _ in range(2):
        _fail(limiter, admin)

    assert limiter.retry_after(admin) is not None  # admin locked
    assert limiter.retry_after(bob) is None  # different key, own bucket


def test_pending_reservations_block_without_confirmed_failures():
    """In-flight reservations alone must block (concurrency guard), even before
    any result is confirmed.
    """
    limiter, _clock = _make(max_attempts=2, window_seconds=300)
    key = "1.2.3.4:admin"

    limiter.reserve_attempt(key)
    limiter.reserve_attempt(key)  # 2 in-flight, none resolved yet

    # Blocked, but only by pending -> short retry hint, not the full window.
    assert limiter.retry_after(key) == _PENDING_RETRY_AFTER_SECONDS


def test_retry_after_is_short_when_block_is_only_from_pending():
    """A request blocked only because in-flight reservations push the total to
    the limit (confirmed failures still below it) should get the short pending
    hint, not the full window -- those reservations may resolve as successes.
    """
    limiter, _clock = _make(max_attempts=5, window_seconds=300)
    key = "1.2.3.4:admin"

    _fail(limiter, key)  # 1 confirmed failure (< max)
    for _ in range(4):
        limiter.reserve_attempt(key)  # 4 in-flight -> total 5 reaches the limit

    assert limiter.retry_after(key) == _PENDING_RETRY_AFTER_SECONDS


def test_retry_after_is_full_window_on_genuine_failure_lockout():
    """Once confirmed failures alone reach the limit, the retry hint is the time
    until the oldest failure ages out (close to the full window).
    """
    limiter, _clock = _make(max_attempts=3, window_seconds=300)
    key = "1.2.3.4:admin"

    for _ in range(3):
        _fail(limiter, key)  # 3 confirmed failures == max

    retry_after = limiter.retry_after(key)
    assert retry_after is not None
    assert retry_after > _PENDING_RETRY_AFTER_SECONDS
    assert 299 <= retry_after <= 300


def test_tracked_keys_are_bounded():
    """A flood of unique keys must not grow the map beyond the cap."""
    limiter, _clock = _make(max_attempts=5, window_seconds=300, max_tracked_keys=3)

    for i in range(100):
        _fail(limiter, f"10.0.0.{i}:admin")

    assert len(limiter._entries) <= 3


def test_expired_keys_are_reclaimed_for_new_keys():
    """Once tracked keys age out of the window their slots are reclaimed."""
    limiter, clock = _make(max_attempts=5, window_seconds=60, max_tracked_keys=2)
    _fail(limiter, "a")
    _fail(limiter, "b")  # table full
    assert len(limiter._entries) == 2

    clock.advance(61)  # "a" and "b" age out of the window
    _fail(limiter, "c")  # expired slot reclaimed -> "c" is tracked

    assert "c" in limiter._entries
    assert len(limiter._entries) <= 2


def test_active_lockout_survives_flood_of_unique_keys():
    """An in-window lockout must NOT be evicted to make room for new keys: an
    attacker could otherwise lock 'admin', then flood the same IP with unique
    fake usernames to push admin's record out and reset its counter.
    """
    limiter, _clock = _make(max_attempts=2, window_seconds=300, max_tracked_keys=3)
    admin = "1.2.3.4:admin"
    _fail(limiter, admin)
    _fail(limiter, admin)  # admin now locked, in-window
    assert limiter.retry_after(admin) is not None

    for i in range(100):  # flood unique usernames from the same IP
        _fail(limiter, f"1.2.3.4:user{i}")

    assert limiter.retry_after(admin) is not None  # lockout preserved
    assert len(limiter._entries) <= 3  # still bounded


def test_successful_login_frees_its_slot(monkeypatch):
    """A successful login must not leave a dead (empty) entry occupying a slot.

    Regression: previously reset() cleared the failures but kept the entry, so
    with the table at capacity a prior success sitting behind a live lockout
    made _make_room falsely report 'table full' and refuse a new key.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=3, window_seconds=300, max_tracked_keys=2)

    _fail(limiter, "1.1.1.1:a")  # A: live lockout candidate (stays at the front)
    _succeed(limiter, "2.2.2.2:b")  # B: success -> entry removed, slot freed
    assert "2.2.2.2:b" not in limiter._entries

    _fail(limiter, "3.3.3.3:c")  # C must still be trackable
    assert "3.3.3.3:c" in limiter._entries
    assert not any("table is full" in m for m in messages)


def test_reservation_alone_never_alerts(monkeypatch):
    """Reserving (and succeeding) must never emit a lockout alert, even when the
    reservation count reaches the limit -- only confirmed failures can.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=2, window_seconds=300)
    key = "1.2.3.4:admin"

    limiter.reserve_attempt(key)
    limiter.reserve_attempt(key)  # count reaches the limit purely by reservation
    assert messages == []

    limiter.reset(key)
    limiter.release(key)
    limiter.release(key)
    assert messages == []
    assert limiter.retry_after(key) is None


def test_concurrent_mixed_results_do_not_false_alert(monkeypatch):
    """Five concurrent attempts, one wrong and four correct, must NOT log a
    lockout: only one attempt is a real failure. Regression for the deque that
    conflated in-flight reservations with confirmed failures.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, _clock = _make(max_attempts=5, window_seconds=300)
    key = "1.2.3.4:admin"

    for _ in range(5):  # 5 requests reserve before any bcrypt resolves
        limiter.reserve_attempt(key)
    assert limiter.retry_after(key) is not None  # gate blocks a 6th (pending)

    limiter.commit_failure(key)  # A: wrong password
    limiter.release(key)
    for _ in range(4):  # B-E: correct password
        limiter.reset(key)
        limiter.release(key)

    assert messages == []  # only one real failure -> no false lockout alert
    assert limiter.retry_after(key) is None  # not locked


def test_concurrent_all_failures_alert_exactly_once(monkeypatch):
    """When all concurrent attempts are genuine failures, the lockout alert is
    logged exactly once (not once per resolving request).
    """
    messages = _capture_warnings(monkeypatch)
    # alert_interval_seconds=0 disables throttle collapsing, so a per-commit
    # alert bug would show as multiple messages.
    limiter, _clock = _make(
        max_attempts=5, window_seconds=300, alert_interval_seconds=0
    )
    key = "1.2.3.4:admin"

    for _ in range(5):
        limiter.reserve_attempt(key)
    for _ in range(5):
        limiter.commit_failure(key)
        limiter.release(key)

    assert len(messages) == 1  # exactly one lockout alert


def test_lockout_alerts_are_throttled_with_suppressed_count(monkeypatch):
    """A burst of lockouts must not flood the log: only one WARNING per interval,
    with the suppressed count reported on the next emitted line.
    """
    messages = _capture_warnings(monkeypatch)
    limiter, clock = _make(
        max_attempts=1, window_seconds=300, alert_interval_seconds=60
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
    limiter, _clock = _make(max_attempts=5, window_seconds=300, max_tracked_keys=1)

    _fail(limiter, "a")  # fills the table (not locked; max_attempts=5)
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
