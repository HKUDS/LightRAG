"""In-process brute-force protection for the /login endpoint (CWE-307).

A sliding-window limiter, keyed by client IP + username, that blocks further
login attempts once a key accumulates too many failures.

Each key tracks two distinct kinds of state, which must not be conflated:

* ``confirmed`` -- timestamps of attempts verified as *wrong* passwords.
* ``pending``  -- in-flight reservations: requests that have passed the check
  and are running bcrypt but whose result is not yet known.

The **rate-limit decision** uses ``confirmed + pending`` so that many
simultaneous requests cannot all pass the check while bcrypt is in flight
(TOCTOU concurrency bypass). The **lockout alert** uses ``confirmed`` only, so a
correct password (or a burst of correct passwords) resolving alongside one wrong
one never produces a false "account locked" alert.

Lifecycle per request: ``reserve_attempt`` (before bcrypt) -> exactly one of
``commit_failure`` (wrong password) / ``reset`` (correct password), and always
``release`` (drop the in-flight reservation, typically in a ``finally``).

State is in-memory and per-process. The API server runs single-worker under
uvicorn (``config.update_uvicorn_mode_config`` forces workers=1); under gunicorn
with N workers the effective limit is N x ``max_attempts``, which still defeats
wordlist brute force. Use a shared store (e.g. Redis) for strict cross-worker
enforcement. All access happens on a single worker's event-loop thread, so no
locking is needed (mirrors the token-renewal cache in ``utils_api.py``).

The number of tracked keys is capped at ``max_tracked_keys``. When at capacity,
only keys that are neither in-window nor in-flight are evicted; a live record
(active lockout or pending request) is never evicted -- otherwise an attacker
could clear a lockout by flooding unique keys. If every tracked key is live, a
brand-new key is simply not tracked until a slot frees. This is a basic
in-process safety bound; large-scale or distributed attacks (including a
sustained flood that keeps the table full of live keys) are the job of an
upstream reverse proxy / WAF.
"""

import time
from collections import OrderedDict, deque
from typing import Callable, Optional

from ..utils import logger

# Upper bound on distinct keys held in memory (see module docstring).
DEFAULT_MAX_TRACKED_KEYS = 10_000

# Minimum seconds between two WARNING logs of the same category. Attack traffic
# is high-volume, so alerts are throttled (with a suppressed-count summary) to
# keep them from filling the log.
DEFAULT_ALERT_INTERVAL_SECONDS = 60.0

# Retry hint (seconds) returned when a key is blocked purely by in-flight
# reservations rather than confirmed failures: those resolve in well under a
# second (bcrypt), so the caller may retry shortly.
_PENDING_RETRY_AFTER_SECONDS = 1.0


def _safe_log_value(value: str, max_length: int = 200) -> str:
    """Make an untrusted value (e.g. an attacker-supplied username embedded in a
    rate-limit key) safe to put in a log line.

    Replaces non-printable characters -- notably CR/LF, which could otherwise be
    used to forge extra log lines (log injection) -- with '?' and truncates
    over-long values. Only the *logged* value is sanitized; the limiter's
    internal key is left untouched so counting is unaffected.
    """
    sanitized = "".join(ch if ch.isprintable() else "?" for ch in value)
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "…(truncated)"
    return sanitized


class _Entry:
    """Per-key state: confirmed-failure timestamps and in-flight reservations."""

    __slots__ = ("failures", "pending")

    def __init__(self) -> None:
        self.failures: deque = deque()  # confirmed wrong-password times, oldest first
        self.pending: int = 0  # reservations whose result is not yet known


class LoginRateLimiter:
    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_keys: int = DEFAULT_MAX_TRACKED_KEYS,
        alert_interval_seconds: float = DEFAULT_ALERT_INTERVAL_SECONDS,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.max_tracked_keys = max_tracked_keys
        self.alert_interval_seconds = alert_interval_seconds
        self._clock = clock
        # Ordered so the least-recently-active key can be evicted in O(1).
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        # Per-category throttle state for _alert.
        self._alert_last: dict[str, float] = {}
        self._alert_suppressed: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Rate limiting is off when either bound is non-positive."""
        return self.max_attempts > 0 and self.window_seconds > 0

    def _prune(self, entry: _Entry, now: float) -> None:
        """Drop ``entry``'s confirmed failures that have aged out of the window."""
        cutoff = now - self.window_seconds
        dq = entry.failures
        while dq and dq[0] <= cutoff:
            dq.popleft()

    def _live_entry(self, key: str, now: float) -> Optional[_Entry]:
        """Return ``key``'s entry after pruning, dropping it if fully dead.

        An entry is dead once it has no in-window failures and no in-flight
        reservation, so idle keys do not accumulate.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._prune(entry, now)
        if not entry.failures and entry.pending == 0:
            del self._entries[key]
            return None
        return entry

    def retry_after(self, key: str) -> Optional[float]:
        """Seconds the caller must wait before attempting, or None if allowed.

        Uses confirmed failures + in-flight reservations so a burst of concurrent
        requests cannot all slip through before their results are known.
        """
        if not self.enabled:
            return None
        now = self._clock()
        entry = self._live_entry(key, now)
        if entry is None:
            return None
        confirmed = len(entry.failures)
        if confirmed + entry.pending < self.max_attempts:
            return None
        if confirmed >= self.max_attempts:
            # Genuine failure lockout: advise waiting until the oldest confirmed
            # failure ages out of the window.
            return max(0.0, self.window_seconds - (now - entry.failures[0]))
        # Blocked only because in-flight reservations push the total to the
        # limit (confirmed failures are still below it). Those resolve in well
        # under a second and may all succeed -- clearing the failures -- so
        # advise a short retry rather than the full window.
        return _PENDING_RETRY_AFTER_SECONDS

    def reserve_attempt(self, key: str) -> None:
        """Reserve one in-flight attempt for ``key`` *before* password verification.

        Counts toward the limit (via ``pending``) so concurrent requests cannot
        bypass the check while bcrypt runs. Pair every call with exactly one
        ``release`` (typically in a ``finally``). Never alerts.
        """
        if not self.enabled:
            return
        now = self._clock()
        entry = self._live_entry(key, now)
        if entry is None:
            if not self._make_room(now):
                # Table is full of live records. Do NOT evict one to make room --
                # that could clear an active lockout. Leave this key untracked;
                # release/commit/reset below all no-op safely for a missing key.
                return
            entry = self._entries[key] = _Entry()
        entry.pending += 1
        self._entries.move_to_end(key)

    def release(self, key: str) -> None:
        """Drop one in-flight reservation for ``key`` (call once per reserve).

        Removes the entry as soon as it is fully idle (no reservation and no
        confirmed failures) so successful logins do not leave dead entries
        occupying capacity. Keeping the map free of idle entries is also what
        lets ``_make_room`` trust that a live front implies a live table.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.pending > 0:
            entry.pending -= 1
        if entry.pending == 0 and not entry.failures:
            del self._entries[key]

    def commit_failure(self, key: str) -> None:
        """Record a confirmed failure (call only after a verified wrong password).

        Emits the throttled lockout alert once, exactly when confirmed failures
        reach the limit -- never on a reservation and never on success.
        """
        if not self.enabled:
            return
        now = self._clock()
        entry = self._entries.get(key)
        if entry is None:
            return  # untracked (table was full when the attempt was reserved)
        self._prune(entry, now)
        entry.failures.append(now)
        self._entries.move_to_end(key)
        if len(entry.failures) == self.max_attempts:
            self._alert(
                now,
                "lockout",
                f"Login rate limit tripped for '{_safe_log_value(key)}': "
                f"{self.max_attempts} failed attempts within "
                f"{self.window_seconds:.0f}s; further attempts are blocked "
                f"(HTTP 429)",
            )

    def reset(self, key: str) -> None:
        """Clear confirmed failures for ``key`` (call on a successful login).

        In-flight reservations for other concurrent requests are left untouched;
        the entry is removed once it has neither a reservation nor a failure.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.failures.clear()
        if entry.pending == 0:
            del self._entries[key]

    def _make_room(self, now: float) -> bool:
        """Ensure there is room for one new key; return True if a slot is free.

        Evicts only dead keys (no in-window failures, no in-flight reservation).
        Keys are ordered least-recently-active first, so once the front key is
        live every remaining key is too and nothing more can be evicted.
        """
        while len(self._entries) >= self.max_tracked_keys:
            key, entry = next(iter(self._entries.items()))  # least-recently-active
            self._prune(entry, now)
            if entry.failures or entry.pending > 0:
                # Capacity exhausted by live records: a likely flood. New keys go
                # untracked until a slot frees, so warn (throttled).
                self._alert(
                    now,
                    "table_full",
                    f"Login rate-limit table is full ({self.max_tracked_keys} "
                    f"active keys); new client/username pairs are temporarily "
                    f"not rate-limited -- possible flood, consider an upstream "
                    f"reverse proxy / WAF rate limit",
                )
                return False
            del self._entries[key]  # dead -> reclaim
        return True

    def _alert(self, now: float, category: str, message: str) -> None:
        """Emit a throttled WARNING for ``category``.

        At most one WARNING per ``alert_interval_seconds`` per category is
        logged; alerts arriving inside that window are counted and reported as a
        summary on the next emitted line, so an attack cannot flood the log.
        """
        last = self._alert_last.get(category)
        if last is not None and now - last < self.alert_interval_seconds:
            self._alert_suppressed[category] = (
                self._alert_suppressed.get(category, 0) + 1
            )
            return
        suppressed = self._alert_suppressed.pop(category, 0)
        self._alert_last[category] = now
        if suppressed:
            message += f" ({suppressed} more since last alert)"
        logger.warning(message)
