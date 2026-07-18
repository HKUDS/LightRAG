"""In-process brute-force protection for the /login endpoint (CWE-307).

A sliding-window limiter that counts *failed* login attempts per key (client
IP + username) and rejects further attempts once the count reaches
``max_attempts`` within ``window_seconds``. A successful login clears the key.

State is in-memory and per-process. The API server runs single-worker under
uvicorn (``config.update_uvicorn_mode_config`` forces workers=1); under gunicorn
with N workers the effective limit is N x ``max_attempts``, which still defeats
wordlist brute force. Use a shared store (e.g. Redis) if strict cross-worker
enforcement is required.

All access happens on a single worker's event-loop thread, so no locking is
needed (mirrors the token-renewal cache in ``utils_api.py``).

The number of tracked keys is capped at ``max_tracked_keys`` so a flood of
unique IP/username pairs cannot grow the map without bound (the map would
otherwise be its own memory-exhaustion DoS). When at capacity, only keys whose
failures have already aged out of the window are evicted; a key that is still
in-window (possibly an active lockout) is never evicted -- otherwise an attacker
could clear a lockout by flooding unique keys. If every tracked key is in-window,
a brand-new key is simply not tracked until a slot frees. This is a basic
in-process safety bound only; defending against large-scale or distributed
attacks (including a sustained flood that keeps the table full of live keys) is
the job of an upstream reverse proxy / WAF.
"""

import time
from collections import OrderedDict, deque
from typing import Callable, Optional

from ..utils import logger

# Upper bound on distinct keys held in memory (see module docstring). Each key
# holds at most ``max_attempts`` timestamps, so total memory stays small.
DEFAULT_MAX_TRACKED_KEYS = 10_000

# Minimum seconds between two WARNING logs of the same category. Attack traffic
# is high-volume, so alerts are throttled (with a suppressed-count summary) to
# keep them from filling the log.
DEFAULT_ALERT_INTERVAL_SECONDS = 60.0


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
        # key -> timestamps of recent failed attempts (oldest first). Ordered so
        # the least-recently-active key can be evicted in O(1) when at capacity.
        self._failures: "OrderedDict[str, deque]" = OrderedDict()
        # Per-category throttle state for _alert: last emit time and how many
        # alerts have been suppressed since then.
        self._alert_last: dict[str, float] = {}
        self._alert_suppressed: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Rate limiting is off when either bound is non-positive."""
        return self.max_attempts > 0 and self.window_seconds > 0

    def _recent_failures(self, key: str, now: float) -> Optional[deque]:
        """Return the in-window failures for ``key``, pruning expired ones.

        Drops the key entirely (and returns None) once nothing is in-window, so
        the map does not grow without bound.
        """
        dq = self._failures.get(key)
        if dq is None:
            return None
        cutoff = now - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()
        if not dq:
            del self._failures[key]
            return None
        return dq

    def retry_after(self, key: str) -> Optional[float]:
        """Seconds the caller must wait before retrying, or None if allowed."""
        if not self.enabled:
            return None
        now = self._clock()
        dq = self._recent_failures(key, now)
        if dq is None or len(dq) < self.max_attempts:
            return None
        # Locked until the oldest in-window failure ages out.
        return max(0.0, self.window_seconds - (now - dq[0]))

    def reserve_attempt(self, key: str) -> None:
        """Reserve one attempt for ``key`` *before* password verification.

        Counts toward the limit so that many concurrent requests cannot all pass
        ``retry_after`` while bcrypt is in flight (TOCTOU). Does NOT alert -- the
        attempt may still succeed. Call ``commit_failure`` after a confirmed
        wrong password, or ``reset`` after a successful login.
        """
        if not self.enabled:
            return
        now = self._clock()
        dq = self._recent_failures(key, now)
        if dq is None:
            if not self._make_room(now):
                # Table is full of in-window records. Do NOT evict a live record
                # to make space -- that would let an attacker clear an active
                # lockout by flooding unique keys. Skip tracking this brand-new
                # key instead; a slot frees once existing records age out.
                return
            dq = self._failures[key] = deque()
        dq.append(now)
        self._failures.move_to_end(key)  # mark most-recently-active

    def commit_failure(self, key: str) -> None:
        """Confirm a reserved attempt as a genuine failure (wrong password).

        Emits the throttled lockout alert if this failure has tripped the limit.
        Because it runs only after the password is verified *wrong*, a correct
        password on the Nth attempt no longer produces a false lockout alert.
        """
        if not self.enabled:
            return
        now = self._clock()
        dq = self._recent_failures(key, now)
        if dq is not None and len(dq) >= self.max_attempts:
            self._alert(
                now,
                "lockout",
                f"Login rate limit tripped for '{_safe_log_value(key)}': "
                f"{self.max_attempts} failed attempts within "
                f"{self.window_seconds:.0f}s; further attempts are blocked "
                f"(HTTP 429)",
            )

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

    def _make_room(self, now: float) -> bool:
        """Ensure there is room for one new key; return True if a slot is free.

        Evicts only keys whose most recent failure has aged out of the window.
        Keys are ordered least-recently-active first, so once the front key is
        still in-window every remaining key is too, and nothing more can be
        evicted (returns False without touching any live record).
        """
        cutoff = now - self.window_seconds
        while len(self._failures) >= self.max_tracked_keys:
            _key, dq = next(iter(self._failures.items()))  # least-recently-active
            if dq and dq[-1] > cutoff:
                # Oldest key is live -> so are all others. Capacity is exhausted
                # by in-window records: a likely flood. New keys go untracked
                # until a slot frees, so warn (throttled) to surface it.
                self._alert(
                    now,
                    "table_full",
                    f"Login rate-limit table is full ({self.max_tracked_keys} "
                    f"active keys); new client/username pairs are temporarily "
                    f"not rate-limited -- possible flood, consider an upstream "
                    f"reverse proxy / WAF rate limit",
                )
                return False
            self._failures.popitem(last=False)  # fully expired -> reclaim
        return True

    def reset(self, key: str) -> None:
        """Clear all recorded failures for ``key`` (called on successful login)."""
        self._failures.pop(key, None)
