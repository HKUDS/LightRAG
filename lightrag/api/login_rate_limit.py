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
"""

import time
from collections import deque
from typing import Callable, Optional


class LoginRateLimiter:
    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._clock = clock
        # key -> timestamps of recent failed attempts (oldest first)
        self._failures: dict[str, deque] = {}

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

    def record_failure(self, key: str) -> None:
        """Record one failed attempt for ``key``."""
        if not self.enabled:
            return
        self._failures.setdefault(key, deque()).append(self._clock())

    def reset(self, key: str) -> None:
        """Clear all recorded failures for ``key`` (called on successful login)."""
        self._failures.pop(key, None)
