"""Gunicorn worker class: uvicorn plus the orphan check uvicorn drops.

gunicorn's own workers exit when their parent changes — ``sync.py`` checks
``self.ppid != os.getppid()`` on every loop turn. ``UvicornWorker`` replaces
``run()`` with an asyncio server and never makes that check, so a master that
dies without signalling its children (``kill -9 <master_pid>``, which reaches
one process only; POSIX does not cascade to children) leaves the workers
serving forever:

* nothing restarts a worker that then crashes, and ``--reload`` / SIGHUP have
  no master to drive them;
* the workers keep the inherited listening socket open, so the port stays bound
  and a replacement master cannot start (gunicorn sets ``SO_REUSEADDR``, and
  ``reuse_port`` is off by default) — the operator sees ``EADDRINUSE`` from a
  server they believe is dead.

Detection rides uvicorn's own notify tick (``callback_notify``, driven roughly
every ``timeout / 2`` seconds — the value gunicorn's arbiter hands each worker),
so this adds no timer of its own. Shutdown goes through SIGTERM rather than
poking the server object: ``callback_notify`` has no handle on the ``Server``
instance (``UvicornWorker._serve`` keeps it local), and uvicorn installs its own
SIGTERM handler for the whole of ``serve()`` (``capture_signals`` →
``handle_exit`` → ``should_exit``), which is the documented graceful path — it
stops accepting, drains in-flight requests, then exits.

``UvicornWorker`` does not forward gunicorn's ``graceful_timeout`` to
uvicorn, whose graceful-shutdown timeout otherwise defaults to unlimited.
Forwarding it here bounds the drain of active ASGI requests after SIGTERM.
LightRAG's application-managed background work keeps using the lifespan
cancellation-and-join cleanup path.
"""

from __future__ import annotations

import os
import signal
from typing import Any

from uvicorn.workers import UvicornWorker


class LightRAGUvicornWorker(UvicornWorker):
    """Uvicorn worker that stops serving once its gunicorn master is gone."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config.timeout_graceful_shutdown = self.cfg.graceful_timeout

    async def callback_notify(self) -> None:
        await super().callback_notify()
        self.exit_if_orphaned()

    def exit_if_orphaned(self) -> bool:
        """Request a graceful shutdown when the master is gone. Returns whether
        this worker was orphaned (kept small and separate so the condition is
        testable without a live arbiter)."""
        if self.ppid == os.getppid():
            return False
        # Logs the pids directly rather than ``%s`` on self: gunicorn's
        # ``Worker.__str__`` reads ``self.pid``, and a shutdown path is the worst
        # place to depend on another attribute being populated.
        self.log.info(
            "Parent changed (master was %s, now %s), shutting down worker %s",
            self.ppid,
            os.getppid(),
            os.getpid(),
        )
        os.kill(os.getpid(), signal.SIGTERM)
        return True
