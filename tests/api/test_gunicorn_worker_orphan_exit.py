"""A worker whose gunicorn master died must stop serving.

gunicorn's sync worker exits when ``self.ppid != os.getppid()``; UvicornWorker
replaces ``run()`` with an asyncio server and never checks, so ``kill -9`` on the
master (which reaches one process only — POSIX does not cascade) left workers
serving forever, holding the inherited listening socket so the port stayed bound
and a replacement master could not start.
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest
from uvicorn_worker import UvicornWorker

from lightrag.api.gunicorn_worker import LightRAGUvicornWorker

pytestmark = pytest.mark.offline


class _Log:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def _worker(ppid: int) -> LightRAGUvicornWorker:
    """A worker with only the attributes the check touches.

    ``__new__`` on purpose: ``UvicornWorker.__init__`` builds a uvicorn ``Config``
    from a live gunicorn ``cfg``, which is a whole arbiter's worth of setup for a
    branch that reads two integers.
    """
    worker = LightRAGUvicornWorker.__new__(LightRAGUvicornWorker)
    worker.ppid = ppid
    worker.log = _Log()
    return worker


def test_gunicorn_graceful_timeout_is_forwarded_to_uvicorn(monkeypatch):
    """Without this mapping uvicorn waits forever for active ASGI requests after
    the orphan-triggered SIGTERM because its shutdown timeout defaults to None."""

    def fake_init(worker, *args, **kwargs):
        worker.cfg = SimpleNamespace(graceful_timeout=47)
        worker.config = SimpleNamespace(timeout_graceful_shutdown=None)

    monkeypatch.setattr(UvicornWorker, "__init__", fake_init)

    worker = LightRAGUvicornWorker()

    assert worker.config.timeout_graceful_shutdown == 47


def test_an_orphaned_worker_requests_a_graceful_shutdown(monkeypatch):
    """Fix-proof: nothing used to notice, and the worker kept serving."""
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

    worker = _worker(ppid=os.getppid() + 1000)  # a master this process never had
    assert worker.exit_if_orphaned() is True

    # SIGTERM to ourselves: uvicorn installs its own handler for the whole of
    # serve(), which stops accepting and drains in-flight requests. SIGKILL would
    # drop them, and touching Server.should_exit is not reachable from here.
    assert signals == [(os.getpid(), signal.SIGTERM)]
    assert any("Parent changed" in message for message in worker.log.messages)


def test_a_live_master_is_left_alone(monkeypatch):
    """The check must be silent on every ordinary tick — it runs for the life of
    the process, so a false positive would kill a healthy worker."""
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

    worker = _worker(ppid=os.getppid())
    assert worker.exit_if_orphaned() is False
    assert signals == []
    assert worker.log.messages == []


@pytest.mark.asyncio
async def test_the_check_rides_the_notify_tick_and_still_notifies(monkeypatch):
    """It must hang off uvicorn's existing notify callback (no timer of its own)
    AND keep doing what that callback was for — gunicorn murders a worker that
    stops notifying."""
    notified: list[bool] = []
    monkeypatch.setattr(
        LightRAGUvicornWorker, "notify", lambda self: notified.append(True)
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

    worker = _worker(ppid=os.getppid() + 1000)
    await worker.callback_notify()

    assert notified == [True]
    assert signals == [(os.getpid(), signal.SIGTERM)]


def test_the_configured_worker_class_resolves(monkeypatch):
    """The class is wired in by dotted path, so a rename would only surface at
    deploy time — gunicorn resolves it here instead."""
    from gunicorn.util import load_class

    monkeypatch.setattr("sys.argv", ["pytest"])
    from lightrag.api import gunicorn_config

    assert load_class(gunicorn_config.worker_class) is LightRAGUvicornWorker, (
        gunicorn_config.worker_class
    )
