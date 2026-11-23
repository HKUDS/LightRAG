"""Start the critical step of several probe processes together.

Each probe calls :func:`arrive_and_wait` once it has imported and set up
everything, right before the step that must overlap; the test calls
:func:`release_when_ready` after starting them. A fixed start time would have
to allow for the slowest interpreter start on any machine, and every run
would pay that wait.

Imported by the probes as ``_rendezvous`` (they run as scripts from this
directory) and by the tests as ``tests.config_store._rendezvous``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

_POLL_SECONDS = 0.005


def arrive_and_wait(directory: str, timeout: float = 60.0) -> None:
    """Report this process ready, then block until the test says go."""
    rendezvous = Path(directory)
    (rendezvous / f"ready-{os.getpid()}").touch()
    deadline = time.monotonic() + timeout
    while not (rendezvous / "go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no go signal in {directory}")
        time.sleep(_POLL_SECONDS)


def release_when_ready(
    directory: Path, procs: list[subprocess.Popen], timeout: float = 60.0
) -> None:
    """Say go once every probe in ``procs`` is ready.

    Says go early when a probe exits first, so the others do not wait out
    their timeout: the exit code and stderr the caller asserts on say why.
    """
    deadline = time.monotonic() + timeout
    while len(list(directory.glob("ready-*"))) < len(procs):
        if any(proc.poll() is not None for proc in procs):
            break
        if time.monotonic() > deadline:
            raise AssertionError(f"probes not ready within {timeout}s")
        time.sleep(_POLL_SECONDS)
    (directory / "go").touch()
