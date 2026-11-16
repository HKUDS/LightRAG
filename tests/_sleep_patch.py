"""Patch ``time.sleep`` for one module's calls only.

``patch("lightrag.x.time.sleep")`` does not patch a name in ``lightrag.x``: it
replaces ``sleep`` on the ``time`` module itself, for every caller in the
process. Any other thread that sleeps in a loop while the patch is active then
calls the mock too. ragas' analytics thread does (``time.sleep(1)`` every
second, started unconditionally when ragas is imported): with a mock that
returns at once it spins and floods ``call_args_list`` with thousands of
calls, failing assertions such as "slept twice" at random.

``patch_module_sleep`` records and short-circuits only the calls made directly
from ``module_name``, on any thread; every other caller gets the real
``time.sleep``.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Iterator
from unittest.mock import MagicMock, patch


@contextlib.contextmanager
def patch_module_sleep(module_name: str) -> Iterator[MagicMock]:
    """Mock ``time.sleep`` as seen by ``module_name``; yield the recorder."""
    real_sleep = time.sleep
    recorder = MagicMock(name=f"{module_name}.time.sleep")

    def _sleep(seconds):
        if sys._getframe(1).f_globals.get("__name__") == module_name:
            return recorder(seconds)
        return real_sleep(seconds)

    with patch(f"{module_name}.time.sleep", _sleep):
        yield recorder
