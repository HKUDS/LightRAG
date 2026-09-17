"""Shared teardown for the hf tests: join the inference executor's workers.

Every test file in this directory loads ``lightrag.llm.hf`` fresh
(``sys.modules.pop()`` then ``import_module``) so its fake ``torch`` /
``transformers`` stubs take effect, and every fresh module object builds its
OWN module-level ``ThreadPoolExecutor``. Any test that submits inference
starts that pool's non-daemon worker thread.

Dropping the module does not clean that up, and no garbage-collection trick
can: importing ``lightrag.llm.hf`` hands a module-level function to
``os.register_at_fork``, CPython keeps a strong reference to it forever
(there is no ``os.unregister_at_fork``), that function's ``__globals__`` IS
the discarded module's dict, and the dict holds the executor. So every
discarded module keeps its pool -- and its worker thread -- alive for the
rest of the session, and they accumulate file by file.

Each leaked thread is harmless on its own, but they accumulate for the rest
of the session and leave every later fork in this process -- this directory's
and other directories' -- forking a more crowded parent. So the teardown below
also asserts that nothing survived it: that assertion is the only alarm, since
CPython's multi-threaded-fork warning is advisory and cannot be promoted to an
error.

An explicit shutdown is therefore the only fix, and this fixture is the single
place that performs it -- do not add a per-file teardown back.
"""

from __future__ import annotations

import sys
import threading

import pytest


@pytest.fixture(autouse=True)
def _shutdown_hf_inference_executor():
    """Shut down whatever hf inference pool the test left behind."""
    yield
    module = sys.modules.pop("lightrag.llm.hf", None)
    if module is not None:
        executor = getattr(module, "_HF_INFERENCE_EXECUTOR", None)
        module._HF_INFERENCE_EXECUTOR = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    # A name still here belongs to a module copy this fixture never saw -- a
    # test that rebuilt the module mid-run, say. Nothing else will ever join
    # it, so report it against the test that produced it.
    orphans = sorted(
        t.name
        for t in threading.enumerate()
        if t.name.startswith("lightrag-hf-inference")
    )
    assert orphans == [], f"hf inference worker threads outlived the test: {orphans}"
