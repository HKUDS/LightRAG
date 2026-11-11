"""``tests._sleep_patch.patch_module_sleep`` scopes a time.sleep mock to one module."""

import threading
import time

import pytest

from tests._sleep_patch import patch_module_sleep

pytestmark = pytest.mark.offline


def _sleep_here(seconds):
    time.sleep(seconds)


def test_the_target_modules_sleep_is_mocked_and_recorded():
    started = time.monotonic()
    with patch_module_sleep(__name__) as sleep:
        _sleep_here(5)
    assert time.monotonic() - started < 1
    assert [call.args[0] for call in sleep.call_args_list] == [5]


def test_another_threads_sleep_stays_real_and_unrecorded():
    """The CI flake: a background loop (ragas' analytics thread) sleeping
    while a test mocked ``time.sleep`` process-wide spun on the mock and
    flooded its call list. Here it must keep sleeping for real."""
    stop = threading.Event()
    counter = [0]
    # The loop must live in ANOTHER module, as ragas' thread does: only
    # calls made from the patched module are mocked.
    namespace = {"__name__": "some_library._analytics", "time": time}
    exec(
        "def loop(stop, counter):\n"
        "    while not stop.is_set():\n"
        "        time.sleep(0.01)\n"
        "        counter[0] += 1\n",
        namespace,
    )

    worker = threading.Thread(
        target=namespace["loop"], args=(stop, counter), daemon=True
    )
    worker.start()
    try:
        with patch_module_sleep(__name__) as sleep:
            time.sleep(0.2)  # this module's call: recorded, returns at once
            stop.wait(0.2)  # let the background loop run under the patch
    finally:
        stop.set()
        worker.join(timeout=5)

    assert [call.args[0] for call in sleep.call_args_list] == [0.2]
    # ~20 real 10 ms sleeps fit in 0.2 s; a loop spinning on a mock does
    # hundreds of thousands.
    assert counter[0] < 200
