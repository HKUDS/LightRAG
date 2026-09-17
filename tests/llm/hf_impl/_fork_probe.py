"""Fork probe for lightrag.llm.hf, run as a subprocess by
test_inference_executor_resets_after_fork.

Not a test module (the leading underscore keeps pytest from collecting it):
it must run in a FRESH interpreter. A pytest session that has executed a few
thousand tests is multi-threaded -- other tests leave long-lived pools
running, and on macOS a single httpx.Client() with trust_env on reaches
_scproxy and leaves two libdispatch workqueue threads no Python code can join.
Forking there makes CPython emit its multi-threaded-fork DeprecationWarning,
which cannot be promoted to an error (it is emitted after fork() has already
returned), so in-process the warning could only ever be filtered away.

Here the process is single-threaded, which is also what the scenario under
test actually looks like -- a gunicorn pre-fork master imports the app and
forks. That lets the warning be asserted ABSENT rather than suppressed: its
presence means something made this process multi-threaded and the probe fails.

Exit code 0 means every check passed; anything else is a failure explained on
stderr.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import warnings


def _install_stubs() -> None:
    # Inlined rather than reused from the test module: the helper there drives
    # monkeypatch, which does not exist in a plain script.
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoModelForCausalLM = object
    sys.modules["transformers"] = fake_transformers

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.device = lambda name: name
    fake_torch.no_grad = _NullContext
    sys.modules["torch"] = fake_torch

    import pipmaster as pm

    pm.is_installed = lambda name: True


def main() -> int:
    _install_stubs()

    import lightrag.llm.hf as hf

    failures = []

    # Build the executor but submit nothing: a live worker thread would make
    # this process multi-threaded, which is exactly the state this probe
    # exists to avoid. What is under test is the at-fork reset, not the pool.
    hf._get_hf_inference_executor()
    if hf._HF_INFERENCE_EXECUTOR is None:
        failures.append("parent: _get_hf_inference_executor() left the slot empty")

    alive = sorted(t.name for t in threading.enumerate())
    if alive != ["MainThread"]:
        failures.append(f"parent: expected a single thread before fork, got {alive}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pid = os.fork()
        if pid == 0:
            # Exit through os._exit so nothing in the child returns to the
            # parent's control flow: the guard must be a fresh unlocked Lock
            # and the executor slot must be empty for the child to rebuild.
            ok = (
                hf._HF_INFERENCE_EXECUTOR is None
                and not hf._HF_INFERENCE_EXECUTOR_GUARD.locked()
            )
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)

    if not os.WIFEXITED(status):
        failures.append(f"child: did not exit normally (status={status})")
    elif os.WEXITSTATUS(status) != 0:
        failures.append(
            "child: os.register_at_fork did not reset the executor and its guard"
        )

    for warning in caught:
        failures.append(f"parent: fork() warned -- {warning.message}")

    for line in failures:
        print(line, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
