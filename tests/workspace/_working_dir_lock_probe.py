"""Process-tree probe for ``lightrag.kg.working_dir_lock``, run as a
subprocess by ``test_working_dir_lock.py``.

Not a test module (the leading underscore keeps pytest from collecting it):
each command runs in a FRESH interpreter, which is what "another process tree"
means for the claim under test, and the only place ``fork`` can be exercised
cleanly. A pytest session that has executed a few thousand tests is
multi-threaded -- other tests leave long-lived pools running, and on macOS
libdispatch adds workqueue threads no Python code can join. Forking there makes
CPython emit its multi-threaded-fork DeprecationWarning, which cannot be
promoted to an error (it is emitted after ``fork()`` has already returned), so
in-process the warning could only ever be filtered away. Here the process is
single-threaded, which is also what a Gunicorn pre-fork master actually looks
like, so the warning is asserted ABSENT rather than suppressed.

Commands (the directory is the second argument):

- ``foreign``  -- try to claim it; prints ``ADMITTED`` or ``REFUSED``.
- ``dying``    -- claim it and die through ``os._exit`` without releasing.
- ``workers``  -- claim it, fork four children that each claim it again, and
  print one line per child (``OK`` or ``REFUSED``).

Exit code 0 means the command ran as described; anything else is a failure
explained on stderr.
"""

from __future__ import annotations

import os
import sys
import warnings


def _attempt(path: str) -> str:
    from lightrag.exceptions import WorkingDirectoryInUseError
    from lightrag.kg.working_dir_lock import acquire_working_dir_lock

    try:
        acquire_working_dir_lock(path)
    except WorkingDirectoryInUseError:
        return "REFUSED"
    return "ADMITTED"


def foreign(path: str) -> int:
    print(_attempt(path))
    return 0


def dying(path: str) -> int:
    from lightrag.kg.working_dir_lock import acquire_working_dir_lock

    acquire_working_dir_lock(path)
    os._exit(0)  # dies holding it: no release, no cleanup


def workers(path: str) -> int:
    from lightrag.kg.working_dir_lock import acquire_working_dir_lock

    acquire_working_dir_lock(path)  # the master claims BEFORE forking

    failures = []
    for _ in range(4):
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pid = os.fork()
            if pid == 0:  # pragma: no cover - runs in the child
                os.close(read_fd)
                answer = "OK" if _attempt(path) == "ADMITTED" else "REFUSED"
                os.write(write_fd, answer.encode())
                os._exit(0)
        os.close(write_fd)
        answer = os.read(read_fd, 16).decode()
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        for warning in caught:
            failures.append(f"parent: fork() warned -- {warning.message}")
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            failures.append(f"child: did not exit cleanly (status={status})")
        print(answer)

    for line in failures:
        print(line, file=sys.stderr)
    return 1 if failures else 0


COMMANDS = {"foreign": foreign, "dying": dying, "workers": workers}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in COMMANDS:
        print(
            f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}} DIRECTORY", file=sys.stderr
        )
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2]))
