"""Process-tree probe for ``lightrag.kg.working_dir_lock``, run as a
subprocess by ``test_working_dir_lock.py``.

Each command runs in a fresh interpreter so it cannot inherit the calling
process's claim bookkeeping or lock descriptor.

Commands (the directory is the second argument):

- ``foreign``  -- try to claim it; prints ``ADMITTED`` or ``REFUSED``.
- ``dying``    -- claim it and die through ``os._exit`` without releasing.
- ``serve``    -- no directory: answer ``foreign`` for every directory read
  from stdin, one per line, and release an admitted claim right after, so
  the next attempt finds the claim as a newly started process would. One
  interpreter then stands in for a fresh one per attempt.

Exit code 0 means the command ran as described; anything else is a failure
explained on stderr.
"""

from __future__ import annotations

import os
import sys


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


def serve() -> int:
    from lightrag.kg import working_dir_lock as wdl

    for line in sys.stdin:
        path = line.rstrip("\n")
        answer = _attempt(path)
        if answer == "ADMITTED":
            wdl.release_working_dir_lock(path)
        if wdl._claims:
            print(f"claims left behind: {list(wdl._claims)}", file=sys.stderr)
            return 1
        print(answer, flush=True)
    return 0


COMMANDS = {"foreign": foreign, "dying": dying}


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        sys.exit(serve())
    if len(sys.argv) != 3 or sys.argv[1] not in COMMANDS:
        print(
            f"usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}} DIRECTORY", file=sys.stderr
        )
        sys.exit(2)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2]))
