from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileLock:
    """A small cross-process file lock.

    This avoids adding a third-party dependency. The lock is advisory and meant
    to serialize prompt git operations across workers.
    """

    path: Path
    _fh: Optional[object] = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+", encoding="utf-8")
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            else:  # pragma: no cover
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
        except Exception:
            fh.close()
            raise
        self._fh = fh
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        fh = self._fh
        self._fh = None
        if fh is None:
            return
        try:
            if os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            else:  # pragma: no cover
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            fh.close()
