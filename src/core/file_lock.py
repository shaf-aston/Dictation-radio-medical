"""Cross-process file locks: one holder at a time, across every process.

The desktop app and the web app run side by side over the same data folder,
so a ``threading.Lock`` (one process) protects nothing between them. The OS
releases these locks when the holding process exits, however it exits, which
makes a lock a liveness test as well: a lock that can be taken has no living
holder.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

_POLL_SEC = 0.01


class FileLock:
    """An exclusive lock on *path* (created if missing), held until released.

    Two ``FileLock`` objects on one path exclude each other even inside one
    process, so it serialises threads as well as processes.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fd: Optional[int] = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Take the lock; wait up to *timeout* seconds (None waits forever, 0 never)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = None if timeout is None else time.monotonic() + timeout
        while not _try_lock(fd):
            if deadline is not None and time.monotonic() >= deadline:
                os.close(fd)
                return False
            time.sleep(_POLL_SEC)
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
