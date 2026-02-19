from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Optional, Tuple


class HardwareRuntimeLock:
    """
    Cross-process non-blocking lock for hardware ownership.
    """

    def __init__(self, lock_path: str | Path = "/tmp/doggo_hardware.lock") -> None:
        self._path = str(lock_path)
        self._fd: Optional[object] = None
        self.owner: Optional[str] = None

    def acquire(self, owner: str) -> Tuple[bool, str]:
        if self._fd is not None:
            return True, "already held"

        fd = open(self._path, "a+")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fd.seek(0)
            held_by = fd.read().strip()
            fd.close()
            if held_by:
                return False, f"hardware lock held by '{held_by}'"
            return False, "hardware lock is held by another process"

        fd.seek(0)
        fd.truncate(0)
        fd.write(owner)
        fd.flush()
        os.fsync(fd.fileno())
        self._fd = fd
        self.owner = owner
        return True, "ok"

    def release(self) -> None:
        if self._fd is None:
            self.owner = None
            return
        try:
            self._fd.seek(0)
            self._fd.truncate(0)
            self._fd.flush()
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
        except Exception:
            pass
        finally:
            self._fd = None
            self.owner = None

