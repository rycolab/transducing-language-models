"""
Pickle-based result storage with atomic writes and file locking.

Copied from benchmarking/utils/bechmarking_utils.py to keep the new
module self-contained.
"""

from __future__ import annotations

import os
import pickle
from contextlib import contextmanager


def safe_load_pickle(path: str, default):
    """Load a pickle file, returning *default* on missing/corrupt file."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return default
    except (pickle.UnpicklingError, EOFError):
        return default


@contextmanager
def file_lock(lock_path: str):
    """Exclusive file lock using fcntl.flock (auto-released on process exit)."""
    import fcntl
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def atomic_pickle_dump(obj, path: str):
    """Write *obj* to *path* atomically via tmp + rename."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
