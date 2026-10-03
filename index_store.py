"""
index_store.py — the only safe way to load and save stories_index.json for writing.

stories_index.json is the canonical store (every story, its text and its tags).
Many scripts rewrite it, on more than one machine (it lives in a synced
folder). Three things can go wrong, and this module handles each:

  1. Two writers at once. A write lock file (stories_index.json.lock) records
     host, PID, command and start time. load_index(..., for_write=True) takes it
     for the rest of the process; a second writer refuses to start and says who
     holds it. A lock left behind by a crashed process on the same machine is
     cleared automatically.
  2. Lost updates the lock can't see (e.g. the other machine's lock hasn't
     synced yet). save_index() refuses to overwrite the file if its size or
     mtime changed since this process loaded (or last saved) it.
  3. Partial writes. save_index() writes a uniquely named temp file in the same
     folder, fsyncs it, and os.replace()s it over the original.

Read-only users (the browser, stats) don't need any of this; they read
stories_index.json or the derived SQLite DBs directly.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class IndexLockedError(RuntimeError):
    """Another process holds the write lock."""


class IndexChangedError(RuntimeError):
    """The index changed on disk after this process loaded it."""


_fingerprints: dict[str, tuple[int, int]] = {}   # abs path -> (size, mtime_ns) at load/save
_held_locks: set[str] = set()                    # lock paths this process holds


def _key(path) -> str:
    return str(Path(path).absolute())


def _fingerprint(path) -> tuple[int, int] | None:
    try:
        st = Path(path).stat()
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _lock_path(path) -> Path:
    return Path(str(Path(path).absolute()) + ".lock")


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows; ask the OS instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_write_lock(path) -> None:
    """Take the write lock for `path` until this process exits. Idempotent."""
    lock = _lock_path(path)
    if str(lock) in _held_locks:
        return
    info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]) if sys.argv else "",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                holder = {}
            same_host = holder.get("host") == info["host"]
            if same_host and isinstance(holder.get("pid"), int) and not _pid_alive(holder["pid"]):
                lock.unlink(missing_ok=True)   # left by a crashed process on this machine
                continue
            raise IndexLockedError(
                f"{Path(path).name} is being written by another job:\n"
                f"  host={holder.get('host')} pid={holder.get('pid')} "
                f"started={holder.get('started_at')}\n  command: {holder.get('command')}\n"
                f"Wait for it to finish. If no such job is running, delete {lock}"
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f)
        _held_locks.add(str(lock))
        atexit.register(release_write_lock, path)
        return


def release_write_lock(path) -> None:
    lock = _lock_path(path)
    if str(lock) in _held_locks:
        _held_locks.discard(str(lock))
        lock.unlink(missing_ok=True)


def load_index(path, *, for_write: bool = False) -> dict:
    """Load the index. for_write=True takes the write lock and remembers the
    file's fingerprint so save_index() can detect concurrent changes."""
    if for_write:
        acquire_write_lock(path)
    fp = _fingerprint(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if for_write:
        _fingerprints[_key(path)] = fp
    return data


def save_index(data: dict, path, *, indent: int | None = 2, validate: bool = False) -> None:
    """Atomically replace the index with `data`.

    Raises IndexChangedError if the file changed since load_index(for_write=True)
    or the previous save_index() in this process. validate=True re-reads the
    temp file and checks the story count before replacing (slow on a 1 GB file).
    """
    path = Path(path)
    acquire_write_lock(path)
    expected = _fingerprints.get(_key(path))
    current = _fingerprint(path)
    if _key(path) in _fingerprints and current != expected:
        raise IndexChangedError(
            f"{path} changed on disk since this job loaded it (another process or "
            f"machine wrote it). Not overwriting — re-run this job to apply its "
            f"changes on top of the current file.")

    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        os.chmod(tmp, mode)   # mkstemp creates 0600; keep the index's normal permissions
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if validate:
            with open(tmp, encoding="utf-8") as f:
                check = json.load(f)
            if len(check.get("stories", [])) != len(data.get("stories", [])):
                raise RuntimeError("Story count mismatch after write — original index untouched")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fingerprints[_key(path)] = _fingerprint(path)
