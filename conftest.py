# conftest.py — pytest configuration for MCScraper
#
# Registers custom marks so pytest doesn't warn about unknown marks
# and -m filtering works correctly.
#
# Marks:
#   network  — tests that make live HTTP requests (skipped by default)
#   semantic — tests that require sentence-transformers to be installed

import sys
from pathlib import Path

# Ensure project root is always on sys.path, regardless of invocation directory
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: marks tests that make live HTTP requests "
        "(deselect with: -m 'not network')"
    )
    config.addinivalue_line(
        "markers",
        "semantic: marks tests that require sentence-transformers "
        "(run with: make test-semantic)"
    )


# ── Guard: tests must never modify the real data folder ─────────────────────────
# output/ links to the real (Dropbox) data. Tests use temp dirs, but background
# threads or relative paths can leak; this fails the run if any real file changed.

import threading
import time

import pytest


def _snapshot(folder: Path) -> dict:
    snap = {}
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file() and not p.name.endswith(("-wal", "-shm", "-journal")):
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


@pytest.fixture(scope="session", autouse=True)
def real_data_untouched():
    out = PROJECT_DIR / "output"
    before = _snapshot(out)
    yield
    # Give daemon threads started by app tests a moment to finish writing
    deadline = time.time() + 10
    while time.time() < deadline and any(
            t.daemon and t.is_alive() for t in threading.enumerate()):
        time.sleep(0.2)
    after = _snapshot(out)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, "Tests modified the real data folder:\n  " + "\n  ".join(changed)
