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
