"""
build_stories_db.py — Build output/stories_meta.db from stories_index.json.

Stores all story metadata (no full_text) in a SQLite database so the browser
can load at startup without parsing the 1.2GB JSON file.

Usage:
    python build_stories_db.py [--index PATH] [--force]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_INDEX = "./output/stories_index.json"
DEFAULT_DB    = "./output/stories_meta.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    author        TEXT,
    author_url    TEXT,
    story_url     TEXT,
    categories    TEXT,
    description   TEXT,
    word_count    INTEGER,
    chapter_count INTEGER,
    added_date    TEXT,
    updated_date  TEXT,
    fetch_error   TEXT,
    source        TEXT,
    auto_tags     TEXT,
    has_text      INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_author     ON stories(author);
CREATE INDEX IF NOT EXISTS idx_word_count ON stories(word_count);
CREATE INDEX IF NOT EXISTS idx_added_date ON stories(added_date);
CREATE INDEX IF NOT EXISTS idx_source     ON stories(source);
"""


def build(index_path: str, db_path: str, force: bool = False) -> None:
    index_p = Path(index_path)
    db_p    = Path(db_path)

    if not index_p.exists():
        print(f"Error: index not found at {index_path}", file=sys.stderr)
        sys.exit(1)

    index_mtime = index_p.stat().st_mtime

    if not force and db_p.exists():
        db_mtime = db_p.stat().st_mtime
        if db_mtime >= index_mtime:
            print(f"stories_meta.db is up to date (index unchanged). Use --force to rebuild.")
            return

    print(f"Loading {index_path} ...")
    t0 = time.time()
    with open(index_p, encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    stories = data.get("stories", [])
    print(f"  {len(stories):,} stories")

    # Remove old DB so we get a clean write
    tmp_path = str(db_p) + ".tmp"
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    conn = sqlite3.connect(tmp_path)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    t1 = time.time()
    rows = []
    for s in stories:
        rows.append((
            s.get("id"),
            s.get("title"),
            s.get("author"),
            s.get("author_url"),
            s.get("story_url"),
            json.dumps(s.get("categories") or [], ensure_ascii=False),
            s.get("description"),
            s.get("word_count") or 0,
            s.get("chapter_count") or 1,
            s.get("added_date"),
            s.get("updated_date"),
            s.get("fetch_error"),
            s.get("source"),
            json.dumps(s.get("auto_tags") or {}, ensure_ascii=False),
            1 if s.get("full_text") else 0,
        ))

    conn.executemany(
        """INSERT OR REPLACE INTO stories
           (id, title, author, author_url, story_url, categories, description,
            word_count, chapter_count, added_date, updated_date, fetch_error,
            source, auto_tags, has_text)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )

    # Store top-level metadata
    for key in ("scraped_at", "last_updated", "total_stories"):
        val = data.get(key)
        if val is not None:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (key, str(val)),
            )

    conn.commit()
    conn.close()

    # Atomic replace
    if db_p.exists():
        db_p.unlink()
    os.rename(tmp_path, str(db_p))

    print(f"  Written {len(rows):,} rows in {time.time()-t1:.1f}s → {db_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build stories_meta.db from stories_index.json")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--db",    default=DEFAULT_DB)
    parser.add_argument("--force", action="store_true", help="Rebuild even if DB is newer than index")
    args = parser.parse_args()
    build(args.index, args.db, args.force)
