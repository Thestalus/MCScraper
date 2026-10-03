#!/usr/bin/env python3
"""
build_search_db.py — Build output/search.db with FTS5 full-text search.

Usage:
    .venv/bin/python3 build_search_db.py
    .venv/bin/python3 build_search_db.py --index ./output/stories_index.json
    .venv/bin/python3 build_search_db.py --force    # rebuild even if up to date

Creates:
    output/search.db   — SQLite with stories, story_tags, and stories_fts tables.

Run via:
    make build-search-db
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(text: str) -> str:
    """Strip HTML tags before FTS indexing. No-op for plain-text stories."""
    if not text or "<" not in text:
        return text or ""
    return _TAG_RE.sub(" ", text)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_INDEX  = "./output/stories_index.json"
DEFAULT_DB     = "./output/search.db"
TEXT_CACHE_DB  = "./output/full_text_cache.db"

# All auto_tag list dimensions (stored as multi-value rows in story_tags).
# "length_class" is scalar — stored as a column in stories, not in story_tags.
LIST_TAG_DIMS = (
    "mind_control_type",
    "setting",
    "themes",
    "tone",
    "dominance",
    "pacing",
    "awareness",
    "sexual_acts",
    "fetish",
    "relationship",
)

_MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3,    'april': 4,
    'may': 5,     'june': 6,     'july': 7,      'august': 8,
    'september': 9,'october': 10, 'november': 11, 'december': 12,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def slug_to_title(slug: str) -> str:
    """Convert CamelCase/PascalCase slug to spaced title. Mirrors browser.py."""
    if not slug:
        return slug
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', slug)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced


def display_title(s: dict) -> str:
    """Return display title for a story dict. Mirrors browser.py dt()."""
    title = s.get("title") or ""
    sid   = s.get("id") or ""
    if not title:
        return slug_to_title(sid) if sid else ""
    if " " in title or len(title) <= 4:
        return title
    return slug_to_title(title)


def story_date_iso(date_str: str) -> str:
    """Convert '15 January 2024' → '2024-01-15', or '' on failure."""
    if not date_str:
        return ""
    parts = date_str.strip().lower().split()
    if len(parts) == 3:
        try:
            day   = int(parts[0])
            month = _MONTH_MAP.get(parts[1], 0)
            year  = int(parts[2])
            if month:
                return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
    return ""


# ── Schema ─────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS stories (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    title_spaced    TEXT,
    author          TEXT,
    description     TEXT,
    word_count      INTEGER DEFAULT 0,
    chapter_count   INTEGER DEFAULT 1,
    added_date_iso  TEXT,
    added_date_raw  TEXT,
    length_class    TEXT,
    source          TEXT,
    has_text        INTEGER DEFAULT 0,
    fetch_error     INTEGER DEFAULT 0,
    story_url       TEXT,
    author_url      TEXT
);

CREATE INDEX IF NOT EXISTS idx_stories_author      ON stories(author COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_stories_word_count  ON stories(word_count);
CREATE INDEX IF NOT EXISTS idx_stories_added_date  ON stories(added_date_iso);
CREATE INDEX IF NOT EXISTS idx_stories_length_class ON stories(length_class);
CREATE INDEX IF NOT EXISTS idx_stories_source      ON stories(source);

CREATE TABLE IF NOT EXISTS story_tags (
    story_id   TEXT NOT NULL,
    tag_type   TEXT NOT NULL,
    tag_value  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_story_tags_story    ON story_tags(story_id);
CREATE INDEX IF NOT EXISTS idx_story_tags_lookup   ON story_tags(tag_type, tag_value);
CREATE INDEX IF NOT EXISTS idx_story_tags_covering ON story_tags(story_id, tag_type, tag_value);

CREATE VIRTUAL TABLE IF NOT EXISTS stories_fts USING fts5(
    story_id    UNINDEXED,
    title,
    title_spaced,
    author,
    description,
    full_text,
    categories_flat,
    tags_flat,
    tokenize='unicode61 remove_diacritics 1'
);
"""


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_DDL)
    return conn


# ── Full-text cache reader ─────────────────────────────────────────────────────

def _load_full_text_cache(cache_path: str) -> dict:
    """Load id→full_text from full_text_cache.db into memory (~150 MB)."""
    if not os.path.exists(cache_path):
        print(f"  Warning: full_text_cache.db not found at {cache_path}")
        print("  Full-text search will not cover story body text.")
        print("  Run: make browse (it builds the cache on first start)")
        return {}
    print(f"  Loading full-text cache from {cache_path}…", end=" ", flush=True)
    conn = sqlite3.connect(cache_path)
    rows = conn.execute("SELECT id, full_text FROM texts").fetchall()
    conn.close()
    result = {row[0]: row[1] for row in rows if row[1]}
    print(f"{len(result):,} stories")
    return result


# ── Build ──────────────────────────────────────────────────────────────────────

def build(index_path: str, db_path: str, text_cache_path: str,
          force: bool = False, no_fulltext: bool = False):
    """Write search.db: a stories table, a tags table, and an FTS5 index over title,
    author, description and tags (plus story text unless no_fulltext). Skips the rebuild
    when search.db is newer than the index, unless force=True. Builds into a temp DB and
    swaps it in, so a failed build leaves the old one usable.
    """
    index_p = Path(index_path)
    db_p    = Path(db_path)

    if not index_p.exists():
        print(f"Error: index not found at {index_path}")
        sys.exit(1)

    # mtime guard — skip if search.db is newer than the index
    if not force and db_p.exists():
        if db_p.stat().st_mtime >= index_p.stat().st_mtime:
            print(f"search.db is up to date (newer than index) — skipping rebuild.")
            print("  Use --force to rebuild anyway.")
            return

    if no_fulltext:
        print(f"Building search.db (metadata + tags only, no full-text) from {index_path}…")
    else:
        print(f"Building search.db from {index_path}…")

    full_texts = _load_full_text_cache(text_cache_path)

    # Build into a fresh temp DB (so schema changes apply cleanly) and swap it in
    # at the end; if the build fails, the old search.db stays usable.
    final_db_path = db_path
    db_path = f"{final_db_path}.tmp-{os.getpid()}"
    if os.path.exists(db_path):
        os.unlink(db_path)

    conn = _open_db(db_path)

    stories_rows  = []
    tags_rows     = []
    fts_rows      = []
    count         = 0
    text_count    = 0

    def _stream_index():
        """Stream stories from index; uses ijson if available, else json.load."""
        try:
            import ijson
            with open(index_path, "rb") as f:
                for s in ijson.items(f, "stories.item"):
                    yield s
        except ImportError:
            print("  (ijson not installed — loading full JSON into memory)")
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("stories", []):
                yield s

    print("  Streaming index…", end=" ", flush=True)

    for s in _stream_index():
        sid        = s.get("id", "")
        title      = s.get("title") or ""
        title_sp   = display_title(s)
        author     = s.get("author") or ""
        desc       = s.get("description") or ""
        word_count = s.get("word_count") or 0
        ch_count   = s.get("chapter_count") or 1
        date_raw   = s.get("added_date") or ""
        date_iso   = story_date_iso(date_raw)
        at         = s.get("auto_tags") or {}
        length_cls = at.get("length_class") or ""
        source     = s.get("source") or None
        has_text   = 1 if (full_texts.get(sid) or s.get("full_text")) else 0
        fetch_err  = 1 if s.get("fetch_error") else 0
        story_url  = s.get("url") or s.get("story_url") or ""
        author_url = s.get("author_url") or ""
        cats       = s.get("categories") or []

        stories_rows.append((
            sid, title, title_sp, author, desc,
            word_count, ch_count, date_iso, date_raw,
            length_cls, source, has_text, fetch_err,
            story_url, author_url,
        ))

        # Tags — categories
        for cat in cats:
            tags_rows.append((sid, "category", cat.lower()))

        # Tags — auto_tag list dimensions
        for dim in LIST_TAG_DIMS:
            vals = at.get(dim) or []
            for v in vals:
                tags_rows.append((sid, dim, v.lower()))

        # FTS row
        full_text      = "" if no_fulltext else _strip_html(full_texts.get(sid) or s.get("full_text") or "")
        categories_flat = " ".join(cats)
        tags_flat = " ".join(
            v for dim in LIST_TAG_DIMS
            for v in (at.get(dim) or [])
        )
        if full_text:
            text_count += 1
        fts_rows.append((
            sid, title, title_sp, author, desc,
            full_text, categories_flat, tags_flat,
        ))

        count += 1
        if count % 2000 == 0:
            print(f"{count:,}…", end=" ", flush=True)

    print(f"\n  {count:,} stories parsed, {text_count:,} have full text")

    # Bulk insert
    print("  Inserting into stories…", end=" ", flush=True)
    conn.executemany(
        "INSERT OR REPLACE INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        stories_rows,
    )
    print("done")

    print("  Inserting into story_tags…", end=" ", flush=True)
    conn.executemany(
        "INSERT INTO story_tags VALUES (?,?,?)",
        tags_rows,
    )
    print(f"{len(tags_rows):,} rows")

    print("  Inserting into stories_fts…", end=" ", flush=True)
    conn.executemany(
        "INSERT INTO stories_fts VALUES (?,?,?,?,?,?,?,?)",
        fts_rows,
    )
    print("done")

    print("  Committing…", end=" ", flush=True)
    conn.commit()
    print("done")

    print("  Running PRAGMA optimize…", end=" ", flush=True)
    conn.execute("INSERT INTO stories_fts(stories_fts) VALUES('optimize')")
    conn.commit()
    conn.close()
    print("done")

    os.replace(db_path, final_db_path)
    db_path = final_db_path
    size_mb = db_p.stat().st_size / 1_048_576
    print(f"\n  search.db built → {db_path}  ({size_mb:.0f} MB)")
    print(f"  {count:,} stories  ·  {len(tags_rows):,} tag rows  ·  {text_count:,} full-text docs")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point: --index, --db, --text-cache, --force, --no-fulltext."""
    p = argparse.ArgumentParser(description="Build FTS search database for MCScraper browser")
    p.add_argument("--index",    default=DEFAULT_INDEX,    help="Path to stories_index.json")
    p.add_argument("--db",       default=DEFAULT_DB,       help="Output path for search.db")
    p.add_argument("--text-cache", default=TEXT_CACHE_DB,  help="Path to full_text_cache.db")
    p.add_argument("--force",        action="store_true", help="Rebuild even if DB is up to date")
    p.add_argument("--no-fulltext",  action="store_true",
                   help="Omit full_text from FTS index (smaller DB, faster cold start, no body search)")
    args = p.parse_args()

    build(
        index_path      = args.index,
        db_path         = args.db,
        text_cache_path = args.text_cache,
        force           = args.force,
        no_fulltext     = args.no_fulltext,
    )


if __name__ == "__main__":
    main()
