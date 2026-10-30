#!/usr/bin/env python3
"""
build_search_db.py — Build output/search.db with FTS5 full-text search.

Usage:
    .venv/bin/python3 build_search_db.py
    .venv/bin/python3 build_search_db.py --index ./output/stories_index.json
    .venv/bin/python3 build_search_db.py --force    # rebuild from scratch

Creates or updates:
    output/search.db   — SQLite with stories, story_tags, fts_docs and stories_fts tables.

Updates are incremental: only stories that are new, changed or removed since the
last run are re-indexed. The FTS index is contentless (no second copy of the text).

Run via:
    make build-search-db
"""

import argparse
import hashlib
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

# Bump when the schema changes; an older search.db is then rebuilt from scratch,
# and the browser ignores it until it is.
SCHEMA_VERSION = 2

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

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

-- One row per indexed story: its stories_fts rowid and a hash of what was
-- indexed, so an update re-indexes only stories whose text or metadata changed.
CREATE TABLE IF NOT EXISTS fts_docs (
    rowid     INTEGER PRIMARY KEY,
    story_id  TEXT NOT NULL UNIQUE,
    sig       TEXT NOT NULL
);

-- Contentless: stores only the word index, not a second copy of every story's
-- text. Columns can't be read back; join fts_docs on rowid for the story ID.
CREATE VIRTUAL TABLE IF NOT EXISTS stories_fts USING fts5(
    title,
    title_spaced,
    author,
    description,
    full_text,
    categories_flat,
    tags_flat,
    content='',
    contentless_delete=1,
    tokenize='unicode61 remove_diacritics 1'
);
"""


def _open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.executescript(_DDL)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


def _db_info(db_path: str) -> tuple:
    """(schema version, fulltext flag) of an existing search.db; (0, None) if unreadable."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < SCHEMA_VERSION:
                return version, None
            row = conn.execute("SELECT value FROM meta WHERE key='fulltext'").fetchone()
            return version, (row[0] == "1") if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return 0, None


# ── Full-text cache reader ─────────────────────────────────────────────────────

class _TextCache:
    """Per-story lookups in full_text_cache.db, for stories whose index entry has no
    text. Looked up one at a time so the build never holds every story's text in memory.
    """

    def __init__(self, cache_path: str):
        self.conn = None
        if os.path.exists(cache_path):
            self.conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)

    def get(self, sid: str) -> str:
        if not self.conn:
            return ""
        row = self.conn.execute("SELECT full_text FROM texts WHERE id=?", (sid,)).fetchone()
        return (row[0] or "") if row else ""

    def close(self):
        if self.conn:
            self.conn.close()


def _signature(fields: tuple) -> str:
    """Hash of everything that goes into a story's FTS row."""
    h = hashlib.blake2b(digest_size=16)
    for f in fields:
        h.update(f.encode("utf-8", "surrogatepass"))
        h.update(b"\x1f")
    return h.hexdigest()


# ── Build ──────────────────────────────────────────────────────────────────────

def build(index_path: str, db_path: str, text_cache_path: str,
          force: bool = False, no_fulltext: bool = False):
    """Write search.db: a stories table, a tags table, and an FTS5 index over title,
    author, description and tags (plus story text unless no_fulltext).

    Skips the run when search.db is newer than the index and was built in the same
    mode, unless force=True. Otherwise an existing search.db of the current schema is
    updated in place: stories whose indexed fields are unchanged are left alone, and
    only new, changed and removed stories touch the FTS index (one transaction, so a
    failed update leaves the old index intact). force=True, an old schema or no DB
    builds a fresh one in a temp file and swaps it in.

    Story text comes from the index; full_text_cache.db is only a fallback for stories
    whose index entry lacks it.
    """
    index_p = Path(index_path)
    db_p    = Path(db_path)

    if not index_p.exists():
        print(f"Error: index not found at {index_path}")
        sys.exit(1)

    version, db_fulltext = _db_info(db_path) if db_p.exists() else (0, None)
    same_mode = db_fulltext is not None and db_fulltext == (not no_fulltext)

    # mtime guard — skip if search.db is newer than the index and in the requested mode
    if not force and same_mode and db_p.stat().st_mtime >= index_p.stat().st_mtime:
        print(f"search.db is up to date (newer than index) — skipping rebuild.")
        print("  Use --force to rebuild anyway.")
        return

    incremental = not force and version == SCHEMA_VERSION
    mode = "metadata + tags only, no full-text" if no_fulltext else "with full text"
    if incremental:
        print(f"Updating search.db ({mode}) from {index_path}…")
        work_path = db_path
    else:
        print(f"Building search.db ({mode}) from {index_path}…")
        # Build into a fresh temp DB (so schema changes apply cleanly) and swap it in
        # at the end; if the build fails, the old search.db stays usable.
        work_path = f"{db_path}.tmp-{os.getpid()}"
        if os.path.exists(work_path):
            os.unlink(work_path)

    if not no_fulltext and not os.path.exists(text_cache_path):
        print(f"  Note: {text_cache_path} not found; using story text from the index only.")
    text_cache = _TextCache(text_cache_path) if not no_fulltext else _TextCache("")

    conn = _open_db(work_path)
    docs = {sid: (rowid, sig) for rowid, sid, sig in
            conn.execute("SELECT rowid, story_id, sig FROM fts_docs")}

    count      = 0
    text_count = 0
    added = changed = removed = 0
    tags_rows  = 0
    seen       = set()

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

    conn.execute("BEGIN IMMEDIATE")
    try:
        # stories and story_tags are small: rewrite them whole.
        conn.execute("DELETE FROM stories")
        conn.execute("DELETE FROM story_tags")

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
            raw_text   = "" if no_fulltext else (s.get("full_text") or text_cache.get(sid))
            has_text   = 1 if (raw_text or s.get("full_text")) else 0
            fetch_err  = 1 if s.get("fetch_error") else 0
            story_url  = s.get("url") or s.get("story_url") or ""
            author_url = s.get("author_url") or ""
            cats       = s.get("categories") or []

            conn.execute(
                "INSERT OR REPLACE INTO stories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, title, title_sp, author, desc,
                 word_count, ch_count, date_iso, date_raw,
                 length_cls, source, has_text, fetch_err,
                 story_url, author_url),
            )

            # Tags — categories, then auto_tag list dimensions
            tags = [(sid, "category", cat.lower()) for cat in cats]
            for dim in LIST_TAG_DIMS:
                for v in at.get(dim) or []:
                    tags.append((sid, dim, v.lower()))
            conn.executemany("INSERT INTO story_tags VALUES (?,?,?)", tags)
            tags_rows += len(tags)

            # FTS row — re-indexed only if what goes into it changed
            categories_flat = " ".join(cats)
            tags_flat = " ".join(
                v for dim in LIST_TAG_DIMS
                for v in (at.get(dim) or [])
            )
            if raw_text:
                text_count += 1
            sig = _signature((title, title_sp, author, desc,
                              raw_text, categories_flat, tags_flat))
            seen.add(sid)
            old = docs.get(sid)
            if old is None or old[1] != sig:
                if old is None:
                    rowid = conn.execute(
                        "INSERT INTO fts_docs(story_id, sig) VALUES (?,?)", (sid, sig)
                    ).lastrowid
                    added += 1
                else:
                    rowid = old[0]
                    conn.execute("DELETE FROM stories_fts WHERE rowid=?", (rowid,))
                    conn.execute("UPDATE fts_docs SET sig=? WHERE rowid=?", (sig, rowid))
                    changed += 1
                conn.execute(
                    "INSERT INTO stories_fts(rowid, title, title_spaced, author, description,"
                    " full_text, categories_flat, tags_flat) VALUES (?,?,?,?,?,?,?,?)",
                    (rowid, title, title_sp, author, desc,
                     _strip_html(raw_text), categories_flat, tags_flat),
                )
                docs[sid] = (rowid, sig)

            count += 1
            if count % 2000 == 0:
                print(f"{count:,}…", end=" ", flush=True)

        print(f"\n  {count:,} stories parsed, {text_count:,} have full text")

        # Stories no longer in the index
        for sid in set(docs) - seen:
            rowid = docs.pop(sid)[0]
            conn.execute("DELETE FROM stories_fts WHERE rowid=?", (rowid,))
            conn.execute("DELETE FROM fts_docs WHERE rowid=?", (rowid,))
            removed += 1

        conn.execute("INSERT OR REPLACE INTO meta VALUES ('fulltext', ?)",
                     ("0" if no_fulltext else "1",))

        print("  Committing…", end=" ", flush=True)
        conn.execute("COMMIT")
        print("done")
    except BaseException:
        conn.execute("ROLLBACK")
        conn.close()
        text_cache.close()
        if not incremental and os.path.exists(work_path):
            os.unlink(work_path)
        raise
    text_cache.close()

    if not incremental:
        # Merging every segment rewrites the whole index, so only do it on a fresh
        # build; FTS5's automerge keeps an incrementally updated index in shape.
        print("  Optimizing FTS index…", end=" ", flush=True)
        conn.execute("INSERT INTO stories_fts(stories_fts) VALUES('optimize')")
        print("done")
    conn.close()

    if not incremental:
        os.replace(work_path, db_path)
    size_mb = db_p.stat().st_size / 1_048_576
    print(f"\n  search.db {'updated' if incremental else 'built'} → {db_path}  ({size_mb:.0f} MB)")
    print(f"  {count:,} stories  ·  {tags_rows:,} tag rows  ·  {text_count:,} full-text docs")
    print(f"  FTS: {added:,} added  ·  {changed:,} re-indexed  ·  {removed:,} removed")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point: --index, --db, --text-cache, --force, --no-fulltext."""
    p = argparse.ArgumentParser(description="Build FTS search database for MCScraper browser")
    p.add_argument("--index",    default=DEFAULT_INDEX,    help="Path to stories_index.json")
    p.add_argument("--db",       default=DEFAULT_DB,       help="Output path for search.db")
    p.add_argument("--text-cache", default=TEXT_CACHE_DB,  help="Path to full_text_cache.db")
    p.add_argument("--force",        action="store_true", help="Rebuild from scratch, even if DB is up to date")
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
