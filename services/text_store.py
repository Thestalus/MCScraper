"""
services/text_store.py — Full-text story content storage (SQLite cache + fallback).
"""
from __future__ import annotations
import json
import os
import sqlite3
from pathlib import Path


def _text_db_path(state) -> str:
    index_path = state.data.get("_index_path", state.index_path)
    return str(Path(index_path).parent / "full_text_cache.db")


def _get_text_conn(state):
    if state.text_conn is not None:
        return state.text_conn
    db_path = _text_db_path(state)
    if os.path.exists(db_path):
        state.text_conn = sqlite3.connect(db_path, check_same_thread=False)
        state.text_conn.execute("PRAGMA journal_mode=WAL")
    return state.text_conn


def build_text_cache(state) -> None:
    """Build SQLite cache of id→full_text in a background thread (fast after first run)."""
    index_path = state.data.get("_index_path", state.index_path)
    db_path    = _text_db_path(state)
    # Skip if cache is newer than index AND has the same story count
    if os.path.exists(db_path) and os.path.getmtime(db_path) >= os.path.getmtime(index_path):
        try:
            cache_conn   = sqlite3.connect(db_path)
            cached_count = cache_conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
            cache_conn.close()
            index_count = len([s for s in state.data.get("stories", []) if s.get("_has_text")])
            if cached_count >= index_count:
                print(f"Full-text cache up to date ({db_path})")
                return
            print(f"Full-text cache has {cached_count} stories but index has {index_count} — rebuilding…")
        except Exception:
            pass
    print("Building full-text SQLite cache (first run — takes ~1 min)…")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS texts (id TEXT PRIMARY KEY, full_text TEXT)")
    conn.execute("DELETE FROM texts")
    count = 0
    try:
        import ijson
        with open(index_path, "rb") as f:
            for s in ijson.items(f, "stories.item"):
                ft = s.get("full_text")
                if ft:
                    conn.execute("INSERT OR REPLACE INTO texts VALUES (?,?)", (s["id"], ft))
                    count += 1
    except ImportError:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("stories", []):
            ft = s.get("full_text")
            if ft:
                conn.execute("INSERT OR REPLACE INTO texts VALUES (?,?)", (s["id"], ft))
                count += 1
    conn.commit()
    conn.close()
    state.text_conn = None  # force reconnect so next request picks up the new cache
    print(f"Full-text cache built: {count} stories → {db_path}")


def get_story_text(state, story_id: str) -> str:
    """Fetch full_text for a story — O(1) via persistent SQLite connection."""
    conn = _get_text_conn(state)
    if conn is not None:
        try:
            row = conn.execute("SELECT full_text FROM texts WHERE id=?",
                               (story_id,)).fetchone()
            if row:
                return row[0]
        except Exception:
            pass
    # Fallback: stream index (slow — cache not built yet or story missing from cache)
    index_path = state.data.get("_index_path", state.index_path)
    try:
        import ijson
        with open(index_path, "rb") as f:
            for s in ijson.items(f, "stories.item"):
                if s.get("id") == story_id:
                    return s.get("full_text", "")
    except ImportError:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("stories", []):
            if s.get("id") == story_id:
                return s.get("full_text", "")
    return ""
