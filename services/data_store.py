"""
services/data_store.py — Data loading and in-memory index helpers.
All functions take a state: AppState parameter instead of using globals.
"""
from __future__ import annotations
import gc
import json
import sqlite3
import threading
from pathlib import Path

from recommender import EmbeddingEngine, PreferenceStore, RecommendationEngine

PREFS_PATH = "./output/preferences.json"
CACHE_PATH = "./output/recs_cache.json"
OUTPUT_DIR = "./output"


def _load_from_sqlite(db_path: str) -> tuple:
    """Load slim story metadata from stories_meta.db. Returns (stories_slim, top_meta)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM stories").fetchall()
    meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
    conn.close()

    top_meta = {r["key"]: r["value"] for r in meta_rows}
    stories_slim = []
    for row in rows:
        s = dict(row)
        s["categories"] = json.loads(s["categories"] or "[]")
        s["auto_tags"]  = s["auto_tags"] or "{}"   # keep as raw JSON string; parsed on first access
        s["_has_text"]  = bool(s.pop("has_text", 0))
        stories_slim.append(s)
    return stories_slim, top_meta


def get_auto_tags(s: dict) -> dict:
    """Return parsed auto_tags dict for a story, parsing lazily on first access."""
    at = s.get("auto_tags")
    if isinstance(at, str):
        parsed = json.loads(at) if at else {}
        s["auto_tags"] = parsed
        return parsed
    return at or {}


def load_recs_cache(state) -> bool:
    """Load pre-calculated recommendations cache if it exists and is fresh."""
    cache = Path(CACHE_PATH)
    if not cache.exists():
        return False
    index_mtime = Path(state.index_path).stat().st_mtime if Path(state.index_path).exists() else 0
    prefs_mtime = Path(PREFS_PATH).stat().st_mtime if Path(PREFS_PATH).exists() else 0
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("index_mtime", 0) < index_mtime:
            print("Recs cache is stale (index updated) — run: make precalc-recs")
            return False
        if data.get("prefs_mtime", 0) < prefs_mtime:
            print("Recs cache is stale (preferences updated) — run: make precalc-recs")
            return False
        state.recs_cache = data
        print(f"Loaded recs cache: {data.get('story_count', 0)} stories, "
              f"{len(data.get('personalised', []))} personalised recs")
        return True
    except Exception as e:
        print(f"Could not load recs cache: {e}")
        return False


def _open_search_db(state) -> None:
    """Open output/search.db if it exists (and is not stale). Called from load_data()."""
    import os
    index_path = state.data.get("_index_path", state.index_path)
    db_path    = str(Path(index_path).parent / "search.db")
    if not os.path.exists(db_path):
        return
    # Stale check — if index is newer, the DB needs a rebuild
    if os.path.getmtime(db_path) < os.path.getmtime(index_path):
        print("search.db is stale — run: make build-search-db")
        print("  (falling back to in-memory search until rebuilt)")
        return
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=1")
        count = conn.execute("SELECT COUNT(*) FROM stories").fetchone()[0]
        state.search_conn = conn
        print(f"FTS search DB ready ({count:,} stories) — full-text search enabled")
        # Warm up the FTS index in the background so the first user search is fast.
        def _warmup():
            try:
                conn.execute(
                    "SELECT story_id FROM stories_fts WHERE stories_fts MATCH '\"the\"' LIMIT 1"
                ).fetchall()
                print("FTS index warm")
            except Exception:
                pass
        threading.Thread(target=_warmup, daemon=True).start()
    except Exception as e:
        print(f"Could not open search.db: {e} — falling back to in-memory search")


def load_data(state) -> None:
    """Load story data into state, init preference/recommendation engines, start background tasks."""
    from services.text_store import build_text_cache

    p         = Path(state.index_path)
    db_path   = str(Path(state.index_path).parent / "stories_meta.db")
    use_sqlite = Path(db_path).exists()

    if not use_sqlite and not p.exists():
        state.data = {"stories": [], "total_stories": 0}
        return

    if use_sqlite:
        import time
        t0 = time.time()
        stories_slim, top_meta = _load_from_sqlite(db_path)
        print(f"Loaded {len(stories_slim):,} stories from SQLite in {time.time()-t0:.1f}s")
        state.data = top_meta
        state.data["_index_path"] = str(p)
    else:
        print("stories_meta.db not found — loading from JSON (run make build-stories-db to speed this up)")
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        stories_slim = []
        for s in raw.get("stories", []):
            slim = {k: v for k, v in s.items() if k != "full_text"}
            slim["_has_text"] = bool(s.get("full_text"))
            stories_slim.append(slim)
        state.data = {k: v for k, v in raw.items() if k != "stories"}
        state.data["_index_path"] = str(p)
        print(f"Loaded {len(stories_slim):,} stories from JSON (slim)")

    state.data["stories"]   = stories_slim
    state.data["by_id"]     = {s["id"]: s for s in stories_slim}
    state.data["by_author"] = {}
    for s in stories_slim:
        a = s.get("author", "Unknown")
        state.data["by_author"].setdefault(a, []).append(s)

    gc.collect()

    # Init preference store and recommendation engine
    state.prefs = PreferenceStore(Path(PREFS_PATH))
    state.rec   = RecommendationEngine(state.engine, state.prefs, stories_slim)

    # Build full-text SQLite cache in background (instant on subsequent runs)
    threading.Thread(target=build_text_cache, args=(state,), daemon=True).start()

    # Open FTS search DB (synchronous — just opens the file, very fast)
    _open_search_db(state)

    # Try loading pre-calculated cache; only build embeddings if cache is missing/stale
    cache_loaded = load_recs_cache(state)
    if not cache_loaded:
        _emb_cache = Path(OUTPUT_DIR) / "recs_embeddings.npz"
        threading.Thread(
            target=lambda: state.engine.build_index(stories_slim, cache_path=_emb_cache),
            daemon=True
        ).start()


def stories(state) -> list:
    return state.data.get("stories", [])


def valid_stories(state) -> list:
    """Return stories that have text and no fetch errors."""
    return [s for s in stories(state)
            if not s.get("fetch_error") and s.get("_has_text", s.get("full_text"))]


def get_all_tags(state) -> tuple:
    """Return original mcstories category codes and auto_tag dimensions with counts."""
    cats = {}
    auto = {dim: {} for dim in
            ["mind_control_type", "setting", "themes", "tone",
             "dominance", "pacing", "awareness",
             "sexual_acts", "fetish", "relationship"]}
    for s in valid_stories(state):
        for c in s.get("categories", []):
            cats[c] = cats.get(c, 0) + 1
        at = get_auto_tags(s)
        for dim in auto:
            for val in (at.get(dim) or []):
                auto[dim][val] = auto[dim].get(val, 0) + 1
    return (
        sorted(cats.items(), key=lambda x: -x[1]),
        {dim: sorted(v.items(), key=lambda x: -x[1])
         for dim, v in auto.items() if v}
    )


def get_all_authors(state) -> list:
    """Return sorted list of (author, story_count) tuples."""
    counts = {}
    for s in valid_stories(state):
        a = s.get("author") or "Unknown"
        counts[a] = counts.get(a, 0) + 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))
