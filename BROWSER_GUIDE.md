# MC Archive Browser — Developer Guide

The browser is a local Flask web application that lets you search, filter, read, and manage your archived MC stories. This guide covers how to run it, how the code is organised, and where to find things.

---

## Running the browser

```bash
# Standard launch (reads output/stories_index.json on port 5000)
make browse

# Custom index or port
.venv/bin/python3 browser.py --index ./output/stories_index.json --port 5003

# All options
python3 browser.py --index PATH --port PORT --host HOST
```

Then open **http://localhost:5000** in your browser.

---

## File layout

```
MCScraper/
├── browser.py              # Entry point (~40 lines): parse args, load data, start Flask
├── app_factory.py          # AppState dataclass + create_app() + Jinja filter setup
│
├── routes/                 # URL route handlers (one file per feature area)
│   ├── __init__.py
│   ├── browse.py           # /, /search, /tag/<code>, /random
│   ├── story.py            # /story/<id>, /similar/<id>
│   ├── author.py           # /author/<name>, /authors, /author-similar/<name>
│   ├── meta.py             # /tags, /stats, /tag-stats
│   ├── prefs.py            # /liked, /history, /want-to-read, /recommendations, /new-stories
│   ├── api.py              # /api/like, /api/want, /api/search, /api/preview, etc.
│   ├── generate.py         # /generate, /api/generate/stream, /api/llm/models
│   └── bible.py            # /bibles and all bible sub-routes
│
├── services/               # Business logic (no Flask, no globals)
│   ├── __init__.py
│   ├── data_store.py       # load_data(), story lookups, get_all_tags(), get_all_authors()
│   ├── text_store.py       # SQLite text cache, get_story_text(), slug_to_title(), dt()
│   ├── search_service.py   # search_stories(), filter scoring, FTS helpers
│   ├── tag_service.py      # visible_tags(), LOW_QUALITY_SETFIT_TAGS, _AUTO_DIM_LABELS
│   └── search.py           # NLQ parser, pref_boost_score, tag_coverage_stats
│
└── templates/              # Jinja2 HTML templates
    ├── base.html           # Shared layout: CSS, nav, header block
    ├── index.html          # Browse/search page with filter sidebar
    ├── story.html          # Story reader with chapter nav and similar stories
    ├── author.html         # Author page with story list
    ├── authors.html        # Full author directory
    ├── tags.html           # Category/tag directory
    ├── stats.html          # Collection statistics
    ├── similar.html        # Similar stories page
    ├── recommendations.html # Personalised recommendations
    ├── author_similar.html # Authors similar to a given author
    ├── history.html        # Reading history
    ├── liked.html          # Liked stories
    ├── want_to_read.html   # Want-to-read list
    ├── new_stories.html    # Recently added stories
    ├── generate.html       # LLM story generator
    ├── tag_stats.html      # Auto-tag coverage dashboard
    └── bibles/
        ├── list.html       # Story bibles index
        ├── edit.html       # Bible metadata editor
        ├── write.html      # Chapter writer
        └── read.html       # Bible reader
```

---

## How it all fits together

### Startup sequence

1. `browser.py` parses CLI args and creates an `AppState` object
2. `load_data(state)` (in `services/data_store.py`) fills the state with stories, opens databases, starts background threads
3. `create_app(state)` (in `app_factory.py`) creates the Flask app, registers Jinja filters, and registers all Blueprint route modules
4. `app.run()` starts the server

### AppState

All shared application state lives in a single `AppState` dataclass (defined in `app_factory.py`). Routes access it via `current_app.state`.

```python
@dataclass
class AppState:
    index_path: str          # Path to stories_index.json
    data: dict               # {"stories": [...], "by_id": {...}, "by_author": {...}}
    prefs: PreferenceStore   # Likes, dismissals, want-to-read
    rec: RecommendationEngine
    engine: EmbeddingEngine  # Embedding vectors for similarity search
    embeddings_built: bool
    search_conn: sqlite3.Connection   # output/search.db (FTS search)
    text_conn: sqlite3.Connection     # output/text_cache.db (story full text)
    recs_cache: dict          # Pre-calculated recommendation cache
    tag_profile_cache: dict
    tag_profile_nlikes: int
```

### How routes access state

```python
# In any route file:
from flask import current_app

@my_bp.route("/example")
def example():
    state = current_app.state
    stories = state.data["stories"]
    ...
```

### Templates

Templates use standard Jinja2. All templates inherit from `base.html` which provides the shared CSS, navigation, and page structure. Child templates use `{% extends "base.html" %}` and fill in a `{% block content %}` block.

Flask's `render_template()` is used throughout — templates live in the `templates/` folder and Flask finds them automatically.

---

## Data flow for a search request

```
User request: GET /?q=hypnosis&tone=dark

routes/browse.py  →  _index_view()
    reads request.args
    calls parse_natural_query(q)         # services/search.py
    calls search_stories(state, ...)     # services/search_service.py
        → _fts_candidate_ids()           # FTS SQLite lookup
        → filter_score() per story       # tag/metadata filtering
        → relevance_score()              # ranking
    render_template("index.html", ...)
```

---

## Databases

| File | Purpose | Built by |
|------|---------|---------|
| `output/stories_index.json` | Full story metadata + text (legacy) | `scraper.py` / `ingest.py` |
| `output/stories_meta.db` | Slim story metadata in SQLite (fast load) | `make build-stories-db` |
| `output/search.db` | FTS full-text search index | `make build-search-db` |
| `output/text_cache.db` | Story full text in SQLite | built automatically at startup |
| `output/preferences.json` | Likes, dismissals, want-to-read | written by the browser |
| `output/recs_cache.json` | Pre-calculated recommendations | `make precalc-recs` |
| `output/recs_embeddings.npz` | Story embedding vectors | built at startup (background) |

---

## Key services

### `services/data_store.py`

- `load_data(state)` — main startup function; loads stories, opens DBs, starts background threads
- `valid_stories(state)` — returns stories that have text and no fetch errors
- `get_all_tags(state)` — returns tag counts for the tag directory page
- `get_all_authors(state)` — returns author story counts

### `services/search_service.py`

- `search_stories(state, query, ...)` — main search function; combines FTS lookup with in-memory filtering and ranking
- `filter_score(story, ...)` — 0/1 whether a story passes all active filters
- `relevance_score(story, q)` — float score for ranking by relevance

### `services/text_store.py`

- `get_story_text(state, story_id)` — retrieve full story HTML from text cache
- `build_text_cache(state)` — background thread that builds the SQLite text cache from JSON
- `slug_to_title(slug)` — convert a URL slug to a human-readable title
- `dt(ts)` — format a Unix timestamp as a date string

### `services/search.py`

- `parse_natural_query(q)` — scan a free-text query for tag phrases (e.g. "hypnosis dark romance" → mc_type=hypnosis, tone=dark/disturbing, theme=romance)
- `pref_boost_score(story, tag_profile)` — score a story against a user's tag preference profile
- `tag_coverage_stats(stories)` — compute per-dimension tag coverage for the tag stats dashboard

### `services/tag_service.py`

- `visible_tags(story)` — return the auto_tags dict with low-quality SetFit-only tags removed
- `LOW_QUALITY_SETFIT_TAGS` — frozenset of tag values that are hidden from display

---

## Routes reference

| Route | File | Description |
|-------|------|-------------|
| `GET /` | `routes/browse.py` | Browse/search page |
| `GET /search` | `routes/browse.py` | Same as `/` (alias) |
| `GET /story/<id>` | `routes/story.py` | Story reader |
| `GET /similar/<id>` | `routes/story.py` | Similar stories |
| `GET /author/<name>` | `routes/author.py` | Author page |
| `GET /authors` | `routes/author.py` | Full author directory |
| `GET /author-similar/<name>` | `routes/author.py` | Similar authors |
| `GET /tag/<code>` | `routes/browse.py` | Stories in a category |
| `GET /tags` | `routes/meta.py` | Tag/category directory |
| `GET /stats` | `routes/meta.py` | Collection statistics |
| `GET /tag-stats` | `routes/meta.py` | Auto-tag coverage dashboard |
| `GET /random` | `routes/browse.py` | Random story redirect |
| `GET /liked` | `routes/prefs.py` | Liked stories |
| `GET /liked/export` | `routes/prefs.py` | Export liked list as JSON |
| `GET /history` | `routes/prefs.py` | Reading history |
| `GET /want-to-read` | `routes/prefs.py` | Want-to-read list |
| `GET /recommendations` | `routes/prefs.py` | Personalised recommendations |
| `GET /new-stories` | `routes/prefs.py` | Recently added stories |
| `GET /generate` | `routes/generate.py` | LLM story generator |
| `GET /bibles` | `routes/bible.py` | Story bibles index |
| `GET /bibles/<id>/edit` | `routes/bible.py` | Edit bible metadata |
| `GET /bibles/<id>/write` | `routes/bible.py` | Write bible chapter |
| `GET /bibles/<id>/read` | `routes/bible.py` | Read bible |
| `POST /api/like/<id>` | `routes/api.py` | Toggle like |
| `POST /api/like-author/<name>` | `routes/api.py` | Toggle author like |
| `POST /api/dismiss-rec/<id>` | `routes/api.py` | Dismiss recommendation |
| `POST /api/want/<id>` | `routes/api.py` | Toggle want-to-read |
| `GET /api/preview/<id>` | `routes/api.py` | Story preview snippet |
| `POST /api/history/<id>` | `routes/api.py` | Record reading history |
| `GET /api/search` | `routes/api.py` | JSON search endpoint |
| `GET /api/llm/models` | `routes/generate.py` | Available LLM models |
| `POST /api/generate/retrieve` | `routes/generate.py` | Retrieve source stories for generation |
| `GET /api/generate/stream` | `routes/generate.py` | Streaming LLM generation |
| `POST /api/bible/write` | `routes/bible.py` | Generate bible chapter |
| `POST /api/bible/summarise` | `routes/bible.py` | Summarise chapter |
| `POST /api/bible/save-chapter` | `routes/bible.py` | Save chapter |

---

## Making changes

### Adding a new page

1. Add a route function in the appropriate `routes/*.py` file
2. Create a template in `templates/` (use `{% extends "base.html" %}`)
3. No changes needed to `browser.py` or `app_factory.py`

### Adding a new filter dimension

1. Add values to the relevant `*_RULES` list in `tagger.py`
2. Add the filter to `search_stories()` in `services/search_service.py`
3. Add the filter control to `templates/index.html`
4. Add the filter parameter to the `_index_view()` function in `routes/browse.py`

### Modifying the recommendation engine

See `recommender.py` — it is imported by `services/data_store.py` and is independent of the browser.

### Modifying the tagger

See `tagger.py` and `CLASSIFIER_GUIDE.md` — the tagger runs separately from the browser.

---

## Background threads

The browser starts two background threads at startup:

1. **Text cache builder** (`services/text_store.py:build_text_cache`) — reads full text from JSON and inserts it into `output/text_cache.db`. Fast on subsequent runs (skips already-cached stories).

2. **Embedding builder** (`recommender.py:EmbeddingEngine.build_index`) — builds FAISS embedding vectors for similarity search. Only runs if `output/recs_cache.json` is missing or stale. Can take a few minutes for large collections.

Both are daemon threads and will not block server shutdown.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Full-text search not working | Run `make build-search-db` |
| Story text missing | Wait for background text cache to finish, or run `python3 refresh_text.py` |
| Recommendations slow | Run `make precalc-recs` to pre-calculate |
| Recommendations stale | Run `make precalc-recs` after adding new likes |
| `stories_meta.db` missing | Run `make build-stories-db` for faster startup |

---

## Claude Slash Commands

Two Claude Code skills are available in `.claude/commands/` for common development tasks. Run them from inside the MCScraper directory.

### `/mc-debug [description]`

Runs a full health check on the browser and scraper. Checks in order:

1. Whether the browser process is running on port 5000
2. Python syntax errors in `app_factory.py` and all `routes/` files
3. Whether `index.json` is valid and non-empty, and shows the last 3 `update_log.json` entries
4. Chapter count sanity — queries `full_text_cache.db` and counts `<!-- BREAK -->` markers; flags stale cache entries showing 1 chapter when more are expected
5. Template entity escaping — greps for `&#XXXX;` expressions missing `| safe` that would render as raw text
6. Route registration — imports and boots the app, counts registered routes (expect 30+), surfaces any blueprint import errors
7. Whether `recs_cache.json` exists

Pass a description of the symptom to focus the check, or leave it blank for a full sweep. Ends with a bullet-point summary of what is healthy and exact fix commands for anything broken.

**Example:**
```
/mc-debug author pages returning 500
```

### `/mc-classify [story id or title]`

Reviews and optionally updates the `auto_tags` for a story.

1. Reads `CLASSIFIER_GUIDE.md` for the full tag schema
2. Finds the story in `index.json` by ID, title, or author (lists matches if ambiguous)
3. Shows the current `auto_tags` and `categories`
4. Samples the story's full text from `full_text_cache.db`
5. Produces a table of suggested tag changes with reasons
6. Asks for confirmation before writing changes back to `index.json`
7. Reminds you to restart the browser for changes to take effect

**Example:**
```
/mc-classify willing canvas
```
