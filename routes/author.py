"""
routes/author.py — /author/<name>, /authors, /author-similar/<name>
"""
from __future__ import annotations
from pathlib import Path

from flask import Blueprint, render_template, abort, current_app

from services.search_service import dt
from services.data_store import get_all_authors, OUTPUT_DIR

author_bp = Blueprint("author", __name__)


@author_bp.route("/author/<author_name>")
def author(author_name):
    """All of one author's stories, sorted by title, with the user's follow status."""
    state = current_app.state
    author_stories = state.data.get("by_author", {}).get(author_name, [])
    author_stories = [s for s in author_stories if not s.get("fetch_error")]
    author_stories.sort(key=lambda s: dt(s).lower())
    author_status = state.prefs.author_status(author_name) if state.prefs else "none"
    return render_template(
        "author.html",
        author=author_name, stories=author_stories,
        author_status=author_status, prefs=state.prefs
    )


@author_bp.route("/authors")
def authors():
    """Every author with their story count."""
    state = current_app.state
    return render_template("authors.html", authors=get_all_authors(state))


@author_bp.route("/author-similar/<path:author_name>")
def author_similar(author_name):
    """Authors whose stories are most similar to this author's (embedding-based; builds the
    index on first use).
    """
    state = current_app.state
    if not state.rec:
        abort(503)
    if not state.engine._loaded:
        state.engine.build_index(
            state.data["stories"],
            cache_path=Path(OUTPUT_DIR) / "recs_embeddings.npz"
        )
    similar_authors = state.rec.similar_authors(author_name, n=12)
    author_stories  = state.data.get("by_author", {}).get(author_name, [])
    author_status   = state.prefs.author_status(author_name) if state.prefs else "none"
    return render_template(
        "author_similar.html",
        author=author_name, similar=similar_authors,
        story_count=len(author_stories), author_status=author_status,
        index_ready=state.engine._loaded
    )
