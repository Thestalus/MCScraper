"""
app_factory.py — AppState dataclass and Flask application factory.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from recommender import EmbeddingEngine


@dataclass
class AppState:
    index_path: str = "./output/stories_index.json"
    data: dict = field(default_factory=dict)
    prefs: Optional[object] = None
    rec: Optional[object] = None
    engine: object = field(default_factory=EmbeddingEngine)
    embeddings_built: bool = False
    search_conn: Optional[sqlite3.Connection] = None
    text_conn: Optional[sqlite3.Connection] = None
    recs_cache: dict = field(default_factory=dict)
    tag_profile_cache: dict = field(default_factory=dict)
    tag_profile_nlikes: int = -1


def _register_jinja_filters(app, state):
    """Register custom Jinja2 filters and globals on the app."""
    import urllib.parse as _up
    from services.search_service import slug_to_title, dt
    from services.tag_service import visible_tags
    from services.data_store import get_auto_tags

    app.jinja_env.filters["urlencode"]     = _up.quote
    app.jinja_env.filters["slug_to_title"] = slug_to_title
    app.jinja_env.filters["dt"]            = dt
    app.jinja_env.globals["visible_tags"]  = visible_tags
    app.jinja_env.globals["story_tags"]    = get_auto_tags
    app.jinja_env.globals["app_state"]     = state


def create_app(state: AppState):
    """Create and configure the Flask application."""
    from flask import Flask

    app = Flask(__name__, template_folder="templates")
    app.secret_key = "mcscraper-browser"
    app.state = state

    _register_jinja_filters(app, state)

    from routes.browse import browse_bp
    from routes.story import story_bp
    from routes.author import author_bp
    from routes.meta import meta_bp
    from routes.prefs import prefs_bp
    from routes.api import api_bp
    from routes.generate import generate_bp
    from routes.bible import bible_bp

    app.register_blueprint(browse_bp)
    app.register_blueprint(story_bp)
    app.register_blueprint(author_bp)
    app.register_blueprint(meta_bp)
    app.register_blueprint(prefs_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(generate_bp)
    app.register_blueprint(bible_bp)

    return app
