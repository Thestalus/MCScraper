"""
routes/api.py — JSON API endpoints: like, want, preview, history, search, dismiss, tag-stats
"""
from __future__ import annotations
import re as _re

from flask import Blueprint, request, jsonify, current_app

from services.search_service import search_stories
from services.text_store import get_story_text
from services.data_store import get_auto_tags

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/search")
def api_search():
    """JSON API for search — useful for LLM integrations."""
    state   = current_app.state
    q       = request.args.get("q", "")
    limit   = min(int(request.args.get("limit", 20)), 100)
    results, total = search_stories(state, query=q, per_page=limit)
    return jsonify({
        "total": total,
        "stories": [
            {
                "id":          s["id"],
                "title":       s.get("title"),
                "author":      s.get("author"),
                "word_count":  s.get("word_count"),
                "categories":  s.get("categories"),
                "description": s.get("description"),
                "auto_tags":   get_auto_tags(s),
            }
            for s in results
        ],
    })


@api_bp.route("/api/like/<story_id>", methods=["POST"])
def api_like(story_id):
    """Set a story's like status from the JSON body: {"status": "liked" | "disliked" |
    anything else to clear}.
    """
    state  = current_app.state
    if not state.prefs:
        return jsonify({"ok": False, "error": "preferences not loaded"})
    data   = request.get_json(force=True)
    status = data.get("status", "liked")
    if status == "liked":
        state.prefs.like_story(story_id)
    elif status == "disliked":
        state.prefs.dislike_story(story_id)
    else:
        state.prefs.unlike_story(story_id)
    return jsonify({"ok": True, "status": status})


@api_bp.route("/api/like-author/<path:author_name>", methods=["POST"])
def api_like_author(author_name):
    """Follow, dislike or clear an author: {"status": "liked" | "disliked" | anything else
    to clear}.
    """
    state  = current_app.state
    if not state.prefs:
        return jsonify({"ok": False})
    data   = request.get_json(force=True)
    status = data.get("status", "liked")
    if status == "liked":
        state.prefs.like_author(author_name)
    elif status == "disliked":
        state.prefs.dislike_author(author_name)
    else:
        state.prefs.unlike_author(author_name)
    return jsonify({"ok": True, "status": status})


@api_bp.route("/api/dismiss-rec/<story_id>", methods=["POST"])
def api_dismiss_rec(story_id):
    """Hide a story from the 'For You' recommendations."""
    state = current_app.state
    if not state.prefs:
        return jsonify({"ok": False})
    state.prefs.hide_rec(story_id)
    return jsonify({"ok": True})


@api_bp.route("/api/restore-dismissed", methods=["POST"])
def api_restore_dismissed():
    """Un-hide every dismissed recommendation."""
    state = current_app.state
    if not state.prefs:
        return jsonify({"ok": False})
    state.prefs.set_value("hidden_recs", [])
    return jsonify({"ok": True})


@api_bp.route("/api/want/<story_id>", methods=["POST"])
def api_want(story_id):
    """Add ({"wanted": true}) or remove a story from the want-to-read list."""
    state  = current_app.state
    if not state.prefs:
        return jsonify({"ok": False})
    data   = request.get_json(force=True)
    wanted = bool(data.get("wanted", True))
    if wanted:
        state.prefs.want_story(story_id)
    else:
        state.prefs.unwant_story(story_id)
    return jsonify({"ok": True, "wanted": wanted})


@api_bp.route("/api/preview/<story_id>")
def api_preview(story_id):
    """First ~700 characters of a story as plain text, for the inline preview on story
    cards.
    """
    state = current_app.state
    s = state.data.get("by_id", {}).get(story_id)
    if not s:
        return jsonify({"text": ""})
    full_text = s.get("full_text") or get_story_text(state, story_id)
    plain = _re.sub(r"<[^>]+>", " ", full_text) if "<" in full_text else full_text
    plain = _re.sub(r"\s+", " ", plain).strip()
    preview = plain[:700].rsplit(None, 1)[0] if len(plain) > 700 else plain
    return jsonify({"text": preview + ("…" if len(plain) > 700 else "")})


@api_bp.route("/api/history/<story_id>", methods=["POST"])
def api_history(story_id):
    """Record that the user read a story up to a chapter: {"chapter": n, "total": m}."""
    state   = current_app.state
    if not state.prefs:
        return jsonify({"ok": False})
    data    = request.get_json(force=True)
    chapter = int(data.get("chapter", 1))
    total   = int(data.get("total", 1))
    state.prefs.record_visit(story_id, chapter, total)
    return jsonify({"ok": True})


