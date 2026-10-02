"""
routes/prefs.py — /liked, /liked/export, /history, /want-to-read, /new-stories, /recommendations
"""
from __future__ import annotations
import json
from pathlib import Path

from flask import Blueprint, render_template, abort, request, redirect, current_app
from flask import Response

from services.data_store import OUTPUT_DIR

prefs_bp = Blueprint("prefs", __name__)


@prefs_bp.route("/liked")
def liked():
    state = current_app.state
    if not state.prefs:
        abort(503)
    liked_ids     = state.prefs.liked_stories
    liked_stories = [state.data["by_id"][sid] for sid in liked_ids
                     if sid in state.data.get("by_id", {})]
    liked_authors = state.prefs.liked_authors
    return render_template(
        "liked.html",
        liked_stories=liked_stories,
        liked_authors=liked_authors,
    )


@prefs_bp.route("/liked/export")
def liked_export():
    state = current_app.state
    if not state.prefs:
        abort(503)
    fmt     = request.args.get("fmt", "json")
    ids     = state.prefs.liked_stories
    stories = [state.data["by_id"][sid] for sid in ids
               if sid in state.data.get("by_id", {})]

    if fmt == "csv":
        import csv
        import io
        buf = io.StringIO()
        w   = csv.writer(buf)
        w.writerow(["id", "title", "author", "word_count", "categories", "story_url"])
        for s in stories:
            w.writerow([
                s["id"], s.get("title", ""), s.get("author", ""),
                s.get("word_count", 0),
                " ".join(s.get("categories", [])),
                s.get("story_url", ""),
            ])
        return Response(
            buf.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=liked_stories.csv"},
        )
    elif fmt == "txt":
        lines = [f"{s.get('title') or s['id']} — {s.get('author', '')}" for s in stories]
        return Response(
            "\n".join(lines), mimetype="text/plain",
            headers={"Content-Disposition": "attachment;filename=liked_stories.txt"},
        )
    else:
        export = [
            {
                "id":         s["id"],
                "title":      s.get("title"),
                "author":     s.get("author"),
                "word_count": s.get("word_count"),
                "categories": s.get("categories"),
                "story_url":  s.get("story_url"),
            }
            for s in stories
        ]
        return Response(
            json.dumps(export, indent=2, ensure_ascii=False),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment;filename=liked_stories.json"},
        )


@prefs_bp.route("/history")
def history():
    state = current_app.state
    if not state.prefs:
        abort(503)
    by_id   = state.data.get("by_id", {})
    entries = []
    for sid, info in state.prefs.visited_stories[:100]:
        s = by_id.get(sid)
        if s:
            entries.append((s, info))
    return render_template("history.html", entries=entries)


@prefs_bp.route("/want-to-read")
def want_to_read():
    state = current_app.state
    if not state.prefs:
        abort(503)
    by_id   = state.data.get("by_id", {})
    stories = [by_id[sid] for sid in state.prefs.wanted_stories if sid in by_id]
    return render_template("want_to_read.html", stories=stories, prefs=state.prefs)


@prefs_bp.route("/recommendations")
def recommendations():
    state = current_app.state
    if not state.rec:
        abort(503)
    profile      = state.rec.taste_profile_summary()
    shuffle      = request.args.get("shuffle") == "1"
    hidden_count = len(state.prefs.data.get("hidden_recs", [])) if state.prefs else 0
    cached_pers  = state.recs_cache.get("personalised")
    if cached_pers:
        by_id  = state.data.get("by_id", {})
        hidden = set(state.prefs.data.get("hidden_recs", [])) if state.prefs else set()
        recs   = [(by_id[sid], score, reason) for sid, score, reason in cached_pers
                  if sid in by_id and sid not in hidden]
        if shuffle:
            import random
            random.shuffle(recs)
    else:
        if not state.engine._loaded:
            state.engine.build_index(
                state.data["stories"],
                cache_path=Path(OUTPUT_DIR) / "recs_embeddings.npz",
            )
        recs = state.rec.personalised_recommendations(n=200, shuffle=shuffle)
    return render_template(
        "recommendations.html",
        recs=recs,
        profile=profile,
        liked_count=state.prefs.total_likes if state.prefs else 0,
        index_ready=state.engine._loaded,
        semantic=state.engine.semantic,
        hidden_count=hidden_count,
    )
