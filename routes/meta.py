"""
routes/meta.py — /tags, /stats, /new-stories, /tag-stats
"""
from __future__ import annotations
import json
from pathlib import Path

from flask import Blueprint, render_template, current_app

from search import tag_coverage_stats
from services.data_store import get_all_tags, valid_stories, stories, OUTPUT_DIR

meta_bp = Blueprint("meta", __name__)

_AUTO_DIM_LABELS = {
    "mind_control_type": ("MC Method",       "mc_type"),
    "setting":           ("Setting",         "setting"),
    "themes":            ("Themes",          "theme"),
    "tone":              ("Tone",            "tone"),
    "dominance":         ("Dominance Style", "dominance"),
    "pacing":            ("Pacing",          "pacing"),
    "awareness":         ("Victim Awareness","awareness"),
    "sexual_acts":       ("Sexual Acts",     "sexual_act"),
    "fetish":            ("Fetish",          "fetish_tag"),
    "relationship":      ("Relationship",    "relationship"),
}


@meta_bp.route("/tags")
def tags():
    state = current_app.state
    cats, auto = get_all_tags(state)
    return render_template(
        "tags.html",
        tags=cats, auto_tags=auto, dim_meta=_AUTO_DIM_LABELS
    )


@meta_bp.route("/stats")
def stats():
    state = current_app.state
    vs = valid_stories(state)
    total_words = sum(s.get("word_count", 0) for s in vs)
    tag_counts = {}
    for s in vs:
        for c in s.get("categories", []):
            tag_counts[c] = tag_counts.get(c, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:30]
    stats_data = {
        "total":         len(vs),
        "authors":       len(set(s.get("author") for s in vs)),
        "avg_words":     total_words / max(len(vs), 1),
        "total_words_m": round(total_words / 1_000_000, 1),
        "multi_chapter": sum(1 for s in vs if s.get("chapter_count", 1) > 1),
        "errors":        sum(1 for s in stories(state) if s.get("fetch_error")),
        "top_tags":      top_tags,
    }
    return render_template("stats.html", stats=stats_data)


@meta_bp.route("/new-stories")
def new_stories():
    state = current_app.state
    update_log_path = Path(OUTPUT_DIR) / "update_log.json"
    runs = []
    new  = []

    if update_log_path.exists():
        try:
            with open(update_log_path) as f:
                log_data = json.load(f)
            runs = log_data.get("runs", [])
        except Exception:
            pass

    all_valid = [s for s in stories(state)
                 if not s.get("fetch_error") and s.get("_has_text", True)]
    by_id = state.data.get("by_id", {})

    if runs:
        last = runs[-1]
        new_ids = last.get("new_story_ids") or []
        ch_ids  = last.get("chapter_updated_ids") or []

        if new_ids or ch_ids:
            new_id_set = set(new_ids)
            ch_id_set  = set(ch_ids)
            new_stories_list    = []
            chapter_update_list = []
            for s in all_valid:
                if s["id"] in new_id_set:
                    new_stories_list.append((s, None))
                elif s["id"] in ch_id_set:
                    chapter_update_list.append((s, "New chapter"))
            new = new_stories_list + chapter_update_list
        else:
            last_added = last.get("stories_added_to_index", 0)
            if last_added > 0:
                new = [(s, None) for s in all_valid[-last_added:]]

    if not new:
        last_seen = state.prefs.data.get("last_seen_count", 0) if state.prefs else 0
        new = [(s, None) for s in (all_valid[last_seen:] if last_seen < len(all_valid) else [])]
        if state.prefs and new:
            state.prefs.set_value("last_seen_count", len(all_valid))

    scrape_date = state.data.get("last_updated", state.data.get("scraped_at", ""))[:10] if state.data else ""
    last_run    = runs[-1]["timestamp"][:10] if runs else ""

    return render_template(
        "new_stories.html",
        new_stories=new, scrape_date=scrape_date,
        last_run=last_run, update_runs=runs[-8:],
        prefs=state.prefs
    )


@meta_bp.route("/tag-stats")
def tag_stats():
    state = current_app.state
    stats = tag_coverage_stats(state.data.get("stories", []))
    return render_template("tag_stats.html", stats=stats)
