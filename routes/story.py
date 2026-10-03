"""
routes/story.py — /story/<id>, /similar/<id>
"""
from __future__ import annotations
import html as _html
import re
from pathlib import Path

from flask import Blueprint, request, render_template, abort, current_app

from services.text_store import get_story_text
from services.data_store import OUTPUT_DIR

story_bp = Blueprint("story", __name__)


@story_bp.route("/story/<story_id>")
def story(story_id):
    """Story reader. Splits the text into chapters (HTML stories on <!-- BREAK --> markers;
    plain-text stories on '* * *' lines, repairing soft line-wrap breaks and OCR double
    spaces), then renders the chapter given by ?ch= along with the user's like/want
    status and up to 4 similar stories.
    """
    state = current_app.state
    s = state.data.get("by_id", {}).get(story_id)
    if not s:
        abort(404)
    full_text = s.get("full_text") or get_story_text(state, story_id)
    is_html = full_text.lstrip().startswith("<") or "<!-- BREAK -->" in full_text
    if is_html:
        raw_chapters = re.split(r"\n<!--\s*BREAK\s*-->\n", full_text)
        chapter_chunks = [(i + 1, ch.strip()) for i, ch in enumerate(raw_chapters)]
    else:
        raw_chapters = re.split(r"\n\s*\*\s*\*\s*\*\s*\n", full_text)
        chapter_chunks = []
        for i, ch in enumerate(raw_chapters, 1):
            paras = [p.strip() for p in ch.split("\n\n") if p.strip()]
            # Merge false paragraph breaks: a chunk that ends without sentence-ending
            # punctuation and whose successor starts lowercase is a soft word-wrap
            # break, not a real paragraph boundary.
            merged: list[str] = []
            for p in paras:
                if merged and not re.search(r'[.!?"”]\s*$', merged[-1]) and re.match(r'[a-z‘“]', p):
                    merged[-1] = merged[-1] + " " + p
                else:
                    merged.append(p)
            # Collapse soft single-newline word-wrap breaks within each paragraph
            merged = [" ".join(p.split("\n")) for p in merged]
            # Collapse runs of multiple spaces (OCR artefacts)
            merged = [re.sub(r'  +', ' ', p) for p in merged]
            chapter_chunks.append((i, "\n".join(f"<p>{_html.escape(p)}</p>" for p in merged)))
    chapter_count = len(chapter_chunks)
    chapters = [(i, f"Ch. {i}") for i in range(1, chapter_count + 1)]
    story_status = state.prefs.story_status(story_id) if state.prefs else "none"
    try:
        current_ch = max(1, min(chapter_count, int(request.args.get("ch", 1))))
    except (ValueError, TypeError):
        current_ch = 1
    similar = []
    if state.rec and state.engine._loaded:
        similar = state.rec.similar_to_story(story_id, n=4)
    return render_template(
        "story.html",
        story=s, chapters=chapters, chapter_chunks=chapter_chunks,
        current_ch=current_ch, story_status=story_status,
        similar=similar, prefs=state.prefs, request=request
    )


@story_bp.route("/similar/<story_id>")
def similar(story_id):
    """Stories similar to one story: from the precalculated recs cache when available,
    otherwise computed live (building the embedding index on first use). Optional
    ?min_words= / ?max_words= filters.
    """
    state = current_app.state
    s = state.data.get("by_id", {}).get(story_id)
    if not s:
        abort(404)
    results = []
    cached = state.recs_cache.get("similar", {}).get(story_id)
    if cached:
        by_id = state.data.get("by_id", {})
        results = [(by_id[sid], score) for sid, score in cached if sid in by_id]
    elif state.rec:
        if not state.engine._loaded:
            state.engine.build_index(
                state.data["stories"],
                cache_path=Path(OUTPUT_DIR) / "recs_embeddings.npz"
            )
        results = state.rec.similar_to_story(story_id, n=24)
    try:
        max_words = int(request.args.get("max_words", 0))
    except (ValueError, TypeError):
        max_words = 0
    try:
        min_words = int(request.args.get("min_words", 0))
    except (ValueError, TypeError):
        min_words = 0
    if max_words:
        results = [(sim, sc) for sim, sc in results if sim.get("word_count", 0) <= max_words]
    if min_words:
        results = [(sim, sc) for sim, sc in results if sim.get("word_count", 0) >= min_words]
    return render_template(
        "similar.html",
        story=s, results=results,
        index_ready=bool(cached or state.engine._loaded), request=request
    )
