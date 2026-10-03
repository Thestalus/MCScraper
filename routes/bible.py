"""
routes/bible.py — /bibles, /bibles/new, /bibles/<id>/edit, /bibles/<id>/write,
                  /bibles/<id>/read, /bibles/<id>/delete,
                  /api/bible/write, /api/bible/summarise, /api/bible/save-chapter
"""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify, redirect, abort, current_app
from flask import Response, stream_with_context

from story_bible import (
    list_bibles, load_bible, save_bible, delete_bible,
    empty_bible, next_chapter_prompt, auto_summarise_prompt,
)

bible_bp = Blueprint("bible", __name__)


@bible_bp.route("/bibles")
def bibles():
    """List every story bible."""
    return render_template("bibles/list.html", bibles=list_bibles())


@bible_bp.route("/bibles/new", methods=["GET", "POST"])
def bible_new():
    """Create a story bible: show the empty form, or save it on POST."""
    if request.method == "POST":
        return _save_bible_form(None)
    return render_template("bibles/edit.html", bible=empty_bible())


@bible_bp.route("/bibles/<bible_id>")
def bible_detail(bible_id):
    """A bible's main page is its write page; redirect there."""
    return redirect(f"/bibles/{bible_id}/write")


@bible_bp.route("/bibles/<bible_id>/edit", methods=["GET", "POST"])
def bible_edit(bible_id):
    """Edit a bible's premise, setting, characters and threads; saves on POST."""
    if request.method == "POST":
        return _save_bible_form(bible_id)
    bible = load_bible(bible_id)
    if not bible:
        abort(404)
    return render_template("bibles/edit.html", bible=bible)


def _save_bible_form(bible_id):
    """Parse form data and save the bible, then redirect to the write page."""
    f     = request.form
    bible = load_bible(bible_id) if bible_id else empty_bible(f.get("title", ""))
    bible["title"]       = f.get("title", bible.get("title", ""))
    bible["premise"]     = f.get("premise", "")
    bible["setting"]     = f.get("setting", "")
    bible["tone"]        = f.get("tone", "")
    bible["style_notes"] = f.get("style_notes", "")
    bible["tags"]        = [t.strip() for t in f.get("tags", "").split() if t.strip()]
    bible["threads"]     = [t.strip() for t in f.get("threads", "").splitlines() if t.strip()]
    bible["dynamic"]     = {
        "dominant":      f.get("dynamic_dominant", ""),
        "submissive":    f.get("dynamic_submissive", ""),
        "method":        f.get("dynamic_method", ""),
        "current_state": f.get("dynamic_current_state", ""),
    }
    names  = f.getlist("char_name")
    roles  = f.getlist("char_role")
    descs  = f.getlist("char_description")
    arcs   = f.getlist("char_arc")
    states = f.getlist("char_state")
    bible["characters"] = [
        {"name": n, "role": r, "description": d, "arc": a, "current_state": s}
        for n, r, d, a, s in zip(names, roles, descs, arcs, states)
        if n.strip()
    ]
    save_bible(bible)
    return redirect(f"/bibles/{bible['id']}/write")


@bible_bp.route("/bibles/<bible_id>/delete", methods=["POST"])
def bible_delete(bible_id):
    """Delete a bible (its JSON file in output/bibles/)."""
    delete_bible(bible_id)
    return jsonify({"ok": True})


@bible_bp.route("/bibles/<bible_id>/write")
def bible_write(bible_id):
    """Writing page for the next chapter of a bible."""
    bible = load_bible(bible_id)
    if not bible:
        abort(404)
    next_chapter = len(bible.get("chapters", [])) + 1
    return render_template("bibles/write.html", bible=bible, next_chapter=next_chapter)


@bible_bp.route("/bibles/<bible_id>/read")
def bible_read(bible_id):
    """Read every generated chapter of a bible in order."""
    bible = load_bible(bible_id)
    if not bible:
        abort(404)
    generated = bible.get("generated", [])
    return render_template("bibles/read.html", bible=bible, generated=generated)


# ── Bible API endpoints ────────────────────────────────────────────────────────

@bible_bp.route("/api/bible/write", methods=["POST"])
def api_bible_write():
    """Stream the next chapter from the local LLM, prompted with the whole bible plus the
    user's instruction. Returns plain text as it is generated; 503 if the LLM server
    isn't reachable.
    """
    from generator import LLMClient
    data  = request.get_json(force=True)
    bible = load_bible(data.get("bible_id", ""))
    if not bible:
        return jsonify({"error": "Bible not found"}), 404

    url         = data.get("url", "http://localhost:1234/v1")
    model_name  = data.get("model", "auto")
    temperature = float(data.get("temperature", 0.85))
    max_tokens  = int(data.get("max_tokens", 3000))
    instruction = data.get("instruction", "")

    messages = next_chapter_prompt(bible, instruction)
    client   = LLMClient(base_url=url, api_key="none", model=model_name)

    if not client.ping():
        return jsonify({"error": "Cannot connect to LLM"}), 503

    def gen():
        try:
            for chunk in client.complete(
                messages, max_tokens=max_tokens, temperature=temperature, stream=True
            ):
                yield chunk
        except Exception as e:
            yield f"\n\n[Error: {e}]"

    return Response(
        stream_with_context(gen()),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@bible_bp.route("/api/bible/summarise", methods=["POST"])
def api_bible_summarise():
    """Ask the local LLM for a short summary of a chapter, to store in the bible's chapter
    list.
    """
    from generator import LLMClient
    data        = request.get_json(force=True)
    text        = data.get("text", "")
    chapter_num = data.get("chapter_num", 1)
    url         = data.get("url", "http://localhost:1234/v1")
    messages    = auto_summarise_prompt(text, chapter_num)
    client      = LLMClient(base_url=url, api_key="none")
    try:
        summary = client.complete(messages, max_tokens=200, temperature=0.3)
        return jsonify({"summary": summary.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bible_bp.route("/api/bible/save-chapter", methods=["POST"])
def api_bible_save_chapter():
    """Append a finished chapter to the bible: its summary to `chapters` and its full text
    to `generated`.
    """
    import datetime
    data     = request.get_json(force=True)
    bible_id = data.get("bible_id", "")
    bible    = load_bible(bible_id)
    if not bible:
        return jsonify({"ok": False, "error": "Bible not found"})

    chapter_num = data.get("chapter_num", len(bible.get("chapters", [])) + 1)
    bible.setdefault("chapters", []).append({
        "number":     chapter_num,
        "title":      data.get("title", ""),
        "summary":    data.get("summary", ""),
        "word_count": len(data.get("text", "").split()),
        "saved_at":   datetime.datetime.now().isoformat(),
    })
    bible.setdefault("generated", []).append({
        "chapter":      chapter_num,
        "title":        data.get("title", ""),
        "text":         data.get("text", ""),
        "generated_at": datetime.datetime.now().isoformat(),
    })
    save_bible(bible)
    return jsonify({"ok": True})
