#!/usr/bin/env python3
"""
fp_audit.py — Local Flask UI for auditing candidate false positives in story tags.

Default data paths:
    input:  output/stories_index.json
    output: output/fp_audit_results.jsonl

Run:
    python fp_audit.py

Then open:
    http://127.0.0.1:5010

Features:
- Pick a tag (for example: mind_control_type:possession)
- Restrict to stories where the tag is in auto_tags but absent from Svengali
- Optional source filter (regex / setfit / semantic / svengali / etc.)
- Optional "source only" mode (the tag came ONLY from that source)
- Optional shuffle / random sample
- Keyboard shortcuts:
    1 = TRUE_POSITIVE
    2 = FALSE_POSITIVE
    3 = AMBIGUOUS
    4 = SKIP
- Immediate JSONL persistence
- Resume support (already reviewed tag+story pairs are skipped)
- Export JSONL or CSV

Notes:
- This script expects auto_tags to look like:
      {
        "mind_control_type": ["possession", ...],
        ...
        "_tag_sources": {
            "possession": ["setfit", "regex"],
            ...
        },
        "_svengali_tags": [...]
      }
- It tolerates variants like `svengali_tags` at the story top level or inside auto_tags.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for

DEFAULT_INDEX = Path("output/stories_index.json")
DEFAULT_RESULTS = Path("output/fp_audit_results.jsonl")
DEFAULT_PORT = 5010

TEXT_FIELDS = ("full_text", "text", "story_text")
SV_RAW_FIELDS = ("_svengali_tags", "svengali_tags")

app = Flask(__name__)

STATE: dict[str, Any] = {
    "index_path": DEFAULT_INDEX,
    "results_path": DEFAULT_RESULTS,
    "stories": [],
    "stories_by_id": {},
    "results": [],
    "reviewed_keys": set(),
    "tag_counts": Counter(),
    "tag_vocab": [],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def get_auto_tags(story: dict) -> dict:
    at = story.get("auto_tags")
    return at if isinstance(at, dict) else {}


def get_story_text(story: dict) -> str:
    for key in TEXT_FIELDS:
        val = story.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def get_sv_raw(story: dict) -> list[str]:
    at = get_auto_tags(story)
    for key in SV_RAW_FIELDS:
        val = at.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, str)]
    for key in SV_RAW_FIELDS:
        val = story.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, str)]
    return []


def get_dim_values(story: dict, dim: str) -> list[str]:
    at = get_auto_tags(story)
    vals = at.get(dim, [])
    if isinstance(vals, list):
        return [v for v in vals if isinstance(v, str)]
    return []


def get_tag_sources(story: dict, tag: str) -> list[str]:
    at = get_auto_tags(story)
    src_map = at.get("_tag_sources") or story.get("_tag_sources") or {}
    if not isinstance(src_map, dict):
        return []
    vals = src_map.get(tag, [])
    if isinstance(vals, list):
        return [v for v in vals if isinstance(v, str)]
    return []


def story_has_tag(story: dict, dim: str, tag: str) -> bool:
    return tag in set(get_dim_values(story, dim))


def story_has_sv_raw_tag(story: dict, tag: str) -> bool:
    return tag in set(get_sv_raw(story))


def make_tag_key(dim: str, tag: str) -> str:
    return f"{dim}:{tag}"


def split_tag_key(tag_key: str) -> tuple[str, str]:
    if ":" not in tag_key:
        raise ValueError(f"Tag key must look like dimension:tag, got {tag_key!r}")
    dim, tag = tag_key.split(":", 1)
    return dim.strip(), tag.strip()


def build_tag_vocab(stories: list[dict]) -> tuple[list[str], Counter]:
    counts = Counter()
    for story in stories:
        at = get_auto_tags(story)
        for dim, vals in at.items():
            if dim.startswith("_") or not isinstance(vals, list):
                continue
            for tag in vals:
                if isinstance(tag, str):
                    counts[make_tag_key(dim, tag)] += 1
    tags = sorted(counts.keys(), key=lambda x: (x.split(":", 1)[0], x.split(":", 1)[1]))
    return tags, counts


def load_state(index_path: Path, results_path: Path) -> None:
    data = load_json(index_path)
    stories = data.get("stories", data) if isinstance(data, dict) else data
    if not isinstance(stories, list):
        raise RuntimeError("Expected stories_index.json to contain a top-level 'stories' list.")
    results = load_jsonl(results_path)
    reviewed_keys = {
        (row.get("story_id"), row.get("tag"))
        for row in results
        if row.get("story_id") and row.get("tag")
    }
    tag_vocab, tag_counts = build_tag_vocab(stories)

    STATE["index_path"] = index_path
    STATE["results_path"] = results_path
    STATE["stories"] = stories
    STATE["stories_by_id"] = {s.get("id"): s for s in stories if s.get("id")}
    STATE["results"] = results
    STATE["reviewed_keys"] = reviewed_keys
    STATE["tag_vocab"] = tag_vocab
    STATE["tag_counts"] = tag_counts


def candidate_rows(tag_key: str, source: str = "", source_mode: str = "contains", shuffle: bool = False, sample: int = 0) -> list[dict]:
    dim, tag = split_tag_key(tag_key)
    rows = []
    for story in STATE["stories"]:
        story_id = story.get("id")
        if not story_id:
            continue
        if (story_id, tag_key) in STATE["reviewed_keys"]:
            continue
        if not story_has_tag(story, dim, tag):
            continue
        if story_has_sv_raw_tag(story, tag):
            continue

        sources = get_tag_sources(story, tag)
        if source_mode == "contains" and source:
            if source not in sources:
                continue
        elif source_mode == "only" and source:
            if set(sources) != {source}:
                continue

        rows.append(
            {
                "story_id": story_id,
                "title": story.get("title") or story_id,
                "author": story.get("author") or "",
                "tag": tag,
                "tag_key": tag_key,
                "dimension": dim,
                "sources": sources,
                "svengali_tags": get_sv_raw(story),
                "story": story,
            }
        )

    if shuffle:
        rng = random.Random(42)
        rng.shuffle(rows)

    if sample and sample > 0:
        rows = rows[:sample]

    return rows


def summarize_candidate_counts(tag_key: str) -> dict:
    out = {"total": 0, "by_source": Counter(), "source_only": Counter()}
    dim, tag = split_tag_key(tag_key)
    for story in STATE["stories"]:
        if not story_has_tag(story, dim, tag):
            continue
        if story_has_sv_raw_tag(story, tag):
            continue
        out["total"] += 1
        sources = get_tag_sources(story, tag)
        for src in sources:
            out["by_source"][src] += 1
        if len(sources) == 1:
            out["source_only"][sources[0]] += 1
    out["reviewed"] = sum(1 for row in STATE["results"] if row.get("tag") == tag_key)
    out["remaining"] = max(0, out["total"] - out["reviewed"])
    return out


def reviewed_breakdown(tag_key: str) -> Counter:
    counts = Counter()
    for row in STATE["results"]:
        if row.get("tag") == tag_key:
            counts[row.get("label", "UNKNOWN")] += 1
    return counts


def text_snippet(text: str, chars: int = 2500) -> str:
    if not text:
        return ""
    if len(text) <= chars:
        return text
    return text[:chars].rstrip() + "\n\n[...truncated...]"


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\">
  <title>FP Audit</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 20px; line-height: 1.35; }
    .wrap { max-width: 1200px; margin: 0 auto; }
    .toolbar, .card, .stats, .help { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }
    .toolbar { background: #fafafa; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { display: block; font-size: 13px; color: #333; margin-bottom: 4px; }
    select, input[type=text], input[type=number] { padding: 8px; min-width: 220px; }
    button, .btn { padding: 10px 14px; border: 1px solid #bbb; background: white; border-radius: 8px; cursor: pointer; text-decoration: none; color: inherit; display: inline-block; }
    button.primary { background: #111; color: white; border-color: #111; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .meta { color: #444; font-size: 14px; }
    .pill { display: inline-block; border-radius: 999px; padding: 4px 10px; background: #f0f0f0; margin-right: 6px; margin-bottom: 6px; font-size: 13px; }
    .textbox { white-space: pre-wrap; background: #fcfcfc; border: 1px solid #ddd; border-radius: 8px; padding: 12px; max-height: 70vh; overflow: auto; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    .actions form { display: inline; }
    .btn-1 { background: #e8fff0; }
    .btn-2 { background: #fff0f0; }
    .btn-3 { background: #fff8e6; }
    .btn-4 { background: #f4f4f4; }
    .small { font-size: 12px; color: #666; }
    .muted { color: #666; }
    .empty { padding: 28px; text-align: center; color: #555; }
    .mono { font-family: ui-monospace, Menlo, Consolas, monospace; }
    .toplinks { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  </style>
</head>
<body>
<div class=\"wrap\">
  <h1>False-positive audit</h1>

  <div class=\"toolbar\">
    <form method=\"get\" action=\"/\">
      <div class=\"row\">
        <div>
          <label>Tag</label>
          <select name=\"tag\" required>
            {% for tk in tag_vocab %}
              <option value=\"{{ tk }}\" {% if tk == selected_tag %}selected{% endif %}>{{ tk }} ({{ tag_counts.get(tk, 0) }})</option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Source filter</label>
          <input type=\"text\" name=\"source\" value=\"{{ source }}\" placeholder=\"setfit / regex / semantic / svengali\">
        </div>
        <div>
          <label>Source mode</label>
          <select name=\"source_mode\">
            <option value=\"any\" {% if source_mode == 'any' %}selected{% endif %}>Any source</option>
            <option value=\"contains\" {% if source_mode == 'contains' %}selected{% endif %}>Contains source</option>
            <option value=\"only\" {% if source_mode == 'only' %}selected{% endif %}>Source only</option>
          </select>
        </div>
        <div>
          <label>Sample size</label>
          <input type=\"number\" name=\"sample\" min=\"0\" value=\"{{ sample }}\">
        </div>
        <div>
          <label><input type=\"checkbox\" name=\"shuffle\" value=\"1\" {% if shuffle %}checked{% endif %}> Shuffle</label>
        </div>
        <div>
          <button class=\"primary\" type=\"submit\">Apply filters</button>
        </div>
      </div>
      <div class=\"toplinks\">
        <a class=\"btn\" href=\"{{ export_json_url }}\">Export JSONL</a>
        <a class=\"btn\" href=\"{{ export_csv_url }}\">Export CSV</a>
        <a class=\"btn\" href=\"/stats?tag={{ selected_tag }}\">Tag stats</a>
      </div>
    </form>
  </div>

  <div class=\"grid\">
    <div class=\"stats\">
      <h3 style=\"margin-top:0;\">Progress</h3>
      <div><strong>Candidate pool:</strong> {{ progress.total }}</div>
      <div><strong>Reviewed for this tag:</strong> {{ progress.reviewed }}</div>
      <div><strong>Remaining:</strong> {{ progress.remaining }}</div>
      <div class=\"small\" style=\"margin-top:8px;\">
        TP: {{ review_counts.get('TRUE_POSITIVE', 0) }} ·
        FP: {{ review_counts.get('FALSE_POSITIVE', 0) }} ·
        AMBIG: {{ review_counts.get('AMBIGUOUS', 0) }} ·
        SKIP: {{ review_counts.get('SKIP', 0) }}
      </div>
    </div>
    <div class=\"help\">
      <h3 style=\"margin-top:0;\">Keyboard</h3>
      <div><strong>1</strong> = TRUE_POSITIVE</div>
      <div><strong>2</strong> = FALSE_POSITIVE</div>
      <div><strong>3</strong> = AMBIGUOUS</div>
      <div><strong>4</strong> = SKIP</div>
      <div class=\"small\" style=\"margin-top:8px;\">This page auto-skips already-reviewed story/tag pairs.</div>
    </div>
  </div>

  {% if row %}
  <div class=\"card\">
    <h2 style=\"margin-top:0;\">{{ row.title }}</h2>
    <div class=\"meta\"><strong>Author:</strong> {{ row.author or "Unknown" }}</div>
    <div class=\"meta\"><strong>ID:</strong> <span class=\"mono\">{{ row.story_id }}</span></div>
    <div class=\"meta\"><strong>Tag:</strong> <span class=\"mono\">{{ row.tag_key }}</span></div>
    <div class=\"meta\"><strong>Sources:</strong>
      {% if row.sources %}
        {% for src in row.sources %}<span class=\"pill\">{{ src }}</span>{% endfor %}
      {% else %}
        <span class=\"muted\">none recorded</span>
      {% endif %}
    </div>
    <div class=\"meta\"><strong>Svengali raw tags:</strong>
      {% if row.svengali_tags %}
        {% for t in row.svengali_tags[:20] %}<span class=\"pill\">{{ t }}</span>{% endfor %}
      {% else %}
        <span class=\"muted\">none</span>
      {% endif %}
    </div>
    <div class=\"meta\" style=\"margin-top:8px;\">
      <a class=\"btn\" href=\"{{ url_for('story_json', story_id=row.story_id) }}\" target=\"_blank\">View raw story JSON</a>
    </div>
  </div>

  <div class=\"card\">
    <div class=\"actions\">
      {% for label, key, cls in [
          ('TRUE_POSITIVE','1','btn-1'),
          ('FALSE_POSITIVE','2','btn-2'),
          ('AMBIGUOUS','3','btn-3'),
          ('SKIP','4','btn-4')
      ] %}
      <form method=\"post\" action=\"/review\">
        <input type=\"hidden\" name=\"story_id\" value=\"{{ row.story_id }}\">
        <input type=\"hidden\" name=\"tag\" value=\"{{ row.tag_key }}\">
        <input type=\"hidden\" name=\"label\" value=\"{{ label }}\">
        <input type=\"hidden\" name=\"source\" value=\"{{ source }}\">
        <input type=\"hidden\" name=\"source_mode\" value=\"{{ source_mode }}\">
        <input type=\"hidden\" name=\"shuffle\" value=\"{{ 1 if shuffle else 0 }}\">
        <input type=\"hidden\" name=\"sample\" value=\"{{ sample }}\">
        <button class=\"{{ cls }}\" type=\"submit\">{{ key }} — {{ label }}</button>
      </form>
      {% endfor %}
    </div>
  </div>

  <div class=\"card\">
    <h3 style=\"margin-top:0;\">Story text</h3>
    <details>
      <summary>Short view</summary>
      <div class=\"textbox\">{{ snippet }}</div>
    </details>
    <details>
      <summary>Full text</summary>
      <div class=\"textbox\">{{ full_text }}</div>
    </details>
  </div>
  {% else %}
  <div class=\"card empty\">
    <h2>No stories left for the current filter.</h2>
    <p>Pick another tag or relax the source filter.</p>
  </div>
  {% endif %}
</div>

<script>
(function () {
  const mapping = {
    "1": "TRUE_POSITIVE",
    "2": "FALSE_POSITIVE",
    "3": "AMBIGUOUS",
    "4": "SKIP"
  };
  document.addEventListener("keydown", function (e) {
    if (!mapping[e.key]) return;
    const input = document.querySelector('input[name="label"][value="' + mapping[e.key] + '"]');
    if (input && input.form) {
      e.preventDefault();
      input.form.submit();
    }
  });
})();
</script>
</body>
</html>
"""


def current_filters_from_request(req) -> dict:
    tag = req.values.get("tag") or (STATE["tag_vocab"][0] if STATE["tag_vocab"] else "")
    source = (req.values.get("source") or "").strip()
    source_mode = (req.values.get("source_mode") or "any").strip().lower()
    if source_mode not in {"any", "contains", "only"}:
        source_mode = "any"
    shuffle = str(req.values.get("shuffle") or "") in {"1", "true", "on", "yes"}
    try:
        sample = int(req.values.get("sample") or 0)
    except ValueError:
        sample = 0
    return {"tag": tag, "source": source, "source_mode": source_mode, "shuffle": shuffle, "sample": sample}


@app.route("/")
def index():
    if not STATE["stories"]:
        return "No stories loaded.", 500

    filters = current_filters_from_request(request)
    selected_tag = filters["tag"]
    rows = candidate_rows(selected_tag, source=filters["source"], source_mode=filters["source_mode"], shuffle=filters["shuffle"], sample=filters["sample"])
    row = rows[0] if rows else None
    progress = summarize_candidate_counts(selected_tag)
    review_counts = reviewed_breakdown(selected_tag)

    export_json_url = url_for("export_results", fmt="jsonl", tag=selected_tag, source=filters["source"], source_mode=filters["source_mode"])
    export_csv_url = url_for("export_results", fmt="csv", tag=selected_tag, source=filters["source"], source_mode=filters["source_mode"])

    full_text = get_story_text(row["story"]) if row else ""
    snippet = text_snippet(full_text)

    return render_template_string(
        PAGE_TEMPLATE,
        tag_vocab=STATE["tag_vocab"],
        tag_counts=STATE["tag_counts"],
        selected_tag=selected_tag,
        source=filters["source"],
        source_mode=filters["source_mode"],
        shuffle=filters["shuffle"],
        sample=filters["sample"],
        row=row,
        progress=progress,
        review_counts=review_counts,
        snippet=snippet,
        full_text=full_text,
        export_json_url=export_json_url,
        export_csv_url=export_csv_url,
    )


@app.route("/review", methods=["POST"])
def review():
    filters = current_filters_from_request(request)
    story_id = request.form.get("story_id", "").strip()
    tag_key = request.form.get("tag", "").strip()
    label = request.form.get("label", "").strip()

    if not story_id or not tag_key or label not in {"TRUE_POSITIVE", "FALSE_POSITIVE", "AMBIGUOUS", "SKIP"}:
        return "Bad review payload", 400

    story = STATE["stories_by_id"].get(story_id)
    if not story:
        return "Unknown story_id", 404

    dim, tag = split_tag_key(tag_key)
    row = {
        "reviewed_at": now_iso(),
        "story_id": story_id,
        "title": story.get("title"),
        "author": story.get("author"),
        "tag": tag_key,
        "dimension": dim,
        "tag_value": tag,
        "label": label,
        "sources": get_tag_sources(story, tag),
        "svengali_tags": get_sv_raw(story),
        "source_filter": filters["source"],
        "source_mode": filters["source_mode"],
    }

    append_jsonl(STATE["results_path"], row)
    STATE["results"].append(row)
    STATE["reviewed_keys"].add((story_id, tag_key))

    return redirect(url_for("index", tag=tag_key, source=filters["source"], source_mode=filters["source_mode"], shuffle=1 if filters["shuffle"] else 0, sample=filters["sample"]))


@app.route("/stats")
def stats():
    filters = current_filters_from_request(request)
    tag = filters["tag"]
    if not tag:
        return jsonify({"error": "No tag selected"}), 400

    summary = summarize_candidate_counts(tag)
    review_counts = reviewed_breakdown(tag)
    out = {
        "tag": tag,
        "summary": {
            "candidate_total": summary["total"],
            "reviewed": summary["reviewed"],
            "remaining": summary["remaining"],
        },
        "review_counts": dict(review_counts),
        "by_source": dict(summary["by_source"]),
        "source_only": dict(summary["source_only"]),
    }
    return jsonify(out)


@app.route("/story/<story_id>")
def story_json(story_id: str):
    story = STATE["stories_by_id"].get(story_id)
    if not story:
        return jsonify({"error": "not found"}), 404
    return Response(json.dumps(story, ensure_ascii=False, indent=2), mimetype="application/json")


@app.route("/export")
def export_results():
    fmt = (request.args.get("fmt") or "jsonl").lower()
    tag = (request.args.get("tag") or "").strip()
    source = (request.args.get("source") or "").strip()
    source_mode = (request.args.get("source_mode") or "any").strip().lower()

    rows = STATE["results"]
    if tag:
        rows = [r for r in rows if r.get("tag") == tag]
    if source and source_mode in {"contains", "only"}:
        if source_mode == "contains":
            rows = [r for r in rows if source in (r.get("sources") or [])]
        else:
            rows = [r for r in rows if set(r.get("sources") or []) == {source}]

    if fmt == "csv":
        sio = io.StringIO()
        writer = csv.DictWriter(
            sio,
            fieldnames=[
                "reviewed_at", "story_id", "title", "author", "tag",
                "dimension", "tag_value", "label", "sources",
                "svengali_tags", "source_filter", "source_mode",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["sources"] = ",".join(out.get("sources") or [])
            out["svengali_tags"] = ",".join(out.get("svengali_tags") or [])
            writer.writerow(out)
        return Response(sio.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=fp_audit_results.csv"})

    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return Response(data, mimetype="application/x-ndjson", headers={"Content-Disposition": "attachment; filename=fp_audit_results.jsonl"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local false-positive audit UI for story tags.")
    p.add_argument("--index", default=str(DEFAULT_INDEX), help="Path to stories_index.json")
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to review results JSONL")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_state(Path(args.index), Path(args.results))
    print(f"Loaded {len(STATE['stories']):,} stories")
    print(f"Loaded {len(STATE['results']):,} prior review rows")
    print(f"Results file: {Path(args.results).resolve()}")
    print(f"Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
