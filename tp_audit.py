#!/usr/bin/env python3
"""
tp_audit.py — Flask UI for TP mining: find true-positive training examples
for tags that lack Svengali labels.

The existing fp_audit already collected ~18–37 confirmed TN (FALSE_POSITIVE)
labels per tag. This tool surfaces TP candidates from the full story corpus
via keyword search so we can reach ≥50 TPs per tag for SetFit training.

Run:
    python tp_audit.py

Then open:
    http://127.0.0.1:5011

Keyboard shortcuts:
    1 = TRUE_POSITIVE     (this story genuinely has the tag)
    2 = FALSE_POSITIVE    (keyword matched but tag doesn't apply)
    3 = AMBIGUOUS
    4 = SKIP

Output: output/fp_audit_results.jsonl  (same file / format as fp_audit.py)

TP targets per tag (need to reach 50 TPs total):
    awareness:oblivious                 24 more needed
    dominance:cold / clinical           20 more needed
    mental_changes:corruption arc       50 more needed
    mental_changes:identity loss        50 more needed
    relationship:reluctant subject      50 more needed
    themes:horror / dark                26 more needed
    sexual_acts:anal                    29 more needed
    sexual_acts:creampie                42 more needed
    sexual_acts:facial                  37 more needed
    sexual_acts:toys                    35 more needed
    mind_control_type:chemical          33 more needed
    mind_control_type:magic             27 more needed
    mind_control_type:parasite/symbiote 37 more needed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, redirect, render_template_string, request, url_for

DEFAULT_INDEX   = Path("output/stories_index.json")
DEFAULT_RESULTS = Path("output/fp_audit_results.jsonl")
DEFAULT_PORT    = 5011

# ── Candidate keyword patterns per tag ────────────────────────────────────────
# Broad patterns intentionally — we want recall here, precision comes from the human label.
# Each pattern is tested against full_text (case-insensitive).

CANDIDATE_PATTERNS: dict[str, list[str]] = {
    "awareness:oblivious": [
        r"\bhad no idea\b", r"\bnever (knew|realized|suspected)\b",
        r"\bcompletely (unaware|oblivious)\b", r"\bhad no (memory|recollection)\b",
        r"\bdidn'?t (know|realize|suspect|remember)\b.{0,40}(control|hypnoti|done to|happen)",
    ],
    "dominance:cold / clinical": [
        r"\bclinical\b", r"\bdetach(ed|ment)\b", r"\bdispassionate\b",
        r"\bspecimen\b", r"\basset\b",
        r"\bcold.?(efficiency|calculation|precision|logic|manner|voice|tone)\b",
    ],
    "mental_changes:corruption arc": [
        r"\bcorrupt\w*\b", r"\bdegrad\w*\b", r"\bdevolv\w*\b",
        r"\bno longer (herself|himself|the (same|person|woman|man))\b",
        r"\bbimbo(ize|ification)?\b", r"\bfallen\b.{0,40}(values|morals|person|soul)",
        r"\bbeyond saving\b", r"\bslipping away\b",
    ],
    "mental_changes:identity loss": [
        r"\blost (herself|himself|her sense of self|who she was|who he was)\b",
        r"\bno longer (who|the person) (she|he) (was|used to be)\b",
        r"\bidentity\b.{0,40}(erase|strip|destroy|gone|lost|shatter|dissolv)",
        r"\b(old|former|true) (her|him|self|identity|personality)\b",
        r"\bself was (gone|erased|destroyed|lost|stripped)\b",
        r"\bforgot (who|that) (she|he) (was|used to be)\b",
    ],
    "relationship:reluctant subject": [
        r"\bdidn'?t want (to|this|it)\b", r"\bagainst her will\b", r"\bagainst his will\b",
        r"\breluctant(ly)?\b", r"\btried to (resist|fight|stop|refuse)\b",
        r"\b(fought|resisted|struggled) (against|the)\b",
        r"\bno (choice|say|control)\b", r"\bnot (willing|consenting|agreeing)\b",
        r"\bforced (to|into)\b",
    ],
    "themes:horror / dark": [
        r"\bhorror\b", r"\bnightmare\b", r"\bterror\b", r"\bdread\b",
        r"\bsinister\b", r"\bdisturbing\b", r"\bmalevolent\b", r"\bmalicious\b",
        r"\bhorrif\w*\b", r"\btraumat\w*\b", r"\bmonstrous\b",
    ],
    "sexual_acts:anal": [
        r"\banal\b", r"\bass(hole|fuck|sex)\b", r"\bbackdoor\b",
        r"\brear (entry|penetration|passage)\b", r"\bsodomiz\w*\b",
        r"\bbutt (sex|fuck|plug)\b",
    ],
    "sexual_acts:creampie": [
        r"\bcreampie\b", r"\bcum(ming)? inside\b", r"\bfilled? (her|him|them) (up|with)\b",
        r"\bseed(ed|ing)? (her|him|them|her womb)\b",
        r"\bdripping.{0,20}(out|down).{0,20}(her|him)\b",
    ],
    "sexual_acts:facial": [
        r"\bfacial\b", r"\bcum.{0,10}(on|across|covering).{0,20}(face|cheek|chin|mouth)\b",
        r"\b(her|his).{0,10}face.{0,20}(cum|covered|dripping|coated)\b",
        r"\bcumshot\b",
    ],
    "sexual_acts:toys": [
        r"\bvibrat(or|ing|ed|e)\b", r"\bdildo\b", r"\bbutt plug\b",
        r"\bsex toy\b", r"\bwand\b.{0,20}(used|pressed|applied|against)",
        r"\btoy(s)? (inserted|used|placed|inside)\b",
        r"\bplug(ged)? (into|inside|her|him)\b",
    ],
    "mind_control_type:chemical": [
        r"\bdrug(ged|ging)?\b", r"\bpotion\b", r"\bserum\b",
        r"\bchemi(cal|cally)\b", r"\bsubstance\b.{0,30}(control|compel|affect|alter|mind)",
        r"\bdose(d|ing)?\b.{0,20}(her|him|them)\b",
        r"\bpheromone\b", r"\btoxin\b", r"\bnarcotic\b",
    ],
    "mind_control_type:magic": [
        r"\bspell\b", r"\bmagic(al)?\b", r"\bwitch\b", r"\bwizard\b",
        r"\britual\b", r"\benchant(ed|ment|ing)?\b",
        r"\bcurse(d)?\b.{0,30}(mind|will|control|obey)",
        r"\bsorcery\b", r"\bincantation\b",
    ],
    "mind_control_type:parasite / symbiote": [
        r"\bparasite\b", r"\bsymbiote?\b", r"\binfect(ed|ion|ing)?\b",
        r"\bslug\b.{0,30}(mind|brain|attach|enter|control)",
        r"\bcreature.{0,30}(attach|enter|take|control|host)",
        r"\bhost(ed|ing)?\b.{0,30}(creature|being|entity|parasite)",
        r"\btendr?il(s)?\b.{0,30}(brain|mind|enter|wrap|attach)",
    ],
    "mental_changes:bimboization": [
        r"\bbimbo(ize|ized|ification|tized|tization)?\b",
        r"\b(airhead|empty.?headed|ditz(y|ier)?)\b",
        r"\b(dumb(er|ing down)|getting dumber|made (her|him) dumber)\b",
        r"\bcan'?t (think|remember|focus).{0,30}(anymore|now|properly)\b",
        r"\bthinking.{0,20}(hard|difficult|impossible|getting harder)\b",
        r"\bvalley girl\b",
    ],
    "mental_changes:intelligence change": [
        r"\b(smarter|dumber|more intelligent|less intelligent)\b",
        r"\b(IQ|intelligence).{0,30}(raised|lowered|increased|decreased|changed|altered)\b",
        r"\b(enhanced|heightened|sharpened).{0,30}(mind|intellect|intelligence|cognitive)\b",
        r"\b(dulled|diminished|reduced).{0,30}(mind|intellect|intelligence|cognitive)\b",
        r"\b(mental capacity|mental acuity|cognitive ability).{0,30}(changed|altered|enhanced|reduced)\b",
        r"\bgenius\b.{0,30}(made|became|turned|transformed)\b",
    ],
    "themes:serial recruitment": [
        r"\brecruit(ed|ing|ment)\b", r"\benlist(ed|ing)\b",
        r"\bconvert(ed|ing)\b.{0,30}(more|others|another|next)",
        r"\bspread(ing)?\b.{0,30}(control|influence|conditioning|program)",
        r"\bnew (subject|victim|recruit|slave|drone|convert)\b",
        r"\b(find|bring|get|turn).{0,20}(more|another|next).{0,20}(subject|victim|candidate|girl|woman|man)\b",
        r"\bone (by one|at a time).{0,30}(control|convert|recruit|program)\b",
    ],
    "themes:betrayal": [
        r"\bbetray(ed|al|ing|er)?\b",
        r"\bstab(bed)?.{0,10}back\b", r"\bback.?stab\b",
        r"\bturn(ed)? (against|on) (her|him|them)\b",
        r"\bused (her|him|them).{0,20}(then|and then|only to|before)\b",
        r"\btrust(ed)?.{0,20}(broken|shattered|violated|betrayed|destroyed)\b",
        r"\b(friend|lover|ally|partner).{0,30}(betray|turn|sell|manipulate|trick)\b",
    ],
}

# How many TPs we need total per tag for training
TP_TARGET = 100

# Current TP counts from existing audit (approximate — recalculated at load)
# Format: tag_key -> {"TP": N, "FP": N, "AMB": N}

# Priority order: 0-TP tags first, then scarce-candidate tags, "bad F1 potential" last.
# This order also drives auto_label.py — if credits run out, the most impactful tags
# are done first. "Bad F1 potential" = broad/subtle concepts where SetFit yields lower ROI.
TARGET_TAGS = [
    # ── Tier 1: 0 TPs, high semantic value ────────────────────────────────────
    "relationship:reluctant subject",       # 0 TP, 50 needed, 4,584 cands — high value
    "mental_changes:corruption arc",        # 0 TP, 50 needed, 4,554 cands — high value
    "mental_changes:identity loss",         # 0 TP, 50 needed,   752 cands — scarce!
    # ── Tier 2: partial TPs, scarce candidates (do before pool runs dry) ──────
    "sexual_acts:creampie",                 # 8 TP, 42 needed, 1,204 cands — scarce
    "mind_control_type:parasite / symbiote",# 13 TP, 37 needed, 1,161 cands — scarce
    "sexual_acts:facial",                   # 13 TP, 37 needed, 1,982 cands
    # ── Tier 3: partial TPs, plentiful candidates ─────────────────────────────
    "sexual_acts:toys",                     # 15 TP, 35 needed, 5,497 cands
    "mind_control_type:chemical",           # 17 TP, 33 needed, 4,542 cands
    "sexual_acts:anal",                     # 21 TP, 29 needed, 4,651 cands
    "mind_control_type:magic",              # 23 TP, 27 needed, 7,582 cands
    # ── Tier 4: bad-F1-potential tags (broad/subtle — lower SetFit ROI) ───────
    "themes:horror / dark",                 # 50 TP → 100 target — very broad concept
    "awareness:oblivious",                  # 50 TP → 100 target — subtle awareness state
    "dominance:cold / clinical",            # 50 TP → 100 target — tone-level concept
    # ── Run 2 additions (March 2026) ──────────────────────────────────────────
    "mental_changes:bimboization",          # 0 TP → 100 target — Svengali ground truth exists
    "mental_changes:intelligence change",   # 19 TP → 100 target — same concept family
    "themes:serial recruitment",            # 25 TP → 100 target — incomplete from run 1
    "themes:betrayal",                      # 31 TP → 100 target — incomplete from run 1, 70% prec
]

app = Flask(__name__)

STATE: dict[str, Any] = {
    "index_path":    DEFAULT_INDEX,
    "results_path":  DEFAULT_RESULTS,
    "stories":       [],
    "stories_by_id": {},
    "results":       [],
    "reviewed_keys": set(),   # (story_id, tag_key)
    "audit_counts":  {},      # tag_key -> {"TP": N, "FP": N, "AMB": N}
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def get_auto_tags(story: dict) -> dict:
    at = story.get("auto_tags")
    return at if isinstance(at, dict) else {}


def get_sv_raw(story: dict) -> list[str]:
    at = get_auto_tags(story)
    for key in ("_svengali_tags", "svengali_tags"):
        val = at.get(key) or story.get(key)
        if isinstance(val, list):
            return [x for x in val if isinstance(x, str)]
    return []


def get_story_text(story: dict) -> str:
    for key in ("full_text", "text", "story_text"):
        val = story.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def story_currently_has_tag(story: dict, dim: str, tag: str) -> bool:
    at = get_auto_tags(story)
    return tag in (at.get(dim) or [])


def story_matches_patterns(story: dict, tag_key: str) -> bool:
    patterns = CANDIDATE_PATTERNS.get(tag_key, [])
    if not patterns:
        return False
    text = get_story_text(story)
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def build_audit_counts(results: list[dict]) -> dict:
    counts = {}
    for row in results:
        key = row.get("tag")
        if not key:
            continue
        if key not in counts:
            counts[key] = {"TP": 0, "FP": 0, "AMB": 0, "SKIP": 0}
        label = row.get("label", "")
        if label == "TRUE_POSITIVE":
            counts[key]["TP"] += 1
        elif label == "FALSE_POSITIVE":
            counts[key]["FP"] += 1
        elif label == "AMBIGUOUS":
            counts[key]["AMB"] += 1
        elif label == "SKIP":
            counts[key]["SKIP"] += 1
    return counts


def tp_needed(tag_key: str) -> int:
    counts = STATE["audit_counts"].get(tag_key, {})
    return max(0, TP_TARGET - counts.get("TP", 0))


def load_state(index_path: Path, results_path: Path) -> None:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", data) if isinstance(data, dict) else data
    results = load_jsonl(results_path)
    reviewed_keys = {
        (r.get("story_id"), r.get("tag"))
        for r in results
        if r.get("story_id") and r.get("tag")
    }
    STATE["index_path"]    = index_path
    STATE["results_path"]  = results_path
    STATE["stories"]       = stories
    STATE["stories_by_id"] = {s.get("id"): s for s in stories if s.get("id")}
    STATE["results"]       = results
    STATE["reviewed_keys"] = reviewed_keys
    STATE["audit_counts"]  = build_audit_counts(results)


def candidate_rows(tag_key: str, shuffle: bool = False, sample: int = 0) -> list[dict]:
    """Return stories matching keyword patterns that haven't been reviewed yet."""
    if ":" not in tag_key:
        return []
    dim, tag = tag_key.split(":", 1)
    rows = []
    for story in STATE["stories"]:
        sid = story.get("id")
        if not sid:
            continue
        if (sid, tag_key) in STATE["reviewed_keys"]:
            continue
        if not story_matches_patterns(story, tag_key):
            continue
        rows.append({
            "story_id":      sid,
            "title":         story.get("title") or sid,
            "author":        story.get("author") or "",
            "tag":           tag,
            "tag_key":       tag_key,
            "dimension":     dim,
            "currently_tagged": story_currently_has_tag(story, dim, tag),
            "svengali_tags": get_sv_raw(story),
            "story":         story,
        })

    if shuffle:
        random.Random(42).shuffle(rows)
    elif not shuffle:
        # Put un-tagged stories first (more valuable as new TPs)
        rows.sort(key=lambda r: (r["currently_tagged"], r["story_id"]))

    if sample:
        rows = rows[:sample]
    return rows


def text_snippet(text: str, chars: int = 2500) -> str:
    if not text:
        return ""
    return text[:chars].rstrip() + ("\n\n[...truncated...]" if len(text) > chars else "")


# ── Flask routes ───────────────────────────────────────────────────────────────

PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TP Audit</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; line-height: 1.35; }
    .wrap { max-width: 1200px; margin: 0 auto; }
    .toolbar, .card, .stats, .help { border: 1px solid #ddd; border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }
    .toolbar { background: #fafafa; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { display: block; font-size: 13px; color: #333; margin-bottom: 4px; }
    select, input[type=text], input[type=number] { padding: 8px; min-width: 260px; }
    button, .btn { padding: 10px 14px; border: 1px solid #bbb; background: white; border-radius: 8px; cursor: pointer; text-decoration: none; color: inherit; display: inline-block; }
    button.primary { background: #111; color: white; border-color: #111; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .meta { color: #444; font-size: 14px; margin-bottom: 4px; }
    .pill { display: inline-block; border-radius: 999px; padding: 4px 10px; background: #f0f0f0; margin-right: 6px; margin-bottom: 6px; font-size: 13px; }
    .pill.tagged { background: #d4edda; }
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
    .progress-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .progress-table th { text-align: left; padding: 4px 8px; border-bottom: 1px solid #ddd; }
    .progress-table td { padding: 4px 8px; }
    .done { color: #2d7a2d; font-weight: bold; }
    .need { color: #c00; }
    .bar-outer { background: #eee; border-radius: 4px; height: 8px; width: 100px; display: inline-block; vertical-align: middle; }
    .bar-inner { background: #2d7a2d; border-radius: 4px; height: 8px; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>TP Audit — True-positive mining for SetFit training</h1>

  <div class="toolbar">
    <form method="get" action="/">
      <div class="row">
        <div>
          <label>Tag (focus on tags with need > 0)</label>
          <select name="tag" required>
            {% for tk in target_tags %}
              {% set need = tp_needed_map.get(tk, 0) %}
              <option value="{{ tk }}" {% if tk == selected_tag %}selected{% endif %}>
                {{ tk }} — need {{ need }} TP{% if need == 0 %} ✓{% endif %}
              </option>
            {% endfor %}
          </select>
        </div>
        <div>
          <label>Sample size (0 = all)</label>
          <input type="number" name="sample" min="0" value="{{ sample }}">
        </div>
        <div>
          <label><input type="checkbox" name="shuffle" value="1" {% if shuffle %}checked{% endif %}> Shuffle</label>
        </div>
        <div>
          <button class="primary" type="submit">Apply</button>
        </div>
      </div>
    </form>
  </div>

  <div class="grid2">
    <div class="stats">
      <h3 style="margin-top:0;">Progress: {{ selected_tag }}</h3>
      {% set counts = audit_counts.get(selected_tag, {}) %}
      <div><strong>TPs found:</strong> {{ counts.get('TP', 0) }} / {{ tp_target }}
        <span class="bar-outer"><span class="bar-inner" style="width: {{ [counts.get('TP',0)*100//tp_target, 100]|min }}px;"></span></span>
      </div>
      <div><strong>TNs found (FP labels):</strong> {{ counts.get('FP', 0) }}</div>
      <div><strong>Ambiguous:</strong> {{ counts.get('AMB', 0) }}</div>
      <div><strong>Candidates remaining:</strong> {{ n_remaining }}</div>
      <div style="margin-top:8px;">
        {% if tp_needed_map.get(selected_tag, 0) == 0 %}
          <span class="done">✓ TP target reached for this tag</span>
        {% else %}
          <span class="need">Still need {{ tp_needed_map[selected_tag] }} more TPs</span>
        {% endif %}
      </div>
    </div>
    <div class="stats">
      <h3 style="margin-top:0;">All tags progress</h3>
      <table class="progress-table">
        <tr><th>Tag</th><th>TP</th><th>TN</th><th>Need</th></tr>
        {% for tk in target_tags %}
          {% set c = audit_counts.get(tk, {}) %}
          {% set need = tp_needed_map.get(tk, 0) %}
          <tr {% if tk == selected_tag %}style="background:#f0f7ff;"{% endif %}>
            <td>{{ tk }}</td>
            <td>{{ c.get('TP', 0) }}</td>
            <td>{{ c.get('FP', 0) }}</td>
            <td class="{% if need == 0 %}done{% else %}need{% endif %}">{{ need }}</td>
          </tr>
        {% endfor %}
      </table>
    </div>
  </div>

  {% if row %}
  <div class="card">
    <h2 style="margin-top:0;">{{ row.title }}</h2>
    <div class="meta"><strong>Author:</strong> {{ row.author or "Unknown" }}</div>
    <div class="meta"><strong>ID:</strong> <span class="mono">{{ row.story_id }}</span></div>
    <div class="meta"><strong>Tag being audited:</strong> <span class="mono">{{ row.tag_key }}</span></div>
    <div class="meta"><strong>Currently in auto_tags:</strong>
      {% if row.currently_tagged %}
        <span class="pill tagged">YES</span>
      {% else %}
        <span class="pill">NO — candidate from keyword search</span>
      {% endif %}
    </div>
    <div class="meta"><strong>Svengali tags:</strong>
      {% if row.svengali_tags %}
        {% for t in row.svengali_tags[:20] %}<span class="pill">{{ t }}</span>{% endfor %}
      {% else %}
        <span class="muted">none</span>
      {% endif %}
    </div>
    <div class="meta" style="margin-top:8px;">
      <a class="btn" href="{{ url_for('story_json', story_id=row.story_id) }}" target="_blank">View raw JSON</a>
    </div>
  </div>

  <div class="card">
    <p class="small">Label this story: does it genuinely exhibit <strong>{{ row.tag_key }}</strong>?</p>
    <div class="actions">
      {% for label, key, cls, hint in [
          ('TRUE_POSITIVE',  '1', 'btn-1', 'Yes, this tag applies'),
          ('FALSE_POSITIVE', '2', 'btn-2', 'No, keyword matched but tag does not apply'),
          ('AMBIGUOUS',      '3', 'btn-3', 'Unclear'),
          ('SKIP',           '4', 'btn-4', 'Skip for now')
      ] %}
      <form method="post" action="/review">
        <input type="hidden" name="story_id" value="{{ row.story_id }}">
        <input type="hidden" name="tag" value="{{ row.tag_key }}">
        <input type="hidden" name="label" value="{{ label }}">
        <input type="hidden" name="shuffle" value="{{ 1 if shuffle else 0 }}">
        <input type="hidden" name="sample" value="{{ sample }}">
        <button class="{{ cls }}" type="submit"><strong>{{ key }}</strong> — {{ label }}<br><span class="small">{{ hint }}</span></button>
      </form>
      {% endfor %}
    </div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Story text</h3>
    <details open>
      <summary>Snippet (first 2500 chars)</summary>
      <div class="textbox">{{ snippet }}</div>
    </details>
    <details>
      <summary>Full text</summary>
      <div class="textbox">{{ full_text }}</div>
    </details>
  </div>
  {% else %}
  <div class="card empty">
    <h2>No more candidates for {{ selected_tag }}.</h2>
    <p>Either all keyword-matching stories have been reviewed, or this tag has no matching patterns.<br>
    Pick another tag from the dropdown.</p>
  </div>
  {% endif %}
</div>

<script>
(function () {
  const mapping = { "1": "TRUE_POSITIVE", "2": "FALSE_POSITIVE", "3": "AMBIGUOUS", "4": "SKIP" };
  document.addEventListener("keydown", function (e) {
    if (!mapping[e.key]) return;
    const input = document.querySelector('input[name="label"][value="' + mapping[e.key] + '"]');
    if (input && input.form) { e.preventDefault(); input.form.submit(); }
  });
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    tag_key  = (request.args.get("tag") or TARGET_TAGS[0]).strip()
    shuffle  = str(request.args.get("shuffle") or "") in {"1", "true", "on"}
    try:
        sample = int(request.args.get("sample") or 0)
    except ValueError:
        sample = 0

    rows = candidate_rows(tag_key, shuffle=shuffle, sample=sample)
    row  = rows[0] if rows else None

    tp_needed_map = {tk: tp_needed(tk) for tk in TARGET_TAGS}
    full_text = get_story_text(row["story"]) if row else ""

    return render_template_string(
        PAGE_TEMPLATE,
        target_tags    = TARGET_TAGS,
        selected_tag   = tag_key,
        shuffle        = shuffle,
        sample         = sample,
        row            = row,
        n_remaining    = len(rows),
        tp_target      = TP_TARGET,
        tp_needed_map  = tp_needed_map,
        audit_counts   = STATE["audit_counts"],
        snippet        = text_snippet(full_text),
        full_text      = full_text,
    )


@app.route("/review", methods=["POST"])
def review():
    story_id = request.form.get("story_id", "").strip()
    tag_key  = request.form.get("tag", "").strip()
    label    = request.form.get("label", "").strip()
    shuffle  = str(request.form.get("shuffle") or "") in {"1", "true", "on"}
    try:
        sample = int(request.form.get("sample") or 0)
    except ValueError:
        sample = 0

    if not story_id or not tag_key or label not in {"TRUE_POSITIVE", "FALSE_POSITIVE", "AMBIGUOUS", "SKIP"}:
        return "Bad review payload", 400

    story = STATE["stories_by_id"].get(story_id)
    if not story:
        return "Unknown story_id", 404

    if ":" not in tag_key:
        return "Bad tag_key", 400
    dim, tag = tag_key.split(":", 1)

    row = {
        "reviewed_at":   now_iso(),
        "story_id":      story_id,
        "title":         story.get("title"),
        "author":        story.get("author"),
        "tag":           tag_key,
        "dimension":     dim,
        "tag_value":     tag,
        "label":         label,
        "sources":       ["tp_mining"],   # marks this as LLM/human TP mining, not FP audit
        "svengali_tags": get_sv_raw(story),
        "source_filter": "tp_mining",
        "source_mode":   "tp_mining",
    }

    append_jsonl(STATE["results_path"], row)
    STATE["results"].append(row)
    STATE["reviewed_keys"].add((story_id, tag_key))
    STATE["audit_counts"] = build_audit_counts(STATE["results"])

    return redirect(url_for("index", tag=tag_key,
                            shuffle=1 if shuffle else 0, sample=sample))


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
    rows = [r for r in STATE["results"] if not tag or r.get("tag") == tag]
    if fmt == "csv":
        sio = io.StringIO()
        writer = csv.DictWriter(sio, fieldnames=[
            "reviewed_at", "story_id", "title", "author", "tag",
            "dimension", "tag_value", "label", "sources", "svengali_tags",
        ])
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["sources"] = ",".join(out.get("sources") or [])
            out["svengali_tags"] = ",".join(out.get("svengali_tags") or [])
            writer.writerow(out)
        return Response(sio.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=tp_audit.csv"})
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return Response(data, mimetype="application/x-ndjson",
                    headers={"Content-Disposition": "attachment; filename=tp_audit.jsonl"})


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TP mining audit UI for SetFit training data.")
    p.add_argument("--index",   default=str(DEFAULT_INDEX),   help="Path to stories_index.json")
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to audit results JSONL")
    p.add_argument("--host",    default="127.0.0.1")
    p.add_argument("--port",    type=int, default=DEFAULT_PORT)
    p.add_argument("--debug",   action="store_true")
    p.add_argument("--stats",   action="store_true", help="Print TP gap table and exit (no server)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_state(Path(args.index), Path(args.results))

    if args.stats:
        print(f"\n{'Tag':55}  {'TP':>4}  {'TN':>4}  {'AMB':>4}  {'Need':>6}  {'Candidates':>10}")
        print("-" * 90)
        for tk in TARGET_TAGS:
            c = STATE["audit_counts"].get(tk, {})
            n_cand = sum(1 for s in STATE["stories"] if story_matches_patterns(s, tk)
                         and (s.get("id"), tk) not in STATE["reviewed_keys"])
            need = tp_needed(tk)
            print(f"{tk:55}  {c.get('TP',0):>4}  {c.get('FP',0):>4}  {c.get('AMB',0):>4}  "
                  f"{'DONE' if need==0 else str(need):>6}  {n_cand:>10,}")
        return

    print(f"Loaded {len(STATE['stories']):,} stories")
    print(f"Loaded {len(STATE['results']):,} prior review rows")
    print(f"Results file: {Path(args.results).resolve()}")
    print(f"\nTP gaps:")
    for tk in TARGET_TAGS:
        c = STATE["audit_counts"].get(tk, {})
        need = tp_needed(tk)
        status = "DONE" if need == 0 else f"need {need} more"
        print(f"  {tk:55}  TP={c.get('TP',0)}  TN={c.get('FP',0)}  {status}")
    print(f"\nOpen: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
