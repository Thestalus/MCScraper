#!/usr/bin/env python3
"""
fp_audit_auto_review.py

Status: research/audit tool, not part of the regular pipeline (see ARCHITECTURE.md).
Produced the March 2026 LLM-adjudicated audit in fp_audit_results.jsonl. Needs ANTHROPIC_API_KEY.

Automated FP audit using Claude Haiku via the Anthropic API.
Replaces the manual copy-paste-to-ChatGPT step.

Reads candidate false positives from stories_index.json, sends them to Claude
in batches, parses the response, and writes results to fp_audit_results.jsonl.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python3 fp_audit_auto_review.py                          # all priority tags, 50 each
    python3 fp_audit_auto_review.py --tag mental_changes:obedience programming
    python3 fp_audit_auto_review.py --tag mind_control_type:possession --n 30
    python3 fp_audit_auto_review.py --all-tags --n 50        # every tag in the index
    python3 fp_audit_auto_review.py --dry-run                # show what would run, no API calls
    python3 fp_audit_auto_review.py --model claude-sonnet-4-6  # use Sonnet instead
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INDEX   = Path("output/stories_index.json")
DEFAULT_RESULTS = Path("output/fp_audit_results.jsonl")
DEFAULT_MODEL   = "claude-haiku-4-5-20251001"
DEFAULT_N       = 50
DEFAULT_CHARS   = 4000
BATCH_SIZE      = 5   # stories per API call
SV_RAW_FIELDS   = ("_svengali_tags", "svengali_tags")
TEXT_FIELDS     = ("full_text", "text", "story_text")

# Import priority list from the workflow helper
try:
    from fp_audit_workflow import PRIORITY_TAGS, SKIP_TAGS
except ImportError:
    PRIORITY_TAGS = []
    SKIP_TAGS = set()


# ── Data helpers ─────────────────────────────────────────────────────────────

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
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def get_auto_tags(story: dict) -> dict:
    at = story.get("auto_tags")
    return at if isinstance(at, dict) else {}


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


def get_tag_sources(story: dict, tag: str) -> list[str]:
    at = get_auto_tags(story)
    src_map = at.get("_tag_sources") or story.get("_tag_sources") or {}
    if not isinstance(src_map, dict):
        return []
    vals = src_map.get(tag, [])
    return [v for v in vals if isinstance(v, str)] if isinstance(vals, list) else []


def get_story_text(story: dict) -> str:
    for key in TEXT_FIELDS:
        val = story.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def smart_truncate(text: str, chars: int) -> str:
    text = (text or "").strip()
    if chars <= 0 or len(text) <= chars:
        return text
    part = chars // 3
    mid_start = max(0, len(text) // 2 - part // 2)
    return (
        text[:part].rstrip()
        + "\n\n[...middle...]\n\n"
        + text[mid_start:mid_start + part].rstrip()
        + "\n\n[...end...]\n\n"
        + text[-part:].rstrip()
    )


def split_tag_key(tag_key: str) -> tuple[str, str]:
    if ":" not in tag_key:
        raise ValueError(f"Tag key must be dimension:tag, got {tag_key!r}")
    dim, tag = tag_key.split(":", 1)
    return dim.strip(), tag.strip()


# ── Candidate selection ──────────────────────────────────────────────────────

def build_candidates(
    stories: list[dict],
    tag_key: str,
    already_reviewed: set[tuple[str, str]],
    n: int,
    seed: int = 42,
) -> list[dict]:
    dim, tag = split_tag_key(tag_key)
    pool = []
    for story in stories:
        sid = story.get("id")
        if not sid:
            continue
        if (sid, tag_key) in already_reviewed:
            continue
        at = get_auto_tags(story)
        vals = at.get(dim, [])
        if not isinstance(vals, list) or tag not in vals:
            continue
        if tag in set(get_sv_raw(story)):
            continue  # Svengali already confirmed — not a candidate FP
        pool.append(story)

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def all_tag_keys(stories: list[dict]) -> list[str]:
    """All dimension:tag keys present in the index, sorted by priority then volume."""
    from collections import Counter
    counts: Counter = Counter()
    for story in stories:
        at = get_auto_tags(story)
        for dim, vals in at.items():
            if dim.startswith("_") or not isinstance(vals, list):
                continue
            for tag in vals:
                if isinstance(tag, str):
                    counts[f"{dim}:{tag}"] += 1

    priority_index = {t: i for i, t in enumerate(PRIORITY_TAGS)}
    keys = [k for k in counts if k not in SKIP_TAGS]
    keys.sort(key=lambda k: (priority_index.get(k, len(PRIORITY_TAGS)), -counts[k], k))
    return keys


# ── Claude API ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are auditing auto-generated story tags. For each story excerpt you will:
1. Read the excerpt carefully.
2. Decide if the given tag is a correct label for that story.

Output ONLY a pipe-delimited table, one row per story, no extra text:
story_id | label | short_reason

Labels (use exactly):
  TRUE_POSITIVE  — tag is correct; the auto-tagger got it right
  FALSE_POSITIVE — tag is wrong for this story
  AMBIGUOUS      — genuinely unclear from the excerpt
  SKIP           — excerpt too short/corrupt to judge
"""

def make_user_message(tag_key: str, candidates: list[dict], chars: int) -> str:
    lines = [f"Tag being audited: **{tag_key}**\n"]
    for i, c in enumerate(candidates, 1):
        text = smart_truncate(get_story_text(c), chars)
        lines.append(f"--- STORY {i} ---")
        lines.append(f"story_id: {c['id']}")
        lines.append(f"title: {c.get('title') or c['id']}")
        lines.append(f"text:\n{text or '(no text)'}")
        lines.append("")
    return "\n".join(lines)


LABEL_MAP = {
    "TRUE_POSITIVE": "TRUE_POSITIVE",
    "FALSE_POSITIVE": "FALSE_POSITIVE",
    "AMBIGUOUS": "AMBIGUOUS",
    "SKIP": "SKIP",
    "TP": "TRUE_POSITIVE",
    "FP": "FALSE_POSITIVE",
    "AMB": "AMBIGUOUS",
}

def parse_response(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("|")
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        story_id = re.sub(r"^\d+\.\s*", "", parts[0]).strip()
        raw_label = parts[1].upper().replace(" ", "_").strip("`* ")
        label = LABEL_MAP.get(raw_label)
        if not story_id or not label:
            continue
        reason = parts[2] if len(parts) > 2 else ""
        rows.append({"story_id": story_id, "label": label, "short_reason": reason})
    return rows


def call_claude(client, model: str, tag_key: str, candidates: list[dict], chars: int) -> list[dict]:
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": make_user_message(tag_key, candidates, chars)}],
    )
    raw = msg.content[0].text if msg.content else ""
    return parse_response(raw)


# ── Main ─────────────────────────────────────────────────────────────────────

def review_tag(
    client,
    stories: list[dict],
    story_by_id: dict,
    tag_key: str,
    already_reviewed: set[tuple[str, str]],
    n: int,
    model: str,
    chars: int,
    dry_run: bool,
    results_path: Path,
    verbose: bool,
) -> int:
    candidates = build_candidates(stories, tag_key, already_reviewed, n)
    if not candidates:
        print(f"  {tag_key}: no unreviewed candidates — skipping")
        return 0

    print(f"  {tag_key}: {len(candidates)} candidates")

    if dry_run:
        return len(candidates)

    to_write = []
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i:i + BATCH_SIZE]
        try:
            parsed = call_claude(client, model, tag_key, batch, chars)
        except Exception as e:
            print(f"    API error on batch {i//BATCH_SIZE + 1}: {e}", file=sys.stderr)
            time.sleep(5)
            continue

        by_id = {r["story_id"]: r for r in parsed}
        for c in batch:
            sid = c["id"]
            result = by_id.get(sid)
            label = result["label"] if result else "SKIP"
            reason = result["short_reason"] if result else "not returned by model"
            row = {
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewer": f"claude_auto/{model}",
                "story_id": sid,
                "title": c.get("title"),
                "author": c.get("author"),
                "tag": tag_key,
                "dimension": tag_key.split(":")[0],
                "tag_value": tag_key.split(":", 1)[1],
                "label": label,
                "short_reason": reason,
                "sources": get_tag_sources(c, tag_key.split(":", 1)[1]),
                "svengali_tags": get_sv_raw(c),
            }
            to_write.append(row)
            already_reviewed.add((sid, tag_key))
            if verbose:
                print(f"    {sid}: {label}  — {reason[:60]}")

        append_jsonl(results_path, to_write[-len(batch):])
        time.sleep(0.3)  # gentle rate limiting

    tp = sum(1 for r in to_write if r["label"] == "TRUE_POSITIVE")
    fp = sum(1 for r in to_write if r["label"] == "FALSE_POSITIVE")
    prec = f"{tp/(tp+fp):.0%}" if (tp + fp) else "—"
    print(f"    → {len(to_write)} reviewed  TP={tp}  FP={fp}  precision={prec}")
    return len(to_write)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automated FP audit via Claude API.")
    p.add_argument("--tag",       help="Single tag key to audit, e.g. 'mind_control_type:possession'")
    p.add_argument("--all-tags",  action="store_true", help="Audit every tag in the index (slow)")
    p.add_argument("--n",         type=int, default=DEFAULT_N, help=f"Stories per tag (default: {DEFAULT_N})")
    p.add_argument("--model",     default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")
    p.add_argument("--chars",     type=int, default=DEFAULT_CHARS, help=f"Max chars per story excerpt (default: {DEFAULT_CHARS})")
    p.add_argument("--index",     default=str(DEFAULT_INDEX))
    p.add_argument("--results",   default=str(DEFAULT_RESULTS))
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--dry-run",   action="store_true", help="Show what would run without calling the API")
    p.add_argument("--verbose",   action="store_true", help="Print each story result as it's reviewed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json(Path(args.index))
    stories = data.get("stories", data) if isinstance(data, dict) else data
    if not isinstance(stories, list):
        raise SystemExit("Expected stories_index.json to contain a top-level 'stories' list.")
    story_by_id = {s.get("id"): s for s in stories if s and s.get("id")}

    results_path = Path(args.results)
    prior = load_jsonl(results_path)
    already_reviewed: set[tuple[str, str]] = {
        (r["story_id"], r["tag"]) for r in prior if r.get("story_id") and r.get("tag")
    }
    print(f"Loaded {len(stories):,} stories, {len(already_reviewed):,} already reviewed.\n")

    # Determine which tags to audit
    if args.tag:
        tags_to_audit = [args.tag]
    elif args.all_tags:
        tags_to_audit = all_tag_keys(stories)
    else:
        # Default: priority tags only
        tags_to_audit = PRIORITY_TAGS
        if not tags_to_audit:
            raise SystemExit("No PRIORITY_TAGS found. Use --tag or --all-tags.")

    if args.dry_run:
        print("DRY RUN — no API calls will be made.\n")
        total = 0
        for tag_key in tags_to_audit:
            cands = build_candidates(stories, tag_key, already_reviewed, args.n, args.seed)
            n = len(cands)
            total += n
            batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  {tag_key}: {n} candidates  ({batches} API calls)")
        est_input_tokens = total * (args.chars // 4) + total * 100
        est_haiku_cost = est_input_tokens / 1_000_000 * 0.80
        est_sonnet_cost = est_input_tokens / 1_000_000 * 3.00
        print(f"\nTotal stories: {total}")
        print(f"Est. input tokens: {est_input_tokens:,}")
        print(f"Est. cost (Haiku):  ${est_haiku_cost:.2f}")
        print(f"Est. cost (Sonnet): ${est_sonnet_cost:.2f}")
        return

    try:
        import anthropic
    except ImportError:
        raise SystemExit("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    print(f"Model: {args.model}")
    print(f"Tags to audit: {len(tags_to_audit)}")
    print(f"Target per tag: {args.n}\n")

    total_reviewed = 0
    for tag_key in tags_to_audit:
        reviewed = review_tag(
            client=client,
            stories=stories,
            story_by_id=story_by_id,
            tag_key=tag_key,
            already_reviewed=already_reviewed,
            n=args.n,
            model=args.model,
            chars=args.chars,
            dry_run=False,
            results_path=results_path,
            verbose=args.verbose,
        )
        total_reviewed += reviewed

    print(f"\nDone. Total reviewed this run: {total_reviewed}")
    print(f"Results appended to: {results_path}")
    print(f"\nRun the dashboard to see results:")
    print(f"  python3 fp_audit_dashboard.py")


if __name__ == "__main__":
    main()
