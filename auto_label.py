#!/usr/bin/env python3
"""
auto_label.py — LLM-assisted TP/TN labeling for SetFit training data.

Uses Claude to classify candidate stories for each target tag, writing results
to output/fp_audit_results.jsonl (same format as fp_audit.py / tp_audit.py).

Processes tags in priority order (see tp_audit.py TARGET_TAGS):
    Tier 1: 0-TP tags, high semantic value
    Tier 2: scarce candidates (run dry soonest)
    Tier 3: plentiful candidates
    Tier 4: broad/subtle concepts (lower SetFit ROI — saved for last)

Usage:
    python auto_label.py                  # dry-run: show what would be labeled
    python auto_label.py --run            # actually call the API and write results
    python auto_label.py --run --tag awareness:oblivious  # single tag only
    python auto_label.py --run --limit 20 # cap API calls per tag

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY environment variable set

Cost estimate (claude-haiku-4-5, ~600 input tokens + 5 output tokens per call):
    ~$0.0004 per story
    ~300 stories per tag × 13 tags = ~3,900 stories total = ~$1.56
    (less if we already have TPs and stop early)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
INDEX_PATH   = Path("output/stories_index.json")
RESULTS_PATH = Path("output/fp_audit_results.jsonl")
MODEL        = "claude-haiku-4-5-20251001"
TP_TARGET    = 100
MAX_TEXT_WORDS = 500   # words of story text sent to LLM (cost vs accuracy tradeoff)
RATE_LIMIT_DELAY = 0.3 # seconds between API calls

# ── Tag definitions (used in LLM prompt) ──────────────────────────────────────
TAG_DESCRIPTIONS: dict[str, str] = {
    "relationship:reluctant subject": (
        "The subject of mind control did not want it and resisted or tried to resist — "
        "they fought against the control, tried to refuse, showed defiance or struggle. "
        "The non-consensual or reluctant nature is a central element of the story."
    ),
    "mental_changes:corruption arc": (
        "The story depicts a character undergoing a gradual corruption, moral degradation, "
        "or fundamental personality change — where who they are as a person fundamentally "
        "shifts. They become debased, degraded, or lose their values/morals over the story."
    ),
    "mental_changes:identity loss": (
        "The story is centrally concerned with a character losing their identity, selfhood, "
        "or sense of who they are — forgetting who they were, their personality being erased "
        "or replaced, or their sense of self dissolving."
    ),
    "sexual_acts:creampie": (
        "The story contains a sexual scene where a male character ejaculates inside a "
        "female character (internal ejaculation / creampie), and this act is explicitly "
        "described or is a significant part of the scene."
    ),
    "mind_control_type:parasite / symbiote": (
        "The mind control in the story is achieved by a parasite, symbiote, alien creature, "
        "worm, slug, or living organism that physically attaches to or enters the host and "
        "controls them from within. The organism is the primary MC mechanism."
    ),
    "sexual_acts:facial": (
        "The story contains a sexual scene where a male character ejaculates on a female "
        "character's face, and this act is explicitly described or is a significant part "
        "of the sexual encounter."
    ),
    "sexual_acts:toys": (
        "The story contains a sexual scene where sex toys (vibrators, dildos, butt plugs, "
        "wands, etc.) are used on or by a character as a significant part of the sexual "
        "activity — not merely mentioned in passing."
    ),
    "mind_control_type:chemical": (
        "The mind control in the story is achieved through a drug, chemical, potion, serum, "
        "aphrodisiac, pheromone, or ingested/injected substance that alters the victim's "
        "will or behaviour. The chemical is the primary MC mechanism."
    ),
    "sexual_acts:anal": (
        "The story contains anal sex — penetration of the anus — as an explicit and "
        "significant part of the sexual activity. Not merely referenced in passing."
    ),
    "mind_control_type:magic": (
        "The mind control in the story is achieved through magic: spells, enchantments, "
        "charms, curses, arcane rituals, or mystical forces that override someone's free "
        "will. Magic is the primary MC mechanism (not just the setting)."
    ),
    "themes:horror / dark": (
        "The story has a horror, dark, or deeply disturbing tone — featuring genuine fear, "
        "dread, graphic content, psychological terror, or a bleak/threatening atmosphere "
        "that goes beyond typical dark MC erotica into horror-genre territory."
    ),
    "awareness:oblivious": (
        "The subject of mind control has NO idea they are being controlled — they fully "
        "believe their thoughts and actions are their own. The control is completely "
        "invisible to them. This is a central element, not just a brief moment."
    ),
    "dominance:cold / clinical": (
        "The dominant figure in the story is detached, cold, clinical, or businesslike — "
        "treating the subject as an asset, tool, specimen, or project rather than a person. "
        "The relationship has an impersonal, transactional quality."
    ),
}

# Import candidate patterns from tp_audit.py
from tp_audit import CANDIDATE_PATTERNS, TARGET_TAGS


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_stories() -> tuple[list[dict], dict[str, dict]]:
    with INDEX_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", data) if isinstance(data, dict) else data
    return stories, {s["id"]: s for s in stories if s.get("id")}


def load_results() -> tuple[list[dict], set, dict]:
    rows = []
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    reviewed = {(r["story_id"], r["tag"]) for r in rows if r.get("story_id") and r.get("tag")}
    counts = defaultdict(lambda: {"TP": 0, "FP": 0, "AMB": 0})
    for r in rows:
        k, lbl = r.get("tag", ""), r.get("label", "")
        if lbl == "TRUE_POSITIVE":    counts[k]["TP"] += 1
        elif lbl == "FALSE_POSITIVE": counts[k]["FP"] += 1
        elif lbl == "AMBIGUOUS":      counts[k]["AMB"] += 1
    return rows, reviewed, counts


def append_result(row: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def story_text_sample(story: dict, max_words: int = MAX_TEXT_WORDS) -> str:
    title = story.get("title", "")
    desc  = story.get("description", "") or ""
    text  = story.get("full_text", "") or ""
    words = text.split()
    n = len(words)
    # Take head + mid + tail for better coverage
    head = words[:max_words // 3]
    mid  = words[max(0, n//2 - max_words//6) : n//2 + max_words//6]
    tail = words[max(0, n - max_words//3):]
    body = " [...] ".join(" ".join(c) for c in [head, mid, tail] if c)
    return f"Title: {title}\nDescription: {desc}\n\n{body}".strip()


def find_candidates(stories: list[dict], tag_key: str, reviewed: set,
                    shuffle: bool = True) -> list[dict]:
    patterns = CANDIDATE_PATTERNS.get(tag_key, [])
    results = []
    for s in stories:
        sid = s.get("id", "")
        if (sid, tag_key) in reviewed:
            continue
        text = s.get("full_text", "") or ""
        if not text:
            continue
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            results.append(s)
    if shuffle:
        random.Random(42).shuffle(results)
    return results


def build_prompt(tag_key: str, story: dict) -> str:
    desc = TAG_DESCRIPTIONS.get(tag_key, tag_key)
    text = story_text_sample(story)
    return f"""You are classifying a story for a tagging system. Your job is to decide whether the tag applies.

TAG: {tag_key}
DEFINITION: {desc}

STORY EXCERPT:
{text}

Does this story genuinely exhibit the tag "{tag_key}" as described above?

Respond with EXACTLY ONE of these labels (no explanation):
TRUE_POSITIVE   — yes, this tag clearly applies to this story
FALSE_POSITIVE  — no, this tag does not apply (keyword matched but concept is absent)
AMBIGUOUS       — borderline or unclear"""


def call_llm(client, prompt: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip().upper()
    if "TRUE_POSITIVE" in raw:
        return "TRUE_POSITIVE"
    elif "FALSE_POSITIVE" in raw:
        return "FALSE_POSITIVE"
    elif "AMBIGUOUS" in raw:
        return "AMBIGUOUS"
    else:
        return "AMBIGUOUS"  # fallback for unexpected responses


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-assisted TP/TN labeling for SetFit training")
    parser.add_argument("--run",   action="store_true", help="Actually call the API (default: dry-run)")
    parser.add_argument("--tag",   type=str, default="",  help="Process a single tag only")
    parser.add_argument("--limit", type=int, default=0,   help="Max API calls per tag (0 = until TP target met)")
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args()

    stories, stories_by_id = load_stories()
    print(f"Loaded {len(stories):,} stories")

    _, reviewed, counts = load_results()
    print(f"Loaded {sum(1 for _ in reviewed)} reviewed entries\n")

    tags_to_run = [args.tag] if args.tag else TARGET_TAGS

    if args.run:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            return
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None
        print("DRY RUN — pass --run to call the API\n")

    total_calls = 0
    total_tp    = 0

    for tag_key in tags_to_run:
        c = counts[tag_key]
        need = max(0, TP_TARGET - c["TP"])
        if need == 0:
            print(f"[DONE] {tag_key} — already has {c['TP']} TPs")
            continue

        candidates = find_candidates(stories, tag_key, reviewed)
        print(f"\n{'='*60}")
        print(f"Tag:        {tag_key}")
        print(f"TPs so far: {c['TP']} / {TP_TARGET}  (need {need} more)")
        print(f"TNs so far: {c['FP']}")
        print(f"Candidates: {len(candidates):,}")

        if not candidates:
            print("  No unreviewed candidates — skipping.")
            continue

        if not args.run:
            print(f"  Would label up to {min(len(candidates), need + 20)} candidates")
            continue

        dim, tag = tag_key.split(":", 1)
        tag_tp = 0
        tag_calls = 0
        # Aim to label enough candidates to find `need` TPs.
        # Budget = need * 5 to handle low TP rates (~20%) without running dry.
        # Always collect at least need + 50 to build a useful TN pool too.
        budget = args.limit if args.limit else min(len(candidates), max(need * 5, need + 50))

        for story in candidates[:budget]:
            if tag_tp >= need:
                break

            sid = story.get("id", "")
            prompt = build_prompt(tag_key, story)

            try:
                label = call_llm(client, prompt)
            except Exception as e:
                print(f"  API error for {sid}: {e}")
                time.sleep(2)
                continue

            row = {
                "reviewed_at":   datetime.now(timezone.utc).isoformat(),
                "story_id":      sid,
                "title":         story.get("title"),
                "author":        story.get("author"),
                "tag":           tag_key,
                "dimension":     dim,
                "tag_value":     tag,
                "label":         label,
                "sources":       ["auto_label"],
                "svengali_tags": (story.get("auto_tags") or {}).get("_svengali_tags") or [],
                "source_filter": "auto_label",
                "source_mode":   "auto_label",
            }
            append_result(row)
            reviewed.add((sid, tag_key))
            counts[tag_key][{"TRUE_POSITIVE":"TP","FALSE_POSITIVE":"FP","AMBIGUOUS":"AMB"}.get(label,"AMB")] += 1

            tag_calls += 1
            total_calls += 1
            if label == "TRUE_POSITIVE":
                tag_tp += 1
                total_tp += 1

            print(f"  [{label:14}] {sid[:40]:40}  TP={counts[tag_key]['TP']}/{TP_TARGET}", flush=True)
            time.sleep(RATE_LIMIT_DELAY)

        print(f"  Tag done: {tag_calls} calls, {tag_tp} new TPs, total TPs={counts[tag_key]['TP']}")

    print(f"\n{'='*60}")
    print(f"Total API calls: {total_calls}")
    print(f"Total new TPs:   {total_tp}")
    print(f"\nFinal TP counts:")
    for tk in tags_to_run:
        c = counts[tk]
        need = max(0, TP_TARGET - c["TP"])
        status = "DONE" if need == 0 else f"still need {need}"
        print(f"  {tk:55}  TP={c['TP']:2}  TN={c['FP']:2}  {status}")


if __name__ == "__main__":
    main()
