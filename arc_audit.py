#!/usr/bin/env python3
"""
arc_audit.py — Audit narrative arc tag accuracy by comparing Dolphin's output

Status: research/audit tool, not part of the regular pipeline (see ARCHITECTURE.md).
Measures narrative_arc.py's accuracy against Claude; writes arc_audit_results.jsonl.
against Claude Haiku under two context conditions.

For each sampled story, Claude assigns arc tags independently (blind — it does
NOT see Dolphin's tags) under two conditions:

  narrow  — same 800-word opening + 600-word ending Dolphin saw
  wide    — up to WIDE_WORDS words of the full story

This gives three sets of tags per story:
  dolphin   — what narrative_arc.py assigned
  claude_narrow — Claude with the same limited context
  claude_wide   — Claude with much more context

Agreement rates reveal two things:
  1. How accurate Dolphin is (dolphin vs claude_wide, the "gold standard")
  2. Whether limited context is the bottleneck (claude_narrow vs claude_wide)

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python arc_audit.py --sample 200
  python arc_audit.py --sample 200 --wide-words 8000
  python arc_audit.py --results-only   # print report from saved results
"""

from __future__ import annotations

import json
import random
import argparse
import logging
import time
from pathlib import Path
from collections import defaultdict

log = logging.getLogger(__name__)

INDEX_PATH   = "./output/stories_index.json"
RESULTS_PATH = "./output/arc_audit_results.jsonl"

# Context window sizes
NARROW_OPENING = 800    # words — matches old narrative_arc.py SAMPLE_WORDS
NARROW_ENDING  = 600    # words — matches old narrative_arc.py ENDING_WORDS
MEDIUM_WORDS   = 10000  # words — matches new narrative_arc.py MAX_STORY_WORDS
WIDE_WORDS     = 100000 # words — gold standard (Claude Haiku 200k ctx ~= 140k words max)


# ── Import schema from narrative_arc ──────────────────────────────────────────

from narrative_arc import ARC_SCHEMA, parse_arc_response


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(story: dict, opening: str, ending: str, context_label: str) -> str:
    title   = story.get("title", story.get("id", "Unknown"))
    author  = story.get("author", "Unknown")
    at      = story.get("auto_tags") or {}
    cats    = " ".join(story.get("categories", []))
    themes  = ", ".join(at.get("themes", []))
    mc_type = ", ".join(at.get("mind_control_type", []))

    options = "\n".join(
        f'  "{k}": one of {json.dumps(v)}'
        for k, v in ARC_SCHEMA.items()
    )

    ending_block = f"\nENDING:\n{ending}" if ending and context_label == "narrow" else ""

    return f"""You are analysing a mind control erotica story to classify its narrative arc.
Story: "{title}" by {author}
Site tags: {cats} | Themes: {themes} | MC method: {mc_type}

TEXT ({context_label} context):
{opening}
{ending_block}

Classify this story using ONLY the exact allowed values listed.
Respond with ONLY a valid JSON object. No explanation, no markdown.

Fields and allowed values:
{options}

JSON response:"""


def _sample_words(words: list, limit: int) -> str:
    """Return up to limit words sampled from beginning, middle, and end."""
    if len(words) <= limit:
        return " ".join(words)
    third   = limit // 3
    opening = " ".join(words[:third])
    middle  = " ".join(words[len(words)//2 - third//2 : len(words)//2 + third//2])
    ending  = " ".join(words[-third:])
    return opening + "\n\n[... omitted ...]\n\n" + middle + "\n\n[... omitted ...]\n\n" + ending


def extract_contexts(story: dict) -> tuple[str, str, str, str]:
    """Return (narrow_opening, narrow_ending, medium_text, wide_text)."""
    full  = story.get("full_text", "")
    words = full.split()

    narrow_opening = " ".join(words[:NARROW_OPENING])
    narrow_ending  = " ".join(words[-NARROW_ENDING:]) if len(words) > NARROW_ENDING else ""
    medium_text    = _sample_words(words, MEDIUM_WORDS)
    wide_text      = _sample_words(words, WIDE_WORDS)

    return narrow_opening, narrow_ending, medium_text, wide_text


# ── Claude client ─────────────────────────────────────────────────────────────

def call_claude(client, prompt: str, model: str = "claude-haiku-4-5-20251001") -> str | None:
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        log.warning(f"Claude call failed: {e}")
        return None


# ── Agreement helpers ─────────────────────────────────────────────────────────

def compare_tags(a: dict, b: dict) -> dict[str, bool]:
    """Return {dim: True/False} for each dimension."""
    return {dim: a.get(dim) == b.get(dim) for dim in ARC_SCHEMA}


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[dict]):
    # Only count new-format results (wide-only, no narrow context)
    results = [r for r in results if r.get("claude_wide") and not r.get("claude_narrow")]

    if not results:
        print("No results.")
        return

    # Aggregate per dimension
    dims = list(ARC_SCHEMA.keys())
    agree  = defaultdict(int)
    totals = defaultdict(int)

    for r in results:
        dolph = r.get("dolphin") or {}
        wide  = r.get("claude_wide") or {}
        if not (dolph and wide):
            continue
        for dim in dims:
            totals[dim] += 1
            if dolph.get(dim) == wide.get(dim):
                agree[dim] += 1

    valid = totals.get(dims[0], 0)
    print(f"\n{'='*52}")
    print(f"  Arc Audit Report — {valid} stories evaluated")
    print(f"  Claude = full story context (up to {WIDE_WORDS:,} words)")
    print(f"{'='*52}")
    print(f"\n  Dolph↔Claude = Dolphin accuracy vs Claude gold standard\n")

    header = f"  {'Dimension':<24}  {'Dolph↔Claude':>14}"
    print(header)
    print("  " + "-" * 42)

    for dim in dims:
        t  = totals[dim] or 1
        dw = 100 * agree[dim] / t
        flag = "  ← low" if dw < 70 else ""
        print(f"  {dim:<24}  {dw:>12.1f}%{flag}")

    print()

    # Per-dimension disagreement examples
    print(f"  Sample disagreements (Dolphin vs Claude-wide):\n")
    shown = defaultdict(int)
    for r in results:
        dolph = r.get("dolphin") or {}
        wide  = r.get("claude_wide") or {}
        if not (dolph and wide):
            continue
        for dim in dims:
            if dolph.get(dim) != wide.get(dim) and shown[dim] < 2:
                print(f"    [{dim}] story: {r['story_id']}")
                print(f"      Dolphin : {dolph.get(dim)}")
                print(f"      Claude  : {wide.get(dim)}")
                shown[dim] += 1
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    if args.results_only:
        results = [json.loads(l) for l in Path(RESULTS_PATH).read_text().splitlines() if l.strip()]
        print_report(results)
        return

    import anthropic
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")
    client = anthropic.Anthropic(api_key=api_key)

    log.info(f"Loading {INDEX_PATH}...")
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Only stories with arc tags and full text
    candidates = [
        s for s in data["stories"]
        if (s.get("auto_tags") or {}).get("narrative_arc")
        and s.get("full_text")
        and not s.get("fetch_error")
    ]
    log.info(f"  {len(candidates)} arc-tagged stories available")

    # Skip already audited
    audited_ids: set[str] = set()
    results_path = Path(RESULTS_PATH)
    results_path.parent.mkdir(exist_ok=True)
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    # Only skip new-format results (wide-only, no narrow)
                    if r.get("claude_wide") and not r.get("claude_narrow"):
                        audited_ids.add(r["story_id"])
                except Exception:
                    pass
        log.info(f"  {len(audited_ids)} already audited — skipping")
        candidates = [s for s in candidates if s["id"] not in audited_ids]

    random.seed(args.seed)
    sample = random.sample(candidates, min(args.sample, len(candidates)))
    log.info(f"  Auditing {len(sample)} stories")

    # Estimate cost (Haiku: $0.80/M input, $4/M output)
    avg_tokens   = min(3000, WIDE_WORDS) * 1.35 + 600
    total_input  = len(sample) * avg_tokens
    total_output = len(sample) * 350
    cost = (total_input * 0.80 + total_output * 4.00) / 1_000_000
    log.info(f"  Estimated cost: ${cost:.2f} (Haiku)")

    with open(results_path, "a", encoding="utf-8") as out:
        for i, story in enumerate(sample, 1):
            sid   = story["id"]
            title = story.get("title", sid)
            log.info(f"[{i}/{len(sample)}] {sid} ({story.get('word_count', 0):,} words)")

            _, _, _, wide_text = extract_contexts(story)

            # ── Wide context (gold standard — full story up to WIDE_WORDS) ──
            prompt_wide = build_prompt(story, wide_text, "", "wide")
            resp_wide   = call_claude(client, prompt_wide)
            tags_wide   = parse_arc_response(resp_wide)

            dolphin_tags = story["auto_tags"].get("narrative_arc", {})

            result = {
                "story_id":    sid,
                "title":       title,
                "word_count":  story.get("word_count", 0),
                "dolphin":     dolphin_tags,
                "claude_wide": tags_wide,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(1)

            if tags_wide:
                dw = sum(1 for d in ARC_SCHEMA if dolphin_tags.get(d) == tags_wide.get(d))
                log.info(f"  Dolph↔Claude: {dw}/{len(ARC_SCHEMA)}")
            else:
                log.warning(f"  Parse failed")

    # Print final report
    all_results = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    print_report(all_results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit narrative arc tag accuracy")
    p.add_argument("--sample",       type=int, default=200,
                   help="Number of stories to audit (default: 200)")
    p.add_argument("--wide-words",   type=int, default=WIDE_WORDS,
                   help=f"Words for wide context (default: {WIDE_WORDS})")
    p.add_argument("--seed",         type=int, default=42,
                   help="Random seed for sampling (default: 42)")
    p.add_argument("--results-only", action="store_true",
                   help="Print report from saved results without running new calls")
    p.add_argument("--index",        default=INDEX_PATH)
    args = p.parse_args()

    INDEX_PATH  = args.index
    WIDE_WORDS  = args.wide_words

    run(args)
