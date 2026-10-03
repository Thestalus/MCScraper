#!/usr/bin/env python3
"""
extract_chatbot_data.py — Extract conversation training data from MC stories
for chatbot / SillyTavern RP fine-tuning.

Story format: HTML prose with embedded dialogue (curly or straight quotes).
Output: output/chat_data.jsonl in ChatML conversation format.

Two extraction strategies are combined:
  1. scene_pairs   — Use N paragraphs as "user" context, the next block as
                     "assistant" response. Teaches reactive scene continuation.
  2. dialogue_runs — Extract consecutive quoted exchanges and frame them as
                     alternating user/assistant turns. Teaches direct dialogue.

Usage:
  python extract_chatbot_data.py                  # default output
  python extract_chatbot_data.py --min-pairs 3    # minimum pairs per conversation
  python extract_chatbot_data.py --limit 5000     # cap stories processed
  python extract_chatbot_data.py --stats          # show stats only, no output
"""

from __future__ import annotations

import argparse
import json
import re
import random
import sys
from pathlib import Path

from dataset_utils import is_validation, write_manifest

# ── Config ─────────────────────────────────────────────────────────────────────

INDEX_PATH  = Path("output/stories_index.json")
OUTPUT_PATH = Path("output/chat_data.jsonl")
PREFS_PATH  = Path("output/preferences.json")

# Minimum word count — very short stories have too little dialogue
MIN_WORDS = 1000

# Minimum quoted speech exchanges in a story to include it
MIN_DIALOGUE_LINES = 6

# Target turns per conversation (user + assistant pairs)
MIN_PAIRS = 2
MAX_PAIRS = 8

# Scene window size in paragraphs
SCENE_WINDOW = 3       # paragraphs per "user" turn
SCENE_RESPONSE = 2     # paragraphs per "assistant" turn

# Weight of liked/picked stories
LIKED_WEIGHT = 3
PICK_WEIGHT  = 2

# Reproducibility + held-out validation (see dataset_utils.py). Conversations
# from one story all go to the same side, so validation measures unseen stories.
DATASET_SEED = 42
VAL_FRACTION = 0.05
VAL_PATH     = Path("output/chat_val.jsonl")

# Dominant language signals — these paragraphs become USER turns (the player is dominant)
DOMINANT_SIGNALS = re.compile(
    r'\b(hypnot|trance|sleep|obey|command|compel|control|relax|focus|listen|'
    r'you will|you must|close your eyes|feel yourself|surrender|submit|'
    r'follow my voice|trigger|suggestion|deep|deeper|drift)\b',
    re.IGNORECASE
)

# Submissive/subject language signals — these paragraphs become ASSISTANT turns
SUBMISSIVE_SIGNALS = re.compile(
    r'\b(couldn\'t|can\'t resist|helpless|drifted?|surrendered?|obeyed?|yielded?|'
    r'mind went blank|felt herself|felt himself|felt myself|sank deeper|'
    r'melted|please|i can\'t|my mind|i felt|i tried|couldn\'t think|'
    r'body moved|felt dizzy|felt the|compliance|compelled|going under|'
    r'losing control|no longer|wasn\'t sure|struggled to)\b',
    re.IGNORECASE
)

SYSTEM_PROMPT = (
    "You are the subject in an erotic mind control roleplay. You respond as the person "
    "being hypnotized or controlled — vulnerable, reactive, and increasingly yielding. "
    "Speak in first person. Show internal conflict, physical sensation, and gradual "
    "submission. React directly to what the other person says or does. Your responses "
    "are 2–5 sentences. Never narrate in third person, never break character."
)


# ── HTML stripping ──────────────────────────────────────────────────────────────

_SKIP_CLASSES = re.compile(r'storyCodes|byline|docAuthor|milestone')
_TAG_RE       = re.compile(r'<[^>]+>')
_WS_RE        = re.compile(r'[ \t]+')

def strip_html(html: str) -> str:
    """Strip HTML tags, preserving paragraph breaks. Removes metadata headers."""
    # Remove script/style blocks
    html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL|re.IGNORECASE)
    # Remove known metadata elements (story codes, byline)
    html = re.sub(r'<[^>]+class="[^"]*(?:storyCodes|byline|docAuthor|milestone)[^"]*"[^>]*>.*?</[a-z]+>',
                  '', html, flags=re.DOTALL|re.IGNORECASE)
    # Convert block-level tags to newlines
    html = re.sub(r'<(?:p|div|h[1-6]|br|li|tr)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(?:p|div|h[1-6]|li|tr)>', '\n', html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = _TAG_RE.sub('', html)
    # Normalise whitespace
    lines = []
    for line in html.split('\n'):
        line = _WS_RE.sub(' ', line).strip()
        if line:
            lines.append(line)
    return '\n'.join(lines)


# ── Paragraph splitting ─────────────────────────────────────────────────────────

def split_paragraphs(text: str) -> list[str]:
    """Split clean text into non-empty paragraphs (single or double newlines)."""
    paras = []
    for chunk in re.split(r'\n+', text):
        chunk = chunk.strip()
        if len(chunk) > 30:
            paras.append(chunk)
    return paras


# ── Dialogue extraction ─────────────────────────────────────────────────────────

# Matches curly or straight quoted speech of 5–300 chars
# U+201C/201D = curly double quotes  U+2018/2019 = curly single quotes
_QUOTE_RE = re.compile(r'[“‘](.*?)[”’]|"([^"]{5,300})"')

def extract_quotes(para: str) -> list[str]:
    """Return all quoted speech strings from a paragraph."""
    quotes = []
    for m in _QUOTE_RE.finditer(para):
        q = (m.group(1) or m.group(2) or '').strip()
        if 5 < len(q) < 300 and not q.startswith('http'):
            quotes.append(q)
    return quotes

def para_has_dialogue(para: str) -> bool:
    return bool(extract_quotes(para))

def para_is_dominant(para: str) -> bool:
    return bool(DOMINANT_SIGNALS.search(para))

def para_is_submissive(para: str) -> bool:
    return bool(SUBMISSIVE_SIGNALS.search(para))


# ── Strategy 1: Scene pairs ─────────────────────────────────────────────────────

def make_scene_pairs(paras: list[str], story_tags: str) -> list[dict]:
    """
    Sliding window over paragraphs: SCENE_WINDOW paragraphs → "user" (dominant player),
    next SCENE_RESPONSE paragraphs → "assistant" (submissive subject).
    Only generate where the assistant block contains submissive/subject reactions.
    """
    records = []
    step = max(1, SCENE_WINDOW)
    i = 0
    while i + SCENE_WINDOW + SCENE_RESPONSE <= len(paras):
        user_block = paras[i : i + SCENE_WINDOW]
        asst_block = paras[i + SCENE_WINDOW : i + SCENE_WINDOW + SCENE_RESPONSE]

        user_text = '\n\n'.join(user_block)
        asst_text = '\n\n'.join(asst_block)

        # Only include if assistant block has submissive reactions or dialogue
        asst_joined = '\n'.join(asst_block)
        if (para_is_submissive(asst_joined) or para_has_dialogue(asst_joined)):
            records.append({
                "role_user": user_text,
                "role_asst": asst_text,
            })

        i += step

    if not records:
        return []

    # Build a single multi-turn conversation from the pairs
    # (cap at MAX_PAIRS pairs)
    sampled = records[:MAX_PAIRS]
    if len(sampled) < MIN_PAIRS:
        return []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for pair in sampled:
        messages.append({"role": "user",      "content": pair["role_user"]})
        messages.append({"role": "assistant", "content": pair["role_asst"]})

    return [{"messages": messages}]


# ── Strategy 2: Dialogue runs ───────────────────────────────────────────────────

def make_dialogue_runs(paras: list[str]) -> list[dict]:
    """
    Find runs of consecutive dialogue-heavy paragraphs and frame them as
    alternating user/assistant turns. Dominant speech → user, submissive reactions → assistant.
    """
    # Collect dialogue paragraphs
    dialogue_paras = [(i, p) for i, p in enumerate(paras) if para_has_dialogue(p)]
    if len(dialogue_paras) < MIN_PAIRS * 2:
        return []

    # Find consecutive runs (within 2 paragraphs of each other)
    runs = []
    current_run = [dialogue_paras[0]]
    for prev, curr in zip(dialogue_paras, dialogue_paras[1:]):
        if curr[0] - prev[0] <= 2:
            current_run.append(curr)
        else:
            if len(current_run) >= MIN_PAIRS * 2:
                runs.append(current_run)
            current_run = [curr]
    if len(current_run) >= MIN_PAIRS * 2:
        runs.append(current_run)

    records = []
    for run in runs:
        # Trim to MAX_PAIRS * 2 paragraphs
        run = run[:MAX_PAIRS * 2]
        if len(run) < MIN_PAIRS * 2:
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Assign roles: dominant speech → user (player), submissive reactions → assistant
        # Fallback: alternate starting with user (dominant acts first)
        for k, (_, para) in enumerate(run):
            if para_is_dominant(para):
                role = "user"
            elif para_is_submissive(para):
                role = "assistant"
            else:
                role = "user" if k % 2 == 0 else "assistant"
            # Avoid two consecutive same roles
            if messages[-1]["role"] == role:
                role = "assistant" if role == "user" else "user"
            messages.append({"role": role, "content": para})

        # Ensure it ends on assistant
        if messages[-1]["role"] != "assistant":
            messages.pop()
        if len(messages) < MIN_PAIRS * 2 + 1:  # +1 for system
            continue

        records.append({"messages": messages})

    return records


# ── Main extraction ─────────────────────────────────────────────────────────────

def process_story(story: dict) -> list[dict]:
    """Extract all conversation records from one story."""
    html = story.get("full_text", "")
    if not html:
        return []

    text  = strip_html(html)
    paras = split_paragraphs(text)
    if len(paras) < MIN_PAIRS * 2 + 2:
        return []

    # Count dialogue lines to filter low-dialogue stories
    total_dialogue = sum(1 for p in paras if para_has_dialogue(p))
    if total_dialogue < MIN_DIALOGUE_LINES:
        return []

    # Build a brief tag string for context (not injected into prompts,
    # kept for stats/debugging only)
    at   = story.get("auto_tags") or {}
    cats = ", ".join(story.get("categories", []))

    records = []
    records.extend(make_scene_pairs(paras, cats))
    records.extend(make_dialogue_runs(paras))
    return records


def load_preferences() -> tuple[set, set]:
    """(liked story IDs, disliked story IDs) from preferences.json; empty sets if the file
    is missing.
    """
    if not PREFS_PATH.exists():
        return set(), set()
    prefs = json.loads(PREFS_PATH.read_text())
    return set(prefs.get("liked_stories", [])), set(prefs.get("disliked_stories", []))


def main():
    """Filter usable stories (text, length, not disliked), turn each into chat
    conversations, put validation stories aside, weight liked ones, and write
    chat_data.jsonl, chat_val.jsonl and a manifest. --stats reports without writing.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit",     type=int, default=None, help="Cap stories processed")
    ap.add_argument("--min-pairs", type=int, default=MIN_PAIRS, help="Min turn pairs per record")
    ap.add_argument("--stats",     action="store_true", help="Show stats only, no output")
    ap.add_argument("--output",    default=str(OUTPUT_PATH), help="Output JSONL path")
    args = ap.parse_args()

    if not INDEX_PATH.exists():
        print(f"ERROR: {INDEX_PATH} not found. Run the scraper first.")
        sys.exit(1)

    print(f"Loading {INDEX_PATH}...")
    data    = json.loads(INDEX_PATH.read_text())
    stories = data.get("stories", [])
    print(f"  {len(stories):,} stories in index")

    liked_ids, disliked_ids = load_preferences()

    # Filter
    usable = [
        s for s in stories
        if s.get("full_text")
        and not s.get("fetch_error")
        and s.get("word_count", 0) >= MIN_WORDS
        and s.get("id") not in disliked_ids
    ]
    print(f"  {len(usable):,} usable stories (have text, ≥{MIN_WORDS} words, not disliked)")

    random.seed(DATASET_SEED)   # shuffles below are reproducible
    random.shuffle(usable)
    if args.limit:
        usable = usable[:args.limit]

    # Extract
    all_records, val_records = [], []
    val_stories = 0
    skipped_low_dialogue = 0
    for story in usable:
        recs = process_story(story)
        if not recs:
            skipped_low_dialogue += 1
            continue

        sid = story.get("id", "")
        if is_validation(sid, VAL_FRACTION, DATASET_SEED):
            val_records.extend(recs)      # once, unweighted — it's a measurement
            val_stories += 1
            continue

        # Apply preference weighting by repeating records
        w   = 1
        if sid in liked_ids:
            w = max(w, LIKED_WEIGHT)
        if story.get("readers_pick"):
            w = max(w, PICK_WEIGHT)

        for _ in range(w):
            all_records.extend(recs)

    print(f"\nExtraction results:")
    print(f"  Stories processed:       {len(usable):,}")
    print(f"  Skipped (low dialogue):  {skipped_low_dialogue:,}")
    print(f"  Conversations extracted: {len(all_records):,}")

    if not all_records:
        print("\nNo records extracted. Check MIN_DIALOGUE_LINES or MIN_PAIRS settings.")
        sys.exit(1)

    # Turn count distribution
    turn_counts = [len([m for m in r["messages"] if m["role"] != "system"])
                   for r in all_records]
    avg_turns = sum(turn_counts) / len(turn_counts)
    print(f"  Avg turns per conv:      {avg_turns:.1f}")
    print(f"  Total turns:             {sum(turn_counts):,}")

    if args.stats:
        print("\n(--stats mode: no file written)")
        return

    random.shuffle(all_records)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_manifest(
        out.with_suffix(".manifest.json"),
        outputs={"train": out, "val": VAL_PATH},
        inputs={"index": INDEX_PATH, "preferences": PREFS_PATH},
        settings={"dataset_seed": DATASET_SEED, "val_fraction": VAL_FRACTION,
                  "min_words": MIN_WORDS, "min_dialogue_lines": MIN_DIALOGUE_LINES,
                  "pairs": [MIN_PAIRS, MAX_PAIRS], "limit": args.limit},
        counts={"train_records": len(all_records), "val_records": len(val_records),
                "val_stories": val_stories},
    )

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nWritten to {out}  ({size_mb:.1f} MB)")
    print(f"Validation: {val_stories:,} held-out stories, {len(val_records):,} conversations → {VAL_PATH}")
    print(f"\nNext: python train_chat.py --train")


if __name__ == "__main__":
    main()
