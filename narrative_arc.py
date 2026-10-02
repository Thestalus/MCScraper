#!/usr/bin/env python3
"""
narrative_arc.py — LLM-powered narrative arc detection for mcstories dataset

Detects story dimensions that regex cannot capture because they require
understanding the whole story arc, not just keyword matching:

  - ending_state:      how the subject ends up (willing, unwilling, neutral, ambiguous)
  - relationship_type: nature of the dominant/subject relationship at the end
  - twist:             whether the story has a reversal or surprise
  - resolution:        whether the story feels complete or open-ended
  - subject_change:    how much the subject changes from start to end
  - power_shift:       whether the power dynamic shifts during the story
  - emotional_tone_end: the emotional register of the final scene

These complement your existing auto_tags and significantly improve
recommendation quality — "stories where she ends up happy" is something
readers care about deeply but regex can't detect.

REQUIRES: A running local LLM (LM Studio or Ollama) — run after you have
your fine-tuned model working. All detection is free once you have the model.

Usage:
  # Detect arcs for all stories without arc tags (~4-6 hours for 17k stories)
  python narrative_arc.py

  # Test on 10 stories first
  python narrative_arc.py --limit 10

  # Use a specific endpoint
  python narrative_arc.py --url http://localhost:1234/v1 --model auto

  # Re-run only stories that failed last time
  python narrative_arc.py --failed-only

  # After running, rebuild JSONL training files with new tags
  python narrative_arc.py --rebuild-jsonl
"""

from __future__ import annotations

import json
import signal
import time
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH   = "./output/stories_index.json"
OUTPUT_PATH  = "./output/stories_index.json"   # overwrites in place
LOG_PATH     = "./output/arc_detection.log"
BATCH_SIZE   = 1       # process one story at a time (keeps memory low)
DELAY        = 0.5     # seconds between requests (local model, can be fast)
MAX_RETRIES  = 2
MAX_STORY_WORDS = 10000  # send full story up to this limit; sample beginning/middle/end beyond
SAMPLE_WORDS    = 800    # fallback opening if full-text not available
ENDING_WORDS    = 600    # fallback ending if full-text not available


# ── Arc taxonomy ──────────────────────────────────────────────────────────────

ARC_SCHEMA = {

    # ── Story status ──────────────────────────────────────────────────────────
    "story_status": [
        "complete",                # story has a satisfying ending
        "incomplete",              # ends abruptly, unresolved, or mid-arc
        "open / ongoing",          # intentionally open-ended, series format
        "cliffhanger",             # ends on a dramatic unresolved moment
    ],

    # ── Ending state ──────────────────────────────────────────────────────────
    "ending_state": [
        "fully willing",           # subject embraces their new state happily
        "contentedly submissive",  # accepts and is at peace with submission
        "resigned",                # accepts but not happily — a kind of defeat
        "still resisting",         # still fighting at the end
        "broken",                  # resistance destroyed, hollow shell
        "stepfordized",            # outwardly perfect and compliant, inwardly gone
        "unaware",                 # still oblivious at the end
        "rescued / restored",      # subject was freed or returned to normal
        "escaped",                 # subject got away under their own power
        "ambiguous",               # unclear / up to interpretation
    ],

    # ── Subject change ────────────────────────────────────────────────────────
    "subject_change": [
        "complete transformation", # personality, values, identity all altered
        "bimboized",               # intelligence and agency reduced to sexuality
        "stepfordized",            # turned into a perfectly obedient domestic
        "drone / blank",           # personality removed, empty vessel
        "significant change",      # clearly different but core self traces remain
        "subtle change",           # small shifts, recognisably the same person
        "no change",               # control failed, wore off, or wasn't deep
        "changed then restored",   # transformed but eventually returned to normal
    ],

    # ── Twist ─────────────────────────────────────────────────────────────────
    "twist": [
        "reversal",                # dominant loses control or is themselves controlled
        "revelation",              # something fundamental about the setup is revealed
        "identity",                # a character is not who they appeared to be
        "consent",                 # non-con revealed as consensual or vice versa
        "the controller controlled", # the person doing the controlling gets a taste
        "no twist",
    ],

    # ── Relationship type ─────────────────────────────────────────────────────
    "relationship_type": [
        "romantic",                # ends as a loving relationship
        "ownership / slave",       # subject is property, collar and chain dynamic
        "worshipful",              # subject worships the dominant as a god/goddess
        "parental / protective",   # dominant is caring, nurturing, protective
        "clinical / professional", # detached, transactional, subject is a specimen
        "obsessive",               # dominant is dangerously fixated
        "adversarial",             # remains hostile even with control established
        "harem / collection",      # subject joins a group of others already controlled
        "ambiguous",
    ],

    # ── Emotional tone at ending ──────────────────────────────────────────────
    "emotional_tone_end": [
        "warm / happy",
        "romantic / tender",
        "melancholic / bittersweet",
        "dark / disturbing",
        "erotic / aroused",
        "tense / unresolved",
        "peaceful / satisfied",
        "hollow / empty",
        "triumphant",
    ],

    # ── Dominant character type ───────────────────────────────────────────────
    "dominant_type": [
        "authority figure",        # boss, doctor, teacher, therapist, coach
        "romantic partner",        # spouse, lover, established relationship
        "stranger",                # no prior relationship at story start
        "supernatural being",      # vampire, demon, god, fae, spirit, alien
        "AI / machine / device",   # computer, robot, app, technological entity
        "family member",           # parent, sibling, relative — incest dynamic
        "friend / peer",           # same social level, pre-existing friendship
        "organisation / group",    # cult, corporation, government, secret society
        "stage hypnotist",         # public performance context
        "self",                    # self-hypnosis, self-inflicted
    ],
}

# ── Subject gender — derived from story categories, no LLM needed ─────────────
# mf / fd → female subject; mm / md → male subject; ff → female subject
_FEMALE_SUBJECT_CATS = {"mf", "fd", "ff"}
_MALE_SUBJECT_CATS   = {"mm", "md"}

def derive_subject_gender(categories: list) -> str:
    """Derive subject gender from mcstories category codes."""
    cats = {c.lower() for c in categories}
    has_f = bool(cats & _FEMALE_SUBJECT_CATS)
    has_m = bool(cats & _MALE_SUBJECT_CATS)
    if has_f and has_m:
        return "multiple subjects - mixed"
    if has_f:
        return "female subject"
    if has_m:
        return "male subject"
    return "ambiguous"

# ── LLM client ────────────────────────────────────────────────────────────────

import requests

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


def detect_model(base_url: str) -> str:
    """Get the first available model from the endpoint."""
    try:
        r = session.get(f"{base_url}/models", timeout=5)
        models = r.json().get("data", [])
        return models[0]["id"] if models else "local-model"
    except:
        return "local-model"


def call_llm(base_url: str, model: str, prompt: str,
             max_tokens: int = 300, temperature: float = 0.1) -> Optional[str]:
    """Call the LLM and return the response text."""
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(
                f"{base_url}/chat/completions",
                json={
                    "model":       model,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  max_tokens,
                    "temperature": temperature,
                },
                timeout=300,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"  LLM attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            time.sleep(1)
    return None


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_arc_prompt(story: dict) -> str:
    """
    Build a prompt that asks the LLM to classify the story's narrative arc.
    Sends the full story up to MAX_STORY_WORDS; falls back to opening+ending
    for very long stories.
    """
    title   = story.get("title", story.get("id", "Unknown"))
    author  = story.get("author", "Unknown")
    full    = story.get("full_text", "")
    words   = full.split()
    at      = story.get("auto_tags") or {}
    cats    = " ".join(story.get("categories", []))
    themes  = ", ".join(at.get("themes", []))
    mc_type = ", ".join(at.get("mind_control_type", []))

    if len(words) <= MAX_STORY_WORDS:
        story_text = full
        context_note = f"FULL STORY ({len(words)} words):"
    else:
        third = MAX_STORY_WORDS // 3
        opening = " ".join(words[:third])
        middle  = " ".join(words[len(words)//2 - third//2 : len(words)//2 + third//2])
        ending  = " ".join(words[-third:])
        story_text = (opening
                      + "\n\n[... omitted ...]\n\n"
                      + middle
                      + "\n\n[... omitted ...]\n\n"
                      + ending)
        context_note = f"STORY EXCERPT (opening/middle/ending, ~{third} words each, {len(words)} total):"

    options = "\n".join(
        f'  "{k}": one of {json.dumps(v)}'
        for k, v in ARC_SCHEMA.items()
    )

    return f"""You are analysing a mind control erotica story to classify its narrative arc and key attributes.
Story: "{title}" by {author}
Site tags: {cats} | Auto-detected themes: {themes} | MC method: {mc_type}

{context_note}
{story_text}

Classify this story by filling in ALL fields below.
Use ONLY the exact allowed values listed for each field.
Respond with ONLY a valid JSON object. No explanation, no markdown, no extra text.

Key guidance:
- story_status: is this story complete or does it end abruptly/mid-arc?
- ending_state: how does the SUBJECT end up at the story's conclusion?
- subject_change: how much has the subject changed from who they were at the start?
- dominant_type: what is the DOMINANT character's role/identity in relation to the subject?

Fields and allowed values:
{options}

JSON response:"""


def parse_arc_response(response: str) -> Optional[dict]:
    """Parse the LLM's JSON response, handling common formatting issues."""
    if not response:
        return None

    # Strip markdown code fences if present
    response = re.sub(r"```(?:json)?\s*", "", response).strip()
    response = re.sub(r"```\s*$", "", response).strip()

    # Find the JSON object
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return None

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Try to fix common issues — trailing commas, single quotes
        fixed = re.sub(r",\s*}", "}", m.group(0))
        fixed = fixed.replace("'", '"')
        try:
            data = json.loads(fixed)
        except:
            return None

    # Validate — only keep keys that are in our schema
    valid = {}
    for key, allowed in ARC_SCHEMA.items():
        val = data.get(key, "")
        if isinstance(val, str):
            val_lower = val.lower()
            for a in allowed:
                if val_lower == a.lower() or val_lower in a.lower():
                    valid[key] = a
                    break
            else:
                valid[key] = "ambiguous"
        else:
            valid[key] = "ambiguous"

    return valid if valid else None


# ── Main detection pipeline ───────────────────────────────────────────────────

def load_index() -> dict:
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_index(data: dict):
    tmp = Path(OUTPUT_PATH).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(Path(OUTPUT_PATH))


def run(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH),
        ]
    )

    base_url = args.url
    model    = args.model if args.model != "auto" else detect_model(base_url)
    log.info(f"Using model: {model} at {base_url}")

    # Test connection
    test = call_llm(base_url, model, 'Reply with exactly: {"ok": true}', max_tokens=20)
    if not test or "ok" not in test:
        log.error("Cannot connect to LLM. Start LM Studio and load a model first.")
        return

    log.info("LLM connection confirmed")

    # Load index
    log.info(f"Loading {INDEX_PATH}...")
    data     = load_index()
    stories  = data.get("stories", [])
    log.info(f"Total stories: {len(stories):,}")

    # Decide which stories to process
    if args.failed_only:
        to_process = [
            s for s in stories
            if not s.get("fetch_error")
            and s.get("full_text")
            and s.get("auto_tags")
            and s["auto_tags"].get("_arc_error")
        ]
        log.info(f"Processing {len(to_process)} previously failed stories")
    else:
        to_process = [
            s for s in stories
            if not s.get("fetch_error")
            and s.get("full_text")
            and s.get("auto_tags")
            and not s["auto_tags"].get("narrative_arc")  # skip already done
        ]
        log.info(f"Stories needing arc detection: {len(to_process):,}")

    if args.limit:
        to_process = to_process[:args.limit]
        log.info(f"Limited to {args.limit} stories")

    if not to_process:
        log.info("All stories already have arc tags. Nothing to do.")
        log.info("Use --failed-only to retry failed stories.")
        return

    # Build lookup for fast update
    by_id = {s["id"]: s for s in stories}

    ok = err = skipped = 0
    # Each checkpoint rewrites the whole (>1 GB) index, which the sync folder then
    # re-uploads, so checkpoint sparingly. Ctrl+C saves before exiting instead.
    checkpoint_every = args.checkpoint_every

    stop_requested = False

    def _request_stop(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt  # second Ctrl+C: quit immediately
        stop_requested = True
        log.warning("Ctrl+C — finishing this story, then saving and exiting "
                    "(press Ctrl+C again to quit without saving)")

    signal.signal(signal.SIGINT, _request_stop)

    for i, story in enumerate(to_process, 1):
        if stop_requested:
            break
        sid   = story["id"]
        title = story.get("title") or sid
        wc    = story.get("word_count", 0)

        if wc < 500:
            log.info(f"[{i}/{len(to_process)}] {sid} — too short ({wc} words), skipping")
            skipped += 1
            continue

        log.info(f"[{i}/{len(to_process)}] {sid} ({wc:,} words)")

        prompt   = build_arc_prompt(story)
        response = call_llm(base_url, model, prompt, max_tokens=400, temperature=0.1)
        arc      = parse_arc_response(response)

        if arc:
            by_id[sid]["auto_tags"]["narrative_arc"] = arc
            by_id[sid]["auto_tags"]["subject_gender"] = derive_subject_gender(
                story.get("categories", []))
            by_id[sid]["auto_tags"].pop("_arc_error", None)
            ok += 1
            log.info(f"  ending={arc.get('ending_state')} | "
                     f"dom={arc.get('dominant_type')} | "
                     f"change={arc.get('subject_change')}")
        else:
            by_id[sid]["auto_tags"]["_arc_error"] = True
            err += 1
            log.warning(f"  Failed to parse arc (raw: {response[:100] if response else 'None'})")

        # Checkpoint every N stories
        if i % checkpoint_every == 0:
            data["stories"] = list(by_id.values())
            save_index(data)
            log.info(f"  Checkpoint saved ({ok} ok, {err} failed, {skipped} skipped)")

        time.sleep(DELAY)

    # Final save
    data["stories"] = list(by_id.values())
    save_index(data)

    log.info("=" * 60)
    log.info(f"Done. {ok} detected, {err} failed, {skipped} skipped")
    log.info(f"Saved to {OUTPUT_PATH}")

    if args.rebuild:
        log.info("Rebuilding JSONL training files...")
        import subprocess
        subprocess.run(["python", "tagger.py", "--rebuild-jsonl"])


# ── Wire arc tags into browser filters ────────────────────────────────────────
# After running this, add these filter options to browser.py:
#
# ENDING_STATES     = [v for v in ARC_SCHEMA["ending_state"]]
# RELATIONSHIP_TYPES= [v for v in ARC_SCHEMA["relationship_type"]]
#
# Then in search_stories():
#   if ending_state and not (
#       s.get("auto_tags",{}).get("narrative_arc",{}).get("ending_state") == ending_state):
#       continue
#
# And in the sidebar template:
#   <select name="ending_state">
#     <option value="">Any ending</option>
#     {% for e in ending_states %}...{% endfor %}
#   </select>


# ── Stats report ──────────────────────────────────────────────────────────────

def print_stats():
    """Print a summary of arc tags in the current index."""
    data    = load_index()
    stories = [s for s in data["stories"]
               if s.get("auto_tags", {}).get("narrative_arc")]

    print(f"\nNarrative arc tags on {len(stories):,} stories\n")

    def _show_dim(dim, allowed, values):
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        total = sum(counts.values())
        print(f"{dim}:  ({total} tagged)")
        for val in allowed:
            count = counts.get(val, 0)
            pct   = 100 * count / max(len(stories), 1)
            bar   = "█" * int(pct / 2)
            print(f"  {val:<40} {count:>5}  {pct:4.1f}%  {bar}")
        unexpected = {k: v for k, v in counts.items() if k not in allowed and k != "unknown"}
        if unexpected:
            print(f"  [unexpected values: {unexpected}]")
        print()

    for dim, allowed in ARC_SCHEMA.items():
        values = [s["auto_tags"]["narrative_arc"].get(dim, "unknown") for s in stories]
        _show_dim(dim, allowed, values)

    sg_stories = [s for s in data["stories"] if s.get("auto_tags", {}).get("subject_gender")]
    sg_values  = [s["auto_tags"]["subject_gender"] for s in sg_stories]
    sg_allowed = ["female subject", "male subject", "multiple subjects - mixed", "ambiguous"]
    _show_dim(f"subject_gender (derived, {len(sg_stories)} stories)", sg_allowed, sg_values)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Detect narrative arcs in mcstories dataset using local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Finish training your model and load it in LM Studio
  2. Start LM Studio server (Local Server tab → Start)
  3. Test with a small batch first:
       python narrative_arc.py --limit 20
  4. Check the output looks right, then run the full detection:
       python narrative_arc.py
  5. Rebuild training data with new arc tags:
       python narrative_arc.py --rebuild-jsonl
  6. View statistics:
       python narrative_arc.py --stats

Speed: ~3-6 seconds per story depending on model and story length.
For 17k stories: ~15-30 hours. Run overnight in batches.
Checkpoints every 50 stories (--checkpoint-every). Ctrl+C finishes the
current story and saves before exiting.
Resume interrupted run:
  python narrative_arc.py   (skips already-tagged stories automatically)
        """
    )
    p.add_argument("--url",          default="http://localhost:1234/v1",
                   help="LLM endpoint URL (default: LM Studio)")
    p.add_argument("--model",        default="auto",
                   help="Model name (default: auto-detect)")
    p.add_argument("--limit",        type=int, default=None,
                   help="Max stories to process (for testing)")
    p.add_argument("--failed-only",  action="store_true",
                   help="Only retry stories that failed last time")
    p.add_argument("--rebuild-jsonl","--rebuild", action="store_true",
                   dest="rebuild",
                   help="Rebuild JSONL training files after detection")
    p.add_argument("--stats",        action="store_true",
                   help="Print arc tag statistics and exit")
    p.add_argument("--index",        default=INDEX_PATH,
                   help="Path to stories_index.json")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save the index every N stories (default: 50); each save rewrites the whole index")
    p.add_argument("--delay",        type=float, default=DELAY,
                   help=f"Seconds between requests (default: {DELAY})")

    args = p.parse_args()
    INDEX_PATH  = args.index
    OUTPUT_PATH = args.index
    DELAY       = args.delay

    if args.stats:
        print_stats()
    else:
        run(args)
