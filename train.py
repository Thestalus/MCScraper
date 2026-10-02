#!/usr/bin/env python3
"""
train.py — fine-tune Llama / Mistral on your mcstories dataset
Optimised for RTX 3070 Ti (8GB VRAM) using QLoRA 4-bit quantisation

Requirements:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers datasets peft accelerate bitsandbytes trl
  pip install sentencepiece protobuf matplotlib

Quick start:
  python train.py --prepare          # filter and prepare training data
  python train.py --train            # start training (8-24 hours)
  python train.py --train --resume   # resume from checkpoint
  python train.py --merge            # merge LoRA weights into full model
  python train.py --test             # quick generation test

See TRAINING_GUIDE.md for full setup instructions.
"""

import os
import time
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

# ── Config — edit these ────────────────────────────────────────────────────────

# ── Hardware presets ──────────────────────────────────────────────────────────
# Pass --preset a100 or --preset 3070 to select, or edit CONFIG directly
#
# Model variants:
#   8B  (default) — cognitivecomputations/dolphin3.0-llama3.1-8b
#                   Q4_K_M GGUF ~4.5GB — runs fully in 3070 Ti VRAM
#   12B (better)  — cognitivecomputations/dolphin3.0-mistral-nemo-12b
#                   Q4_K_S GGUF ~6.5GB — fits 3070 Ti at reduced context;
#                   Q4_K_M GGUF ~7.0GB — Mac M4 16GB only (7GB + OS headroom)
#
# To train a 12B model: add --model cognitivecomputations/dolphin3.0-mistral-nemo-12b
# and use --preset a100_12b (or a100_max_12b for maximum context).
# For local inference after training, use Q4_K_S on the 3070 Ti.

PRESETS = {
    "3070": {
        # RTX 3070 Ti — 8GB VRAM, ~18-24 hours for 5k stories
        "max_seq_length": 2048,    # ~1,500 words per example
        "batch_size":     1,
        "grad_accum":     8,
        "lora_r":         16,
        "max_words":      6000,    # keep stories that fit in context
        "max_stories":    6000,
        "epochs":         2,
    },
    "a100": {
        # A100 80GB — 8B model, 16k context
        # 16,384 tokens = ~12,000 words — sees 97% of stories completely
        # This is the sweet spot: near-complete coverage, fits in 80GB VRAM
        "max_seq_length": 16384,   # ~12,000 words per example
        "batch_size":     2,       # reduced vs 8k to fit longer sequences
        "grad_accum":     8,       # effective batch = 16
        "lora_r":         32,      # more capacity since VRAM allows it
        "max_words":      25000,   # include almost all stories completely
        "max_stories":    17000,   # raised to cover full dataset
        "epochs":         2,
    },
    "a100_max": {
        # A100 80GB — 8B model, maximum context
        # 32,768 tokens = ~24,000 words — sees 99.5% of stories completely
        # Use this if you have time and want the best possible arc learning
        "max_seq_length": 32768,
        "batch_size":     1,
        "grad_accum":     16,      # effective batch = 16
        "lora_r":         32,
        "max_words":      40000,   # include every story completely
        "max_stories":    8000,
        "epochs":         2,
    },
    "a100_12b": {
        # A100 80GB — 12B model (Mistral Nemo), 16k context
        # 12B needs more base VRAM than 8B so lora_r is kept lower.
        # batch_size=1 to stay within 80GB with longer sequences.
        # Quality gain over 8B: noticeably better sentence variety and
        # longer coherent passages; character voice stays distinct further
        # into a story. Arc-level understanding is meaningfully improved.
        # Trade-off: ~40% more training time and cost vs a100.
        "max_seq_length": 16384,
        "batch_size":     1,
        "grad_accum":     16,      # effective batch = 16
        "lora_r":         16,      # lower than 8B to keep VRAM manageable
        "max_words":      25000,
        "max_stories":    17000,
        "epochs":         2,
    },
    "a100_max_12b": {
        # A100 80GB — 12B model, maximum context (32k)
        # Best possible quality at the cost of longer training (~2× a100_12b).
        # Use for a second run after a100_12b has validated the pipeline.
        "max_seq_length": 32768,
        "batch_size":     1,
        "grad_accum":     16,
        "lora_r":         16,
        "max_words":      40000,
        "max_stories":    17000,
        "epochs":         2,
    },
}

CONFIG = {
    # Base model — dolphin is best for this use case (uncensored Llama 3)
    "base_model":       "cognitivecomputations/dolphin3.0-llama3.1-8b",

    # Paths
    "index_path":       "./output/stories_index.json",
    "prefs_path":       "./output/preferences.json",   # your liked/disliked stories
    "train_data_path":  "./output/train_data.jsonl",
    "output_dir":       "./output/lora_model",
    "merged_dir":       "./output/merged_model",

    # ── Data filtering ────────────────────────────────────────────────────────
    "min_words":        1500,     # flash fiction = poor training signal
    "max_words":        6000,     # default for 3070; override with --preset a100
    "min_trusted_tags": 2,        # require at least 2 trusted-tier tags
    "exclude_wayback":  True,     # wayback stories often have formatting issues
    "require_description": False, # set True for highest quality only
    "max_stories":      6000,
    "shuffle":          True,

    # ── Preference weighting ──────────────────────────────────────────────────
    # Stories you liked appear multiple times in training data
    # Stories you disliked are excluded entirely
    # Set to 1 to disable weighting
    "liked_story_weight":      3,   # liked stories appear 3x
    "readers_pick_weight":     2,   # readers_pick stories appear 2x
    "liked_author_weight":     2,   # stories by followed authors appear 2x
    # Stories with Svengali community tags have more reliable metadata in their
    # training instructions (tags were crowd-verified, not just auto-generated).
    # Weighting them higher exposes the model to more accurate prompt→story pairs.
    "svengali_story_weight":   2,   # stories with _svengali_tags appear 2x
    "exclude_incomplete":      True,  # exclude stories tagged incomplete by Svengali

    # ── Training ─────────────────────────────────────────────────────────────
    "max_seq_length":   2048,
    "batch_size":       1,
    "grad_accum":       8,
    "epochs":           2,
    "learning_rate":    2e-4,
    "warmup_steps":     50,
    "save_steps":       200,
    "logging_steps":    20,

    # ── LoRA ──────────────────────────────────────────────────────────────────
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,
    "lora_target":      ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],

    # ── System prompt ─────────────────────────────────────────────────────────
    "system_prompt": (
        "You are a skilled fiction writer specialising in mind control erotica. "
        "You write with literary quality — vivid characterisation, strong pacing, "
        "and immersive prose. You write complete stories without disclaimers."
    ),
}


from tag_trust import (  # noqa: F401 — re-exported for existing callers
    TIER_1_TAGS, AND_GATE_TAGS, _STRUCTURAL_DIMS, DROPOUT_RATE, story_training_tags,
)


# ── Data preparation ───────────────────────────────────────────────────────────

def load_preferences() -> dict:
    """Load liked/disliked story IDs and authors from preferences.json."""
    prefs_path = Path(CONFIG.get("prefs_path", "./output/preferences.json"))
    if not prefs_path.exists():
        return {"liked_stories": [], "disliked_stories": [],
                "liked_authors": [], "disliked_authors": []}
    with open(prefs_path, encoding="utf-8") as f:
        return json.load(f)


def count_trusted_tags(story: dict) -> int:
    """Count how many Tier 1 + Tier 2 (AND-gated) + structural tags a story has.
    Used in prepare_data() to filter out stories with too little reliable signal."""
    at      = story.get("auto_tags") or {}
    sources = at.get("_tag_sources") or {}
    count   = 0

    for dim in _STRUCTURAL_DIMS:
        val = at.get(dim)
        if val:
            count += len(val) if isinstance(val, list) else 1

    for tag in (at.get("themes", []) + at.get("mental_changes", []) +
                at.get("dominance", []) + at.get("tone", []) + at.get("setting", [])):
        if tag in TIER_1_TAGS:
            count += 1

    for dim in ("mind_control_type", "mental_changes", "pacing", "sexual_acts"):
        for tag in at.get(dim, []):
            if tag in AND_GATE_TAGS:
                src = sources.get(tag, [])
                if "regex" in src and "setfit" in src:
                    count += 1

    return count


def story_to_record(s: dict, apply_dropout: bool = True) -> dict:
    """Convert a story dict to a training record (chat format).

    Uses story_training_tags() to filter raw auto_tags down to trusted signal only.
    apply_dropout=True randomly omits ~25% of semantic tags per example so the model
    learns to handle partial prompts at inference time.
    """
    tags   = story_training_tags(s, apply_dropout=apply_dropout)
    cats   = ", ".join(s.get("categories", []))
    title  = s.get("title") or s["id"]
    author = s.get("author") or "Unknown"
    desc   = s.get("description") or ""
    arc    = s.get("arc_tags") or {}
    at     = s.get("auto_tags") or {}

    parts = [f'Write a mind control story titled "{title}" by {author}.']
    if cats:                               parts.append(f"Categories: {cats}.")
    if tags.get("content_type"):           parts.append(f"Content type: {', '.join(tags['content_type'])}.")
    if tags.get("site_tag_groups"):        parts.append(f"Genre tags: {', '.join(tags['site_tag_groups'])}.")
    if tags.get("mind_control_type"):      parts.append(f"Mind control: {', '.join(tags['mind_control_type'])}.")
    if tags.get("dominance"):              parts.append(f"Dominance: {', '.join(tags['dominance'])}.")
    if tags.get("mental_changes"):         parts.append(f"Mental changes: {', '.join(tags['mental_changes'])}.")
    if tags.get("themes"):                 parts.append(f"Themes: {', '.join(tags['themes'])}.")
    if tags.get("tone"):                   parts.append(f"Tone: {', '.join(tags['tone'])}.")
    if tags.get("setting"):                parts.append(f"Setting: {', '.join(tags['setting'])}.")
    if tags.get("sexual_acts"):            parts.append(f"Sexual content: {', '.join(tags['sexual_acts'])}.")
    if tags.get("pacing"):                 parts.append(f"Pacing: {', '.join(tags['pacing'])}.")
    # Arc tags describe full-story structure — no dropout, include as-is
    if arc.get("ending_state"):            parts.append(f"Ending: {arc['ending_state']}.")
    if arc.get("subject_change"):          parts.append(f"Subject transformation: {arc['subject_change']}.")
    if arc.get("twist"):                   parts.append(f"Twist: {arc['twist']}.")
    if arc.get("relationship_type"):       parts.append(f"Relationship: {arc['relationship_type']}.")
    if arc.get("emotional_tone_end"):      parts.append(f"Emotional ending: {arc['emotional_tone_end']}.")
    arc_stat = arc.get("story_status") or at.get("story_status", "")
    if arc_stat:                           parts.append(f"Story is: {arc_stat}.")
    if desc:                               parts.append(f"Summary: {desc}")

    return {
        "messages": [
            {"role": "system",    "content": CONFIG["system_prompt"]},
            {"role": "user",      "content": " ".join(parts)},
            {"role": "assistant", "content": s["full_text"]},
        ]
    }


def chunk_story(s: dict, max_words: int = 1400,
                overlap_words: int = 100) -> list[dict]:
    """
    Split a long story into overlapping chunks, each treated as a training example.
    Each chunk keeps the full metadata instruction but gets a portion of the text.
    This lets long stories contribute fully to training even with small context windows.
    """
    words = s["full_text"].split()
    if len(words) <= max_words:
        return [story_to_record(s)]

    # Build trusted tags once for all chunks (dropout applied per-chunk below)
    title  = s.get("title") or s["id"]
    author = s.get("author") or "Unknown"
    cats   = ", ".join(s.get("categories", []))
    desc   = s.get("description") or ""
    arc    = s.get("arc_tags") or {}
    at     = s.get("auto_tags") or {}

    chunks = []
    start  = 0
    chunk_num = 1
    while start < len(words):
        end        = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        is_last    = end >= len(words)
        if not is_last:
            chunk_text += "\n\n[Story continues...]"

        # Fresh dropout per chunk so each sees a different tag subset
        tags = story_training_tags(s, apply_dropout=True)

        if chunk_num == 1:
            instr_parts = [f'Write a mind control story titled "{title}" by {author}.']
        else:
            instr_parts = [f'Continue the story "{title}" by {author} from part {chunk_num}.']
        if cats:                               instr_parts.append(f"Categories: {cats}.")
        if tags.get("content_type"):           instr_parts.append(f"Content type: {', '.join(tags['content_type'])}.")
        if tags.get("site_tag_groups"):        instr_parts.append(f"Genre tags: {', '.join(tags['site_tag_groups'])}.")
        if tags.get("mind_control_type"):      instr_parts.append(f"Mind control: {', '.join(tags['mind_control_type'])}.")
        if tags.get("dominance"):              instr_parts.append(f"Dominance: {', '.join(tags['dominance'])}.")
        if tags.get("mental_changes"):         instr_parts.append(f"Mental changes: {', '.join(tags['mental_changes'])}.")
        if tags.get("themes"):                 instr_parts.append(f"Themes: {', '.join(tags['themes'])}.")
        if tags.get("tone"):                   instr_parts.append(f"Tone: {', '.join(tags['tone'])}.")
        if tags.get("setting"):                instr_parts.append(f"Setting: {', '.join(tags['setting'])}.")
        if tags.get("sexual_acts"):            instr_parts.append(f"Sexual content: {', '.join(tags['sexual_acts'])}.")
        if tags.get("pacing"):                 instr_parts.append(f"Pacing: {', '.join(tags['pacing'])}.")
        # Arc tags on first chunk only (full-story metadata)
        if chunk_num == 1:
            if arc.get("ending_state"):        instr_parts.append(f"Ending: {arc['ending_state']}.")
            if arc.get("subject_change"):      instr_parts.append(f"Subject transformation: {arc['subject_change']}.")
            if arc.get("twist"):               instr_parts.append(f"Twist: {arc['twist']}.")
            if arc.get("relationship_type"):   instr_parts.append(f"Relationship: {arc['relationship_type']}.")
            if arc.get("emotional_tone_end"):  instr_parts.append(f"Emotional ending: {arc['emotional_tone_end']}.")
            arc_stat = arc.get("story_status") or at.get("story_status", "")
            if arc_stat:                       instr_parts.append(f"Story is: {arc_stat}.")
            if desc:                           instr_parts.append(f"Summary: {desc}")

        chunks.append({
            "messages": [
                {"role": "system",    "content": CONFIG["system_prompt"]},
                {"role": "user",      "content": " ".join(instr_parts)},
                {"role": "assistant", "content": chunk_text},
            ]
        })
        start     += max_words - overlap_words
        chunk_num += 1

    return chunks


def prepare_data(use_chunking: bool = False):
    """
    Filter stories, apply preference weighting, and write training JSONL.

    Weighting works by repeating records:
      liked_story_weight=3  → each liked story appears 3x in the training file
      readers_pick_weight=2 → each readers pick appears 2x
      disliked stories      → excluded entirely

    use_chunking=True splits long stories into overlapping chunks so the
    full text is seen during training. Best used with A100 (more VRAM) since
    chunking increases the number of training examples significantly.
    """
    index_path = Path(CONFIG["index_path"])
    if not index_path.exists():
        print(f"ERROR: {index_path} not found. Run the scraper first.")
        sys.exit(1)

    print(f"Loading stories from {index_path}...")
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    stories = data.get("stories", [])
    print(f"Total stories in index: {len(stories):,}")

    # Load user preferences
    prefs = load_preferences()
    liked_ids      = set(prefs.get("liked_stories", []))
    disliked_ids   = set(prefs.get("disliked_stories", []))
    liked_authors  = set(prefs.get("liked_authors", []))
    disliked_authors = set(prefs.get("disliked_authors", []))

    print(f"Preferences loaded: {len(liked_ids)} liked, {len(disliked_ids)} disliked, "
          f"{len(liked_authors)} followed authors")

    # Filter
    kept, reasons = [], {}
    for s in stories:
        wc = s.get("word_count", 0)
        at = s.get("auto_tags") or {}
        sid = s.get("id", "")

        if s.get("fetch_error"):
            reasons["fetch_error"] = reasons.get("fetch_error", 0) + 1; continue
        if not s.get("full_text"):
            reasons["no_text"] = reasons.get("no_text", 0) + 1; continue
        if sid in disliked_ids:
            reasons["disliked"] = reasons.get("disliked", 0) + 1; continue
        if s.get("author") in disliked_authors:
            reasons["disliked_author"] = reasons.get("disliked_author", 0) + 1; continue
        if wc < CONFIG["min_words"]:
            reasons["too_short"] = reasons.get("too_short", 0) + 1; continue
        if wc > CONFIG["max_words"] and not use_chunking:
            reasons["too_long"] = reasons.get("too_long", 0) + 1; continue
        if count_trusted_tags(s) < CONFIG.get("min_trusted_tags", 2):
            reasons["few_trusted_tags"] = reasons.get("few_trusted_tags", 0) + 1; continue
        if CONFIG.get("exclude_wayback") and s.get("source") == "wayback":
            reasons["wayback"] = reasons.get("wayback", 0) + 1; continue
        if CONFIG.get("require_description") and not s.get("description"):
            reasons["no_description"] = reasons.get("no_description", 0) + 1; continue
        # Exclude stories Svengali has confirmed as incomplete
        if CONFIG.get("exclude_incomplete"):
            status = (s.get("auto_tags") or {}).get("story_status", "")
            if status == "incomplete":
                reasons["incomplete"] = reasons.get("incomplete", 0) + 1; continue
        kept.append(s)

    print(f"\nFiltering results:")
    print(f"  Kept:         {len(kept):,}")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:<22} {count:,}")

    if CONFIG["shuffle"]:
        random.shuffle(kept)

    if CONFIG["max_stories"] and len(kept) > CONFIG["max_stories"]:
        kept = kept[:CONFIG["max_stories"]]
        print(f"\nCapped to {CONFIG['max_stories']:,} stories (set max_stories higher to include more)")

    # Apply weights — build weighted list before writing
    liked_weight      = CONFIG.get("liked_story_weight",   3)
    pick_weight       = CONFIG.get("readers_pick_weight",  2)
    author_weight     = CONFIG.get("liked_author_weight",  2)
    svengali_weight   = CONFIG.get("svengali_story_weight", 2)

    weighted = []  # list of (story, weight) tuples
    for s in kept:
        sid    = s.get("id", "")
        author = s.get("author", "")
        at     = s.get("auto_tags") or {}
        w = 1
        if at.get("_svengali_tags"):
            w = max(w, svengali_weight)
        if sid in liked_ids:
            w = max(w, liked_weight)
        if s.get("readers_pick"):
            w = max(w, pick_weight)
        if author in liked_authors:
            w = max(w, author_weight)
        weighted.append((s, w))

    liked_count     = sum(1 for s, w in weighted if s.get("id") in liked_ids)
    picks_count     = sum(1 for s, w in weighted if s.get("readers_pick"))
    author_count    = sum(1 for s, w in weighted if s.get("author") in liked_authors)
    svengali_count  = sum(1 for s, w in weighted if (s.get("auto_tags") or {}).get("_svengali_tags"))
    print(f"\nWeighting:")
    print(f"  Svengali-tagged stories:{svengali_count:,}  (×{svengali_weight})")
    print(f"  Liked stories:          {liked_count:,}  (×{liked_weight})")
    print(f"  Readers' picks:         {picks_count:,}  (×{pick_weight})")
    print(f"  Followed author stories:{author_count:,}  (×{author_weight})")

    # Expand weights and chunk if requested
    max_chunk_words = int(CONFIG["max_seq_length"] * 0.65)  # ~65% of context = one chunk

    records = []
    for s, weight in weighted:
        if use_chunking and s.get("word_count", 0) > max_chunk_words:
            chunks = chunk_story(s, max_words=max_chunk_words, overlap_words=150)
        else:
            chunks = [story_to_record(s)]
        for _ in range(weight):
            records.extend(chunks)

    random.shuffle(records)

    # Write
    out_path = Path(CONFIG["train_data_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_words = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Count words in assistant turn
            for m in rec.get("messages", []):
                if m.get("role") == "assistant":
                    total_words += len(m["content"].split())

    total_tokens_est = total_words * 1.3
    size_mb  = out_path.stat().st_size / 1024 / 1024
    tok_per_sec = 800 if CONFIG["max_seq_length"] <= 2048 else 1200
    hours_est = (total_tokens_est / CONFIG["max_seq_length"]) * CONFIG["epochs"] / tok_per_sec / 3600

    print(f"\nTraining data written to {out_path}")
    print(f"  Base stories:      {len(kept):,}")
    print(f"  Total records:     {len(records):,}  (after weighting{'+ chunking' if use_chunking else ''})")
    print(f"  File size:         {size_mb:.1f} MB")
    print(f"  Est. tokens:       {total_tokens_est/1e6:.1f}M")
    if CONFIG["max_seq_length"] <= 2048:
        print(f"  Est. train time:   {hours_est:.1f} hours (RTX 3070 Ti)")
    else:
        print(f"  Est. train time:   {hours_est:.1f} hours (A100 80GB)")
    print(f"\nNext: python train.py --train")


# ── Training ──────────────────────────────────────────────────────────────────

class LossTracker:
    """
    Training callback that:
    - Prints a live ASCII loss curve in the terminal
    - Saves a loss_curve.png chart to the output directory
    - Estimates time remaining
    - Warns when loss plateaus or ticks up (signal to stop)
    """

    def __init__(self, output_dir: str, total_steps: int):
        self.output_dir  = Path(output_dir)
        self.total_steps = total_steps
        self.losses:  list[float] = []
        self.steps:   list[int]   = []
        self.start_time = time.time()
        self.best_loss  = float("inf")
        self.plateau_count = 0
        self._warned_plateau = False

    def on_log(self, step: int, loss: float):
        self.losses.append(loss)
        self.steps.append(step)

        if loss < self.best_loss:
            self.best_loss = loss
            self.plateau_count = 0
        else:
            self.plateau_count += 1

        # Time estimate
        elapsed   = time.time() - self.start_time
        rate      = step / max(elapsed, 1)
        remaining = (self.total_steps - step) / max(rate, 0.001)
        eta_min   = int(remaining / 60)
        eta_hr    = eta_min // 60
        eta_min  %= 60

        # ASCII sparkline (last 40 values)
        recent = self.losses[-40:]
        if len(recent) > 1:
            lo, hi = min(recent), max(recent)
            rng    = max(hi - lo, 0.001)
            chars  = " ▁▂▃▄▅▆▇█"
            spark  = "".join(chars[min(8, int(8 * (v - lo) / rng))] for v in recent)
        else:
            spark = ""

        # Plateau warning
        warn = ""
        if self.plateau_count >= 5 and not self._warned_plateau:
            warn = "  ⚠ loss plateaued — consider stopping"
            self._warned_plateau = True
        elif self.plateau_count >= 10:
            warn = "  ⚠ loss ticking up — good time to stop"

        print(
            f"  Step {step:>5}/{self.total_steps}  "
            f"Loss: {loss:.4f}  Best: {self.best_loss:.4f}  "
            f"ETA: {eta_hr}h{eta_min:02d}m  "
            f"|{spark}|"
            f"{warn}"
        )

        # Save chart every 100 steps
        if step % 100 == 0 and len(self.losses) > 5:
            self._save_chart()

    def _save_chart(self):
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.steps, self.losses, color="#8b4513", linewidth=1.5,
                    label="Training loss")
            # Smooth trend line
            if len(self.losses) > 10:
                window = max(5, len(self.losses) // 10)
                smooth = [sum(self.losses[max(0,i-window):i+1]) /
                          len(self.losses[max(0,i-window):i+1])
                          for i in range(len(self.losses))]
                ax.plot(self.steps, smooth, color="#c8956a", linewidth=2,
                        linestyle="--", label="Smoothed", alpha=0.8)

            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss — Step {self.steps[-1]}/{self.total_steps} "
                         f"(best: {self.best_loss:.4f})")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            chart_path = self.output_dir / "loss_curve.png"
            fig.savefig(chart_path, dpi=120)
            plt.close(fig)
        except ImportError:
            pass  # matplotlib not installed — skip chart
        except Exception as e:
            pass  # don't crash training over a chart

    def final_summary(self):
        elapsed = time.time() - self.start_time
        hrs     = int(elapsed // 3600)
        mins    = int((elapsed % 3600) // 60)
        print(f"\n  Training complete in {hrs}h {mins:02d}m")
        print(f"  Best loss:  {self.best_loss:.4f}")
        print(f"  Final loss: {self.losses[-1]:.4f}")
        if self.losses[-1] > self.best_loss + 0.05:
            print(f"  ⚠ Final loss is higher than best — consider using an earlier checkpoint")
        self._save_chart()
        if (self.output_dir / "loss_curve.png").exists():
            print(f"  Loss curve: {self.output_dir}/loss_curve.png")


def train(resume: bool = False):
    """Fine-tune using Unsloth + QLoRA."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install unsloth trl datasets transformers accelerate bitsandbytes")
        print("  # For CUDA 12.1 (most common):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    train_path = Path(CONFIG["train_data_path"])
    if not train_path.exists():
        print("No training data found. Run: python train.py --prepare")
        sys.exit(1)

    out_dir = Path(CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MC Stories Fine-Tune")
    print(f"  Model:  {CONFIG['base_model']}")
    print(f"  Data:   {train_path} ({train_path.stat().st_size/1024/1024:.0f} MB)")
    print(f"  Output: {out_dir}")
    print(f"  VRAM:   8GB target (QLoRA 4-bit)")
    print(f"{'='*60}\n")

    # Load model with Unsloth (handles 4-bit quantisation automatically)
    print("Loading model (this downloads ~5GB on first run)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = CONFIG["base_model"],
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,       # auto-detect (bfloat16 on Ampere+)
        load_in_4bit   = True,       # essential for 8GB VRAM
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r              = CONFIG["lora_r"],
        target_modules = CONFIG["lora_target"],
        lora_alpha     = CONFIG["lora_alpha"],
        lora_dropout   = CONFIG["lora_dropout"],
        bias           = "none",
        use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM
        random_state   = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
          f"({100*trainable/total:.1f}%)")

    # Load dataset
    print("Loading training data...")
    dataset = load_dataset("json", data_files=str(train_path), split="train")
    print(f"  {len(dataset):,} examples")

    # Format into chat template
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chat, batched=False)

    # Training arguments — tuned for 8GB
    training_args = TrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = CONFIG["epochs"],
        per_device_train_batch_size = CONFIG["batch_size"],
        gradient_accumulation_steps = CONFIG["grad_accum"],
        warmup_steps                = CONFIG["warmup_steps"],
        learning_rate               = CONFIG["learning_rate"],
        fp16                        = not _is_bfloat16_supported(),
        bf16                        = _is_bfloat16_supported(),
        logging_steps               = CONFIG["logging_steps"],
        save_steps                  = CONFIG["save_steps"],
        save_total_limit            = 3,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 42,
        report_to                   = "none",   # set to "wandb" to enable W&B logging
        resume_from_checkpoint      = resume,
    )

    # Calculate total steps for ETA
    steps_per_epoch = len(dataset) // (CONFIG["batch_size"] * CONFIG["grad_accum"])
    total_steps     = steps_per_epoch * CONFIG["epochs"]
    tracker         = LossTracker(str(out_dir), total_steps)

    # Custom callback to feed loss to tracker
    from transformers import TrainerCallback

    class LossCurveCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                tracker.on_log(state.global_step, logs["loss"])

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset,
        dataset_text_field = "text",
        max_seq_length     = CONFIG["max_seq_length"],
        args               = training_args,
        callbacks          = [LossCurveCallback()],
    )

    print(f"\nStarting training...")
    print(f"  Epochs:      {CONFIG['epochs']}")
    print(f"  Steps:       {total_steps:,}  ({steps_per_epoch:,} per epoch)")
    print(f"  Batch size:  {CONFIG['batch_size']} (effective: "
          f"{CONFIG['batch_size'] * CONFIG['grad_accum']})")
    print(f"  Checkpoints: every {CONFIG['save_steps']} steps → {out_dir}")
    print(f"  Loss curve:  {out_dir}/loss_curve.png  (updates every 100 steps)")
    print(f"  Ctrl+C safe — resume with: python train.py --train --resume\n")

    trainer.train(resume_from_checkpoint=resume)

    tracker.final_summary()

    # Save final LoRA weights
    final_path = out_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n✓ LoRA weights saved to {final_path}")
    print(f"  Next: python train.py --merge")


def _is_bfloat16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except:
        return False


# ── Merge LoRA into full model ────────────────────────────────────────────────

def merge():
    """Merge LoRA adapters into the base model for use in LM Studio."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: pip install unsloth")
        sys.exit(1)

    lora_path   = Path(CONFIG["output_dir"]) / "final"
    merged_path = Path(CONFIG["merged_dir"])

    if not lora_path.exists():
        print(f"No LoRA weights at {lora_path}. Run training first.")
        sys.exit(1)

    print(f"Merging LoRA into base model...")
    print(f"  Source: {lora_path}")
    print(f"  Output: {merged_path}")
    print(f"  This takes ~5 minutes and ~12GB disk space\n")

    merged_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(lora_path),
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
    )

    # Save as float16 for LM Studio / llama.cpp compatibility
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method = "merged_16bit",
    )

    print(f"\n✓ Merged model saved to {merged_path}")
    print(f"\nTo use in LM Studio:")
    print(f"  1. Open LM Studio")
    print(f"  2. My Models → Load from folder")
    print(f"  3. Point to: {merged_path.resolve()}")
    print(f"\nTo convert to GGUF for faster inference:")
    print(f"  pip install llama-cpp-python")
    print(f"  python convert_to_gguf.py  (see TRAINING_GUIDE.md)")


# ── Test generation ────────────────────────────────────────────────────────────

def test_generation():
    """Quick generation test to verify the model works."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: pip install unsloth")
        sys.exit(1)

    lora_path = Path(CONFIG["output_dir"]) / "final"
    if not lora_path.exists():
        print(f"No model at {lora_path}")
        sys.exit(1)

    print(f"Loading model from {lora_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = str(lora_path),
        max_seq_length = CONFIG["max_seq_length"],
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)  # enables 2x faster inference

    prompts = [
        "Write a short hypnosis story set in a therapist's office. "
        "Themes: slow burn, psychological. Tone: dark. Pacing: single scene.",

        "Write a mind control story about a woman discovering her new boss "
        "has unusual methods. Categories: mc mf. Themes: corruption arc, betrayal.",
    ]

    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print(prompt)
        print(f"\n--- Output {i} ---")

        messages = [
            {"role": "system",  "content": CONFIG["system_prompt"]},
            {"role": "user",    "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens  = 800,
                temperature     = 0.85,
                top_p           = 0.92,
                repetition_penalty = 1.15,
                do_sample       = True,
                pad_token_id    = tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(generated[:1500])
        if len(generated) > 1500:
            print(f"\n[...truncated, {len(generated)} chars total]")
        print()


# ── GGUF conversion helper ────────────────────────────────────────────────────

def write_gguf_script():
    """Write a helper script for converting to GGUF format."""
    script = '''#!/usr/bin/env python3
"""
convert_to_gguf.py — convert merged model to GGUF for llama.cpp / LM Studio

Run AFTER: python train.py --merge

Requires llama.cpp:
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp && pip install -r requirements.txt
"""
import subprocess, sys
from pathlib import Path

MERGED_DIR = "./output/merged_model"
GGUF_DIR   = "./output/gguf_model"
LLAMA_CPP  = "./llama.cpp"   # path to your llama.cpp clone

Path(GGUF_DIR).mkdir(parents=True, exist_ok=True)

# Convert to F16 GGUF first
print("Converting to GGUF F16...")
subprocess.run([
    sys.executable, f"{LLAMA_CPP}/convert_hf_to_gguf.py",
    MERGED_DIR, "--outfile", f"{GGUF_DIR}/model-f16.gguf",
    "--outtype", "f16"
], check=True)

# Quantise to Q4_K_M (best quality/size balance for 8GB VRAM)
print("Quantising to Q4_K_M...")
subprocess.run([
    f"{LLAMA_CPP}/llama-quantize",
    f"{GGUF_DIR}/model-f16.gguf",
    f"{GGUF_DIR}/model-Q4_K_M.gguf",
    "Q4_K_M"
], check=True)

print(f"\\nDone! Load in LM Studio: {GGUF_DIR}/model-Q4_K_M.gguf")
print("Q4_K_M is ~4.5GB and runs well on 8GB VRAM at ~40 tok/s")
'''
    path = Path("convert_to_gguf.py")
    path.write_text(script)
    print(f"Written: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(
        description="Fine-tune a story generation model on your mcstories dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. python train.py --prepare          Filter and prepare training data
  2. python train.py --train            Start training (~18-24 hours on 3070 Ti)
  3. python train.py --train --resume   Resume if interrupted
  4. python train.py --test             Quick generation sanity check
  5. python train.py --merge            Merge LoRA → full model for LM Studio

Edit CONFIG at the top of this file to change model, data filters, etc.
        """
    )
    p.add_argument("--prepare", action="store_true",
                   help="Filter stories and write training data")
    p.add_argument("--train",   action="store_true",
                   help="Start/resume fine-tuning")
    p.add_argument("--resume",  action="store_true",
                   help="Resume training from last checkpoint")
    p.add_argument("--test",    action="store_true",
                   help="Run generation test on trained model")
    p.add_argument("--merge",   action="store_true",
                   help="Merge LoRA weights into full model for LM Studio")
    p.add_argument("--gguf",    action="store_true",
                   help="Write GGUF conversion helper script")
    p.add_argument("--model",   default=None,
                   help="Override base model (e.g. unsloth/mistral-7b-instruct-v0.3)")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--max-stories", type=int, default=None)
    p.add_argument("--max-words",   type=int, default=None)
    p.add_argument("--seq-len",     type=int, default=None,
                   help="Max sequence length (reduce to 1024 if OOM)")
    p.add_argument("--preset",      choices=["3070", "a100", "a100_max",
                                                     "a100_12b", "a100_max_12b"], default=None,
                   help="Hardware preset: '3070' local validation, 'a100'/'a100_max' 8B rental, "
                        "'a100_12b'/'a100_max_12b' 12B rental (use with --model dolphin3.0-mistral-nemo-12b)")
    p.add_argument("--chunk",       action="store_true",
                   help="Chunk long stories into overlapping training examples")
    p.add_argument("--min-words",    type=int, default=None,
                   help="Minimum word count to include a story (default: 1500)")
    p.add_argument("--liked-weight",   type=int, default=None,
                   help="Training weight for liked stories (default: 3)")
    p.add_argument("--picks-weight",   type=int, default=None,
                   help="Training weight for readers pick stories (default: 2)")

    args = p.parse_args()

    # Apply preset first, then individual overrides
    if args.preset:
        preset = PRESETS[args.preset]
        for k, v in preset.items():
            CONFIG[k] = v
        print(f"Using {args.preset.upper()} preset: "
              f"seq_len={CONFIG['max_seq_length']}, "
              f"max_words={CONFIG['max_words']}, "
              f"lora_r={CONFIG['lora_r']}")

    if args.model:        CONFIG["base_model"]       = args.model
    if args.epochs:       CONFIG["epochs"]           = args.epochs
    if args.max_stories:  CONFIG["max_stories"]      = args.max_stories
    if args.max_words:    CONFIG["max_words"]        = args.max_words
    if args.min_words:    CONFIG["min_words"]        = args.min_words
    if args.seq_len:      CONFIG["max_seq_length"]   = args.seq_len
    if args.liked_weight: CONFIG["liked_story_weight"] = args.liked_weight
    if args.picks_weight: CONFIG["readers_pick_weight"] = args.picks_weight

    if args.prepare:
        prepare_data(use_chunking=args.chunk)
    elif args.train:
        train(resume=args.resume)
    elif args.test:
        test_generation()
    elif args.merge:
        merge()
    elif args.gguf:
        write_gguf_script()
    else:
        p.print_help()
