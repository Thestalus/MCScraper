"""
tag_trust.py — which auto-tags are trustworthy enough to put in training data.

Single source of truth for the trust tiers. Used by train.py (fine-tuning
records) and by the JSONL exports written by tagger.py, scraper.py and update.py,
so no export carries low-precision tags as if they were facts.
"""
from __future__ import annotations
import random

# ── Tag trust tiers (from CLASSIFIER_GUIDE.md — March 2026 FP audit) ──────────
# Tier 1: regex precision ≥74% — include unconditionally in training prompts.
TIER_1_TAGS = frozenset({
    "trigger phrase",        # themes   — 91%
    "erotic / explicit",     # tone     — 91%
    "manipulative",          # dominance — 90%
    "obedience programming", # mental   — 87%
    "psychological",         # tone     — 86%
    "consent / ethics",      # themes   — 86%
    "slave mentality",       # themes   — 85%
    "induced love",          # mental   — 77%
    "power corrupts",        # themes   — 76%
    "dark / disturbing",     # tone     — 74%
    "home / domestic",       # setting  — 78%
})

# Tier 2: AND-gated — include only when BOTH 'regex' AND 'setfit' in _tag_sources[tag].
AND_GATE_TAGS = frozenset({
    "hypnosis",          # mc_type  — F1=0.596
    "magic",             # mc_type  — F1=0.541
    "chemical",          # mc_type  — F1=0.380
    "memory alteration", # mental   — F1=0.440
    "identity loss",     # mental   — F1=0.523
    "leisurely",         # pacing   — F1=0.625
    "anal",              # sexual   — F1=0.402
    "toys",              # sexual   — F1=0.450
})

# Structural dimensions — Svengali-derived (≥80% precision), always include when present.
_STRUCTURAL_DIMS = ("content_type", "site_tag_groups", "site_tags_expanded")

# Per-tag dropout probability for semantic tags.
# Teaches the model to handle sparse/partial prompts; prevents it learning
# "tag not mentioned → concept absent". Structural dims are never dropped.
DROPOUT_RATE = 0.25


def story_training_tags(story: dict, apply_dropout: bool = True) -> dict:
    """Return a filtered tag dict for use in LLM training prompts.

    Keeps:
      - Structural dims (content_type, site_tag_groups, site_tags_expanded) — always
      - Tier 1 tags (regex precision ≥74%) — always when present
      - Tier 2 tags (AND-gate: both 'regex' AND 'setfit' in _tag_sources) — when agreed

    Drops:
      - Everything else: noisy classifiers, broken pacing tags, low-F1 models, etc.
      - apply_dropout=True: each included semantic tag has DROPOUT_RATE chance of being
        omitted per example, making the model robust to incomplete labels at inference.
        At least one semantic tag is always kept if any were present.
    """
    at      = story.get("auto_tags") or {}
    sources = at.get("_tag_sources") or {}

    def is_and_gated(tag: str) -> bool:
        src = sources.get(tag, [])
        return "regex" in src and "setfit" in src

    def apply_drop(tags: list) -> list:
        if not apply_dropout or not tags:
            return tags
        kept = [t for t in tags if random.random() > DROPOUT_RATE]
        return kept if kept else [random.choice(tags)]  # never drop everything

    result = {}

    # Structural — never dropped (Svengali-derived, high precision)
    for dim in _STRUCTURAL_DIMS:
        val = at.get(dim)
        if val:
            result[dim] = val if isinstance(val, list) else [val]

    # mc_type — Tier 2 only (no Tier 1 mc_type tags exist)
    mc = [t for t in at.get("mind_control_type", []) if t in AND_GATE_TAGS and is_and_gated(t)]
    mc = apply_drop(mc)
    if mc:
        result["mind_control_type"] = mc

    # dominance — Tier 1 only ('manipulative')
    dom = apply_drop([t for t in at.get("dominance", []) if t in TIER_1_TAGS])
    if dom:
        result["dominance"] = dom[:1]

    # mental_changes — Tier 1 + AND-gated Tier 2
    mental = [t for t in at.get("mental_changes", [])
              if t in TIER_1_TAGS or (t in AND_GATE_TAGS and is_and_gated(t))]
    mental = apply_drop(mental)
    if mental:
        result["mental_changes"] = mental[:3]

    # themes — Tier 1 only
    themes = apply_drop([t for t in at.get("themes", []) if t in TIER_1_TAGS])
    if themes:
        result["themes"] = themes[:4]

    # tone — Tier 1 only (erotic/explicit, psychological, dark/disturbing)
    tone = apply_drop([t for t in at.get("tone", []) if t in TIER_1_TAGS])
    if tone:
        result["tone"] = tone[:2]

    # setting — Tier 1 only ('home / domestic')
    setting = apply_drop([t for t in at.get("setting", []) if t in TIER_1_TAGS])
    if setting:
        result["setting"] = setting[:1]

    # sexual_acts — Tier 2 only (anal, toys)
    sex = [t for t in at.get("sexual_acts", []) if t in AND_GATE_TAGS and is_and_gated(t)]
    sex = apply_drop(sex)
    if sex:
        result["sexual_acts"] = sex

    # pacing — Tier 2 only ('leisurely')
    pac = [t for t in at.get("pacing", []) if t in AND_GATE_TAGS and is_and_gated(t)]
    pac = apply_drop(pac)
    if pac:
        result["pacing"] = pac[:1]

    return result


def export_tags(auto_tags: dict | None) -> dict:
    """Trusted tags for the JSONL exports: story_training_tags() without dropout,
    plus length_class and pov (computed, not classifier guesses)."""
    auto_tags = auto_tags or {}
    trusted = story_training_tags({"auto_tags": auto_tags}, apply_dropout=False)
    for key in ("length_class", "pov"):
        if auto_tags.get(key):
            trusted[key] = auto_tags[key]
    return trusted
