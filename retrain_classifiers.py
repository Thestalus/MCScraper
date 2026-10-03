#!/usr/bin/env python3
"""
retrain_classifiers.py — Retrain SetFit classifiers with expanded tag list.

Skips encoder fine-tuning (uses existing output/setfit_model/).
Retrains all per-tag logistic regression classifiers from cached embeddings.
Use this after adding new tags or when existing classifiers need recalibration.

Usage:
    python retrain_classifiers.py           # retrain classifiers only
    python retrain_classifiers.py --eval    # retrain + evaluate on held-out set
    python retrain_classifiers.py --tag     # retrain + apply tags to full index

Tags that CAN be trained now (≥50 Svengali positives):
    All 20 existing tags PLUS 5 new:
        fetish:bdsm                 (777 svengali pos)
        fetish:sadomasochism        (490 svengali pos)
        fetish:fetish clothing      (968 svengali pos)
        themes:serial recruitment   (133 svengali pos)
        themes:betrayal             (160 svengali pos)

Tags that need LLM TP/TN generation before SetFit training (0 Svengali labels):
    awareness:oblivious
    dominance:cold / clinical
    mental_changes:corruption arc
    mental_changes:identity loss
    relationship:reluctant subject
    themes:horror / dark
    sexual_acts:anal, creampie, facial, toys
    mind_control_type:chemical, magic, parasite / symbiote

Low-performing existing models being retrained (oof_f1 at last run):
    mind_control_type:possession        0.288
    mind_control_type:body puppet       0.289
    mental_changes:induced love         0.301
    relationship:authority              0.303
    mental_changes:addiction / need     0.304
    awareness:denial                    0.323
    dominance:brutal                    0.349
    dominance:gentle                    0.378
    mental_changes:memory alteration    0.402
    relationship:strangers              0.405
    pacing:rapid                        0.695 but BROKEN (tags 98% of corpus)
    pacing:leisurely                    0.639 but BROKEN (tags 88% of corpus)
"""

import sys
import pickle
import logging
from pathlib import Path

# Patch TARGET_TAGS before importing setfit_trainer (it reads TARGET_TAGS at module level
# only inside functions, so we can safely import and override).
sys.path.insert(0, str(Path(__file__).parent))

import setfit_trainer as st
import index_store

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

# ── Expanded tag list ──────────────────────────────────────────────────────────
# All original tags + 5 new ones with sufficient Svengali labels.
EXPANDED_TAGS = [
    # ── Awareness ─────────────────────────────────────────────────────────────
    ("awareness",         "valiant but futile struggle"),
    ("awareness",         "denial"),

    # ── Mental changes ────────────────────────────────────────────────────────
    ("mental_changes",    "obedience programming"),
    ("mental_changes",    "speech modification"),
    ("mental_changes",    "induced love"),
    ("mental_changes",    "memory alteration"),
    ("mental_changes",    "addiction / need"),

    # ── Mind control type ─────────────────────────────────────────────────────
    ("mind_control_type", "hypnosis"),
    ("mind_control_type", "possession"),
    ("mind_control_type", "body puppet"),

    # ── Dominance ─────────────────────────────────────────────────────────────
    ("dominance",         "gentle"),
    ("dominance",         "brutal"),

    # ── Pacing ────────────────────────────────────────────────────────────────
    ("pacing",            "leisurely"),
    ("pacing",            "rapid"),

    # ── Relationship ──────────────────────────────────────────────────────────
    ("relationship",      "strangers"),
    ("relationship",      "friends"),
    ("relationship",      "couples"),
    ("relationship",      "peers"),
    ("relationship",      "enemies"),
    ("relationship",      "authority"),

    # ── NEW: Fetish (sufficient Svengali labels) ───────────────────────────────
    ("fetish",            "bdsm"),              # 777 svengali pos
    ("fetish",            "sadomasochism"),      # 490 svengali pos
    ("fetish",            "fetish clothing"),    # 968 svengali pos

    # ── NEW: Themes (sufficient Svengali labels) ───────────────────────────────
    ("themes",            "serial recruitment"), # 133 svengali pos
    ("themes",            "betrayal"),           # 160 svengali pos

    # ── NEW: LLM-labeled (no Svengali coverage — uses auto_label TP/TN) ────────
    ("relationship",      "reluctant subject"),  # 100 LLM TPs (run 2)
    ("mental_changes",    "corruption arc"),     # 100 LLM TPs (run 2)
    ("mental_changes",    "identity loss"),      # 100 LLM TPs (run 2)
    ("mental_changes",    "bimboization"),       # 100 LLM TPs (run 2)
    ("mental_changes",    "intelligence change"),# 100 LLM TPs (run 2)
    ("sexual_acts",       "creampie"),           # 100 LLM TPs
    ("sexual_acts",       "facial"),             # 54 LLM TPs (partial — scarce)
    ("sexual_acts",       "toys"),               # 99 LLM TPs (run 2)
    ("sexual_acts",       "anal"),               # 100 LLM TPs (run 2)
    ("mind_control_type", "parasite / symbiote"),# 100 LLM TPs
    ("mind_control_type", "chemical"),           # 100 LLM TPs (run 2)
    ("mind_control_type", "magic"),              # 97 LLM TPs (run 2)
    ("themes",            "horror / dark"),      # 89 LLM TPs (run 2)
    ("awareness",         "oblivious"),          # 100 LLM TPs (run 2)
    ("dominance",         "cold / clinical"),    # 100 LLM TPs (run 2)
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain SetFit classifiers (classifier phase only)")
    parser.add_argument("--eval",      action="store_true", help="Evaluate on held-out set after training")
    parser.add_argument("--tag",       action="store_true", help="Apply tags to all stories after training")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Override all thresholds (0 = use per-tag calibrated)")
    parser.add_argument("--gold-only", action="store_true",
                        help="Use only audited TP/TN labels, no Svengali fallback")
    args = parser.parse_args()
    use_svengali_fallback = not args.gold_only

    stories = st.load_index()
    audit_labels = st.load_audit_labels()

    if args.gold_only:
        log.info("--gold-only: using audit labels exclusively")

    # Determine active tags using gold positive counts (audit + Svengali, not raw auto_tags)
    active_tags = []
    for dim, tag in EXPANDED_TAGS:
        n_pos = st.count_gold_positives(stories, dim, tag, audit_labels)
        if n_pos >= st.MIN_POSITIVES:
            active_tags.append((dim, tag))
        else:
            log.info(f"  Skipping {dim}:{tag} — only {n_pos} gold positives (< {st.MIN_POSITIVES})")

    log.info(f"{len(active_tags)} tags qualify for training")

    # Phase 2: build / load embeddings (uses existing fine-tuned encoder)
    log.info("Loading embeddings (skip fine-tuning — using existing encoder)...")
    vecs, valid = st.build_embeddings(stories)

    # Phase 3: train classifiers with hybrid labels
    log.info("Training classifiers...")
    classifiers = st.train_classifiers(valid, vecs, active_tags,
                                       audit_labels=audit_labels,
                                       use_svengali_fallback=use_svengali_fallback)

    # Save (atomic write)
    tmp = Path(st.CLF_PATH).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(classifiers, f)
    tmp.replace(st.CLF_PATH)
    log.info(f"Classifiers saved to {st.CLF_PATH} ({len(classifiers)} total)")

    if args.eval:
        log.info("Evaluating on held-out set...")
        st.eval_classifiers(valid, vecs, classifiers, threshold=args.threshold,
                            audit_labels=audit_labels,
                            use_svengali_fallback=use_svengali_fallback)

    if args.tag:
        log.info("Applying tags to full index...")
        # Clear old SetFit tags first then re-apply (retag semantics)
        data = index_store.load_index(st.INDEX_PATH, for_write=True)
        all_dims = set(info["dim"] for info in classifiers.values())
        for s in data.get("stories", []):
            if not s:
                continue
            at = s.get("auto_tags") or {}
            sources = at.get("_tag_sources") or {}
            for dim in all_dims:
                for tag in list(at.get(dim, [])):
                    src = sources.get(tag) or []
                    if "setfit" in src:
                        new_src = [x for x in src if x != "setfit"]
                        if new_src:
                            sources[tag] = new_src
                        else:
                            sources.pop(tag, None)
                            at.get(dim, []).remove(tag)
        index_store.save_index(data, st.INDEX_PATH, indent=None)
        log.info("Cleared old SetFit tags. Re-loading and applying...")
        stories = st.load_index()
        vecs, valid = st.build_embeddings(stories)
        st.tag_all(stories, vecs, classifiers, valid, threshold=args.threshold)


if __name__ == "__main__":
    main()
