#!/usr/bin/env python3
"""
eval_tagger.py — Measure auto-tagger precision/recall against Svengali ground truth.

Run after any tagger.py change to see if precision/recall improved.

Usage:
    python eval_tagger.py                    # full report
    python eval_tagger.py --dim dominance    # one dimension
    python eval_tagger.py --examples brutal  # show FP/FN examples for a tag
    python eval_tagger.py --recompute        # recompute fresh regex tags (slow)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import svengali as sv
import tagger as tg

INDEX   = Path("./output/stories_index.json")

# Dimensions where Svengali has direct coverage
SV_VOCAB = {
    "mind_control_type": set(sv.SV_MC_TYPE.values()),
    "dominance":         set(sv.SV_DOMINANCE.values()),
    "mental_changes":    set(sv.SV_MENTAL.values()),
    "pacing":            set(sv.SV_PACING.values()),
    "awareness":         set(sv.SV_AWARENESS.values()) | set(sv.SV_CONSENT.values()),
    "sexual_acts":       set(sv.SV_SEXUAL_ACTS.values()),
    "fetish":            set(sv.SV_FETISH.values()),
}


def load_index():
    with open(INDEX, encoding="utf-8") as f:
        return json.load(f)["stories"]


def get_regex_vals(story: dict, dim: str) -> set[str]:
    """Tags in dim that came from the regex layer (per _tag_sources)."""
    at      = story.get("auto_tags") or {}
    sources = at.get("_tag_sources") or {}
    dim_val = at.get(dim, [])
    if not isinstance(dim_val, list):
        return set()
    return {t for t in dim_val if "regex" in sources.get(t, [])}


def get_sv_vals(story: dict, dim: str) -> set[str]:
    """Tags in dim derived from the story's _svengali_tags."""
    at     = story.get("auto_tags") or {}
    sv_raw = at.get("_svengali_tags")
    if not sv_raw:
        return set()
    return set((sv.parse_svengali_response({"tags": sv_raw}) or {}).get(dim, []))


def compute_metrics(stories, dims=None, recompute=True):
    """
    Returns {dim: {tag: {tp, fp, fn, precision, recall, f1}}}
    Only counts tags that exist in Svengali's vocabulary for that dim.
    Only includes stories that Svengali has tagged (has _svengali_tags).
    """
    if dims is None:
        dims = list(SV_VOCAB.keys())

    tagged = [s for s in stories if (s.get("auto_tags") or {}).get("_svengali_tags")]
    print(f"Evaluating {len(tagged):,} stories with Svengali tags...\n")

    precomputed = None
    if recompute:
        from retag_parallel import parallel_regex_tag
        print("Pre-computing regex tags in parallel...")
        precomputed = parallel_regex_tag(tagged)
        print(f"Done. Tagged {len(precomputed):,} stories.\n")

    stats = {dim: {} for dim in dims}
    examples_fp = {dim: {} for dim in dims}
    examples_fn = {dim: {} for dim in dims}

    for s in tagged:
        for dim in dims:
            sv_vocab = SV_VOCAB[dim]
            sv_vals  = get_sv_vals(s, dim)
            if recompute:
                rx_vals = set(precomputed.get(s["id"], {}).get(dim, []))
            else:
                rx_vals = get_regex_vals(s, dim)

            # Only evaluate tags that are in SV's vocabulary
            for tag in sv_vocab:
                d = stats[dim].setdefault(tag, {"tp": 0, "fp": 0, "fn": 0})
                has_rx = tag in rx_vals
                has_sv = tag in sv_vals
                if has_rx and has_sv:
                    d["tp"] += 1
                elif has_rx and not has_sv:
                    d["fp"] += 1
                    if tag not in examples_fp[dim]:
                        examples_fp[dim][tag] = s["id"]
                elif not has_rx and has_sv:
                    d["fn"] += 1
                    if tag not in examples_fn[dim]:
                        examples_fn[dim][tag] = s["id"]

    # Compute precision, recall, F1
    for dim in dims:
        for tag, d in stats[dim].items():
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            d["precision"] = tp / (tp + fp) if (tp + fp) else None
            d["recall"]    = tp / (tp + fn) if (tp + fn) else None
            if d["precision"] and d["recall"]:
                p, r = d["precision"], d["recall"]
                d["f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
            else:
                d["f1"] = None
            d["sv_total"]   = tp + fn   # stories SV tagged with this
            d["auto_total"] = tp + fp   # stories auto tagged with this

    return stats, examples_fp, examples_fn


def print_report(stats, examples_fp, examples_fn, min_sv=20):
    """Print precision/recall table for all dimensions."""
    for dim, tag_stats in stats.items():
        # Filter to tags SV uses at least min_sv times
        rows = [
            (tag, d) for tag, d in tag_stats.items()
            if d["sv_total"] >= min_sv
        ]
        if not rows:
            continue

        rows.sort(key=lambda x: (x[1]["f1"] or -1))

        print(f"\n{'═'*72}")
        print(f"  {dim.upper()}")
        print(f"{'═'*72}")
        print(f"  {'Tag':<28} {'SV':>5} {'Auto':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"  {'-'*64}")
        for tag, d in rows:
            prec = f"{d['precision']:.0%}" if d["precision"] is not None else "  —  "
            rec  = f"{d['recall']:.0%}"    if d["recall"]    is not None else "  —  "
            f1   = f"{d['f1']:.2f}"        if d["f1"]        is not None else "  —  "
            fp_ex = examples_fp[dim].get(tag, "")
            fn_ex = examples_fn[dim].get(tag, "")
            note = ""
            if d["precision"] is not None and d["precision"] < 0.4:
                note = f"  ← FP: {fp_ex}"
            elif d["recall"] is not None and d["recall"] < 0.3:
                note = f"  ← FN: {fn_ex}"
            print(f"  {tag:<28} {d['sv_total']:>5,} {d['auto_total']:>5,} {prec:>6} {rec:>6} {f1:>6}{note}")


def show_examples(stories, dim, tag, n=10):
    """Show FP and FN example stories for a specific tag."""
    tagged = [s for s in stories if (s.get("auto_tags") or {}).get("_svengali_tags")]

    fps = []
    fns = []
    for s in tagged:
        sv_vals = get_sv_vals(s, dim)
        rx_vals = get_regex_vals(s, dim)
        if tag in rx_vals and tag not in sv_vals:
            fps.append(s)
        elif tag not in rx_vals and tag in sv_vals:
            fns.append(s)

    print(f"\nFalse positives for {dim}/{tag} (auto fires, SV disagrees) — {len(fps):,} total")
    for s in fps[:n]:
        at = s.get("auto_tags") or {}
        print(f"  {s['id']:<30}  SV: {(s.get('auto_tags') or {}).get('_svengali_tags', [])[:5]}")

    print(f"\nFalse negatives for {dim}/{tag} (SV has it, auto misses) — {len(fns):,} total")
    for s in fns[:n]:
        at = s.get("auto_tags") or {}
        print(f"  {s['id']:<30}  SV: {at.get('_svengali_tags', [])[:5]}")


def main():
    p = argparse.ArgumentParser(description="Evaluate auto-tagger vs Svengali")
    p.add_argument("--dim",      help="Only evaluate this dimension")
    p.add_argument("--examples", metavar="TAG", help="Show FP/FN examples for this tag")
    p.add_argument("--no-recompute", action="store_true",
                   help="Use stored _tag_sources instead of recomputing regex tags (faster but may be stale)")
    p.add_argument("--min-sv",   type=int, default=20,
                   help="Min Svengali stories for a tag to appear in report (default: 20)")
    global INDEX
    p.add_argument("--index",    default=str(INDEX))
    args = p.parse_args()

    INDEX = Path(args.index)

    stories = load_index()
    dims = [args.dim] if args.dim else None

    stats, fp_ex, fn_ex = compute_metrics(stories, dims=dims, recompute=not args.no_recompute)

    if args.examples:
        # Find which dim this tag lives in
        target_dim = args.dim
        if not target_dim:
            for d, vocab in SV_VOCAB.items():
                if args.examples in vocab:
                    target_dim = d
                    break
        if target_dim:
            show_examples(stories, target_dim, args.examples)
        else:
            print(f"Tag '{args.examples}' not found in Svengali vocabulary")
    else:
        print_report(stats, fp_ex, fn_ex, min_sv=args.min_sv)


if __name__ == "__main__":
    main()
