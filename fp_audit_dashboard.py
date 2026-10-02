#!/usr/bin/env python3
"""
fp_audit_dashboard.py

Small terminal dashboard for false-positive audit results.

Reads:
- output/stories_index.json
- output/fp_audit_results.jsonl

Shows:
- overall review totals
- per-tag precision
- per-source precision
- recommended stop / continue decisions

Usage:
    python3 fp_audit_dashboard.py
    python3 fp_audit_dashboard.py --top 20
    python3 fp_audit_dashboard.py --focus-tags mind_control_type:,mental_changes:
    python3 fp_audit_dashboard.py --source setfit
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INDEX = Path("output/stories_index.json")
DEFAULT_RESULTS = Path("output/fp_audit_results.jsonl")
SV_RAW_FIELDS = ("_svengali_tags", "svengali_tags")


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
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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
    if isinstance(vals, list):
        return [v for v in vals if isinstance(v, str)]
    return []


def make_tag_key(dim: str, tag: str) -> str:
    return f"{dim}:{tag}"


def parse_focus_prefixes(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def matches_focus(tag_key: str, focus_prefixes: list[str]) -> bool:
    if not focus_prefixes:
        return True
    return any(tag_key.startswith(prefix) for prefix in focus_prefixes)


def candidate_counts(
    stories: list[dict],
    focus_prefixes: list[str] | None = None,
    source: str = "",
    source_mode: str = "contains",
) -> Counter:
    focus_prefixes = focus_prefixes or []
    counts = Counter()
    for story in stories:
        at = get_auto_tags(story)
        sv_raw = set(get_sv_raw(story))
        for dim, vals in at.items():
            if dim.startswith("_") or not isinstance(vals, list):
                continue
            for tag in vals:
                if not isinstance(tag, str):
                    continue
                if tag in sv_raw:
                    continue
                tag_key = make_tag_key(dim, tag)
                if not matches_focus(tag_key, focus_prefixes):
                    continue
                sources = get_tag_sources(story, tag)
                if source_mode == "contains" and source:
                    if source not in sources:
                        continue
                elif source_mode == "only" and source:
                    if set(sources) != {source}:
                        continue
                counts[tag_key] += 1
    return counts


def precision_from_counts(tp: int, fp: int) -> float | None:
    denom = tp + fp
    if denom == 0:
        return None
    return tp / denom


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100*x:.1f}%"


def decision(reviewed: int, tp: int, fp: int, amb: int) -> str:
    prec = precision_from_counts(tp, fp)
    if reviewed < 10:
        return "too early"
    if reviewed < 30:
        if prec is None:
            return "need clearer labels"
        if prec >= 0.90:
            return "probably solid; get to 30"
        if prec <= 0.60:
            return "likely noisy; inspect false positives"
        return "mixed; get to 30"
    if prec is None:
        return "need more TP/FP labels"
    if prec >= 0.90:
        return "stop soon / keep as-is"
    if prec >= 0.75:
        return "good, refine slightly"
    if prec >= 0.60:
        return "mixed, review 20 more"
    return "noisy, investigate deeply"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precision dashboard for fp audit results.")
    p.add_argument("--index", default=str(DEFAULT_INDEX), help="Path to stories_index.json")
    p.add_argument("--results", default=str(DEFAULT_RESULTS), help="Path to fp_audit_results.jsonl")
    p.add_argument("--top", type=int, default=15, help="How many tags to show")
    p.add_argument("--focus-tags", default="", help="Comma-separated prefixes, e.g. mind_control_type:,mental_changes:")
    p.add_argument("--source", default="", help="Optional candidate pool source filter")
    p.add_argument("--source-mode", choices=["any", "contains", "only"], default="contains")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json(Path(args.index))
    stories = data.get("stories", data) if isinstance(data, dict) else data
    if not isinstance(stories, list):
        raise SystemExit("Expected stories_index.json to contain a top-level 'stories' list.")

    results = load_jsonl(Path(args.results))
    focus_prefixes = parse_focus_prefixes(args.focus_tags)

    overall = Counter(r.get("label", "UNKNOWN") for r in results)
    tp_all = overall.get("TRUE_POSITIVE", 0)
    fp_all = overall.get("FALSE_POSITIVE", 0)
    amb_all = overall.get("AMBIGUOUS", 0)
    skip_all = overall.get("SKIP", 0)

    print("OVERALL")
    print("-------")
    print(f"reviewed rows:      {len(results)}")
    print(f"TRUE_POSITIVE:      {tp_all}")
    print(f"FALSE_POSITIVE:     {fp_all}")
    print(f"AMBIGUOUS:          {amb_all}")
    print(f"SKIP:               {skip_all}")
    print(f"overall precision:  {pct(precision_from_counts(tp_all, fp_all))}")
    print()

    # candidate pool counts from index
    cand_counts = candidate_counts(
        stories=stories,
        focus_prefixes=focus_prefixes,
        source=args.source.strip(),
        source_mode=args.source_mode,
    )

    tag_stats = defaultdict(Counter)
    for r in results:
        tag = r.get("tag")
        if not isinstance(tag, str):
            continue
        if not matches_focus(tag, focus_prefixes):
            continue
        tag_stats[tag][r.get("label", "UNKNOWN")] += 1

    rows = []
    for tag in sorted(set(cand_counts) | set(tag_stats)):
        c = tag_stats[tag]
        tp = c.get("TRUE_POSITIVE", 0)
        fp = c.get("FALSE_POSITIVE", 0)
        amb = c.get("AMBIGUOUS", 0)
        skip = c.get("SKIP", 0)
        reviewed = tp + fp + amb + skip
        rows.append({
            "tag": tag,
            "candidate_total": cand_counts.get(tag, 0),
            "reviewed": reviewed,
            "remaining_est": max(0, cand_counts.get(tag, 0) - reviewed),
            "tp": tp,
            "fp": fp,
            "amb": amb,
            "skip": skip,
            "precision": precision_from_counts(tp, fp),
            "decision": decision(reviewed, tp, fp, amb),
        })

    rows.sort(key=lambda r: (-r["remaining_est"], -(r["reviewed"]), r["tag"]))

    print("PER TAG")
    print("-------")
    print("tag\treviewed\tTP\tFP\tAMB\tprecision\tremaining_est\tdecision")
    for r in rows[: args.top]:
        print(
            f'{r["tag"]}\t{r["reviewed"]}\t{r["tp"]}\t{r["fp"]}\t{r["amb"]}\t'
            f'{pct(r["precision"])}\t{r["remaining_est"]}\t{r["decision"]}'
        )
    print()

    # best / worst among sufficiently reviewed tags
    eligible = [r for r in rows if (r["tp"] + r["fp"]) >= 5 and r["precision"] is not None]
    best = sorted(eligible, key=lambda r: (-r["precision"], -r["reviewed"], r["tag"]))[:10]
    worst = sorted(eligible, key=lambda r: (r["precision"], -r["reviewed"], r["tag"]))[:10]

    print("BEST TAGS (min 5 TP/FP decisions)")
    print("-------------------------------")
    for r in best:
        print(f'{r["tag"]}: {pct(r["precision"])}  (TP={r["tp"]}, FP={r["fp"]}, reviewed={r["reviewed"]})')
    print()

    print("WORST TAGS (min 5 TP/FP decisions)")
    print("--------------------------------")
    for r in worst:
        print(f'{r["tag"]}: {pct(r["precision"])}  (TP={r["tp"]}, FP={r["fp"]}, reviewed={r["reviewed"]})')
    print()

    # per-source from review rows
    source_stats = defaultdict(Counter)
    for r in results:
        sources = r.get("sources") or []
        if not sources:
            source_stats["(none)"][r.get("label", "UNKNOWN")] += 1
        else:
            for s in sources:
                source_stats[s][r.get("label", "UNKNOWN")] += 1

    print("PER SOURCE")
    print("----------")
    print("source\treviewed\tTP\tFP\tAMB\tprecision\tdecision")
    for src, c in sorted(source_stats.items(), key=lambda kv: -(kv[1].get("TRUE_POSITIVE", 0)+kv[1].get("FALSE_POSITIVE", 0)+kv[1].get("AMBIGUOUS", 0)+kv[1].get("SKIP", 0))):
        tp = c.get("TRUE_POSITIVE", 0)
        fp = c.get("FALSE_POSITIVE", 0)
        amb = c.get("AMBIGUOUS", 0)
        skip = c.get("SKIP", 0)
        reviewed = tp + fp + amb + skip
        print(f"{src}\t{reviewed}\t{tp}\t{fp}\t{amb}\t{pct(precision_from_counts(tp, fp))}\t{decision(reviewed, tp, fp, amb)}")

    print()
    print("SUGGESTED NEXT STEP")
    print("-------------------")
    prioritized = [r for r in rows if r["remaining_est"] > 0]
    prioritized.sort(key=lambda r: (
        # prefer tags with some evidence but not enough
        0 if 10 <= r["reviewed"] < 50 else 1,
        # then highest remaining
        -r["remaining_est"],
        r["tag"],
    ))
    if prioritized:
        nxt = prioritized[0]
        print(f'Next tag to keep auditing: {nxt["tag"]}')
        print(f'Suggested target: {"50-60" if nxt["reviewed"] < 30 else "20 more"} reviews total')
    else:
        print("No remaining candidate tags found.")


if __name__ == "__main__":
    main()
