#!/usr/bin/env python3
"""
search_stories.py — search and browse your local mcstories JSON index

Usage:
  python search_stories.py                         # interactive mode
  python search_stories.py -q "hypnosis"           # keyword search
  python search_stories.py -a "AuthorName"         # filter by author
  python search_stories.py -c "MC" "FF"            # filter by categories
  python search_stories.py --stats                 # show dataset stats
  python search_stories.py --export results.jsonl  # export results to JSONL
"""

import json
import argparse
import re
import sys
from pathlib import Path


INDEX_FILE = Path("output/stories_index.json")


def load_index(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["stories"]


def search(stories: list[dict], query: str = "", author: str = "",
           categories: list[str] = None, min_words: int = 0,
           themes: list[str] = None, mc_type: str = "",
           setting: str = "", tone: str = "", length_class: str = "") -> list[dict]:
    q = query.lower()
    a = author.lower()
    cats = [c.lower() for c in (categories or [])]

    results = []
    for s in stories:
        if s.get("fetch_error") or not s.get("full_text"):
            continue
        # keyword — searches title, description, full text
        if q and not (
            q in s.get("title", "").lower()
            or q in s.get("description", "").lower()
            or q in s.get("full_text", "").lower()
        ):
            continue
        # author filter
        if a and a not in s.get("author", "").lower():
            continue
        # category filter (all requested cats must be present)
        if cats:
            story_cats = [c.lower() for c in s.get("categories", [])]
            if not all(c in story_cats for c in cats):
                continue
        # word count filter
        if s.get("word_count", 0) < min_words:
            continue

        # auto_tag filters
        at = s.get("auto_tags") or {}
        if themes:
            story_themes = [t.lower() for t in at.get("themes", [])]
            if not all(t.lower() in story_themes for t in themes):
                continue
        if mc_type and mc_type.lower() not in [x.lower() for x in at.get("mind_control_type", [])]:
            continue
        if setting and not any(setting.lower() in x.lower() for x in at.get("setting", [])):
            continue
        if tone and not any(tone.lower() in x.lower() for x in at.get("tone", [])):
            continue
        if length_class and at.get("length_class", "").lower() != length_class.lower():
            continue

        results.append(s)
    return results


def print_story(s: dict, verbose: bool = False):
    print(f"\n{'─'*60}")
    print(f"  Title   : {s.get('title', '—')}")
    print(f"  Author  : {s.get('author', '—')}")
    print(f"  Words   : {s.get('word_count', 0):,}")
    print(f"  Added   : {s.get('added_date', '—')}")
    cats = ", ".join(s.get("categories", [])) or "—"
    print(f"  Tags    : {cats}")
    print(f"  URL     : {s.get('story_url', '—')}")
    if s.get("description"):
        print(f"  Blurb   : {s['description'][:160]}{'…' if len(s['description'])>160 else ''}")
    at = s.get("auto_tags") or {}
    if at.get("mind_control_type"):
        print(f"  MC Type : {', '.join(at['mind_control_type'])}")
    if at.get("setting"):
        print(f"  Setting : {', '.join(at['setting'])}")
    if at.get("themes"):
        print(f"  Themes  : {', '.join(at['themes'])}")
    if at.get("tone"):
        print(f"  Tone    : {', '.join(at['tone'])}")
    if at.get("length_class"):
        print(f"  Length  : {at['length_class']}  |  POV: {at.get('pov','?')}")
    if verbose and s.get("full_text"):
        preview = s["full_text"][:500].replace("\n", " ")
        print(f"  Preview : {preview}…")


def print_stats(stories: list[dict]):
    ok = [s for s in stories if not s.get("fetch_error") and s.get("full_text")]
    total_words = sum(s.get("word_count", 0) for s in ok)
    authors = {s.get("author") for s in ok}
    all_cats: dict[str, int] = {}
    for s in ok:
        for c in s.get("categories", []):
            all_cats[c] = all_cats.get(c, 0) + 1

    top_cats = sorted(all_cats.items(), key=lambda x: -x[1])[:15]

    print(f"\n{'═'*60}")
    print(f"  Dataset statistics")
    print(f"{'─'*60}")
    print(f"  Stories        : {len(ok):,}")
    print(f"  Unique authors : {len(authors):,}")
    print(f"  Total words    : {total_words:,}")
    print(f"  Avg words/story: {total_words//max(len(ok),1):,}")
    print(f"\n  Top categories:")
    for cat, count in top_cats:
        bar = "█" * min(count // max(1, len(ok) // 40), 30)
        print(f"    {cat:<20} {count:>4}  {bar}")
    print()


def interactive(stories: list[dict]):
    print("\n📚 mcstories local search")
    print("  Commands: search <query> | author <name> | cat <tag> | stats | quit\n")
    while True:
        try:
            line = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        parts = line.split(None, 1)
        cmd, rest = parts[0].lower(), parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "stats":
            print_stats(stories)
        elif cmd == "search":
            results = search(stories, query=rest)
            print(f"\n  {len(results)} results for '{rest}'")
            for s in results[:10]:
                print_story(s)
            if len(results) > 10:
                print(f"\n  … and {len(results)-10} more. Refine your query.")
        elif cmd == "author":
            results = search(stories, author=rest)
            print(f"\n  {len(results)} stories by '{rest}'")
            for s in results[:20]:
                print_story(s)
        elif cmd == "cat":
            tags = rest.split()
            results = search(stories, categories=tags)
            print(f"\n  {len(results)} stories tagged: {tags}")
            for s in results[:10]:
                print_story(s)
        else:
            # treat unknown input as a search query
            results = search(stories, query=line)
            print(f"\n  {len(results)} results")
            for s in results[:10]:
                print_story(s)


def main():
    p = argparse.ArgumentParser(description="Search your local mcstories JSON index")
    p.add_argument("-i", "--index", default=str(INDEX_FILE),
                   help="Path to stories_index.json")
    p.add_argument("-q", "--query", default="", help="Full-text keyword search")
    p.add_argument("-a", "--author", default="", help="Filter by author name")
    p.add_argument("-c", "--categories", nargs="+", default=[],
                   help="Filter by category tags (all must match)")
    p.add_argument("--min-words", type=int, default=0,
                   help="Minimum word count filter")
    p.add_argument("--theme", nargs="+", default=[], metavar="THEME",
                   help="Filter by auto-detected theme(s)")
    p.add_argument("--mc-type", default="", metavar="TYPE",
                   help="Filter by mind control type (e.g. hypnosis)")
    p.add_argument("--setting", default="", help="Filter by setting (e.g. office)")
    p.add_argument("--tone", default="", help="Filter by tone (e.g. dark, romantic)")
    p.add_argument("--length", default="", dest="length_class",
                   help="Filter by length class: flash / short story / novelette / novella")
    p.add_argument("--stats", action="store_true", help="Print dataset statistics")
    p.add_argument("--export", metavar="FILE", default="",
                   help="Export results to a JSONL file")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show text preview in results")
    args = p.parse_args()

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"Index not found: {idx_path}\nRun scraper.py first.", file=sys.stderr)
        sys.exit(1)

    stories = load_index(idx_path)

    if args.stats:
        print_stats(stories)
        return

    if not any([args.query, args.author, args.categories, args.min_words,
                args.theme, args.mc_type, args.setting, args.tone, args.length_class]):
        interactive(stories)
        return

    results = search(stories, query=args.query, author=args.author,
                     categories=args.categories, min_words=args.min_words,
                     themes=args.theme, mc_type=args.mc_type,
                     setting=args.setting, tone=args.tone,
                     length_class=args.length_class)

    print(f"\n{len(results)} results")
    for s in results:
        print_story(s, verbose=args.verbose)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            for s in results:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\nExported {len(results)} stories → {args.export}")


if __name__ == "__main__":
    main()
