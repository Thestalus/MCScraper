#!/usr/bin/env python3
"""
export_stories.py — Pull specific stories out of the scraped archive into one text file.

Give it a list of story titles or ids (as printed on mcstories.com, or the slug used
in the story's URL) and it looks each one up in output/stories_meta.db +
output/full_text_cache.db and writes the matches into a single .txt file.

Usage:
    python export_stories.py "Corrupting Little Sister" "Love Pill" -o my_stories.txt
    python export_stories.py --from-file wanted.txt -o my_stories.txt
    python export_stories.py --author "Jay Inkwell" -o jay_inkwell.txt

    # wanted.txt: one title or id per line, blank lines and "#" comments ignored.
    # --author matches the stored author name exactly (case-insensitive) and pulls
    # every story credited to them, in title order. Combine with explicit names/
    # --from-file to build one file out of several authors' catalogs plus one-offs.

Matching order per name: exact id -> exact title (case-insensitive) -> unique
substring match on title. Anything ambiguous or not found is reported and skipped
rather than guessed.
"""
from __future__ import annotations
import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
META_DB = ROOT / "output" / "stories_meta.db"
TEXT_DB = ROOT / "output" / "full_text_cache.db"


def _norm(s: str) -> str:
    """Lowercase and strip everything but letters/digits, so 'Judas Goat' matches
    the stored title 'JudasGoat' (most titles in this DB have no spaces/punctuation)."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _norm_author(s: str) -> str:
    """Lowercase and fold curly quotes to straight ones (data has e.g. 'Alecta’s Shadow')."""
    return s.lower().replace("’", "'").replace("‘", "'").strip()


def load_stories() -> list[dict]:
    """Every story's id, title, author, URL and word count from stories_meta.db."""
    conn = sqlite3.connect(META_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, author, story_url, word_count, has_text FROM stories"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_story(name: str, stories: list[dict]) -> dict | None:
    """Resolve one requested name to a story row, or None (printing why) if it can't be."""
    by_id = {s["id"]: s for s in stories}
    if name in by_id:
        return by_id[name]

    name_norm = _norm(name)
    exact_title = [s for s in stories if _norm(s["title"]) == name_norm]
    if len(exact_title) == 1:
        return exact_title[0]
    if len(exact_title) > 1:
        print(f"  ✗ '{name}': ambiguous — {len(exact_title)} stories share this title "
              f"({', '.join(s['id'] for s in exact_title)}). Use the id instead.")
        return None

    substr = [s for s in stories if name_norm in _norm(s["title"])]
    if len(substr) == 1:
        return substr[0]
    if len(substr) > 1:
        print(f"  ✗ '{name}': matches {len(substr)} titles — be more specific, or use an id:")
        for s in substr[:10]:
            print(f"      {s['id']}  —  {s['title']}")
        if len(substr) > 10:
            print(f"      ... and {len(substr) - 10} more")
        return None

    print(f"  ✗ '{name}': no story found")
    return None


def get_full_text(story_id: str) -> str:
    """A story's text from full_text_cache.db ('' if missing)."""
    conn = sqlite3.connect(TEXT_DB)
    row = conn.execute("SELECT full_text FROM texts WHERE id=?", (story_id,)).fetchone()
    conn.close()
    return row[0] if row else ""


def main():
    """Match the requested titles/ids/authors against the archive and write the stories
    into one text file.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="Story titles or ids to export")
    parser.add_argument("--from-file", "-f", help="Text file with one title/id per line")
    parser.add_argument("--author", "-a", action="append", default=[],
                         help="Pull every story by this author (exact name, case-insensitive). Repeatable.")
    parser.add_argument("--output", "-o", default="exported_stories.txt", help="Output text file path")
    args = parser.parse_args()

    names = list(args.names)
    if args.from_file:
        for line in Path(args.from_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)

    if not names and not args.author:
        parser.error("give at least one story name/id, --from-file, or --author")

    if not META_DB.exists() or not TEXT_DB.exists():
        sys.exit(f"Expected databases not found under {ROOT / 'output'}")

    stories = load_stories()
    print(f"Loaded {len(stories):,} stories from {META_DB.name}\n")

    matched: list[dict] = []
    seen_ids: set[str] = set()

    for author in args.author:
        by_author = sorted([s for s in stories if _norm_author(s["author"]) == _norm_author(author)],
                            key=lambda s: s["title"])
        if not by_author:
            print(f"  ✗ author '{author}': no stories found")
            continue
        print(f"  ✓ author '{author}': {len(by_author)} stories")
        for s in by_author:
            if s["id"] not in seen_ids:
                matched.append(s)
                seen_ids.add(s["id"])

    for name in names:
        story = find_story(name, stories)
        if story:
            print(f"  ✓ '{name}' -> {story['id']}  ({story['title']} — {story['author']})")
            if story["id"] not in seen_ids:
                matched.append(story)
                seen_ids.add(story["id"])

    if not matched:
        sys.exit("\nNo stories matched — nothing written.")

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as out:
        for i, s in enumerate(matched):
            if i:
                out.write("\n\n" + "=" * 80 + "\n\n")
            out.write(f"Title: {s['title']}\n")
            out.write(f"Author: {s['author']}\n")
            out.write(f"URL: {s['story_url']}\n")
            out.write(f"Word count: {s['word_count']}\n")
            out.write("-" * 80 + "\n\n")
            text = get_full_text(s["id"]) if s["has_text"] else ""
            out.write(text if text else "[no cached full text for this story]")

    print(f"\nWrote {len(matched)} stories to {out_path.resolve()}")


if __name__ == "__main__":
    main()
