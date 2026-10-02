#!/usr/bin/env python3
"""
clean_text.py — clean encoding artifacts and HTML junk from scraped stories

Fixes common problems in Wayback Machine scraped text:
  - UTF-8 mojibake (â€™ → ', â€" → —, etc.)
  - Windows-1252 control characters (\x92 → ', \x97 → —, etc.)
  - Undecoded HTML entities (&rsquo; → ', &mdash; → —, etc.)
  - Wayback Machine injected toolbar HTML
  - Null bytes and non-printable control characters
  - Excessive whitespace / blank lines
  - Repeated punctuation artifacts

Can be run as a standalone script to clean all stories in the index,
or imported and called per-story.

Usage:
  python clean_text.py                    # clean all stories in index
  python clean_text.py --dry-run          # preview changes without saving
  python clean_text.py --limit 20         # test on 20 stories first
  python clean_text.py --story StorySlug  # clean one specific story
  python clean_text.py --stats            # show artifact frequency stats
"""

from __future__ import annotations

import re
import json
import html
import argparse
import unicodedata
from pathlib import Path
from typing import Optional

INDEX_PATH = Path("./output/stories_index.json")


# ── Mojibake tables ───────────────────────────────────────────────────────────
# UTF-8 bytes interpreted as Latin-1/Windows-1252 — very common in old HTML

MOJIBAKE = [
    # Double curly quotes
    ("â€œ",  "\u201c"),   # "
    ("â€\x9d", "\u201d"), # "
    ("â€˜",  "\u2018"),   # '
    ("â€™",  "\u2019"),   # '
    # Dashes — the em/en dash mojibake sequences
    ("\u00e2\u20ac\u201c",  "\u2014"),   # â€" → — em dash
    ("\u00e2\u20ac\u201d",  "\u2013"),   # â€• → – en dash
    # Ellipsis
    ("â€¦",  "\u2026"),   # …
    # Other common ones
    ("Ã©",   "é"),
    ("Ã¨",   "è"),
    ("Ã ",   "à"),
    ("Ã¢",   "â"),
    ("Ã®",   "î"),
    ("Ã´",   "ô"),
    ("Ã»",   "û"),
    ("Ã«",   "ë"),
    ("Ãª",   "ê"),
    ("Ã¯",   "ï"),
    ("Ã¼",   "ü"),
    ("Ã¶",   "ö"),
    ("Ã£",   "ã"),
    ("Ãµ",   "õ"),
    ("Ã±",   "ñ"),
    ("Ã§",   "ç"),
    ("Ã¤",   "ä"),
]

# Windows-1252 smart punctuation stored as raw bytes
WIN1252 = {
    "\x80": "€",
    "\x82": "‚",
    "\x83": "ƒ",
    "\x84": "„",
    "\x85": "…",   # ellipsis
    "\x86": "†",
    "\x87": "‡",
    "\x88": "ˆ",
    "\x89": "‰",
    "\x8a": "Š",
    "\x8b": "‹",
    "\x8c": "Œ",
    "\x91": "\u2018",  # ' left single quote
    "\x92": "\u2019",  # ' right single quote / apostrophe
    "\x93": "\u201c",  # " left double quote
    "\x94": "\u201d",  # " right double quote
    "\x95": "•",   # bullet
    "\x96": "\u2013",  # – en dash
    "\x97": "\u2014",  # — em dash
    "\x98": "˜",
    "\x99": "™",
    "\x9a": "š",
    "\x9b": "›",
    "\x9c": "œ",
    "\x9e": "ž",
    "\x9f": "Ÿ",
}

# HTML entities to decode (html.unescape handles most, these are extras)
EXTRA_ENTITIES = {
    "&apos;":   "'",
    "&#39;":    "'",
    "&#x27;":   "'",
    "&lsquo;":  "\u2018",
    "&rsquo;":  "\u2019",
    "&ldquo;":  "\u201c",
    "&rdquo;":  "\u201d",
    "&mdash;":  "\u2014",
    "&ndash;":  "\u2013",
    "&hellip;": "\u2026",
    "&nbsp;":   " ",
    "&bull;":   "•",
    "&trade;":  "™",
    "&copy;":   "©",
    "&reg;":    "®",
    "&lt;":     "<",
    "&gt;":     ">",
    "&amp;":    "&",   # must be last
}

# Wayback Machine injected patterns to strip
WAYBACK_PATTERNS = [
    re.compile(r'<!--\s*BEGIN WAYBACK TOOLBAR.*?END WAYBACK TOOLBAR[^>]*-->', re.DOTALL | re.IGNORECASE),
    re.compile(r'<div[^>]*id=["\']?wm-ipp["\']?.*?</div>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<div[^>]*id=["\']?wm-[^"\']*["\']?.*?</div>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<!-- Wayback.*?-->', re.DOTALL | re.IGNORECASE),
    re.compile(r'<script[^>]*>.*?__wm\b.*?</script>', re.DOTALL | re.IGNORECASE),
    re.compile(r'/web/\d{14}/', re.IGNORECASE),   # embedded wayback URLs in text
]

# HTML tags that might have slipped through scraping
HTML_TAG_PAT  = re.compile(r'<[^>]{1,200}>')
HTML_COMMENT  = re.compile(r'<!--.*?-->', re.DOTALL)


# ── Core clean function ───────────────────────────────────────────────────────

def clean(text: str) -> str:
    """
    Clean a story text string. Returns cleaned version.
    Applies all fixes in a safe order.
    """
    if not text:
        return text

    # 1. Strip Wayback Machine injected content
    for pat in WAYBACK_PATTERNS:
        text = pat.sub("", text)

    # 2. Strip any remaining HTML tags and comments
    text = HTML_COMMENT.sub(" ", text)
    text = HTML_TAG_PAT.sub(" ", text)

    # 3. Fix mojibake — must happen before HTML entity decoding
    for bad, good in MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)

    # 4. Fix Windows-1252 control characters
    for bad, good in WIN1252.items():
        if bad in text:
            text = text.replace(bad, good)

    # 5. Decode HTML entities
    for entity, replacement in EXTRA_ENTITIES.items():
        if entity in text:
            text = text.replace(entity, replacement)
    text = html.unescape(text)  # handles &amp; &#123; etc.

    # 6. Remove null bytes and other non-printable control chars
    # Keep: \t (tab), \n (newline), \r (carriage return)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 7. Normalise Unicode — NFC form (composed characters)
    text = unicodedata.normalize("NFC", text)

    # 8. Fix specific common replacements that survive the above
    text = text.replace("\r\n", "\n")  # Windows line endings
    text = text.replace("\r",   "\n")  # old Mac line endings

    # 9. Clean up excessive whitespace
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing whitespace on each line
    text = re.sub(r'[ \t]+\n', '\n', text)
    # Collapse multiple spaces to one (but not at line start — preserve indentation)
    text = re.sub(r'(?<=\S) {2,}', ' ', text)
    # Strip leading/trailing whitespace from whole text
    text = text.strip()

    return text


def count_artifacts(text: str) -> dict[str, int]:
    """Count how many artifacts are in a text string."""
    counts = {}
    for bad, _ in MOJIBAKE:
        if bad in text:
            counts[f"mojibake:{bad[:6]}"] = text.count(bad)
    for bad, _ in WIN1252.items():
        if bad in text:
            counts[f"win1252:{repr(bad)}"] = text.count(bad)
    for entity in EXTRA_ENTITIES:
        if entity in text:
            counts[f"entity:{entity}"] = text.count(entity)
    for i, pat in enumerate(WAYBACK_PATTERNS):
        matches = len(pat.findall(text))
        if matches:
            counts[f"wayback_pat_{i}"] = matches
    html_tags = len(HTML_TAG_PAT.findall(text))
    if html_tags:
        counts["html_tags"] = html_tags
    null_bytes = text.count('\x00')
    if null_bytes:
        counts["null_bytes"] = null_bytes
    return counts


# ── Main processing ───────────────────────────────────────────────────────────

def process_index(index_path: Path, dry_run: bool = False,
                  limit: Optional[int] = None,
                  story_id: Optional[str] = None) -> None:

    print(f"Loading {index_path}...")
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    stories = data.get("stories", [])
    print(f"Total stories: {len(stories):,}")

    cleaned    = 0
    unchanged  = 0
    total_fixes = 0
    artifact_counts: dict[str, int] = {}

    for i, story in enumerate(stories):
        # Filter by story_id if specified
        if story_id and story.get("id") != story_id:
            continue

        text = story.get("full_text", "")
        if not text:
            continue

        if limit and cleaned + unchanged >= limit:
            break

        # Count artifacts before
        artifacts = count_artifacts(text)
        if artifacts:
            for k, v in artifacts.items():
                artifact_counts[k] = artifact_counts.get(k, 0) + v

        new_text = clean(text)

        if new_text != text:
            cleaned += 1
            total_fixes += sum(artifacts.values()) if artifacts else 1
            if dry_run:
                sid = story.get("id", "?")
                print(f"\n  {sid}:")
                # Show a diff snippet
                for j, (old_line, new_line) in enumerate(
                    zip(text.split('\n'), new_text.split('\n'))
                ):
                    if old_line != new_line:
                        print(f"    Before: {repr(old_line[:80])}")
                        print(f"    After:  {repr(new_line[:80])}")
                        if j > 3:
                            print(f"    ... ({sum(1 for a,b in zip(text.split(chr(10)), new_text.split(chr(10))) if a!=b)} lines changed)")
                            break
            else:
                story["full_text"] = new_text
                # Update word count
                story["word_count"] = len(new_text.split())
        else:
            unchanged += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{len(stories):,} "
                  f"({cleaned} cleaned, {unchanged} unchanged)...")

    print(f"\nResults:")
    print(f"  Stories cleaned:   {cleaned:,}")
    print(f"  Already clean:     {unchanged:,}")
    print(f"  Total fixes:       {total_fixes:,}")

    if artifact_counts:
        print(f"\nArtifact types found:")
        for k, v in sorted(artifact_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {k:<35} {v:>6}")

    if dry_run:
        print("\nDry run — no changes saved.")
        return

    if cleaned == 0:
        print("Nothing to clean.")
        return

    # Save
    tmp = index_path.with_suffix(".tmp")
    print(f"\nSaving to {tmp}...")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Validate
    try:
        with open(tmp, encoding="utf-8") as f:
            check = json.load(f)
        assert len(check.get("stories", [])) == len(stories)
        print(f"Validation OK — {len(stories):,} stories in output")
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Validation failed — original untouched: {e}")

    tmp.replace(index_path)
    print(f"Saved to {index_path}")
    print(f"\nRun 'make rebuild-jsonl' to update training files.")


def show_stats(index_path: Path) -> None:
    """Show what artifacts exist in the index without modifying anything."""
    print(f"Scanning {index_path} for artifacts...")
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    stories = data.get("stories", [])
    total_artifacts: dict[str, int] = {}
    stories_with_artifacts = 0

    for story in stories:
        text = story.get("full_text", "")
        if not text:
            continue
        counts = count_artifacts(text)
        if counts:
            stories_with_artifacts += 1
            for k, v in counts.items():
                total_artifacts[k] = total_artifacts.get(k, 0) + v

    print(f"\nStories with artifacts: {stories_with_artifacts:,} / {len(stories):,}")
    print(f"\nArtifact types:")
    for k, v in sorted(total_artifacts.items(), key=lambda x: -x[1]):
        print(f"  {k:<40} {v:>8} occurrences")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Clean encoding artifacts from scraped story text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_text.py --stats            # see what artifacts exist
  python clean_text.py --dry-run --limit 20  # preview on 20 stories
  python clean_text.py                    # clean everything
  python clean_text.py --story AbbyAscent # clean one story

After cleaning, rebuild training files:
  make rebuild-jsonl
        """
    )
    p.add_argument("--dry-run",  action="store_true",
                   help="Preview changes without saving")
    p.add_argument("--stats",    action="store_true",
                   help="Show artifact statistics and exit")
    p.add_argument("--limit",    type=int, default=None,
                   help="Only process N stories (for testing)")
    p.add_argument("--story",    default=None,
                   help="Clean a specific story by ID/slug")
    p.add_argument("--index",    default=str(INDEX_PATH),
                   help=f"Path to index file (default: {INDEX_PATH})")

    args = p.parse_args()
    idx  = Path(args.index)

    if args.stats:
        show_stats(idx)
    else:
        process_index(idx, dry_run=args.dry_run,
                      limit=args.limit, story_id=args.story)
