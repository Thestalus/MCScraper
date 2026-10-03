#!/usr/bin/env python3
"""
refresh_text.py — re-fetch full_text for live-site stories using the fixed
                  HTML extractor, without touching wayback stories.

Why this exists:
  The original scraper used BeautifulSoup's get_text("\n", strip=True) which
  concatenated inline-element text with surrounding words when no whitespace
  existed in the HTML source (e.g. <em>surely</em>that → "surelythat").
  scraper.py now uses _extract_text() which inserts spaces around inline
  elements first.  This script re-fetches story text with that fix applied.

Safety guarantees:
  - Stories with source="wayback", "local", or "manual" are NEVER touched.
  - Only full_text and word_count are updated; all other fields (auto_tags,
    categories, preferences, etc.) are preserved exactly.
  - Resumable: already-refreshed stories are skipped on re-run.
  - Atomic saves: index is written via a .tmp file to prevent corruption.

Usage:
  .venv/bin/python3 refresh_text.py                        # refresh all
  .venv/bin/python3 refresh_text.py --dry-run              # preview only
  .venv/bin/python3 refresh_text.py --dry-run --limit 20   # preview 20
  .venv/bin/python3 refresh_text.py --story ThingOfIt      # one story
  .venv/bin/python3 refresh_text.py --resume               # skip already-done
  .venv/bin/python3 refresh_text.py --workers 3            # parallel fetches
"""

import json
import time
import logging
import argparse
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from scraper import fetch_chapter_text, get, DELAY
import index_store
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHECKPOINT_EVERY = 100  # save index after every N stories refreshed


def fetch_story_text(story: dict) -> tuple[str, int, str | None]:
    """Re-fetch full_text for one story. Returns (full_text, word_count, error)."""
    chapter_urls = story.get("chapter_urls") or []
    if not chapter_urls:
        # Fall back to deriving chapter URLs from story_url
        story_url = story.get("story_url", "")
        if not story_url:
            return "", 0, "No chapter_urls and no story_url"
        # Single-chapter: StorySlug/StorySlug.html
        from urllib.parse import urlparse
        slug = urlparse(story_url).path.strip("/").split("/")[0]
        from scraper import BASE_URL
        chapter_urls = [f"{BASE_URL}/{slug}/{slug}.html"]

    texts = []
    missing_chapters = []
    for i, ch_url in enumerate(chapter_urls, 1):
        if i > 1:
            time.sleep(DELAY)
        text = fetch_chapter_text(ch_url)
        if not text and i == 1:
            # Try numbered fallback
            fallback = ch_url.replace(".html", "1.html")
            time.sleep(DELAY)
            text = fetch_chapter_text(fallback)
        if text:
            texts.append(text)
        else:
            missing_chapters.append(i)
            log.warning(f"  ch {i} returned no text: {ch_url}")

    if not texts:
        return "", 0, "No chapter text retrieved"

    # Safety: if any chapter failed to fetch, refuse to update.
    # The story may have had chapters recovered from the Wayback Machine that
    # are no longer on the live site — overwriting would silently delete them.
    if missing_chapters:
        return "", 0, (
            f"Skipped — {len(missing_chapters)} of {len(chapter_urls)} chapter(s) "
            f"returned no text (ch {missing_chapters}). Original text preserved."
        )

    full_text = "\n<!-- BREAK -->\n".join(texts)
    from scraper import _strip_html
    word_count = len(_strip_html(full_text).split())
    return full_text, word_count, None


def main():
    """Re-download the text of live-site stories (all, one --story, or --limit N; --resume
    skips stories already refreshed) with the current HTML extractor and save through
    index_store. Wayback, local and manual stories are never touched, and a story keeps
    its old text when a chapter can't be fetched.
    """
    parser = argparse.ArgumentParser(description="Re-fetch story text with fixed HTML extractor")
    parser.add_argument("--index",   default="./output/stories_index.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without saving")
    parser.add_argument("--story",   default=None,
                        help="Only refresh this story ID")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Stop after N stories (for testing)")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip stories that already have refresh_text=true marker")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel fetch workers (default 1 — be polite to the server)")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        log.error(f"Index not found: {index_path}")
        return

    log.info(f"Loading {index_path} …")
    data = index_store.load_index(index_path, for_write=not args.dry_run)

    stories = data.get("stories", [])

    # Build work list: live-site stories only
    _PROTECTED = {"wayback", "local", "manual"}
    wayback_skipped = 0
    already_done    = 0
    work = []
    for s in stories:
        if s.get("source") in _PROTECTED:
            wayback_skipped += 1
            continue
        if args.story and s.get("id") != args.story:
            continue
        if args.resume and s.get("_text_refreshed"):
            already_done += 1
            continue
        work.append(s)

    log.info(f"Total stories in index : {len(stories):,}")
    log.info(f"Protected (skipped)    : {wayback_skipped:,}  (wayback/local/manual)")
    log.info(f"Already refreshed      : {already_done:,}")
    log.info(f"To refresh             : {len(work):,}")

    if args.limit:
        work = work[:args.limit]
        log.info(f"Limited to             : {len(work):,}")

    if args.dry_run:
        log.info("DRY RUN — no changes will be saved")
        for s in work[:20]:
            log.info(f"  Would refresh: {s.get('id')} — {s.get('title','')[:60]}")
        if len(work) > 20:
            log.info(f"  … and {len(work) - 20} more")
        return

    # Build a lookup so we can update stories in-place by id
    story_by_id = {s.get("id"): s for s in stories}

    refreshed       = 0
    failed          = 0
    unchanged       = 0
    chapter_skipped = []   # stories where a chapter was missing — text preserved

    def save_index():
        index_store.save_index(data, index_path)

    def process_one(s: dict) -> tuple[str, str, str | None]:
        """Returns (story_id, new_text, word_count, error)."""
        new_text, new_wc, error = fetch_story_text(s)
        return s["id"], new_text, new_wc, error

    log.info("Starting refresh …")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, s): s for s in work}
        done_count = 0
        for future in as_completed(futures):
            story_id, new_text, new_wc, error = future.result()
            done_count += 1
            s = story_by_id.get(story_id)
            if s is None:
                continue

            if error:
                if "Skipped" in error:
                    # Chapter(s) missing from live site — original text kept intact
                    log.warning(f"  [{done_count}/{len(work)}] SKIP  {story_id}: {error}")
                    chapter_skipped.append((story_id, s.get("title", ""), error))
                    # Still mark as visited so --resume doesn't retry endlessly
                    s["_text_refreshed"] = True
                else:
                    log.warning(f"  [{done_count}/{len(work)}] FAIL  {story_id}: {error}")
                    failed += 1
            else:
                old_text = s.get("full_text", "")
                if new_text == old_text:
                    unchanged += 1
                    log.debug(f"  [{done_count}/{len(work)}] SAME  {story_id}")
                else:
                    delta = new_wc - len(old_text.split())
                    log.info(f"  [{done_count}/{len(work)}] OK    {story_id}"
                             f"  (Δ{delta:+d} words)")
                    s["full_text"]  = new_text
                    s["word_count"] = new_wc
                    refreshed += 1
                s["_text_refreshed"] = True

            # Checkpoint
            if done_count % CHECKPOINT_EVERY == 0:
                save_index()
                log.info(f"  Checkpoint saved — {done_count}/{len(work)} processed")

            # Polite delay between requests (only needed for sequential)
            if args.workers == 1:
                time.sleep(DELAY)

    save_index()

    # Write a report of stories whose chapters were partially missing
    if chapter_skipped:
        skip_path = index_path.parent / "refresh_text_skipped.txt"
        with open(skip_path, "w", encoding="utf-8") as f:
            f.write(f"Stories skipped by refresh_text.py — chapter(s) missing from live site\n")
            f.write(f"Original full_text preserved for all {len(chapter_skipped)} stories below.\n")
            f.write("=" * 70 + "\n\n")
            for sid, title, reason in chapter_skipped:
                f.write(f"{sid}  |  {title}\n  {reason}\n\n")
        log.info(f"Skipped stories written to {skip_path}")

    log.info("")
    log.info("Refresh complete:")
    log.info(f"  Updated          : {refreshed:,}")
    log.info(f"  Unchanged        : {unchanged:,}")
    log.info(f"  Chapter-skipped  : {len(chapter_skipped):,}  (original text kept — see refresh_text_skipped.txt)")
    log.info(f"  Failed (network) : {failed:,}")
    log.info(f"Index saved to {index_path}")
    log.info("")
    log.info("Next step: run  make post-update  to re-tag and rebuild caches.")


if __name__ == "__main__":
    main()
