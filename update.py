#!/usr/bin/env python3
"""
update.py — weekly incremental update for mcstories.com

Much faster than running the full scraper weekly. Uses the site's own
/WhatsNew.html page to find only stories added since your last update,
scrapes just those, and merges them into your existing index.

The site updates ~weekly. Run this once a week to stay current.

Usage:
  make update                    # check and scrape new stories
  make update --dry-run          # see what's new without scraping
  python update.py               # same as make update
  python update.py --dry-run
  python update.py --weeks 4     # look back 4 weeks instead of 1
  python update.py --full-check  # fall back to full letter-page crawl if needed

How it works:
  1. Fetch /WhatsNew.html — the site's own "recently added" page
  2. Extract story landing URLs from it
  3. Compare against your existing stories_index.json
  4. Scrape only the new ones
  5. Merge into index + append to JSONL training files
  6. Re-tag new stories with regex
  7. Run SetFit tagging pass over the updated index

The "New Stories" page in the browser shows whatever was added in the
most recent update run.
"""

import re
import json
import time
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from tag_trust import export_tags
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
import index_store
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL      = "https://mcstories.com"
WHATSNEW_URL  = f"{BASE_URL}/WhatsNew.html"
HEADERS       = {"User-Agent": "Mozilla/5.0 (compatible; story-archiver/1.0; +research)"}
DELAY         = 1.5
MAX_RETRIES   = 3

OUTPUT_DIR    = Path("./output")
INDEX_PATH    = OUTPUT_DIR / "stories_index.json"
UPDATE_LOG    = OUTPUT_DIR / "update_log.json"

NAV_SLUGS = {
    "titles", "authors", "tags", "readerspicks", "whatsnew",
    "whats_new", "faq", "forum", "garden", "recentadditions"
}

log = logging.getLogger(__name__)
session = requests.Session()
session.headers.update(HEADERS)


# ── HTTP ──────────────────────────────────────────────────────────────────────

def get(url: str) -> BeautifulSoup | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log.warning(f"[attempt {attempt+1}] {url}: {e}")
            time.sleep(DELAY * (attempt + 1))
    return None


# ── Discover new story URLs from WhatsNew ─────────────────────────────────────

def get_whatsnew_urls(weeks: int = 2) -> list[tuple[str, str]]:
    """
    Fetch /WhatsNew.html and return list of (landing_url, date_string) tuples
    for stories added recently.

    The WhatsNew page lists stories with their dates — we grab all of them
    and let the caller filter against the existing index.
    """
    log.info(f"Fetching {WHATSNEW_URL}...")
    soup = get(WHATSNEW_URL)
    if not soup:
        log.error("Could not fetch WhatsNew page")
        return []

    results = []
    seen = set()

    # WhatsNew page structure: story links with /StorySlug/index.html pattern
    # alongside date text — we collect both
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(WHATSNEW_URL, href)

        # Match story landing pages
        m = re.match(r"https://mcstories\.com/([^/]+)/index\.html$", full, re.I)
        if not m:
            # Also try relative patterns like ../StorySlug/index.html
            m2 = re.match(r"\.\./([^/]+)/index\.html$", href, re.I)
            if m2:
                slug = m2.group(1)
                full = f"{BASE_URL}/{slug}/index.html"
                m = re.match(r"https://mcstories\.com/([^/]+)/index\.html$", full, re.I)

        if not m:
            continue

        slug = m.group(1)
        if slug.lower() in NAV_SLUGS:
            continue
        if full in seen:
            continue
        seen.add(full)

        # Try to find a nearby date in the page text
        date_str = ""
        parent = a.parent
        for _ in range(4):
            if parent is None:
                break
            text = parent.get_text(" ", strip=True)
            date_m = re.search(r"\d{1,2}\s+\w+\s+\d{4}", text)
            if date_m:
                date_str = date_m.group(0)
                break
            parent = parent.parent

        results.append((full, date_str))

    log.info(f"WhatsNew page: {len(results)} story links found")
    return results


def get_new_story_urls_fallback() -> list[str]:
    """
    Fallback: crawl the first few letter pages (A, B, C...) looking for
    story slugs not in the existing index. Slower but catches anything
    WhatsNew misses (e.g. if it only shows 2 weeks of history).
    """
    log.info("Fallback: checking letter pages for new stories...")
    from scraper import get_letter_urls, get_story_urls_from_letter

    existing = load_existing_slugs()
    letter_urls = get_letter_urls()
    new_urls = []

    for letter_url in letter_urls:
        for url in get_story_urls_from_letter(letter_url):
            slug = url.rstrip("/").split("/")[-2]
            if slug not in existing:
                new_urls.append(url)
        time.sleep(DELAY)
        if len(new_urls) > 100:
            break  # safety valve — if there are 100+ new stories something is wrong

    return new_urls


# ── Index management ──────────────────────────────────────────────────────────

def load_existing_stories() -> dict[str, dict]:
    """Load existing story id → {chapter_count, chapter_urls} without loading full_text.

    Returns a dict keyed by story id (slug) with just the metadata needed to
    detect chapter additions. full_text is intentionally excluded to keep memory use low.
    """
    if not INDEX_PATH.exists():
        return {}

    stories: dict[str, dict] = {}

    try:
        import ijson
        with open(INDEX_PATH, "rb") as f:
            for item in ijson.items(f, "stories.item"):
                sid = item.get("id")
                if sid:
                    stories[sid] = {
                        "chapter_count": item.get("chapter_count", 1),
                        "chapter_urls":  item.get("chapter_urls", []),
                        "source":        item.get("source"),
                    }
        return stories
    except ImportError:
        pass

    # Fallback: full JSON load (ijson not installed)
    log.info("ijson not available — loading full index to read chapter counts")
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for s in data.get("stories", []):
        sid = s.get("id")
        if sid:
            stories[sid] = {
                "chapter_count": s.get("chapter_count", 1),
                "chapter_urls":  s.get("chapter_urls", []),
                "source":        s.get("source"),
            }
    return stories


def load_existing_slugs() -> set[str]:
    return set(load_existing_stories().keys())


def fetch_new_chapters(landing_url: str, existing_chapter_count: int,
                       existing_chapter_urls: list) -> dict | None:
    """
    For an existing story that appeared in WhatsNew, check if it has new chapters.
    If so, fetches only the new chapters and returns a partial update dict:
      {"id", "new_chapters_text", "new_chapter_urls", "new_chapter_count"}
    Returns None if no new chapters were found.
    """
    from scraper import parse_landing, fetch_chapter_text, make_id

    meta = parse_landing(landing_url)
    if meta.get("fetch_error"):
        log.warning(f"Landing page error for {landing_url}: {meta['fetch_error']}")
        return None

    site_chapter_urls  = meta.get("chapter_urls", [])
    site_chapter_count = len(site_chapter_urls)

    if site_chapter_count <= existing_chapter_count:
        return None  # no new chapters

    new_urls = site_chapter_urls[existing_chapter_count:]
    log.info(f"  {make_id(landing_url)}: {existing_chapter_count} → {site_chapter_count} chapters "
             f"({len(new_urls)} new)")

    texts = []
    for i, ch_url in enumerate(new_urls):
        if i > 0:
            time.sleep(DELAY)
        chapter_num = existing_chapter_count + i + 1
        log.info(f"    ch {chapter_num}/{site_chapter_count}: {ch_url.split('/')[-1]}")
        text = fetch_chapter_text(ch_url)
        if text:
            texts.append(text)
        else:
            log.warning(f"    ch {chapter_num} returned no text")

    if not texts:
        return None

    return {
        "id":                make_id(landing_url),
        "new_chapters_text": "\n\n* * *\n\n".join(texts),
        "new_chapter_urls":  new_urls,
        "new_chapter_count": site_chapter_count,
        "updated_date":      meta.get("updated_date", ""),
    }


def apply_chapter_updates(chapter_updates: list[dict]) -> int:
    """
    Append new chapter text to existing stories in the index.
    Returns the number of stories updated.
    """
    if not chapter_updates or not INDEX_PATH.exists():
        return 0

    log.info(f"Loading index to apply {len(chapter_updates)} chapter updates...")
    data = index_store.load_index(INDEX_PATH, for_write=True)

    updates_by_id = {u["id"]: u for u in chapter_updates}
    applied = 0
    for s in data["stories"]:
        upd = updates_by_id.get(s.get("id"))
        if not upd:
            continue
        sep = "\n<!-- BREAK -->\n" if (s.get("full_text") or "").lstrip().startswith("<") else "\n\n* * *\n\n"
        s["full_text"]     = (s.get("full_text") or "") + sep + upd["new_chapters_text"]
        from scraper import _strip_html
        s["word_count"]    = len(_strip_html(s["full_text"]).split())
        s["chapter_count"] = upd["new_chapter_count"]
        s["chapter_urls"]  = (s.get("chapter_urls") or []) + upd["new_chapter_urls"]
        if upd.get("updated_date"):
            s["updated_date"] = upd["updated_date"]
        # Re-tag with regex since there's new text. tag_story() returns a fresh
        # regex-only dict; SetFit/Svengali tags come back on the next
        # `make post-update`, but arc results (hours of LLM time) would not —
        # so carry them over.
        if s.get("full_text") and not s.get("fetch_error"):
            from tagger import tag_story
            old_at = s.get("auto_tags") or {}
            s["auto_tags"] = tag_story(s)
            for key in ("narrative_arc", "subject_gender", "_arc_error"):
                if key in old_at:
                    s["auto_tags"][key] = old_at[key]
        applied += 1

    if applied == 0:
        return 0

    index_store.save_index(data, INDEX_PATH, validate=True)
    log.info(f"Applied chapter updates to {applied} existing stories")
    return applied


def merge_new_stories(new_stories: list[dict]) -> int:
    """
    Safely append new stories to stories_index.json.
    Loads the full index into memory — safe and simple.
    Writes to .tmp first and only replaces the original once
    the new file is confirmed valid JSON.

    Returns the number of genuinely new stories added to the index.
    """
    if not new_stories:
        return 0

    # Tag new stories with regex layer
    from tagger import tag_story

    tagged_stories = []
    for s in new_stories:
        if s.get("full_text") and not s.get("fetch_error"):
            s["auto_tags"] = tag_story(s)
        tagged_stories.append(s)

    if not INDEX_PATH.exists():
        to_add = tagged_stories
        data = {
            "source": BASE_URL,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_stories": len(tagged_stories),
            "stories": tagged_stories,
        }
    else:
        log.info(f"Loading {INDEX_PATH} ({INDEX_PATH.stat().st_size // 1024 // 1024}MB)...")
        data = index_store.load_index(INDEX_PATH, for_write=True)

        existing_ids = {s["id"] for s in data.get("stories", [])}
        to_add = [s for s in tagged_stories if s.get("id") not in existing_ids]
        log.info(f"Adding {len(to_add)} new stories to {len(existing_ids):,} existing")

        data["stories"].extend(to_add)
        data["total_stories"] = len(data["stories"])
        data["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Validated, atomic save — never touches the original until the new file
    # is confirmed to hold every story
    try:
        index_store.save_index(data, INDEX_PATH, validate=True)
    except RuntimeError as e:
        raise RuntimeError(f"Merge validation failed — original index untouched: {e}")
    log.info(f"Merged {len(to_add)} genuinely new stories into {INDEX_PATH}")

    # Append only newly-added stories to JSONL training files
    # Use dict access — stories here are plain dicts, not Story dataclasses
    try:
        with open(OUTPUT_DIR / "training_raw_pretraining.jsonl", "a", encoding="utf-8") as f1, \
             open(OUTPUT_DIR / "training_instruction_response.jsonl", "a", encoding="utf-8") as f2, \
             open(OUTPUT_DIR / "training_completion.jsonl", "a", encoding="utf-8") as f3:

            for s in to_add:
                if not s.get("full_text") or s.get("fetch_error"):
                    continue
                at    = export_tags(s.get("auto_tags"))
                cats  = ", ".join(s.get("categories", []))
                title = s.get("title", "")
                auth  = s.get("author", "")
                desc  = s.get("description", "")
                from scraper import _strip_html
                text  = _strip_html(s["full_text"])

                # Raw pretraining
                parts = [f"Title: {title}", f"Author: {auth}"]
                if cats:                    parts.append(f"Categories: {cats}")
                if at.get("themes"):        parts.append(f"Themes: {', '.join(at['themes'])}")
                if at.get("mind_control_type"): parts.append(f"Mind Control Method: {', '.join(at['mind_control_type'])}")
                if at.get("setting"):       parts.append(f"Setting: {', '.join(at['setting'])}")
                if at.get("tone"):          parts.append(f"Tone: {', '.join(at['tone'])}")
                if desc:                    parts.append(f"Summary: {desc}")
                f1.write(json.dumps({"text": "\n".join(parts) + "\n\n" + text}, ensure_ascii=False) + "\n")

                # Instruction/response
                msg = f'Write a {cats or "fiction"} story titled "{title}" by {auth}.'
                if at.get("themes"):            msg += f" Themes: {', '.join(at['themes'])}."
                if at.get("mind_control_type"): msg += f" Mind control method: {', '.join(at['mind_control_type'])}."
                if at.get("setting"):           msg += f" Setting: {', '.join(at['setting'])}."
                f2.write(json.dumps({"messages": [
                    {"role": "system",    "content": "You are a skilled fiction writer specializing in mind control erotica."},
                    {"role": "user",      "content": msg},
                    {"role": "assistant", "content": text},
                ]}, ensure_ascii=False) + "\n")

                # Completion
                cp = [f"Title: {title}", f"Author: {auth}", f"Categories: {cats or 'fiction'}"]
                if at.get("themes"):            cp.append(f"Themes: {', '.join(at['themes'])}")
                if at.get("mind_control_type"): cp.append(f"Mind Control Method: {', '.join(at['mind_control_type'])}")
                f3.write(json.dumps({"prompt": "\n".join(cp) + "\n\n", "completion": text}, ensure_ascii=False) + "\n")

    except Exception as e:
        log.warning(f"JSONL append failed (index is fine): {e}")

    return len(to_add)


def run_setfit_tagging(output_dir: Path) -> bool:
    """
    Run SetFit tagging as a second pass after new stories are merged.

    This calls setfit_trainer.py --tag using the same Python interpreter.
    Safe behavior:
      - If the SetFit script is missing, skip with a warning.
      - If the trained model/classifiers are missing, skip with a warning.
      - If the subprocess fails, keep the update results and report the error.
    """
    setfit_script = Path(__file__).with_name("setfit_trainer.py")
    clf_path = output_dir / "setfit_classifiers.pkl"
    model_dir = output_dir / "setfit_model"

    if not setfit_script.exists():
        log.warning("SetFit step skipped: setfit_trainer.py not found")
        return False

    if not clf_path.exists():
        log.warning(f"SetFit step skipped: classifiers not found at {clf_path}")
        return False

    if not model_dir.exists():
        log.warning(f"SetFit step skipped: model directory not found at {model_dir}")
        return False

    cmd = [
        sys.executable,
        str(setfit_script),
        "--tag",
    ]

    log.info("")
    log.info("Running SetFit tagging pass on updated index...")
    log.info("Command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            check=True,
        )
        log.info("SetFit tagging complete")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log.warning(f"SetFit tagging failed (update still succeeded): exit code {e.returncode}")
        return False
    except Exception as e:
        log.warning(f"SetFit tagging failed (update still succeeded): {e}")
        return False


# ── Update log ────────────────────────────────────────────────────────────────

def load_update_log() -> dict:
    if UPDATE_LOG.exists():
        with open(UPDATE_LOG, encoding="utf-8") as f:
            return json.load(f)
    return {"runs": [], "last_seen_slugs": []}


def save_update_log(log_data: dict):
    with open(UPDATE_LOG, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def _save_run_log(ok: int, err: int, added_to_index: int, setfit_ok: bool,
                  chapters_updated: int = 0,
                  new_story_ids: list | None = None,
                  chapter_updated_ids: list | None = None):
    log_data = load_update_log()
    log_data["runs"].append({
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "stories_scraped_ok":   ok,
        "errors":               err,
        "stories_added_to_index": added_to_index,
        "chapters_updated":     chapters_updated,
        "setfit_applied":       setfit_ok,
        "new_story_ids":        new_story_ids or [],
        "chapter_updated_ids":  chapter_updated_ids or [],
    })
    # Keep last 52 runs (one year of weekly updates)
    log_data["runs"] = log_data["runs"][-52:]
    save_update_log(log_data)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    global OUTPUT_DIR, INDEX_PATH, UPDATE_LOG

    OUTPUT_DIR = Path(args.output)
    INDEX_PATH = OUTPUT_DIR / "stories_index.json"
    UPDATE_LOG = OUTPUT_DIR / "update_log.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "update.log"),
        ]
    )

    log.info("=" * 60)
    log.info("mcstories.com weekly update check")
    log.info("=" * 60)

    # Step 1: get candidate URLs from WhatsNew
    candidates = get_whatsnew_urls(weeks=args.weeks)

    if not candidates and args.full_check:
        log.info("WhatsNew returned nothing — falling back to full letter crawl")
        candidates = [(url, "") for url in get_new_story_urls_fallback()]

    if not candidates:
        log.info("No new story URLs found. Site may not have updated yet.")
        return

    # Step 2: split candidates into brand-new stories vs existing stories to check for new chapters
    existing = load_existing_stories()
    log.info(f"Existing stories in index: {len(existing):,}")

    new_urls      = []  # slugs not in index at all
    existing_urls = []  # slugs already in index — check for new chapters

    # Sources that must never be overwritten or have chapters fetched from mcstories.com
    _PROTECTED_SOURCES = {"wayback", "manual", "local"}

    for url, date in candidates:
        slug = url.rstrip("/").split("/")[-2]
        if slug not in existing:
            new_urls.append((url, date))
        else:
            meta = existing[slug]
            if meta.get("source") in _PROTECTED_SOURCES:
                log.info(f"  Skipping {slug} (source={meta['source']!r} — protected)")
                continue
            existing_urls.append((url, meta))

    log.info(f"Truly new stories:          {len(new_urls)}")
    log.info(f"Existing stories in WhatsNew (checking for new chapters): {len(existing_urls)}")

    if not new_urls and not existing_urls:
        log.info("No new stories — index is up to date.")
        _save_run_log(0, 0, 0, False)
        return

    if args.dry_run:
        if new_urls:
            print(f"\nDry run — {len(new_urls)} new stories would be scraped:")
            for url, date in new_urls:
                slug = url.rstrip("/").split("/")[-2]
                print(f"  {slug:<45} {date}")
        if existing_urls:
            print(f"\nDry run — {len(existing_urls)} existing stories in WhatsNew (may have new chapters):")
            for url, meta in existing_urls:
                slug = url.rstrip("/").split("/")[-2]
                print(f"  {slug:<45} (currently {meta['chapter_count']} ch stored)")
        return

    # Step 3a: scrape brand-new stories (full fetch)
    from scraper import fetch_full_story, make_id

    new_stories = []
    ok = 0
    err = 0

    if new_urls:
        log.info(f"\nScraping {len(new_urls)} new stories...")
        for i, (url, date) in enumerate(new_urls, 1):
            slug = url.rstrip("/").split("/")[-2]
            log.info(f"[{i}/{len(new_urls)}] {slug}")

            data = fetch_full_story(url)
            story = {
                "id":           make_id(url),
                "title":        data.get("title", ""),
                "author":       data.get("author", ""),
                "author_url":   data.get("author_url", ""),
                "story_url":    url,
                "categories":   data.get("categories", []),
                "description":  data.get("description", ""),
                "word_count":   data.get("word_count", 0),
                "added_date":   data.get("added_date", "") or date,
                "updated_date": data.get("updated_date", ""),
                "full_text":    data.get("full_text", ""),
                "fetch_error":  data.get("fetch_error"),
                "chapter_count": data.get("chapter_count", 1),
                "chapter_urls": data.get("chapter_urls", []),
                "auto_tags":    None,
                "source":       "update",
            }
            new_stories.append(story)

            if story.get("full_text") and not story.get("fetch_error"):
                ok += 1
            else:
                err += 1

            time.sleep(DELAY)

    # Step 3b: check existing stories in WhatsNew for new chapters (landing page only)
    chapter_updates = []
    if existing_urls:
        log.info(f"\nChecking {len(existing_urls)} existing stories for new chapters...")
        for i, (url, meta) in enumerate(existing_urls, 1):
            slug = url.rstrip("/").split("/")[-2]
            log.info(f"[{i}/{len(existing_urls)}] {slug} (stored: {meta['chapter_count']} ch)")
            upd = fetch_new_chapters(url, meta["chapter_count"], meta["chapter_urls"])
            if upd:
                chapter_updates.append(upd)
            time.sleep(DELAY)

        log.info(f"Found new chapters in {len(chapter_updates)} existing stories")

    # Step 4a: merge new stories into index (regex tagging happens inside merge_new_stories)
    added_to_index = 0
    if new_stories:
        log.info(f"\nMerging {ok} new stories into index...")
        added_to_index = merge_new_stories(new_stories)

    # Step 4b: apply chapter updates to existing stories
    chapters_updated = 0
    if chapter_updates:
        log.info(f"\nApplying new chapters to {len(chapter_updates)} existing stories...")
        chapters_updated = apply_chapter_updates(chapter_updates)

    # Step 5: run SetFit tagging as a second pass
    setfit_ok = False
    if (added_to_index > 0 or chapters_updated > 0) and not args.skip_setfit:
        setfit_ok = run_setfit_tagging(OUTPUT_DIR)
    elif args.skip_setfit:
        log.info("SetFit step skipped by --skip-setfit")

    # Step 6: update log — browser uses this for "New Stories" page
    new_ids      = [s["id"] for s in new_stories if s.get("id") and not s.get("fetch_error")]
    ch_upd_ids   = [u["id"] for u in chapter_updates]
    _save_run_log(ok, err, added_to_index, setfit_ok, chapters_updated,
                  new_story_ids=new_ids, chapter_updated_ids=ch_upd_ids)

    log.info("=" * 60)
    log.info(f"Update complete: {added_to_index} new stories added, "
             f"{chapters_updated} existing stories got new chapters, {err} errors")

    if setfit_ok:
        log.info("New stories received regex + SetFit tags")
    else:
        log.info("New stories received regex tags only")

    log.info("View new stories in browser: New Stories (● New) tab")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Weekly incremental update for mcstories.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update.py                 # check for new stories and scrape them
  python update.py --dry-run       # show what's new without scraping
  python update.py --weeks 4       # look back 4 weeks (catches up if you missed some)
  python update.py --full-check    # use full letter crawl if WhatsNew is empty
  python update.py --skip-setfit   # scrape + regex-tag only

Schedule weekly on Mac:
  crontab -e
  0 9 * * 1 cd /path/to/MCScraper && .venv/bin/python3 update.py >> output/cron.log 2>&1

Schedule weekly on Windows (Task Scheduler):
  Action: .venv\\Scripts\\python.exe update.py
  Start in: C:\\MCTraining
  Trigger: Weekly, Monday 9:00 AM
        """
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--weeks", type=int, default=2,
                   help="How many weeks back to check (default: 2)")
    p.add_argument("--full-check", action="store_true",
                   help="Fall back to full letter crawl if WhatsNew is empty")
    p.add_argument("--skip-setfit", action="store_true",
                   help="Skip SetFit tagging after merge")
    p.add_argument("--output", default="./output")
    args = p.parse_args()

    run(args)
