#!/usr/bin/env python3
"""
wayback_scraper.py — scrape deleted mcstories.com stories from the Internet Archive

Workflow:
  1. Query the Wayback CDX API to find all story landing pages captured
     on/near a given date
  2. Compare against your existing stories_index.json to find stories
     that have since been deleted from the live site
  3. Scrape the archived versions of those deleted stories
  4. Merge them into a separate wayback_index.json (or optionally into
     the main index with source="wayback")

The CDX API is free, rate-limit friendly, and returns structured data —
no HTML scraping of archive.org itself needed for the discovery phase.

Usage:
  # Find and scrape stories deleted since 2020
  .venv/bin/python3 wayback_scraper.py --date 20200101

  # Broader search — scan multiple yearly snapshots
  .venv/bin/python3 wayback_scraper.py --date 20180101 --date 20200101 --date 20220101

  # Dry run — just show what would be scraped
  .venv/bin/python3 wayback_scraper.py --date 20200101 --dry-run

  # Merge results into main index when done
  .venv/bin/python3 wayback_scraper.py --merge

Via Makefile:
  make wayback DATE=20200101
"""

import re
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import index_store
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL      = "https://mcstories.com"
WB_BASE       = "https://web.archive.org"
CDX_API       = "https://web.archive.org/cdx/search/cdx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; story-archiver/1.0; +research)",
}
DELAY       = 2.0   # be polite to archive.org — they ask for >=1s between requests
MAX_RETRIES = 2   # wayback 404s are definitive — fewer retries saves time

NAV_SLUGS = {"titles", "authors", "tags", "readerspicks", "whatsnew",
             "whats_new", "faq", "forum", "garden", "recentadditions"}

OUTPUT_DIR    = Path("./output")
MAIN_INDEX    = OUTPUT_DIR / "stories_index.json"
WAYBACK_INDEX = OUTPUT_DIR / "wayback_index.json"

log = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WaybackStory:
    id:            str
    title:         str
    author:        str
    author_url:    str
    story_url:     str          # original mcstories.com URL
    wayback_url:   str          # archive.org URL used to scrape
    snapshot_date: str          # YYYYMMDDHHMMSS
    categories:    list[str]    = field(default_factory=list)
    description:   str          = ""
    word_count:    int          = 0
    added_date:    str          = ""
    updated_date:  str          = ""
    full_text:     str          = ""
    fetch_error:   Optional[str] = None
    auto_tags:     Optional[dict] = None
    chapter_count: int           = 1
    chapter_urls:  list[str]     = field(default_factory=list)
    source:        str           = "wayback"


# ── HTTP ──────────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url: str, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=30)
            # 404 = snapshot doesn't exist — no point retrying
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            # Don't retry on 4xx client errors
            if hasattr(e, 'response') and e.response is not None:
                if 400 <= e.response.status_code < 500:
                    return None
            log.warning(f"[attempt {attempt+1}/{retries}] {url}: {e}")
            time.sleep(DELAY * (attempt + 1))
    return None

def get_json(url: str) -> Optional[dict | list]:
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning(f"[CDX attempt {attempt+1}] {url}: {e}")
            time.sleep(DELAY * (attempt + 1))
    return None


# ── CDX API: discover archived story URLs ────────────────────────────────────

def cdx_get_story_slugs(target_date: str) -> dict[str, str]:
    """
    Query the CDX API for all mcstories.com story slugs archived near target_date.

    Uses text output format streamed line-by-line to avoid truncation on large
    responses. Each line is "original timestamp" space-separated.
    """
    log.info(f"Querying CDX API for mcstories.com near {target_date}...")

    year  = target_date[:4]

    # Use text output (not JSON) — avoids truncated-JSON errors on large responses.
    # Stream line by line so memory usage stays low.
    # Narrow to a ±6 month window to keep response size manageable.
    from_date = f"{int(year)-1}0101"
    to_date   = f"{int(year)+1}1231"

    url = (
        f"{CDX_API}"
        f"?url=mcstories.com"
        f"&matchType=domain"
        f"&filter=statuscode:200"
        f"&fl=original,timestamp"
        f"&output=text"
        f"&collapse=urlkey"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&limit=500000"
    )
    log.info(f"CDX: {url}")

    slugs: dict[str, str] = {}
    total_lines = 0
    try:
        with session.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(chunk_size=8192):
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                total_lines += 1

                # Each line: "http://www.mcstories.com:80/Slug/Slug.html 20150209074734"
                parts = line.split(" ")
                if len(parts) < 2:
                    continue
                original, timestamp = parts[0], parts[1]

                m = re.match(
                    r"https?://(?:www\.)?mcstories\.com(?::\d+)?/([^/?#]+)/([^/?#]+)\.html",
                    original, re.I
                )
                if not m:
                    continue

                dir_slug  = m.group(1)
                file_name = m.group(2)

                if dir_slug.lower() in NAV_SLUGS:
                    continue
                if not file_name.lower().startswith(dir_slug.lower()):
                    continue
                suffix = file_name[len(dir_slug):]
                if suffix and not suffix.isdigit():
                    continue

                # Prefer chapter-1 (no number) as canonical timestamp
                if suffix == "":
                    slugs[dir_slug] = timestamp
                elif dir_slug not in slugs:
                    slugs[dir_slug] = timestamp

    except Exception as e:
        log.error(f"CDX stream failed: {e}")
        return {}

    log.info(f"CDX: processed {total_lines:,} lines, found {len(slugs):,} story slugs")
    return slugs


def find_closest_snapshot(slug: str, target_date: str) -> Optional[str]:
    """Find the closest archived timestamp for a specific story."""
    for pattern in [
        f"www.mcstories.com:80/{slug}/{slug}.html",
        f"www.mcstories.com/{slug}/{slug}.html",
        f"mcstories.com/{slug}/{slug}.html",
    ]:
        url = (f"{CDX_API}?url={pattern}&output=json&fl=timestamp"
               f"&limit=1&closest={target_date}&filter=statuscode:200")
        try:
            r = session.get(url, timeout=20)
            rows = r.json()
            if rows and len(rows) >= 2:
                return rows[1][0]
        except:
            pass
    return None


def get_all_snapshots_for_slug(slug: str) -> list[str]:
    """Return all archived timestamps for a specific story slug, oldest first.
    Queries every known URL pattern with no snapshot limit."""
    timestamps: set[str] = set()
    patterns = [
        # Standard subdirectory layout (most common)
        f"www.mcstories.com:80/{slug}/{slug}.html",
        f"www.mcstories.com/{slug}/{slug}.html",
        f"mcstories.com/{slug}/{slug}.html",
        # index.html variant
        f"www.mcstories.com:80/{slug}/index.html",
        f"www.mcstories.com/{slug}/index.html",
        f"mcstories.com/{slug}/index.html",
        # Flat layout (older stories sometimes stored at root)
        f"www.mcstories.com/{slug}.html",
        f"mcstories.com/{slug}.html",
        f"www.mcstories.com:80/{slug}.html",
        # https variants
        f"https://www.mcstories.com/{slug}/{slug}.html",
        f"https://mcstories.com/{slug}/{slug}.html",
    ]
    for pattern in patterns:
        url = (f"{CDX_API}?url={pattern}&output=json&fl=timestamp"
               f"&filter=statuscode:200")  # no limit — get all snapshots
        for attempt in range(3):
            try:
                r = session.get(url, timeout=30)
                rows = r.json()
                if rows and len(rows) >= 2:
                    for row in rows[1:]:
                        timestamps.add(row[0])
                break
            except Exception as e:
                if attempt == 2:
                    log.debug(f"CDX query failed for {pattern}: {e}")
                time.sleep(1)
    return sorted(timestamps)


def wayback_url(original_url: str, timestamp: str) -> str:
    """Construct a Wayback Machine URL from an original URL."""
    return f"{WB_BASE}/web/{timestamp}/{original_url}"


def make_wayback_chapter_url(slug: str, chapter_num: int, timestamp: str) -> list[str]:
    """Return candidate Wayback URLs for a chapter, trying all known variants."""
    fname = f"{slug}.html" if chapter_num == 1 else f"{slug}{chapter_num}.html"
    originals = [
        f"http://www.mcstories.com:80/{slug}/{fname}",
        f"http://www.mcstories.com/{slug}/{fname}",
        f"https://mcstories.com/{slug}/{fname}",
    ]
    return [f"{WB_BASE}/web/{timestamp}/{o}" for o in originals]


def load_existing_slugs() -> set[str]:
    """
    Extract only story IDs from the main index without loading full_text.
    Uses ijson for streaming if available, falls back to a lightweight
    regex scan to avoid loading 500MB of story text into RAM.
    """
    if not MAIN_INDEX.exists():
        log.warning(f"Main index not found at {MAIN_INDEX} — treating all archived stories as new")
        return set()

    slugs: set[str] = set()

    # Try ijson first (streaming JSON parser — pip install ijson)
    try:
        import ijson
        with open(MAIN_INDEX, "rb") as f:
            for slug in ijson.items(f, "stories.item.id"):
                slugs.add(slug)
        log.info(f"Main index has {len(slugs)} stories (streamed via ijson)")
        return slugs
    except ImportError:
        pass

    # Fallback: regex scan for "id": "Slug" lines — fast, low memory
    import re
    id_pat = re.compile(r'"id"\s*:\s*"([^"]+)"')
    with open(MAIN_INDEX, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = id_pat.search(line)
            if m:
                slugs.add(m.group(1))
    log.info(f"Main index has {len(slugs)} stories (scanned via regex)")
    return slugs


def load_wayback_slugs() -> set[str]:
    """
    Load IDs of successfully scraped wayback stories.
    Stories with no text (fetch_error or empty full_text) are excluded
    so they will be retried on the next run.
    """
    if not WAYBACK_INDEX.exists():
        return set()

    slugs: set[str] = set()
    try:
        import ijson
        with open(WAYBACK_INDEX, "rb") as f:
            for story in ijson.items(f, "stories.item"):
                # Only mark as done if text was actually retrieved
                if story.get("full_text") and not story.get("fetch_error"):
                    slugs.add(story["id"])
        return slugs
    except ImportError:
        pass

    # Fallback: full load
    with open(WAYBACK_INDEX, encoding="utf-8") as f:
        data = json.load(f)
    for s in data.get("stories", []):
        if s.get("full_text") and not s.get("fetch_error"):
            slugs.add(s["id"])
    return slugs


# ── Scraping archived pages ───────────────────────────────────────────────────

def scrape_wayback_landing(wb_url: str, story_slug: str) -> dict:
    """
    Scrape a story landing page from the Wayback Machine.
    Same logic as scraper.py but adapted for archive.org URLs.
    """
    soup = get_soup(wb_url)
    if not soup:
        return {"fetch_error": f"Failed to fetch {wb_url}"}

    result: dict = {"story_slug": story_slug}

    # Remove Wayback Machine toolbar markup to avoid it polluting text
    for el in soup.find_all(id=re.compile(r"^wm")):
        el.decompose()

    # Title — from meta or h2
    def meta(name):
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else ""

    title = meta("dcterms.title")
    if not title:
        for tag in soup.find_all(["h1", "h2"]):
            txt = tag.get_text(strip=True)
            if txt and "mind-control story archive" not in txt.lower():
                title = txt
                break
    result["title"] = title or story_slug

    # Author
    author = meta("dcterms.creator")
    author_url = ""
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Wayback hrefs can be absolute archive.org paths or relative
        if "Authors/" in href and href.endswith(".html") and "/index.html" not in href:
            if not author:
                author = a.get_text(strip=True)
            # Reconstruct original URL
            if "web.archive.org" in href:
                m = re.search(r"mcstories\.com(/.*)", href)
                if m:
                    author_url = BASE_URL + m.group(1)
            else:
                author_url = urljoin(BASE_URL, href)
            break
    result["author"]     = author
    result["author_url"] = author_url

    # Tags
    subject = meta("dcterms.subject")
    if subject:
        cats = [t.strip() for t in subject.split() if t.strip()]
    else:
        cats = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "Tags/" in href and href.endswith(".html"):
                m = re.search(r"/Tags/([^/]+)\.html", href)
                if m:
                    code = m.group(1)
                    if code.lower() != "index" and "all stories" not in a.get_text().lower():
                        cats.append(code)
    result["categories"] = list(dict.fromkeys(cats))

    # Description
    desc = meta("dcterms.description")
    if not desc:
        for p in soup.find_all("p"):
            txt = p.get_text(strip=True)
            if 20 < len(txt) < 1000 and not re.search(
                    r"copyright|mcstories\.com|mind-control story archive", txt, re.I):
                desc = txt
                break
    result["description"] = desc

    # Dates
    page_text = soup.get_text(" ", strip=True)
    added   = re.search(r"Added[:\s]+(\d{1,2}\s+\w+\s+\d{4})", page_text, re.I)
    updated = re.search(r"Updated[:\s]+(\d{1,2}\s+\w+\s+\d{4})", page_text, re.I)
    result["added_date"]   = added.group(1)   if added   else ""
    result["updated_date"] = updated.group(1) if updated else ""

    # Chapter URLs — find links to StorySlug.html and StorySlugN.html
    chapter_map: dict[int, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Normalise Wayback hrefs back to original URLs
        if "web.archive.org" in href:
            m = re.search(r"mcstories\.com(/.*)", href)
            if m:
                href = m.group(1)

        # Match chapter file patterns
        prefix = f"/{story_slug}/{story_slug}"
        # Handle relative hrefs like "StorySlug.html" or "../StorySlug/StorySlug2.html"
        if href.endswith(".html"):
            fname = href.split("/")[-1]
            base  = f"{story_slug}"
            if fname == f"{base}.html":
                chapter_map[1] = f"{BASE_URL}/{story_slug}/{fname}"
            else:
                num_m = re.match(rf"^{re.escape(base)}(\d+)\.html$", fname, re.I)
                if num_m:
                    chapter_map[int(num_m.group(1))] = f"{BASE_URL}/{story_slug}/{fname}"

    if not chapter_map:
        chapter_map[1] = f"{BASE_URL}/{story_slug}/{story_slug}.html"

    result["chapter_urls"] = [chapter_map[n] for n in sorted(chapter_map)]
    return result


def _clean_extracted_text(text: str) -> str:
    """Fix common scraping artefacts in extracted story text."""
    # Collapse runs of whitespace (spaces/tabs) to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Insert missing space between a word-char and an uppercase letter
    # catches "surelyThat" → "surely That" (inline tag concatenation)
    text = re.sub(r'([a-z,;:!?\'"])([A-Z])', r'\1 \2', text)
    # Clean up space before punctuation that crept in from separators
    text = re.sub(r' ([.,;:!?\'"])', r'\1', text)
    # Normalise line endings
    text = re.sub(r'\r\n?', '\n', text)
    return text.strip()


def scrape_wayback_chapter(timestamp: str, chapter_url: str,
                           chapter_num: int = 1) -> str:
    """
    Fetch one chapter from the Wayback Machine.
    Tries URL variants in order but stops immediately on 404
    (no point trying other variants if the snapshot doesn't exist).
    """
    m = re.search(r"/([^/]+)/[^/]+\.html$", chapter_url)
    slug = m.group(1) if m else None

    candidates = []
    if slug:
        candidates = make_wayback_chapter_url(slug, chapter_num, timestamp)
    candidates.append(f"{WB_BASE}/web/{timestamp}/{chapter_url}")
    # Deduplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for wb in candidates:
        try:
            r = session.get(wb, timeout=30)
            if r.status_code == 404:
                # Snapshot doesn't exist — skip remaining variants for this timestamp
                # (they'll all 404 too since the content wasn't captured)
                break
            if r.status_code != 200:
                continue
        except Exception as e:
            log.warning(f"  Connection error: {wb}: {e}")
            time.sleep(1)
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        for el in soup.find_all(id=re.compile(r"^wm")):
            el.decompose()
        body = (soup.find("div", id="mcstories")
                or soup.find("div", class_="story")
                or soup.find("div", id="story"))
        if body:
            text = _clean_extracted_text(body.get_text(" ", strip=True))
            if len(text) > 100:
                return text
        # Fallback: paragraphs after first heading
        paras, capture = [], False
        for tag in soup.find_all(["h1", "h2", "p"]):
            if tag.name in ("h1", "h2"):
                capture = True
                continue
            if capture:
                t = _clean_extracted_text(tag.get_text(" ", strip=True))
                if t:
                    paras.append(t)
        text = "\n\n".join(paras)
        if len(text) > 100:
            return text
    return ""


def scrape_wayback_story(slug: str, timestamp: str) -> dict:
    """Fetch landing page + all chapters from Wayback Machine."""
    landing_original = f"{BASE_URL}/{slug}/index.html"
    landing_wb       = wayback_url(landing_original, timestamp)

    log.info(f"  Landing: {landing_wb}")
    meta = scrape_wayback_landing(landing_wb, slug)
    if meta.get("fetch_error"):
        return meta

    chapter_urls  = meta["chapter_urls"]
    chapter_count = len(chapter_urls)
    texts         = []

    for i, ch_url in enumerate(chapter_urls, 1):
        time.sleep(DELAY)
        if chapter_count > 1:
            log.info(f"    ch {i}/{chapter_count}: {ch_url.split('/')[-1]}")
        text = scrape_wayback_chapter(timestamp, ch_url, i)
        if text:
            texts.append(text)
        else:
            # Try with +1/-1 on the numbered fallback
            fallback = ch_url.replace(".html", "1.html") if i == 1 else None
            if fallback:
                time.sleep(DELAY)
                text = scrape_wayback_chapter(timestamp, fallback, 1)
                if text:
                    texts.append(text)
                    meta["chapter_urls"][0] = fallback

    meta["full_text"]     = "\n\n* * *\n\n".join(texts)
    meta["word_count"]    = len(meta["full_text"].split())
    meta["chapter_count"] = chapter_count
    if not meta["full_text"]:
        meta["fetch_error"] = "No chapter text retrieved from Wayback Machine"
    return meta


# ── Index management ──────────────────────────────────────────────────────────

def load_wayback_index() -> dict:
    """Load wayback index. For 434 stories this is fine to keep in memory fully."""
    if not WAYBACK_INDEX.exists():
        return {"source": "wayback", "stories": [], "dates_scanned": []}
    with open(WAYBACK_INDEX, encoding="utf-8") as f:
        return json.load(f)


def save_wayback_story(story_dict: dict):
    """
    Append a single story to wayback_index.json without loading the whole file.
    Uses a JSONL sidecar file for stories (one JSON object per line) and a
    separate small metadata file for stats — avoids ever holding all story
    text in memory at once.
    """
    # Stories go into wayback_stories.jsonl — one per line, append-only
    stories_file = WAYBACK_INDEX.with_suffix(".jsonl")
    with open(stories_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(story_dict, ensure_ascii=False) + "\n")


def save_wayback_meta(meta: dict):
    """Save just the metadata/stats for the wayback index."""
    tmp = WAYBACK_INDEX.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    tmp.replace(WAYBACK_INDEX)


def save_wayback_index(data: dict):
    """Legacy full save — only used for final consolidation."""
    tmp = WAYBACK_INDEX.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(WAYBACK_INDEX)


def consolidate_wayback_index():
    """
    Merge wayback_stories.jsonl sidecar into wayback_index.json.
    Call this at the end of a run or when you want a single clean file.

    Priority order when a story ID appears in multiple places:
      1. Existing wayback_index.json entry that has full_text (retry-recovered)
      2. JSONL sidecar entry that has full_text
      3. Existing wayback_index.json entry (any)
      4. JSONL sidecar entry (any, last wins for duplicates)

    This ensures --retry-failed progress is never overwritten by a new date scan.
    """
    stories_file = WAYBACK_INDEX.with_suffix(".jsonl")
    if not stories_file.exists():
        return

    meta = {}
    # Load existing index first — this may contain retry-recovered stories
    # with full_text that we must not overwrite with stale JSONL entries.
    existing_by_id: dict[str, dict] = {}
    if WAYBACK_INDEX.exists():
        with open(WAYBACK_INDEX, encoding="utf-8") as f:
            meta = json.load(f)
        for s in meta.get("stories", []):
            if s.get("id"):
                existing_by_id[s["id"]] = s

    # Read JSONL sidecar (append-only, may have stale failed entries)
    jsonl_by_id: dict[str, dict] = {}
    with open(stories_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    s = json.loads(line)
                    if s.get("id"):
                        jsonl_by_id[s["id"]] = s  # last entry wins for duplicates
                except Exception:
                    pass

    # Merge: prefer existing entry with full_text (retry-recovered) over JSONL
    merged: dict[str, dict] = {}
    all_ids = set(existing_by_id) | set(jsonl_by_id)
    for sid in all_ids:
        existing = existing_by_id.get(sid)
        jsonl    = jsonl_by_id.get(sid)
        if existing and existing.get("full_text") and not existing.get("fetch_error"):
            merged[sid] = existing   # retry-recovered: keep
        elif jsonl and jsonl.get("full_text") and not jsonl.get("fetch_error"):
            merged[sid] = jsonl      # JSONL has text: use it
        elif existing:
            merged[sid] = existing   # fall back to existing (failed or partial)
        else:
            merged[sid] = jsonl      # only in JSONL

    stories = list(merged.values())
    meta["stories"]       = stories
    meta["total_stories"] = len(stories)
    save_wayback_index(meta)
    log.info(f"Consolidated {len(stories)} wayback stories into {WAYBACK_INDEX} "
             f"({len(existing_by_id)} existing, {len(jsonl_by_id)} in JSONL)")


def merge_into_main(dry_run: bool = False):
    """
    Merge wayback stories into stories_index.json: add stories missing from it,
    and refresh the text of wayback stories it already holds when a longer copy
    has since been recovered. Loads the full main index — safe and simple. The main index is
    ~500MB but this is a one-time operation done occasionally.
    """
    # Find wayback stories
    stories_file = WAYBACK_INDEX.with_suffix(".jsonl")
    if not stories_file.exists() and not WAYBACK_INDEX.exists():
        print("No wayback stories found — run scraper first.")
        return
    if not MAIN_INDEX.exists():
        print("No main stories_index.json found.")
        return

    # Load wayback stories from both the index and the .jsonl sidecar. Either can
    # hold the most complete copy of a story (--retry-failed recovers into the
    # index; date scans append to the sidecar), so keep the one with the most text.
    candidates: list[dict] = []
    if WAYBACK_INDEX.exists():
        with open(WAYBACK_INDEX, encoding="utf-8") as f:
            candidates.extend(json.load(f).get("stories", []))
    if stories_file.exists():
        with open(stories_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except:
                        pass

    best: dict[str, dict] = {}
    for s in candidates:
        if not s.get("id") or s.get("fetch_error") or not s.get("full_text"):
            continue
        cur = best.get(s["id"])
        if cur is None or (s.get("word_count") or 0) > (cur.get("word_count") or 0):
            best[s["id"]] = s
    wb_stories = list(best.values())

    if not wb_stories:
        print("No wayback stories found in index or sidecar.")
        return

    # Load main index fully — safe approach
    print(f"Loading main index...")
    main_data = index_store.load_index(MAIN_INDEX, for_write=not dry_run)

    existing = {s["id"]: s for s in main_data.get("stories", [])}
    print(f"Main index: {len(existing):,} stories")

    to_add = [s for s in wb_stories if s["id"] not in existing]
    # Wayback stories already merged from an earlier, shorter snapshot (e.g. only
    # chapter 1 had been posted when it was captured). Live-site stories are never
    # touched. Small word-count gains are ignored: they come from the main index's
    # text having been cleaned since, and the raw copy must not replace it.
    def _is_longer(new: dict, old: dict) -> bool:
        if (new.get("chapter_count") or 1) > (old.get("chapter_count") or 1):
            return True
        return (new.get("word_count") or 0) > 1.2 * (old.get("word_count") or 0)

    to_update = [s for s in wb_stories
                 if s["id"] in existing
                 and existing[s["id"]].get("source") == "wayback"
                 and _is_longer(s, existing[s["id"]])]

    print(f"Wayback stories total:        {len(wb_stories):,}")
    print(f"Already in main index:        {len(wb_stories) - len(to_add):,}")
    print(f"New (deleted from live site): {len(to_add):,}")
    print(f"Longer copy than main index:  {len(to_update):,}")

    if dry_run:
        print("\nDry run — would add:")
        for s in to_add[:20]:
            print(f"  {s.get('id',''):<40} {s.get('title','')[:50]}")
        if len(to_add) > 20:
            print(f"  ... and {len(to_add)-20} more")
        if to_update:
            print("\nDry run — would update text:")
            for s in to_update[:20]:
                old = existing[s["id"]]
                print(f"  {s['id']:<40} {old.get('word_count', 0):>7,} → {s.get('word_count', 0):,} words")
        return

    if not to_add and not to_update:
        print("Nothing new to merge.")
        return

    if to_update:
        print(f"Updating {len(to_update):,} stories with longer text "
              f"(tags kept — re-run tagging to refresh them)...")
        for s in to_update:
            old = existing[s["id"]]
            for k in ("author_url", "wayback_url", "snapshot_date", "word_count",
                      "updated_date", "full_text", "chapter_count", "chapter_urls"):
                if k in s:
                    old[k] = s[k]

    print(f"Merging {len(to_add):,} stories...")

    main_data["stories"].extend(to_add)
    main_data["total_stories"] = len(main_data["stories"])
    main_data["last_updated"]  = datetime.now(timezone.utc).isoformat()

    total_expected = len(main_data["stories"])
    print(f"Writing {total_expected:,} stories to {MAIN_INDEX}...")
    # Validated save: re-reads the written file and checks the story count
    # before replacing; on failure the original index is untouched.
    try:
        index_store.save_index(main_data, MAIN_INDEX, validate=True)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    print(f"✓ Merged {len(to_add):,} wayback stories into {MAIN_INDEX}"
          f"{f', updated {len(to_update):,}' if to_update else ''}")
    print(f"  Total stories now: {total_expected:,}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def probe_archive():
    """Check what mcstories.com content is actually in the Wayback Machine."""
    print("\nProbing Wayback Machine for mcstories.com content...\n")

    tests = [
        # Old site: flat .html files at root e.g. www.mcstories.com/StoryName.html
        ("Old-style root .html pages (www)",
         f"{CDX_API}?url=www.mcstories.com/*.html&matchType=prefix&output=text&limit=10&fl=original,timestamp,statuscode&collapse=urlkey&filter=statuscode:200"),
        ("Old-style root .html pages (no www)",
         f"{CDX_API}?url=mcstories.com/*.html&matchType=prefix&output=text&limit=10&fl=original,timestamp,statuscode&collapse=urlkey&filter=statuscode:200"),
        # Check Authors directory (old site had /Authors/Name.html)
        ("Authors pages",
         f"{CDX_API}?url=mcstories.com/Authors/*.html&matchType=prefix&output=text&limit=10&fl=original,timestamp,statuscode&collapse=urlkey&filter=statuscode:200"),
        # Check if new-style subdir stories exist at all
        ("New-style subdir pages",
         f"{CDX_API}?url=mcstories.com/AToZeb/&matchType=prefix&output=text&limit=5&fl=original,timestamp,statuscode"),
        # Count total pages archived
        ("Total mcstories.com pages in archive",
         f"{CDX_API}?url=mcstories.com&matchType=domain&output=text&limit=20&fl=original,timestamp,statuscode&collapse=urlkey&filter=statuscode:200"),
    ]

    for label, url in tests:
        print(f"  {label}")
        print(f"  {url}")
        try:
            r = session.get(url, timeout=30)
            print(f"  HTTP {r.status_code}  {len(r.text)} bytes")
            if r.text.strip():
                lines = r.text.strip().splitlines()
                print(f"  {len(lines)} results, first 8:")
                for l in lines[:8]:
                    print(f"    {l}")
            else:
                print(f"  (empty response)")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()
        import time; time.sleep(1.5)


def run(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "wayback_scraper.log"),
        ],
    )

    if args.merge:
        merge_into_main(dry_run=args.dry_run)
        return

    existing_slugs  = load_existing_slugs()
    wayback_done    = load_wayback_slugs()
    wb_index        = load_wayback_index()

    all_deleted: dict[str, str] = {}   # slug -> timestamp

    for date in args.date:
        log.info(f"\n{'='*60}")
        log.info(f"Scanning Wayback Machine for date: {date}")
        log.info(f"{'='*60}")

        archived = cdx_get_story_slugs(date)
        time.sleep(DELAY)

        deleted = {slug: ts for slug, ts in archived.items()
                   if slug not in existing_slugs and slug not in wayback_done}

        log.info(f"Date {date}: {len(archived)} archived, "
                 f"{len(archived) - len(deleted)} already in main index, "
                 f"{len(deleted)} new deleted stories to scrape")

        for slug, ts in deleted.items():
            if slug not in all_deleted:
                all_deleted[slug] = ts

        # Free large CDX result — no longer needed
        archived.clear()
        del archived, deleted

    # Free slug sets — no longer needed after building all_deleted
    existing_slugs.clear()
    wayback_done.clear()
    del existing_slugs, wayback_done

    if not all_deleted:
        log.info("No new deleted stories found across all dates.")
        return

    log.info(f"\nTotal unique deleted stories to scrape: {len(all_deleted)}")

    if args.dry_run:
        print(f"\nDry run — would scrape {len(all_deleted)} deleted stories:")
        for slug, ts in sorted(all_deleted.items()):
            dt = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
            print(f"  {slug:<45} (snapshot: {dt})")
        return

    if args.limit:
        items = list(all_deleted.items())[:args.limit]
        all_deleted = dict(items)
        log.info(f"Limiting to {args.limit} stories (--limit flag)")

    # Scrape each deleted story from Wayback
    from tagger import tag_story
    import gc

    # Prepare metadata update once (reuse same dict, no growing lists)
    wb_index.setdefault("dates_scanned", [])
    for d in args.date:
        if d not in wb_index["dates_scanned"]:
            wb_index["dates_scanned"].append(d)

    ok = err = 0
    slugs_list = list(all_deleted.items())
    # Free the dict — no longer needed
    all_deleted.clear()
    del all_deleted

    for i, (slug, timestamp) in enumerate(slugs_list, 1):
        log.info(f"[{i}/{len(slugs_list)}] {slug}  (snapshot: {timestamp[:8]})")

        data = scrape_wayback_story(slug, timestamp)
        story = WaybackStory(
            id            = slug,
            title         = data.get("title", ""),
            author        = data.get("author", ""),
            author_url    = data.get("author_url", ""),
            story_url     = f"{BASE_URL}/{slug}/index.html",
            wayback_url   = wayback_url(f"{BASE_URL}/{slug}/index.html", timestamp),
            snapshot_date = timestamp,
            categories    = data.get("categories", []),
            description   = data.get("description", ""),
            word_count    = data.get("word_count", 0),
            added_date    = data.get("added_date", ""),
            updated_date  = data.get("updated_date", ""),
            full_text     = data.get("full_text", ""),
            fetch_error   = data.get("fetch_error"),
            chapter_count = data.get("chapter_count", 1),
            chapter_urls  = data.get("chapter_urls", []),
            source        = "wayback",
        )

        if story.full_text and not story.fetch_error:
            # Tag using only the fields the tagger needs — don't pass full_text twice
            tag_input = {
                "id": story.id, "title": story.title, "author": story.author,
                "categories": story.categories, "description": story.description,
                "word_count": story.word_count, "full_text": story.full_text,
            }
            story.auto_tags = tag_story(tag_input)
            ok += 1
        else:
            err += 1

        # Serialise once, save immediately, then discard
        story_dict = asdict(story)
        save_wayback_story(story_dict)

        # Free story text from memory immediately — it's on disk now
        del story_dict, story, data
        gc.collect()

        # Tell the OS allocator to release free pages back — prevents
        # memory creep from Python holding onto freed string allocations
        try:
            import ctypes, sys
            if sys.platform == "darwin":
                ctypes.cdll.LoadLibrary("libSystem.B.dylib").malloc_zone_pressure_relief(None, 0)
            else:
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass  # non-fatal — just means OS may see higher RAM for longer

        # Update lightweight metadata
        wb_index["last_updated"]  = datetime.now(timezone.utc).isoformat()
        wb_index["total_stories"] = i
        save_wayback_meta(wb_index)

        # Full GC pass every 20 stories to catch any circular refs
        if i % 20 == 0:
            gc.collect(2)  # generation 2 = full collection
            log.info(f"  Memory checkpoint at story {i}")

        time.sleep(DELAY)

    log.info(f"\nDone. {ok} scraped, {err} errors/404s.")

    # Consolidate .jsonl sidecar into wayback_index.json
    consolidate_wayback_index()
    log.info(f"Saved to {WAYBACK_INDEX}")
    log.info("Run: make wayback-merge  to add them to your main index.")


def run_retry_failed(args):
    """
    Retry all stories in wayback_index.json that previously failed.
    Uses concurrent workers for speed. Safe to Ctrl-C — checkpoints every 10
    recoveries and on interrupt.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tagger import tag_story

    OUTPUT_DIR    = Path(args.output)
    WAYBACK_INDEX = OUTPUT_DIR / "wayback_index.json"
    WORKERS       = getattr(args, "workers", 5)
    REQ_DELAY     = max(0.3, args.delay / WORKERS)  # per-worker delay

    if not WAYBACK_INDEX.exists():
        log.error("No wayback_index.json found. Run --date first.")
        return

    with open(WAYBACK_INDEX, encoding="utf-8") as f:
        wb_data = json.load(f)

    stories = wb_data.get("stories", [])
    failed  = [s for s in stories if s.get("fetch_error") and not s.get("full_text")]
    log.info(f"Found {len(failed)} previously-failed stories to retry "
             f"({WORKERS} workers, {REQ_DELAY:.1f}s delay each)")

    if not failed:
        log.info("Nothing to retry.")
        return

    if args.limit:
        failed = failed[:args.limit]

    story_by_id = {s["id"]: s for s in stories}
    lock        = threading.Lock()
    counters    = {"ok": 0, "err": 0}

    # Circuit breaker: when consecutive failures hit threshold, all workers pause.
    CIRCUIT_THRESHOLD = 15   # consecutive story failures before pausing
    CIRCUIT_PAUSE     = 300  # seconds to pause (5 minutes)
    circuit = {"consec": 0, "paused_until": 0.0}

    def checkpoint():
        wb_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = WAYBACK_INDEX.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(wb_data, f, indent=2, ensure_ascii=False)
        tmp.replace(WAYBACK_INDEX)

    def process_story(story):
        slug = story["id"]

        # Circuit breaker: wait if we're in a pause window
        with lock:
            wait_until = circuit["paused_until"]
        if wait_until > time.time():
            sleep_s = wait_until - time.time()
            log.warning(f"  [{slug}] Circuit breaker active — waiting {sleep_s:.0f}s")
            time.sleep(sleep_s)

        timestamps = get_all_snapshots_for_slug(slug)
        if not timestamps:
            return False, slug, 0

        # Limit snapshots tried per story to avoid multi-hour hangs.
        # Each failed fetch can take up to 60s (30s timeout × 2 retries);
        # trying 30+ timestamps sequentially is the root cause of 1500s/story.
        MAX_SNAPSHOTS = getattr(args, "max_snapshots", 10)
        if len(timestamps) > MAX_SNAPSHOTS:
            log.debug(f"  [{slug}] {len(timestamps)} snapshots found, "
                      f"trying {MAX_SNAPSHOTS} most recent")
            timestamps = timestamps[-MAX_SNAPSHOTS:]

        consec_fetch_errors = 0
        for ts in timestamps:
            time.sleep(REQ_DELAY)
            data = scrape_wayback_story(slug, ts)
            if data.get("full_text") and not data.get("fetch_error"):
                tag_input = {
                    "id":          slug,
                    "title":       data.get("title",       story.get("title", "")),
                    "author":      data.get("author",      story.get("author", "")),
                    "categories":  data.get("categories",  story.get("categories", [])),
                    "description": data.get("description", story.get("description", "")),
                    "word_count":  data.get("word_count",  0),
                    "full_text":   data.get("full_text",   ""),
                }
                with lock:
                    auto_tags = tag_story(tag_input)
                    s = story_by_id[slug]
                    s["full_text"]     = data["full_text"]
                    s["word_count"]    = data.get("word_count", 0)
                    s["chapter_count"] = data.get("chapter_count", 1)
                    s["chapter_urls"]  = data.get("chapter_urls", [])
                    s["snapshot_date"] = ts
                    s["fetch_error"]   = None
                    s["auto_tags"]     = auto_tags
                    if data.get("title"):
                        s["title"] = data["title"]
                    counters["ok"] += 1
                    circuit["consec"] = 0  # reset on success
                    if counters["ok"] % 10 == 0:
                        checkpoint()
                return True, slug, data.get("word_count", 0)

            # Track consecutive fetch/connection failures.
            # If archive.org is throttling this worker, all remaining
            # timestamps for this story will also fail — bail out early.
            if data.get("fetch_error"):
                consec_fetch_errors += 1
                if consec_fetch_errors >= 3:
                    log.warning(f"  [{slug}] 3 consecutive fetch errors — "
                                f"archive.org may be throttling; skipping "
                                f"remaining snapshots for this story")
                    break
            else:
                consec_fetch_errors = 0

        # All timestamps failed — update circuit breaker
        with lock:
            circuit["consec"] += 1
            if circuit["consec"] >= CIRCUIT_THRESHOLD:
                circuit["paused_until"] = time.time() + CIRCUIT_PAUSE
                circuit["consec"] = 0
                log.warning(
                    f"  Circuit breaker triggered: {CIRCUIT_THRESHOLD} consecutive "
                    f"failures — all workers will pause {CIRCUIT_PAUSE}s"
                )

        return False, slug, 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(failed), desc="Retrying", unit="story",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    except ImportError:
        pbar = None

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(process_story, s): s for s in failed}
            for fut in as_completed(futures):
                success, slug, words = fut.result()
                if not success:
                    with lock:
                        counters["err"] += 1
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"ok={counters['ok']} err={counters['err']}", refresh=True)
    except KeyboardInterrupt:
        log.info("\nInterrupted — saving progress…")
    finally:
        if pbar:
            pbar.close()

    log.info(f"Retry done: {counters['ok']} recovered, {counters['err']} still failed.")

    if counters["ok"] > 0:
        checkpoint()
        log.info(f"Saved → {WAYBACK_INDEX}")
        log.info("Run: make wayback-merge  to add recovered stories to your main index.")
    else:
        log.info("No stories recovered — index unchanged.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Scrape deleted mcstories.com stories from the Wayback Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check what's actually in the archive
  .venv/bin/python3 wayback_scraper.py --probe

  # Dry run to see what deleted stories would be found
  .venv/bin/python3 wayback_scraper.py --date 20200101 --dry-run

  # Scrape stories deleted since 2020
  .venv/bin/python3 wayback_scraper.py --date 20200101

  # Check multiple time periods
  .venv/bin/python3 wayback_scraper.py --date 20150101 --date 20180101 --date 20210101

  # Test with just 10 stories
  .venv/bin/python3 wayback_scraper.py --date 20200101 --limit 10

  # After scraping, merge into your main index
  .venv/bin/python3 wayback_scraper.py --merge
        """
    )
    p.add_argument("--date",     action="append", default=[],
                   metavar="YYYYMMDD",
                   help="Target date for Wayback snapshot (can specify multiple)")
    p.add_argument("--merge",    action="store_true",
                   help="Merge wayback_index.json into stories_index.json")
    p.add_argument("--dry-run",  action="store_true",
                   help="Show what would be scraped without actually scraping")
    p.add_argument("--limit",    type=int, default=None,
                   help="Max stories to scrape (for testing)")
    p.add_argument("--output",   default="./output",
                   help="Output directory (default: ./output)")
    p.add_argument("--delay",    type=float, default=DELAY,
                   help=f"Seconds between requests (default: {DELAY})")
    p.add_argument("--probe",    action="store_true",
                   help="Check what mcstories.com URLs are indexed in the Wayback Machine")
    p.add_argument("--consolidate", action="store_true",
                   help="Merge wayback_stories.jsonl sidecar into wayback_index.json")
    p.add_argument("--retry-failed", action="store_true",
                   help="Retry all previously-failed stories by exhausting all available snapshots")
    p.add_argument("--workers",  type=int, default=5,
                   help="Concurrent workers for --retry-failed (default: 5)")
    p.add_argument("--max-snapshots", type=int, default=10,
                   help="Max snapshots to try per story in --retry-failed (default: 10)")
    args = p.parse_args()

    OUTPUT_DIR    = Path(args.output)
    MAIN_INDEX    = OUTPUT_DIR / "stories_index.json"
    WAYBACK_INDEX = OUTPUT_DIR / "wayback_index.json"
    DELAY         = args.delay

    if args.probe:
        probe_archive()
        import sys; sys.exit(0)

    if args.consolidate:
        consolidate_wayback_index()
        import sys; sys.exit(0)

    if getattr(args, "retry_failed", False):
        run_retry_failed(args)
        import sys; sys.exit(0)

    if not args.merge and not args.date:
        p.error("Specify at least one --date YYYYMMDD, or use --merge / --retry-failed")

    run(args)