#!/usr/bin/env python3
"""
mcstories.com scraper — confirmed URL structure:

  /Titles/index.html
    → /Titles/A.html, /Titles/B.html, … /Titles/Z.html, /Titles/0.html
      → /StoryName/index.html          (story landing: title, author, tags, description)
        → /StoryName/StoryName.html    (single chapter OR chapter 1 of many)
        → /StoryName/StoryName2.html   (chapter 2, if multi-chapter)
        → /StoryName/StoryName3.html   (chapter 3, if multi-chapter) etc.

  Single-chapter story: only /StoryName/StoryName.html exists (no number)
  Multi-chapter story:  /StoryName/StoryName.html + StoryName2.html + StoryName3.html …
                        (chapter 1 never has a number, chapters 2+ always do)
"""

import re
import json
import time
import argparse
import logging
from pathlib import Path
from tag_trust import export_tags
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tagger import tag_story
import index_store

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL     = "https://mcstories.com"
TITLES_INDEX = f"{BASE_URL}/Titles/index.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; story-archiver/1.0; +research)",
    "Accept":     "text/html,application/xhtml+xml",
}
DELAY       = 1.5
MAX_RETRIES = 3

# Top-level site directories that are NOT stories
NAV_SLUGS = {"titles", "authors", "tags", "readerspicks", "whatsnew",
             "whats_new", "faq", "forum", "garden", "recentadditions"}

log = logging.getLogger(__name__)


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Story:
    id:            str
    title:         str
    author:        str
    author_url:    str
    story_url:     str
    categories:    list[str]      = field(default_factory=list)
    description:   str            = ""
    word_count:    int            = 0
    added_date:    str            = ""
    updated_date:  str            = ""
    full_text:     str            = ""
    fetch_error:   Optional[str]  = None
    auto_tags:     Optional[dict] = None
    chapter_count: int            = 1
    chapter_urls:  list[str]      = field(default_factory=list)


# ─── HTTP ─────────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update(HEADERS)

def get(url: str) -> Optional[BeautifulSoup]:
    """Fetch a page and parse it, retrying MAX_RETRIES times with a growing delay. Returns
    None if every attempt fails.
    """
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            log.warning(f"[attempt {attempt+1}] GET {url} → {e}")
            time.sleep(DELAY * (attempt + 1))
    return None


# ─── Level 1: Titles/index.html → letter page URLs ───────────────────────────

def get_letter_urls() -> list[str]:
    """Level 1: the per-letter title pages (A.html, B.html, ...) linked from
    Titles/index.html.
    """
    soup = get(TITLES_INDEX)
    if not soup:
        log.error("Could not fetch Titles index")
        return []
    urls, seen = [], set()
    for a in soup.find_all("a", href=True):
        full = urljoin(TITLES_INDEX, a["href"])
        # /Titles/A.html, /Titles/B.html … /Titles/0.html, /Titles/#.html
        if re.match(r"https://mcstories\.com/Titles/[^/]+\.html$", full):
            slug = full.rsplit("/", 1)[-1]
            # exclude index.html itself and nav pages
            if slug != "index.html" and full not in seen:
                seen.add(full)
                urls.append(full)
    log.info(f"Found {len(urls)} letter pages")
    return urls


# ─── Level 2: letter page → story landing URLs ────────────────────────────────

def get_story_urls_from_letter(letter_url: str) -> list[str]:
    """Level 2: every story landing page (https://mcstories.com/<Story>/index.html) linked
    from one letter page.
    """
    soup = get(letter_url)
    if not soup:
        return []
    urls, seen = [], set()
    for a in soup.find_all("a", href=True):
        full = urljoin(letter_url, a["href"])
        m = re.match(r"https://mcstories\.com/([^/]+)/index\.html$", full)
        if m and m.group(1).lower() not in NAV_SLUGS:
            if full not in seen:
                seen.add(full)
                urls.append(full)
    return urls


# ─── Level 3: story landing page → metadata + chapter URLs ───────────────────

def parse_landing(url: str) -> dict:
    """
    Parse /StoryName/index.html.
    Discovers chapter URLs from links on the landing page.
    Chapter 1 = StoryName/StoryName.html (no number)
    Chapter N = StoryName/StoryNameN.html (N >= 2)
    """
    soup = get(url)
    if not soup:
        return {"fetch_error": f"Failed to fetch {url}"}

    story_slug = urlparse(url).path.strip("/").split("/")[0]
    result: dict = {"story_slug": story_slug}

    # ── Meta tags — primary source (dcterms in <head>) ────────────────────────
    # <meta name="dcterms.title"       content="Story Title">
    # <meta name="dcterms.creator"     content="Author Name">
    # <meta name="dcterms.subject"     content="mc mf fd">
    # <meta name="dcterms.description" content="Story blurb">
    def meta(name):
        tag = soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.get("content") else ""

    # ── Title ─────────────────────────────────────────────────────────────────
    title = meta("dcterms.title")
    if not title:
        # Fallback: first h1/h2 that isn't the site masthead
        for tag in soup.find_all(["h1", "h2"]):
            txt = tag.get_text(strip=True)
            if txt and "mind-control story archive" not in txt.lower():
                title = txt
                break
    result["title"] = title or story_slug

    # ── Author ────────────────────────────────────────────────────────────────
    author = meta("dcterms.creator")
    author_url = ""
    # Get author URL from the Authors link regardless
    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        if re.match(r"https://mcstories\.com/Authors/[^/]+\.html$", full):
            if not author:
                author = a.get_text(strip=True)
            author_url = full
            break
    result["author"]     = author
    result["author_url"] = author_url

    # ── Tags ──────────────────────────────────────────────────────────────────
    # Primary: dcterms.subject contains space-separated tag codes e.g. "mc mf fd"
    subject = meta("dcterms.subject")
    if subject:
        cats = [t.strip() for t in subject.split() if t.strip()]
    else:
        # Fallback: scrape /Tags/xx.html links
        cats = []
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"])
            m = re.match(r"https://mcstories\.com/Tags/([^/]+)\.html$", full)
            if m:
                tag_code  = m.group(1)
                link_text = a.get_text(strip=True)
                if tag_code.lower() == "index":
                    continue
                if "all stories" in link_text.lower():
                    continue
                cats.append(tag_code)
    result["categories"] = list(dict.fromkeys(cats))

    # ── Description ───────────────────────────────────────────────────────────
    desc = meta("dcterms.description")
    if not desc:
        for p in soup.find_all("p"):
            txt = p.get_text(strip=True)
            if 20 < len(txt) < 1000 and not re.search(
                    r"copyright|mcstories\.com|mind-control story archive", txt, re.I):
                desc = txt
                break
    result["description"] = desc

    # ── Dates ─────────────────────────────────────────────────────────────────
    page_text = soup.get_text(" ", strip=True)
    added   = re.search(r"Added[:\s]+(\d{1,2}\s+\w+\s+\d{4})", page_text, re.I)
    updated = re.search(r"Updated[:\s]+(\d{1,2}\s+\w+\s+\d{4})", page_text, re.I)
    result["added_date"]   = added.group(1)   if added   else ""
    result["updated_date"] = updated.group(1) if updated else ""

    # ── Chapter URLs ──────────────────────────────────────────────────────────
    # Links on the landing page to chapter files follow the pattern:
    #   /StoryName/StoryName.html   (chapter 1, no number)
    #   /StoryName/StoryName2.html  (chapter 2)
    #   /StoryName/StoryName3.html  (chapter 3) etc.
    chapter_map: dict[int, str] = {}   # chapter_num → full URL

    for a in soup.find_all("a", href=True):
        full = urljoin(url, a["href"])
        # Must be inside this story's directory
        prefix = f"https://mcstories.com/{story_slug}/{story_slug}"
        if not full.startswith(prefix):
            continue
        suffix = full[len(prefix):]   # either ".html" or "2.html", "3.html" etc.
        if suffix == ".html":
            chapter_map[1] = full
        else:
            m = re.match(r"^(\d+)\.html$", suffix)
            if m:
                chapter_map[int(m.group(1))] = full

    if not chapter_map:
        # Fallback: construct chapter 1 URL from the slug
        chapter_map[1] = f"{BASE_URL}/{story_slug}/{story_slug}.html"

    result["chapter_urls"] = [chapter_map[n] for n in sorted(chapter_map)]
    return result


# ─── Level 4: fetch chapter text ─────────────────────────────────────────────

# Regex for stripping HTML tags — used for word counts, tagger input, FTS indexing.
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML tags, returning plain text suitable for word counting and analysis.
    Safe to call on legacy plain-text full_text values (no-op if no < present).
    """
    if not text or "<" not in text:
        return text or ""
    return _TAG_RE.sub(" ", text)


def _extract_html(body) -> str:
    """Extract story HTML from a BeautifulSoup tag, preserving all author formatting.

    Removes only the story-title heading (already in metadata) and stray nav
    elements, then returns the inner HTML so that <em>, <strong>, <blockquote
    class="textMessage">, <pre>, <table>, screenplay markup, etc. are kept exactly
    as authored and rendered faithfully by the browser.
    """
    for tag in body.find_all("h3", class_="title"):
        tag.decompose()
    for tag in body.find_all("nav"):
        tag.decompose()
    html = body.decode_contents()
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def fetch_chapter_text(url: str) -> str:
    """Download one chapter and return its text as cleaned HTML, from the <article
    id="mcstories"> block (or older layouts' equivalents), falling back to the <p>/<hr>
    tags after the first heading. Returns '' if the page can't be fetched.
    """
    soup = get(url)
    if not soup:
        log.warning(f"  Could not fetch chapter: {url}")
        return ""
    # Story text is in <article id="mcstories"> (current site) or the older
    # <div id="mcstories"> (pre-2024 snapshots / Wayback pages)
    body = (soup.find("article", id="mcstories")
            or soup.find("div", id="mcstories")
            or soup.find("div", class_="story")
            or soup.find("div", id="story"))
    if body:
        return _extract_html(body)
    # Fallback: collect <p> and <hr> tags after the first <h1>/<h2>, wrap as HTML
    paras, capture = [], False
    for tag in soup.find_all(["h1", "h2", "p", "hr"]):
        if tag.name in ("h1", "h2"):
            capture = True
            continue
        if capture:
            if tag.name == "hr":
                paras.append("<hr>")
                continue
            inner = tag.decode_contents().strip()
            if inner:
                paras.append(f"<p>{inner}</p>")
    return "\n".join(paras)


def fetch_full_story(landing_url: str) -> dict:
    """Fetch landing page + all chapters, return complete story dict."""
    meta = parse_landing(landing_url)
    if meta.get("fetch_error"):
        return meta

    chapter_urls  = meta["chapter_urls"]
    chapter_count = len(chapter_urls)
    texts         = []

    for i, ch_url in enumerate(chapter_urls, 1):
        if i > 1:
            time.sleep(DELAY)
        if chapter_count > 1:
            log.info(f"  ch {i}/{chapter_count}: {ch_url.split('/')[-1]}")
        text = fetch_chapter_text(ch_url)
        if not text and i == 1:
            # Fallback: some single-chapter stories use StorySlug1.html not StorySlug.html
            fallback = ch_url.replace(".html", "1.html")
            log.info(f"  ch 1 empty, trying numbered fallback: {fallback.split('/')[-1]}")
            time.sleep(DELAY)
            text = fetch_chapter_text(fallback)
            if text:
                meta["chapter_urls"] = [fallback] + chapter_urls[1:]
        if text:
            texts.append(text)
        else:
            log.warning(f"  ch {i} returned no text")

    meta["full_text"]     = "\n<!-- BREAK -->\n".join(texts)
    meta["word_count"]    = len(_strip_html(meta["full_text"]).split())
    meta["chapter_count"] = chapter_count
    if not meta["full_text"]:
        meta["fetch_error"] = "No chapter text retrieved"
    return meta


# ─── Training format builders ─────────────────────────────────────────────────

def to_raw_pretraining(story: Story) -> dict:
    """Training-export record: a metadata header (title, author, trusted tags, summary)
    followed by the plain story text.
    """
    at    = export_tags(story.auto_tags)
    parts = [f"Title: {story.title}", f"Author: {story.author}"]
    if story.categories:              parts.append(f"Categories: {', '.join(story.categories)}")
    if at.get("themes"):              parts.append(f"Themes: {', '.join(at['themes'])}")
    if at.get("mind_control_type"):   parts.append(f"Mind Control Method: {', '.join(at['mind_control_type'])}")
    if at.get("setting"):             parts.append(f"Setting: {', '.join(at['setting'])}")
    if at.get("tone"):                parts.append(f"Tone: {', '.join(at['tone'])}")
    if at.get("length_class"):        parts.append(f"Length: {at['length_class']}")
    if at.get("pov"):                 parts.append(f"POV: {at['pov']}")
    if story.description:             parts.append(f"Summary: {story.description}")
    return {"text": "\n".join(parts) + "\n\n" + _strip_html(story.full_text)}


def to_instruction_response(story: Story) -> dict:
    """Training-export record in chat format: a 'write a story titled X with these trusted
    tags' request and the story as the reply.
    """
    at      = export_tags(story.auto_tags)
    cats    = ", ".join(story.categories) if story.categories else "fiction"
    themes  = ", ".join(at.get("themes", []))
    mc_type = ", ".join(at.get("mind_control_type", []))
    setting = ", ".join(at.get("setting", []))
    msg = f'Write a {cats} story titled "{story.title}" by {story.author}.'
    if themes:  msg += f" Themes: {themes}."
    if mc_type: msg += f" Mind control method: {mc_type}."
    if setting: msg += f" Setting: {setting}."
    return {
        "messages": [
            {"role": "system",    "content": "You are a skilled fiction writer specializing in mind control erotica."},
            {"role": "user",      "content": msg},
            {"role": "assistant", "content": _strip_html(story.full_text)},
        ]
    }


def to_completion(story: Story) -> dict:
    """Training-export record as a prompt (metadata header) / completion (story text) pair."""
    at    = export_tags(story.auto_tags)
    cats  = ", ".join(story.categories) if story.categories else "fiction"
    parts = [f"Title: {story.title}", f"Author: {story.author}", f"Categories: {cats}"]
    if at.get("themes"):            parts.append(f"Themes: {', '.join(at['themes'])}")
    if at.get("mind_control_type"): parts.append(f"Mind Control Method: {', '.join(at['mind_control_type'])}")
    return {"prompt": "\n".join(parts) + "\n\n", "completion": _strip_html(story.full_text)}


def make_id(landing_url: str) -> str:
    """Story ID from its landing URL: the path segment, e.g.
    .../TheWillingCanvas/index.html -> 'TheWillingCanvas'.
    """
    return urlparse(landing_url).path.strip("/").split("/")[0]


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(args):
    """Full scrape: collect every story URL, skip ones already in the index (unless
    --no-resume), fetch and tag the rest, append them to the training exports, and save
    the index every CHECKPOINT_EVERY (50) stories through index_store.
    """
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "scraper.log"),
        ],
    )

    # ── Step 1: collect all story landing URLs ────────────────────────────────
    log.info("Building story URL list from Titles index...")
    letter_urls = get_letter_urls()
    if not letter_urls:
        log.error("No letter pages found — check Titles index URL")
        return

    landing_urls, seen = [], set()
    for letter_url in letter_urls:
        log.info(f"  Letter page: {letter_url}")
        for u in get_story_urls_from_letter(letter_url):
            if u not in seen:
                seen.add(u)
                landing_urls.append(u)
        # Stop crawling letter pages early if we already have enough
        if args.limit and len(landing_urls) >= args.limit:
            log.info(f"  Reached limit of {args.limit} — stopping letter crawl early")
            break
        time.sleep(DELAY)

    if args.limit:
        landing_urls = landing_urls[:args.limit]
        log.info(f"Limiting to {args.limit} stories (--limit flag)")
    log.info(f"Total story landing pages to scrape: {len(landing_urls)}")

    CHECKPOINT_EVERY = 50   # write index to disk every N stories

    # ── Step 2: load existing index if resuming ───────────────────────────────
    index_path    = out / "stories_index.json"
    stories: list[Story] = []
    already_done: set[str] = set()

    index_store.acquire_write_lock(index_path)   # this run rewrites the index
    if index_path.exists() and not args.no_resume:
        try:
            saved = index_store.load_index(index_path, for_write=True)
            for s in saved.get("stories", []):
                already_done.add(s["story_url"])
                stories.append(Story(**{k: v for k, v in s.items()
                                        if k in Story.__dataclass_fields__}))
            log.info(f"Resuming: loaded {len(stories)} previously scraped stories")
        except Exception as e:
            log.warning(f"Could not load existing index ({e}) — starting fresh")

    to_scrape = [u for u in landing_urls if u not in already_done]
    if len(already_done) > 0:
        log.info(f"Skipping {len(already_done)} already-scraped stories")
    log.info(f"Stories remaining to scrape: {len(to_scrape)}")

    # ── Step 3: scrape each story ─────────────────────────────────────────────
    # JSONL files: append if resuming, write fresh otherwise
    mode    = "a" if already_done else "w"
    f_raw   = open(out / "training_raw_pretraining.jsonl",      mode, encoding="utf-8")
    f_instr = open(out / "training_instruction_response.jsonl", mode, encoding="utf-8")
    f_comp  = open(out / "training_completion.jsonl",           mode, encoding="utf-8")

    def write_index():
        index = {
            "source":        BASE_URL,
            "scraped_at":    datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "total_stories": len(stories),
            "stories":       [asdict(s) for s in stories],
        }
        # Atomic, locked, and refuses to clobber changes made by another job
        index_store.save_index(index, index_path)

    for i, landing_url in enumerate(to_scrape, 1):
        log.info(f"[{i}/{len(to_scrape)}] {landing_url}")
        data = fetch_full_story(landing_url)

        story = Story(
            id            = make_id(landing_url),
            title         = data.get("title", ""),
            author        = data.get("author", ""),
            author_url    = data.get("author_url", ""),
            story_url     = landing_url,
            categories    = data.get("categories", []),
            description   = data.get("description", ""),
            word_count    = data.get("word_count", 0),
            added_date    = data.get("added_date", ""),
            updated_date  = data.get("updated_date", ""),
            full_text     = data.get("full_text", ""),
            fetch_error   = data.get("fetch_error"),
            chapter_count = data.get("chapter_count", 1),
            chapter_urls  = data.get("chapter_urls", []),
        )
        stories.append(story)

        if story.full_text and not story.fetch_error:
            story.auto_tags = tag_story(asdict(story),
                                        use_semantic=args.semantic,
                                        threshold=args.threshold)
            f_raw.write(  json.dumps(to_raw_pretraining(story),      ensure_ascii=False) + "\n")
            f_instr.write(json.dumps(to_instruction_response(story), ensure_ascii=False) + "\n")
            f_comp.write( json.dumps(to_completion(story),           ensure_ascii=False) + "\n")

        # Checkpoint: flush JSONL and write index every N stories
        if i % CHECKPOINT_EVERY == 0:
            f_raw.flush(); f_instr.flush(); f_comp.flush()
            write_index()
            log.info(f"  Checkpoint saved ({len(stories)} stories total)")

        time.sleep(DELAY)

    f_raw.close(); f_instr.close(); f_comp.close()

    # ── Step 4: final index write ─────────────────────────────────────────────
    write_index()

    ok    = sum(1 for s in stories if not s.fetch_error and s.full_text)
    err   = sum(1 for s in stories if s.fetch_error)
    empty = sum(1 for s in stories if not s.fetch_error and not s.full_text)
    log.info("─" * 60)
    log.info(f"Done.  {ok} scraped,  {err} errors,  {empty} empty")
    log.info(f"Output: {out.resolve()}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="mcstories.com scraper")
    p.add_argument("-o", "--output",   default="./mcstories_output")
    p.add_argument("--limit",          type=int,   default=None)
    p.add_argument("--delay",          type=float, default=DELAY)
    p.add_argument("--semantic",       action="store_true")
    p.add_argument("--threshold",      type=float, default=0.35)
    p.add_argument("--no-resume",      action="store_true",
                   help="Ignore existing index and start from scratch")
    args  = p.parse_args()
    DELAY = args.delay
    run(args)
