# MCScraper — User Guide

A personal archive and browser for mcstories.com, including deleted/offline stories recovered from the Wayback Machine.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Setup](#setup)
3. [The Browser](#the-browser)
4. [Searching & Filtering](#searching--filtering)
5. [Story Cards — Preview & History](#story-cards--preview--history)
6. [Preferences & Liking](#preferences--liking)
7. [Recommendations](#recommendations)
8. [Keeping Up To Date](#keeping-up-to-date)
9. [Scraping (First Time / Rebuild)](#scraping-first-time--rebuild)
10. [Wayback Machine (Deleted Stories)](#wayback-machine-deleted-stories)
11. [Local Content Ingestion](#local-content-ingestion)
12. [Tagging](#tagging)
13. [Maintenance](#maintenance)
14. [Key Files Reference](#key-files-reference)
15. [Common Make Commands](#common-make-commands)

---

## What This Is

MCScraper downloads and indexes the entire mcstories.com story archive locally. You can then browse, search, and filter ~18,300 stories offline through a web browser at `http://localhost:5003`, with features like:

- Full-text search across all stories
- Filtering by tags, author, word count, category, and AI-generated tags
- Personalised recommendations based on stories you like
- Recovery of deleted stories via the Wayback Machine
- Story generation using a local AI model

---

## Setup

**First time only:**

```bash
cd ~/Code/MCScraper
make install           # Creates .venv and installs dependencies
make install-semantic  # Also installs sentence-transformers (needed for recommendations)
make check             # Verify everything is working
```

You only need to do this once. The `.venv` persists between sessions.

---

## The Browser

The browser is a local website that runs at `http://127.0.0.1:5003`.

**Start it:**
```bash
make browse
# or directly:
.venv/bin/python3 browser.py --port 5003
```

There is also a **Launch Browser.command** file you can double-click from Finder.

**Pages:**

| Page | URL | What it does |
|------|-----|-------------|
| Search | `/` | Main search and browse with filters |
| Story | `/story/StoryId` | Read a story |
| Author | `/author/Name` | All stories by one author |
| Authors | `/authors` | Full author list |
| Tags | `/tags` | Browse all tag categories |
| For You | `/recommendations` | Personalised story recommendations |
| Liked | `/liked` | Your liked stories and authors |
| Want to Read | `/want-to-read` | Bookmarked stories saved for later |
| History | `/history` | Recently read stories with chapter position |
| New | `/new-stories` | Recently added stories |
| Similar | `/similar/StoryId` | Stories similar to a given story |
| Random | `/random` | Jump to a random story |
| Stats | `/stats` | Dataset statistics |
| Generate | `/generate` | AI story generation (requires LM Studio) |
| Bibles | `/bibles` | Story continuation tool |

---

## Searching & Filtering

### Text Search

The search box at the top searches titles, authors, descriptions, categories, and AI-generated tags. Two modes are available via radio buttons in the sidebar:

| Mode | Behaviour |
|------|-----------|
| **Closest match** (default) | All words are searched; stories matching more words rank higher. Nothing is hidden — partial matches appear lower. |
| **Literal** | Every word must appear somewhere (AND logic). Anything missing a word is excluded. |

- Searches CamelCase story IDs too — `thing of it` matches `ThingOfIt`
- Case-insensitive
- Filters set in the sidebar are preserved when you change the search text

**FTS search database** — if `output/search.db` exists (built with `make build-search-db`), searches run against a fast SQLite FTS5 index. The browser prints `"FTS search DB ready"` on startup when active. Without it, search falls back to a Python loop (slightly slower, same results).

### Sidebar Filters

| Filter | What it does |
|--------|-------------|
| Category | Official mcstories.com category codes (mc, md, mf, ff, etc.) |
| Exclude categories | Hide stories with these codes (e.g. `nc sn`) |
| Author | Filter to one author |
| Min/Max words | Word count range |
| MC Method | How the mind control works (hypnosis, drug, tech, etc.) |
| Setting | Where the story takes place |
| Themes | Subject matter themes (ctrl/cmd+click for multiple, any/all toggle) |
| Tone | Story tone (dark, romantic, playful, etc.) |
| Dominance style | Relationship dynamic |
| Pacing | Slow burn vs fast |
| Victim awareness | Whether the subject knows what's happening |
| Sexual Acts | Explicit acts (ctrl/cmd+click for multiple) |
| Fetish | Fetish tags (ctrl/cmd+click for multiple) |
| Relationship | Controller/subject relationship (teacher/student, etc.) |

Themes, Sexual Acts, and Fetish all have an **any / all** toggle below the list:
- **any** — story matches if it has at least one of your selected tags
- **all** (default for Themes) — story must have every selected tag

### Soft Filters (Preference-based)

Check **"Soft filters"** in the sidebar to switch from hard filtering (exclude non-matching stories) to **preference ranking** — stories matching more filters appear first, but nothing is hidden. Good for exploration.

### Category Codes (Quick Reference)

| Code | Meaning |
|------|---------|
| `mc` | Mind control |
| `md` | Male dominant |
| `fd` | Female dominant |
| `mf` | Male/female sex |
| `ff` | Female/female sex |
| `nc` | Non-consensual |
| `in` | Incest |
| `hm` | Humiliation |
| `ft` | Fetish |
| `sf` | Science fiction |
| `rb` | Robot/AI |

Full list at `/tags`.

---

## Story Cards — Preview & History

Each story card in search results has a few interactive features beyond the title and tags:

### Inline Preview

Click **▶ Preview** on any card to expand the first ~600 characters of the story inline, without opening a new page. Query words are highlighted in yellow. Click again (now **▾ Preview**) to collapse. Preview text is cached in the browser so re-opening is instant.

### Continue Reading / History

Every time you open a story page, your position is recorded automatically. Story cards show a badge:

| Badge | Meaning |
|-------|---------|
| `▶ Ch 3/8` | You've read up to chapter 3 of 8 |
| `▶ read` | You've opened this story (single-chapter) |

The **History** page (`/history`) shows your 100 most recently read stories in order, with a link that jumps straight to where you left off.

History is stored in `output/preferences.json` under `"history"` and persists across browser restarts.

### Chapter Navigation (Story Page)

On a story page with multiple chapters:

- Click the chapter buttons at the top and bottom
- **← →** arrow keys jump to previous/next chapter
- **↑ ↓** arrow keys also work (up = previous, down = next)
- Opening `/story/StoryId?ch=3` jumps directly to chapter 3

### Tag Tooltips

Hovering over a two-letter category tag (e.g. `mc`, `hm`, `mf`) shows its full name. Tags that already display their full name (AI-generated tags like "hypnosis", "corruption arc") don't show a tooltip — the label is already visible.

---

## Preferences & Liking

### Liking Stories

- **From a story page:** click the ♥ Like button (alongside ✕ Dislike and 🔖 Want to read)
- **From search results or author pages:** click the ♡ Like button on any story card — no need to open the story

### Want to Read

Click **🔖 Want** on any story card or story page to bookmark it for later. Bookmarked stories appear on the **Want to Read** page (`/want-to-read`) and show a teal badge on their cards. Clicking again removes the bookmark.

### Following Authors

- Open any author's page and click **☆ Follow**
- Followed authors' stories get a green highlight in search results

### Exporting

From the **Liked** page (`/liked`), use the export links to download your liked stories as:
- **JSON** — full metadata (id, title, author, word count, tags)
- **CSV** — spreadsheet-compatible
- **txt** — plain list of titles

### Visual Highlights on Story Cards

Story cards are colour-coded based on your preferences:

| Colour | Meaning |
|--------|---------|
| Amber/gold border | **Fave type** — has `md` or `mc` + `mf` tags (your preferred combination) |
| Green border | **Liked author** |
| Blue border | **Liked story** |
| Combined | Fave type + liked author = amber + green badge |
| Blue `▶ Ch X/Y` badge | You've previously read this story (see [History](#story-cards--preview--history)) |

### Editing Preferences Directly

Preferences are stored in `output/preferences.json`. You can edit it manually:

```json
{
  "liked_stories": ["StoryId", "AnotherStory"],
  "liked_authors": ["Author Name", "Another Author"],
  "disliked_stories": [],
  "disliked_authors": []
}
```

Story IDs are CamelCase slugs (e.g. `ThingOfIt`, `MindController`). Author names are as shown on the site.

After editing preferences manually, **restart the browser** and **re-run `make precalc-recs`** to update recommendations.

---

## Recommendations

The **For You** page (`/recommendations`) ranks stories by similarity to your liked stories using semantic embeddings.

**Building the recommendation cache:**
```bash
make precalc-recs
```

This takes under a minute and only needs re-running when:
- You've liked/unliked a significant number of stories
- You've run `make post-update` or any other pipeline step
- The browser shows a "Recs cache is stale" warning

The cache is saved to `output/recs_cache.json`.

### Similar Stories

On any story page, click **≡ Similar stories** to see stories ranked by similarity to that one. This uses the same recommendation engine but anchored to a single story rather than your full liked list. Also available at `/similar/StoryId`.

### Recommendation quality modes

`make precalc-recs` uses whichever mode is available:

| Mode | How to get it | Quality | Speed |
|------|--------------|---------|-------|
| **Tag-overlap** (default) | No setup needed | Good — based on tags and descriptions | ~1 min |
| **Semantic** | `make install-semantic` | Better — understands meaning, not just keywords | ~2 min (first run encodes all stories; subsequent runs load from `output/recs_embeddings.npz` cache and are nearly instant) |

To upgrade to semantic mode:
```bash
cd ~/Code/MCScraper
make install-semantic
make precalc-recs
```

---

## Keeping Up To Date

New stories are added to mcstories.com regularly.

**One-shot weekly update** (checks for new stories + rebuilds everything):
```bash
make update-full
```

This runs in sequence: fetch new stories → regex tag → semantic tag → rebuild JSONL → rebuild recommendations.

**Step by step if you prefer:**
```bash
make update          # Check last week for new stories only
make update-catchup  # Check last 4 weeks (if you've fallen behind)
make update-dry      # Preview what would be added without scraping
make post-update     # Re-tag + rebuild JSONL + rebuild recs + rebuild FTS DB (after index change)
```

**Protected stories:** `update.py` automatically skips any story with `source="wayback"`, `source="local"`, or `source="manual"` when checking for new chapters. These stories will never have their text overwritten by the update pipeline. Manually added stories must include `"source": "local"` in their JSON to be protected (see [OCR section](#adding-stories-from-screenshots-ocr)).

---

## Scraping (First Time / Rebuild)

You've already done the initial scrape. These commands are for reference or starting fresh.

```bash
make scrape          # Full scrape, ~17 hours, safe to Ctrl+C and resume
make resume          # Continue an interrupted scrape
make test-scrape     # Quick 20-story smoke test (~35 seconds)
```

The scraper is resumable — it saves progress and skips already-scraped stories.

---

## Wayback Machine (Deleted Stories)

Some stories have been removed from mcstories.com but are preserved in the Internet Archive.

### Basic Workflow

```bash
# Scrape deleted stories from a specific year
make wayback DATE=20200101

# Or scan multiple years at once (maximum coverage)
make wayback-multi

# Merge recovered stories into the main index
make wayback-merge

# Then run the full rebuild pipeline
make post-update
```

### Retrying Failed Stories

```bash
make wayback-retry     # 5 parallel workers, tries all available snapshots
```

Safe to stop and resume — skips stories already successfully recovered.

### Checking Coverage

```bash
make wayback-probe     # See what's available in the archive
make wayback-dry       # Preview what would be scraped (no actual scraping)
```

---

## Adding Stories from Screenshots (OCR)

For stories that exist only as phone screenshots, use `ingest.py` with OCR — it handles extraction, chapter assembly, and index insertion automatically. See [Local Content Ingestion](#local-content-ingestion) for the full workflow.

**Quick start:**

```bash
make install-ocr   # one-time setup (installs tesseract)
make ingest-file FILE="/path/to/screenshots/folder" TITLE="Story Title" AUTHOR="Author Name"
```

macOS screenshots are sorted by timestamp automatically. For other images, rename to `001.png`, `002.png`, … if they don't sort in reading order.

---

## Local Content Ingestion

`ingest.py` imports stories from local files into the archive. It handles all common formats automatically — metadata is extracted from the file where possible (title, author, chapter structure), and the result is a properly formatted index entry ready for tagging.

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Detects SOL (StoriesOnline) export headers automatically |
| EPUB | `.epub` | Reads OPF manifest for spine order and Dublin Core metadata |
| PDF | `.pdf` | Uses PyMuPDF; each file becomes one chapter |
| Word document | `.docx` | Extracts via zip/XML; strips copyright front-matter |
| Screenshot / image | `.png` `.jpg` `.jpeg` | OCR via macOS tesseract; requires `make install-ocr` |

### SOL Export Format

Plain `.txt` files exported from StoriesOnline contain a structured header block:

```
Title: The Story Title
by Author Name
Genre: Mind Control
Published: 2022-03-01
Summary: A description of the story.
Chapters: 4
...

[story body starts here]
```

`ingest.py` parses this header automatically — title, author, description, and chapter count are all extracted. The body is split into chapters on `Chapter N` headings or `* * *` scene breaks.

### Batch Folder Import

The most common use case: import an entire author folder of mixed file types.

```bash
# Preview without saving (shows what would be added, what would be skipped)
make ingest-dry FOLDER="/path/to/author folder" AUTHOR="Author Name"

# Import and write to index
make ingest-folder FOLDER="/path/to/author folder" AUTHOR="Author Name"
```

Files are grouped into stories automatically:
- Numbered series (`Story_Part_1.txt`, `Story_Part_2.txt`) → one story, multiple chapters
- EPUBs are always standalone (one EPUB = one story)
- Already-indexed stories are skipped (by ID) — safe to re-run

### Single File Import

```bash
make ingest-file FILE="/path/to/story.epub" AUTHOR="Author Name"
make ingest-file FILE="/path/to/story.txt" TITLE="My Title" AUTHOR="Author Name"
```

Or directly:

```bash
.venv/bin/python3 ingest.py "/path/to/story.epub" --author "Author Name" --patch
.venv/bin/python3 ingest.py "/path/to/story.txt" --title "My Title" --author "Author Name" --dry-run
```

### OCR (Screenshots / Images)

For stories that exist only as screenshots:

```bash
make install-ocr   # one-time: installs tesseract + pytesseract
make ingest-file FILE="/path/to/screenshots/folder" TITLE="Story Title" AUTHOR="Author Name"
```

macOS screenshots (`Screenshot 2024-01-15 at 10.23.45 AM.png`) are sorted by their embedded timestamp automatically. For other image files, rename to `001.png`, `002.png`, … if they don't sort in reading order. Each image becomes one page; pages are assembled into a single chapter.

### After Ingesting

Ingested stories have `"source": "local"` and are **never overwritten** by the scraper or `refresh_text.py`.

After any ingest run, rebuild the indexes:

```bash
make post-update   # re-tag + rebuild FTS + rebuild recs + rebuild stories_meta.db
```

Or if you want to skip retagging (already-tagged content):

```bash
make build-search-db-force
make build-stories-db
```

---

## Tagging

Stories have two types of tags:

1. **mcstories.com categories** — scraped directly from the site (mc, md, mf, etc.)
2. **AI-generated tags** — computed by a layered pipeline

### How the layers work

Tags are built up in passes. Every tag carries a `_tag_sources` record showing which layers contributed it (e.g. `["regex", "svengali"]`).

**How passes interact:** `retag_parallel.py` owns a fixed set of *regex-only dimensions* (`site_tags_expanded`, `site_tag_groups`, `content_type`, `pov`, `length_class`, `prevalence`) which it always overwrites. For all other dimensions (the ML-enhanced ones like `mind_control_type`, `setting`, `themes`, etc.) it merges in its results while preserving any tags already contributed by SetFit or Svengali. `_tag_sources` is rebuilt from scratch for regex on every run — stale entries are purged, so the AND-gate cannot be falsely satisfied by a pattern that no longer fires.

| Layer | What it does | When it runs |
|-------|-------------|--------------|
| **Regex** | Pattern-matching on full text. High precision, fast. | `make tag` |
| **Structural** | Infers pacing (epic/rapid/leisurely) from word count + chapter count. No ML needed. | `make tag` (built into regex pass) |
| **Semantic** | Sentence-transformer similarity for setting, themes, tone. Slower, catches things regex misses. | `make tag-semantic` |
| **Svengali** | Community-voted tags from ~14,800 stories merged in as ground truth. | `make svengali-import` |
| **SetFit** | Fine-tuned ML classifier for dimensions that need narrative understanding. | `make tag-setfit` |

### Which dimensions use which layers

Post March 2026 audit. Tags with <60% precision were dropped from regex; some are handled by AND-gated SetFit instead (see below).

| Dimension | Regex | SetFit | Notes |
|-----------|-------|--------|-------|
| MC Method | virus, drug, pheromones, nanotechnology, psionic, tech, subliminal, reality manipulation, time stop | hypnosis†, magic†, chemical† | possession/body puppet permanently dropped (0% Sv recall) |
| Relationship | boss/employee, teacher/student, doctor/patient, coworkers | reluctant subject† | All social dynamics tags (strangers, friends, couples, enemies, etc.) permanently dropped — 0% Sv recall |
| Sexual acts | oral, masturbation, rimming, group sex, titfuck | anal†, toys† | anal/toys AND-gated due to low audit precision but high Sv recall (61–72%) |
| Fetish | bdsm, sadomasochism, fetish clothing | — | Regex-only (below-threshold; under ML review) |
| Setting | home / domestic, supernatural, workplace | — | Most setting tags dropped (<50% precision); only these three retained |
| Themes | trigger phrase, consent / ethics, slave mentality, power corrupts, induced love | horror / dark†, identity loss†, memory alteration†, corruption arc†, serial recruitment | Many theme tags dropped (<40% audit precision) |
| Tone | dark / disturbing, erotic / explicit, psychological | — | Other tone tags dropped (<20% precision) — melancholic, romantic/sweet, humorous/playful, tense/suspenseful all gone |
| Dominance | manipulative, sadistic, loving control, cold / clinical, self-inflicted | brutal‡, gentle‡ | gentle/brutal are feature-only (recs only, not shown in browser) |
| Pacing | episodic | leisurely† | epic from structural; rapid dropped (44% after tightening); single-scene dropped (6%) |
| Mental changes | obedience programming, memory alteration, emotion control, addiction / need, orgasm control, speech modification†, induced love, stepfordization | identity loss†, corruption arc†, memory alteration† | speech modification feature-only |
| Victim awareness | denial (partial) | oblivious†, denial | valiant but futile struggle permanently dropped — 41% precision, 1% Sv recall |

† **AND-gated**: tag only appears when **both** regex AND SetFit agree. SetFit-only instances are purged from the index.
‡ **Feature-only**: present in `auto_tags` for recommendations, not shown as browser filter options.

### AND-gate tiers and layer trust

For tags with unreliable regex precision, an **AND-gate** is applied: the tag only appears when both regex AND SetFit agree. This is the single biggest quality improvement from the March 2026 audit — for example, `hypnosis` precision rises from **25% (regex alone) to 89% (AND-gate)**.

**Tier 1 — High trust (use as-is):** regex precision ≥75%, or structural (Svengali-derived). Includes all `site_tag_groups` / `site_tags_expanded`, `obedience programming`, `trigger phrase`, `consent / ethics`, `slave mentality`, `power corrupts`, `tone:erotic/explicit`, `tone:psychological`.

**Tier 2 — AND-gated:** requires both regex AND SetFit agreement. Active AND-gates: `hypnosis`, `magic`, `chemical`, `identity loss`, `memory alteration`, `corruption arc`, `horror/dark`, `leisurely`, `reluctant subject`, `anal`, `toys`.

Check `_tag_sources` to see which layers contributed:
- `["regex"]` — regex only
- `["svengali"]` — community-voted only
- `["regex", "svengali"]` — both agreed (highest confidence)
- `["setfit"]` — SetFit only (non-AND-gated tags only)
- `["regex", "setfit"]` — AND-gate satisfied

For the full precision audit results (82 tags, 4,597 stories reviewed, March 2026) and per-tag rationale, see `CLASSIFIER_GUIDE.md`.

### Re-tag all stories

```bash
make tag          # Regex + structural, ~8 min parallel (recommended)
make tag-slow     # ~24 min, single-process fallback
make tag-semantic # ~30 min, also runs semantic layer
```

### SetFit ML classifiers

Trained once on Svengali-labeled stories using PU learning (treats unlabeled stories as unknown, not negative).

```bash
make train-setfit    # Fine-tune encoder + train classifiers (~25 min Windows CUDA / ~3h Mac MPS — prefer Windows)
make train-setfit-fast  # Skip encoder fine-tuning, just retrain classifiers (~30 sec, uses cached embeddings)
make eval-setfit     # Check precision/recall against held-out Svengali stories
make tag-setfit      # Apply to all stories (additive)
make retag-setfit    # Clear existing SetFit tags and re-apply (use after threshold changes)
```

For full details on the tagging pipeline, evaluation, and improvement strategy see `CLASSIFIER_GUIDE.md`.

### Svengali Community Tags

```bash
make svengali-scrape         # Download community tags (slow)
make svengali-import         # Merge into main index
make svengali-stats          # Coverage report
make svengali-import-dry     # Preview before importing
```

---

## Maintenance

### Fix Text Artifacts

Some stories have run-on words caused by HTML tags being stripped without inserting spaces (e.g. `hishuge` instead of `his huge`). `fix_text.py` uses `wordninja` (probabilistic word splitter) with a layered set of guards to repair these while avoiding false splits on proper names, genre terms, and real compound words.

```bash
make fix-text          # Fix all stories and rebuild SQLite cache
make fix-text-dry      # Preview changes on 500 stories without saving
```

Or directly:

```bash
.venv/bin/python3 fix_text.py                    # Fix all stories
.venv/bin/python3 fix_text.py --dry-run          # Preview without saving
.venv/bin/python3 fix_text.py --dry-run --limit 500   # Preview first 500
.venv/bin/python3 fix_text.py --story ThingOfIt  # Fix one story only
```

Rebuilds the SQLite full-text cache automatically after saving.

**Note:** The root cause (HTML tag stripping without spaces) is fixed in `clean_text.py`. `fix_text.py` is for repairing stories already stored in the index from before that fix.

### Re-fetch Story Text (Full Refresh)

For a deeper fix, re-fetch the full text of all live-site stories using the improved HTML extractor. Stories are stored as **raw HTML** — the extractor preserves all formatting tags (`<em>`, `<strong>`, `<br>`, scene-break elements, etc.) and inserts spaces around inline elements to prevent word concatenation. This eliminates concatenation errors at the source rather than patching them after the fact.

```bash
make refresh-text          # re-fetch all live-site stories (~1 hr for 17k)
make refresh-text-resume   # resume an interrupted run
make refresh-text-dry      # preview first 20 stories without fetching
```

**Safety guarantees:**
- Stories with `source="wayback"`, `source="local"`, or `source="manual"` are **never touched**
- If any chapter of a story fails to fetch (e.g. chapters removed from the live site since the original scrape), the entire story is **skipped** and the original text preserved
- Skipped stories are listed in `output/refresh_text_skipped.txt` for review
- Resumable — safe to Ctrl+C and continue with `make refresh-text-resume`
- Only `full_text` and `word_count` are updated; all tags, preferences, and metadata are preserved

After completing, run `make post-update` to re-tag and rebuild caches.

### Backups

The `output/` directory contains everything important. Back it up before major operations. A backup copy is at `output_backup/`.

**⚠ Warning:** `make clean` and `make clean-output` delete your scraped data permanently.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `output/stories_index.json` | Source of truth — all story metadata and full text (written by scraper/tagger) |
| `output/stories_meta.db` | SQLite metadata index — browser loads from here instead of JSON (15× faster startup, 73 MB vs 1.2 GB) |
| `output/preferences.json` | Your liked stories, authors, and reading history |
| `output/full_text_cache.db` | SQLite cache — fast story text lookup |
| `output/search.db` | SQLite FTS5 index — fast search (build with `make build-search-db`) |
| `output/recs_cache.json` | Pre-calculated recommendations |
| `output/wayback_index.json` | Recovered deleted stories (before merge) |
| `browser.py` | Web browser (Flask app) |
| `scraper.py` | Live site scraper |
| `wayback_scraper.py` | Deleted story recovery |
| `tagger.py` | Tag generation (regex + semantic layers) |
| `retag_parallel.py` | Fast parallel retagging (~8 min, 10 cores) |
| `eval_tagger.py` | Measure regex precision/recall vs Svengali |
| `setfit_trainer.py` | ML tag classifier (fine-tuned sentence transformer) |
| `recommender.py` | Recommendation engine |
| `ingest.py` | Local content ingestion (.txt SOL, .epub, .pdf, .docx, images) |
| `fix_text.py` | Text artifact fixer (wordninja-based whitespace repair) |
| `refresh_text.py` | Re-fetch story text with fixed HTML extractor (skips wayback) |
| `update.py` | New story checker |
| `old/` | Archived old versions (scraper variants, svengali v2/v3, train v2) |

---

## Common Make Commands

```bash
make help              # Full command list

# Daily use
make browse            # Start the browser
make update-full       # Check for new stories + full rebuild (one-shot)
make precalc-recs      # Rebuild recommendation cache only

# After any manual index change
make post-update       # Re-tag + rebuild JSONL + rebuild recs + FTS DB + stories_meta.db

# Story metadata index (browser fast-load)
make build-stories-db  # Build/update stories_meta.db from stories_index.json

# FTS search database
make build-search-db        # Build/update FTS index (slim, ~150 MB, skips if up to date)
make build-search-db-force  # Force rebuild slim FTS index
make build-search-db-full   # Build with full story text (~1.6 GB, enables body-text search)

# Retagging
make tag               # Re-tag all stories, parallel (~8 min) — recommended
make tag-setfit        # Apply SetFit ML classifiers (after make tag)
make eval-tags         # Check regex tagger agreement with Svengali
make eval-setfit       # Check SetFit classifier agreement with Svengali

# Wayback
make wayback DATE=20200101   # Scrape deleted stories from 2020
make wayback-multi           # Scrape multiple years
make wayback-merge           # Merge into main index
make wayback-retry           # Retry failed recoveries

# Local content ingestion
make ingest-dry FOLDER=/path/to/folder AUTHOR="Name"   # Preview
make ingest-folder FOLDER=/path/to/folder AUTHOR="Name" # Import folder
make ingest-file FILE=/path/story.epub AUTHOR="Name"    # Import single file
make ingest-pan                                          # Import Pan/PanWhoWrites

# Text quality
make fix-text          # Wordninja whitespace repair on stored text
make refresh-text      # Re-fetch all live-site text with fixed HTML extractor (~1 hr)
make refresh-text-resume  # Resume an interrupted refresh

# Diagnostics
make check             # Verify environment
make stats             # Dataset statistics
```

---

## Hardware Reference

Two machines are used for this project. They have different strengths:

| Task | Mac M4 (16GB unified) | Windows PC (32GB RAM + RTX 3070 Ti 8GB VRAM) |
|------|-----------------------|----------------------------------------------|
| Scraping, data prep, browser | ✅ Primary machine | ✅ Works fine |
| Story generation (interactive) | ✅ ~50-60 tok/sec | ✅ Fast |
| SetFit classifier training | ⚠️ ~3h on MPS | ✅ ~25 min on CUDA |
| Arc tagging (bulk, 17k stories) | ⚠️ ~60 sec/story — memory pressure | ✅ ~25 sec/story — model fits in VRAM |
| LoRA fine-tuning | ❌ Too slow | ⚠️ Validates only; use A100 rental for real run |

**Why the Mac struggles with bulk inference:** The 8B model uses ~4.7GB. On the Mac
this comes out of the same 16GB pool used by the OS and all apps — at capacity,
macOS compresses memory and inference slows 3-4× . The RTX 3070 Ti has dedicated
VRAM; the model sits there independently and system RAM stays free.

**LM Studio context settings:**
- Mac: **8192** (interactive generation only — higher causes memory pressure)
- Windows: **32768** (fits comfortably in 8GB VRAM for arc tagging)

**Arc tagging** syncs automatically via Dropbox — run `narrative_arc.py` on Windows,
results appear on Mac when the checkpoint saves (every 5 stories).

---

---

## Further Improvements to Consider

Ideas that would be straightforward to implement, roughly in priority order:

### Browser / Reading

| Idea | Effort | Notes |
|------|--------|-------|
| **Per-story notes** | Low | The `notes` field already exists in `preferences.json` — just needs a text box on the story page and a notes view. |
| **Story download** | Low | A button on the story page to save the full text as a `.txt` file — useful for offline reading or feeding to an LLM. |
| **Author notes** | Low | Same as story notes but per author — useful for remembering why you follow/avoid someone. |
| **Reading time estimate** | Low | Add "~X min read" to story cards based on word count (250 wpm). |
| **Keyboard shortcut to focus search** | Low | Press `/` anywhere to jump the cursor to the search box (common in many tools). |
| **5-star rating** | Medium | Replace liked/not-liked binary with 1–5 stars. Better training signal for recommendations. Requires migrating existing likes (treat as 5 stars). |
| **Bulk like/unlike** | Medium | Checkbox multi-select on search results, then "Like all selected". Useful after a big search to batch-save results. |
| **Dark/light mode toggle** | Low | CSS variables are already set up for theming. Just needs a toggle button and `prefers-color-scheme` media query. |

### Search

| Idea | Effort | Notes |
|------|--------|-------|
| **Saved searches / presets** | Medium | Save a filter combination (e.g. "long hypnosis with corruption arc") and recall it in one click. Stored in `preferences.json`. |
| **Full-body text search** | Low | Run `make build-search-db-full` to enable searching inside story text (1.6 GB index, ~4 sec cold start on Dropbox — acceptable after warmup). |

### Recommendations

| Idea | Effort | Notes |
|------|--------|-------|
| **Recommendation explanation** | Medium | On the For You page, show which liked stories drove each recommendation ("because you liked X, Y, Z"). The scoring already tracks this internally. |

---

*Guide covers the project as of March 2026, ~18,300 stories indexed.*
