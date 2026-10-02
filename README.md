# MCScraper

A personal, offline archive and browser for mcstories.com — including deleted stories recovered from the Wayback Machine and locally ingested files — with search, multi-layer tagging, recommendations, and local-LLM generation tools.

The browser runs locally at **http://127.0.0.1:5003**.

---

## Repository layout

Code and data are kept separate:

| What | Where | Backed up by |
|------|-------|--------------|
| Code (this repo) | `~/Code/MCScraper` | git / GitHub |
| Data (indexes, databases, caches, training files, preferences) | `Dropbox/MCScraper-data` | Dropbox |

`output/` in the repo is a **link** to the Dropbox data folder and is ignored by git. All scripts read and write `./output/...`, so they work unchanged on any machine where that link exists.

```
MCScraper/
├── browser.py, app_factory.py   Flask browser entry point + app factory
├── routes/                      Page and API blueprints (browse, story, prefs, api, generate, bible, …)
├── services/                    Data loading, search, tag display, full-text store
├── templates/                   Jinja2 templates
├── scraper.py, update.py        Live scraping and incremental updates
├── wayback_scraper.py           Deleted-story recovery from the Wayback Machine
├── ingest.py                    Local file / folder / OCR ingestion
├── tagger.py, knn_tagger.py     Regex + semantic tagging, kNN community-tag transfer
├── setfit_trainer.py            SetFit classifier training and tagging
├── recommender.py               Preferences + embedding-based recommendations
├── build_*_db.py                SQLite metadata and FTS search databases
├── train.py, train_chat.py      LLM fine-tuning data prep / training
├── tests.py                     pytest suite
├── Makefile                     All common commands (`make help`)
└── output/  →  Dropbox/MCScraper-data   (not in git)
```

---

## Setup

### Mac

```bash
git clone https://github.com/Thestalus/MCScraper.git ~/Code/MCScraper
cd ~/Code/MCScraper
ln -s ~/Library/CloudStorage/Dropbox/MCScraper-data output
make install            # create .venv + core dependencies
make install-semantic   # sentence-transformers etc. (recommendations, semantic tagging)
make check              # verify dependencies and that scripts parse
```

To reproduce the exact package versions of the main Mac environment instead:
`.venv/bin/python3 -m pip install -r requirements-lock.txt`

### Windows (e.g. arc tagging on the GPU machine)

```bat
cd %USERPROFILE%
git clone https://github.com/Thestalus/MCScraper.git
cd MCScraper
mklink /J output "%USERPROFILE%\Dropbox\MCScraper-data"
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The Makefile's `PROJECT_DIR` is a Mac path — on Windows run scripts directly, e.g.
`.venv\Scripts\python narrative_arc.py --limit 2000`.

> Don't run jobs that write `stories_index.json` on both machines at the same time — Dropbox will create a conflicted copy.

---

## Everyday use

```bash
make browse        # start the browser on :5003
make update        # fetch new stories from the last week
make post-update   # re-tag, rebuild training JSONL, recs cache and FTS DB after index changes
make test          # run the test suite
make help          # full command reference
```

Keeping machines in sync:

```bash
git add -A && git commit -m "…" && git push   # after changing code
git pull                                        # before running on another machine
```

Data needs no git step — it syncs through Dropbox.

---

## How it fits together

```
mcstories.com / Wayback / local files
        │  scraper.py · wayback_scraper.py · ingest.py
        ▼
output/stories_index.json  (canonical story + tag data)
        │  tagger.py (regex + semantic) · setfit_trainer.py · knn_tagger.py (Svengali)
        ▼
auto_tags ──► stories_meta.db / search.db  ──► browser (search, filters)
          ──► recs_cache.json / embeddings ──► recommendations
          ──► train.py (trust-tier filtered)  ──► fine-tuning data
```

After anything changes `stories_index.json`, run `make post-update` so the derived databases and caches catch up.

---

## Documentation

| Guide | Covers |
|-------|--------|
| [GUIDE.md](GUIDE.md) | Using the browser, updating, Wayback recovery, ingestion, tagging, maintenance |
| [BROWSER_GUIDE.md](BROWSER_GUIDE.md) | Browser architecture: startup, AppState, routes, services, making changes |
| [CLASSIFIER_GUIDE.md](CLASSIFIER_GUIDE.md) | Tag schema, tagging layers, trust tiers, audits |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Fine-tuning pipeline, hardware, Windows/LM Studio setup |
