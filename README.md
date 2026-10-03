# MCScraper

A personal, offline archive and browser for mcstories.com — including deleted stories recovered from the Wayback Machine and locally ingested files — with search, multi-layer tagging, recommendations, and local-LLM generation tools.

The browser runs locally at **http://127.0.0.1:5003**.

---

## Repository layout

Code and data are kept separate. The repo holds only code; everything the scripts produce (story index, databases, caches, embeddings, training files, preferences) goes in `output/`, which is ignored by git.

`output/` can be an ordinary folder, or a link to a folder somewhere else — for example a Dropbox or other synced folder, so the multi-gigabyte data is backed up without going into git. All scripts read and write `./output/...`, so either way works unchanged.

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
├── index_store.py               the only safe way to write stories_index.json (lock + atomic save)
├── tag_utils.py, tag_trust.py   reading a story's tags; which tags are trusted for training
├── dataset_utils.py             story-level validation splits and dataset manifests
├── tests.py                     pytest suite
├── Makefile                     All common commands (`make help`)
└── output/                      Data folder or link to one (not in git)
```

---

## Setup

Requires **Python 3** (developed and tested on 3.13) and git. Developed on macOS; Linux and Windows are supported via the instructions below.

### 1. Clone

```bash
git clone https://github.com/Thestalus/MCScraper.git
cd MCScraper
```

### 2. Choose where the data lives

Either create a plain folder (`mkdir output`), or link `output` to an existing data folder:

| OS | Link `output` to an existing folder |
|----|-------------------------------------|
| macOS / Linux | `ln -s /path/to/data output` |
| Windows (Command Prompt) | `mklink /J output "C:\path\to\data"` |

### 3. Install dependencies

**macOS / Linux** — the Makefile handles the virtual environment:

```bash
make install            # create .venv + core dependencies
make install-semantic   # sentence-transformers (recommendations, semantic tagging)
make check              # verify dependencies and that scripts parse
```

or install everything in `requirements.txt` at once:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

**Windows:**

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The Makefile uses Unix paths, so on Windows run the scripts directly instead of through `make` — e.g. `.venv\Scripts\python browser.py --port 5003`. The Makefile shows the exact command and arguments behind each target.

Optional extras:
- **OCR ingestion** needs the Tesseract binary (`brew install tesseract`, your Linux package manager, or the Windows installer).
- **LLM fine-tuning** needs `torch`, `transformers`, `peft`, `bitsandbytes` and `accelerate`, installed to match your GPU/CUDA setup — see [TRAINING_GUIDE.md](TRAINING_GUIDE.md).
- `requirements-lock.txt` pins the exact package versions of a known-good macOS environment.

---

## Everyday use

```bash
make browse        # start the browser on :5003
make update        # fetch new stories from the last week
make post-update   # re-tag, rebuild training JSONL, recs cache and FTS DB after index changes
make test          # run the test suite
make help          # full command reference
```

### Using more than one machine

Code syncs through git (`git push` after changes, `git pull` before running elsewhere). If `output` links to a synced folder on each machine, the data follows automatically — but don't run jobs that write `stories_index.json` on two machines at the same time, or the sync service will create a conflicted copy.

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
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Start here for the code**: what every script and data file is for, what reads and writes what, rules the code relies on |
| [GUIDE.md](GUIDE.md) | Using the browser, updating, Wayback recovery, ingestion, tagging, maintenance |
| [BROWSER_GUIDE.md](BROWSER_GUIDE.md) | Browser architecture: startup, AppState, routes, services, making changes |
| [CLASSIFIER_GUIDE.md](CLASSIFIER_GUIDE.md) | Tag schema, tagging layers, trust tiers, audits |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Fine-tuning pipeline, model choices, LM Studio setup |

> **Note:** TRAINING_GUIDE.md is written around specific hardware — an RTX 3070 Ti (8 GB VRAM) on Windows, an Apple M4 Mac with 16 GB, and rented A100s for larger runs. Model sizes, context lengths, timings and costs in it assume that hardware; adjust them for yours.
