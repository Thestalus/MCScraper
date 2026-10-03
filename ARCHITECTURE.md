# MCScraper architecture and file map

What every file is for, what it reads and writes, and how the pieces connect. For day-to-day usage see [GUIDE.md](GUIDE.md); for the browser's internals see [BROWSER_GUIDE.md](BROWSER_GUIDE.md).

## How data flows

```
 mcstories.com      Wayback Machine      local files / OCR      Svengali (community tags)
      │                    │                     │                       │
  scraper.py         wayback_scraper.py      ingest.py               svengali.py
  update.py                │                     │                       │
      └────────────────────┴──────────┬──────────┴───────────────────────┘
                                      ▼
                       output/stories_index.json        ← canonical: every story, its text, its tags
                                      │                    (written only through index_store.py)
         ┌────────────────────────────┼──────────────────────────────────┐
         │ tagging                    │ derived databases                │ training data
         ▼                            ▼                                  ▼
  tagger.py / retag_parallel.py   build_stories_db.py → stories_meta.db  train.py → train_data.jsonl, val_data.jsonl
  setfit_trainer.py               build_search_db.py  → search.db        extract_chatbot_data.py → chat_data.jsonl, chat_val.jsonl
  retrain_classifiers.py          recommender.py      → recs_cache.json  tagger.py --jsonl-only → training_*.jsonl (export)
  narrative_arc.py (LLM)                                 recs_embeddings.npz
  (write tags back into the index)  services/text_store → full_text_cache.db
                                      │
                                      ▼
                       browser.py (Flask, http://127.0.0.1:5003)
                       reads stories_meta.db, search.db, full_text_cache.db,
                       recs_*; writes only preferences.json
```

After anything changes `stories_index.json`, run `make post-update`. It re-tags (regex), re-imports Svengali tags, retrains and applies the SetFit classifiers, rebuilds the JSONL export, the recommendation cache, `search.db` and `stories_meta.db`, then prints stats. `make browse` also rebuilds `stories_meta.db` by itself when the index is newer.

## Rules the code relies on

- **Only `index_store.py` writes `stories_index.json`.** It takes a write lock (`stories_index.json.lock`), refuses to save if another process or machine changed the file since it was loaded, and replaces the file atomically. Any new script that modifies the index must use `index_store.load_index(path, for_write=True)` and `index_store.save_index(...)`.
- **Read tags with `tag_utils.get_auto_tags(story)`.** `stories_meta.db` stores `auto_tags` as a JSON string that is parsed on first access; reading `story["auto_tags"]` directly can return a string. Templates use `story_tags(s)`.
- **Training trust lives in `tag_trust.py`.** Which tags are reliable enough to put in training prompts is defined there and nowhere else; `train.py` and every JSONL exporter import it. The browser's display filter (`services/tag_service.py`, `LOW_QUALITY_SETFIT_TAGS`) is a separate, related list — see "Tag quality lists" below.
- **Tests never touch the real data.** `conftest.py` fails the run if any file under `output/` changes.

## Data files (`output/`)

`output/` is a link to the data folder (Dropbox). "Rebuild" means the file can be regenerated from the index; "keep" means it can't, so it must stay backed up.

| File | Made by | Used by | If lost |
|------|---------|---------|---------|
| `stories_index.json` | scraper, update, wayback merge, ingest, all taggers | everything | **keep** — the canonical store |
| `preferences.json` | browser (likes, history, want-to-read, hidden recs) | browser, recommender, train.py | **keep** |
| `svengali_raw.jsonl` | `svengali.py --scrape` | `svengali.py --import` | keep (re-scrape is slow) |
| `fp_audit_results.jsonl` | audit tools, `auto_label.py` | SetFit training (audit labels) | **keep** — months of labelling |
| `gold_labels.jsonl` | you (optional; human-verified labels) | `setfit_trainer.py --eval` | keep |
| `wayback_index.json(l)` | `wayback_scraper.py` | wayback merge/retry | keep (slow to rebuild) |
| `setfit_model/` | `setfit_trainer.py --train` (fine-tuned encoder) | SetFit embeddings | rebuild (hours, GPU) |
| `setfit_classifiers.pkl` | `setfit_trainer.py --train`, `retrain_classifiers.py` | SetFit tagging | rebuild (`make retrain-classifiers`) |
| `setfit_embeddings.npz` | `setfit_trainer.py` | SetFit training/tagging | rebuild (hours; cache keyed by encoder + story text) |
| `stories_meta.db` | `build_stories_db.py` | browser (fast metadata load) | rebuild (`make build-stories-db`) |
| `search.db` | `build_search_db.py` | browser search (FTS5) | rebuild (`make build-search-db`) |
| `full_text_cache.db` | browser startup (`services/text_store.py`) | browser story pages, `export_stories.py` | rebuild (automatic) |
| `recs_embeddings.npz` | `recommender.py` | recommendations | rebuild (~1 min; cache keyed by story signature) |
| `recs_cache.json` | `make precalc-recs` | browser "For You" + similar stories | rebuild (`make precalc-recs`) |
| `training_*.jsonl` (3 files) | `tagger.py --jsonl-only`, scraper, update | external fine-tuning only | rebuild (`make rebuild-jsonl`) |
| `train_data.jsonl`, `val_data.jsonl` + `.manifest.json` | `train.py --prepare` | `train.py --train` | rebuild |
| `chat_data.jsonl`, `chat_val.jsonl` + `.manifest.json` | `extract_chatbot_data.py` | `train_chat.py` | rebuild |
| `knn_svengali*.npy/json`, `fulltext_pool*` | `knn_tagger.py` (research) | `knn_tagger.py` | rebuild |
| `bibles/` | browser story-bible editor | browser | **keep** (your writing) |
| `*.log`, `update_log.json` | scrapers, update, arc tagging | `/new-stories` page reads `update_log.json` | logs |

## Scripts by role

### Getting stories in
| File | Run with | What it does |
|------|----------|--------------|
| `scraper.py` | `make scrape`, `make resume`, `make test-scrape` | Full scrape of mcstories.com into the index (resumable). Also writes the three `training_*.jsonl` exports. |
| `update.py` | `make update`, `make update-full` | Weekly incremental update: new stories and new chapters of existing ones. Skips `wayback`/`local`/`manual` stories. |
| `wayback_scraper.py` | `make wayback*` | Recovers deleted stories from the Internet Archive into `wayback_index.json`, then merges them into the index. |
| `ingest.py` | `make ingest-*` | Imports local files (txt/md including StoriesOnline exports, EPUB, PDF, DOCX, and screenshots via OCR) as stories. |
| `refresh_text.py` | `make refresh-text` | Re-fetches the text of live-site stories with the current HTML extractor. |
| `svengali.py` | `make svengali-*` | Scrapes community tags from Svengali and imports them into `auto_tags` (source `svengali`). |

### Cleaning text
| File | Run with | What it does |
|------|----------|--------------|
| `clean_text.py` | `python clean_text.py` | Removes encoding artefacts and HTML junk from story text. |
| `fix_text.py` | `make fix-text` | Applies text clean-up rules to every story and refreshes the text cache. |

### Tagging
| File | Run with | What it does |
|------|----------|--------------|
| `tagger.py` | `make tag-slow`, `make tag-semantic`, `make rebuild-jsonl` | Regex tag rules (the first tagging layer), optional sentence-embedding layer, and the `training_*.jsonl` export. |
| `retag_parallel.py` | `make tag` | Runs the regex tagger over every story on all CPU cores. |
| `setfit_trainer.py` | `make train-setfit`, `make eval-setfit`, `make tag-setfit` | SetFit classifiers: fine-tune the encoder, train per-tag classifiers, evaluate on a held-out test split, tag stories. |
| `retrain_classifiers.py` | `make retrain-classifiers` (also `post-update` step 3) | Retrains the SetFit classifiers on existing embeddings (no encoder fine-tune) and re-applies them. |
| `narrative_arc.py` | `make arc-test`, `make arc`, `make arc-stats` | Uses a local LLM (LM Studio) to tag each story's arc (ending, dominant type, subject change). Usually run on the GPU PC. |
| `tag_trust.py` | (library) | Trust tiers: which tags are reliable enough for training prompts. |
| `tag_utils.py` | (library) | `get_auto_tags()` — the single way to read a story's tags. |

### Derived databases and recommendations
| File | Run with | What it does |
|------|----------|--------------|
| `build_stories_db.py` | `make build-stories-db` (auto in `make browse`) | Builds `stories_meta.db`: every story without its text, so the browser loads in under a second. |
| `build_search_db.py` | `make build-search-db` | Builds `search.db` (SQLite FTS5) for fast search; `-full` variant indexes story text too. |
| `recommender.py` | `make precalc-recs` | Embedding engine, preference store and recommendation engine; `--precalc` writes `recs_cache.json`. |

### The browser
| File | What it does |
|------|--------------|
| `browser.py` | Entry point: loads data, starts Flask on port 5003. |
| `app_factory.py` | `AppState` (everything loaded at startup) and `create_app()` (blueprints, template helpers). |
| `routes/` | One file per area: `browse` (search/home), `story`, `author`, `meta` (tags, stats, new stories), `prefs` (liked, history, recommendations), `api` (JSON endpoints), `generate`, `bible`. |
| `services/data_store.py` | Loads stories (from `stories_meta.db` or the index), preferences, recs cache, starts background cache builds. |
| `services/search_service.py` | Search, filtering and ranking. |
| `services/tag_service.py` | Which tags to display (hides low-quality classifier tags unless Svengali agrees). |
| `services/text_store.py` | Story text from `full_text_cache.db`, building it in the background when needed. |
| `search.py` | Ranking helpers and natural-language query parsing ("long hypnosis office story" → filters). |
| `generator.py` | Story generation through any OpenAI-compatible local LLM server. |
| `story_bible.py` | Story bibles (characters, world, chapter summaries) for long-form generation. |
| `templates/` | Jinja2 templates; `base.html` is the shared layout. |

### Training (LoRA fine-tuning)
| File | Run with | What it does |
|------|----------|--------------|
| `train.py` | `python train.py --prepare` / `--train` / `--merge` | Story LoRA: builds a seeded, filtered dataset with a story-level validation split, then trains with Unsloth. |
| `extract_chatbot_data.py` | `python extract_chatbot_data.py` | Turns dialogue-heavy stories into chat conversations (`chat_data.jsonl`, `chat_val.jsonl`). |
| `train_chat.py` | `python train_chat.py --train` | Chat/RP LoRA on the extracted conversations. |
| `dataset_utils.py` | (library) | Story-level validation split and dataset manifests. |
| `training_monitor.py` | `python training_monitor.py --cmd "..."` | Live progress window around any training command. |

Current code defaults (check before a run — [TRAINING_GUIDE.md](TRAINING_GUIDE.md) discusses 12B models that the code does not default to yet): `train.py` uses `cognitivecomputations/dolphin3.0-llama3.1-8b`; `train_chat.py` uses `Vdr1/L3-8B-Sunfall-v0.5-Stheno-v3.2` at 2,048 tokens with LoRA rank 8.

### Command-line utilities
| File | Run with | What it does |
|------|----------|--------------|
| `search_stories.py` | `make search`, `make stats` | Terminal search REPL and dataset stats over the index. |
| `export_stories.py` | `python export_stories.py "Title" ... -o out.txt` | Copies chosen stories into one text file. |
| `eval_tagger.py` | `make eval-tags` | Regex tagger precision/recall against Svengali tags. |
| `audit_scrape.py` | `python audit_scrape.py` | Checks scraper output for errors and missing text. |

### Research and audit tools
Used to build the labelled data and measure the classifiers. Not part of the regular pipeline; kept because their output (`fp_audit_results.jsonl`, audit reports) feeds training and the trust tiers.

| File | What it does |
|------|--------------|
| `fp_audit.py` | Local web UI to label suspected false-positive tags by hand. |
| `fp_audit_auto_review.py` | The same review done in batches by Claude Haiku (API key needed). |
| `fp_audit_dashboard.py` | Terminal summary of audit results. |
| `tp_audit.py` | Web UI to find true-positive examples for tags without Svengali labels. |
| `auto_label.py` | Claude-assisted labelling of candidate stories per tag. |
| `arc_audit.py` | Compares local-LLM arc tags with Claude's, to measure arc-tag accuracy. |
| `knn_tagger.py` | Experimental nearest-neighbour tag transfer from Svengali-tagged stories. |

### Shared infrastructure and tests
| File | What it does |
|------|--------------|
| `index_store.py` | Locked, change-checked, atomic loading and saving of the index. |
| `tests.py` | The test suite (`make test`). Covers tagging, exports, every browser page, caches, write safety, dataset splits. |
| `conftest.py` | Test setup, plus the guard that fails a run if tests modify real data. |
| `Makefile` | All common commands — `make help` lists them. |

### Legacy (`old/`, private, not in git)
`make estimate`, `make probe`, `make import-aiden` and `make import-mindspark` run scripts from the private `old/` folder (`estimate_size.py`, `probe_site.py`, `import_local_stories.py`, `import_local_mindspark.py`). They are superseded by `ingest.py` and kept for reference.

## Tag quality lists

Two lists decide how much a tag is trusted, for different purposes:

- `tag_trust.py` (training) — Tier 1 tags (regex precision ≥ 74%) and AND-gated tags (only when regex **and** SetFit agree).
- `services/tag_service.py` `LOW_QUALITY_SETFIT_TAGS` (display) — classifier tags hidden in the browser unless Svengali also has them.

They overlap on purpose in one place: `leisurely` is listed as a broken classifier for display (the classifier alone fires on most stories), but is trusted for training when AND-gated, because requiring the regex rule to agree filters out most of those false hits. If either list changes, check the other.

## Evidence terms

Scores in the guides come from different kinds of evidence. They are not interchangeable:

| Term | Meaning |
|------|---------|
| Human gold | Labels you verified by hand (`gold_labels.jsonl`, manual `fp_audit.py` reviews). |
| LLM-adjudicated | Labels decided by Claude (`fp_audit_auto_review.py`, `auto_label.py`). Good but not ground truth. |
| Community agreement | Agreement with Svengali tags. Svengali is incomplete, so a "false positive" against it is sometimes a real tag. |
| Cross-validated (OOF) | Out-of-fold scores within the training data. Optimistic when the same data also chose the threshold. |
| Validation | Scores on the val split, used to choose thresholds. |
| Held-out test | Scores on the test split, used for nothing else. The only number to quote as "test performance". |
| Subjective rating | Your judgement of generated text. |
