# ═══════════════════════════════════════════════════════════════════════════════
# MCScraper — Makefile
# All commands run from the repository root
# ═══════════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  := $(CURDIR)
VENV         := $(PROJECT_DIR)/.venv
PY           := $(VENV)/bin/python3
PIP          := $(PY) -m pip

OUTPUT_DIR   := $(PROJECT_DIR)/output
INDEX        := $(OUTPUT_DIR)/stories_index.json

# ── Colours ────────────────────────────────────────────────────────────────────
BOLD  := \033[1m
GREEN := \033[0;32m
CYAN  := \033[0;36m
RESET := \033[0m

.DEFAULT_GOAL := help

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help
help:
	@echo ""
	@echo "$(BOLD)MCScraper — command reference$(RESET)"
	@echo ""
	@echo "$(CYAN)Setup$(RESET)"
	@echo "  make install          Create venv + install all dependencies"
	@echo "  make install-semantic Also install sentence-transformers (semantic tagging)"
	@echo "  make install-ocr      Also install tesseract + pytesseract (image/screenshot ingestion)"
	@echo "  make check            Verify all deps are installed and scripts parse cleanly"
	@echo ""
	@echo "$(CYAN)Estimate$(RESET)"
	@echo "  make estimate         Sample 30 live stories and estimate full dataset size"
	@echo "  make estimate-large   Sample 100 stories for a more accurate estimate"
	@echo ""
	@echo "$(CYAN)Scrape$(RESET)"
	@echo "  make test-scrape      Scrape first 20 stories (smoke test, \~35 seconds)"
	@echo "  make scrape           Full scrape, regex tagging (\~3 hours)"
	@echo "  make scrape-semantic  Full scrape + semantic tagging (\~3 hours)"
	@echo ""
	@echo "$(CYAN)Tag (post-scrape)$(RESET)"
	@echo "  make tag              Re-tag all stories in parallel (\~8 min, all CPU cores)"
	@echo "  make tag-slow         Re-tag with regex only, single-process (\~24 min)"
	@echo "  make tag-semantic     Re-tag with regex + semantic (\~30 min)"
	@echo "  make rebuild-jsonl    Rebuild JSONL training files from current index"
	@echo "  make eval-tags        Measure regex tagger precision/recall vs Svengali"
	@echo ""
	@echo "$(CYAN)SetFit ML classifiers$(RESET)"
	@echo "  make train-setfit         Fine-tune encoder + train classifiers (hours, GPU recommended)"
	@echo "  make retrain-classifiers  Retrain classifiers only — no encoder fine-tune (~30 sec)"
	@echo "  make eval-setfit          Evaluate SetFit classifiers (OOF F1, precision, recall)"
	@echo "  make tag-setfit           Apply trained SetFit classifiers to all stories"
	@echo ""
	@echo "$(CYAN)Update$(RESET)"
	@echo "  make update           Check for new stories (last week)"
	@echo "  make update-full      Check for new stories + run full post-update pipeline"
	@echo "  make post-update      Re-tag + rebuild JSONL + rebuild recs + FTS DB (after any index change)"
	@echo "  make build-search-db  Build FTS search database only (skips if up to date)"
	@echo "  make build-search-db-force  Force-rebuild FTS search database"
	@echo ""
	@echo "$(CYAN)Search$(RESET)"
	@echo "  make search           Open interactive search REPL"
	@echo "  make stats            Print dataset statistics"
	@echo ""
	@echo "$(CYAN)Test$(RESET)"
	@echo "  make test             Run full test suite"
	@echo "  make test-fast        Run fast tests only (no network calls)"
	@echo "  make test-verbose     Run tests with detailed output"
	@echo "  make test-semantic    Run semantic tagger tests (requires install-semantic)"
	@echo ""
	@echo "$(CYAN)Local Content Ingestion$(RESET)"
	@echo "  make ingest-dry FOLDER=/path  Preview batch import from a folder (no changes)"
	@echo "  make ingest-folder FOLDER=/path AUTHOR=\"Name\"  Batch import folder → index"
	@echo "  make ingest-file FILE=/path/story.txt TITLE=\"...\" AUTHOR=\"Name\"  Single file"
	@echo "  make ingest-pan       Import Pan/PanWhoWrites folder (pre-configured)"
	@echo ""
	@echo "$(CYAN)Maintenance$(RESET)"
	@echo "  make clean-output     Delete scraped output (keeps venv)"
	@echo "  make clean            Delete venv and output"
	@echo ""


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# install always runs pip — safe because pip skips already-current packages.
# This means re-running make install after the venv exists still catches any
# missing packages (e.g. pytest added after the venv was first created).
.PHONY: install
install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(BOLD)Creating virtual environment...$(RESET)"; \
		python3 -m venv $(VENV); \
	fi
	@$(PIP) install --upgrade pip --quiet
	@$(PIP) install requests beautifulsoup4 lxml pytest --quiet
	@echo "$(GREEN)✓ Environment ready$(RESET)"
	@echo "  Python: $$($(PY) --version)"
	@echo "  Venv:   $(VENV)"

.PHONY: install-semantic
install-semantic: install
	@echo "$(BOLD)Installing sentence-transformers (semantic tagging)...$(RESET)"
	$(PIP) install sentence-transformers --quiet
	@echo "$(GREEN)✓ Semantic tagging ready$(RESET)"
	@echo "  Note: first run will download the model (\~23 MB, cached after)"

.PHONY: install-ocr
install-ocr: install
	@echo "$(BOLD)Installing OCR dependencies (image/screenshot ingestion)...$(RESET)"
	@which tesseract >/dev/null 2>&1 || (echo "  Installing tesseract via Homebrew..." && brew install tesseract)
	$(PIP) install pytesseract pillow --quiet
	@echo "$(GREEN)✓ OCR ready — use ingest.py with .png/.jpg/.jpeg files$(RESET)"

# check: uses absolute paths so it works regardless of cwd
.PHONY: check
check: install
	@echo "$(BOLD)Checking dependencies...$(RESET)"
	@$(PY) -c "import requests; print('  requests            ', requests.__version__)"
	@$(PY) -c "import bs4; print('  beautifulsoup4      ', bs4.__version__)"
	@$(PY) -c "import lxml; print('  lxml                ', lxml.__version__)"
	@$(PY) -c "import pytest; print('  pytest              ', pytest.__version__)"
	@$(PY) -c "import sentence_transformers; print('  sentence-transformers', sentence_transformers.__version__)" 2>/dev/null \
		|| echo "  sentence-transformers   NOT installed  (run: make install-semantic)"
	@echo ""
	@echo "$(BOLD)Checking scripts...$(RESET)"
	@for f in scraper.py tagger.py search_stories.py estimate_size.py tests.py conftest.py; do \
		if [ -f "$(PROJECT_DIR)/$$f" ]; then \
			$(PY) -c "import ast; ast.parse(open('$(PROJECT_DIR)/$$f').read()); print('  ' + '$$f'.ljust(22) + 'OK')"; \
		else \
			echo "  $$f$(shell printf '%*s' $$((22 - $${#f})) '')MISSING (copy from repo)"; \
		fi; \
	done
	@echo "$(GREEN)✓ Check complete$(RESET)"


# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMATE
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: probe
probe: install
	@echo "$(BOLD)Probing mcstories.com URL structure...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) probe_site.py --follow

.PHONY: estimate
estimate: install
	@echo "$(BOLD)Estimating dataset size (sampling 30 stories)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) estimate_size.py --sample 30

.PHONY: estimate-large
estimate-large: install
	@echo "$(BOLD)Estimating dataset size (sampling 100 stories for accuracy)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) estimate_size.py --sample 100

.PHONY: estimate-from-index
estimate-from-index:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Reporting exact sizes from local index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) estimate_size.py --from-index --index $(INDEX)


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPE
# ═══════════════════════════════════════════════════════════════════════════════

$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

.PHONY: test-scrape
test-scrape: install $(OUTPUT_DIR)
	@echo "$(BOLD)Test scrape — 20 stories (\~35 seconds)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) scraper.py \
		--output $(OUTPUT_DIR) \
		--limit 20
	@echo ""
	@echo "$(GREEN)✓ Test scrape complete. Output in $(OUTPUT_DIR)$(RESET)"
	@ls -l $(OUTPUT_DIR)

.PHONY: scrape
scrape: install $(OUTPUT_DIR)
	@echo "$(BOLD)Full scrape with regex tagging...$(RESET)"
	@echo "  Estimated time: \~17 hours"
	@echo "  Output:         $(OUTPUT_DIR)"
	@echo "  Safe to Ctrl+C — resumes automatically from where it stopped"
	@echo ""
	cd $(PROJECT_DIR) && $(PY) scraper.py \
		--output $(OUTPUT_DIR) \
		--delay 1.5
	@echo "$(GREEN)✓ Scrape complete$(RESET)"
	$(MAKE) estimate-from-index

.PHONY: resume
resume: install
	@echo "$(BOLD)Resuming scrape from last checkpoint...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) scraper.py \
		--output $(OUTPUT_DIR) \
		--delay 1.5
	@echo "$(GREEN)✓ Done$(RESET)"
	$(MAKE) estimate-from-index

.PHONY: scrape-semantic
scrape-semantic: install-semantic $(OUTPUT_DIR)
	@echo "$(BOLD)Full scrape with regex + semantic tagging...$(RESET)"
	@echo "  Estimated time: \~3 hours (semantic overhead negligible vs network delay)"
	@echo "  Note: model downloads on first run (\~23 MB)"
	@echo ""
	cd $(PROJECT_DIR) && $(PY) scraper.py \
		--output $(OUTPUT_DIR) \
		--delay 1.5 \
		--semantic
	@echo "$(GREEN)✓ Scrape complete$(RESET)"
	$(MAKE) estimate-from-index


# ═══════════════════════════════════════════════════════════════════════════════
# TAG  (post-scrape — run on an existing index)
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: tag
tag:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Re-tagging all stories in parallel (\~8 min)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) retag_parallel.py
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: tag-slow
tag-slow:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Re-tagging with regex layer, single-process (\~24 min)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) tagger.py --index $(INDEX)
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: eval-tags
eval-tags:
	@echo "$(BOLD)Evaluating regex tagger precision/recall vs Svengali...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) eval_tagger.py
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: train-setfit
train-setfit: install-semantic
	@echo "$(BOLD)Training SetFit classifiers (fine-tune + classifiers)...$(RESET)"
	@echo "  GPU strongly recommended. Use --skip-finetune to skip fine-tuning."
	cd $(PROJECT_DIR) && $(PY) setfit_trainer.py --train
	@echo "$(GREEN)✓ Done. Run: make tag-setfit$(RESET)"

.PHONY: train-setfit-fast
train-setfit-fast: install-semantic
	@echo "$(BOLD)Retraining SetFit classifiers only (skips encoder fine-tuning, ~30 sec)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) setfit_trainer.py --train --skip-finetune
	@echo "$(GREEN)✓ Done. Run: make retag-setfit$(RESET)"

.PHONY: retrain-classifiers
retrain-classifiers: install-semantic
	@echo "$(BOLD)Retraining classifiers from cached embeddings (no encoder fine-tuning)...$(RESET)"
	@echo "  Uses expanded tag list from retrain_classifiers.py"
	cd $(PROJECT_DIR) && $(PY) retrain_classifiers.py --eval
	@echo "$(GREEN)✓ Done. Run: make tag-setfit to apply to index$(RESET)"

.PHONY: eval-setfit
eval-setfit: install-semantic
	@echo "$(BOLD)Evaluating SetFit classifier agreement with Svengali...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) setfit_trainer.py --eval
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: tag-setfit
tag-setfit: install-semantic
	@echo "$(BOLD)Applying SetFit classifiers to all stories...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) setfit_trainer.py --tag
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: retag-setfit
retag-setfit: install-semantic
	@echo "$(BOLD)Clearing and re-applying SetFit classifiers (use after threshold changes)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) setfit_trainer.py --retag
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: tag-semantic
tag-semantic: install-semantic
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Re-tagging with regex + semantic (\~30 minutes)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) tagger.py --index $(INDEX) --semantic
	@echo "$(GREEN)✓ Done$(RESET)"


.PHONY: rebuild-jsonl
rebuild-jsonl:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Rebuilding JSONL training files from current index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) tagger.py --index $(INDEX) --jsonl-only
	@echo "$(GREEN)✓ Done$(RESET)"
	@ls -l $(OUTPUT_DIR)/*.jsonl


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

# build-search-db uses --no-fulltext by default: ~30 MB, instant cold start,
# searches title/author/description/categories/tags.
# For full in-body text search (1.6 GB, 4-5s cold start on network drives):
#   make build-search-db-full
.PHONY: build-stories-db
build-stories-db: install
	@echo "$(BOLD)Building stories_meta.db from stories_index.json...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) build_stories_db.py --index $(INDEX) --force
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: build-search-db
build-search-db: install
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	@echo "$(BOLD)Building FTS search database (slim)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) build_search_db.py --index $(INDEX) --no-fulltext
	@echo "$(GREEN)✓ search.db ready$(RESET)"

.PHONY: build-search-db-force
build-search-db-force: install
	@echo "$(BOLD)Rebuilding FTS search database (slim, forced)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) build_search_db.py --index $(INDEX) --force --no-fulltext
	@echo "$(GREEN)✓ search.db ready$(RESET)"

.PHONY: build-search-db-full
build-search-db-full: install
	@echo "$(BOLD)Building FTS search database with full-text (1.6 GB, slow cold start)...$(RESET)"
	@echo "  Enables searching inside story body text."
	cd $(PROJECT_DIR) && $(PY) build_search_db.py --index $(INDEX) --force
	@echo "$(GREEN)✓ search.db ready (full-text)$(RESET)"

.PHONY: post-update
# Pass SKIP_RETAG=1 to skip the full regex retag (use after make update, which already
# tags new stories during fetch). Run without the flag to force a full retag of all
# stories — needed when the regex tagging rules change.
SKIP_RETAG ?= 0
post-update: install-semantic
	@echo "$(BOLD)═══ Post-update pipeline ═══$(RESET)"
	@echo ""
	@if [ "$(SKIP_RETAG)" = "1" ]; then \
		echo "$(CYAN)Step 1/7 — Regex tagging skipped (new stories tagged during fetch)$(RESET)"; \
	else \
		echo "$(CYAN)Step 1/7 — Regex tagging (parallel, ~8 min)...$(RESET)"; \
		cd $(PROJECT_DIR) && $(PY) retag_parallel.py; \
	fi
	@echo ""
	@echo "$(CYAN)Step 2/7 — Re-importing Svengali community tags...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --import --index $(INDEX)
	@echo ""
	@echo "$(CYAN)Step 3/7 — SetFit ML classifier tagging (if trained)...$(RESET)"
	@if [ -f "$(OUTPUT_DIR)/setfit_classifiers.pkl" ]; then \
		cd $(PROJECT_DIR) && $(PY) retrain_classifiers.py --tag; \
	else \
		echo "  (skipped — run make train-setfit to enable)"; \
	fi
	@echo ""
	@echo "$(CYAN)Step 4/7 — Rebuilding JSONL training files...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) tagger.py --index $(INDEX) --jsonl-only
	@echo ""
	@echo "$(CYAN)Step 5/7 — Rebuilding recommendation cache...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) recommender.py --precalc --n-recs 200
	@echo ""
	@echo "$(CYAN)Step 6/7 — Rebuilding FTS search database (slim)...$(RESET)"
	rm -f $(PROJECT_DIR)/output/search.db $(PROJECT_DIR)/output/search.db-shm $(PROJECT_DIR)/output/search.db-wal
	cd $(PROJECT_DIR) && $(PY) build_search_db.py --index $(INDEX) --force --no-fulltext
	@echo ""
	@echo "$(CYAN)Step 6.5/7 — Rebuilding stories_meta.db (browser fast-load index)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) build_stories_db.py --index $(INDEX) --force
	@echo ""
	@echo "$(CYAN)Step 7/7 — Stats...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) search_stories.py --index $(INDEX) --stats
	@echo ""
	@echo "$(GREEN)✓ All done. Restart the browser to see new stories.$(RESET)"

.PHONY: update-full
update-full: install-semantic
	@echo "$(BOLD)═══ Check for new stories + full rebuild ═══$(RESET)"
	@echo ""
	@echo "$(CYAN)Fetching new stories...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) update.py --output $(OUTPUT_DIR)
	@echo ""
	$(MAKE) post-update SKIP_RETAG=1

.PHONY: precalc-recs
precalc-recs: install-semantic
	@echo "$(BOLD)Pre-calculating recommendation cache...$(RESET)"
	@echo "  Builds embeddings + saves similar stories for all 17k stories."
	@echo "  Re-run after: make tag-semantic, make svengali-import, or liking many stories."
	cd $(PROJECT_DIR) && $(PY) recommender.py --precalc --n-recs 200
	@echo "$(GREEN)✓ Cache saved to output/recs_cache.json$(RESET)"

# Only set up the venv if it doesn't exist yet — browse shouldn't hit the network
# (pip upgrade + reinstalls) on every launch.
$(PY):
	@$(MAKE) --no-print-directory install

# The browser's fast-load metadata DB, rebuilt only when the index is newer
# than it (e.g. after tagging, an update, or arc tagging on another machine).
$(OUTPUT_DIR)/stories_meta.db: $(INDEX) | $(PY)
	@echo "$(BOLD)stories_meta.db is older than the index — rebuilding...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) build_stories_db.py --index $(INDEX) --force

.PHONY: browse
browse: $(if $(wildcard $(INDEX)),$(OUTPUT_DIR)/stories_meta.db) | $(PY)
	@$(PY) -c "import flask" 2>/dev/null || $(PIP) install flask --quiet
	@echo "$(BOLD)Starting MC Archive browser...$(RESET)"
	@echo "  Open: http://localhost:5003"
	@echo "  Press Ctrl+C to stop"
	cd $(PROJECT_DIR) && $(PY) browser.py --index $(INDEX) --port 5003

.PHONY: search
search:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	cd $(PROJECT_DIR) && $(PY) search_stories.py --index $(INDEX)

.PHONY: stats
stats:
	@if [ ! -f "$(INDEX)" ]; then \
		echo "No index yet — run make scrape first."; exit 1; fi
	cd $(PROJECT_DIR) && $(PY) search_stories.py --index $(INDEX) --stats


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: test
test: install
	@echo "$(BOLD)Running full test suite...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) -m pytest tests.py -v

.PHONY: test-fast
test-fast: install
	@echo "$(BOLD)Running fast tests (no network)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) -m pytest tests.py -v -m "not network"

.PHONY: test-verbose
test-verbose: install
	@echo "$(BOLD)Running tests with full output...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) -m pytest tests.py -v -s --tb=long

.PHONY: test-semantic
test-semantic: install-semantic
	@echo "$(BOLD)Running semantic tagger tests...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) -m pytest tests.py -v -m "semantic"


# ═══════════════════════════════════════════════════════════════════════════════
# SVENGALI COMMUNITY TAGS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: svengali-probe
svengali-probe: install
	@echo "$(BOLD)Probing Svengali API...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --probe

.PHONY: svengali-scrape
svengali-scrape: install
	@echo "$(BOLD)Scraping Svengali tags (slow — be patient and polite)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --scrape

.PHONY: svengali-scrape-test
svengali-scrape-test: install
	@echo "$(BOLD)Test scrape — 50 stories only...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --scrape --limit 50

.PHONY: import-aiden
import-aiden: install
	@echo "$(BOLD)Importing Aiden PDF stories into index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_stories.py

.PHONY: import-aiden-update
import-aiden-update: install
	@echo "$(BOLD)Importing Aiden PDF stories (with full-text update)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_stories.py --update-existing

.PHONY: import-aiden-dry
import-aiden-dry: install
	@echo "$(BOLD)Dry run — previewing Aiden import...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_stories.py --dry-run

.PHONY: import-mindspark
import-mindspark: install
	@echo "$(BOLD)Importing MindSpark PDF stories into index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_mindspark.py

.PHONY: import-mindspark-update
import-mindspark-update: install
	@echo "$(BOLD)Importing MindSpark PDF stories (with full-text update)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_mindspark.py --update-existing

.PHONY: import-mindspark-dry
import-mindspark-dry: install
	@echo "$(BOLD)Dry run — previewing MindSpark import...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) import_local_mindspark.py --dry-run


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL CONTENT INGESTION
# Supports: .txt (plain / SOL export), .epub, .pdf, .docx, .png/.jpg (OCR)
# ═══════════════════════════════════════════════════════════════════════════════

# Pass FOLDER=, AUTHOR=, FILE=, TITLE= on the command line:
#   make ingest-folder FOLDER=/path/to/stories AUTHOR="Author Name"
FOLDER ?=
AUTHOR ?=
FILE   ?=
TITLE  ?=

.PHONY: ingest-dry
ingest-dry: install
	@if [ -z "$(FOLDER)" ]; then \
		echo "Usage: make ingest-dry FOLDER=/path/to/folder [AUTHOR=\"Name\"]"; exit 1; fi
	@echo "$(BOLD)Dry run — previewing batch import from $(FOLDER)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) ingest.py \
		--folder "$(FOLDER)" \
		$(if $(AUTHOR),--author "$(AUTHOR)",) \
		--index $(INDEX) \
		--dry-run --verbose

.PHONY: ingest-folder
ingest-folder: install
	@if [ -z "$(FOLDER)" ]; then \
		echo "Usage: make ingest-folder FOLDER=/path/to/folder [AUTHOR=\"Name\"]"; exit 1; fi
	@echo "$(BOLD)Batch ingesting $(FOLDER) → index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) ingest.py \
		--folder "$(FOLDER)" \
		$(if $(AUTHOR),--author "$(AUTHOR)",) \
		--index $(INDEX) \
		--patch --verbose
	@echo ""
	@echo "$(GREEN)✓ Ingest complete.  Run: make post-update$(RESET)"

.PHONY: ingest-file
ingest-file: install
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make ingest-file FILE=/path/to/story.txt [TITLE=\"...\"] [AUTHOR=\"Name\"]"; exit 1; fi
	@echo "$(BOLD)Ingesting single file: $(FILE)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) ingest.py \
		"$(FILE)" \
		$(if $(TITLE),--title "$(TITLE)",) \
		$(if $(AUTHOR),--author "$(AUTHOR)",) \
		--index $(INDEX) \
		--patch
	@echo ""
	@echo "$(GREEN)✓ Ingest complete.  Run: make post-update$(RESET)"

-include local.mk
PAN_FOLDER ?= $(error Set PAN_FOLDER in local.mk)

.PHONY: ingest-pan
ingest-pan: install
	@echo "$(BOLD)Batch ingesting Pan / PanWhoWrites folder...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) ingest.py \
		--folder $(PAN_FOLDER) \
		--author "Pan" \
		--index $(INDEX) \
		--patch --verbose
	@echo ""
	@echo "$(GREEN)✓ Pan import complete.  Run: make post-update$(RESET)"

.PHONY: ingest-pan-dry
ingest-pan-dry: install
	@echo "$(BOLD)Dry run — previewing Pan / PanWhoWrites import...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) ingest.py \
		--folder $(PAN_FOLDER) \
		--author "Pan" \
		--index $(INDEX) \
		--dry-run --verbose

.PHONY: svengali-import
svengali-import: install
	@echo "$(BOLD)Importing Svengali tags into index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --import

.PHONY: svengali-import-dry
svengali-import-dry: install
	@echo "$(BOLD)Dry run — previewing Svengali import...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --import --dry-run

.PHONY: svengali-stats
svengali-stats: install
	@echo "$(BOLD)Svengali coverage stats...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) svengali.py --stats

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT ARTIFACT FIXING (fix-text, not clean-text — avoid confusion with make clean)
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: fix-text
fix-text: install
	@echo "$(BOLD)Cleaning encoding artifacts from story text...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) fix_text.py --index $(INDEX)
	@echo "$(GREEN)✓ Done$(RESET)"

.PHONY: fix-text-dry
fix-text-dry: install
	@echo "$(BOLD)Dry run — previewing text fixes on 500 stories (no changes saved)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) fix_text.py --index $(INDEX) --dry-run --limit 500
	@echo "$(GREEN)✓ Done$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT REFRESH — re-fetch story text with fixed inline-element HTML extractor
# Skips wayback stories. Resumable. Only updates full_text + word_count.
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: refresh-text
refresh-text: install
	@echo "$(BOLD)Re-fetching story text with fixed HTML extractor (live stories only)...$(RESET)"
	@echo "$(CYAN)Wayback stories are skipped. Safe to Ctrl+C and resume with make refresh-text-resume$(RESET)"
	cd $(PROJECT_DIR) && $(PY) refresh_text.py --index $(INDEX)
	@echo "$(GREEN)✓ Done — run make post-update to re-tag and rebuild caches$(RESET)"

.PHONY: refresh-text-resume
refresh-text-resume: install
	@echo "$(BOLD)Resuming text refresh (skipping already-refreshed stories)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) refresh_text.py --index $(INDEX) --resume
	@echo "$(GREEN)✓ Done — run make post-update to re-tag and rebuild caches$(RESET)"

.PHONY: refresh-text-dry
refresh-text-dry: install
	@echo "$(BOLD)Dry run — previewing refresh on first 20 stories (no fetching)...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) refresh_text.py --index $(INDEX) --dry-run --limit 20
	@echo "$(GREEN)✓ Done$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# WEEKLY UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: update
update: install
	@echo "$(BOLD)Checking for new stories...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) update.py --output $(OUTPUT_DIR)

.PHONY: update-dry
update-dry: install
	@echo "$(BOLD)Dry run — showing new stories without scraping...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) update.py --dry-run --output $(OUTPUT_DIR)

.PHONY: update-catchup
update-catchup: install
	@echo "$(BOLD)Catching up — checking last 4 weeks...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) update.py --weeks 4 --full-check --output $(OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# NARRATIVE ARC DETECTION (run after fine-tuned model is in LM Studio)
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: arc-test
arc-test: install
	@echo "$(BOLD)Testing narrative arc detection on 20 stories...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) narrative_arc.py --limit 20

.PHONY: arc
arc: install
	@echo "$(BOLD)Detecting narrative arcs for all stories...$(RESET)"
	@echo "  Requires: LM Studio running with model loaded on port 1234"
	@echo "  ~15-30 hours for 17k stories. Safe to Ctrl+C and resume."
	cd $(PROJECT_DIR) && $(PY) narrative_arc.py --rebuild-jsonl

.PHONY: arc-stats
arc-stats: install
	@echo "$(BOLD)Narrative arc tag statistics...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) narrative_arc.py --stats

# ═══════════════════════════════════════════════════════════════════════════════
# WAYBACK MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

# Usage:  make wayback DATE=20200101
#         make wayback DATE=20150101 DATE=20180101 DATE=20210101
DATE ?= 20200101

.PHONY: wayback
wayback: install
	@echo "$(BOLD)Scraping deleted stories from Wayback Machine (date: $(DATE))...$(RESET)"
	@echo "  This finds stories archived on $(DATE) that no longer exist on the live site."
	@echo "  Delay: 2s between requests (archive.org requirement)"
	@echo ""
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py --date $(DATE) --output $(OUTPUT_DIR)

.PHONY: wayback-dry
wayback-dry: install
	@echo "$(BOLD)Dry run — showing what would be scraped (date: $(DATE))...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py --date $(DATE) --output $(OUTPUT_DIR) --dry-run

.PHONY: wayback-merge
wayback-merge: install
	@echo "$(BOLD)Merging Wayback stories into main index...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py --merge --output $(OUTPUT_DIR)

.PHONY: wayback-retry
wayback-retry: install
	@echo "$(BOLD)Retrying previously-failed wayback stories (polite single-worker mode)...$(RESET)"
	@echo "  NOTE: After a successful retry run, run 'make wayback-merge' to add new stories."
	@echo "  NOTE: Stories already in the main index are not re-merged even if recovered."
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py --retry-failed --output $(OUTPUT_DIR) --workers 1 --delay 3.0 --max-snapshots 10

.PHONY: wayback-probe
wayback-probe: install
	@echo "$(BOLD)Checking what mcstories.com content is in the Wayback Machine...$(RESET)"
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py --probe

.PHONY: wayback-multi
wayback-multi: install
	@echo "$(BOLD)Scanning multiple dates for maximum deleted story coverage...$(RESET)"
	@echo "  Dates: 2010, 2013, 2016, 2019, 2022"
	cd $(PROJECT_DIR) && $(PY) wayback_scraper.py \
		--date 20100101 \
		--date 20130101 \
		--date 20160101 \
		--date 20190101 \
		--date 20220101 \
		--output $(OUTPUT_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: clean-output
# ⚠ WARNING: clean and clean-output DELETE YOUR SCRAPED DATA
# Only run these if you want to start completely from scratch.
# Your stories_index.json, preferences.json etc will be GONE.
clean-output:
	@echo "$(BOLD)$(RED)WARNING: This will delete all scraped data in output/$(RESET)"
	@read -p "Type YES to confirm: " confirm && [ "$$confirm" = "YES" ]
	@echo "Removing output directory..."
	rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)✓ Output cleared$(RESET)"

.PHONY: clean
clean:
	@echo "$(BOLD)$(RED)WARNING: This will delete venv AND all scraped data$(RESET)"
	@read -p "Type YES to confirm: " confirm && [ "$$confirm" = "YES" ]
	@echo "Removing venv and output..."
	rm -rf $(VENV) $(OUTPUT_DIR)
	@echo "$(GREEN)✓ Clean$(RESET)"
