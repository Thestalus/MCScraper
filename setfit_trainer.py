#!/usr/bin/env python3
"""
setfit_trainer.py — Fine-tune all-mpnet-base-v2 on Svengali labels using SetFit.

SetFit adapts the embedding model to this domain via contrastive learning on
positive/negative story pairs, then trains per-tag classifiers with calibrated
thresholds. Expected improvement: +10–20% F1 over frozen embeddings.

Requirements:
    pip install setfit datasets

Workflow:
    python setfit_trainer.py --train    # fine-tune encoder + train classifiers
    python setfit_trainer.py --eval     # evaluate on held-out Svengali stories
    python setfit_trainer.py --tag      # apply tags to all stories in the index

Outputs:
    output/setfit_model/            fine-tuned sentence transformer
    output/setfit_classifiers.pkl   per-tag classifiers + calibrated thresholds
    output/setfit_embeddings.npz    story embeddings (multi-chunk max-pooled)
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Must be set before torch/transformers import to avoid MPS OOM on large multi-label batches
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_PATH       = "output/stories_index.json"
AUDIT_PATH       = "output/fp_audit_results.jsonl"
BASE_MODEL       = "sentence-transformers/all-mpnet-base-v2"
SETFIT_MODEL_DIR = "output/setfit_model"
CLF_PATH         = "output/setfit_classifiers.pkl"
EMBED_CACHE      = "output/setfit_embeddings.npz"

# ── Sampling ──────────────────────────────────────────────────────────────────
CHAPTER_SEP   = "\n\n* * *\n\n"  # chapter delimiter (same as tagger.py)
SAMPLE_WORDS  = 150   # words per position (head/mid/tail) per chapter for contrastive training
MAX_CHAPTERS  = 6     # cap chapters sampled per story to keep text length manageable
N_CHUNKS      = 5     # chunks per story for full classification embeddings
CHUNK_WORDS   = 300   # words per chunk

# ── SetFit hyperparameters ────────────────────────────────────────────────────
# Mac MPS (current):   NUM_ITERATIONS=5,  BATCH_SIZE=4,  MAX_TRAIN_STORIES=2000  (~3h)
# Windows CUDA:        NUM_ITERATIONS=20, BATCH_SIZE=16, MAX_TRAIN_STORIES=4000  (~25min)
NUM_ITERATIONS  = 15    # contrastive pairs per label per epoch
NUM_EPOCHS      = 1     # contrastive training epochs
BATCH_SIZE      = 8     # Mac MPS: 4 | Windows CUDA: 8 (16 fills 8GB VRAM)
MAX_TRAIN_STORIES = 4000  # Mac MPS: 2000 | Windows CUDA: 4000
FINETUNE_DEVICE = "cuda"  # Mac: "mps" | Windows: "cuda"

# ── Classifier ────────────────────────────────────────────────────────────────
MIN_POSITIVES  = 50   # skip tags with fewer labeled training examples

# ── Tags to train ─────────────────────────────────────────────────────────────
TARGET_TAGS = [
    # ── Awareness ────────────────────────────────────────────────────────────
    ("awareness",         "valiant but futile struggle"),
    # consensual   → disabled: P3% against Svengali, essentially noise
    # denial       → feature-only: kept in auto_tags for recs but not browser filter
    ("awareness",         "denial"),
    # playing along / skeptical+confused → require full narrative arc, dropped

    # ── Mental changes ────────────────────────────────────────────────────────
    ("mental_changes",    "obedience programming"),
    # speech modification → feature-only: P16% precision, high recall only
    ("mental_changes",    "speech modification"),
    # emotion control → disabled: P4% against Svengali, noise
    ("mental_changes",    "induced love"),
    ("mental_changes",    "memory alteration"),
    ("mental_changes",    "addiction / need"),

    # ── Mind control type ─────────────────────────────────────────────────────
    # virus / nanotechnology / pheromones / psionic / virtual reality → regex-only
    # (SetFit was getting P5-11% on these — they're keyword-definitive, not semantic)
    ("mind_control_type", "hypnosis"),
    ("mind_control_type", "possession"),
    ("mind_control_type", "body puppet"),

    # ── Dominance ─────────────────────────────────────────────────────────────
    # self-inflicted → regex handles (DOMINANCE_RULES); removed from SetFit (P6%)
    # subtle / gradual → regex handles
    # gentle → feature-only: P27% in eval, high recall but too many FPs for visible filter
    ("dominance",         "gentle"),
    # brutal → feature-only: P10% in eval, antonym confusion with gentle
    ("dominance",         "brutal"),

    # ── Pacing ────────────────────────────────────────────────────────────────
    ("pacing",            "leisurely"),
    ("pacing",            "rapid"),
    # epic → removed from SetFit: now covered by infer_pacing_from_structure (word-count based)

    # ── Relationship ──────────────────────────────────────────────────────────
    # teacher/student / boss/employee / doctor/patient → regex handles role keywords
    ("relationship",      "strangers"),
    ("relationship",      "friends"),
    ("relationship",      "couples"),
    ("relationship",      "peers"),
    ("relationship",      "enemies"),
    ("relationship",      "authority"),
]

# Tags generated by SetFit but only used as recommendation features — not shown
# as browser filter options. Kept in training so they remain in auto_tags.
FEATURE_ONLY_TAGS = {
    ("awareness",      "denial"),
    ("dominance",      "gentle"),
    ("dominance",      "brutal"),
    ("mental_changes", "speech modification"),
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_index() -> list[dict]:
    log.info(f"Loading index from {INDEX_PATH}...")
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", data) if isinstance(data, dict) else data
    log.info(f"  {len(stories):,} stories loaded")
    return stories


def story_sample(story: dict) -> str:
    """
    Chapter-aware text sample for SetFit contrastive training.
    For each chapter (up to MAX_CHAPTERS): takes head + mid + tail (SAMPLE_WORDS each).
    Falls back to whole-story head/mid/tail if no chapter markers found.
    """
    title = story.get("title", "")
    desc  = story.get("description", "") or ""
    text  = story.get("full_text", "") or ""

    def _sample_chunk(chunk_text: str) -> str:
        words = chunk_text.split()
        n     = len(words)
        head  = words[:SAMPLE_WORDS]
        mid   = words[max(0, n//2 - SAMPLE_WORDS//2) : n//2 + SAMPLE_WORDS//2]
        tail  = words[max(0, n - SAMPLE_WORDS):]
        return " [...] ".join(" ".join(c) for c in [head, mid, tail] if c)

    if CHAPTER_SEP in text:
        chapters = text.split(CHAPTER_SEP)[:MAX_CHAPTERS]
        body = " [ch] ".join(_sample_chunk(ch) for ch in chapters)
    else:
        body = _sample_chunk(text)

    return f"{title}. {desc}. {body}".strip()


def story_chunks(story: dict) -> list[str]:
    """Multi-chunk text sampling for full classification embeddings (same as sv_classifier)."""
    title  = story.get("title", "")
    desc   = story.get("description", "") or ""
    text   = story.get("full_text", "") or ""
    prefix = f"{title}. {desc}. "
    words  = text.split()
    n      = len(words)
    chunks = []
    for i in range(N_CHUNKS):
        start  = int((i / N_CHUNKS) * max(0, n - CHUNK_WORDS))
        window = words[start: start + CHUNK_WORDS]
        chunks.append((prefix + " ".join(window)).strip())
    return chunks


def svengali_stories(stories: list[dict]) -> list[dict]:
    """Return only stories that have Svengali community tags."""
    return [s for s in stories
            if (s.get("auto_tags") or {}).get("_svengali_tags")]


def has_text(story: dict) -> bool:
    return bool(story.get("full_text") or story.get("_has_text"))


# ── Audit label loading ────────────────────────────────────────────────────────

def load_audit_labels(path: str = AUDIT_PATH) -> dict:
    """
    Load LLM/human audit labels from fp_audit_results.jsonl.

    Returns dict: (story_id, "dim:tag") -> int
        1  = positive  (TRUE_POSITIVE or FALSE_NEGATIVE)
        0  = negative  (FALSE_POSITIVE or TRUE_NEGATIVE)
        (AMBIGUOUS / SKIP entries are excluded entirely)

    Later entries in the file override earlier ones for the same (story_id, tag) pair.
    """
    labels: dict[tuple[str, str], int] = {}
    p = Path(path)
    if not p.exists():
        log.info(f"No audit file found at {path} — using Svengali labels only")
        return labels
    positive_labels = {"TRUE_POSITIVE", "FALSE_NEGATIVE"}
    negative_labels = {"FALSE_POSITIVE", "TRUE_NEGATIVE"}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid   = row.get("story_id", "")
            tkey  = row.get("tag", "")      # e.g. "awareness:oblivious"
            label = row.get("label", "")
            if not sid or not tkey:
                continue
            if label in positive_labels:
                labels[(sid, tkey)] = 1
            elif label in negative_labels:
                labels[(sid, tkey)] = 0
            # AMBIGUOUS / SKIP → skip
    n_pos = sum(1 for v in labels.values() if v == 1)
    n_neg = sum(1 for v in labels.values() if v == 0)
    log.info(f"Audit labels loaded: {n_pos} positive, {n_neg} negative "
             f"across {len({k[1] for k in labels})} tags")
    return labels


def count_gold_positives(stories: list[dict], dim: str, tag: str,
                         audit_labels: dict) -> int:
    """
    Count unique stories with a confirmed positive label for (dim, tag).
    Priority: audit gold first, then Svengali-sourced tag.
    Used to decide whether a tag has enough data to train on.
    """
    tag_key = f"{dim}:{tag}"
    counted: set[str] = set()
    # Audit gold positives
    for (sid, tkey), lbl in audit_labels.items():
        if tkey == tag_key and lbl == 1:
            counted.add(sid)
    # Svengali fallback (skip if already counted)
    for s in stories:
        sid = s.get("id", "")
        if sid in counted:
            continue
        at      = s.get("auto_tags") or {}
        sources = at.get("_tag_sources") or {}
        sv_tags = [t for t in at.get(dim, [])
                   if "svengali" in (sources.get(t) or [])]
        if tag in sv_tags:
            counted.add(sid)
    return len(counted)


# ── Phase 1: SetFit contrastive fine-tuning ───────────────────────────────────

def finetune_encoder(stories: list[dict], active_tags: list[tuple[str, str]],
                     audit_labels: dict | None = None,
                     use_svengali_fallback: bool = True) -> None:
    """
    Fine-tune the sentence transformer using SetFit contrastive learning.
    Label vectors use hybrid evidence: audit gold first, then Svengali fallback.
    Saves fine-tuned model to SETFIT_MODEL_DIR.
    """
    try:
        from setfit import SetFitModel, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError:
        log.error("SetFit not installed. Run: pip install setfit datasets")
        sys.exit(1)

    if audit_labels is None:
        audit_labels = {}

    # Training pool: stories with audit labels OR Svengali labels (and have text)
    auditlabeled_ids = {sid for (sid, _) in audit_labels}
    sv_ids = {s.get("id") for s in svengali_stories(stories)}
    pool = [s for s in stories
            if has_text(s) and (s.get("id") in auditlabeled_ids or s.get("id") in sv_ids)]

    if len(pool) > MAX_TRAIN_STORIES:
        import random
        random.seed(42)
        pool = random.sample(pool, MAX_TRAIN_STORIES)
    log.info(f"SetFit contrastive training on {len(pool):,} labeled stories "
             f"({len(auditlabeled_ids & {s.get('id') for s in pool}):,} with audit labels)")

    label_names = [f"{dim}:{tag}" for dim, tag in active_tags]

    # Build dataset: text + multi-hot label vector per story
    # Priority: audit gold label > Svengali > implicit 0 (unknown)
    texts  = []
    labels = []
    for s in pool:
        sid = s.get("id", "")
        at      = s.get("auto_tags") or {}
        sources = at.get("_tag_sources") or {}
        text = story_sample(s)
        vec = []
        for dim, tag in active_tags:
            tag_key = f"{dim}:{tag}"
            audit_key = (sid, tag_key)
            if audit_key in audit_labels:
                # Gold audit label
                vec.append(audit_labels[audit_key])
            elif use_svengali_fallback:
                # Svengali label (only svengali-sourced tags, not regex/setfit)
                sv_tags = [t for t in at.get(dim, [])
                           if "svengali" in (sources.get(t) or [])]
                vec.append(1 if tag in sv_tags else 0)
            else:
                vec.append(0)  # unknown → implicit 0 (SetFit limitation)
        texts.append(text)
        labels.append(vec)

    train_ds = Dataset.from_dict({"text": texts, "label": labels})

    import platform
    if platform.system() == "Darwin" and FINETUNE_DEVICE == "mps":
        estimated_minutes = MAX_TRAIN_STORIES * NUM_ITERATIONS * NUM_EPOCHS * 3 * SAMPLE_WORDS // (300 * 60)
        log.warning(
            f"Running on Mac MPS with SAMPLE_WORDS={SAMPLE_WORDS} (multi-chunk). "
            f"Estimated fine-tuning time: ~{estimated_minutes}+ minutes. "
            "For faster runs, reduce SAMPLE_WORDS to 300 or use a Windows/CUDA machine."
        )

    # Start from existing fine-tuned encoder if available, else base model
    start_model = SETFIT_MODEL_DIR if Path(SETFIT_MODEL_DIR).exists() else BASE_MODEL
    log.info(f"Building SetFit model from {start_model} (device={FINETUNE_DEVICE})...")
    model = SetFitModel.from_pretrained(
        start_model,
        multi_target_strategy="one-vs-rest",
        labels=label_names,
        device=FINETUNE_DEVICE,
        use_differentiable_head=False,
    )

    args = TrainingArguments(
        output_dir=SETFIT_MODEL_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        num_iterations=NUM_ITERATIONS,
        evaluation_strategy="no",
        save_strategy="no",
        show_progress_bar=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )

    log.info("Starting contrastive fine-tuning...")
    t0 = time.time()
    trainer.train()
    log.info(f"Fine-tuning complete in {(time.time()-t0)/60:.1f} min")

    # Save only the encoder (sentence transformer), not the classification head
    # The head will be replaced by our own calibrated classifiers
    Path(SETFIT_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.model_body.save(SETFIT_MODEL_DIR)
    log.info(f"Fine-tuned encoder saved to {SETFIT_MODEL_DIR}/")


# ── Phase 2: Build embeddings with fine-tuned model ───────────────────────────

def encoder_fingerprint(model_path: str) -> str:
    """Identify the encoder (and chunking settings) that produced cached vectors.

    For a local fine-tuned model this hashes each file's name, size and mtime,
    so re-running finetune_encoder() changes the fingerprint and forces a full
    re-embed. For a hub model name it hashes the name.
    """
    h = hashlib.sha1(f"chunks={N_CHUNKS}x{CHUNK_WORDS}|".encode())
    p = Path(model_path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                st = f.stat()
                h.update(f"{f.relative_to(p)}:{st.st_size}:{int(st.st_mtime)}|".encode())
    else:
        h.update(model_path.encode())
    return h.hexdigest()


def _content_hash(chunks: list[str]) -> str:
    return hashlib.sha1("\x00".join(chunks).encode("utf-8")).hexdigest()


def build_embeddings(stories: list[dict], rebuild: bool = False) -> np.ndarray:
    """
    Embed all stories using multi-chunk max-pooling with the fine-tuned model.
    Returns (array of shape (n_stories, dim), valid stories). Caches to EMBED_CACHE.

    The cache is reused only when it was built by the same encoder (see
    encoder_fingerprint) and a story's text chunks are unchanged; otherwise the
    affected vectors are recomputed. rebuild=True ignores the cache entirely.
    """
    from sentence_transformers import SentenceTransformer

    model_path = SETFIT_MODEL_DIR if Path(SETFIT_MODEL_DIR).exists() else BASE_MODEL
    fingerprint = encoder_fingerprint(model_path)
    log.info(f"Loading encoder from {model_path}...")
    model = SentenceTransformer(model_path)

    valid   = [s for s in stories if has_text(s)]
    chunks  = {s["id"]: story_chunks(s) for s in valid}
    hashes  = {sid: _content_hash(c) for sid, c in chunks.items()}

    # Load existing cache: id -> (vector, content hash)
    cached: dict[str, tuple[np.ndarray, str]] = {}
    if Path(EMBED_CACHE).exists() and not rebuild:
        log.info(f"Loading embedding cache from {EMBED_CACHE}...")
        npz = np.load(EMBED_CACHE, allow_pickle=True)
        cache_fp = str(npz["encoder_fp"]) if "encoder_fp" in npz.files else None
        if cache_fp is not None and cache_fp != fingerprint:
            log.warning("  Encoder changed since the cache was built — re-embedding every story")
        else:
            if cache_fp is None:
                # Caches from before fingerprints were stored. Re-embedding all
                # stories takes hours, so adopt it and stamp the fingerprint; run
                # with --rebuild-embeddings if it may predate the current encoder.
                log.warning("  Cache has no encoder fingerprint; assuming it matches the "
                            "current encoder (use --rebuild-embeddings to be sure)")
            old_hashes = (npz["content_hashes"].tolist() if "content_hashes" in npz.files
                          else [None] * len(npz["ids"]))
            for sid, vec, ch in zip(npz["ids"].tolist(), npz["vecs"], old_hashes):
                # A missing content hash (old cache) is trusted, like the fingerprint
                cached[str(sid)] = (vec, ch if ch is not None else hashes.get(str(sid), ""))
        log.info(f"  Reusable cached embeddings: {len(cached):,}")

    to_embed = [s for s in valid
                if s["id"] not in cached or cached[s["id"]][1] != hashes[s["id"]]]

    if to_embed:
        log.info(f"Embedding {len(to_embed):,} new/changed stories × {N_CHUNKS} chunks...")
        all_chunks = []
        for s in to_embed:
            all_chunks.extend(chunks[s["id"]])
        t0  = time.time()
        raw = model.encode(
            all_chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        dim      = raw.shape[1]
        reshaped = raw.reshape(len(to_embed), N_CHUNKS, dim)
        new_vecs = np.max(reshaped, axis=1)
        log.info(f"  Embedded in {(time.time()-t0)/60:.1f} min")
        for s, v in zip(to_embed, new_vecs):
            cached[s["id"]] = (v, hashes[s["id"]])

    # Keep only current stories; save if anything changed or the file needs stamping
    keep_ids = [s["id"] for s in valid if s["id"] in cached]
    cached_ids  = keep_ids
    cached_vecs = np.vstack([cached[sid][0] for sid in keep_ids]) if keep_ids else np.zeros((0, 1), np.float32)
    needs_save = bool(to_embed) or not Path(EMBED_CACHE).exists() or rebuild
    if not needs_save:
        npz_files = np.load(EMBED_CACHE, allow_pickle=True).files
        needs_save = "encoder_fp" not in npz_files or "content_hashes" not in npz_files
    if needs_save and keep_ids:
        tmp_base = Path(EMBED_CACHE).parent / f"_setfit_embed_tmp_{os.getpid()}"
        np.savez_compressed(str(tmp_base), ids=np.array(keep_ids), vecs=cached_vecs,
                            content_hashes=np.array([cached[sid][1] for sid in keep_ids]),
                            encoder_fp=np.array(fingerprint))
        Path(str(tmp_base) + ".npz").replace(EMBED_CACHE)
        log.info(f"  Cache saved: {len(keep_ids):,} embeddings")

    # Return in valid order
    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
    result = np.zeros((len(valid), cached_vecs.shape[1]), dtype=np.float32)
    for i, s in enumerate(valid):
        idx = id_to_idx.get(s["id"])
        if idx is not None:
            result[i] = cached_vecs[idx]
    return result, valid


# ── Phase 3: Train classifiers with calibrated thresholds ─────────────────────

def _hybrid_labels(stories: list[dict], candidate_idx: np.ndarray,
                   dim: str, tag: str,
                   audit_labels: dict,
                   use_svengali_fallback: bool = True,
                   max_neg_ratio: int = 5) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Hybrid label construction using audit gold labels + Svengali fallback.

    Label priority for each story:
      1. LLM/human audit gold label (overrides everything)
      2. Svengali positive for this exact tag  → positive
      3. Svengali labeled different tag in same dimension → negative
      4. Otherwise → unknown, SKIPPED (not used as negative)

    This is strictly better than PU labeling because:
      - Svengali absence is NOT treated as negative
      - Audit-verified TPs/TNs are preferred over Svengali inference
      - New tags with no Svengali coverage still get labels from audit

    Returns:
        all_idx:  np.ndarray of story indices
        y:        np.ndarray of labels (1=pos, 0=neg)
        sources:  dict with counts: gold_pos, gold_neg, sv_pos, sv_neg
    """
    tag_key = f"{dim}:{tag}"
    rng = np.random.default_rng(42)

    gold_pos, gold_neg = [], []
    sv_pos,   sv_neg   = [], []

    for i in candidate_idx:
        story = stories[int(i)]
        sid   = story.get("id", "")
        audit_key = (sid, tag_key)

        if audit_key in audit_labels:
            # Priority 1: gold audit label
            if audit_labels[audit_key] == 1:
                gold_pos.append(int(i))
            else:
                gold_neg.append(int(i))
            continue

        if not use_svengali_fallback:
            continue

        # Priority 2/3: Svengali fallback
        at      = (story.get("auto_tags") or {})
        sources_map = (at.get("_tag_sources") or {})
        sv_dim_tags = [t for t in at.get(dim, [])
                       if "svengali" in (sources_map.get(t) or [])]
        if tag in sv_dim_tags:
            sv_pos.append(int(i))
        elif sv_dim_tags:
            # Has a different Svengali-labeled tag in this dimension → reliable negative
            sv_neg.append(int(i))
        # else: no label for this dimension → skip

    pos_idx = gold_pos + sv_pos
    neg_idx = gold_neg + sv_neg

    # Cap negatives so they don't swamp positives
    max_neg = len(pos_idx) * max_neg_ratio
    if len(neg_idx) > max_neg:
        neg_idx = list(rng.choice(neg_idx, max_neg, replace=False))

    all_idx = np.array(pos_idx + neg_idx, dtype=int)
    y       = np.array([1] * len(pos_idx) + [0] * len(neg_idx), dtype=int)
    label_sources = {
        "gold_pos": len(gold_pos), "gold_neg": len(gold_neg),
        "sv_pos":   len(sv_pos),   "sv_neg":   len(sv_neg),
    }
    return all_idx, y, label_sources


def train_classifiers(stories: list[dict], vecs: np.ndarray,
                      active_tags: list[tuple[str, str]],
                      audit_labels: dict | None = None,
                      use_svengali_fallback: bool = True) -> dict:
    """
    Train per-tag logistic regression classifiers with OOF threshold calibration.
    Uses hybrid labels: audit gold first, Svengali fallback second.
    Svengali absence is never treated as negative.
    """
    if audit_labels is None:
        audit_labels = {}

    # Candidate pool: all stories with embeddings (not just Svengali-tagged)
    all_idx = np.arange(len(stories))

    classifiers = {}
    print(f"\n{'Dimension':<20} {'Tag':<35} {'Pos':>5} {'Neg':>6}  "
          f"{'gPos':>5} {'gNeg':>5} {'sPos':>5} {'sNeg':>5}  "
          f"{'CV F1':>6}  {'OOF Thr':>7}  {'OOF F1':>6}")
    print("-" * 115)

    for dim, tag in active_tags:
        tag_idx, y, srcs = _hybrid_labels(
            stories, all_idx, dim, tag, audit_labels,
            use_svengali_fallback=use_svengali_fallback,
        )
        tag_vecs = vecs[tag_idx]
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < MIN_POSITIVES:
            log.info(f"  Skipping {dim}:{tag} — only {n_pos} positives "
                     f"(gold={srcs['gold_pos']}, sv={srcs['sv_pos']})")
            continue

        clf = LogisticRegression(C=0.5, class_weight="balanced",
                                 max_iter=1000, random_state=42)

        # CV F1 estimate
        from sklearn.model_selection import StratifiedKFold
        cv_preds = cross_val_predict(clf, tag_vecs, y, cv=5,
                                     method="predict", n_jobs=-1)
        cv_f1 = f1_score(y, cv_preds, zero_division=0)

        # OOF probability for threshold calibration
        oof_proba = cross_val_predict(clf, tag_vecs, y, cv=5,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        best_thr, best_f1 = 0.5, 0.0
        for thr in np.arange(0.25, 0.95, 0.01):
            preds = (oof_proba >= thr).astype(int)
            f = f1_score(y, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(thr)

        # Train final classifier on all confident-label data
        clf.fit(tag_vecs, y)
        classifiers[f"{dim}:{tag}"] = {
            "clf":           clf,
            "dim":           dim,
            "tag":           tag,
            "opt_threshold": best_thr,
            "oof_f1":        best_f1,
            "n_pos":         int(n_pos),
        }
        print(f"  {dim:<20} {tag:<35} {n_pos:>5,} {n_neg:>6,}  "
              f"{srcs['gold_pos']:>5} {srcs['gold_neg']:>5} "
              f"{srcs['sv_pos']:>5} {srcs['sv_neg']:>5}  "
              f"{cv_f1:>6.3f}  {best_thr:>7.2f}  {best_f1:>6.3f}")

    print(f"\nTrained {len(classifiers)} classifiers.")
    return classifiers


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_classifiers(stories: list[dict], vecs: np.ndarray,
                     classifiers: dict, threshold: float = 0.0,
                     audit_labels: dict | None = None,
                     use_svengali_fallback: bool = True) -> None:
    """Evaluate classifiers using hybrid labels (audit gold first, Svengali fallback).

    Uses a held-out 20% split of all confidently-labeled stories (not just Svengali).
    N varies per row because each tag has a different confident-example pool.
    """
    if audit_labels is None:
        audit_labels = {}

    # Eval pool: all stories (hybrid labels determine which have confident labels)
    all_idx = np.arange(len(stories))
    # Hold out 20% by story index for eval
    _, eval_idx = train_test_split(all_idx, test_size=0.2, random_state=42)

    print(f"\nEvaluating on up to {len(eval_idx):,} held-out stories (N varies per tag)...\n")
    print(f"{'Dimension':<20} {'Tag':<35} {'OOF Thr':>7}  "
          f"{'N':>5} {'P':>6} {'R':>6} {'F1':>6}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 105)

    all_tp = all_fp = all_fn = 0
    rows = []
    for key, info in classifiers.items():
        dim, tag = info["dim"], info["tag"]

        # Hybrid labels; don't cap negatives for eval
        tag_eval_idx, y, _ = _hybrid_labels(
            stories, eval_idx, dim, tag, audit_labels,
            use_svengali_fallback=use_svengali_fallback,
            max_neg_ratio=9999,
        )
        if len(tag_eval_idx) == 0 or y.sum() == 0:
            continue

        tag_eval_vecs = vecs[tag_eval_idx]
        thr   = info.get("opt_threshold", 0.5) if threshold == 0.0 else threshold
        proba = info["clf"].predict_proba(tag_eval_vecs)[:, 1]
        preds = (proba >= thr).astype(int)

        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        all_tp += tp; all_fp += fp; all_fn += fn
        rows.append((f1, dim, tag, thr, len(tag_eval_idx), p, r, f1, tp, fp, fn))

    rows.sort(key=lambda x: x[0])
    for _, dim, tag, thr, n, p, r, f1, tp, fp, fn in rows:
        print(f"  {dim:<20} {tag:<35} {thr:>7.2f}  "
              f"{n:>5,} {p:>5.0%} {r:>5.0%} {f1:>6.3f}  {tp:>5,} {fp:>5,} {fn:>5,}")

    mp = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    mr = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0
    print("-" * 105)
    print(f"  {'MICRO AVERAGE (confident labels only)':<63} {mp:>5.0%} {mr:>5.0%} {mf:>6.3f}  "
          f"{all_tp:>5,} {all_fp:>5,} {all_fn:>5,}")


# ── Tagging ───────────────────────────────────────────────────────────────────

def tag_all(stories: list[dict], vecs: np.ndarray,
            classifiers: dict, valid: list[dict],
            threshold: float = 0.0) -> None:
    """Apply classifier tags to all stories in the index."""
    id_to_vec = {s["id"]: vecs[i] for i, s in enumerate(valid)}
    tagged = 0
    for s in stories:
        vec = id_to_vec.get(s["id"])
        if vec is None:
            continue
        at      = s.setdefault("auto_tags", {})
        sources = at.setdefault("_tag_sources", {})
        changed = False
        for key, info in classifiers.items():
            dim, tag = info["dim"], info["tag"]
            thr  = info.get("opt_threshold", 0.5) if threshold == 0.0 else threshold
            prob = info["clf"].predict_proba(vec.reshape(1, -1))[0, 1]
            if prob >= thr:
                tags_list = at.setdefault(dim, [])
                if tag not in tags_list:
                    tags_list.append(tag)
                    changed = True
                src = sources.setdefault(tag, [])
                if "setfit" not in src:
                    src.append("setfit")
        if changed:
            tagged += 1

    log.info(f"Tagged {tagged:,} stories")

    tmp = Path(INDEX_PATH).with_suffix(".tmp")
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    story_map = {s["id"]: s for s in stories}
    target = data.get("stories", data) if isinstance(data, dict) else data
    for s in target:
        if s["id"] in story_map:
            s["auto_tags"] = story_map[s["id"]].get("auto_tags", {})
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(INDEX_PATH)
    log.info(f"Index updated: {INDEX_PATH}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SetFit trainer for story tags")
    parser.add_argument("--train",     action="store_true", help="Fine-tune encoder + train classifiers")
    parser.add_argument("--eval",      action="store_true", help="Evaluate classifiers on held-out set")
    parser.add_argument("--tag",       action="store_true", help="Apply tags to all stories (additive — won't remove existing SetFit tags)")
    parser.add_argument("--retag",     action="store_true", help="Clear existing SetFit tags then re-apply (use after threshold changes)")
    parser.add_argument("--rebuild-embeddings", action="store_true",
                        help="Ignore output/setfit_embeddings.npz and re-embed every story")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Override per-tag threshold (0 = use calibrated)")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="Skip SetFit fine-tuning, use existing model in output/setfit_model/")
    parser.add_argument("--gold-only", action="store_true",
                        help="Train/eval using only audited TP/TN labels — no Svengali fallback")
    args = parser.parse_args()

    if not any([args.train, args.eval, args.tag, args.retag]):
        parser.print_help()
        return

    use_svengali_fallback = not args.gold_only
    if args.gold_only:
        log.info("--gold-only: using audit labels exclusively, no Svengali fallback")

    stories = load_index()
    audit_labels = load_audit_labels()

    # Determine active tags — count gold positives (audit + Svengali, NOT raw auto_tags)
    active_tags = []
    for dim, tag in TARGET_TAGS:
        n_pos = count_gold_positives(stories, dim, tag, audit_labels)
        if n_pos >= MIN_POSITIVES:
            active_tags.append((dim, tag))
        else:
            log.info(f"  Skipping {dim}:{tag} — only {n_pos} gold positives (< {MIN_POSITIVES})")
    log.info(f"{len(active_tags)} tags with sufficient training data")

    if args.train:
        # Phase 1: fine-tune encoder (unless skipped)
        if not args.skip_finetune:
            finetune_encoder(stories, active_tags,
                             audit_labels=audit_labels,
                             use_svengali_fallback=use_svengali_fallback)
        else:
            log.info("Skipping fine-tuning (--skip-finetune)")

        # Phase 2: build embeddings
        vecs, valid = build_embeddings(stories, rebuild=args.rebuild_embeddings)

        # Phase 3: train classifiers (use valid only — vecs rows match valid, not stories)
        classifiers = train_classifiers(valid, vecs, active_tags,
                                        audit_labels=audit_labels,
                                        use_svengali_fallback=use_svengali_fallback)

        tmp = Path(CLF_PATH).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(classifiers, f)
        tmp.replace(CLF_PATH)
        log.info(f"Classifiers saved to {CLF_PATH}")

    if args.eval or args.tag or args.retag:
        if not Path(CLF_PATH).exists():
            log.error(f"No classifiers found at {CLF_PATH}. Run --train first.")
            sys.exit(1)
        with open(CLF_PATH, "rb") as f:
            classifiers = pickle.load(f)
        vecs, valid = build_embeddings(stories, rebuild=args.rebuild_embeddings)

    if args.eval:
        eval_classifiers(valid, vecs, classifiers, threshold=args.threshold,
                         audit_labels=audit_labels,
                         use_svengali_fallback=use_svengali_fallback)

    if args.retag:
        log.info("Clearing existing SetFit tags from index...")
        with open(INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        all_dims = set(info["dim"] for info in classifiers.values())
        for s in data.get("stories", []):
            if not s: continue
            at = s.get("auto_tags") or {}
            sources = at.get("_tag_sources") or {}
            for dim in all_dims:
                for tag in list(at.get(dim, [])):
                    src = sources.get(tag) or []
                    if "setfit" in src:
                        new_src = [x for x in src if x != "setfit"]
                        if new_src:
                            sources[tag] = new_src
                        else:
                            sources.pop(tag, None)
                            at.get(dim, []).remove(tag)
        tmp = Path(INDEX_PATH).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(INDEX_PATH)
        log.info("SetFit tags cleared. Re-loading and applying...")
        stories = load_index()
        vecs, valid = build_embeddings(stories, rebuild=args.rebuild_embeddings)
        tag_all(stories, vecs, classifiers, valid, threshold=args.threshold)

    if args.tag:
        tag_all(stories, vecs, classifiers, valid, threshold=args.threshold)


if __name__ == "__main__":
    main()
