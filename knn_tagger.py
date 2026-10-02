#!/usr/bin/env python3
"""
knn_tagger.py — RAG-style tag inference via nearest-neighbour retrieval

Two complementary ideas, both using the sentence-transformer model already
in tagger.py (all-MiniLM-L6-v2, 384-dim):

┌─────────────────────────────────────────────────────────────────────────────┐
│  IDEA 1 — k-NN vote from Svengali corpus                                    │
│                                                                             │
│  Embed all ~14,800 Svengali-tagged stories once.  At tag time, embed the   │
│  new story, find its k nearest neighbours in that corpus, and weight-vote  │
│  their tags.  Svengali tags are community ground truth — this leverages    │
│  them at inference, not just at training time.                             │
│                                                                             │
│  Best for: nuanced/subjective tags (dominance style, tone, awareness)      │
│  where the static description cosine in SemanticTagger is imprecise.       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  IDEA 2 — Mean-pooled full-text embedding                                   │
│                                                                             │
│  SemanticTagger only sees 600 words.  For novellas/novels the arc-level    │
│  tags (slow burn, corruption arc, identity loss, leisurely) live in the    │
│  middle and end of the story — they're invisible at 600 words.             │
│                                                                             │
│  Fix: split story into overlapping 500-word chunks, embed all of them,    │
│  mean-pool into one vector.  Drop-in replacement for the 600-word sample. │
│                                                                             │
│  Best for: pacing, themes (arc tags), awareness (gradually losing),        │
│  any tag that requires reading most of the story to judge.                 │
└─────────────────────────────────────────────────────────────────────────────┘

COMPUTING REQUIREMENTS
──────────────────────

  Model:      all-MiniLM-L6-v2 (already used in tagger.py)
              22 MB on disk, ~50 MB in RAM

  Idea 1 — build Svengali index (run once, ~2 min on Apple Silicon CPU):
    • 14,800 stories × 600 words each
    • batch encode at 64 stories/batch  →  232 batches × ~0.15 s  ≈  35 s
    • output: float32 array  14,800 × 384 × 4 B  =  22.7 MB  (.npy file)
    • + tags JSON:  ~1.5 MB
    • k-NN query at runtime:  embed 1 story (~1 ms)
      + numpy dot product 14,800×384  (<0.1 ms)  =  ~2 ms overhead per story

  Idea 2 — full-text embedding (run once during retag, ~5 min on Apple Silicon CPU):
    • average story ~6,000 words  →  ~23 chunks (500-word, 250-stride)
    • 17,600 stories × 23 chunks  =  ~405,000 encode calls
    • batched at 64:  ~6,300 batches × ~0.15 s  ≈  16 min on CPU
    • with MPS (Apple Silicon GPU):  ≈  3–4 min
    • mean-pooled output:  17,600 × 384 × 4 B  =  27 MB  (.npy file)
    • at runtime, chunking + encoding 1 story:  ~23 × 1 ms  ≈  25 ms
      (vs current ~1 ms for 600-word sample — acceptable for a retag run)

  Both ideas combined:
    • Total one-time build:  ~6–20 min depending on CPU vs MPS
    • Total index storage:  ~55 MB
    • No GPU required, no FAISS required (14,800 vectors fit easily in RAM)
    • For >200,000 stories you'd want FAISS; not needed here

  To enable MPS (Apple Silicon) for the encode step:
    import torch; device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

Usage:
    python knn_tagger.py --build           # build both indexes (run once)
    python knn_tagger.py --build --mps     # same, using Apple Silicon GPU
    python knn_tagger.py --demo            # tag a random story and print results
"""

from __future__ import annotations

import json
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

INDEX_PATH   = Path("./output/stories_index.json")
KNN_EMB_PATH = Path("./output/knn_svengali.npy")       # Idea 1 embeddings
KNN_TAG_PATH = Path("./output/knn_svengali_tags.json")  # Idea 1 tags
POOL_EMB_PATH = Path("./output/fulltext_pool.npy")      # Idea 2 embeddings
POOL_ID_PATH  = Path("./output/fulltext_pool_ids.json") # Idea 2 story ID order

# Dimensions to vote on (exclude regex-only dims — kNN adds nothing there)
KNN_DIMS = {
    "mind_control_type", "setting", "themes", "tone",
    "dominance", "pacing", "awareness", "prevalence", "mental_changes",
}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(model_name: str = "all-MiniLM-L6-v2", use_mps: bool = False):
    from sentence_transformers import SentenceTransformer
    device = "cpu"
    if use_mps:
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                log.info("Using Apple Silicon MPS for encoding")
        except ImportError:
            pass
    log.info(f"Loading {model_name} on {device}...")
    return SentenceTransformer(model_name, device=device)


def _short_sample(story: dict) -> str:
    """600-word sample — same as SemanticTagger._story_sample()."""
    title = story.get("title", "")
    desc  = story.get("description", "") or ""
    text  = story.get("full_text", "") or ""
    return f"{title}. {desc}. " + " ".join(text.split()[:600])


def _load_stories() -> list[dict]:
    with open(INDEX_PATH, encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", data) if isinstance(data, dict) else data
    return [s for s in stories if s.get("full_text") and not s.get("fetch_error")]


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 1 — k-NN SVENGALI INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_index(use_mps: bool = False) -> None:
    """
    Embed all Svengali-tagged stories and save the index to disk.
    Run once; takes ~35 s on CPU or ~8 s on MPS.
    """
    stories = _load_stories()

    # Only index stories that have Svengali community tags
    sv_stories = [
        s for s in stories
        if (s.get("auto_tags") or {}).get("_svengali_tags")
    ]
    log.info(f"Svengali-tagged stories to index: {len(sv_stories):,}")

    model = _load_model(use_mps=use_mps)
    samples = [_short_sample(s) for s in sv_stories]

    log.info("Encoding Svengali corpus...")
    embeddings = model.encode(
        samples,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )  # shape: (N, 384)

    # Save embeddings
    np.save(KNN_EMB_PATH, embeddings.astype(np.float32))
    log.info(f"Saved embeddings → {KNN_EMB_PATH}  ({embeddings.nbytes / 1e6:.1f} MB)")

    # Save just the tag dims we care about (keeps the JSON small)
    tag_list = []
    for s in sv_stories:
        at = s.get("auto_tags") or {}
        tag_list.append({dim: at.get(dim, []) for dim in KNN_DIMS})

    with open(KNN_TAG_PATH, "w", encoding="utf-8") as f:
        json.dump(tag_list, f)
    log.info(f"Saved tag index  → {KNN_TAG_PATH}")


class KNNTagger:
    """
    At tag time: embed the new story, dot-product against all 14,800 Svengali
    embeddings, take the top-k, weight-vote their tags.

    Plugs into the merge step in tagger.py after SemanticTagger.
    """

    def __init__(self, k: int = 15, sim_threshold: float = 0.60,
                 vote_threshold: float = 0.25,
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        k               nearest neighbours to consider
        sim_threshold   min cosine similarity to count a neighbour (filters noise)
        vote_threshold  fraction of k that must vote for a tag to include it
                        e.g. 0.25 → tag must appear in 4 of 15 neighbours
        """
        self.k              = k
        self.sim_threshold  = sim_threshold
        self.vote_threshold = vote_threshold
        self.model_name     = model_name
        self._model         = None
        self._embeddings: Optional[np.ndarray] = None  # (N, 384)
        self._tags: Optional[list[dict]]       = None

    def _load(self):
        if self._model is not None:
            return
        if not KNN_EMB_PATH.exists() or not KNN_TAG_PATH.exists():
            raise RuntimeError(
                "k-NN index not built. Run:  python knn_tagger.py --build")
        self._model      = _load_model(self.model_name)
        self._embeddings = np.load(KNN_EMB_PATH)  # mmap is fine: read-only
        with open(KNN_TAG_PATH, encoding="utf-8") as f:
            self._tags = json.load(f)
        log.info(f"KNNTagger loaded: {len(self._tags):,} stories, "
                 f"{self._embeddings.shape[1]}-dim")

    def predict(self, story: dict) -> dict[str, list[str]]:
        """
        Returns {dim: [tag, ...]} for dims where neighbours agree strongly enough.
        These are meant to be merged into auto_tags alongside regex + semantic.
        """
        self._load()

        # Embed the query story (same 600-word sample as SemanticTagger)
        query_vec = self._model.encode(
            _short_sample(story), normalize_embeddings=True)  # (384,)

        # Cosine similarities — fast numpy dot (embeddings are L2-normalised)
        sims = self._embeddings @ query_vec  # (N,)

        # Top-k indices, sorted descending
        k = min(self.k, len(sims))
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        # Weighted vote — each neighbour's vote is weighted by its similarity
        votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_weight = 0.0

        for idx in top_idx:
            sim = float(sims[idx])
            if sim < self.sim_threshold:
                continue
            total_weight += sim
            neighbor_tags = self._tags[idx]
            for dim in KNN_DIMS:
                for tag in neighbor_tags.get(dim, []):
                    votes[dim][tag] += sim

        if total_weight == 0:
            return {}

        # Keep tags where weighted vote fraction exceeds vote_threshold
        results: dict[str, list[str]] = {}
        for dim, tag_votes in votes.items():
            accepted = [
                tag for tag, w in sorted(tag_votes.items(), key=lambda x: -x[1])
                if (w / total_weight) >= self.vote_threshold
            ]
            if accepted:
                results[dim] = accepted

        return results


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 2 — MEAN-POOLED FULL-TEXT EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def _three_position_samples(story: dict, sample_words: int = 250) -> list[str]:
    """
    Return 3 text samples covering start, middle, and end of the story.
    Each is `sample_words` words long (padded to full story if shorter).
    For short stories this naturally deduplicates into 1-2 unique samples.

    Why 3 positions instead of a sliding window:
      - 44 chunks/story × 18k stories = 815k encode calls → very slow
      - 3 samples/story × 18k stories =  55k encode calls → ~5 min on MPS
      - Start captures setup/premise; middle captures arc development;
        end captures resolution — the three most tag-relevant moments.
    """
    title = story.get("title", "")
    desc  = story.get("description", "") or ""
    text  = story.get("full_text", "") or ""
    words = text.split()
    n     = len(words)

    if n <= sample_words:
        # Short story: one sample covers everything
        return [f"{title}. {desc}. {text}"]

    # Head — first sample_words words (with title+desc for context)
    head = f"{title}. {desc}. " + " ".join(words[:sample_words])

    # Middle — centred on the midpoint
    mid_start = max(0, n // 2 - sample_words // 2)
    mid  = " ".join(words[mid_start : mid_start + sample_words])

    # Tail — last sample_words words
    tail = " ".join(words[max(0, n - sample_words):])

    return [head, mid, tail]


def build_fulltext_pool(use_mps: bool = False,
                        sample_words: int = 250,
                        encode_batch: int = 64) -> None:
    """
    Compute 3-position mean-pooled embeddings for every story and save to disk.

    Each story is represented by 3 samples (start, middle, end — each
    `sample_words` words), encoded together and mean-pooled into one vector.
    This covers the full narrative arc while keeping encode calls to
    3 × 18k = ~55k total (vs 44 × 18k = 815k for a sliding window).

    Estimated time: ~5-7 min on MPS (Apple Silicon), ~20 min on CPU.
    """
    stories = _load_stories()
    log.info(f"Stories to embed: {len(stories):,}")

    model = _load_model(use_mps=use_mps)

    # ── Step 1: build flat sample list with story boundaries
    all_samples: list[str]          = []
    story_ids:   list[str]          = []
    story_ranges: list[tuple[int, int]] = []   # (start, end) into all_samples

    log.info("Building 3-position samples...")
    for story in stories:
        start_idx = len(all_samples)
        samples   = _three_position_samples(story, sample_words=sample_words)
        all_samples.extend(samples)
        story_ranges.append((start_idx, len(all_samples)))
        story_ids.append(story.get("id", ""))

    total = len(all_samples)
    avg   = total / max(len(stories), 1)
    log.info(f"Total samples: {total:,}  (avg {avg:.1f} per story)")

    # ── Step 2: encode all samples in one batched pass
    log.info(f"Encoding {total:,} samples "
             f"(batch={encode_batch}, device={'mps' if use_mps else 'cpu'})...")
    embeddings = model.encode(
        all_samples,
        normalize_embeddings=False,
        batch_size=encode_batch,
        show_progress_bar=True,
    )  # (total_samples, 384)

    # ── Step 3: mean-pool per story and renormalise
    log.info("Mean-pooling per story...")
    all_vecs = []
    for start, end in story_ranges:
        chunk_vecs = embeddings[start:end]
        mean_vec   = chunk_vecs.mean(axis=0)
        mean_vec  /= (np.linalg.norm(mean_vec) + 1e-9)
        all_vecs.append(mean_vec.astype(np.float32))

    pool = np.stack(all_vecs, axis=0)  # (N, 384)
    np.save(POOL_EMB_PATH, pool)
    with open(POOL_ID_PATH, "w") as f:
        json.dump(story_ids, f)

    log.info(f"Saved full-text pool → {POOL_EMB_PATH}  ({pool.nbytes / 1e6:.1f} MB)")


class MeanPoolSemanticTagger:
    """
    Drop-in replacement for SemanticTagger._story_sample() that uses
    mean-pooled full-text embeddings instead of a 600-word prefix.

    Use this in SemanticTagger.tag_story() / tag_stories_batch() by
    replacing the _story_sample() call:

        # In tagger.py SemanticTagger.tag_story():
        # OLD: story_vec = self._model.encode(_story_sample(story), ...)
        # NEW: story_vec = mean_pool_tagger.encode_story(story)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_mps: bool = False):
        self.model_name = model_name
        self.use_mps    = use_mps
        self._model     = None

    def _load(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.use_mps)

    def encode_story(self, story: dict, sample_words: int = 250) -> np.ndarray:
        """Returns a normalised (384,) float32 vector covering the full story arc."""
        self._load()
        samples = _three_position_samples(story, sample_words=sample_words)
        vecs    = self._model.encode(samples, normalize_embeddings=False)
        mean    = vecs.mean(axis=0)
        mean   /= (np.linalg.norm(mean) + 1e-9)
        return mean.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO PLUG INTO tagger.py
# ══════════════════════════════════════════════════════════════════════════════
#
# In tagger.py  merge_tags(), add a knn_result parameter:
#
#   def merge_tags(regex_result, semantic_result=None, knn_result=None):
#       ...
#       for dim in all_sem_dims:
#           rx_tags  = regex_result.get(dim, [])
#           sem_tags = semantic_result.get(dim, []) if semantic_result else []
#           knn_tags = knn_result.get(dim, []) if knn_result else []
#           merged[dim] = _dedup(rx_tags
#                                + [t for t in sem_tags if t not in rx_tags]
#                                + [t for t in knn_tags if t not in rx_tags
#                                                       and t not in sem_tags])
#
# Then in tag_story() / process_index(), instantiate KNNTagger once and call:
#
#   knn = KNNTagger()
#   knn_result = knn.predict(story)
#   return merge_tags(rx, sem, knn_result)
#
# Source labeling: add "knn" as a valid source in _tag_sources so you can
# tell which layer fired each tag during debugging.


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Build kNN / full-text pool indexes")
    p.add_argument("--build",     action="store_true", help="Build both indexes")
    p.add_argument("--knn-only",  action="store_true", help="Build Svengali kNN index only")
    p.add_argument("--pool-only", action="store_true", help="Build full-text pool only")
    p.add_argument("--mps",       action="store_true", help="Use Apple Silicon GPU")
    p.add_argument("--demo",      action="store_true", help="Tag a random story and print results")
    args = p.parse_args()

    if args.build or args.knn_only:
        build_knn_index(use_mps=args.mps)

    if args.build or args.pool_only:
        build_fulltext_pool(use_mps=args.mps)

    if args.demo:
        import random
        stories = _load_stories()
        story   = random.choice([s for s in stories
                                 if (s.get("auto_tags") or {}).get("_svengali_tags")])
        print(f"\nStory: {story['title']} by {story['author']}")
        print(f"Length: {story.get('word_count',0):,} words")

        knn = KNNTagger()
        result = knn.predict(story)
        print("\n── k-NN votes ──")
        for dim, tags in result.items():
            if tags:
                print(f"  {dim:25} {tags}")

        print("\n── Svengali ground truth ──")
        at = story.get("auto_tags") or {}
        for dim in KNN_DIMS:
            tags = at.get(dim, [])
            if tags:
                print(f"  {dim:25} {tags}")
