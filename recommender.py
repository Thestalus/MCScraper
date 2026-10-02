#!/usr/bin/env python3
"""
recommender.py — similarity engine and preference learner for the MC Archive browser

Three capabilities:
  1. Story similarity    — find stories like a given one (embedding cosine similarity)
  2. Preference learning — like/dislike stories/authors, build a taste profile,
                           get personalised recommendations that improve over time
  3. Author profiling    — find authors with similar style/themes to ones you like

Uses sentence-transformers for embeddings (optional — falls back to tag-overlap
similarity if not installed).

Preferences are stored in output/preferences.json and persist across sessions.
"""

from __future__ import annotations

import json
import math
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)


def _at(s: dict) -> dict:
    """Return parsed auto_tags dict, parsing lazily from raw JSON string on first access."""
    at = s.get("auto_tags")
    if isinstance(at, str):
        at = json.loads(at) if at else {}
        s["auto_tags"] = at
    return at or {}


# ── Embedding engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Lazy-loading wrapper around sentence-transformers.
    Falls back to TF-IDF-style tag overlap if not installed.
    """

    def __init__(self):
        self._model      = None
        self._embeddings: dict[str, list[float]] = {}   # story_id -> vector
        self._loaded     = False
        self.semantic    = False

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model   = SentenceTransformer("all-MiniLM-L6-v2")
            self.semantic = True
            log.info("Recommendation engine: semantic mode (sentence-transformers)")
        except ImportError:
            self.semantic = False
            log.info("Recommendation engine: tag-overlap mode (install sentence-transformers for better results)")

    def story_signature(self, s: dict) -> str:
        """Build a text signature for a story used for embedding."""
        at = _at(s)
        parts = [
            s.get("title", ""),
            s.get("description", ""),
            " ".join(s.get("categories", [])),
            " ".join(at.get("themes", [])),
            " ".join(at.get("mind_control_type", [])),
            " ".join(at.get("setting", [])),
            " ".join(at.get("tone", [])),
            " ".join(at.get("dominance", [])),
            " ".join(at.get("pacing", [])),
            # Arc tags — populated after `make arc`, empty until then
            at.get("story_status", ""),
            at.get("ending_state", ""),
            at.get("subject_change", ""),
            at.get("emotional_tone_end", ""),
            at.get("dominant_type", ""),
        ]
        return " ".join(p for p in parts if p).lower()

    def build_index(self, stories: list[dict],
                    cache_path: "Path | None" = None):
        """Compute and cache embeddings for all valid stories.

        If cache_path is given, embeddings are loaded from disk and only
        new/missing stories are re-encoded, then the cache is updated.
        """
        if self._loaded:
            return
        self._try_load_model()
        valid = [s for s in stories if not s.get("fetch_error") and s.get("_has_text", s.get("full_text"))]
        log.info(f"Building recommendation index for {len(valid)} stories...")
        t0 = time.time()

        if self.semantic:
            import numpy as np
            # ── Load existing embedding cache ──────────────────────────────
            cached: dict[str, object] = {}
            if cache_path and Path(cache_path).exists():
                try:
                    data = np.load(cache_path, allow_pickle=False)
                    for sid, vec in zip(data["ids"].tolist(), data["vecs"]):
                        cached[str(sid)] = vec
                    log.info(f"  Loaded {len(cached)} cached embeddings from {cache_path}")
                except Exception as e:
                    log.warning(f"  Could not load embedding cache: {e}")

            # ── Encode only stories missing from cache ─────────────────────
            to_encode = [s for s in valid if s["id"] not in cached]
            if to_encode:
                log.info(f"  Encoding {len(to_encode)} stories "
                         f"({len(valid) - len(to_encode)} already cached)...")
                sigs = [self.story_signature(s) for s in to_encode]
                vecs = self._model.encode(sigs, normalize_embeddings=True,
                                          batch_size=64, show_progress_bar=True)
                for s, v in zip(to_encode, vecs):
                    cached[s["id"]] = v
            else:
                log.info(f"  All {len(valid)} embeddings loaded from cache — skipping encode")

            for s in valid:
                v = cached.get(s["id"])
                if v is not None:
                    self._embeddings[s["id"]] = v if isinstance(v, np.ndarray) else np.array(v, dtype=np.float32)

            # ── Save updated cache ─────────────────────────────────────────
            if cache_path and to_encode:
                try:
                    ids_arr  = np.array(list(self._embeddings.keys()))
                    vecs_arr = np.array([self._embeddings[sid] for sid in ids_arr],
                                        dtype=np.float32)
                    tmp = Path(cache_path).with_suffix(".tmp.npz")
                    np.savez(tmp, ids=ids_arr, vecs=vecs_arr)
                    tmp.replace(cache_path)
                    log.info(f"  Saved embedding cache ({len(ids_arr)} stories) → {cache_path}")
                except Exception as e:
                    log.warning(f"  Could not save embedding cache: {e}")
        else:
            # Tag-overlap fallback: represent each story as a set of tag tokens
            for s in valid:
                sig   = self.story_signature(s)
                tokens = sig.split()
                freq   = defaultdict(int)
                for t in tokens:
                    freq[t] += 1
                total = max(sum(freq.values()), 1)
                # Normalised term frequency vector stored as {token: weight}
                self._embeddings[s["id"]] = {t: c / total for t, c in freq.items()}

        self._loaded = True
        log.info(f"Index built in {time.time()-t0:.1f}s")

    def similarity(self, id_a: str, id_b: str) -> float:
        """Cosine similarity between two story embeddings."""
        va = self._embeddings.get(id_a)
        vb = self._embeddings.get(id_b)
        if va is None or vb is None:
            return 0.0
        if self.semantic:
            import numpy as np
            return float(np.dot(va, vb))
        else:
            # Dict dot product
            keys = set(va) | set(vb)
            dot  = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
            na   = math.sqrt(sum(v*v for v in va.values()))
            nb   = math.sqrt(sum(v*v for v in vb.values()))
            return dot / (na * nb + 1e-9)

    def vec(self, story_id: str) -> Optional[list]:
        return self._embeddings.get(story_id)

    def mean_vec(self, story_ids: list[str]):
        """Average embedding of a list of story IDs."""
        vecs = [self._embeddings[sid] for sid in story_ids
                if sid in self._embeddings]
        if not vecs:
            return None
        if self.semantic:
            import numpy as np
            arr = np.array(vecs)
            mean = arr.mean(axis=0)
            norm = np.linalg.norm(mean)
            return (mean / norm) if norm > 0 else mean
        else:
            # Merge dicts
            merged: dict[str, float] = defaultdict(float)
            for v in vecs:
                for k, val in v.items():
                    merged[k] += val / len(vecs)
            total = math.sqrt(sum(v*v for v in merged.values())) + 1e-9
            return {k: v/total for k, v in merged.items()}

    def score_against_vec(self, story_id: str, target_vec) -> float:
        """Score a story against an arbitrary target vector."""
        sv = self._embeddings.get(story_id)
        if sv is None or target_vec is None:
            return 0.0
        if self.semantic:
            import numpy as np
            return float(np.dot(sv, target_vec))
        else:
            keys = set(sv) | set(target_vec)
            dot  = sum(sv.get(k, 0) * target_vec.get(k, 0) for k in keys)
            na   = math.sqrt(sum(v*v for v in sv.values()))
            nb   = math.sqrt(sum(v*v for v in target_vec.values()))
            return dot / (na * nb + 1e-9)


# ── Preference store ──────────────────────────────────────────────────────────

class PreferenceStore:
    """
    Persists user likes/dislikes and derives a taste profile vector.

    Storage format (preferences.json):
    {
      "liked_stories":    ["story_id", ...],
      "disliked_stories": ["story_id", ...],
      "liked_authors":    ["Author Name", ...],
      "disliked_authors": ["Author Name", ...],
      "notes":            {"story_id": "optional personal note"}
    }
    """

    def __init__(self, prefs_path: Path):
        self.path = prefs_path
        self.data = {
            "liked_stories":    [],
            "disliked_stories": [],
            "liked_authors":    [],
            "disliked_authors": [],
            "notes":            {},
            "hidden_recs":      [],
            "history":          {},
            "want_to_read":     [],
        }
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                # Ensure all keys exist
                for k in ("liked_stories","disliked_stories",
                          "liked_authors","disliked_authors","notes","hidden_recs"):
                    self.data.setdefault(k, {} if k == "notes" else [])
                self.data.setdefault("history", {})
                self.data.setdefault("want_to_read", [])
            except Exception as e:
                log.warning(f"Could not load preferences: {e}")

    def _save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def hide_rec(self, story_id: str):
        if story_id not in self.data["hidden_recs"]:
            self.data["hidden_recs"].append(story_id)
        self._save()

    def unhide_rec(self, story_id: str):
        if story_id in self.data["hidden_recs"]:
            self.data["hidden_recs"].remove(story_id)
        self._save()

    def like_story(self, story_id: str, note: str = ""):
        if story_id not in self.data["liked_stories"]:
            self.data["liked_stories"].append(story_id)
        if story_id in self.data["disliked_stories"]:
            self.data["disliked_stories"].remove(story_id)
        if note:
            self.data["notes"][story_id] = note
        self._save()

    def dislike_story(self, story_id: str):
        if story_id not in self.data["disliked_stories"]:
            self.data["disliked_stories"].append(story_id)
        if story_id in self.data["liked_stories"]:
            self.data["liked_stories"].remove(story_id)
        self._save()

    def unlike_story(self, story_id: str):
        for lst in ("liked_stories", "disliked_stories"):
            if story_id in self.data[lst]:
                self.data[lst].remove(story_id)
        self._save()

    def like_author(self, author: str):
        if author not in self.data["liked_authors"]:
            self.data["liked_authors"].append(author)
        if author in self.data["disliked_authors"]:
            self.data["disliked_authors"].remove(author)
        self._save()

    def dislike_author(self, author: str):
        if author not in self.data["disliked_authors"]:
            self.data["disliked_authors"].append(author)
        if author in self.data["liked_authors"]:
            self.data["liked_authors"].remove(author)
        self._save()

    def unlike_author(self, author: str):
        for lst in ("liked_authors", "disliked_authors"):
            if author in self.data[lst]:
                self.data[lst].remove(author)
        self._save()

    def record_visit(self, story_id: str, chapter: int, total: int):
        import time
        self.data["history"][story_id] = {
            "last_chapter": chapter,
            "total":        total,
            "ts":           int(time.time()),
        }
        self._save()

    def visit_info(self, story_id: str):
        return self.data["history"].get(story_id)

    @property
    def visited_stories(self):
        h = self.data.get("history", {})
        return sorted(h.items(), key=lambda x: x[1]["ts"], reverse=True)

    def story_status(self, story_id: str) -> str:
        if story_id in self.data["liked_stories"]:    return "liked"
        if story_id in self.data["disliked_stories"]: return "disliked"
        return "none"

    def author_status(self, author: str) -> str:
        if author in self.data["liked_authors"]:    return "liked"
        if author in self.data["disliked_authors"]: return "disliked"
        return "none"

    def want_story(self, story_id: str):
        lst = self.data.setdefault("want_to_read", [])
        if story_id not in lst:
            lst.append(story_id)
            self._save()

    def unwant_story(self, story_id: str):
        lst = self.data.get("want_to_read", [])
        if story_id in lst:
            lst.remove(story_id)
            self._save()

    def is_wanted(self, story_id: str) -> bool:
        return story_id in self.data.get("want_to_read", [])

    @property
    def wanted_stories(self):
        return list(self.data.get("want_to_read", []))

    @property
    def liked_stories(self):    return self.data["liked_stories"]
    @property
    def disliked_stories(self): return self.data["disliked_stories"]
    @property
    def liked_authors(self):    return self.data["liked_authors"]
    @property
    def total_likes(self):      return len(self.data["liked_stories"]) + len(self.data["liked_authors"])


# ── Recommendation engine ────────────────────────────────────────────────────

class RecommendationEngine:

    def __init__(self, engine: EmbeddingEngine, prefs: PreferenceStore,
                 stories: list[dict]):
        self.engine  = engine
        self.prefs   = prefs
        self.stories = {s["id"]: s for s in stories
                        if not s.get("fetch_error") and s.get("_has_text", s.get("full_text"))}

    # ── Tag helpers ───────────────────────────────────────────────────────────

    _TAG_DIMS = ["themes", "mind_control_type", "setting", "tone",
                 "dominance", "pacing", "awareness"]

    # Gender/pairing categories (mf, ff, md, fd, etc.) get a mild boost —
    # half the weight of auto_tags so they nudge rather than dominate.
    _CATEGORY_WEIGHT = 0.5

    @staticmethod
    def _tag_confidence(at: dict, tag: str) -> float:
        """
        Return a confidence multiplier for a tag based on its source(s).
          regex only               → 1.0  (baseline)
          setfit/svengali only     → 1.5  (ML or ground-truth label)
          setfit/svengali + regex  → 2.0  (two independent layers agree)
        """
        sources = set(at.get("_tag_sources", {}).get(tag, []))
        high_conf = sources & {"setfit", "svengali"}
        if high_conf and "regex" in sources:
            return 2.0
        if high_conf:
            return 1.5
        return 1.0

    def _corpus_category_freq(self) -> dict[str, float]:
        """Fraction of corpus stories that have each category tag.
        Used to normalise liked-category rates so MF/MD don't get boosted
        just because they're the most common categories on the site."""
        from collections import Counter
        counts: Counter = Counter()
        n = len(self.stories)
        if n == 0:
            return {}
        for s in self.stories.values():
            for cat in s.get("categories", []):
                counts[cat] += 1
        return {cat: count / n for cat, count in counts.items()}

    def _build_tag_profile(self, liked_ids: list[str]) -> dict[str, dict[str, float]]:
        """
        Build a weighted tag profile from liked stories.
        Returns {dim: {tag: weight}} where weight is confidence-weighted frequency / n_liked.
        Tags confirmed by the Svengali-trained classifier count up to 2× more.

        Category weights are normalised by corpus frequency so that liking MF at
        the same rate it appears in the corpus gives no boost — only genuine
        over-representation in your likes (e.g. liking FF 3× more than baseline)
        produces a positive signal.
        """
        from collections import Counter
        counts: dict[str, Counter] = {dim: Counter() for dim in self._TAG_DIMS}
        valid = [s for lid in liked_ids
                 if (s := self.stories.get(lid)) is not None]
        if not valid:
            return {}
        for s in valid:
            at = _at(s)
            for dim in self._TAG_DIMS:
                for tag in at.get(dim, []):
                    counts[dim][tag] += self._tag_confidence(at, tag)
        # Categories — liked rate normalised by corpus rate, then half-weighted.
        # ratio = liked_freq / corpus_freq:
        #   ~1.0  → no preference vs average → small/no boost
        #   >1.0  → genuine preference → boost proportional to over-representation
        # Capped at 4× to prevent rare categories inflating from a few lucky likes.
        corpus_freq = self._corpus_category_freq()
        cat_counts: Counter = Counter()
        for s in valid:
            for cat in s.get("categories", []):
                cat_counts[cat] += 1
        n = len(valid)
        result = {dim: {tag: count / n for tag, count in counter.items()}
                  for dim, counter in counts.items() if counter}
        if cat_counts and corpus_freq:
            result["categories"] = {
                cat: min((count / n) / max(corpus_freq.get(cat, 1.0), 0.01), 4.0)
                     * self._CATEGORY_WEIGHT
                for cat, count in cat_counts.items()
            }
        return result

    def _tag_boost(self, story: dict, profile: dict[str, dict[str, float]]) -> float:
        """
        Score how well a story's tags match a tag profile.
        Returns 0-1: sum of matching tag weights, normalised by total profile weight.
        Tags confirmed by the Svengali-trained classifier contribute up to 2× more.
        """
        if not profile:
            return 0.0
        at = _at(story)
        total_weight = sum(sum(w for w in tags.values()) for tags in profile.values())
        if total_weight == 0:
            return 0.0
        score = 0.0
        for dim, tag_weights in profile.items():
            if dim == "categories":
                story_tags = set(story.get("categories", []))
                for tag, weight in tag_weights.items():
                    if tag in story_tags:
                        score += weight * self._CATEGORY_WEIGHT
            else:
                story_tags = set(at.get(dim, []))
                for tag, weight in tag_weights.items():
                    if tag in story_tags:
                        score += weight * self._tag_confidence(at, tag)
        return min(score / total_weight, 1.0)

    def _story_tag_overlap(self, story: dict, target_tags: dict[str, set]) -> float:
        """Tag overlap between a story and a target story's tags (Jaccard-style)."""
        at = _at(story)
        matches = total = 0
        for dim, t_tags in target_tags.items():
            s_tags = set(at.get(dim, []))
            if t_tags or s_tags:
                matches += len(s_tags & t_tags)
                total   += len(s_tags | t_tags)
        # Categories mild boost (half weight)
        s_cats = set(story.get("categories", []))
        t_cats = target_tags.get("_categories", set())
        if s_cats or t_cats:
            matches += self._CATEGORY_WEIGHT * len(s_cats & t_cats)
            total   += self._CATEGORY_WEIGHT * len(s_cats | t_cats)
        return matches / total if total > 0 else 0.0

    # ── Core methods ──────────────────────────────────────────────────────────

    def similar_to_story(self, story_id: str, n: int = 12,
                         exclude_same_author: bool = False) -> list[tuple[dict, float]]:
        """Find n stories most similar to the given one.
        Blends semantic similarity (80%) with tag overlap (20%)."""
        target = self.stories.get(story_id)
        if not target:
            return []
        target_author = target.get("author", "")
        target_tags = {dim: set(_at(target).get(dim, []))
                       for dim in self._TAG_DIMS}
        target_tags["_categories"] = set(target.get("categories", []))
        scores = []
        for sid, s in self.stories.items():
            if sid == story_id:
                continue
            if exclude_same_author and s.get("author") == target_author:
                continue
            if sid in self.prefs.disliked_stories:
                continue
            sem_score = self.engine.similarity(story_id, sid)
            tag_score = self._story_tag_overlap(s, target_tags)
            score = 0.8 * sem_score + 0.2 * tag_score
            scores.append((s, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:n]

    def personalised_recommendations(self, n: int = 20,
                                     exclude_read: bool = True,
                                     shuffle: bool = False) -> list[tuple[dict, float, str]]:
        """
        Recommend stories based on taste profile.
        Returns list of (story, score, reason) tuples.
        """
        if self.prefs.total_likes == 0:
            # No preferences yet — return a varied selection of popular/well-tagged stories
            candidates = [s for s in self.stories.values()
                          if len(_at(s).get("themes", [])) >= 2
                          and s.get("word_count", 0) > 2000]
            import random
            random.shuffle(candidates)
            return [(s, 0.5, "well-tagged story") for s in candidates[:n]]

        # Build positive taste vector from liked stories + liked author stories
        liked_ids = list(self.prefs.liked_stories)
        for author in self.prefs.liked_authors:
            for s in self.stories.values():
                if s.get("author") == author and s["id"] not in liked_ids:
                    liked_ids.append(s["id"])

        taste_vec = self.engine.mean_vec(liked_ids)
        if taste_vec is None:
            return []

        # Build negative vector from dislikes
        dislike_ids  = list(self.prefs.disliked_stories)
        dislike_authors = self.prefs.data.get("disliked_authors", [])
        for author in dislike_authors:
            for s in self.stories.values():
                if s.get("author") == author:
                    dislike_ids.append(s["id"])

        dislike_vec = self.engine.mean_vec(dislike_ids) if dislike_ids else None

        # Build tag profile from liked stories for tag boost
        tag_profile = self._build_tag_profile(liked_ids)

        # Score all candidates (defer _explain to top-N only)
        disliked_authors = set(self.prefs.data.get("disliked_authors") or [])
        hidden_recs      = set(self.prefs.data.get("hidden_recs") or [])
        raw = []
        for sid, s in self.stories.items():
            if exclude_read and sid in self.prefs.liked_stories:
                continue
            if sid in self.prefs.disliked_stories:
                continue
            if sid in hidden_recs:
                continue
            if s.get("author") in disliked_authors:
                continue

            pos_score = self.engine.score_against_vec(sid, taste_vec)
            neg_score = self.engine.score_against_vec(sid, dislike_vec) if dislike_vec else 0
            tag_score = self._tag_boost(s, tag_profile)
            # 70% semantic similarity + 30% tag match, penalise dislike-similar
            final = 0.7 * pos_score + 0.3 * tag_score - 0.3 * neg_score
            raw.append((s, final))

        raw.sort(key=lambda x: -x[1])

        # Diversify: cap how many recs any single liked story can anchor.
        # Work on a 5× pool so there are enough candidates after capping.
        pool = raw[:min(len(raw), n * 5)]
        if shuffle:
            import random
            random.shuffle(pool)

        # Use only explicit likes (not liked-author additions) as anchors so
        # the cap reflects the user's actual stated preferences.
        anchor_ids = list(self.prefs.liked_stories)[:20]
        # Allow each anchor to contribute at most this many recs.
        # Minimum of 3 so a small like list still gets enough results.
        max_per_anchor = max(3, n // max(len(anchor_ids), 1))

        per_anchor: dict[str, int] = {}
        top: list = []
        overflow: list = []

        for s, score in pool:
            # Find the closest liked story for this candidate
            best_anchor = None
            best_sim = -1.0
            for lid in anchor_ids:
                sc = self.engine.similarity(s["id"], lid)
                if sc > best_sim:
                    best_sim = sc
                    best_anchor = lid

            count = per_anchor.get(best_anchor, 0)
            if best_anchor is None or count < max_per_anchor:
                top.append((s, score))
                if best_anchor:
                    per_anchor[best_anchor] = count + 1
            else:
                overflow.append((s, score))

            if len(top) >= n:
                break

        # Fill any remaining slots from overflow (keeps total at n)
        if len(top) < n:
            top.extend(overflow[:n - len(top)])

        return [(s, score, self._explain(s, liked_ids, tag_profile)) for s, score in top]

    def _explain(self, story: dict, liked_ids: list[str],
                 tag_profile: dict | None = None) -> str:
        """Generate a human-readable reason for a recommendation."""
        at = _at(story)

        # Find the most similar liked story (check up to 20 liked stories)
        best_score = 0.0
        best_title = ""
        for lid in liked_ids[:20]:
            sc = self.engine.similarity(story["id"], lid)
            if sc > best_score:
                best_score = sc
                liked = self.stories.get(lid)
                if liked:
                    best_title = liked.get("title") or lid

        parts = []
        if best_title and best_score > 0.3:
            pct = int(round(best_score * 100))
            parts.append(f"{pct}% match with \"{best_title}\"")

        # Show matched profile tags grouped by dimension for clarity
        if tag_profile:
            _DIM_SHORT = {
                "themes":           "themes",
                "mind_control_type":"mc",
                "tone":             "tone",
                "setting":          "setting",
                "dominance":        "dominance",
                "pacing":           "pacing",
            }
            dim_matches: dict[str, list[str]] = {}
            for dim, short in _DIM_SHORT.items():
                profile_tags = tag_profile.get(dim, {})
                hits = [tag for tag in at.get(dim, [])
                        if tag in profile_tags and profile_tags[tag] >= 0.15]
                hits.sort(key=lambda t: -profile_tags[t])
                if hits:
                    dim_matches[short] = hits[:2]
            if dim_matches:
                tag_parts = [f"{short}: {', '.join(tags)}"
                             for short, tags in list(dim_matches.items())[:3]]
                parts.append(" · ".join(tag_parts))
        elif at.get("themes"):
            parts.append("themes: " + ", ".join(at["themes"][:2]))

        return "  ·  ".join(parts) if parts else "matches your taste profile"

    def similar_authors(self, author_name: str, n: int = 8) -> list[tuple[str, float, dict]]:
        """
        Find authors with similar style/themes.
        Returns list of (author_name, similarity_score, profile_summary).
        """
        # Build author vectors as mean of their stories
        by_author: dict[str, list[str]] = defaultdict(list)
        for sid, s in self.stories.items():
            a = s.get("author")
            if a:
                by_author[a].append(sid)

        if author_name not in by_author:
            return []

        target_vec = self.engine.mean_vec(by_author[author_name])
        if target_vec is None:
            return []

        results = []
        for other_author, sids in by_author.items():
            if other_author == author_name:
                continue
            if len(sids) < 2:   # skip one-story authors
                continue
            other_vec = self.engine.mean_vec(sids)
            if other_vec is None:
                continue

            if self.engine.semantic:
                import numpy as np
                score = float(np.dot(target_vec, other_vec))
            else:
                keys  = set(target_vec) | set(other_vec)
                dot   = sum(target_vec.get(k,0) * other_vec.get(k,0) for k in keys)
                na    = math.sqrt(sum(v*v for v in target_vec.values()))
                nb    = math.sqrt(sum(v*v for v in other_vec.values()))
                score = dot / (na * nb + 1e-9)

            # Build profile summary
            their_stories = [self.stories[sid] for sid in sids]
            all_themes    = []
            all_mc        = []
            for s in their_stories:
                at = _at(s)
                all_themes.extend(at.get("themes", []))
                all_mc.extend(at.get("mind_control_type", []))

            theme_counts = defaultdict(int)
            mc_counts    = defaultdict(int)
            for t in all_themes: theme_counts[t] += 1
            for m in all_mc:     mc_counts[m]    += 1

            profile = {
                "story_count":    len(sids),
                "top_themes":     sorted(theme_counts, key=lambda k: -theme_counts[k])[:3],
                "top_mc_types":   sorted(mc_counts,    key=lambda k: -mc_counts[k])[:2],
                "liked":          author_name in self.prefs.liked_authors,
            }
            results.append((other_author, score, profile))

        results.sort(key=lambda x: -x[1])
        return results[:n]

    def save_cache(self, cache_path: Path, index_mtime: float = 0.0,
                   prefs_mtime: float = 0.0, n_similar: int = 24,
                   n_recs: int = 50) -> None:
        """
        Pre-calculate all recommendations and save to a cache file.
        Fast path uses a single matrix multiply for semantic mode (~seconds for 17k stories).
        """
        from datetime import datetime, timezone

        story_ids = list(self.stories.keys())
        n = len(story_ids)
        log.info(f"Pre-calculating similar stories for {n} stories...")

        disliked = set(self.prefs.disliked_stories)
        similar: dict[str, list] = {}

        import numpy as np

        if self.engine.semantic:
            ids_with_vecs = [sid for sid in story_ids if self.engine.vec(sid) is not None]
            mat = np.array([self.engine.vec(sid) for sid in ids_with_vecs], dtype=np.float32)
        else:
            # Build a dense matrix from tag-overlap dict vectors for fast matrix multiply.
            # Vocabulary is small (~200-400 unique tokens) so this fits easily in RAM.
            ids_with_vecs = [sid for sid in story_ids if self.engine.vec(sid) is not None]
            vocab = {}
            for sid in ids_with_vecs:
                for tok in self.engine.vec(sid):
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
            V = len(vocab)
            mat = np.zeros((len(ids_with_vecs), V), dtype=np.float32)
            for i, sid in enumerate(ids_with_vecs):
                for tok, w in self.engine.vec(sid).items():
                    mat[i, vocab[tok]] = w
            # L2-normalise rows so dot product == cosine similarity
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat /= norms

        log.info(f"  Matrix multiply ({len(ids_with_vecs)} × {mat.shape[1]})...")
        sims = mat @ mat.T  # full similarity matrix in one shot
        for i, sid in enumerate(ids_with_vecs):
            row = sims[i].copy()
            row[i] = -1.0   # exclude self
            top_idx = np.argpartition(row, -min(n_similar, len(row) - 1))[-n_similar:]
            top = sorted(
                [(ids_with_vecs[j], float(row[j])) for j in top_idx
                 if ids_with_vecs[j] not in disliked],
                key=lambda x: -x[1]
            )[:n_similar]
            similar[sid] = [[s_id, round(sc, 4)] for s_id, sc in top]
            if (i + 1) % 2000 == 0:
                log.info(f"  [{i+1}/{len(ids_with_vecs)}]")

        log.info("Computing personalised recommendations...")
        pers = self.personalised_recommendations(n=n_recs)
        personalised = [[r[0]["id"], round(r[1], 4), r[2]] for r in pers]

        cache = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "index_mtime":  index_mtime,
            "prefs_mtime":  prefs_mtime,
            "story_count":  len(similar),
            "similar":      similar,
            "personalised": personalised,
        }
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        tmp.replace(cache_path)
        log.info(f"Saved → {cache_path}  ({len(similar)} stories, {len(personalised)} recs)")

    def taste_profile_summary(self) -> dict:
        """Summarise the user's taste based on liked stories."""
        liked = [self.stories[sid] for sid in self.prefs.liked_stories
                 if sid in self.stories]
        if not liked:
            return {}

        theme_counts   = defaultdict(int)
        mc_counts      = defaultdict(int)
        setting_counts = defaultdict(int)
        tone_counts    = defaultdict(int)
        dom_counts     = defaultdict(int)
        pac_counts     = defaultdict(int)
        cat_counts     = defaultdict(int)
        word_counts    = []

        for s in liked:
            at = _at(s)
            for t in at.get("themes", []):          theme_counts[t]   += 1
            for m in at.get("mind_control_type", []): mc_counts[m]     += 1
            for x in at.get("setting", []):          setting_counts[x] += 1
            for x in at.get("tone", []):             tone_counts[x]    += 1
            for x in at.get("dominance", []):        dom_counts[x]     += 1
            for x in at.get("pacing", []):           pac_counts[x]     += 1
            for c in s.get("categories", []):        cat_counts[c]     += 1
            if s.get("word_count"):
                word_counts.append(s["word_count"])

        def top(d, n=3):
            return sorted(d.items(), key=lambda x: -x[1])[:n]

        return {
            "liked_count":       len(liked),
            "top_themes":        top(theme_counts),
            "top_mc_types":      top(mc_counts),
            "top_settings":      top(setting_counts),
            "top_tones":         top(tone_counts),
            "top_dominance":     top(dom_counts),
            "top_pacing":        top(pac_counts),
            "top_categories":    top(cat_counts, 5),
            "avg_length":        int(sum(word_counts) / len(word_counts)) if word_counts else 0,
            "liked_authors":     self.prefs.liked_authors,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Pre-calculate recommendation cache")
    p.add_argument("--precalc",    action="store_true",
                   help="Build and save recommendations cache")
    p.add_argument("--index",      default="./output/stories_index.json")
    p.add_argument("--prefs",      default="./output/preferences.json")
    p.add_argument("--cache",      default="./output/recs_cache.json")
    p.add_argument("--n-similar",  type=int, default=24,
                   help="Similar stories to store per story (default: 24)")
    p.add_argument("--n-recs",     type=int, default=200,
                   help="Personalised recs to store (default: 200)")
    args = p.parse_args()

    if args.precalc:
        index_path = Path(args.index)
        prefs_path = Path(args.prefs)
        cache_path = Path(args.cache)

        log.info(f"Loading index from {index_path}...")
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)
        stories = data.get("stories", [])
        log.info(f"  {len(stories)} stories loaded")

        engine = EmbeddingEngine()
        prefs  = PreferenceStore(prefs_path)
        rec    = RecommendationEngine(engine, prefs, stories)

        index_mtime = index_path.stat().st_mtime
        prefs_mtime = prefs_path.stat().st_mtime if prefs_path.exists() else 0.0

        emb_cache = index_path.parent / "recs_embeddings.npz"
        engine.build_index(stories, cache_path=emb_cache)
        rec.save_cache(cache_path,
                       index_mtime=index_mtime,
                       prefs_mtime=prefs_mtime,
                       n_similar=args.n_similar,
                       n_recs=args.n_recs)
    else:
        p.print_help()
