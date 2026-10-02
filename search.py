#!/usr/bin/env python3
"""
search.py — Pure helper functions for search ranking, NLQ parsing, and tag stats.
No Flask, no globals, no database connections. Imported by browser.py.
"""

from __future__ import annotations
import re
from services.data_store import get_auto_tags

# ── Natural language query parsing ────────────────────────────────────────────
#
# Maps lowercase phrases (and aliases) → (dimension, canonical_value).
# Sorted longest-first so greedy matching works correctly.
#
_NLQ_MAP: list[tuple[str, str, str]] = [
    # MC types
    ("reality manipulation", "mc_type",  "reality manipulation"),
    ("virtual reality",      "mc_type",  "virtual reality"),
    ("nanotechnology",       "mc_type",  "nanotechnology"),
    ("time stop",            "mc_type",  "time stop"),
    ("subliminal",           "mc_type",  "subliminal"),
    ("pheromones",           "mc_type",  "pheromones"),
    ("pheromone",            "mc_type",  "pheromones"),
    ("nanotech",             "mc_type",  "nanotechnology"),
    ("chemical",             "mc_type",  "chemical"),
    ("hypnosis",             "mc_type",  "hypnosis"),
    ("hypnotised",           "mc_type",  "hypnosis"),
    ("hypnotized",           "mc_type",  "hypnosis"),
    ("hypnotic",             "mc_type",  "hypnosis"),
    ("psionic",              "mc_type",  "psionic"),
    ("technology",           "mc_type",  "technology"),
    ("virus",                "mc_type",  "virus"),
    ("magic",                "mc_type",  "magic"),

    # Settings (longer phrases first)
    ("science fiction",      "setting",  "science fiction"),
    ("fantasy / historical", "setting",  "fantasy / historical"),
    ("bar / club / party",   "setting",  "bar / club / party"),
    ("school / college",     "setting",  "school / college"),
    ("online / virtual",     "setting",  "online / virtual"),
    ("home / domestic",      "setting",  "home / domestic"),
    ("gym / sports",         "setting",  "gym / sports"),
    ("workplace",            "setting",  "workplace"),
    ("university",           "setting",  "school / college"),
    ("college",              "setting",  "school / college"),
    ("campus",               "setting",  "school / college"),
    ("school",               "setting",  "school / college"),
    ("hospital",             "setting",  "medical"),
    ("medical",              "setting",  "medical"),
    ("domestic",             "setting",  "home / domestic"),
    ("historical",           "setting",  "fantasy / historical"),
    ("fantasy",              "setting",  "fantasy / historical"),
    ("supernatural",         "setting",  "supernatural"),
    ("office",               "setting",  "workplace"),
    ("at work",              "setting",  "workplace"),
    ("sports",               "setting",  "gym / sports"),
    ("sci-fi",               "setting",  "science fiction"),
    ("scifi",                "setting",  "science fiction"),
    ("online",               "setting",  "online / virtual"),
    ("virtual",              "setting",  "online / virtual"),
    ("party",                "setting",  "bar / club / party"),
    ("club",                 "setting",  "bar / club / party"),
    ("gym",                  "setting",  "gym / sports"),
    ("bar",                  "setting",  "bar / club / party"),
    ("home",                 "setting",  "home / domestic"),

    # Themes (longer phrases first)
    ("serial recruitment",   "theme",    "serial recruitment"),
    ("corruption arc",       "theme",    "corruption arc"),
    ("stepfordization",      "theme",    "stepfordization"),
    ("slave mentality",      "theme",    "slave mentality"),
    ("bimboization",         "theme",    "bimboization"),
    ("mythology / religion", "theme",    "mythology / religion"),
    ("reluctant subject",    "theme",    "reluctant subject"),
    ("willing subject",      "theme",    "willing subject"),
    ("drone / hive mind",    "theme",    "drone / hive mind"),
    ("consent / ethics",     "theme",    "consent / ethics"),
    ("horror / dark",        "theme",    "horror / dark"),
    ("identity loss",        "theme",    "identity loss"),
    ("induced love",         "theme",    "induced love"),
    ("power corrupts",       "theme",    "power corrupts"),
    ("trigger phrase",       "theme",    "trigger phrase"),
    ("found family",         "theme",    "found family"),
    ("slow burn",            "theme",    "slow burn"),
    ("time travel",          "theme",    "time travel"),
    ("hive mind",            "theme",    "drone / hive mind"),
    ("trigger word",         "theme",    "trigger phrase"),
    ("mythology",            "theme",    "mythology / religion"),
    ("conspiracy",           "theme",    "conspiracy"),
    ("reluctant",            "theme",    "reluctant subject"),
    ("betrayal",             "theme",    "betrayal"),
    ("obsession",            "theme",    "obsession"),
    ("celebrity",            "theme",    "celebrity"),
    ("tentacles",            "theme",    "tentacles"),
    ("religion",             "theme",    "mythology / religion"),
    ("revenge",              "theme",    "revenge"),
    ("romance",              "theme",    "romance"),
    ("willing",              "theme",    "willing subject"),
    ("consent",              "theme",    "consent / ethics"),
    ("harem",                "theme",    "harem"),
    ("rescue",               "theme",    "rescue"),
    ("horror",               "theme",    "horror / dark"),
    ("drone",                "theme",    "drone / hive mind"),
    ("bimbo",                "theme",    "bimboization"),

    # Tones (longer phrases first)
    ("dark / disturbing",    "tone",     "dark / disturbing"),
    ("romantic / sweet",     "tone",     "romantic / sweet"),
    ("humorous / playful",   "tone",     "humorous / playful"),
    ("erotic / explicit",    "tone",     "erotic / explicit"),
    ("tense / suspenseful",  "tone",     "tense / suspenseful"),
    ("psychological",        "tone",     "psychological"),
    ("melancholic",          "tone",     "melancholic"),
    ("disturbing",           "tone",     "dark / disturbing"),
    ("suspenseful",          "tone",     "tense / suspenseful"),
    ("humorous",             "tone",     "humorous / playful"),
    ("explicit",             "tone",     "erotic / explicit"),
    ("thriller",             "tone",     "tense / suspenseful"),
    ("playful",              "tone",     "humorous / playful"),
    ("romantic",             "tone",     "romantic / sweet"),
    ("erotic",               "tone",     "erotic / explicit"),
    ("humour",               "tone",     "humorous / playful"),
    ("humor",                "tone",     "humorous / playful"),
    ("tense",                "tone",     "tense / suspenseful"),
    ("funny",                "tone",     "humorous / playful"),
    ("sweet",                "tone",     "romantic / sweet"),
    ("dark",                 "tone",     "dark / disturbing"),
    ("sad",                  "tone",     "melancholic"),

    # Length
    ("flash fiction",        "length",   "flash"),
    ("short story",          "length",   "short story"),
    ("novelette",            "length",   "novelette"),
    ("novella",              "length",   "novella"),
    ("novel",                "length",   "novel"),
    ("flash",                "length",   "flash"),
    ("long",                 "length",   "novel"),
    ("short",                "length",   "short story"),

    # Pacing
    ("single scene",         "pacing",   "single scene"),
    ("one shot",             "pacing",   "single scene"),
    ("episodic",             "pacing",   "episodic"),
    ("leisurely",            "pacing",   "leisurely"),
    ("oneshot",              "pacing",   "single scene"),
    ("rapid",                "pacing",   "rapid"),
    ("epic",                 "pacing",   "epic"),

    # Dominance
    ("loving control",       "dominance","loving control"),
    ("cold / clinical",      "dominance","cold / clinical"),
    ("self-inflicted",       "dominance","self-inflicted"),
    ("manipulative",         "dominance","manipulative"),
    ("sadistic",             "dominance","sadistic"),
    ("clinical",             "dominance","cold / clinical"),
    ("gentle",               "dominance","gentle"),
    ("brutal",               "dominance","brutal"),
    ("cold",                 "dominance","cold / clinical"),

    # Awareness
    ("valiant but futile struggle", "awareness", "valiant but futile struggle"),
    ("gradually losing awareness",  "awareness", "gradually losing awareness"),
    ("futile struggle",      "awareness","valiant but futile struggle"),
    ("oblivious",            "awareness","oblivious"),
    ("accepting",            "awareness","accepting"),
    ("gradually",            "awareness","gradually losing awareness"),
    ("struggling",           "awareness","valiant but futile struggle"),
]

# Sort longest phrase first for greedy left-to-right matching
_NLQ_MAP.sort(key=lambda x: -len(x[0]))

# Human-readable dimension labels for the banner
_DIM_LABELS = {
    "mc_type":   "MC type",
    "setting":   "Setting",
    "theme":     "Theme",
    "tone":      "Tone",
    "length":    "Length",
    "pacing":    "Pacing",
    "dominance": "Dominance",
    "awareness": "Awareness",
}


def parse_natural_query(q: str) -> tuple[str, dict]:
    """
    Scan a free-text query for tag-like phrases and extract them as filter hints.

    Returns:
        (cleaned_query, detected)
        - cleaned_query : original query with detected phrases removed
        - detected      : {dimension: value_or_list}
            "theme" key always holds a list; all other dimensions hold a string.
            Dimension names match browser.py URL parameter names.
    """
    if not q:
        return q, {}

    text = q.lower()
    detected: dict[str, str | list[str]] = {}
    consumed: set[int] = set()

    for phrase, dim, value in _NLQ_MAP:
        start = 0
        while True:
            pos = text.find(phrase, start)
            if pos == -1:
                break
            end = pos + len(phrase)
            # Require word boundaries (don't match inside a longer word)
            if pos > 0 and text[pos - 1].isalpha():
                start = pos + 1
                continue
            if end < len(text) and text[end].isalpha():
                start = pos + 1
                continue
            # Skip already-matched positions
            if any(i in consumed for i in range(pos, end)):
                start = pos + 1
                continue
            # Accept
            for i in range(pos, end):
                consumed.add(i)
            if dim == "theme":
                lst = detected.setdefault("theme", [])
                if value not in lst:
                    lst.append(value)
            else:
                detected.setdefault(dim, value)  # first match wins per scalar dim
            start = pos + 1

    if not detected:
        return q, {}

    # Remove consumed characters; collapse extra whitespace
    clean = "".join(c for i, c in enumerate(q) if i not in consumed)
    clean = re.sub(r"\s{2,}", " ", clean).strip(" ,;-")
    return clean, detected


def nlq_apply_url(clean_query: str, detected: dict, base_url: str = "/") -> str:
    """
    Build a URL that applies detected NLQ filters as explicit query params.
    clean_query is the query with tag phrases stripped.
    """
    parts = []
    if clean_query:
        parts.append(f"q={_urlencode(clean_query)}")
    for dim, val in detected.items():
        if dim == "theme":
            for v in val:
                parts.append(f"theme={_urlencode(v)}")
        elif dim == "length":
            parts.append(f"length={_urlencode(val)}")
        else:
            parts.append(f"{dim}={_urlencode(val)}")
    qs = "&".join(parts)
    return f"{base_url}?{qs}" if qs else base_url


def _urlencode(s: str) -> str:
    """Minimal percent-encoding for URL query values."""
    from urllib.parse import quote_plus
    return quote_plus(s)


# ── Preference-based boost scoring ────────────────────────────────────────────

def pref_boost_score(story: dict, tag_profile: dict) -> float:
    """
    Score a story's alignment with a user tag profile (0.0–1.0).

    tag_profile: {dimension: {value: weight}} as returned by
    RecommendationEngine._build_tag_profile().
    """
    if not tag_profile:
        return 0.0
    at = get_auto_tags(story)
    total_weight = 0.0
    matched_weight = 0.0
    for dim, weights in tag_profile.items():
        if not weights:
            continue
        max_w = max(weights.values())
        total_weight += max_w
        story_vals = [v.lower() for v in (at.get(dim) or [])]
        for val, w in weights.items():
            if val.lower() in story_vals:
                matched_weight += w
                break  # one match per dimension
    if total_weight == 0:
        return 0.0
    return min(1.0, matched_weight / total_weight)


# ── Tag coverage statistics ───────────────────────────────────────────────────

_ALL_DIMS = [
    "mind_control_type", "setting", "themes", "tone",
    "dominance", "pacing", "awareness", "sexual_acts",
    "fetish", "relationship",
]

_DIM_DISPLAY = {
    "mind_control_type": "MC Type",
    "setting":           "Setting",
    "themes":            "Themes",
    "tone":              "Tone",
    "dominance":         "Dominance",
    "pacing":            "Pacing",
    "awareness":         "Awareness",
    "sexual_acts":       "Sexual Acts",
    "fetish":            "Fetish",
    "relationship":      "Relationship",
}


def tag_coverage_stats(stories: list[dict]) -> dict:
    """
    Compute per-dimension tag coverage statistics.

    Returns:
        {
            "total":      int,
            "dimensions": [
                {
                    "name":      str,   # dimension key
                    "label":     str,   # human-readable label
                    "covered":   int,   # stories with ≥1 tag in this dim
                    "pct":       float, # 0–100
                    "top":       [(value, count), ...],  # top 15 by count
                    "tag_count": int,   # unique tag values
                }
            ]
        }
    """
    valid = [s for s in stories
             if not s.get("fetch_error") and s.get("_has_text", s.get("full_text"))]
    total = len(valid)
    if total == 0:
        return {"total": 0, "dimensions": []}

    dims_out = []
    for dim in _ALL_DIMS:
        counts: dict[str, int] = {}
        covered = 0
        for s in valid:
            at = get_auto_tags(s)
            vals = at.get(dim) or []
            if vals:
                covered += 1
                for v in vals:
                    counts[v] = counts.get(v, 0) + 1
        top = sorted(counts.items(), key=lambda x: -x[1])
        dims_out.append({
            "name":      dim,
            "label":     _DIM_DISPLAY.get(dim, dim),
            "covered":   covered,
            "pct":       round(100.0 * covered / total, 1) if total else 0.0,
            "top":       top[:15],
            "tag_count": len(counts),
        })

    dims_out.sort(key=lambda d: -d["covered"])
    return {"total": total, "dimensions": dims_out}
