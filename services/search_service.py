"""
services/search_service.py — Story search, filtering, and ranking.
"""
from __future__ import annotations
import re
import sqlite3

from search import pref_boost_score
from services.data_store import valid_stories, get_auto_tags


# ── Display title helper ───────────────────────────────────────────────────────

def slug_to_title(slug: str) -> str:
    """Convert CamelCase/PascalCase slug to spaced title."""
    if not slug:
        return slug
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', slug)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced


def dt(story) -> str:
    """
    Display title for a story dict or string.
    Handles all cases:
      - story has a real title with spaces → use it
      - story title looks like a slug (no spaces) → run slug_to_title on it
      - story has no title → derive from id
    Works with both dicts and strings.
    """
    if isinstance(story, str):
        if not story:
            return ""
        if " " in story or len(story) <= 4:
            return story
        return slug_to_title(story)
    # Dict
    title = story.get("title") or ""
    sid   = story.get("id") or ""
    if not title:
        return slug_to_title(sid) if sid else ""
    if " " in title or len(title) <= 4:
        return title
    return slug_to_title(title)


# ── Scoring ────────────────────────────────────────────────────────────────────

def relevance_score(s: dict, q: str) -> float:
    """
    Score a story's relevance to a search query.
    Higher = more relevant. Used to rank results when sort=relevance.
    """
    if not q:
        return 0.0
    score    = 0.0
    q_lower  = q.lower()
    title    = (s.get("title") or "").lower()
    title_spaced = dt(s).lower()
    author   = (s.get("author") or "").lower()
    desc     = (s.get("description") or "").lower()
    cats     = " ".join(s.get("categories", [])).lower()
    at       = get_auto_tags(s)
    themes   = " ".join(at.get("themes", [])).lower()
    mc       = " ".join(at.get("mind_control_type", [])).lower()

    words = q_lower.split()

    def word_hits(field):
        return sum(1 for w in words if w in field)

    all_words = len(words)

    if q_lower == title or q_lower == title_spaced:
        score += 100
    elif title.startswith(q_lower) or title_spaced.startswith(q_lower):
        score += 60
    elif q_lower in title or q_lower in title_spaced:
        score += 40
    else:
        hits = word_hits(title + " " + title_spaced)
        if hits == all_words: score += 30
        elif hits > 0:        score += hits * 8

    if q_lower == author:       score += 50
    elif q_lower in author:     score += 25
    elif word_hits(author) > 0: score += 10

    if q_lower in cats:            score += 20
    elif word_hits(cats) == all_words: score += 12

    if q_lower in themes:          score += 15
    elif word_hits(themes) > 0:    score += 8

    if q_lower in mc:              score += 15
    elif word_hits(mc) > 0:        score += 8

    if q_lower in desc:
        score += 10
    else:
        hits = word_hits(desc)
        if hits == all_words: score += 8
        elif hits > 0:        score += hits * 2

    return score


def _match_multi(selected, story_vals, logic="any"):
    """Check if story_vals satisfies the selected list under 'any' (OR) or 'all' (AND) logic."""
    if logic == "all":
        return all(any(sel.lower() in sv for sv in story_vals) for sel in selected)
    return any(any(sel.lower() in sv for sv in story_vals) for sel in selected)


def filter_score(s: dict, categories=None, mc_type="", setting="",
                 themes=None, theme_logic="all", tone="", dominance="", pacing="",
                 awareness="", length_class="",
                 sexual_acts=None, sexual_act_logic="any",
                 fetish_tags=None, fetish_tag_logic="any",
                 relationship="") -> float:
    """Return a score (0–1) for how well a story matches the given soft filters."""
    at = get_auto_tags(s)
    checks = []
    if categories:
        story_cats = [c.lower() for c in s.get("categories", [])]
        checks.append(all(c.lower() in story_cats for c in categories))
    if mc_type:
        checks.append(any(mc_type.lower() in x.lower()
                          for x in at.get("mind_control_type", [])))
    if setting:
        checks.append(any(setting.lower() in x.lower()
                          for x in at.get("setting", [])))
    if themes:
        story_themes = [x.lower() for x in at.get("themes", [])]
        checks.append(_match_multi(themes, story_themes, theme_logic))
    if tone:
        checks.append(any(tone.lower() in x.lower() for x in at.get("tone", [])))
    if dominance:
        checks.append(any(dominance.lower() in x.lower()
                          for x in at.get("dominance", [])))
    if pacing:
        checks.append(any(pacing.lower() in x.lower() for x in at.get("pacing", [])))
    if awareness:
        checks.append(any(awareness.lower() in x.lower()
                          for x in at.get("awareness", [])))
    if length_class:
        checks.append(at.get("length_class", "").lower() == length_class.lower())
    if sexual_acts:
        story_acts = [x.lower() for x in at.get("sexual_acts", [])]
        checks.append(_match_multi(sexual_acts, story_acts, sexual_act_logic))
    if fetish_tags:
        story_fetish = [x.lower() for x in at.get("fetish", [])]
        checks.append(_match_multi(fetish_tags, story_fetish, fetish_tag_logic))
    if relationship:
        checks.append(any(relationship.lower() in x.lower()
                          for x in at.get("relationship", [])))
    if not checks:
        return 1.0
    return sum(checks) / len(checks)


_MONTH_MAP = {
    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
}


def _parse_story_date(s):
    """Parse '15 January 2024' → (year, month) tuple, or None."""
    if not s:
        return None
    parts = s.strip().lower().split()
    if len(parts) == 3:
        try:
            return (int(parts[2]), _MONTH_MAP.get(parts[1], 0))
        except ValueError:
            pass
    return None


def _fts_candidate_ids(state, q: str, search_mode: str) -> frozenset:
    """
    Query stories_fts and return a frozenset of matching story IDs.
    Returns frozenset() on FTS error (bad query syntax).
    """
    if search_mode == "literal":
        fts_q = " AND ".join(f'"{w}"' for w in q.split())
    else:
        fts_q = " OR ".join(f'"{w}"' for w in q.split())
    try:
        rows = state.search_conn.execute(
            "SELECT story_id FROM stories_fts WHERE stories_fts MATCH ?",
            (fts_q,)
        ).fetchall()
        return frozenset(row[0] for row in rows)
    except sqlite3.OperationalError:
        return frozenset()


def search_stories(state, query="", author="", categories=None, mc_type="",
                   setting="", themes=None, theme_logic="all", tone="", dominance="", pacing="",
                   awareness="", length_class="",
                   sexual_acts=None, sexual_act_logic="any",
                   fetish_tags=None, fetish_tag_logic="any",
                   relationship="",
                   min_words=0, date_from="", date_to="",
                   sort="relevance", page=1, per_page=30, soft=False,
                   archived=False, search_mode="fuzzy", liked_filter=False,
                   exclude_cats=None):
    """Search stories with full filtering, sorting, and pagination."""
    fts_ids = None
    q = query.lower().strip()
    if state.search_conn and q:
        fts_ids = _fts_candidate_ids(state, q, search_mode)

    results = []
    word_coverage = {}
    _themes_param = themes
    for s in valid_stories(state):
        if s.get("fetch_error"):
            continue
        at = get_auto_tags(s)

        # ── Text search ────────────────────────────────────────────────────────
        if q:
            sid = s.get("id", "")
            if fts_ids is not None:
                if sid not in fts_ids:
                    continue
                if search_mode == "fuzzy":
                    combined = " ".join([
                        (s.get("title") or "").lower(), dt(s).lower(),
                        (s.get("author") or "").lower(),
                        (s.get("description") or "").lower(),
                    ])
                    matched = sum(1 for w in q.split() if w in combined)
                    word_coverage[sid] = matched / len(q.split()) if q.split() else 0
            else:
                title        = (s.get("title") or "").lower()
                title_spaced = dt(s).lower()
                auth         = (s.get("author") or "").lower()
                desc         = (s.get("description") or "").lower()
                cats         = " ".join(s.get("categories", [])).lower()
                themes_str   = " ".join(at.get("themes", [])).lower()
                mc_t         = " ".join(at.get("mind_control_type", [])).lower()
                setting_t    = " ".join(at.get("setting", [])).lower()
                tone_t       = " ".join(at.get("tone", [])).lower()
                aware_t      = " ".join(at.get("awareness", [])).lower()
                combined = " ".join([title, title_spaced, auth, desc, cats,
                                     themes_str, mc_t, setting_t, tone_t, aware_t])
                words = q.split()
                if search_mode == "literal":
                    if not all(w in combined for w in words):
                        continue
                else:
                    matched = sum(1 for w in words if w in combined)
                    if matched == 0:
                        continue
                    word_coverage[s.get("id")] = matched / len(words)

        # ── Hard filters ───────────────────────────────────────────────────────
        if author and author.lower() not in (s.get("author") or "").lower():
            continue
        if min_words and s.get("word_count", 0) < min_words:
            continue
        if date_from or date_to:
            ym = _parse_story_date(s.get("added_date", ""))
            if ym:
                if date_from:
                    try:
                        y, m = int(date_from[:4]), int(date_from[5:7]) if len(date_from) > 4 else 1
                        if ym < (y, m):
                            continue
                    except ValueError:
                        pass
                if date_to:
                    try:
                        y, m = int(date_to[:4]), int(date_to[5:7]) if len(date_to) > 4 else 12
                        if ym > (y, m):
                            continue
                    except ValueError:
                        pass
        if archived and s.get("source") != "wayback":
            continue
        if liked_filter and state.prefs:
            sid = s.get("id", "")
            auth = s.get("author", "")
            if sid not in state.prefs.liked_stories and auth not in state.prefs.liked_authors:
                continue

        if soft:
            results.append(s)
        else:
            if categories:
                story_cats = [c.lower() for c in s.get("categories", [])]
                if not all(c.lower() in story_cats for c in categories):
                    continue
            if exclude_cats:
                story_cats_l = [c.lower() for c in s.get("categories", [])]
                if any(c.lower() in story_cats_l for c in exclude_cats):
                    continue
            if mc_type and not any(mc_type.lower() in x.lower()
                                   for x in at.get("mind_control_type", [])):
                continue
            if setting and not any(setting.lower() in x.lower()
                                   for x in at.get("setting", [])):
                continue
            if _themes_param:
                story_themes = [x.lower() for x in at.get("themes", [])]
                if not _match_multi(_themes_param, story_themes, theme_logic):
                    continue
            if tone and not any(tone.lower() in x.lower()
                                for x in at.get("tone", [])):
                continue
            if dominance and not any(dominance.lower() in x.lower()
                                     for x in at.get("dominance", [])):
                continue
            if pacing and not any(pacing.lower() in x.lower()
                                  for x in at.get("pacing", [])):
                continue
            if awareness and not any(awareness.lower() in x.lower()
                                     for x in at.get("awareness", [])):
                continue
            if length_class and at.get("length_class","").lower() != length_class.lower():
                continue
            if sexual_acts:
                story_acts = [x.lower() for x in at.get("sexual_acts", [])]
                if not _match_multi(sexual_acts, story_acts, sexual_act_logic):
                    continue
            if fetish_tags:
                story_fetish = [x.lower() for x in at.get("fetish", [])]
                if not _match_multi(fetish_tags, story_fetish, fetish_tag_logic):
                    continue
            if relationship and not any(relationship.lower() in x.lower()
                                        for x in at.get("relationship", [])):
                continue
            results.append(s)

    # ── Preference boost ──────────────────────────────────────────────────────
    _pref_profile: dict = {}
    if state.rec and state.prefs and state.prefs.liked_stories:
        n = len(state.prefs.liked_stories)
        if n != state.tag_profile_nlikes:
            state.tag_profile_cache  = state.rec._build_tag_profile(list(state.prefs.liked_stories))
            state.tag_profile_nlikes = n
        _pref_profile = state.tag_profile_cache

    # ── Sort ──────────────────────────────────────────────────────────────────
    if soft and (q or any([categories, mc_type, setting, _themes_param,
                            tone, dominance, pacing, awareness, length_class,
                            sexual_acts, fetish_tags, relationship])):
        def soft_score(s):
            rs = relevance_score(s, q) if q else 0.0
            fs = filter_score(s, categories=categories, mc_type=mc_type,
                              setting=setting, themes=_themes_param, theme_logic=theme_logic,
                              tone=tone, dominance=dominance, pacing=pacing,
                              awareness=awareness, length_class=length_class,
                              sexual_acts=sexual_acts, sexual_act_logic=sexual_act_logic,
                              fetish_tags=fetish_tags, fetish_tag_logic=fetish_tag_logic,
                              relationship=relationship)
            ps = pref_boost_score(s, _pref_profile) * 20 if _pref_profile else 0.0
            return rs + fs * 50 + ps
        results.sort(key=soft_score, reverse=True)
    elif sort == "relevance" or (sort == "relevance" and q):
        if q and search_mode == "fuzzy":
            results.sort(key=lambda s: (word_coverage.get(s.get("id"), 0),
                                        relevance_score(s, q) +
                                        pref_boost_score(s, _pref_profile) * 20),
                         reverse=True)
        elif q:
            results.sort(key=lambda s: relevance_score(s, q) +
                                        pref_boost_score(s, _pref_profile) * 20,
                         reverse=True)
        else:
            results.sort(key=lambda s: s.get("added_date") or "", reverse=True)
    elif sort == "words_desc":
        results.sort(key=lambda s: s.get("word_count", 0), reverse=True)
    elif sort == "words_asc":
        results.sort(key=lambda s: s.get("word_count", 0))
    elif sort == "author":
        results.sort(key=lambda s: (s.get("author") or "").lower())
    elif sort == "date":
        results.sort(key=lambda s: s.get("added_date") or "", reverse=True)
    elif sort == "title":
        results.sort(key=lambda s: dt(s).lower())
    else:
        results.sort(key=lambda s: dt(s).lower())

    total = len(results)
    start = (page - 1) * per_page
    return results[start:start + per_page], total
