"""
routes/browse.py — /, /search, /tag/<tag_code>, /random
"""
from __future__ import annotations
import random

from flask import Blueprint, request, render_template, redirect, abort, current_app

from search import parse_natural_query, nlq_apply_url
from services.search_service import search_stories
from services.data_store import valid_stories

from tagger import (SETTING_RULES, THEME_RULES, TONE_RULES,
                    MIND_CONTROL_TYPE_RULES, DOMINANCE_RULES, PACING_RULES,
                    AWARENESS_RULES)
from svengali import SV_SEXUAL_ACTS, SV_FETISH, SV_RELATIONSHIP

MC_TYPES      = [r[0] for r in MIND_CONTROL_TYPE_RULES]
SETTINGS      = [r[0] for r in SETTING_RULES]
THEMES        = [r[0] for r in THEME_RULES]
TONES         = [r[0] for r in TONE_RULES]
DOMINANCES    = [r[0] for r in DOMINANCE_RULES
                 if r[0] not in ("gentle", "brutal")]
PACINGS       = [r[0] for r in PACING_RULES] + ["rapid", "epic"]
AWARENESS     = [r[0] for r in AWARENESS_RULES
                 if r[0] not in ("consensual", "denial")]
LENGTHS       = ["flash", "short story", "novelette", "novella", "novel"]
SEXUAL_ACTS   = sorted(set(SV_SEXUAL_ACTS.values()))
FETISH_TAGS   = sorted(set(SV_FETISH.values()))
RELATIONSHIPS = sorted(set(SV_RELATIONSHIP.values()))

browse_bp = Blueprint("browse", __name__)


def _filter_params():
    """Current filter params as URL query string (without page)."""
    args = request.args.to_dict(flat=False)
    args.pop("page", None)
    parts = []
    for k, vals in args.items():
        for v in vals:
            if v:
                parts.append(f"{k}={v}")
    return "&".join(parts)


def _index_view():
    """Shared view for / and /search: read every filter from the query string, run
    search_stories(), and render index.html with the results, pagination, filter
    options, and any filters detected in a natural-language query.
    """
    state = current_app.state
    q            = request.args.get("q", "")
    cat          = request.args.get("cat", "")
    mc_type      = request.args.get("mc_type", "")
    setting      = request.args.get("setting", "")
    themes       = request.args.getlist("theme")
    theme_logic  = request.args.get("theme_logic", "all")
    tone         = request.args.get("tone", "")
    dominance    = request.args.get("dominance", "")
    pacing       = request.args.get("pacing", "")
    awareness    = request.args.get("awareness", "")
    sexual_acts       = request.args.getlist("sexual_act")
    sexual_act_logic  = request.args.get("sexual_act_logic", "any")
    fetish_tags       = request.args.getlist("fetish_tag")
    fetish_tag_logic  = request.args.get("fetish_tag_logic", "any")
    relationship = request.args.get("relationship", "")
    length_class = request.args.get("length", "")
    date_from    = request.args.get("date_from", "")
    date_to      = request.args.get("date_to", "")
    sort         = request.args.get("sort", "relevance")
    page         = max(1, int(request.args.get("page", 1)))
    soft         = request.args.get("soft", "") == "1"
    archived     = request.args.get("archived", "") == "1"
    liked_filter = request.args.get("liked_filter", "") == "1"
    search_mode  = request.args.get("search_mode", "fuzzy")
    exclude_cat  = request.args.get("exclude_cat", "")
    per_page     = 30

    cats      = [c.strip() for c in cat.split() if c.strip()]
    excl_cats = [c.strip() for c in exclude_cat.split() if c.strip()]

    nlq_clean, nlq_detected = parse_natural_query(q) if q else ("", {})
    nlq_url = nlq_apply_url(nlq_clean, nlq_detected) if nlq_detected else ""

    results, total = search_stories(
        state,
        query=q, categories=cats, mc_type=mc_type, setting=setting,
        themes=themes, theme_logic=theme_logic, tone=tone, dominance=dominance, pacing=pacing,
        awareness=awareness, length_class=length_class,
        sexual_acts=sexual_acts, sexual_act_logic=sexual_act_logic,
        fetish_tags=fetish_tags, fetish_tag_logic=fetish_tag_logic,
        relationship=relationship,
        date_from=date_from, date_to=date_to,
        sort=sort, page=page, per_page=per_page, soft=soft,
        archived=archived, search_mode=search_mode, liked_filter=liked_filter,
        exclude_cats=excl_cats
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template(
        "index.html",
        results=results, total=total, page=page, total_pages=total_pages,
        q=q, cat=cat, mc_type=mc_type, setting=setting, selected_themes=themes,
        theme_logic=theme_logic, tone=tone,
        dominance=dominance, pacing=pacing, awareness=awareness,
        selected_sexual_acts=sexual_acts, sexual_act_logic=sexual_act_logic,
        selected_fetish_tags=fetish_tags, fetish_tag_logic=fetish_tag_logic,
        relationship=relationship,
        length_class=length_class, date_from=date_from, date_to=date_to,
        sort=sort, soft=soft, archived=archived, liked_filter=liked_filter,
        search_mode=search_mode,
        exclude_cat=exclude_cat,
        params=_filter_params(),
        nlq_detected=nlq_detected, nlq_url=nlq_url,
        mc_types=MC_TYPES, settings=SETTINGS, themes=THEMES, tones=TONES,
        dominances=DOMINANCES, pacings=PACINGS, awareness_opts=AWARENESS,
        sexual_acts=SEXUAL_ACTS, fetish_tags=FETISH_TAGS, relationships=RELATIONSHIPS,
        lengths=LENGTHS, prefs=state.prefs, request=request
    )


@browse_bp.route("/")
def index():
    """Home page: search box, filter sidebar and results (same view as /search)."""
    return _index_view()


@browse_bp.route("/search")
def search():
    """Search results page; identical to / so filter URLs can use either path."""
    return _index_view()


@browse_bp.route("/tag/<tag_code>")
def tag(tag_code):
    """All stories with one mcstories.com category code (e.g. /tag/mf), sorted by title,
    unpaginated.
    """
    state = current_app.state
    results, total = search_stories(
        state, categories=[tag_code], sort="title", per_page=1000
    )
    return render_template(
        "index.html",
        results=results, total=total, page=1, total_pages=1,
        q="", cat=tag_code, mc_type="", setting="", selected_themes=[],
        theme_logic="all", tone="",
        dominance="", pacing="", awareness="",
        selected_sexual_acts=[], sexual_act_logic="any",
        selected_fetish_tags=[], fetish_tag_logic="any", relationship="",
        length_class="", sort="title", soft=False,
        params=f"cat={tag_code}",
        exclude_cat="", archived=False, liked_filter=False, search_mode="fuzzy",
        date_from="", date_to="",
        nlq_detected={}, nlq_url="",
        mc_types=MC_TYPES, settings=SETTINGS, themes=THEMES, tones=TONES,
        dominances=DOMINANCES, pacings=PACINGS, awareness_opts=AWARENESS,
        sexual_acts=SEXUAL_ACTS, fetish_tags=FETISH_TAGS, relationships=RELATIONSHIPS,
        lengths=LENGTHS, prefs=state.prefs, request=request
    )


@browse_bp.route("/random")
def random_story():
    """Redirect to a random story that has text."""
    state = current_app.state
    pool = valid_stories(state)
    if not pool:
        abort(404)
    s = random.choice(pool)
    return redirect(f"/story/{s['id']}")
