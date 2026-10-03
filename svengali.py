#!/usr/bin/env python3
"""
svengali.py — scrape and import tags from Svengali (gregariousfrog.com)

Svengali is a community tagging system for mcstories.com stories.
It stores human-verified tags contributed by readers since 2010.
These are significantly higher quality than auto-tags.

TWO MODES:

1. SCRAPE — fetch tags from Svengali's API for all stories in your index
   python svengali.py --scrape
   python svengali.py --scrape --limit 100   # test first

2. IMPORT — merge already-scraped Svengali data into stories_index.json
   python svengali.py --import
   python svengali.py --import --dry-run

The scraper saves raw API responses to output/svengali_raw.jsonl (one per line)
so you can re-import without re-scraping.

API endpoint (discovered from Svengali source):
  https://www.gregariousfrog.com/svengali/servers/pixi.php
  ?url=http://www.mcstories.com/StorySlug/index.html

Please be respectful — rate limit is set to 2 seconds between requests by default.
The site is run by a single person (Gregarious Frog) as a community service.
Consider emailing them before bulk scraping: the help page suggests contacting "Frog".

If they share a data export directly, use --import-file to load it:
  python svengali.py --import-file svengali_export.json
"""

from __future__ import annotations

import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import requests
import index_store

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL       = "https://www.gregariousfrog.com/svengali/servers/pixi.php"
STORY_BASE     = "http://www.mcstories.com"   # Svengali uses http not https
DELAY          = 1.0    # seconds between requests
MAX_RETRIES    = 3
INDEX_PATH     = Path("./output/stories_index.json")
RAW_PATH       = Path("./output/svengali_raw.jsonl")
HEADERS        = {
    "User-Agent": "Mozilla/5.0 (compatible; personal-archiver/1.0)",
    "Referer":    "http://www.mcstories.com/",
}

session = requests.Session()
session.headers.update(HEADERS)


# ── Svengali tag taxonomy → our tag dimensions ────────────────────────────────
# Maps Svengali's tag names (from the help doc) to our auto_tags dimensions.
# Svengali uses a hierarchical tree; we flatten to our flat tag structure.

# Mind control type
SV_MC_TYPE = {
    "Body Puppet":          "body puppet",
    "Possession":           "possession",
    "Chemical":             "chemical",
    "Hypnosis":             "hypnosis",
    "Induction":            "hypnosis",
    "Breast Fixation":      "hypnosis",
    "Crystals / Pendants":  "hypnosis",
    "Eye Fixation":         "hypnosis",
    "Spirals":              "hypnosis",
    "Magic":                "magic",
    "Demon":                "magic",
    "Genie":                "magic",
    "Leprechaun":           "magic",
    "Operant Conditioning": "operant conditioning",
    "Parasite / Symbiote":  "parasite / symbiote",
    "Pheromones":           "pheromones",
    "Psionic":              "psionic",
    "Reality Manipulation": "reality manipulation",
    "Subliminal":           "subliminal",
    "Technology":           "technology",
    "Nanotechnology":       "nanotechnology",
    "Virtual Reality":      "virtual reality",
    "Virus":                "virus",
}

# Reaction → awareness
SV_AWARENESS = {
    "Denial":                    "denial",
    "Playing Along":             "playing along",
    "Skepticism":                "skeptical / confused",
    "Valiant but Futile Struggle": "valiant but futile struggle",
}

# Application → dominance
SV_DOMINANCE = {
    "Accidental":    "self-inflicted",
    "Brutal":        "brutal",
    "Gentle":        "gentle",
    "Gradual":       "gradual",
    "Self-Inflicted":"self-inflicted",
    "Speedy":        "speedy",
    "Subtle":        "subtle",
}

# Mental Manipulation → mental_changes
SV_MENTAL = {
    "Addiction":              "addiction / need",
    "Semen":                  "addiction / need",
    "Can't Say No":           "obedience programming",
    "Conform to Clothing":    "obedience programming",
    "Emotion Control":        "emotion control",
    "Induced Love":           "induced love",
    "Enhanced Libido":        "obedience programming",
    "Bimboization":           "bimboization",
    "IQ Lowering":            "intelligence change",
    "Heightened Intelligence":"intelligence change",
    "Himboization":           "bimboization",
    "Illiterate":             "intelligence change",
    "Mental Age Regression":  "intelligence change",
    "Forgets Name":           "memory alteration",
    "Relationship Amnesia":   "memory alteration",
    "Animal":                 "speech modification",
    "Bimbo":                  "speech modification",
    "Foul-Mouthed":           "speech modification",
    "Infantile":              "speech modification",
    "Mute":                   "speech modification",
    "Restricted Topics":      "speech modification",
    "Third-person":           "speech modification",
    "Multiple Personalities": "personality removal",
    "Oblivious to Change":    "obedience programming",
    "Orgasm Control":         "orgasm control",
    "Personality Removal":    "personality removal",
    "Pregnancy Obsession":    "obedience programming",
    "Purification":           "purification",
    "Slave Mentality":        "obedience programming",
    "Slut Mentality":         "obedience programming",
    "Stepfordization":        "stepfordization",
}

# Orientation changes → themes
SV_THEMES_FROM_ORIENTATION = {
    "To Bi-Sexual":     "identity loss",
    "To Gay":           "identity loss",
    "To Lesbian":       "identity loss",
    "To Animal":        "identity loss",
    "To Robot":         "identity loss",
    "To Whore":         "slave mentality",
}

# Physical Alterations → themes
SV_PHYSICAL = {
    "Age Regression":           "body modification",
    "Breast Enhancement":       "body modification",
    "Butt Expansion":           "body modification",
    "Faux Pregnancy":           "body modification",
    "Fertility Control":        "body modification",
    "Hair Removal":             "body modification",
    "Lactation":                "body modification",
    "Penis Enhancement":        "body modification",
    "Weight Gain":              "body modification",
    "Weight Loss":              "body modification",
    "Full-Body Transformation": "body modification",
    "To Anthropomorphic Animal":"body modification",
    "Body Switch":              "body modification",
    "Gender Switch":            "body modification",
    "To Hermaphrodite":         "body modification",
    "To Mannequin / Doll":      "body modification",
    "To Robot":                 "body modification",
}

# Social Hierarchy → site tags
SV_GENDER_DYNAMIC = {
    "Female Dominant":  "fd",
    "Female Submission":"mf",
    "Male Dominant":    "md",
    "Male Submission":  "mf",
}

# Level of Consent → awareness
SV_CONSENT = {
    "Consensual":       "consensual",
    "Resistant":        "valiant but futile struggle",
    "Threat / Blackmail":"valiant but futile struggle",
    "Non-Consensual":   "valiant but futile struggle",
    "Rape":             "valiant but futile struggle",
}

# Themes → our themes
SV_THEMES = {
    "Betrayal":           "betrayal",
    "Celebrity":          "celebrity",
    "College / School":   "college",
    "Conspiracy":         "conspiracy",
    "Corporation":        "conspiracy",
    "Drone / Hive":       "drone / hive mind",
    "Harem":              "harem",
    "Rescue":             "rescue",
    "Revenge":            "revenge",
    "Reversal of Fortune":"power corrupts",
    "Serial Recruitment": "serial recruitment",
    "Tentacles":          "tentacles",
    "Vampire":            "mythology / religion",
    "Power Corrupts":     "power corrupts",
    "Religion":           "mythology / religion",
    "Time Stop":          "time travel",
    "Time Travel":        "time travel",
    "Workplace":          "workplace",
    # Incest subtypes
    "Brother / Brother":  "incest",
    "Brother / Sister":   "incest",
    "Sister / Sister":    "incest",
    "Father / Daughter":  "incest",
    "Father / Son":       "incest",
    "Mother / Daughter":  "incest",
    "Mother / Son":       "incest",
}

# Genre → setting
SV_SETTING = {
    "Comic Book":       "comic book",
    "Fantasy":          "fantasy / historical",
    "Historical":       "fantasy / historical",
    "Horror":           "supernatural",
    "Humor":            "humor",
    "Manga / Anime":    "anime",
    "Romantic":         "romance",
    "Science Fiction":  "science fiction",
    "Supernatural":     "supernatural",
}

# Pacing
SV_PACING = {
    "Epic":     "epic",
    "Leisurely":"leisurely",
    "Rapid":    "rapid",
}

# POV
SV_POV = {
    "First-person":  "first person",
    "Second-person": "second person",
    "Third-person":  "third person",
    "Controller":    "third person",
    "Victim":        "first person",
}

# Story status
SV_STATUS = {
    "Complete":   "complete",
    "Incomplete": "incomplete",
}

# Sexual Activities → sexual_acts
# Note: "Animal" and "Cow"/"Dog"/"Horse" are skipped — they are ambiguous with
# the speech-modification "Animal" tag and the orientation "To Animal" subtypes.
# "Necrophilia" excluded by design.
SV_SEXUAL_ACTS = {
    "Anal Sex":         "anal",
    "Masturbation":     "masturbation",
    "Fingering":        "fingering",
    "Rimming":          "rimming",
    "Fisting":          "fisting",
    "Cuckold":          "cuckold",
    "Exhibitionism":    "exhibitionism",
    "Public Nudity":    "exhibitionism",
    "Facial":           "facial",
    "Creampie":         "creampie",
    "Impregnation":     "impregnation",
    "Titfuck":          "titfuck",
    "Striptease":       "striptease",
    "Voyeurism":        "voyeurism",
    # Oral sex and its subtypes
    "Oral Sex":         "oral",
    "Snowball":         "oral",
    "Female to Female": "oral",
    "Female to Male":   "oral",
    "Female to Animal": "oral",
    "Male to Female":   "oral",
    "Male to Male":     "oral",
    # Toys and subtypes
    "Toy":              "toys",
    "Toys":             "toys",
    "Bottle":           "toys",
    "Butt Plug":        "toys",
    "Dildo":            "toys",
    "Vegetable":        "toys",
    "Vibrator":         "toys",
}

# Fetish / Clothing / Occupations / Human Body / Extreme → fetish (flat list)
SV_FETISH = {
    # BDSM
    "BDSM":               "bdsm",
    "Bondage":             "bdsm",
    "Gag":                 "bdsm",
    "Handcuffs":           "bdsm",
    "Rope Play":           "bdsm",
    "Sadomasochism":       "sadomasochism",
    "Flogging":            "sadomasochism",
    "Spanking":            "sadomasochism",
    # Animal roleplay (distinct from bestiality)
    "Animal Role-play":    "animal roleplay",
    "Cow-girl":            "animal roleplay",
    "Furry":               "animal roleplay",
    "Ponygirl":            "animal roleplay",
    # Bestiality (parent tag only — subtypes Cow/Dog/Horse are ambiguous)
    "Bestiality":          "bestiality",
    # Body decoration
    "Big Beautiful Woman": "bbw",
    "Bodypaint":           "body decoration",
    "Piercings":           "body decoration",
    "Tattoos":             "body decoration",
    # Clothing fetishes
    "Boots":               "fetish clothing",
    "Collar":              "fetish clothing",
    "Corset":              "fetish clothing",
    "Cross-Dressing":      "cross-dressing",
    "Diaper":              "fetish clothing",
    "Lingerie":            "fetish clothing",
    "Rubber / Latex":      "fetish clothing",
    "Sheer / Tight":       "fetish clothing",
    "Shoes":               "fetish clothing",
    "Swimsuit":            "fetish clothing",
    "Stockings":           "fetish clothing",
    "Other Clothing":      "fetish clothing",
    "French Maid":         "costume",
    "Uniform":             "costume",
    # Occupations (character role as fetish element)
    "Babysitter":          "occupation",
    "Cheerleader":         "occupation",
    "Housewife":           "occupation",
    "Medical":             "occupation",
    "Therapist":           "occupation",
    "Law Enforcement":     "occupation",
    "Legal Profession":    "occupation",
    "Military":            "occupation",
    "Religious":           "occupation",
    "Royalty":             "occupation",
    "Stage Hypnotist":     "occupation",
    "Teacher":             "occupation",
    # Human body fetishes
    "Feet":                "feet",
    "Breasts":             "breast fetish",
    "Lactation":           "lactation",
    "Small Penis":         "humiliation",
    "Large Penis":         "size",
    "Virgin":              "virgin",
    "Pregnancy":           "pregnancy fetish",
    "Cannibalism":         "extreme",
    "Snuff":               "snuff",
    "Torture":             "pain / torture",
    # Watersports / scat
    "Enema":               "watersports / scat",
    "Scatophilia":         "watersports / scat",
    "Golden Shower":       "watersports / scat",
    # Other
    "Smoking":             "smoking",
    "Chocolate":           "food play",
}

# Relationship / Power Imbalance → relationship
SV_RELATIONSHIP = {
    # Relationship types
    "Enemies":            "enemies",
    "Strangers":          "strangers",
    "Peers":              "peers",
    "Classmates":         "peers",
    "Teammates":          "peers",
    "Rivals":             "rivals",
    "Couples":            "couples",
    "Coworkers":          "coworkers",
    "Friends":            "friends",
    "Neighbors":          "friends",
    "Inlaws":             "family adjacent",
    # Power imbalance dynamics
    "Boss / Employee":    "boss / employee",
    "Coach / Player":     "coach / player",
    "Customer / Employee":"service worker",
    "Doctor / Patient":   "doctor / patient",
    "Teacher / Student":  "teacher / student",
    "Authority Dominant": "authority",
    "Subordinate Dominant": "subordinate dominant",
}



# ── API ───────────────────────────────────────────────────────────────────────
# All endpoints confirmed from svengali.min.js source.
# All use JSONP (callback=?) and are fully public — no auth required.

SERVER  = "https://www.gregariousfrog.com/svengali/servers/"

_tag_id_map:   dict = {}  # numeric id str → tag name
_tag_name_map: dict = {}  # tag name → numeric id str


def _jsonp_get(endpoint: str, params: dict) -> Optional[dict]:
    """Make a JSONP request. Strips the cb(...) wrapper to get raw JSON."""
    params["callback"] = "cb"
    url = SERVER + endpoint
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            text = r.text.strip()
            # Strip JSONP wrapper: cb({...}) or cb([...])
            start = text.find("(")
            end   = text.rfind(")")
            if start != -1 and end != -1:
                return json.loads(text[start+1:end])
            if text.startswith(("{", "[")):
                return json.loads(text)
            log.debug(f"  Unexpected response from {endpoint}: {text[:100]}")
            return None
        except requests.exceptions.ConnectionError:
            log.warning(f"  Connection error ({endpoint}, attempt {attempt+1})")
            time.sleep(DELAY * 2)
        except json.JSONDecodeError as e:
            log.warning(f"  JSON parse error ({endpoint}): {e}")
            return None
        except Exception as e:
            log.warning(f"  Error ({endpoint}, attempt {attempt+1}): {e}")
            time.sleep(DELAY)
    return None


def _walk_tags(node):
    """Recursively walk the GetTagList.php response tree."""
    if isinstance(node, dict):
        nid  = str(node.get("id", ""))
        name = node.get("name", "")
        if nid and name:
            _tag_id_map[nid]    = name
            _tag_name_map[name] = nid
        children = node.get("children", {})
        if children:
            tags = children.get("tag", [])
            if isinstance(tags, dict):
                tags = [tags]
            for child in tags:
                _walk_tags(child)
    elif isinstance(node, list):
        for item in node:
            _walk_tags(item)


def fetch_tag_list() -> dict:
    """Fetch the full tag taxonomy. Returns {id: name} map."""
    global _tag_id_map, _tag_name_map
    if _tag_id_map:
        return _tag_id_map
    log.info("Fetching Svengali tag list...")
    data = _jsonp_get("GetTagList.php", {})
    if not data:
        log.error("Could not fetch tag list")
        return {}
    _walk_tags(data)
    log.info(f"  Loaded {len(_tag_id_map)} tags")
    return _tag_id_map


def fetch_title_tags(slug: str, title_name: str = "") -> Optional[list]:
    """
    Fetch tags for a story from GetTitleTags.php.
    Returns list of tag name strings, [] if untagged, None on error.
    JS call: GetTitleTags.php?titlename=...&titleurl=...&callback=?
    """
    data = _jsonp_get("GetTitleTags.php", {
        "titlename": title_name or slug,
        "titleurl":  slug,
    })
    if data is None:
        return None
    tag_ids = data.get("TitleTags", [])
    if not tag_ids:
        return []
    fetch_tag_list()
    return [_tag_id_map.get(str(tid), str(tid)) for tid in tag_ids]


def probe_api() -> None:
    """Probe the API with the tag list and a few test stories."""
    print("Probing Svengali API...")
    tag_map = fetch_tag_list()
    if not tag_map:
        print("  FAILED: could not fetch tag list")
        return
    print(f"  Tag list OK: {len(tag_map)} tags loaded")
    for tid, name in list(tag_map.items())[:8]:
        print(f"    id={tid}: {name}")
    for slug in ["WheeHearAndObey", "HypnoticInfluences", "ThePerfectWife"]:
        time.sleep(1)
        tags = fetch_title_tags(slug)
        if tags is None:
            print(f"  {slug}: error/timeout")
        elif not tags:
            print(f"  {slug}: no tags assigned")
        else:
            print(f"  {slug}: {tags[:6]}")
            break


# ── Tag parser ────────────────────────────────────────────────────────────────
# ── Tag parser ────────────────────────────────────────────────────────────────

def parse_svengali_response(raw: dict) -> dict:
    """
    Parse a Svengali API response into our auto_tags format.
    The exact structure depends on what the API returns —
    this handles the most likely formats based on the help doc taxonomy.
    """
    if not raw:
        return {}

    tags = raw.get("tags", [])
    if not isinstance(tags, list):
        return {}

    # tags is a list of name strings from fetch_title_tags()
    tag_names = set(t for t in tags if isinstance(t, str) and t)

    if not tag_names:
        return {}

    result = {
        "mind_control_type": [],
        "awareness":         [],
        "dominance":         [],
        "mental_changes":    [],
        "themes":            [],
        "setting":           [],
        "pacing":            [],
        "sexual_acts":       [],
        "fetish":            [],
        "relationship":      [],
        "pov":               None,
        "story_status":      None,
        "_svengali_tags":    sorted(tag_names),  # keep raw for reference
    }

    for tag in tag_names:
        if tag in SV_MC_TYPE:
            v = SV_MC_TYPE[tag]
            if v not in result["mind_control_type"]:
                result["mind_control_type"].append(v)

        if tag in SV_AWARENESS:
            v = SV_AWARENESS[tag]
            if v not in result["awareness"]:
                result["awareness"].append(v)

        if tag in SV_CONSENT:
            v = SV_CONSENT[tag]
            if v not in result["awareness"]:
                result["awareness"].append(v)

        if tag in SV_DOMINANCE:
            v = SV_DOMINANCE[tag]
            if v not in result["dominance"]:
                result["dominance"].append(v)

        if tag in SV_MENTAL:
            v = SV_MENTAL[tag]
            if v not in result["mental_changes"]:
                result["mental_changes"].append(v)

        if tag in SV_THEMES:
            v = SV_THEMES[tag]
            if v not in result["themes"]:
                result["themes"].append(v)

        if tag in SV_THEMES_FROM_ORIENTATION:
            v = SV_THEMES_FROM_ORIENTATION[tag]
            if v not in result["themes"]:
                result["themes"].append(v)

        if tag in SV_PHYSICAL:
            v = SV_PHYSICAL[tag]
            if v not in result["themes"]:
                result["themes"].append(v)

        if tag in SV_SETTING:
            v = SV_SETTING[tag]
            if v not in result["setting"]:
                result["setting"].append(v)

        if tag in SV_PACING:
            v = SV_PACING[tag]
            if v not in result["pacing"]:
                result["pacing"].append(v)

        if tag in SV_SEXUAL_ACTS:
            v = SV_SEXUAL_ACTS[tag]
            if v not in result["sexual_acts"]:
                result["sexual_acts"].append(v)

        if tag in SV_FETISH:
            v = SV_FETISH[tag]
            if v not in result["fetish"]:
                result["fetish"].append(v)

        if tag in SV_RELATIONSHIP:
            v = SV_RELATIONSHIP[tag]
            if v not in result["relationship"]:
                result["relationship"].append(v)

        if tag in SV_POV:
            result["pov"] = SV_POV[tag]

        if tag in SV_STATUS:
            result["story_status"] = SV_STATUS[tag]

    # Clean up empty lists
    return {k: v for k, v in result.items() if v}


# ── Scraper ───────────────────────────────────────────────────────────────────

def fetch_all_tagged_stories_via_search() -> dict:
    """
    Bulk approach: search each tag, invert results to get story→tags map.
    ~100 requests instead of 17k. Takes ~2 minutes instead of hours.
    SearchOnTagsV2.php returns top 100 results per tag as:
      ["Title%%slug%%cssclass%%percent", ...]
    """
    fetch_tag_list()
    if not _tag_id_map:
        log.error("Could not load tag list")
        return {}

    story_tags: dict = {}
    total = len(_tag_id_map)
    log.info(f"Bulk: fetching {total} tags (~{total * DELAY / 60:.0f} min)...")

    for i, (tag_id, tag_name) in enumerate(_tag_id_map.items(), 1):
        data = _jsonp_get("SearchOnTagsV2.php", {
            "tags": tag_id,
            "rnd":  str(int(time.time())),
        })
        if data and data.get("Titles"):
            for entry in data["Titles"]:
                if not entry:
                    continue
                parts = str(entry).split("%%")
                if len(parts) >= 2:
                    slug = parts[1].strip("/").split("/")[0]
                    if slug:
                        story_tags.setdefault(slug, [])
                        if tag_name not in story_tags[slug]:
                            story_tags[slug].append(tag_name)
        if i % 10 == 0:
            log.info(f"  [{i}/{total}] {len(story_tags)} stories found so far")
        time.sleep(DELAY)

    log.info(f"Bulk done: {len(story_tags)} stories with tags")
    return story_tags


def scrape(index_path: Path, raw_path: Path,
           limit: Optional[int] = None, resume: bool = True,
           workers: int = 2, bulk: bool = True) -> None:
    """
    Fetch Svengali tags.

    BULK MODE (default): search each tag (~100 requests, ~2 min).
    Covers stories appearing in any tag's top-100 results.

    PER-STORY MODE (--no-bulk): one request per story with concurrent
    workers. Complete coverage, ~2.4 hours at 2 workers / 1s delay.
    Run after bulk to fill any gaps.
    """
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if bulk:
        story_tags = fetch_all_tagged_stories_via_search()
        if not story_tags:
            log.error("Bulk returned nothing — try --no-bulk")
            return
        done: set = set()
        if resume and raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["slug"])
                    except:
                        pass
        written = 0
        with open(raw_path, "a", encoding="utf-8") as f:
            for slug, tags in story_tags.items():
                if slug in done:
                    continue
                if limit and written >= limit:
                    break
                f.write(json.dumps({
                    "slug": slug,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "tags": tags,
                }, ensure_ascii=False) + "\n")
                written += 1
        log.info(f"Written {written} records to {raw_path}")
        log.info("Run: python svengali.py --import")
        return

    # ── Per-story mode ────────────────────────────────────────────────────────
    import concurrent.futures, threading
    log.info(f"Per-story mode: {workers} workers, {DELAY}s delay")
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    all_slugs = [s["id"] for s in data.get("stories", [])
                 if not s.get("fetch_error") and s.get("_has_text", True)]

    done: set = set()
    if resume and raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["slug"])
                except:
                    pass
        log.info(f"Already done: {len(done):,}")

    to_scrape = [s for s in all_slugs if s not in done]
    if limit:
        to_scrape = to_scrape[:limit]

    log.info(f"To scrape: {len(to_scrape):,}  "
             f"(est. {len(to_scrape) * DELAY / workers / 3600:.1f}h)")

    fetch_tag_list()
    by_id   = {s["id"]: s for s in data.get("stories", [])}
    ok = not_found = err = 0
    lock = threading.Lock()

    def fetch_one(slug):
        title = by_id.get(slug, {}).get("title") or slug
        tags  = fetch_title_tags(slug, title)
        time.sleep(DELAY)
        return {"slug": slug,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "tags": tags}

    with open(raw_path, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_one, s): s for s in to_scrape}
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                rec  = fut.result()
                tags = rec.get("tags")
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    if tags is None: err += 1
                    elif not tags:   not_found += 1
                    else:            ok += 1
                if i % 200 == 0 or i <= 3:
                    log.info(f"  [{i}/{len(to_scrape)}] "
                             f"{ok} tagged, {not_found} no tags, {err} errors")

    log.info(f"Done. {ok} tagged, {not_found} no tags, {err} errors")



# ── Importer ──────────────────────────────────────────────────────────────────

def import_tags(index_path: Path, raw_path: Path,
                import_file: Optional[Path] = None,
                dry_run: bool = False) -> None:
    """
    Merge Svengali tags from raw_path into stories_index.json.

    Svengali tags are stored under auto_tags._svengali_tags (raw list)
    and merged into the relevant auto_tags dimensions.

    Svengali tags are treated as high-confidence — they override
    auto-tagger values for the same dimension when present.
    """
    source = import_file or raw_path
    if not source.exists():
        log.error(f"No Svengali data found at {source}")
        log.error("Run: python svengali.py --scrape   first")
        return

    # Load raw Svengali data
    log.info(f"Loading Svengali data from {source}...")
    sv_by_slug: dict[str, dict] = {}

    if source.suffix == ".jsonl":
        with open(source, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    slug = rec.get("slug")
                    tags = rec.get("tags")
                    if slug and tags:
                        sv_by_slug[slug] = parse_svengali_response({"tags": tags})
                except:
                    pass
    else:
        with open(source, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            for item in raw:
                slug = item.get("slug") or item.get("id")
                tags = item.get("tags") or item.get("tag_names") or []
                if slug:
                    sv_by_slug[slug] = parse_svengali_response({"tags": tags})
        elif isinstance(raw, dict):
            for slug, val in raw.items():
                tags = val if isinstance(val, list) else val.get("tags", [])
                sv_by_slug[slug] = parse_svengali_response({"tags": tags})

    tagged_count = sum(1 for v in sv_by_slug.values() if v)
    log.info(f"Svengali records: {len(sv_by_slug):,}  ({tagged_count:,} with tags)")

    # Resolve to absolute path so errors are unambiguous
    index_path = index_path.resolve()
    if not index_path.exists():
        log.error(f"Index not found: {index_path}")
        log.error("Pass the correct path with: --index /path/to/stories_index.json")
        return

    # Load main index
    log.info(f"Loading {index_path}...")
    data = index_store.load_index(index_path, for_write=not dry_run)

    stories = data.get("stories", [])
    log.info(f"Main index: {len(stories):,} stories")

    updated = skipped = no_sv = 0

    for story in stories:
        slug = story.get("id")
        sv   = sv_by_slug.get(slug)

        if not sv:
            no_sv += 1
            continue

        at = story.get("auto_tags") or {}

        # Merge: Svengali tags augment (and for some dims override) auto_tags
        changed = False
        for dim, sv_vals in sv.items():
            if dim == "_svengali_tags":
                # Store raw svengali tags for reference
                at["_svengali_tags"] = sv_vals
                changed = True
                continue

            if dim == "pov" and sv_vals:
                if at.get("pov") != sv_vals:
                    at["pov"] = sv_vals
                    changed = True
                continue

            if dim == "story_status" and sv_vals:
                if not at.get("story_status"):
                    at["story_status"] = sv_vals
                    changed = True
                continue

            if isinstance(sv_vals, list) and sv_vals:
                existing = set(at.get(dim, []))
                new_vals = [v for v in sv_vals if v not in existing]
                if new_vals:
                    at[dim] = list(existing) + new_vals
                    changed = True
                # Mark all svengali-contributed tags in _tag_sources
                sources = at.setdefault("_tag_sources", {})
                for v in sv_vals:
                    src = sources.get(v) or []
                    if "svengali" not in src:
                        sources[v] = list(src) + ["svengali"]
                        changed = True

        if changed:
            story["auto_tags"] = at
            updated += 1
        else:
            skipped += 1

    log.info(f"\nResults:")
    log.info(f"  Updated:         {updated:,}")
    log.info(f"  Already current: {skipped:,}")
    log.info(f"  No Svengali data:{no_sv:,}")

    if dry_run:
        log.info("Dry run — no changes saved.")

        # Show a sample of what would change
        sample = [s for s in stories if s.get("auto_tags", {}).get("_svengali_tags")][:5]
        for s in sample:
            print(f"\n  {s['id']}:")
            print(f"    Svengali tags: {s['auto_tags'].get('_svengali_tags', [])[:8]}")
            print(f"    story_status:  {s['auto_tags'].get('story_status')}")
            print(f"    mc_type:       {s['auto_tags'].get('mind_control_type', [])[:3]}")
        return

    if updated == 0:
        log.info("Nothing to update.")
        return

    # Save (validated: re-reads the written file and checks the story count)
    log.info(f"\nSaving {len(stories):,} stories to {index_path}...")
    index_store.save_index(data, index_path, validate=True)
    log.info(f"Saved to {index_path}")
    log.info("Run: make rebuild-jsonl   to update training files")


# ── Stats ─────────────────────────────────────────────────────────────────────

def show_stats(index_path: Path) -> None:
    """Show how many stories have Svengali tags and what dimensions are covered."""
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    stories = data.get("stories", [])

    sv_stories  = [s for s in stories if s.get("auto_tags", {}).get("_svengali_tags")]
    status_tags = [s for s in stories if s.get("auto_tags", {}).get("story_status")]

    print(f"\nSvengali coverage:")
    print(f"  Stories with Svengali tags: {len(sv_stories):,} / {len(stories):,}")
    print(f"  Stories with story_status:  {len(status_tags):,}")

    if sv_stories:
        from collections import Counter
        status_dist = Counter(
            s["auto_tags"]["story_status"]
            for s in status_tags
            if s.get("auto_tags", {}).get("story_status")
        )
        print(f"\n  Story status distribution:")
        for status, count in status_dist.most_common():
            print(f"    {status:<20} {count:,}")

        # Show dimension coverage
        dims = ["mind_control_type", "awareness", "dominance", "mental_changes",
                "themes", "setting", "pacing"]
        print(f"\n  Dimension coverage (stories with Svengali data):")
        for dim in dims:
            count = sum(1 for s in sv_stories
                       if s.get("auto_tags", {}).get(dim))
            pct = 100 * count / max(len(sv_stories), 1)
            print(f"    {dim:<25} {count:>5}  ({pct:.0f}%)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("./output").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("./output/svengali.log"),
        ]
    )

    p = argparse.ArgumentParser(
        description="Scrape and import Svengali community tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. First probe the API to confirm it's accessible:
       python svengali.py --probe

  2. Test on a small batch:
       python svengali.py --scrape --limit 50

  3. Check the raw data looks right:
       python svengali.py --import --dry-run

  4. Run the full scrape (this will take many hours):
       python svengali.py --scrape

  5. Import into your index:
       python svengali.py --import

  6. Rebuild training files:
       make rebuild-jsonl

  If Gregarious Frog shares a data export:
       python svengali.py --import-file svengali_export.json

  View stats:
       python svengali.py --stats
        """
    )

    p.add_argument("--scrape",       action="store_true",
                   help="Scrape Svengali tags (bulk mode by default, ~2 min)")
    p.add_argument("--no-bulk",      action="store_true",
                   help="Per-story mode instead of bulk (complete but slow)")
    p.add_argument("--workers",      type=int, default=2,
                   help="Workers for per-story mode (default: 2)")
    p.add_argument("--import",       action="store_true", dest="do_import",
                   help="Import scraped Svengali tags into stories_index.json")
    p.add_argument("--import-file",  default=None,
                   help="Import from a specific file (JSON or JSONL)")
    p.add_argument("--probe",        action="store_true",
                   help="Probe the API with one story to check it's working")
    p.add_argument("--stats",        action="store_true",
                   help="Show Svengali coverage statistics")
    p.add_argument("--dry-run",      action="store_true",
                   help="Preview import without saving")
    p.add_argument("--limit",        type=int, default=None,
                   help="Only scrape N stories (for testing)")
    p.add_argument("--no-resume",    action="store_true",
                   help="Scrape from scratch, ignoring existing raw data")
    p.add_argument("--delay",        type=float, default=DELAY,
                   help=f"Seconds between requests (default: {DELAY})")
    p.add_argument("--index",        default=str(INDEX_PATH))
    p.add_argument("--raw",          default=str(RAW_PATH))

    args = p.parse_args()
    INDEX_PATH = Path(args.index)
    RAW_PATH   = Path(args.raw)
    DELAY      = args.delay

    Path("./output").mkdir(exist_ok=True)

    if args.probe:
        probe_api()
    elif args.scrape:
        scrape(INDEX_PATH, RAW_PATH,
               limit=args.limit, resume=not args.no_resume,
               workers=getattr(args, "workers", 2),
               bulk=not getattr(args, "no_bulk", False))
    elif args.do_import:
        import_file = Path(args.import_file) if args.import_file else None
        import_tags(INDEX_PATH, RAW_PATH,
                    import_file=import_file, dry_run=args.dry_run)
    elif args.stats:
        show_stats(INDEX_PATH)
    else:
        p.print_help()
