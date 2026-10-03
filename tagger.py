#!/usr/bin/env python3
"""
tagger.py — two-layer automatic tagging for mcstories stories

Layer 1 (always runs): regex tagger
  - Decodes site shortcodes (mc, mf, hm ...) into human-readable labels
  - Detects POV via pronoun counting
  - Classifies length by word count
  - Catches explicit keyword mentions for mind_control_type, setting, themes, tone

Layer 2 (optional, requires sentence-transformers): semantic tagger
  - Embeds a sample of story text and compares to tag description vectors
  - Adds tags the regex layer misses due to oblique language
  - Only runs on the fuzzy dimensions: setting, themes, tone, mind_control_type
  - Degrades gracefully if sentence-transformers is not installed

The two layers are merged. Site shortcode tags always win if there is a conflict.
Each tag carries a _tag_sources annotation so you can see which layer fired it.

Install semantic layer:
  pip install sentence-transformers

Usage:
  python tagger.py                              # regex only (fast)
  python tagger.py --semantic                   # regex + semantic
  python tagger.py --semantic --rebuild-jsonl   # retag + regenerate training files
  python tagger.py --semantic --threshold 0.32  # tune cosine cutoff (default 0.35)
"""

from __future__ import annotations

import re
import json
import sys
import logging
from pathlib import Path
from tag_trust import export_tags
from typing import Any, Optional
import index_store

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")

def _strip_html(text: str) -> str:
    """Strip HTML tags for text analysis. No-op for legacy plain-text stories."""
    if not text or "<" not in text:
        return text or ""
    return _TAG_RE.sub(" ", text)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

REGEX_ONLY_DIMS  = {"site_tags_expanded", "site_tag_groups", "content_type",
                    "pov", "length_class", "sexual_acts", "relationship"}
SEMANTIC_EXTRA_DIMS = {"dominance", "pacing"}  # new dims handled by both layers
SEMANTIC_DIMS    = {"mind_control_type", "setting", "themes", "tone", "dominance", "pacing", "awareness", "prevalence", "mental_changes"}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — REGEX TAGGER
# ══════════════════════════════════════════════════════════════════════════════

# ── Official mcstories.com tag codes ─────────────────────────────────────────
# Source: mcstories.com/Tags/index.html — complete confirmed list of 25 tags.
# These are the ONLY real site tags. Everything else in stories_index.json
# categories that isn't in this list is either a scraping artefact or unknown.

SITE_TAG_LABELS: dict[str, str] = {
    "mc":  "mind control",                 # 17,272 — the defining tag
    "mf":  "male/female sex",             # 11,724
    "ff":  "female/female sex",           # 6,723
    "md":  "male dominant",               # 8,084
    "fd":  "female dominant",             # 4,163
    "mm":  "male/male sex",               # 1,797
    "in":  "incest",                      # 1,555
    "gr":  "growth / enlargement",        # 1,518
    "ft":  "fetish (usually clothing)",   # 1,433
    "cb":  "comic book / super-hero",     # 533
    "sf":  "science fiction",             # 699
    "ds":  "dominance and/or submission", # 815
    "hm":  "humiliation",                 # 826
    "bd":  "bondage and/or discipline",   # 807
    "ex":  "exhibitionism",               # 415
    "rb":  "robots / robotification",     # 407
    "la":  "lactation",                   # 379
    "hu":  "humor",                       # 335
    "be":  "bestiality",                  # 174
    "fu":  "furry",                       # 127
    "ws":  "watersports",                 # 64
    "ts":  "time stop",                   # 62
    "sc":  "scatology",                   # 57
    "nc":  "non-consensual",              # 32
    "ca":  "cannibalism",                 # 5
}

SITE_TAG_GROUPS: dict[str, list[str]] = {
    "sex_act":        ["mf", "ff", "mm", "ma"],
    "gender_dynamic": ["fd", "md"],
    "kink":           ["bd", "ds", "hm", "ex", "la", "ws", "nc", "in", "sc", "be"],
    "genre":          ["sf", "ft", "cb", "rb", "fu", "gr", "ts", "ca", "hu"],
    "content_flag":   ["nc"],
}


def _r(pattern: str) -> re.Pattern:
    # Normalize curly apostrophes/quotes to straight so patterns match the
    # normalized corpus regardless of how they were typed in the source file.
    pattern = pattern.replace('\u2019', "'").replace('\u2018', "'")
    return re.compile(pattern, re.IGNORECASE | re.DOTALL)

RuleList = list[tuple[str, re.Pattern, int]]

SETTING_RULES: RuleList = [
    # workplace: removed 'job' and 'career' (too generic); raised min_n 2→3
    ("workplace",           _r(r"\b(office|boss|coworker|cubicle|boardroom|secretary|corporate|manager|HR|workplace|employee|employer)\b"), 3),
    ("school / college",    _r(r"\b(school|university|college|campus|professor|student|classroom|dorm|lecture|teacher|class|semester)\b"), 2),
    ("home / domestic",     _r(r"\b(home|house|apartment|kitchen|bedroom|husband|wife|roommate|neighbor|domestic|household|married|marriage)\b"), 3),
    ("medical",             _r(r"\b(doctor|hospital|clinic|nurse|patient|therapist|therapy|hypnotherapist|psychiatrist|session|treatment|medical)\b"), 2),
    ("fantasy / historical",_r(r"\b(castle|kingdom|dragon|elf|sword|tavern|medieval|ancient|potion|quest|realm|sorcerer|knight)\b"), 2),
    ("science fiction",     _r(r"\b(spaceship|alien|android|future|galaxy|planet|laser|nanobot|AI|cyberpunk|dystopia|colony|starship)\b"), 2),
    # supernatural: removed 'spirit', 'mystical', 'ritual' (fire on non-supernatural contexts); kept creature/faction words
    ("supernatural",        _r(r"\b(vampire|werewolf|demon|ghost|haunted|occult|paranormal|coven)\b"), 2),
    ("gym / sports",        _r(r"\b(gym|workout|trainer|exercise|locker.?room|weights|fitness|coach|athlete|team|sport|cheerleader)\b"), 2),
    ("bar / club / party",  _r(r"\b(bar|nightclub|party|dance.?floor|drinks|cocktail|drunk|club|rave|concert|venue|social)\b"), 2),
    ("online / virtual",    _r(r"\b(online|chat|website|avatar|VR|virtual.?reality|internet|screen|app|discord|forum|game|stream)\b"), 2),
]

MIND_CONTROL_TYPE_RULES: RuleList = [
    # Svengali-aligned taxonomy
    ("hypnosis",              _r(r"\b(hypno|hypnot|pendulum|pocket.?watch|post-hypnotic|look into my eyes|you are getting sleepy)\b"), 1),
    ("pheromones",            _r(r"\bpheromone\w*\b"
                               r"|\bmusk\b.{0,60}(aroused?|arousal|intoxicat\w*|overwhelm\w*|cloud\w*.{0,20}(mind|thought|think)|obey|comply)"
                               r"|(scent|smell|aroma|fragrance|perfume).{0,80}(cloud\w*.{0,20}(mind|thought|think)|intoxicat\w*|overwhelm\w*.{0,15}(sense|mind)|couldn.?t think|obey|comply|compel)"
                               r"|(obey|comply|compel|intoxicat\w*|overwhelm\w*|cloud\w*).{0,80}(scent|smell|aroma|fragrance|perfume)"), 1),
    ("psionic",               _r(r"\b(psion|psychic|telepathy|telekinesis|mind.?read|mental.?power|esp|sixth sense|empathic)\b"), 1),
    # chemical: broad patterns — AND-gated with SetFit for precision; standalone was 37% prec
    ("chemical",          _r(r"\bdrug(ged|ging)?\b"
                             r"|\bpotion\b|\bserum\b"
                             r"|\bchemi(cal|cally)\b"
                             r"|\bsubstance\b.{0,30}(control|compel|affect|alter|mind)"
                             r"|\bdose(d|ing)?\b.{0,20}(her|him|them)"
                             r"|\btoxin\b|\bnarcotic\b"), 2),
    ("subliminal",            _r(r"\b(subliminal|reprogramm|brainwash|hidden message|earworm)\b"), 1),
    ("technology",            _r(r"\b(mind.?control (device|chip|implant|machine|collar|headset|app)"
                               r"|brain.?(chip|implant|interface|hack)"
                               r"|neural (implant|chip|interface|override|link|jack)"
                               r"|control (chip|implant|device|collar|headset|module)"
                               r"|remote.?control(led)?.{0,15}(mind|brain|behavior|her|him)"
                               r"|behavioral (modification|programming) (device|tech|implant)"
                               r"|nanotechnology|nanobots?.{0,20}(control|brain|mind|obey)"
                               r"|virtual reality.{0,30}(control|program|trap|stuck|mind)"
                               r"|subliminal (audio|video|signal|message|frequen)"
                               r"|mind.?hack(ing|ed)|electrodes?.{0,20}brain)\b"), 1),
    ("nanotechnology",        _r(r"\b(nano|nanobot|nanite|microscopic|injected.*bloodstream|bloodstream.*(obey|control))\b"), 1),
    ("virtual reality",       _r(r"\b(virtual.?reality|vr headset|simulation|simulated|matrix|digital.?world)\b"), 1),
    ("virus",                 _r(r"\b(pathogen|contagion)\b"
                               # biological/MC virus: require MC-specific words close by,
                               # short window (80 chars) to avoid matching computer-virus plots
                               r"|\bvirus\b.{0,80}(obey|comply|compel|submit|trance|mind control|thought|arousal|desire|lust|docil)"
                               r"|(obey|comply|compel|submit|trance|arousal|desire|lust|docil).{0,80}\bvirus\b"
                               r"|(infected|infection).{0,80}(obey|comply|compel|behav\w*|mind|brain|trance|arousal|desire)"
                               r"|(obey|comply|compel|behav\w*|trance|arousal|desire).{0,80}(infected|infection)"
                               # explicit MC-virus compounds unambiguous regardless of context
                               r"|\b(mind.?virus|lust.?virus|obedience.?(virus|strain|pathogen)|"
                               r"compliance.?(virus|pathogen)|arousal.?virus|nanite.?virus|rage.?virus)\b"), 1),
    # magic: AND-gated with SetFit; standalone was 51% prec (fantasy setting ≠ magic MC)
    ("magic",             _r(r"\bspell\b|\bwitch\b|\bwizard\b|\bsorcery\b|\bincantation\b"
                             r"|\britual\b"
                             r"|\benchant(ed|ment|ing)?\b"
                             r"|\bcurse(d)?\b.{0,30}(mind|will|control|obey)"
                             r"|\bmagic(al)?\b.{0,30}(control|obey|compel|power|enslave|bind)"), 2),
    ("reality manipulation",  _r(r"\b(reality (bend|warp|alter|shift|rewrite|rewrit|manipulat)"
                               r"|(bend|warp|alter|rewrite|manipulat).{0,15}reality"
                               r"|rewrote reality|reality itself (changed|bent|shifted|altered)"
                               r"|(genie|djinn|wish).{0,30}(grant|granted|came true|reality)"
                               r"|alter(ed|ing) (the |their )?(world|reality|existence|timeline)"
                               r"|changed.?reality|reality.?altering)\b"), 1),
    # time stop: rare tag — only fire on unambiguous time-manipulation phrases.
    # "everyone froze / stopped" excluded: matches fear/shock/command paralysis, not time stop.
    # "froze.*time" / "pause.*time" excluded: too loose, matches metaphors and body-freeze.
    ("time stop",             _r(r"\b(time.?stop(?:ped|ping|s)?|time.?freeze|time.?froze|freeze.?time|frozen.?in.?time)\b"
                               r"|\btime\s+(stopped|froze|stood still|was frozen|has stopped)\b"
                               r"|\b(stopped|paused)\s+time\b"
                               r"|\bworld\s+(froze|frozen|stopped|paused)\s+around\b"), 1),
]

SEXUAL_ACT_RULES: RuleList = [
    # These are unambiguous explicit terms — 1 match is sufficient.
    # anal: AND-gated with SetFit; standalone was 46% prec
    ("anal",          _r(r"\banal\b|\bsodomiz\w*\b"
                         r"|\bass(hole|fuck|sex)\b"
                         r"|\bbackdoor\b.{0,20}(sex|penetrat|entry|passage)"
                         r"|\brear (entry|penetration|passage)\b"
                         r"|\bbutt (sex|fuck)\b"), 1),

    ("oral",          _r(r"\b(blow.?job|blowjob|fellatio|cunnilingus|deepthroat\w*|face.?fuck\w*"
                         r"|oral sex"
                         r"|going.?down.?on|went.?down.?on|goes.?down.?on|get.?down.?on|kneel\w*.{0,20}(cock|dick|him)"
                         r"|suck(ed|ing)?.{0,5}(his|her|my|the).?(cock|dick|length|shaft|tip|head)"
                         r"|suck\b.{0,5}(cock|dick)\b"
                         r"|lick(ed|ing).?(his|her|my).?(cock|pussy|clit|folds|slit|lips)"
                         r"|tongue.{0,20}(his|her).{0,10}(cock|dick|pussy|clit|shaft|folds|slit|tip|head)"
                         r"|gave?.{0,10}(him|her)\s+head\b|getting\s+head\b|giving\s+head\b"
                         r"|mouth.{0,10}(on|around|over).{0,10}(his|her|the).{0,10}(cock|dick|length|shaft|pussy|clit)"
                         r"|took.{0,10}(him|it|his.?(cock|dick|length)).{0,20}(in|into).{0,10}(her|his|my).?mouth"
                         r"|(cock|dick).{0,10}(in|into|inside).{0,5}(her|his|my|your).{0,5}mouth"
                         r"|lips.{0,10}(around|wrapped.{0,10}around|on).{0,10}(his|her).{0,10}(cock|dick|shaft|pussy)"
                         r"|swallow(ed|ing).{0,20}(cum|come|seed|load|every drop)"
                         r"|ate?.{0,10}(her|him).{0,10}out\b"
                         r"|eat(ing)?.{0,10}(her|him).{0,10}out\b"
                         r"|her.?mouth.{0,10}(on|around).{0,10}(him|his|it|the.?cock|the.?shaft))\b"), 1),

    ("masturbation",  _r(r"\b(masturbat\w*|jerk(ed|ing).?off|jack(ed|ing).?off"
                         r"|rub(bed|bing).{0,10}(her|his).{0,10}(clit|pussy|cock|dick|self)"
                         r"|rub(bed|bing).{0,5}(herself|himself)\b"
                         r"|strok(ed|ing).{0,10}(him.?self|his.?(cock|dick|length|shaft))"
                         r"|strok(ed|ing).{0,10}(her.?(clit|pussy|slit|sex|folds|core))"
                         r"|finger(ed|ing).{0,10}(her.?self|him.?self)"
                         r"|play(ed|ing).{0,10}with.{0,5}(her.?self|him.?self|her.?(pussy|clit|sex|slit)|his.?(cock|dick))"
                         r"|pleasur(ed|ing).{0,10}(her.?self|him.?self)"
                         r"|brought?.{0,10}(her.?self|him.?self).{0,10}(off|to.?orgasm|to.?climax)"
                         r"|hand.{0,5}(between|inside|down).{0,20}(her|his).{0,10}(legs|thighs|pants|underwear))\b"), 2),

    # exhibitionism removed — too context-dependent for regex; covered by SetFit + Svengali
    # voyeurism removed — context-dependent; covered by SetFit + Svengali

    # toys: AND-gated with SetFit; standalone was 31% prec (toy mention ≠ toy used sexually)
    ("toys",          _r(r"\bvibrat(or|ing|ed|e)\b|\bdildo\b|\bbutt plug\b|\bsex toy\b"
                         r"|\btoy(s)? (inserted|used|placed|inside|in her|in him)\b"
                         r"|\bplug(ged)? (into|inside|her|him)\b"
                         r"|\bwand\b.{0,20}(used|pressed|applied|against|vibrat)"), 2),

    # cuckold removed — context-dependent; covered by SetFit + Svengali

    # impregnation removed — 39% audit precision; left to SetFit/LLM

    ("fingering",     _r(r"\b(finger(ed|ing).{0,20}(her.?pussy|her.?clit|her.?folds|her.?slit|her.?sex|her.?core|inside.?her|her.?wet|her.?entrance)"
                         r"|finger(ed|ing).{0,5}(her.?self|him.?self)"
                         r"|slid?.{0,10}(a|her|his|two|three).{0,5}finger\w*.{0,15}(into|inside|between).{0,10}(her.?(pussy|clit|folds|slit|sex|core|entrance|opening)|him.?(ass|entrance))"
                         r"|thrust(ed|ing).{0,10}(a|his|her|two|three).{0,5}finger\w*.{0,10}(into|inside).{0,10}(her.?(pussy|clit|folds|slit|entrance)|him.?(ass|entrance))"
                         r"|push(ed|ing).{0,10}(a|his|her|two|three).{0,5}finger\w*.{0,10}(into|inside).{0,10}(her.?(pussy|clit|folds|slit|entrance|opening)|him.?(ass|entrance))"
                         r"|curled?.{0,10}(a|his|her).{0,5}finger\w*.{0,10}(inside|within).{0,10}(her.?(pussy|clit|folds|slit)|him.?(ass))"
                         r"|two.?fingers.{0,10}(inside|into|within).{0,10}(her.?(pussy|slit|core|entrance)|his.?ass)"
                         r"|three.?fingers.{0,10}(inside|into|within).{0,10}(her.?(pussy|slit|core|entrance)|his.?ass)"
                         r"|fingers?.{0,10}(buried|pumping|working).{0,10}(inside|within).{0,10}(her.?(pussy|slit|core|folds)|him.?(ass))"
                         r"|thumb.{0,15}(against|on|over|inside|at).{0,10}(her|his).{0,10}(clit|pussy|entrance|folds|slit|opening))\b"), 1),

    ("rimming",       _r(r"\b(rimm(ed|ing)(?!.{0,15}glass)|rim.?job|analingus"
                         r"|lick(ed|ing).{0,20}(her.?ass|his.?ass|her.?anus|his.?anus|between.{0,10}(her|his).?cheeks|her.?rear|his.?rear)"
                         r"|lick\w*.{0,15}(her|his).{0,10}hole\b"
                         r"|tongue.{0,20}(her.?ass|his.?ass|her.?anus|his.?anus|her.?rear|his.?rear|between.{0,10}(her|his).?cheeks)"
                         r"|tong(ue|uing).{0,10}(her|his).{0,10}(hole|rear|ass|anus|rosebud))\b"), 1),

    # creampie: AND-gated with SetFit; standalone was 18% prec
    ("creampie",      _r(r"\bcreampie\b"
                         r"|\bcum(ming)? inside\b"
                         r"|\bseed(ed|ing)? (her|him|them|her womb)\b"
                         r"|\bfill(ed|ing)? (her|him|them) (up|with cum|with seed)\b"), 1),

    # striptease removed — 4% audit precision; dropped entirely
    # titfuck removed — 14% audit precision; dropped entirely

    # facial: AND-gated with SetFit; standalone was 32% prec
    ("facial",        _r(r"\bcumshot\b"
                         r"|\bcum.{0,10}(on|across|covering).{0,20}(face|cheek|chin|mouth)\b"
                         r"|\b(her|his).{0,10}face.{0,20}(cum|covered|dripping|coated)\b"), 1),

    ("group sex",     _r(r"\b(group.?sex|threesome|foursome|orgy|gang.?bang|gangbang"
                         r"|double.?penetration|dp\b.{0,10}(her|him|sex|penetrat)"
                         r"|spit.?roast\w*|double.?team\w*"
                         r"|taking.?turns.{0,20}(fucking|using|with|on).{0,10}(her|him)"
                         r"|passed.?around.{0,20}(the|between).{0,10}(men|guys|them|women))\b"), 2),
]

THEME_RULES: RuleList = [
    # Narrative arc
    ("slow burn",              _r(r"\b(gradual|slowly|over time|weeks? later|months? later|resist(ance|ing)|reluctant|inch by inch|day after day)\b"), 2),
    # corruption arc: removed 'becoming' and bare 'transform' (too generic); require personal subject/change context
    ("corruption arc",         _r(r"\b(corrupt\w*.{0,25}(her|him|them|mind|soul|self|values|morals|person)"
                                   r"|degrad\w*"
                                   r"|devolv\w*"
                                   r"|no longer (herself|himself)"
                                   r"|bimbo(ize|ification)"
                                   r"|transform\w*.{0,30}(into|her|him|something else|stranger|new person))\b"), 2),
    ("stepfordization",        _r(r"\b(stepford|perfect (wife|housewife|woman)|obedient.?wife|happy.?homemaker|smile.?and.?obey|prim and proper)\b"), 1),
    ("slave mentality",        _r(r"\b(slave|slavery|owned|property|master's|your property|serve and obey|exist to serve|born to serve|only purpose)\b"), 2),
    # bimboization: removed 'ditzy' and 'giggly' (too common in general fiction); raised min_n 1→2
    ("bimboization",           _r(r"\b(bimbo|airhead|dumb(er|ing down)|valley girl|can't think|thinking.?is.?hard|empty.?headed)\b"), 2),
    ("drone / hive mind",      _r(r"\b(drone|hive|collective|assimilat|one of us|join us|resistance is futile|borg|unit|node|link)\b"), 1),
    ("induced love",           _r(r"\b(fell in love|love (him|her) now|can't help.?loving|heart.*(belong|his|hers)|devoted(ly)?|adore.?him|worship.?her)\b"), 2),
    ("harem",                  _r(r"\b(harem|collection|another one|adding (her|him)|joining the others|stable|his (girls|women|slaves))\b"), 1),
    # serial recruitment: removed spread/network/growing (too generic); tightened convert/recruit to require new targets
    ("serial recruitment",     _r(r"\b(one by one"
                                   r"|next (target|victim|recruit)"
                                   r"|convert(ing|ed).{0,30}(more|them|others|girl|woman|women|men)"
                                   r"|recruit(ing|ment).{0,20}(more|new|others|victim|target))\b"), 1),
    ("obsession",              _r(r"\b(obsess|fixat|can't stop thinking|consumed by|haunt(ed)?|compulsion|possessive|fixation)\b"), 2),
    # betrayal: 70% audit precision — kept as regex. Fires on interpersonal deception that is central to the MC plot.
    ("betrayal",               _r(r"\b(betray|backstab)\b"
                                   r"|\bdeceiv\w+.{0,30}(her|him|them|me)\b"
                                   r"|\btricked?.{0,30}(her|him|them|me|into)\b"
                                   r"|\blied.{0,20}(to|about).{0,20}(her|him|them|me)\b"), 1),
    ("revenge",                _r(r"\b(revenge|payback|get back at|punish|deserve it|retribution|settle the score|get even)\b"), 2),
    ("power corrupts",         _r(r"\b(power.*(corrupt|change|gone too far)|drunk on power|intoxicated by|losing (himself|herself|control)|couldn't stop|addicted to control)\b"), 1),
    ("reluctant subject",      _r(r"\b(fight it|didn't want|tried to refuse|fought against|willpower|defiance|valiant|futile.?struggle)\b"), 2),
    ("willing subject",        _r(r"\b(wanted to be (hypnotized|controlled|enslaved|dominated|used)|"
                                   r"asked (him|her) to (hypnotize|control|take over|enslave)|"
                                   r"chose to (submit|obey|surrender|be controlled)|"
                                   r"volunteered|consented to|agreed to be|"
                                   r"always wanted this|fantasized about being controlled|"
                                   r"eager to be (controlled|used|hypnotized|enslaved))\b"), 2),
    ("conspiracy",             _r(r"\b(conspiracy|secret.?organisation|secret.?program|government|they.?knew|cover.?up|operation|experiment)\b"), 2),
    ("celebrity",              _r(r"\b(celebrity|famous|star|actor|actress|singer|pop star|idol|public figure)\b"), 2),
    ("romance",                _r(r"\b(love|romance|feel(ings?)|heart|kiss|tender|intimate|affection|companion)\b"), 3),
    ("horror / dark",          _r(r"\b(horror|dark|terrif|fear|scream|blood|dread|nightmare|doom|trap|sinister)\b"), 2),
    ("consent / ethics",       _r(r"\b(consent|ethics?|moral|wrong|right|guilty|shame|justify|deserve)\b"), 2),
    ("identity loss",          _r(r"\b(identity|who (am|are|was) I|memories|forget|personality|no longer herself|no longer himself"
                                   r"|not a person|not people|not human|dehumaniz|you are (just )?a (thing|beast|pet|animal|doll|slave|toy|object)"
                                   r"|she is (just )?a (thing|beast|pet|animal|doll|slave|toy|object)"
                                   r"|he is (just )?a (thing|beast|pet|animal|doll|slave|toy|object)"
                                   r"|just an? (thing|beast|pet|animal|doll|toy|object)"
                                   r"|no longer (a )?human|stripped (of|away).{0,20}(self|identity|name|who))\b"), 3),
    ("trigger phrase",         _r(r"\b(trigger|phrase|keyword|command.?word|activation|password|signal|codeword|when.?I.?say)\b"), 2),
    ("tentacles",              _r(r"\b(tentacle|tendril|appendage|squid|octopus|wrapped around|coiled)\b"), 1),
    ("mythology / religion",   _r(r"\b(god|goddess|deity|divine|myth|legend|ancient|cult|worship|sacrifice|temple|prophet|chosen)\b"), 2),
    ("rescue",                 _r(r"\b(rescue|save (her|him|them)|break.?the.?spell|free.?(her|him)|snap out|wake up|counter.?program)\b"), 2),
    ("time travel",            _r(r"\b(time.?travel|time.?travell|travelled back in time|future self|past self|paradox|changed.?history|alternate timeline)\b"), 1),
    ("found family",           _r(r"\b(found family|chosen family|our (little )?family|they.{0,10}family now|her family now|his family now|new family|become (a )?family|family of (their|her|his) own|sisterhood|brotherhood|sense of belonging)\b"), 2),
]

# ── Fetish tags (keyword-definitive subset only) ──────────────────────────────
# Svengali covers ~14,800 stories — these rules fill in the remaining ~2,800.
# Only adding tags where patterns are unambiguous and high-precision.

FETISH_RULES: RuleList = [
    ("bdsm",              _r(r"\b(bdsm"
                             r"|bondage"
                             r"|handcuff\w*"
                             r"|hogti\w*"
                             r"|ball.?gag|mouth.?gag|bit.?gag"
                             r"|rope.{0,15}(tied|bound|restrained|around.{0,10}wrist)"
                             r"|tied.{0,10}(to.{0,10}(bed|chair|post|frame)|spread.?eagle)"
                             r"|wrist.?cuff|ankle.?cuff|spreader.?bar)\b"), 1),

    ("sadomasochism",     _r(r"\b(masoch\w*"
                             r"|spanking|spanked"
                             r"|flogg\w*"
                             r"|paddl\w*.{0,20}(butt|ass|bottom|backside)"
                             r"|whip(ped|ping).{0,20}(her|him|back|ass|butt|body)"
                             r"|crop(ped|ping).{0,20}(her|him|butt|ass|thigh))\b"), 1),

    ("fetish clothing",   _r(r"\b(latex|catsuit"
                             r"|rubber.{0,10}(suit|dress|outfit|catsuit|gloves|mask)"
                             r"|corset\w*"
                             r"|fishnet.{0,10}(stocking|tights|pantyhose)"
                             r"|thigh.?high.{0,10}stock"
                             r"|garter.?belt"
                             r"|pvc.{0,10}(dress|outfit|suit|catsuit))\b"), 1),

    ("body decoration",   _r(r"\b(tattoo\w*|tatted\b|inked\b"
                             r"|pierc\w*.{0,20}(nipple|navel|clit|belly|tongue|nose|lip\b)"
                             r"|nipple.{0,10}pierc"
                             r"|bodypaint\w*|body.?paint\w*)\b"), 1),

    ("virgin",            _r(r"\b(hymen|deflower\w*"
                             r"|pop(ped|ping).{0,10}her.{0,10}cherry"
                             r"|took?.{0,10}(her|his).{0,10}virginity"
                             r"|losing.{0,10}(her|his).{0,10}virginity)\b"), 1),

    ("lactation",         _r(r"\b(lactat\w*"
                             r"|milk(ed|ing).{0,20}(breast|nipple|tit)"
                             r"|breast.?milk|nipple.?milk"
                             r"|milk.{0,15}(flow|drip\w*|leak\w*).{0,20}(breast|nipple|tit)"
                             r"|engorged?.{0,20}with.{0,5}milk)\b"), 1),

    ("feet",              _r(r"\b(foot.?fetish|feet.?fetish|foot.?worship|toe.?worship"
                             r"|worship(p?ed|p?ing).{0,20}(her|his).{0,10}feet"
                             r"|lick(ed|ing).{0,20}(her|his).{0,10}(feet|toes?|soles?)"
                             r"|smell(ed|ing).{0,20}(her|his).{0,10}(feet|toes?|soles?)"
                             r"|suck(ed|ing).{0,20}(her|his).{0,10}toes?"
                             r"|(her|his).{0,10}(feet|toes?|soles?).{0,20}(in|into).{0,10}(his|her|my).{0,10}(mouth|face))\b"), 1),

    ("costume",           _r(r"\b(french.?maid.{0,20}(outfit|costume|uniform|dress)"
                             r"|(maid|cheerleader|schoolgirl|nurse).{0,20}(outfit|costume|uniform)"
                             r"|wearing.{0,20}(a.{0,5})?(maid|cheerleader|schoolgirl|nurse).{0,10}(outfit|costume|uniform))\b"), 1),
]

# ── Application / dominance style (Svengali "Application" + "Dominance") ──────

DOMINANCE_RULES: RuleList = [
    ("gentle",             _r(r"\b(gentle|tender|soft|kind|patient|nurtur|careful|lovingly|softly|easy|with care)\b"), 3),
    ("brutal",             _r(r"\b(brutally (treated|used|punished|broken|trained|conditioned|forced|hurt|violated)"
                               r"|sadistic|sadism|without mercy.{0,20}(train|punish|condit|control)"
                               r"|(whipped|flogged|caned|beaten|struck).{0,20}(control|train|obey|punish)"
                               r"|breaking (her|him|you).{0,20}(will|spirit|resistance|mind)"
                               r"|cruel(ty)?.{0,20}(master|owner|dominant|controller|mistress))\b"), 1),
    ("self-inflicted",     _r(r"\b(self.?hypno(sis|tize|tizing|tized)?"
                               r"|hypnotized? (herself|himself|themselves)"
                               r"|put (herself|himself) (under|in a trance|to sleep|into trance)"
                               r"|did it (to herself|to himself)|sought.{0,15}(control|enslave|hypnot|program)"
                               r"|volunteer(ed|ing).{0,20}(subject|slave|test)"
                               r"|wanted to be (controlled|hypnotized|programmed|enslaved|owned)"
                               r"|chose to be (controlled|hypnotized|enslaved|owned|blank)"
                               r"|willing(ly)? (submit|surrender|enter|accept).{0,15}(control|trance|hypno)"
                               r"|accidental(ly)?.{0,20}(hypnotize|trigger|put (herself|himself)))\b"), 1),
    ("loving control",     _r(r"(for (her|his|your) own good"
                                r"|because (he|she) loved (her|him)"
                                r"|out of love"
                                r"|loved (her|him) too much"
                                r"|(his|her) love for (her|him)"
                                r"|lovingly (control|program|condition|hypnoti|train|guid)"
                                r"|wanted (only )?(to )?(protect|care for|look after|keep.{0,10}safe)"
                                r"|(cared|caring) for (her|him|you) deeply"
                                r"|(gentle|loving) (master|owner|controller|dominant))"), 1),
    # cold/clinical: dropped 'asset' (fires on financial uses); clinical gets negative
    # lookahead to exclude medical collocates (depression, treatment, review, etc.)
    ("cold / clinical",    _r(r"\b(clinical(?!\s+(depression|trial|review|treatment|diagnosis"
                               r"|condition|care|study|health|staff|setting))"
                               r"|detach(ed|ment)|dispassionate|specimen"
                               r"|cold.?(efficiency|calculation|precision|logic|manner|voice|tone))\b"), 3),
    ("manipulative",       _r(r"\b(manipulat|deceiv|trick|gaslight|scheming|subtle manipulation|pull.?strings|puppet)\b"), 2),
    ("sadistic",           _r(r"\b(sadis|cruel|enjoy.*(pain|suffering|tears)|pleasure from|delight in|torment|humiliat)\b"), 2),
]

# ── Pacing (Svengali "Style and Pacing") ──────────────────────────────────────

PACING_RULES: RuleList = [
    ("leisurely",          _r(r"\b(over (weeks|months|years|time)|day by day|little by little|leisurely|unhurried"
                               r"|slowly (began|developed|realized|changed|grew|worked)"
                               r"|(weeks|months) (passed|went by|later)"
                               r"|gradual (process|change|transformation|conditioning))\b"), 4),
    # episodic: requires explicit story-structure markers; includes spelled-out ordinals and roman numerals
    ("episodic",           _r(r"\b(chapter (one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+|[ivxlIVXL]{1,6})\b"
                               r"|part (one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\b"
                               r"|episode [0-9]+\b"
                               r"|to be continued"
                               r"|instalment"
                               r"|each (session with|visit to|time she (came|returned|arrived)|time he (came|returned|arrived)))\b"), 2),
    ("single scene",       _r(r"\b(afternoon|evening|that night|one day|an hour|the session|the evening|single encounter)\b"), 2),
]

# ── Pacing from document structure ────────────────────────────────────────────

# Precompile patterns used by infer_pacing_from_structure
_RAPID_PAT = re.compile(
    r'\b(instantly|immediately|right away|at once|'
    r'in an instant|within (seconds|minutes|moments)|'
    r'snapped (her|his|their) fingers|with a snap|'
    r'just like that|like a switch|'
    r'at the snap|first (session|meeting|touch|glance|word))\b',
    re.I,
)
_TIME_SKIP_PAT = re.compile(
    r'\b(weeks? later|months? later|years? later|'
    r'the next (week|month|year)|several (days|weeks|months|years)|'
    r'over (the next|many|several) (days?|weeks?|months?|years?)|'
    r'by the time|as the (days|weeks|months) (passed|went by)|'
    r'a month had|a year had|six months|three months)\b',
    re.I,
)


def infer_pacing_from_structure(story: dict) -> list[str]:
    """
    Infer pacing tags from document-level structure rather than keyword matches.
    Covers 'rapid' and 'epic' (not reachable by short-window regex) and supplements
    'leisurely' with time-skip density.  Returns a list of matched tags.
    """
    text = _strip_html(story.get("full_text", "") or "")
    wc   = story.get("word_count", 0) or len(text.split())
    chs  = story.get("chapter_count", 1) or 1

    if not text or wc < 100:
        return []

    tags = []

    # ── Epic: very long multi-chapter works ───────────────────────────────────
    if (wc > 40_000 and chs >= 5) or wc > 65_000:
        tags.append("epic")

    # ── Rapid: short story where control is established immediately ───────────
    rapid_hits = len(_RAPID_PAT.findall(text))
    if wc < 3_000 and rapid_hits >= 3:
        tags.append("rapid")
    elif wc < 6_000 and rapid_hits >= 5:
        tags.append("rapid")
    elif rapid_hits >= 8:   # even long stories if saturated with instant-control signals
        tags.append("rapid")

    # ── Leisurely supplement: high time-skip density ──────────────────────────
    # Regex catches exact phrases; this catches density across the whole text.
    skips = len(_TIME_SKIP_PAT.findall(text))
    if wc > 2_000 and skips / max(wc / 1_000, 1) >= 1.5:
        tags.append("leisurely")

    return tags


# ── Reaction / Awareness (Svengali "Reaction") ────────────────────────────────
# "playing along" and "skeptical / confused" dropped — require full narrative arc; SetFit handles them.

AWARENESS_RULES: RuleList = [
    # min_n=1: specific multi-word phrases; a single match is meaningful.
    # "playing along" and "skeptical / confused" dropped — require narrative arc; SetFit handles them.

    ("valiant but futile struggle",
        _r(r"(mind (screamed|cried|begged|fought|protested|raged|pleaded)|"
           r"trapped inside (her|his|their)|"
           r"still (herself|himself) inside|"
           r"tried to (resist|fight|break free|break out|push back) but|"
           r"fought (against|back) but|"
           r"screamed (inside|silently|within)|"
           r"part of (her|him) (still|was) (fighting|resisting|screaming))"), 2),

    ("denial",
        _r(r"(not under (control|hypnosis|his|her)|thinking clearly|perfectly fine|"
           r"nothing wrong|in control|not affected|immune to|won't work on|"
           r"can't be hypnotized|too strong.?willed)"), 2),

    ("oblivious",
        # removed 'her/his own idea/choice/decision' (fires on consensual stories)
        # kept bare 'had no idea' and 'never knew/realized/suspected' — these are specific enough in MC context
        _r(r"(had no idea"
           r"|never (knew|realized|suspected)"
           r"|completely (unaware|oblivious)"
           r"|had no (memory|recollection))"), 1),

    ("gradually losing awareness",
        _r(r"(slowly (forgot|lost)|stopped (remembering|questioning)|"
           r"(memories|sense of self) (faded|slipped|eroded)|"
           r"no longer (remembered|questioned)|used to (know|think|believe))"), 1),

    ("accepting",
        _r(r"(gave in|gave herself|gave himself|stopped (fighting|resisting)|"
           r"surrender(ed|ing)|chose to (obey|submit|comply)|"
           r"no longer (cared|resisted|fought))"), 2),

    ("consensual",
        _r(r"(asked (him|her) to (hypnotize|control|take control)|"
           r"wanted to be controlled|consented? to|agreed to give up|"
           r"chose to be controlled|willingly submitted)"), 1),
]

# ── Prevalence / scale (Svengali "Prevalence") ────────────────────────────────

PREVALENCE_RULES: RuleList = [
    ("single target",      _r(r"\b(only (her|him|you)|just (her|him|one)|singled out|chosen|specific target|you alone)\b"), 2),
    ("small group",        _r(r"\b(a few|several|group|them all|the (women|men|girls|boys)|one by one|each of them|the whole (office|class|team))\b"), 2),
    ("widespread",         _r(r"\b(everyone|the whole (city|town|world|country)|spreading|contagious|pandemic|mass|population|nation)\b"), 2),
    ("secret / obscure",   _r(r"\b(secret|hidden|no one knows|underground|covert.?operation|shadow|in the dark|unknown to (most|all|others))\b"), 2),
]

# ── Mental changes (Svengali "Mental Manipulation") ───────────────────────────

MENTAL_CHANGE_RULES: RuleList = [
    ("bimboization",         _r(r"\b(bimbo|airhead|dumb(er|ing)|can't think|thinking.?is.?hard|empty.?headed|giggly|ditzy)\b"), 1),
    ("intelligence change",  _r(r"\b(smarter|dumber|IQ|intelligence|mental.?capacity|cognitive|heightened|lowered|enhanced.?mind)\b"), 2),
    ("memory alteration",    _r(r"\b(amnesia|blank.?out|wipe.?(her|his|the).?memor|can't recall|false memor|implanted memor|eras\w*.{0,15}(her|his|the|all).{0,10}memor|memor(y|ies).?wipe|wiped?.?clean)\b"), 1),
    ("emotion control",      _r(r"\b(feel (happy|joy|love|content|calm|peace|bliss)|emotions.*(control|manipulat|adjust|program)|feel nothing|emotionless|detached)\b"), 2),
    ("obedience programming",_r(r"\b(obey|obedient|compliance|must (do|follow|obey)|programmed to|compelled to|cannot refuse|trained to|conditioned to)\b"), 3),
    ("addiction / need",     _r(r"\b(addicted|addiction|need(s)? (him|her|it)|craving|withdrawal|can't live without|depend(ent|ence)|obsessive need)\b"), 2),
    ("orgasm control",       _r(
                               # "don't / can't / not allowed to cum without/unless/until permission"
                               r"(don.?t|cannot|can.?t|not allowed to|may not|must not|forbidden to|not permitted to|you don.?t get to)"
                               r".{0,20}(cum|come|orgasm|climax).{0,30}(without|unless|until|permission|told|allowed|permitted|I say|given permission)"
                               # "cum only when I/you say/allow/permit, on command"
                               r"|(cum|come|orgasm|climax).{0,20}only.{0,20}"
                               r"(when (I|you).{0,10}(say|allow|permit|tell|give permission|decide)|on (my |your )?command|when (told|permitted|allowed|given permission))"
                               # "you may only cum when …" / "only I decide when you cum"
                               r"|(you may only|only I (decide|say|determine|control) when).{0,30}(cum|come|orgasm|climax)"
                               # "ask/beg (me/for) permission to cum/come"
                               r"|(ask|beg).{0,20}(me |for )?(permission|to be allowed|to cum|to come|to orgasm|to climax)"
                               r".{0,30}(cum|come|orgasm|climax|release)"
                               # "I decide when you cum/come/orgasm"
                               r"|I (decide|say|control|determine).{0,20}when.{0,20}(you )?(cum|come|orgasm|climax)"
                               # explicit denial with context
                               r"|denied.{0,5}(an? )?(orgasm|release|climax).{0,30}(unless|until|without|permission|command|I say)"
                               r"|(withhold|withheld|withholds).{0,20}(orgasm|release|climax)"
                               r"|(ruined|ruin).{0,5}(her|his|your).{0,5}(orgasm|climax)"
                               r"|no (release|orgasm|climax).{0,20}(without|unless|until).{0,20}(permission|told|I say|allowed|command)"
                               # edging with denial
                               r"|edg(ed|ing).{0,30}(denied|permission|command|without (release|coming|cumming|climax))"
                               # "kept from coming/on the edge until …"
                               r"|kept (her|him|you) (from coming|on the edge|from cumming|from (an? )?orgasm|from climax)"
                               r".{0,30}(until|unless|permission|command|I say)"
                               ), 1),
    ("speech modification",  _r(r"\b(speak.*(only|like|as)|refer to (herself|himself) as|third.?person|no longer.*(say|speak|talk)|restricted (topics?|words?)|mute|silent)\b"), 2),
    ("induced love",         _r(r"\b(fell in love|love (him|her) now|can't help.?loving|heart.*(belong|his|hers))\b"), 1),
    ("stepfordization",      _r(r"\b(stepford|perfect (wife|housewife|woman)|obedient.?wife|happy.?homemaker|smile.?and.?obey)\b"), 1),
]

TONE_RULES: RuleList = [
    ("dark / disturbing",  _r(r"\b(dark|disturb|brutal|violent|horror|grim|bleak|sinister|twisted)\b"), 2),
    ("romantic / sweet",   _r(r"\b(love|tender|sweet|gentle|warm|caring|affectionate|cuddle|cherish)\b"), 3),
    ("humorous / playful", _r(r"\b(laugh|humor|joke|comedy|absurd|ridiculous|funny|silly|ironic)\b"), 2),
    ("erotic / explicit",  _r(r"\b(moan|thrust|climax|orgasm|aroused|erotic|explicit|sensual|pleasure)\b"), 2),
    ("psychological",      _r(r"\b(mind|psyche|thought|cognitive|conscious|aware|perceive|realize|introspect)\b"), 4),
    ("melancholic",        _r(r"\b(sad|grief|loss|longing|wistful|hollow|empty|tears|mourn|bittersweet)\b"), 2),
    ("tense / suspenseful",_r(r"\b(tense|suspense|dread|danger|risk|caught|panic|escape|caught|watch)\b"), 2),
]


RELATIONSHIP_RULES: RuleList = [
    # Relationship between controller and subject — requires both roles to appear
    # or possessive pronouns attaching them to each other. min_n=2 ensures the
    # relationship is central, not a passing mention.

    ("teacher / student",   _r(r"\b(her|his|my|the)\s+(teacher|professor|instructor|prof|lecturer|tutor)\b"
                               r"|(professor|teacher|instructor|tutor).{0,80}(student|pupil|class|grade|semester|assignment|homework|office.?hour)"
                               r"|(student|pupil).{0,80}(professor|teacher|instructor|tutor)"
                               r"|after.?class\b|extra.?credit\b|office.?hours\b"), 2),

    ("boss / employee",     _r(r"\b(her|his|my|the)\s+(boss|manager|supervisor|employer)\b"
                               r"|(boss|manager|supervisor).{0,80}(employee|subordinate|secretary|assistant|staff|worker)"
                               r"|(employee|subordinate|secretary|assistant).{0,80}(boss|manager|supervisor)"
                               r"|called.{0,10}into.{0,15}(the|his|her).{0,5}office\b"
                               r"|performance.?review\b|(quit|resign\w*).{0,20}(job|work|position)"), 2),

    ("doctor / patient",    _r(r"\b(her|his|my|the)\s+(doctor|therapist|psychiatrist|psychologist|physician|hypnotherapist)\b"
                               r"|(doctor|therapist|psychiatrist|psychologist|physician|nurse).{0,80}(patient|client|appointment|session|treatment)"
                               r"|(patient|client).{0,80}(doctor|therapist|physician|nurse|psychiatrist|psychologist)"
                               r"|next.?appointment\b|prescription\b|diagnosis\b"
                               r"|on.{0,5}the.{0,10}(couch|table|chair).{0,50}(therapist|doctor|session)"), 2),

    ("coworkers",           _r(r"\b(coworker|co-worker|colleague)\b"
                               r"|same.{0,10}(office|department|company|floor|team).{0,50}(him|her|them|both|together)"
                               r"|work\w*.{0,30}(together|alongside|side.?by.?side).{0,30}(office|company|job|work)"
                               r"|\bfellow.{0,5}(worker|employee|staff)\b"), 2),
]


POV_PATTERNS = [
    ("first person",  _r(r"\b(I |I'm|I was|I felt|I had|my |me )\b")),
    ("third person",  _r(r"\b(he |she |they |his |her |their )\b")),
    ("second person", _r(r"\b(you |your |you're|you've)\b")),
]


def classify_length(word_count: int) -> str:
    """Length class from a word count: flash (<1k), short story (<7.5k), novelette
    (<17.5k), novella (<40k), novel.
    """
    if word_count < 1000:   return "flash"
    if word_count < 7500:   return "short story"
    if word_count < 17500:  return "novelette"
    if word_count < 40000:  return "novella"
    return "novel"


def detect_pov(text: str) -> str:
    """First, second or third person, by which pronoun pattern is most common in the first
    3,000 characters; 'unknown' if the winner has fewer than 5 hits.
    """
    if not text:
        return "unknown"
    sample = text[:3000]
    counts = {label: len(pat.findall(sample)) for label, pat in POV_PATTERNS}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] >= 5 else "unknown"


def apply_rules(text: str, rules: RuleList) -> list[str]:
    """Labels whose regex matches the text at least min_n times (the third element of each
    rule).
    """
    results = []
    for label, pat, min_n in rules:
        if min_n == 1:
            # Fast path: just check if there's any match
            if pat.search(text):
                results.append(label)
        else:
            # Count matches but stop early once we hit min_n
            count = 0
            for _ in pat.finditer(text):
                count += 1
                if count >= min_n:
                    results.append(label)
                    break
    return results


def expand_site_tags(raw_tags: list[str]) -> dict[str, list[str]]:
    """Turn mcstories.com category codes (mc, mf, fd, ...) into readable labels plus the
    groups they belong to (sex_act, gender_dynamic, kink, genre, content_flag). Unknown
    codes are ignored.
    """
    expanded, groups = [], set()
    for tag in raw_tags:
        t = tag.lower().strip()
        if t in SITE_TAG_LABELS:
            expanded.append(SITE_TAG_LABELS[t])
            for group, members in SITE_TAG_GROUPS.items():
                if t in members:
                    groups.add(group)
    return {"labels": list(dict.fromkeys(expanded)), "groups": sorted(groups)}


def regex_tag(story: dict[str, Any]) -> dict[str, Any]:
    """Run Layer 1. Returns a flat auto_tags dict."""
    full_text  = _strip_html(story.get("full_text", ""))
    title      = story.get("title", "")
    desc       = story.get("description", "")
    site_cats  = story.get("categories", [])
    word_count = story.get("word_count", 0)

    # Title + description weighted 3x for strong signal.
    # Full text used for all rules — sampling was missing acts that appear mid-story.
    # Normalize curly apostrophes/quotes so patterns with ' match both ' and '.
    corpus = (f"{title} {desc} " * 3 + full_text).lower()
    corpus = corpus.replace('\u2019', "'").replace('\u2018', "'")
    site   = expand_site_tags(site_cats)

    tags = {
        "site_tags_expanded": site["labels"],
        "site_tag_groups":    site["groups"],
        "content_type":       [SITE_TAG_LABELS[t.lower()] for t in site_cats
                               if t.lower() in SITE_TAG_LABELS
                               and t.lower() in SITE_TAG_GROUPS.get("gender_dynamic", [])],
        "mind_control_type":  apply_rules(corpus, MIND_CONTROL_TYPE_RULES),
        "setting":            apply_rules(corpus, SETTING_RULES),
        "themes":             apply_rules(corpus, THEME_RULES),
        "tone":               apply_rules(corpus, TONE_RULES),
        "dominance":          apply_rules(corpus, DOMINANCE_RULES),
        "pacing":             list(dict.fromkeys(
                                  apply_rules(corpus, PACING_RULES)
                                  + infer_pacing_from_structure(story)
                              )),
        "awareness":          apply_rules(corpus, AWARENESS_RULES),
        "prevalence":         apply_rules(corpus, PREVALENCE_RULES),
        "mental_changes":     apply_rules(corpus, MENTAL_CHANGE_RULES),
        "sexual_acts":        apply_rules(corpus, SEXUAL_ACT_RULES),
        "fetish":             apply_rules(corpus, FETISH_RULES),
        "relationship":       apply_rules(corpus, RELATIONSHIP_RULES),
        "pov":                detect_pov(full_text),
        "length_class":       classify_length(word_count),
    }

    # ── Contradiction suppression ──────────────────────────────────────────────
    # Remove tags that are logically incompatible with other strong signals.

    cats_lower = [c.lower() for c in site_cats]
    dom  = tags["dominance"]
    thms = tags["themes"]
    tone = tags["tone"]

    # "loving control" cannot coexist with dehumanisation, betrayal, humiliation,
    # sadism, or brutal treatment — remove it if any of these are present.
    if "loving control" in dom:
        if (any(t in thms for t in ("betrayal", "horror / dark", "slave mentality"))
                or "hm" in cats_lower
                or any(t in dom for t in ("sadistic", "brutal"))
                or "dark / disturbing" in tone):
            dom.remove("loving control")

    # "gentle" is incompatible with betrayal, humiliation, or brutal/sadistic signals.
    if "gentle" in dom:
        if (any(t in thms for t in ("betrayal", "horror / dark"))
                or "hm" in cats_lower
                or any(t in dom for t in ("sadistic", "brutal"))):
            dom.remove("gentle")

    return tags


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SEMANTIC TAGGER
# ══════════════════════════════════════════════════════════════════════════════

# Richer descriptions = better embeddings.
# Each key must exactly match the label used in the regex layer.
SEMANTIC_TAG_DESCRIPTIONS: dict[str, dict[str, str]] = {

    "mind_control_type": {
        "hypnosis": (
            "a story where someone is hypnotised, placed in a trance, subjected to "
            "hypnotic induction, given post-hypnotic suggestions, or made to follow "
            "commands while in an altered state of consciousness"
        ),
        "chemical": (
            "a story where mind control is achieved through a drug, chemical, potion, "
            "serum, aphrodisiac, injection, or ingested substance that alters the "
            "victim's will or behaviour"
        ),
        "magic": (
            "a story involving magical control — spells, enchantments, charms, curses, "
            "arcane rituals, or mystical forces that override someone's free will"
        ),
        "technology": (
            "a story where control is achieved through technology — a neural implant, "
            "mind-control device, app, algorithm, collar, VR headset, nanobots, or "
            "other scientific or electronic means"
        ),
        "subliminal": (
            "a story where influence works below conscious awareness — subliminal "
            "messages, brainwashing, conditioning, or gradual psychological "
            "reprogramming that the victim does not consciously notice"
        ),
        "psionic": (
            "a story involving alien mind control, psychic domination, telepathy, "
            "pheromone influence, hive-mind absorption, or parasitic possession"
        ),
        "dominant personality": (
            "a story where control comes purely from a compelling, charismatic, or "
            "overwhelmingly dominant personality — no device, drug, or magic, just "
            "psychological power and force of will"
        ),
    },

    "setting": {
        "workplace": (
            "a story set in an office, corporation, workplace, or professional "
            "environment — with bosses, coworkers, meetings, cubicles, or a career context"
        ),
        "school / college": (
            "a story set in a school, university, college, or academic environment "
            "involving students, professors, classrooms, dormitories, or campus life"
        ),
        "home / domestic": (
            "a story set primarily in a home, house, or domestic environment — "
            "involving housemates, spouses, family members, or everyday home life"
        ),
        "fantasy / historical": (
            "a story set in a fantasy world with magic, wizards, castles, dragons, "
            "medieval settings, enchanted kingdoms, or mythological creatures"
        ),
        "science fiction": (
            "a story set in a science-fiction universe — space travel, alien worlds, "
            "the far future, advanced technology, robots, or interstellar society"
        ),
        "supernatural": (
            "a story set in a world with supernatural elements — vampires, werewolves, "
            "demons, ghosts, occult forces, or creatures of the night"
        ),
        "gym / sports": (
            "a story set in a gym, fitness studio, sports facility, or athletic "
            "environment involving trainers, athletes, or physical exercise"
        ),
        "bar / club / party": (
            "a story set in a bar, nightclub, party, or social gathering with "
            "alcohol, dancing, crowds, or a nightlife atmosphere"
        ),
        "online / virtual": (
            "a story that takes place online, in a virtual environment, chat room, "
            "VR world, or where the internet or digital technology is the primary setting"
        ),
        "medical": (
            "a story set in a therapy office, medical clinic, hospital, or healthcare "
            "environment involving patients, doctors, nurses, or therapists"
        ),
    },

    "themes": {
        "slow burn": (
            "a story where mind control or a relationship develops slowly and gradually "
            "over time, with resistance, reluctance, or incremental change rather than "
            "an immediate takeover"
        ),
        "corruption arc": (
            "a story about the gradual corruption, transformation, or personality change "
            "of a character — where who they are fundamentally shifts over the course "
            "of the story"
        ),
        "romance": (
            "a story with strong romantic elements — love, emotional connection, "
            "tenderness, longing, or a relationship that goes beyond purely physical"
        ),
        "revenge": (
            "a story driven by revenge, power struggle, dominance games, or deliberate "
            "humiliation — where control is used as a weapon or payback"
        ),
        "consent / ethics": (
            "a story that explicitly explores the ethics of mind control, grappling "
            "with consent, guilt, moral ambiguity, or the right and wrong of having "
            "power over another person"
        ),
        "humor / comedy": (
            "a story that is primarily comedic or humorous in tone — with jokes, "
            "absurdist situations, irony, or a generally light and funny approach"
        ),
        "identity loss": (
            "a story centrally concerned with loss of identity, selfhood, or memory — "
            "where a character forgets who they are, loses their personality, or their "
            "sense of self is erased or replaced"
        ),
        "exhibitionism": (
            "a story with strong exhibitionist themes — public exposure, performance, "
            "being watched, stripping, or display in front of an audience"
        ),
        "consensual fantasy": (
            "a story framed as consensual roleplay, a mutual fantasy, or a game where "
            "all parties agree to the scenario and it is presented as a shared desire"
        ),
        "horror / dark": (
            "a story with a horror, dark, or disturbing tone featuring fear, dread, "
            "graphic content, psychological terror, or a bleak or threatening atmosphere"
        ),
        "body modification": (
            "a story involving physical transformation or body modification — breast "
            "expansion, growth, physical enhancement, or changes to the body as part "
            "of the mind control experience"
        ),
    },

    "tone": {
        "dark / disturbing": (
            "a story with a dark, disturbing, or unsettling tone — brutal, violent, "
            "sinister, or psychologically distressing"
        ),
        "romantic / sweet": (
            "a story with a warm, tender, romantic, or sweet tone — gentle, caring, "
            "emotionally intimate, and focused on connection rather than dominance"
        ),
        "humorous / playful": (
            "a story with a light, humorous, playful, or comedic tone — not taking "
            "itself too seriously, with wit, banter, or absurdist fun"
        ),
        "erotic / explicit": (
            "a story that is primarily erotic in tone — explicitly sexual, arousing, "
            "with detailed physical descriptions and a focus on sexual content"
        ),
        "psychological": (
            "a story with a psychological, introspective, or cerebral tone — focused "
            "on the inner experience of characters, their thoughts, consciousness, "
            "and the mental dimension of control"
        ),
        "melancholic": (
            "a story with a sad, melancholic, or bittersweet tone — grief, longing, "
            "loss, or a sense of hollowness"
        ),
        "tense / suspenseful": (
            "a story with tension, suspense, or danger — where the stakes feel real "
            "and the reader fears what will happen next"
        ),
    },

    "dominance": {
        "gentle": (
            "a story where the dominant figure is caring, tender, protective, and "
            "nurturing — control expressed through love and patience rather than force"
        ),
        "firm but fair": (
            "a story where the dominant figure is strict and demanding but not cruel — "
            "clear rules, discipline, and high expectations but without sadism"
        ),
        "sadistic": (
            "a story where the dominant figure is cruel, sadistic, brutal, or merciless — "
            "enjoying the subject's pain, humiliation, or suffering"
        ),
        "manipulative": (
            "a story where control is achieved through subtle manipulation, deception, "
            "gaslighting, or psychological scheming rather than overt force"
        ),
        "loving control": (
            "a story where dominance and control are an expression of deep love and "
            "devotion — the dominant figure cherishes and adores their subject"
        ),
        "cold / clinical": (
            "a story where the dominant figure is detached, cold, clinical, or businesslike — "
            "treating the subject as an asset, tool, or project rather than a person"
        ),
    },

    "pacing": {
        "leisurely": (
            "a story where the mind control or relationship develops very slowly over "
            "a long period — weeks, months, gradual incremental change"
        ),
        "fast / immediate": (
            "a story where control is established quickly or immediately — sudden "
            "snapping into trance, instant effect, rapid submission"
        ),
        "episodic": (
            "a story told in chapters or episodes, each a separate session or encounter "
            "that advances the ongoing relationship or control"
        ),
        "single scene": (
            "a story that takes place in a single continuous scene or encounter — "
            "one evening, one session, one afternoon"
        ),
    },

    "awareness": {
        "fully aware / helpless": (
            "the subject is completely aware of what is happening to them — they can "
            "see themselves obeying, feel the control, know it is wrong — but are "
            "utterly powerless to stop it. Their mind is intact but their body or will "
            "acts against them."
        ),
        "accepting": (
            "the subject knows what is happening and has stopped fighting it — they "
            "have surrendered, given in, or chosen to accept the control. Not quite "
            "willing but no longer resisting."
        ),
        "confused / questioning": (
            "the subject senses something is wrong or different but cannot identify it — "
            "they feel strange urges, foggy thinking, or behaviour that puzzles them, "
            "but do not understand the cause."
        ),
        "oblivious": (
            "the subject has no idea they are being controlled — they believe their "
            "thoughts and actions are entirely their own. The control is invisible to "
            "them."
        ),
        "gradually losing awareness": (
            "the subject begins aware but slowly loses their sense of self, memories, "
            "or resistance over time — a progressive erosion of their original "
            "personality or knowledge of what is happening."
        ),
        "forced / non-consensual awareness": (
            "the subject is aware, actively resisting and trying to fight back, "
            "but their body or voice acts against their will — a strong non-consent "
            "dynamic where the inner person and the outer body are at war."
        ),
        "willingly induced unawareness": (
            "the subject consented to or requested the control, trance, or programming — "
            "they chose to give up their awareness or will, which makes the subsequent "
            "loss of control a fulfilment of their own desire."
        ),
    },
}


class SemanticTagger:
    """
    Wraps a SentenceTransformer model. Tag description embeddings are
    computed once at load time and reused for every story.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 threshold: float = 0.35):
        self.model_name = model_name
        self.threshold  = threshold
        self._model     = None
        self._tag_vecs: dict[str, dict[str, Any]] = {}

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run:  pip install sentence-transformers\n"
                "Then re-run with --semantic."
            )
        log.info(f"Loading model '{self.model_name}'…")
        self._model = SentenceTransformer(self.model_name)
        log.info("Pre-computing tag description embeddings…")
        for dim, tag_map in SEMANTIC_TAG_DESCRIPTIONS.items():
            labels = list(tag_map.keys())
            descs  = list(tag_map.values())
            vecs   = self._model.encode(descs, normalize_embeddings=True)
            self._tag_vecs[dim] = dict(zip(labels, vecs))
        log.info("Semantic tagger ready.")

    def _story_sample(self, story: dict[str, Any]) -> str:
        """Build the text sample used for embedding a story."""
        title     = story.get("title", "")
        desc      = story.get("description", "")
        full_text = _strip_html(story.get("full_text", ""))
        return f"{title}. {desc}. " + " ".join(full_text.split()[:600])

    def tag_story(self, story: dict[str, Any]) -> dict[str, list[str]]:
        """Return {dim: [matched_label, ...]} for semantic dims only."""
        self._load()
        import numpy as np
        story_vec = self._model.encode(
            self._story_sample(story), normalize_embeddings=True)
        return self._score_vec(story_vec, np)

    def _score_vec(self, story_vec, np) -> dict[str, list[str]]:
        """Score a single embedding vector against all tag descriptions."""
        results: dict[str, list[str]] = {}
        for dim, tag_vecs in self._tag_vecs.items():
            hits = []
            for label, tvec in tag_vecs.items():
                score = float(np.dot(story_vec, tvec))
                if score >= self.threshold:
                    hits.append((label, score))
            results[dim] = [lbl for lbl, _ in sorted(hits, key=lambda x: -x[1])]
        return results

    def tag_stories_batch(self, stories: list[dict[str, Any]],
                          batch_size: int = 64) -> list[dict[str, list[str]]]:
        """
        Encode and score a list of stories in batches.
        Much faster than calling tag_story() in a loop — the GPU/CPU
        processes batch_size stories simultaneously.
        """
        self._load()
        import numpy as np

        samples = [self._story_sample(s) for s in stories]
        results = []

        for i in range(0, len(samples), batch_size):
            batch   = samples[i:i + batch_size]
            vecs    = self._model.encode(
                batch,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            for vec in vecs:
                results.append(self._score_vec(vec, np))

        return results


# ══════════════════════════════════════════════════════════════════════════════
# MERGE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _dedup(lst: list) -> list:
    return list(dict.fromkeys(lst))


def merge_tags(regex_result: dict[str, Any],
               semantic_result: Optional[dict[str, list[str]]] = None
               ) -> dict[str, Any]:
    """
    Merge regex and semantic results into a single auto_tags dict.

    Per dimension:
      - REGEX_ONLY_DIMS  : always use regex, ignore semantic
      - SEMANTIC_DIMS    : union of both; regex first (high-precision anchors),
                           then semantic-only additions appended after

    _tag_sources records which layer(s) fired each tag — useful for
    debugging and threshold tuning.
    """
    merged:  dict[str, Any]        = {}
    sources: dict[str, list[str]]  = {}

    for dim in REGEX_ONLY_DIMS:
        val = regex_result.get(dim)
        merged[dim] = val if val is not None else ([] if dim not in ("pov", "length_class") else "")

    sem = semantic_result or {}
    all_sem_dims = SEMANTIC_DIMS | SEMANTIC_EXTRA_DIMS
    for dim in all_sem_dims:
        rx_tags  = regex_result.get(dim, [])
        sem_tags = sem.get(dim, [])

        for t in rx_tags:
            sources.setdefault(t, [])
            if "regex" not in sources[t]:
                sources[t].append("regex")
        for t in sem_tags:
            sources.setdefault(t, [])
            if "semantic" not in sources[t]:
                sources[t].append("semantic")

        merged[dim] = _dedup(rx_tags + [t for t in sem_tags if t not in rx_tags])

    merged["_tag_sources"]    = sources
    merged["_tagger_version"] = "2.0"
    merged["_semantic_used"]  = semantic_result is not None
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# SVENGALI → DIMENSION PROMOTION
# ══════════════════════════════════════════════════════════════════════════════

def promote_svengali_tags(auto_tags: dict[str, Any]) -> None:
    """
    Convert raw _svengali_tags display names into dimension fields in-place.
    Uses the SV_* maps from svengali.py to map display names → our tag values.
    Tags added this way are recorded as "svengali" in _tag_sources.

    This is the highest-quality tagging layer — community-voted ground truth.
    Called after regex/semantic tagging so Svengali data is always preserved.
    """
    sv_raw = auto_tags.get("_svengali_tags")
    if not sv_raw:
        return

    try:
        from svengali import (SV_MC_TYPE, SV_AWARENESS, SV_DOMINANCE, SV_MENTAL,
                              SV_PACING, SV_SEXUAL_ACTS, SV_RELATIONSHIP, SV_SETTING,
                              SV_THEMES, SV_CONSENT)
    except ImportError:
        return

    # Map each Svengali display-name dict to its target dimension
    dim_maps = [
        ("mind_control_type", SV_MC_TYPE),
        ("awareness",         SV_AWARENESS),
        ("dominance",         SV_DOMINANCE),
        ("mental_changes",    SV_MENTAL),
        ("pacing",            SV_PACING),
        ("sexual_acts",       SV_SEXUAL_ACTS),
        ("relationship",      SV_RELATIONSHIP),
        ("setting",           SV_SETTING),
        ("themes",            SV_THEMES),
    ]

    sources = auto_tags.setdefault("_tag_sources", {})
    for dim, sv_map in dim_maps:
        lookup = {k.lower(): v for k, v in sv_map.items()}
        existing = auto_tags.get(dim, [])
        for raw_tag in sv_raw:
            mapped = lookup.get(raw_tag.lower())
            if mapped and mapped not in existing:
                existing.append(mapped)
            if mapped:
                src = sources.setdefault(mapped, [])
                if "svengali" not in src:
                    src.append("svengali")
        auto_tags[dim] = existing


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API  (called by scraper.py)
# ══════════════════════════════════════════════════════════════════════════════

_semantic_tagger: Optional[SemanticTagger] = None


def tag_story(story: dict[str, Any],
              use_semantic: bool = False,
              threshold: float = 0.35) -> dict[str, Any]:
    """
    Tag a single story dict. Drop-in replacement for the old tag_story().

    Args:
        story:        story dict from stories_index.json
        use_semantic: enable the semantic layer (requires sentence-transformers)
        threshold:    cosine similarity cutoff (default 0.35)
    """
    global _semantic_tagger

    rx = regex_tag(story)

    if not use_semantic:
        return merge_tags(rx)

    if _semantic_tagger is None or _semantic_tagger.threshold != threshold:
        _semantic_tagger = SemanticTagger(threshold=threshold)

    sem = _semantic_tagger.tag_story(story)
    return merge_tags(rx, sem)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def process_index(index_path: Path,
                  output_path: Optional[Path] = None,
                  use_semantic: bool = False,
                  threshold: float = 0.35) -> None:
    """Tag all stories in stories_index.json and save."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    in_place = output_path is None or Path(output_path) == Path(index_path)
    data = index_store.load_index(index_path, for_write=in_place)

    stories = data["stories"]
    total   = len(stories)
    tagged  = 0

    if use_semantic:
        global _semantic_tagger
        if _semantic_tagger is None:
            _semantic_tagger = SemanticTagger(threshold=threshold)
        _semantic_tagger._load()   # load once before loop

        # Batch semantic encoding — much faster than one-at-a-time
        BATCH = 64
        valid = [(i, s) for i, s in enumerate(stories)
                 if not s.get("fetch_error") and s.get("full_text")]
        print(f"  Batch encoding {len(valid)} stories (batch={BATCH})...")
        for chunk_start in range(0, len(valid), BATCH):
            chunk     = valid[chunk_start:chunk_start + BATCH]
            idxs      = [c[0] for c in chunk]
            strs      = [c[1] for c in chunk]
            sem_tags  = _semantic_tagger.tag_stories_batch(strs, batch_size=BATCH)
            for idx, story, sem in zip(idxs, strs, sem_tags):
                regex_result  = regex_tag(story)
                old_sv = (story.get("auto_tags") or {}).get("_svengali_tags")
                new_at = merge_tags(regex_result, sem)
                if old_sv:
                    new_at["_svengali_tags"] = old_sv
                    promote_svengali_tags(new_at)
                stories[idx]["auto_tags"] = new_at
                tagged += 1
            done = min(chunk_start + BATCH, len(valid))
            print(f"  [{done}/{len(valid)}] semantic batch done")
    else:
        for i, story in enumerate(stories, 1):
            if story.get("fetch_error") or not story.get("full_text"):
                story["auto_tags"] = None
                continue
            old_sv = (story.get("auto_tags") or {}).get("_svengali_tags")
            new_at = tag_story(story, use_semantic=False)
            if old_sv:
                new_at["_svengali_tags"] = old_sv
                promote_svengali_tags(new_at)
            story["auto_tags"] = new_at
            tagged += 1
            if i % 500 == 0:
                print(f"  [regex] Tagged {i}/{total}...")

    data["auto_tagged"]      = True
    data["tagged_count"]     = tagged
    data["semantic_tagging"] = use_semantic

    out = output_path or index_path
    print(f"Writing {len(stories):,} stories to {out}...")
    # Validated, atomic save: the original is untouched unless the new file
    # holds every story
    try:
        index_store.save_index(data, out, validate=True)
    except RuntimeError as e:
        raise RuntimeError(f"Tagger output validation failed — original untouched: {e}")
    mode = "regex + semantic" if use_semantic else "regex only"
    print(f"Done. Tagged {tagged}/{total} stories ({mode}) -> {out}")


# ══════════════════════════════════════════════════════════════════════════════
# JSONL REGENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def rebuild_jsonl_with_tags(index_path: Path, output_dir: Path) -> None:
    """Regenerate all three JSONL training files from an already-tagged index."""
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)

    # Write to .tmp files first — protect existing training data from partial writes
    raw_f   = open(output_dir / "training_raw_pretraining.jsonl.tmp",      "w", encoding="utf-8")
    instr_f = open(output_dir / "training_instruction_response.jsonl.tmp", "w", encoding="utf-8")
    comp_f  = open(output_dir / "training_completion.jsonl.tmp",           "w", encoding="utf-8")

    written = 0
    for story in data["stories"]:
        if not story.get("full_text") or story.get("fetch_error"):
            continue

        at      = export_tags(story.get("auto_tags"))
        cats    = ", ".join(story.get("categories", [])) or "fiction"
        themes  = ", ".join(at.get("themes", []))
        mc_type = ", ".join(at.get("mind_control_type", []))
        setting = ", ".join(at.get("setting", []))
        tone    = ", ".join(at.get("tone", []))
        length  = at.get("length_class", "")
        pov     = at.get("pov", "")

        meta = [f"Title: {story['title']}", f"Author: {story['author']}",
                f"Categories: {cats}"]
        if themes:  meta.append(f"Themes: {themes}")
        if mc_type: meta.append(f"Mind Control Method: {mc_type}")
        if setting: meta.append(f"Setting: {setting}")
        if tone:    meta.append(f"Tone: {tone}")
        if length:  meta.append(f"Length: {length}")
        if pov:     meta.append(f"POV: {pov}")
        if story.get("description"):
            meta.append(f"Summary: {story['description']}")

        meta_block = "\n".join(meta)

        raw_f.write(json.dumps(
            {"text": meta_block + "\n\n" + story["full_text"]},
            ensure_ascii=False) + "\n")

        user_msg = f'Write a {cats} story titled "{story["title"]}" by {story["author"]}.'
        if themes:  user_msg += f" Themes: {themes}."
        if mc_type: user_msg += f" Mind control method: {mc_type}."
        if setting: user_msg += f" Setting: {setting}."

        instr_f.write(json.dumps({
            "messages": [
                {"role": "system",    "content": "You are a skilled fiction writer specializing in mind control erotica."},
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": story["full_text"]},
            ]
        }, ensure_ascii=False) + "\n")

        comp_f.write(json.dumps({
            "prompt":     meta_block + "\n\n",
            "completion": story["full_text"],
        }, ensure_ascii=False) + "\n")

        written += 1

    raw_f.close(); instr_f.close(); comp_f.close()

    # Rename .tmp → final only after all three wrote successfully
    for stem in ["training_raw_pretraining.jsonl",
                 "training_instruction_response.jsonl",
                 "training_completion.jsonl"]:
        tmp = output_dir / (stem + ".tmp")
        if tmp.exists():
            tmp.replace(output_dir / stem)

    print(f"Done. Rebuilt JSONL files for {written} stories -> {output_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Tag stories — regex layer always runs, semantic layer optional",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tagger.py                               # regex only (fast, no extra deps)
  python tagger.py --semantic                    # regex + semantic
  python tagger.py --semantic --threshold 0.30   # lower threshold = more tags
  python tagger.py --semantic --rebuild-jsonl    # retag and rebuild training files
        """
    )
    p.add_argument("-i", "--index",   default="./mcstories_output/stories_index.json")
    p.add_argument("-o", "--output",  default=None,
                   help="Output path (default: overwrite input)")
    p.add_argument("--semantic",      action="store_true",
                   help="Enable semantic layer (requires: pip install sentence-transformers)")
    p.add_argument("--threshold",     type=float, default=0.35,
                   help="Cosine similarity threshold for semantic tags (default: 0.35)")
    p.add_argument("--rebuild-jsonl", action="store_true",
                   help="Rebuild all JSONL training files after tagging")
    p.add_argument("--jsonl-only",   action="store_true",
                   help="Rebuild JSONL from existing tags only — skip regex/semantic tagging")
    args = p.parse_args()

    idx = Path(args.index)
    if not idx.exists():
        print(f"Index not found: {idx}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output) if args.output else None

    if args.jsonl_only:
        rebuild_jsonl_with_tags(idx, idx.parent)
    else:
        process_index(idx, out, use_semantic=args.semantic, threshold=args.threshold)
        if args.rebuild_jsonl:
            final_idx = out or idx
            rebuild_jsonl_with_tags(final_idx, final_idx.parent)
