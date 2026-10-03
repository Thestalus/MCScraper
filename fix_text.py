#!/usr/bin/env python3
"""
fix_text.py — apply text-cleanup to all full_text fields in stories_index.json
              and rebuild the SQLite full-text cache.

Usage:
  .venv/bin/python3 fix_text.py
  .venv/bin/python3 fix_text.py --index ./output/stories_index.json
"""

import json
import re
import sqlite3
import argparse
from pathlib import Path
import index_store

try:
    import wordninja
    _WORDNINJA = True
    _WORDCOST = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
except ImportError:
    _WORDNINJA = False
    _WORDCOST = {}

# System dictionary — real English words we should never split.
# Loaded from /usr/share/dict/words (235k entries) + proper names.
def _load_sys_dict() -> set:
    words: set[str] = set()
    for path in ('/usr/share/dict/words', '/usr/share/dict/propernames'):
        try:
            with open(path) as f:
                words.update(w.strip().lower() for w in f if w.strip())
        except FileNotFoundError:
            pass
    return words

_SYS_DICT = _load_sys_dict()

# Only accept a wordninja split if all parts are common words.
# Cost is negative log-frequency from Wikipedia; lower = more common.
# Threshold 11.0 rejects splits of rare/unknown words (nubbins, nanites, etc.)
# while accepting splits of common concatenated words (his+huge, and+who, etc.)
_SPLIT_COST_THRESHOLD = 11.0

# Words wordninja would incorrectly split — genre-specific and common compounds
_GENRE_WHITELIST = {
    # Genre-specific
    'stepfordized', 'stepfordization', 'bimboification', 'bimboified',
    'bimboize', 'mindcontrolled', 'brainwashing', 'brainwashed',
    'dominatrix', 'submissive', 'submissiveness', 'hypnotized',
    'hypnotised', 'mesmerized', 'mesmerised', 'sleepwalking',
    'sleepwalker', 'mindless', 'mindlessly', 'enslavement',
    'enslaved', 'conditioned', 'conditioning', 'reprogrammed',
    'reprogramming', 'thoughtless', 'thoughtlessness', 'obedience',
    'disobedience', 'entranced', 'entrancement', 'compelled',
    'compulsion', 'manipulated', 'manipulation', 'corrupted',
    'corruption', 'overwritten', 'overwriting', 'mcstories',
    'hypnodomme', 'hypnoslave', 'hypnoslaves', 'hypnodommes',
    'trancelike', 'posthypnotic', 'zombified', 'mindwipe',
    'mindwiped', 'mindfuck', 'mindfucked', 'mindbreak',
    'mindbroken', 'overwhelmingness', 'thoughtcontrol',
    'braindrain', 'mindcontrol',
    # Reflexive pronouns
    'yourself', 'myself', 'himself', 'herself', 'themselves',
    'itself', 'oneself', 'yourselves', 'ourselves',
    # Common compound words
    'without', 'within', 'throughout', 'although', 'already',
    'always', 'almost', 'also',
    'someone', 'something', 'somewhere', 'somehow', 'sometimes',
    'sometime', 'somewhat',
    'anyone', 'anything', 'anywhere', 'anytime', 'anyhow',
    'everyone', 'everything', 'everywhere', 'everyday',
    'nobody', 'nothing', 'nowhere',
    'because', 'before', 'beyond', 'between', 'beneath', 'behind',
    'beside', 'besides', 'became', 'become', 'becomes', 'becoming',
    'underneath', 'understand', 'understood', 'understanding',
    'however', 'whatever', 'whenever', 'wherever', 'whichever',
    'whoever', 'whatever', 'otherwise', 'therefore', 'together',
    'another', 'anymore', 'anymore', 'tonight', 'today', 'tomorrow',
    'himself', 'herself', 'yourself', 'myself',
    'cannot', 'couldn', 'wouldn', 'shouldn', 'doesn', 'haven',
    'hadn', 'wasn', 'weren', 'isn', 'aren',
    # Genre / erotica terms that wordninja splits but are legitimate single words
    'hardon', 'hardons', 'catsuit', 'catsuits', 'keycard', 'keycards',
    'callbox', 'earbuds', 'earbud', 'titties', 'titty', 'tittie',
    'shlong', 'schlongs', 'sextoy', 'sextoys',
    # Common English words that split badly
    'soften', 'softened', 'softening', 'softens',
    'texted', 'texting',
    'indrawn', 'alight', 'boffin', 'boffins',
    'persona', 'personas', 'demeanour', 'demeanor',
    'repost', 'reposted', 'reposting',
    'theirs', 'awhile',
    'inform', 'informs', 'informed', 'informing',
    'barkeep', 'barkeeps', 'barkeeper',
    'herein', 'hereto', 'hereof', 'hereby', 'herewith',
    'carryon', 'carryout',
    'defocus', 'defocused', 'defocusing', 'defocuses',
    'nametag', 'nametags', 'eyeroll', 'eyerolls', 'mudroom', 'mudrooms',
    'byline', 'bylines', 'stupour', 'stupours',
    'catgirl', 'catgirls', 'catboy', 'catboys', 'petplay',
    'soused', 'doused', 'dicked', 'crewcut', 'kneepads', 'kneepad',
    'bowtie', 'bowties',
    'empath', 'empaths', 'empathic', 'empathy',
    'fellate', 'fellated', 'fellating', 'fellatio',
    'unfocus', 'unfocused', 'unfocusing', 'unfocuses',
    'whoring', 'whored', 'whores',
    # Real words not in /usr/share/dict
    'frisson', 'fellas', 'doofus', 'ardour', 'kissable', 'feedlot', 'lasing',
    'forking', 'forked', 'foreskin', 'foreskins', 'duking',
    'externalities',
    # Dialect / informal contractions
    'shoulda', 'woulda', 'coulda', 'gonna', 'wanna', 'gotta',
    'kindda',
    # Compound words that split correctly but shouldn't
    'payback', 'paybacks', 'subzero',
    # Genre/fiction words
    'pantied', 'childified',
    'lapine', 'lapines',  # rabbit/furry fiction
    'unblank', 'unblanked', 'unblanking', 'unblanks',  # MC genre
    'derriere', 'derrieres',
    # Misspellings that get badly split — preserve the typo as-is
    'orgasem', 'orgasam', 'orgasms',  # orgasm misspellings
    'forskin',  # foreskin
    'noticable', 'unnoticable',  # noticeable / unnoticeable
    'presure',  # pressure
    'octupus',  # octopus
    'younest',  # youngest
    'tounge',  # tongue
    'essense',  # essence
    'farsend',  # fasten (typo/concatenation)
    'manking',  # lovemaking (concatenation)
    'mouthes',  # mouths
    'breaked',  # broke (non-standard past tense)
    'forhead',  # forehead
    'heared',   # heard
    'unitard', 'unitards',  # real garment word
    'pendula',  # plural of pendulum
    'fundies',  # slang for fundamentalists
}

def _maybe_split(token: str) -> str:
    """Use wordninja to split a suspiciously long token, unless it's whitelisted."""
    if not _WORDNINJA:
        return token
    if token.lower() in _GENRE_WHITELIST:
        return token
    # Skip tokens that exist in the system dictionary — they're real words.
    if token.lower() in _SYS_DICT:
        return token
    if len(token) < 6:
        return token
    # Skip tokens made of repeated characters (intentional stuttering: "Iiiiii")
    if len(set(token.lower())) <= 2:
        return token
    # Skip tokens that start with an uppercase letter — likely a proper noun.
    # Exception: tokens starting with 'I' + lowercase where wordninja's first
    # split part is actually 'i' (the English pronoun), e.g. Iknowthat → I know that.
    # This prevents splitting names like Ivanna → Iv anna or Illora → Ill or a.
    if token[0].isupper():
        if not (token[0] == 'I' and len(token) > 1 and token[1].islower()):
            return token
        # Peek at what wordninja would produce before committing
        parts = wordninja.split(token)
        if not parts or parts[0].lower() != 'i':
            return token
    else:
        parts = wordninja.split(token)
    if len(parts) == 1:
        return token
    # Reject single-char fragments other than 'a' and 'i' (real English words).
    # Letters like t, s, d, b have low wordninja costs but aren't standalone words.
    if any(len(p) == 1 and p.lower() not in ('a', 'i') for p in parts):
        return token
    # Reject if any part is unknown or rare in wordninja's model.
    # This prevents splitting real words (nubbins, nanites) that happen to
    # decompose into two obscure fragments.
    if any(_WORDCOST.get(p.lower()) is None or
           _WORDCOST[p.lower()] > _SPLIT_COST_THRESHOLD
           for p in parts):
        return token
    # Preserve original capitalisation of first char
    result = ' '.join(parts)
    if token[0].isupper() and result[0].islower():
        result = result[0].upper() + result[1:]
    return result


# Compound words that wordninja incorrectly split — repair pass
# Maps "split form" → "correct form" (word-boundary matched)
_REJOIN_MAP = {
    r'your self\b': 'yourself',
    r'my self\b': 'myself',
    r'him self\b': 'himself',
    r'her self\b': 'herself',
    r'them selves\b': 'themselves',
    r'it self\b': 'itself',
    r'your selves\b': 'yourselves',
    r'our selves\b': 'ourselves',
    r'with out\b': 'without',
    r'with in\b': 'within',
    r'through out\b': 'throughout',
    r'al though\b': 'although',
    r'al ready\b': 'already',
    r'al ways\b': 'always',
    r'al most\b': 'almost',
    r'some one\b': 'someone',
    r'some thing\b': 'something',
    r'some where\b': 'somewhere',
    r'some how\b': 'somehow',
    r'some times\b': 'sometimes',
    r'some time\b': 'sometime',
    r'some what\b': 'somewhat',
    r'any one\b': 'anyone',
    r'any thing\b': 'anything',
    r'any where\b': 'anywhere',
    r'any time\b': 'anytime',
    r'any how\b': 'anyhow',
    r'every one\b': 'everyone',
    r'every thing\b': 'everything',
    r'every where\b': 'everywhere',
    r'every day\b': 'everyday',
    r'no body\b': 'nobody',
    r'no thing\b': 'nothing',
    r'no where\b': 'nowhere',
    r'be cause\b': 'because',
    r'be fore\b': 'before',
    r'be yond\b': 'beyond',
    r'be tween\b': 'between',
    r'be neath\b': 'beneath',
    r'be hind\b': 'behind',
    r'be side\b': 'beside',
    r'be sides\b': 'besides',
    r'be came\b': 'became',
    r'be come\b': 'become',
    r'be comes\b': 'becomes',
    r'be coming\b': 'becoming',
    r'under neath\b': 'underneath',
    r'under stand\b': 'understand',
    r'under stood\b': 'understood',
    r'under standing\b': 'understanding',
    r'how ever\b': 'however',
    r'what ever\b': 'whatever',
    r'when ever\b': 'whenever',
    r'where ver\b': 'wherever',
    r'which ever\b': 'whichever',
    r'who ever\b': 'whoever',
    r'other wise\b': 'otherwise',
    r'there fore\b': 'therefore',
    r'to gether\b': 'together',
    r'an other\b': 'another',
    r'any more\b': 'anymore',
    r'to night\b': 'tonight',
    r'to morrow\b': 'tomorrow',
    r'can not\b': 'cannot',
}
# Compile once at import time
_REJOIN_RE = [(re.compile(r'\b' + pat, re.IGNORECASE), repl)
              for pat, repl in _REJOIN_MAP.items()]


def _rejoin_splits(text: str) -> str:
    """Repair compound words that wordninja incorrectly split in a prior pass."""
    for pattern, repl in _REJOIN_RE:
        # Preserve capitalisation: if match starts uppercase, capitalise replacement
        def _sub(m, r=repl):
            return r[0].upper() + r[1:] if m.group()[0].isupper() else r
        text = pattern.sub(_sub, text)
    return text


def clean_text(text: str) -> str:
    """Fix common scraping artefacts (inline-element concatenation, etc.)."""
    if not text:
        return text
    # Repair compound words split by a prior wordninja pass
    text = _rejoin_splits(text)
    # Collapse runs of spaces/tabs to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Insert missing space between lowercase/punctuation and an uppercase letter
    # catches "surelyThat" → "surely That" (adjacent inline tags with no space)
    text = re.sub(r'([a-z,;:!?\'"])([A-Z])', r'\1 \2', text)
    # Insert missing space after sentence-ending period: "word.Next" → "word. Next"
    # Only matches lowercase + period + uppercase to avoid breaking abbreviations (D.W., C.E.O.)
    text = re.sub(r'([a-z]\.)([A-Z])', r'\1 \2', text)
    # Insert missing space after common contractions (both straight and curly apostrophes)
    # "that'sactually" / "that\u2019sactually" → "that's actually"
    text = re.sub(r"(['\u2019](?:s|re|ve|ll|d|t))([a-z])", r'\1 \2', text)
    # Insert missing space between an uppercase letter and a following lowercase run
    # when preceded by a lowercase word: "wordThe" already caught above;
    # catches "Thefuck" style (uppercase start of concatenated word mid-sentence)
    # Only fires when a lowercase letter immediately precedes the uppercase: "isThe"
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
    # Remove space inserted before punctuation by the separator
    text = re.sub(r' ([.,;:!?])', r'\1', text)
    # Normalise line endings
    text = re.sub(r'\r\n?', '\n', text)
    # Split remaining long concatenated tokens using wordninja.
    # Use a negative lookbehind/lookahead to skip tokens inside email addresses
    # or adjacent to digits (e.g. donwin345 → don win345 is wrong).
    if _WORDNINJA:
        # \b ensures we only match whole words, not substrings within hyphenated words.
        # (?<!-) skips tokens after a hyphen (e.g. sing-songed).
        # (?!['\u2019]) skips dialect contractions like headin', lookin', campin'.
        text = re.sub(r"(?<!-)\b[A-Za-z]{6,}\b(?!\d)(?!['\u2019])", lambda m: _maybe_split(m.group()), text)
    return text.strip()


def _show_diff(story_id: str, old: str, new: str) -> None:
    """Print changed lines for a story."""
    old_lines = old.split('\n')
    new_lines = new.split('\n')
    shown = 0
    for o, n in zip(old_lines, new_lines):
        if o != n:
            print(f"  [{story_id}]")
            print(f"    Before: {repr(o[:120])}")
            print(f"    After:  {repr(n[:120])}")
            shown += 1
            if shown >= 3:
                remaining = sum(1 for a, b in zip(old_lines, new_lines) if a != b) - shown
                if remaining:
                    print(f"    ... ({remaining} more changed lines)")
                break


def main():
    """Apply the text clean-up rules to every story's full_text (or one --story), report
    what changed, save through index_store unless --dry-run, and refresh
    full_text_cache.db.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="./output/stories_index.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without saving")
    parser.add_argument("--story", default=None,
                        help="Only process this story ID")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process N stories")
    args = parser.parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Index not found: {index_path}")
        return

    print(f"Loading {index_path} …")
    data = index_store.load_index(index_path, for_write=not args.dry_run)

    stories = data.get("stories", [])
    changed = 0
    processed = 0
    for s in stories:
        if args.story and s.get("id") != args.story:
            continue
        if args.limit and processed >= args.limit:
            break
        ft = s.get("full_text")
        if not ft:
            continue
        processed += 1
        cleaned = clean_text(ft)
        if cleaned != ft:
            changed += 1
            if args.dry_run:
                _show_diff(s.get("id", "?"), ft, cleaned)
            else:
                s["full_text"] = cleaned

    print(f"\nCleaned text in {changed:,} / {processed:,} stories")

    if args.dry_run:
        print("Dry run — no changes saved.")
        return

    if changed == 0:
        print("Nothing to fix.")
        return

    print(f"Saving {index_path} …")
    index_store.save_index(data, index_path)
    print("Saved.")

    # Rebuild SQLite cache
    db_path = index_path.parent / "full_text_cache.db"
    print(f"Rebuilding SQLite cache → {db_path} …")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS texts (id TEXT PRIMARY KEY, full_text TEXT)")
    conn.execute("DELETE FROM texts")
    count = 0
    for s in stories:
        ft = s.get("full_text")
        if ft:
            conn.execute("INSERT OR REPLACE INTO texts VALUES (?,?)", (s["id"], ft))
            count += 1
    conn.commit()
    conn.close()
    print(f"SQLite cache rebuilt: {count:,} stories")


if __name__ == "__main__":
    main()
