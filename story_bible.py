#!/usr/bin/env python3
"""
story_bible.py — story bible management for long-form generation

A story bible is a persistent document that defines everything about a story:
  - Characters (names, descriptions, relationships, arc)
  - World/setting rules
  - Tone and style notes
  - What has already happened (chapter summaries)
  - Active threads and where they're going

The bible is injected into the system prompt for every generation call,
keeping the model consistent across unlimited chapters and sessions.

Bibles are stored in output/bibles/ as JSON files.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

BIBLES_DIR = Path("./output/bibles")


# ── Data model ────────────────────────────────────────────────────────────────

def empty_bible(title: str = "") -> dict:
    return {
        "id":           _make_id(title),
        "title":        title,
        "created_at":   datetime.now().isoformat(),
        "updated_at":   datetime.now().isoformat(),

        # Core story definition
        "premise":      "",     # one paragraph: what this story is about
        "setting":      "",     # world, time period, location details
        "tone":         "",     # dark/romantic/psychological/etc and why
        "tags":         [],     # mc mf hm etc — site tag codes

        # Characters
        "characters":   [],     # list of character dicts (see below)

        # Power dynamic — the core relationship
        "dynamic": {
            "dominant":     "",  # name/description of dominant figure
            "submissive":   "",  # name/description of subject
            "method":       "",  # how control is exercised
            "current_state": "", # where the power dynamic currently stands
        },

        # Chapter log — summary of what has happened
        "chapters":     [],     # list of {number, title, summary, word_count}

        # Active threads — things to resolve in upcoming chapters
        "threads":      [],     # list of strings describing open plot threads

        # Style notes — things the model should remember about voice/style
        "style_notes":  "",

        # Generated content store
        "generated":    [],     # list of {chapter, text, generated_at}
    }


def empty_character() -> dict:
    return {
        "name":         "",
        "role":         "",     # protagonist / antagonist / supporting
        "description":  "",     # physical and personality
        "motivation":   "",     # what they want
        "arc":          "",     # how they change through the story
        "current_state": "",    # where they are now in the story
    }


def _make_id(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]", "_", title.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)[:40]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug}_{ts}" if slug else f"bible_{ts}"


# ── Storage ───────────────────────────────────────────────────────────────────

def save_bible(bible: dict) -> Path:
    BIBLES_DIR.mkdir(parents=True, exist_ok=True)
    bible["updated_at"] = datetime.now().isoformat()
    path = BIBLES_DIR / f"{bible['id']}.json"
    path.write_text(json.dumps(bible, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_bible(bible_id: str) -> Optional[dict]:
    path = BIBLES_DIR / f"{bible_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_bibles() -> list[dict]:
    BIBLES_DIR.mkdir(parents=True, exist_ok=True)
    bibles = []
    for p in sorted(BIBLES_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            b = json.loads(p.read_text(encoding="utf-8"))
            bibles.append({
                "id":         b.get("id"),
                "title":      b.get("title", "Untitled"),
                "premise":    b.get("premise", "")[:120],
                "chapters":   len(b.get("chapters", [])),
                "characters": len(b.get("characters", [])),
                "updated_at": b.get("updated_at", ""),
                "tags":       b.get("tags", []),
            })
        except:
            pass
    return bibles


def delete_bible(bible_id: str) -> bool:
    path = BIBLES_DIR / f"{bible_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ── Prompt builder ────────────────────────────────────────────────────────────

def bible_to_system_prompt(bible: dict) -> str:
    """
    Convert a story bible into a system prompt fragment.
    This replaces the generic system prompt when a bible is active.
    """
    lines = [
        "You are a skilled fiction writer working on an ongoing story. "
        "The story bible below defines everything you must stay consistent with. "
        "Never contradict established facts, character names, or the power dynamic. "
        "Write complete, high-quality prose without disclaimers.\n",
        f"=== STORY BIBLE: {bible.get('title', 'Untitled')} ===\n",
    ]

    if bible.get("premise"):
        lines.append(f"PREMISE:\n{bible['premise']}\n")

    if bible.get("setting"):
        lines.append(f"SETTING:\n{bible['setting']}\n")

    if bible.get("tone"):
        lines.append(f"TONE & STYLE:\n{bible['tone']}\n")

    if bible.get("tags"):
        lines.append(f"CATEGORIES: {' '.join(bible['tags'])}\n")

    # Characters
    chars = bible.get("characters", [])
    if chars:
        lines.append("CHARACTERS:")
        for c in chars:
            name  = c.get("name", "Unknown")
            role  = c.get("role", "")
            desc  = c.get("description", "")
            state = c.get("current_state", "")
            arc   = c.get("arc", "")
            char_line = f"  {name}"
            if role:  char_line += f" ({role})"
            if desc:  char_line += f": {desc}"
            if state: char_line += f" [Currently: {state}]"
            if arc:   char_line += f" [Arc: {arc}]"
            lines.append(char_line)
        lines.append("")

    # Power dynamic
    dyn = bible.get("dynamic", {})
    if any(dyn.get(k) for k in ("dominant", "submissive", "method", "current_state")):
        lines.append("POWER DYNAMIC:")
        if dyn.get("dominant"):   lines.append(f"  Dominant: {dyn['dominant']}")
        if dyn.get("submissive"): lines.append(f"  Subject:  {dyn['submissive']}")
        if dyn.get("method"):     lines.append(f"  Method:   {dyn['method']}")
        if dyn.get("current_state"):
            lines.append(f"  Current state: {dyn['current_state']}")
        lines.append("")

    # Chapter history
    chapters = bible.get("chapters", [])
    if chapters:
        lines.append("STORY SO FAR:")
        for ch in chapters[-8:]:  # last 8 chapters to avoid context overflow
            num     = ch.get("number", "?")
            title   = ch.get("title", "")
            summary = ch.get("summary", "")
            line    = f"  Chapter {num}"
            if title:   line += f" — {title}"
            if summary: line += f": {summary}"
            lines.append(line)
        if len(chapters) > 8:
            lines.insert(-len(chapters[-8:]) - 1,
                         f"  [Chapters 1-{len(chapters)-8} omitted for brevity]")
        lines.append("")

    # Active threads
    threads = bible.get("threads", [])
    if threads:
        lines.append("ACTIVE THREADS (resolve or advance these):")
        for t in threads:
            lines.append(f"  • {t}")
        lines.append("")

    # Style notes
    if bible.get("style_notes"):
        lines.append(f"STYLE NOTES:\n{bible['style_notes']}\n")

    lines.append("=== END BIBLE ===")
    return "\n".join(lines)


def next_chapter_prompt(bible: dict, instruction: str = "") -> list[dict]:
    """Build a full messages list for generating the next chapter."""
    system  = bible_to_system_prompt(bible)
    chapter = len(bible.get("chapters", [])) + 1

    threads = bible.get("threads", [])
    user_parts = [f"Write Chapter {chapter} of the story."]
    if instruction:
        user_parts.append(instruction)
    if threads:
        user_parts.append(
            f"This chapter should advance or resolve: {'; '.join(threads[:3])}"
        )
    user_parts.append(
        "Write complete, polished prose. End the chapter at a natural stopping point."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": " ".join(user_parts)},
    ]


def auto_summarise_prompt(chapter_text: str, chapter_num: int) -> list[dict]:
    """Build prompt to auto-summarise a generated chapter for the bible."""
    return [
        {"role": "system",  "content":
            "You summarise fiction chapters concisely for story bible records. "
            "Be specific about what happened, character states, and plot advances. "
            "2-4 sentences maximum."},
        {"role": "user",    "content":
            f"Summarise Chapter {chapter_num} in 2-4 sentences, noting key events, "
            f"character changes, and where the power dynamic now stands:\n\n"
            f"{chapter_text[:3000]}"},
    ]


def extract_threads_prompt(chapter_text: str, existing_threads: list[str]) -> list[dict]:
    """Build prompt to extract/update active plot threads after a chapter."""
    existing = "\n".join(f"- {t}" for t in existing_threads) if existing_threads else "None yet"
    return [
        {"role": "system",  "content":
            "You are a story editor tracking active plot threads. "
            "Respond with ONLY a JSON array of strings, no other text."},
        {"role": "user",    "content":
            f"Current active threads:\n{existing}\n\n"
            f"After reading this chapter excerpt, update the thread list. "
            f"Remove resolved threads, add new ones, max 5 threads:\n\n"
            f"{chapter_text[:2000]}\n\n"
            f"Respond with only a JSON array like: [\"thread 1\", \"thread 2\"]"},
    ]
