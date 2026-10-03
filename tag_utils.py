"""
tag_utils.py — the one place that knows how a story's `auto_tags` is stored.

stories_meta.db keeps auto_tags as a raw JSON string so the browser can load
18k stories quickly; it is parsed on first access and cached back onto the
story dict. Everything that reads tags from a story loaded that way must go
through get_auto_tags() — reading s["auto_tags"] directly can return a string.
"""
from __future__ import annotations
import json


def get_auto_tags(s: dict) -> dict:
    """Return the story's parsed auto_tags dict, parsing it in place if needed."""
    at = s.get("auto_tags")
    if isinstance(at, str):
        at = json.loads(at) if at else {}
        s["auto_tags"] = at
    return at or {}
