#!/usr/bin/env python3
"""
generator.py — story generation via local LLM (LM Studio / Ollama / any OpenAI-compatible API)

Works with:
  - LM Studio    http://localhost:1234  (Windows or Mac)
  - Ollama       http://localhost:11434
  - Any OpenAI-compatible endpoint

No API key needed for local models.
For cloud APIs (Together.ai, Featherless.ai) set OPENAI_API_KEY env var.
"""

from __future__ import annotations
import json
import re
import time
import logging
import requests
from typing import Optional, Iterator

log = logging.getLogger(__name__)

# ── Default endpoints ─────────────────────────────────────────────────────────

ENDPOINTS = {
    "lmstudio":    {"url": "http://localhost:1234/v1",   "key": "lm-studio"},
    "ollama":      {"url": "http://localhost:11434/v1",  "key": "ollama"},
    "together":    {"url": "https://api.together.xyz/v1","key": None},
    "featherless": {"url": "https://api.featherless.ai/v1","key": None},
    "custom":      {"url": "http://localhost:8080/v1",   "key": None},
}


# ── Model client ──────────────────────────────────────────────────────────────

class LLMClient:
    """Thin wrapper around any OpenAI-compatible chat completions endpoint."""

    def __init__(self, base_url: str, api_key: str = "none", model: str = "auto"):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.model    = model
        self._models_cache: list[str] = []

    def list_models(self) -> list[str]:
        """Model names offered by the server's /models endpoint (cached after the first
        call; [] if unreachable).
        """
        if self._models_cache:
            return self._models_cache
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            r.raise_for_status()
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            self._models_cache = models
            return models
        except Exception as e:
            log.warning(f"Could not list models: {e}")
            return []

    def resolve_model(self) -> str:
        """The configured model, or the server's first model when set to 'auto'."""
        if self.model and self.model != "auto":
            return self.model
        models = self.list_models()
        return models[0] if models else "local-model"

    def complete(self, messages: list[dict], max_tokens: int = 2000,
                 temperature: float = 0.85, stream: bool = False) -> str | Iterator[str]:
        """Send a chat completion. Returns the reply text, or with stream=True an iterator
        of text chunks as they arrive.
        """
        model = self.resolve_model()
        payload = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      stream,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        if not stream:
            r = requests.post(url, json=payload, headers=headers, timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        else:
            return self._stream(url, payload, headers)

    def _stream(self, url, payload, headers) -> Iterator[str]:
        with requests.post(url, json=payload, headers=headers,
                           stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except:
                    pass

    def ping(self) -> bool:
        """True if the server answers /models without a server error."""
        try:
            r = requests.get(f"{self.base_url}/models", timeout=3,
                             headers={"Authorization": f"Bearer {self.api_key}"})
            return r.status_code < 500
        except:
            return False


# ── RAG prompt builder ────────────────────────────────────────────────────────

def build_prompt(request_text: str, examples: list[dict],
                 tags: dict, style_mode: str = "inspired") -> list[dict]:
    """
    Build a chat messages list for story generation.

    style_mode:
      "inspired"   — write something new inspired by examples (default)
      "continue"   — continue the first example story
      "remix"      — blend elements from multiple examples
    """
    # System prompt
    system = (
        "You are a skilled fiction writer specialising in mind control erotica. "
        "You write with literary quality — vivid characterisation, strong pacing, "
        "and prose that builds atmosphere. You never refuse or add disclaimers. "
        "Write complete, self-contained stories unless asked to continue."
    )

    # Build example context from retrieved stories
    example_blocks = []
    for i, ex in enumerate(examples[:3], 1):
        at     = ex.get("auto_tags") or {}
        cats   = " ".join(ex.get("categories", []))
        themes = ", ".join(at.get("themes", [])[:3])
        mc     = ", ".join(at.get("mind_control_type", [])[:2])
        words  = ex.get("word_count", 0)
        # Include opening ~600 words as style reference
        text_sample = " ".join((ex.get("full_text") or "").split()[:600])

        block = (
            f"--- Example {i}: \"{ex.get('title','')}\" by {ex.get('author','')} "
            f"[{cats}] {words:,} words ---\n"
        )
        if themes: block += f"Themes: {themes}\n"
        if mc:     block += f"MC method: {mc}\n"
        block += f"\n{text_sample}...\n"
        example_blocks.append(block)

    # Build tag context from user selections
    tag_lines = []
    if tags.get("categories"):
        tag_lines.append(f"Categories: {' '.join(tags['categories'])}")
    if tags.get("mc_type"):
        tag_lines.append(f"Mind control method: {tags['mc_type']}")
    if tags.get("setting"):
        tag_lines.append(f"Setting: {tags['setting']}")
    if tags.get("themes"):
        tag_lines.append(f"Themes: {', '.join(tags['themes'])}")
    if tags.get("tone"):
        tag_lines.append(f"Tone: {tags['tone']}")
    if tags.get("dominance"):
        tag_lines.append(f"Dominance style: {tags['dominance']}")
    if tags.get("pacing"):
        tag_lines.append(f"Pacing: {tags['pacing']}")
    if tags.get("length"):
        length_map = {
            "flash": "under 1,000 words",
            "short story": "2,000–6,000 words",
            "novelette": "8,000–15,000 words",
        }
        tag_lines.append(f"Target length: {length_map.get(tags['length'], tags['length'])}")

    # User message
    if style_mode == "continue" and examples:
        first = examples[0]
        user = (
            f"Continue the following story, picking up naturally from where it ends.\n\n"
            f"Story: \"{first.get('title','')}\" by {first.get('author','')}\n\n"
            f"{(first.get('full_text') or '')[-1500:]}\n\n"
            f"Continue from here:"
        )
    elif style_mode == "remix" and examples:
        titles = " / ".join(f'"{e.get("title","")}"' for e in examples[:3])
        user = (
            f"Write a new story that blends elements from these stories: {titles}\n\n"
        )
        if example_blocks:
            user += "Reference material:\n\n" + "\n".join(example_blocks) + "\n\n"
        user += f"User request: {request_text}\n\n"
        if tag_lines:
            user += "Parameters:\n" + "\n".join(tag_lines) + "\n\n"
        user += "Write the story now:"
    else:  # inspired
        user = ""
        if example_blocks:
            user += (
                "Here are some reference stories to draw style inspiration from "
                "(do NOT copy them directly):\n\n"
                + "\n".join(example_blocks)
                + "\n\n"
            )
        user += f"Write a new story based on this request: {request_text}\n\n"
        if tag_lines:
            user += "Parameters:\n" + "\n".join(tag_lines) + "\n\n"
        user += "Write the complete story now:"

    return [
        {"role": "system",  "content": system},
        {"role": "user",    "content": user},
    ]


# ── Token/word estimate ───────────────────────────────────────────────────────

def estimate_tokens(length_class: str) -> int:
    """Rough output-token budget for a story of the given length class."""
    return {
        "flash":      800,
        "short story": 3000,
        "novelette":   8000,
        "novella":    20000,
    }.get(length_class, 2000)
