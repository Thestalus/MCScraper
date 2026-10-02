#!/usr/bin/env python3
"""
Audit mcstories scraper output.

Usage:
    python3 audit_scrape.py
    python3 audit_scrape.py --output ./output
    python3 audit_scrape.py --show-errors 20
    python3 audit_scrape.py --write-retry-list retry_urls.txt
"""

import argparse
import json
from collections import Counter
from pathlib import Path


DATASET_FILES = [
    "training_raw_pretraining.jsonl",
    "training_instruction_response.jsonl",
    "training_completion.jsonl",
]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def load_index(index_path: Path) -> dict:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "stories" not in data or not isinstance(data["stories"], list):
        raise RuntimeError(
            f"Unexpected index format in {index_path}. Expected a dict with a 'stories' list."
        )
    return data


def classify_story(story: dict) -> str:
    fetch_error = story.get("fetch_error")
    full_text = story.get("full_text") or ""
    if fetch_error:
        return "fetch_error"
    if not isinstance(full_text, str) or not full_text.strip():
        return "empty_text"
    return "success"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit mcstories scrape output.")
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--show-errors", type=int, default=0, help="Show up to N failed/empty stories")
    parser.add_argument(
        "--write-retry-list",
        default=None,
        help="Write URLs with fetch_error or empty_text to this file",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    index_path = out_dir / "stories_index.json"

    if not index_path.exists():
        raise SystemExit(f"Index file not found: {index_path}")

    data = load_index(index_path)
    stories = data["stories"]

    statuses = Counter()
    urls = []
    duplicates = 0
    seen = set()
    retry_urls = []
    failed_or_empty = []

    for story in stories:
        status = classify_story(story)
        statuses[status] += 1

        url = story.get("story_url")
        if url:
            urls.append(url)
            if url in seen:
                duplicates += 1
            else:
                seen.add(url)

        if status in {"fetch_error", "empty_text"}:
            if url:
                retry_urls.append(url)
            failed_or_empty.append(story)

    print("MCStories scrape audit")
    print("=" * 24)
    print(f"Output directory:        {out_dir}")
    print(f"Index file:              {index_path}")
    print(f"Stories indexed:         {len(stories)}")
    print(f"Successful stories:      {statuses['success']}")
    print(f"Fetch errors:            {statuses['fetch_error']}")
    print(f"Empty text:              {statuses['empty_text']}")
    print(f"Duplicate story URLs:    {duplicates}")

    print("\nDataset rows")
    print("-" * 24)
    for name in DATASET_FILES:
        path = out_dir / name
        print(f"{name:<38} {count_lines(path)}")

    if args.write_retry_list:
        retry_path = Path(args.write_retry_list)
        retry_path.write_text("\n".join(retry_urls), encoding="utf-8")
        print(f"\nWrote retry list:        {retry_path} ({len(retry_urls)} URLs)")

    if args.show_errors:
        print(f"\nFailed / empty stories (up to {args.show_errors})")
        print("-" * 48)
        for story in failed_or_empty[: args.show_errors]:
            title = story.get("title") or "<untitled>"
            url = story.get("story_url") or "<no url>"
            err = story.get("fetch_error")
            status = classify_story(story)
            print(f"[{status}] {title}")
            print(f"URL: {url}")
            if err:
                print(f"Reason: {err}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
