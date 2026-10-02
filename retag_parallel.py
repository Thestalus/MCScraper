#!/usr/bin/env python3
"""Retag all stories in the index using full-text regex, parallelized."""
import json, sys, time, tempfile, os
from multiprocessing import Pool, cpu_count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

INDEX_PATH = "output/stories_index.json"


def tag_story(story):
    """Worker: returns (id, new_auto_tags) or None if no full_text."""
    if not story or not story.get("full_text"):
        return None
    from tagger import regex_tag
    return (story["id"], regex_tag(story))


def parallel_regex_tag(stories: list, chunksize: int = 200) -> dict:
    """
    Run regex_tag on a list of stories in parallel.
    Returns {story_id: tags_dict} for all stories that have full_text.
    Importable by other modules (eval_tagger, etc.) to avoid serial loops.
    """
    valid = [s for s in stories if s and s.get("full_text")]
    results = {}
    with Pool(cpu_count()) as pool:
        for result in pool.imap_unordered(tag_story, valid, chunksize=chunksize):
            if result:
                results[result[0]] = result[1]
    return results


def main():
    print("Loading index...")
    with open(INDEX_PATH) as f:
        data = json.load(f)

    stories = data["stories"]
    valid = [s for s in stories if s and s.get("full_text")]
    print(f"{len(valid)} stories with full text (of {len(stories)} total)")

    n_workers = cpu_count()
    print(f"Using {n_workers} workers...")
    t0 = time.time()

    results = {}
    chunk = 200
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(tag_story, valid, chunksize=chunk)):
            if result:
                results[result[0]] = result[1]
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(valid) - i - 1) / rate
                print(f"  {i+1}/{len(valid)}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    print(f"\nTagged {len(results)} stories in {time.time()-t0:.1f}s")

    # Dimensions that regex owns exclusively (no ML layer adds tags here).
    # These are always overwritten; all other dims preserve ML-sourced tags.
    REGEX_ONLY_DIMS = {
        "site_tags_expanded", "site_tag_groups", "content_type",
        "pov", "length_class", "prevalence",
    }

    # Write results back, rebuilding _tag_sources for regex from scratch so
    # stale entries can't falsely satisfy the AND-gate.
    updated = 0
    for s in stories:
        if s and s.get("id") in results:
            if s.get("auto_tags") is None:
                s["auto_tags"] = {}
            tags = results[s["id"]]

            # Strip all existing "regex" sources — we're rebuilding from current results.
            sources = {
                tag: [src for src in src_list if src != "regex"]
                for tag, src_list in (s["auto_tags"].get("_tag_sources") or {}).items()
            }
            sources = {tag: src for tag, src in sources.items() if src}

            for dim, tag_list in tags.items():
                if dim.startswith("_") or not isinstance(tag_list, list):
                    continue
                if dim in REGEX_ONLY_DIMS:
                    s["auto_tags"][dim] = tag_list
                else:
                    # Preserve tags that have a non-regex source (SetFit/Svengali).
                    existing = s["auto_tags"].get(dim, [])
                    kept = [t for t in existing if sources.get(t)]
                    s["auto_tags"][dim] = list(dict.fromkeys(tag_list + kept))
                for tag in tag_list:
                    if "regex" not in sources.get(tag, []):
                        sources.setdefault(tag, []).append("regex")

            s["auto_tags"]["_tag_sources"] = sources
            updated += 1

    print(f"Saving index ({updated} stories updated)...")
    out = Path(INDEX_PATH)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=out.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, out)
    print("Done.")


if __name__ == "__main__":
    main()
