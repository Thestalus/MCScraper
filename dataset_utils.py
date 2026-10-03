"""
dataset_utils.py — reproducibility helpers for building fine-tuning datasets.

Used by train.py (story LoRA) and extract_chatbot_data.py / train_chat.py
(chat LoRA):

  * is_validation(story_id) — a stable, story-level train/validation split.
    It hashes the story ID, so a story always lands on the same side (all of
    its chunks or dialogue turns together), and adding new stories never
    reshuffles existing ones.
  * write_manifest() — records what produced a dataset (seed, split, input
    fingerprints, code version, package versions, file hashes) next to it, so
    two training runs can be compared and a dataset can be rebuilt exactly.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROJECT_DIR = Path(__file__).parent


def is_validation(story_id: str, fraction: float, seed: int) -> bool:
    """True if this story belongs to the validation split."""
    if fraction <= 0:
        return False
    h = hashlib.sha1(f"{seed}:{story_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < fraction


def file_sha256(path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(path) -> dict | None:
    """Cheap identity for big inputs (the 1 GB index): size + mtime, not a hash."""
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    return {"path": str(path), "size": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=PROJECT_DIR, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions(names=("torch", "transformers", "trl", "peft", "unsloth",
                            "datasets", "accelerate", "bitsandbytes")) -> dict:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_manifest(path, *, outputs: dict, inputs: dict, settings: dict, counts: dict) -> dict:
    """Write a JSON manifest describing how a dataset was built.

    outputs:  {name: file path} — hashed (sha256) so the exact files can be verified
    inputs:   {name: file path} — fingerprinted (size + mtime), cheap for big files
    settings: seed, split fraction, filters, model … anything that changes the data
    counts:   record / story counts per split
    """
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "settings": settings,
        "counts": counts,
        "inputs": {k: file_fingerprint(v) for k, v in inputs.items()},
        "outputs": {k: {"path": str(v), "sha256": file_sha256(v)} for k, v in outputs.items()},
        "code": {"tag_trust.py": file_sha256(PROJECT_DIR / "tag_trust.py")},
        "packages": package_versions(),
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
