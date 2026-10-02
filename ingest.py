#!/usr/bin/env python3
"""
ingest.py — Universal local content ingestion for the MC Archive.

Converts plain text, EPUB, PDF, DOCX, and OCR screenshots into properly
formatted HTML story entries and patches them into stories_index.json.

Supported source formats:
  .txt            Plain text file(s) — with optional StoriesOnline header
  .epub           EPUB ebook — Dublin Core metadata + chapters
  .pdf            PDF file(s) — via PyMuPDF (already installed)
  .docx           Word documents — via zip/XML extraction
  .png/.jpg/...   Screenshot/scan — via Tesseract OCR (optional, see below)

OCR setup (one-time, only needed for images):
  brew install tesseract
  .venv/bin/pip install pytesseract pillow

Usage:
  # Single file → print JSON snippet
  .venv/bin/python3 ingest.py story.txt

  # Single file → patch directly into index
  .venv/bin/python3 ingest.py story.txt --patch

  # Override metadata
  .venv/bin/python3 ingest.py story.txt --title "My Story" --author "Jane" --patch

  # Multiple files = multiple chapters of one story
  .venv/bin/python3 ingest.py ch1.txt ch2.txt ch3.txt --title "Long Story" --patch

  # EPUB (metadata auto-extracted from OPF)
  .venv/bin/python3 ingest.py book.epub --patch

  # PDF folder (auto-sorted, grouped by series name)
  .venv/bin/python3 ingest.py ./StoryFolder/ --author "Jane" --patch

  # Batch ingest entire author folder (skips already-indexed stories)
  .venv/bin/python3 ingest.py --folder /path/to/Pan/ --author "Pan" --patch
  .venv/bin/python3 ingest.py --folder /path/to/Pan/ --author "Pan" --dry-run

  # OCR screenshot(s)
  .venv/bin/python3 ingest.py scan.png --title "Scanned Story" --author "Jane" --patch

After patching the index run:
  make tag && make post-update
"""

from __future__ import annotations

import re
import sys
import json
import html as htmlmod
import argparse
import unicodedata
import zipfile
from pathlib import Path
from typing import Optional
from collections import defaultdict

# ── Optional imports ───────────────────────────────────────────────────────────

try:
    import fitz as _fitz       # PyMuPDF
    _HAVE_FITZ = True
except ImportError:
    _HAVE_FITZ = False

try:
    from bs4 import BeautifulSoup as _BS
    _HAVE_BS4 = True
except ImportError:
    _HAVE_BS4 = False

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_INDEX  = "./output/stories_index.json"
DEFAULT_SOURCE = "local"

_TEXT_EXTS  = {".txt", ".md"}
_PDF_EXTS   = {".pdf"}
_EPUB_EXTS  = {".epub"}
_DOCX_EXTS  = {".docx"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"}
_ALL_EXTS   = _TEXT_EXTS | _PDF_EXTS | _EPUB_EXTS | _DOCX_EXTS | _IMAGE_EXTS

# macOS screenshot filename: "Screenshot 2024-01-15 at 10.23.45 AM.png"
_MACOS_SCREENSHOT_RE = re.compile(
    r"Screenshot (\d{4}-\d{2}-\d{2}) at (\d+)\.(\d{2})\.(\d{2})\s*(AM|PM)",
    re.IGNORECASE,
)

def _image_sort_key(path: Path) -> tuple:
    """Sort key for image files: parses macOS screenshot timestamps, falls back to filename."""
    m = _MACOS_SCREENSHOT_RE.search(path.name)
    if m:
        date, h, mins, secs, ampm = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5).upper()
        if ampm == "PM" and h != 12:
            h += 12
        elif ampm == "AM" and h == 12:
            h = 0
        return (date, h, mins, secs)
    return (path.name, 0, 0, 0)

# Sentence-terminal characters — a line ending with one of these is a real
# paragraph end; otherwise it's word-wrap and gets joined to the next line.
_SENT_END = re.compile(r'[.!?"\u201d\u2019\u2026\u2025…]\s*$')

# Chapter-break patterns in plain text
_CHAPTER_BREAK = re.compile(
    r'^\s*(?:'
    r'\*\s*\*\s*\*'           # * * *
    r'|[-~=]{3,}'             # --- ~~~ ===
    r'|chapter\s+\d+'         # Chapter 1 / CHAPTER 1
    r'|part\s+\d+'            # Part 1
    r'|#{1,3}\s+\S'           # Markdown headings
    r')\s*$',
    re.IGNORECASE,
)

# Structural header lines to skip (chapter/part titles, TOC entries, not prose)
_HEADER_LINE = re.compile(
    r'^\s*(?:'
    r'chapter|part|section|prologue|epilogue|interlude|act'  # structural labels
    r'|table of contents|contents'                            # TOC heading
    r')\b.{0,60}$',
    re.IGNORECASE,
)

# Trailing chapter number in a filename stem: "Mad Monday 1", "Sex_Virus_1", etc.
_SERIES_NUM = re.compile(
    r'\s*[-–_]?\s*(?:part\s*|chapter\s*|ch\.?\s*)?(\d{1,3})\s*$',
    re.IGNORECASE,
)

# SOL (StoriesOnline) metadata header — matches any "Key: value" line where
# the key is 1–4 words with no sentence punctuation (i.e. a metadata field).
_SOL_META_LINE = re.compile(
    r'^((?:[A-Za-z][A-Za-z]*\s*){1,4})\s*:\s*(.*)$',
)

# Lines in a TABLE OF CONTENTS / chapter list — skip entirely
_TOC_LINE = re.compile(
    r'^\s*(?:table of contents|contents)\s*$',
    re.IGNORECASE,
)


# ── Text formatting ───────────────────────────────────────────────────────────

def _clean_raw_text(text: str) -> str:
    """Normalise encoding and whitespace before paragraph detection."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces per line
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()


def _lines_to_paragraphs(lines: list[str]) -> list[str]:
    """
    Convert a list of lines into semantic paragraphs.

    Lines separated by a blank line are distinct paragraphs.
    Within a non-blank block, a line ending with sentence-terminal punctuation
    closes the paragraph; otherwise it's word-wrap and joins the next line.
    """
    paragraphs: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            para = " ".join(current).strip()
            if para:
                paragraphs.append(para)
            current.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if _HEADER_LINE.match(line):
            flush()
            continue
        current.append(line)
        if _SENT_END.search(line):
            flush()

    flush()
    return paragraphs


def paragraphs_to_html(paragraphs: list[str]) -> str:
    return "\n".join(f"<p>{htmlmod.escape(p)}</p>" for p in paragraphs if p.strip())


def text_block_to_html(text: str) -> str:
    """Convert a raw text block (one chapter) to HTML paragraphs."""
    text = _clean_raw_text(text)
    paras = _lines_to_paragraphs(text.split("\n"))
    return paragraphs_to_html(paras)


def split_into_chapters(text: str) -> list[str]:
    """
    Split a multi-chapter plain text into a list of chapter strings.

    Filters out chunks that are pure structural skeleton (TOC entries,
    lone chapter headings, etc.) — requires at least 30 words of real prose.
    """
    text = _clean_raw_text(text)
    chapters: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if _CHAPTER_BREAK.match(line):
            if current:
                chapters.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        chapters.append("\n".join(current).strip())

    # Filter: keep only chunks with enough actual prose (≥ 30 words)
    real = [c for c in chapters if len(c.split()) >= 30]
    return real or [text]


def format_chapters_as_html(chapters: list[str]) -> str:
    """Convert chapter text strings to the archive's HTML chapter format."""
    parts = [text_block_to_html(ch) for ch in chapters if ch.strip()]
    return "\n<!-- BREAK -->\n".join(parts)


# ── SOL header parser ─────────────────────────────────────────────────────────

def _parse_sol_header(text: str) -> tuple[dict, str]:
    """
    Parse a StoriesOnline-style export header from the top of a txt file.

    Returns (metadata_dict, story_body_text).
    The header ends at the first double-blank-line after the metadata block.
    """
    lines = text.lstrip("\ufeff").split("\n")  # strip BOM if present
    meta: dict[str, str] = {}
    story_lines: list[str] = []
    in_header = True
    header_ended = False
    blank_count = 0
    i = 0

    # First non-blank line = title
    # Second non-blank line = "by Author"
    title_line = ""
    author_line = ""
    header_line_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if in_header and not header_ended:
            if not stripped:
                blank_count += 1
                # Header ends after metadata key-value lines + a blank line
                if meta and blank_count >= 2:
                    header_ended = True
                continue
            else:
                blank_count = 0

            # Try structured metadata
            m = _SOL_META_LINE.match(stripped)
            if m:
                meta[m.group(1).lower().replace(" ", "_")] = m.group(2).strip()
                header_line_count += 1
                continue

            # First two non-blank, non-meta lines = title + "by Author"
            if not title_line:
                title_line = stripped
                continue
            if not author_line and stripped.lower().startswith("by "):
                author_line = stripped[3:].strip()
                continue

            # If we hit a non-meta, non-title/author line and have meta → header is done
            if header_line_count > 0:
                header_ended = True
                story_lines.append(line)
            else:
                # No meta found — this is just regular prose
                in_header = False
                story_lines.append(line)
        else:
            story_lines.append(line)

    if title_line:
        meta.setdefault("title", title_line)
    if author_line:
        meta.setdefault("author", author_line)

    body = "\n".join(lines[i:] if not story_lines else story_lines)
    return meta, body


# ── Source extractors ─────────────────────────────────────────────────────────

class StoryData:
    """Intermediate representation before building the final index entry."""
    def __init__(self):
        self.title:       str = ""
        self.author:      str = ""
        self.description: str = ""
        self.story_url:   str = ""
        self.author_url:  str = ""
        self.added_date:  str = ""
        self.categories:  list[str] = []
        self.chapter_texts: list[str] = []   # raw text, not yet HTML


def extract_txt(paths: list[Path]) -> StoryData:
    sd = StoryData()

    if len(paths) == 1:
        raw = paths[0].read_text(encoding="utf-8-sig", errors="replace")
        meta, body = _parse_sol_header(raw)

        sd.title       = meta.get("title", "")
        sd.author      = meta.get("author", "")
        sd.description = meta.get("summary", "")
        sd.story_url   = meta.get("story_url", "")
        sd.author_url  = meta.get("author_url", "")

        # Convert SOL date "2019-10-19 23:34:50" → "19 October 2019"
        raw_date = meta.get("published", meta.get("updated", ""))
        sd.added_date = _parse_sol_date(raw_date)

        # Genre → categories (SOL genre field: "Mind Control / Romance")
        genre = meta.get("genre", "")
        if genre:
            sd.categories = [g.strip() for g in re.split(r'[,/]', genre) if g.strip()]

        # Strip repeated story title from top of body (common in SOL exports)
        body_stripped = body.lstrip()
        if sd.title and body_stripped.lower().startswith(sd.title.lower()):
            body = body_stripped[len(sd.title):].lstrip()

        # Use SOL chapter count to decide splitting strategy:
        # if SOL says 1 chapter, treat * * * as scene breaks (keep one block)
        sol_chapters = int(meta.get("chapters", "0") or "0")
        if sol_chapters <= 1:
            sd.chapter_texts = [_clean_raw_text(body)]
        else:
            sd.chapter_texts = split_into_chapters(body)
    else:
        # Multiple files = one chapter each
        for p in paths:
            raw = p.read_text(encoding="utf-8-sig", errors="replace")
            _, body = _parse_sol_header(raw)
            sd.chapter_texts.append(_clean_raw_text(body))

    if not sd.description and sd.chapter_texts:
        sd.description = _auto_description(sd.chapter_texts[0])
    return sd


def extract_pdf(paths: list[Path]) -> StoryData:
    if not _HAVE_FITZ:
        sys.exit("PyMuPDF not installed. Run: .venv/bin/pip install pymupdf")
    sd = StoryData()
    for path in sorted(paths, key=lambda p: p.name):
        doc = _fitz.open(str(path))
        pages = []
        for page in doc:
            t = page.get_text()
            if len(t.split()) > 10:
                pages.append(t.strip())
        doc.close()
        if pages:
            sd.chapter_texts.append("\n\n".join(pages))
    if not sd.description and sd.chapter_texts:
        sd.description = _auto_description(sd.chapter_texts[0])
    return sd


def extract_epub(path: Path) -> StoryData:
    if not _HAVE_BS4:
        sys.exit("BeautifulSoup4 not installed. Run: .venv/bin/pip install beautifulsoup4")
    if not zipfile.is_zipfile(path):
        sys.exit(f"{path} does not look like a valid EPUB")

    sd = StoryData()
    opf_title = ""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        opf_path = _epub_find_opf(zf, names)
        ordered_items: list[str] = []

        if opf_path:
            opf_xml = zf.read(opf_path).decode("utf-8", errors="replace")
            soup = _BS(opf_xml, "xml")
            title_tag = soup.find("dc:title")
            if title_tag:
                opf_title = title_tag.get_text().strip()
            creator_tag = soup.find("dc:creator")
            if creator_tag:
                sd.author = creator_tag.get_text().strip()

            spine = soup.find("spine")
            manifest = soup.find("manifest")
            if spine and manifest:
                id_to_href = {
                    item.get("id"): item.get("href", "")
                    for item in manifest.find_all("item")
                    if "html" in item.get("media-type", "")
                }
                opf_dir = str(Path(opf_path).parent)
                for itemref in spine.find_all("itemref"):
                    href = id_to_href.get(itemref.get("idref", ""), "")
                    if href:
                        full = (Path(opf_dir) / href).as_posix() if opf_dir != "." else href
                        ordered_items.append(full)

        if not ordered_items:
            ordered_items = sorted(n for n in names
                                   if n.lower().endswith((".html", ".xhtml", ".htm")))

        for item_path in ordered_items:
            matched = _epub_resolve_path(item_path, names)
            if not matched:
                continue
            try:
                text = _epub_html_to_text(zf.read(matched))
                if len(text.split()) > 30:
                    sd.chapter_texts.append(text)
            except Exception:
                continue

    if not sd.chapter_texts:
        sys.exit(f"Could not extract text from {path}")

    # OPF title is often a generic container name ("Pan's Writing", "Author's Stories")
    # — prefer it only when it looks specific (doesn't contain author name or "writing")
    if opf_title and not _is_generic_epub_title(opf_title, sd.author):
        sd.title = opf_title

    # Try to parse a SOL-style title/author header from the first chapter body
    first_meta, first_body = _parse_sol_header(sd.chapter_texts[0])
    if first_meta.get("title"):
        sd.title = sd.title or first_meta["title"]
        sd.author = sd.author or first_meta.get("author", "")
        sd.description = sd.description or first_meta.get("summary", "")
        if first_body.strip():
            sd.chapter_texts[0] = first_body
    elif not sd.title:
        # Strip bare "Title\nby Author" from top of body
        sd.chapter_texts[0] = _strip_inline_header(sd.chapter_texts[0], sd)

    if not sd.description:
        sd.description = _auto_description(sd.chapter_texts[0])
    return sd


def _is_generic_epub_title(title: str, author: str) -> bool:
    """Return True if the OPF title looks like a generic container name."""
    tl = title.lower()
    if author and author.lower().split()[0] in tl:
        return True   # "Pan's Writing", "Pan Stories"
    generic_words = {"writing", "writings", "stories", "collection", "works", "anthology"}
    return bool(generic_words & set(tl.split()))


_FRONT_MATTER_SKIP = re.compile(
    r'\b(?:copyright|all rights reserved|licensed for your personal|'
    r'this ebook|ebook may not|purchase.*copy|depicting.*fiction|'
    r'characters depicted|18 years of age)\b',
    re.IGNORECASE,
)

def _strip_front_matter(text: str) -> str:
    """
    Skip copyright/license paragraphs that precede the actual story in ebooks.
    Walks paragraphs forward until one has > 20 words and no copyright keywords.
    """
    paras = re.split(r'\n\n+', text.strip())
    for i, para in enumerate(paras):
        if len(para.split()) > 20 and not _FRONT_MATTER_SKIP.search(para):
            return "\n\n".join(paras[i:])
    return text


def _strip_inline_header(text: str, sd: StoryData) -> str:
    """
    Strip a bare 'Title\\nby Author' header from the top of extracted body text.
    Updates sd.title and sd.author as side-effects when found.
    """
    lines = text.lstrip().split("\n")
    i = 0
    # Skip blank lines
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return text
    first_nonblank = lines[i].strip()
    # Check if next non-blank line is "by Author"
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    second_nonblank = lines[j].strip() if j < len(lines) else ""

    if (not _SENT_END.search(first_nonblank) and len(first_nonblank.split()) <= 8
            and second_nonblank.lower().startswith("by ")):
        if not sd.title:
            sd.title = first_nonblank
        if not sd.author:
            sd.author = second_nonblank[3:].strip()
        return "\n".join(lines[j + 1:]).lstrip()

    # Just a lone short title line with no "by"
    if (not _SENT_END.search(first_nonblank) and len(first_nonblank.split()) <= 6
            and first_nonblank == (sd.title or "")):
        return "\n".join(lines[i + 1:]).lstrip()

    return text


def extract_docx(paths: list[Path]) -> StoryData:
    """Extract text from one or more .docx files using zip/XML parsing."""
    sd = StoryData()
    for path in sorted(paths, key=lambda p: p.name):
        try:
            with zipfile.ZipFile(path) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  Warning: could not read {path.name}: {e}", file=sys.stderr)
            continue

        # Extract text from <w:t> elements, paragraph breaks at <w:p>
        paragraphs: list[str] = []
        for para_xml in re.split(r'<w:p[ >]', xml):
            texts = re.findall(r'<w:t(?:\s[^>]*)?>([^<]*)</w:t>', para_xml)
            line = "".join(texts).strip()
            if line:
                paragraphs.append(line)

        chapter_text = "\n\n".join(paragraphs)
        if len(chapter_text.split()) > 20:
            sd.chapter_texts.append(chapter_text)

    if sd.chapter_texts:
        # Strip copyright/front matter, then inline title/author header
        sd.chapter_texts[0] = _strip_front_matter(sd.chapter_texts[0])
        sd.chapter_texts[0] = _strip_inline_header(sd.chapter_texts[0], sd)

    if not sd.description and sd.chapter_texts:
        sd.description = _auto_description(sd.chapter_texts[0])
    return sd


def extract_images(paths: list[Path]) -> StoryData:
    try:
        import pytesseract
        from PIL import Image as _Image
    except ImportError:
        sys.exit(
            "pytesseract/Pillow not installed:\n"
            "  brew install tesseract\n"
            "  .venv/bin/pip install pytesseract pillow"
        )
    sd = StoryData()
    pages: list[str] = []
    for path in sorted(paths, key=_image_sort_key):
        img = _Image.open(path)
        text = pytesseract.image_to_string(img, lang="eng").strip()
        if text:
            pages.append(text)
    if not pages:
        sys.exit("OCR extracted no text from the provided images.")
    combined = "\n\n".join(pages)
    sd.chapter_texts = split_into_chapters(combined)
    if sd.chapter_texts:
        sd.description = _auto_description(sd.chapter_texts[0])
    return sd


# ── EPUB helpers ──────────────────────────────────────────────────────────────

def _epub_find_opf(zf: zipfile.ZipFile, names: list[str]) -> Optional[str]:
    if "META-INF/container.xml" in names:
        try:
            xml = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
            m = re.search(r'full-path=["\']([^"\']+\.opf)["\']', xml, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception:
            pass
    for n in names:
        if n.lower().endswith(".opf"):
            return n
    return None


def _epub_resolve_path(item_path: str, names: list[str]) -> Optional[str]:
    if item_path in names:
        return item_path
    lc = item_path.lower()
    for n in names:
        if n.lower() == lc:
            return n
    base = Path(item_path).name.lower()
    for n in names:
        if Path(n).name.lower() == base:
            return n
    return None


def _epub_html_to_text(html_bytes: bytes) -> str:
    if not _HAVE_BS4:
        return ""
    try:
        m = re.search(rb'charset=["\']?([a-zA-Z0-9_-]+)', html_bytes[:500])
        encoding = m.group(1).decode("ascii", errors="replace") if m else "utf-8"
        html_str = html_bytes.decode(encoding, errors="replace")
    except Exception:
        html_str = html_bytes.decode("utf-8", errors="replace")

    soup = _BS(html_str, "html.parser")
    for tag in soup(["script", "style", "head", "nav", "aside"]):
        tag.decompose()
    for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "br", "li"]):
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")

    text = soup.get_text(" ")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ── Misc helpers ──────────────────────────────────────────────────────────────

def _auto_description(text: str, max_chars: int = 280) -> str:
    for para in re.split(r'\n\n+', text):
        para = para.strip()
        if len(para.split()) < 15:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        desc = ""
        for s in sentences:
            if len(desc) + len(s) + 1 > max_chars:
                break
            desc += (" " if desc else "") + s
        if desc:
            return desc.strip()
    return ""


def _make_id(title: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", title).split()
    return "".join(w.capitalize() for w in words)


def _split_camel(s: str) -> str:
    """'RoommateAgreement' → 'Roommate Agreement'"""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def _normalize_title(title: str) -> str:
    """Normalize a title for fuzzy duplicate detection.
    Strips leading articles (the/a/an), lowercases, removes punctuation."""
    t = title.lower().strip()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _word_count(html_text: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", html_text).split())


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def _parse_sol_date(raw: str) -> str:
    """Convert '2019-10-19 23:34:50' → '19 October 2019', or '' on failure."""
    if not raw:
        return ""
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', raw.strip())
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{d} {_MONTH_NAMES[mo - 1]} {y}"
    return ""


def _title_from_filename(path: Path, author_hint: str = "") -> str:
    """Derive a story title from a filename, removing author suffix and extension."""
    stem = path.stem
    # Remove trailing " - AuthorName" suffix if present
    if author_hint:
        pattern = re.compile(r'\s*[-–]\s*' + re.escape(author_hint) + r'\S*\s*$', re.IGNORECASE)
        stem = pattern.sub("", stem)
    # Clean underscores and extra spaces
    stem = stem.replace("_", " ").strip()
    return stem


def _series_base(stem: str) -> str:
    """Strip trailing chapter/part number from a filename stem for grouping."""
    return _SERIES_NUM.sub("", stem).strip().rstrip("-–_ ")


def _author_from_filename(path: Path) -> str:
    """Extract author from " - AuthorName" suffix in filename."""
    m = re.search(r'\s*[-–]\s*([A-Z][a-zA-Z]+(?:Who[A-Z][a-zA-Z]+)?)\s*$', path.stem)
    return m.group(1) if m else ""


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract(paths: list[Path]) -> StoryData:
    """Auto-detect format and extract story data from the given file(s)."""
    if not paths:
        sys.exit("No source files provided.")

    exts = {p.suffix.lower() for p in paths}

    if exts <= _TEXT_EXTS:
        return extract_txt(paths)
    elif exts <= _PDF_EXTS:
        return extract_pdf(paths)
    elif exts <= _EPUB_EXTS:
        if len(paths) > 1:
            sys.exit("Provide a single .epub file.")
        return extract_epub(paths[0])
    elif exts <= _DOCX_EXTS:
        return extract_docx(paths)
    elif exts <= _IMAGE_EXTS:
        return extract_images(paths)
    elif len(exts) == 1:
        sys.exit(f"Unsupported file type: {exts}")
    else:
        sys.exit(f"Mixed file types not supported in a single call: {exts}")


def resolve_paths(sources: list[str]) -> list[Path]:
    """Resolve source arguments (files or directory) to a sorted list of Paths."""
    out: list[Path] = []
    for src in sources:
        p = Path(src)
        if not p.exists():
            sys.exit(f"Source not found: {src}")
        if p.is_dir():
            children = sorted(p.iterdir())
            for ext_set in [_PDF_EXTS, _DOCX_EXTS, _TEXT_EXTS, _IMAGE_EXTS]:
                batch = [c for c in children if c.suffix.lower() in ext_set]
                if batch:
                    out.extend(batch)
                    break
            else:
                sys.exit(f"No supported files found in {src}")
        else:
            out.append(p)
    return out


# ── Index operations ──────────────────────────────────────────────────────────

def _load_index(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_index(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(tmp, encoding="utf-8") as f:
        check = json.load(f)
    assert len(check.get("stories", [])) == len(data.get("stories", [])), \
        "Story count mismatch after save"
    tmp.replace(path)


def patch_index(entry: dict, index_path: Path) -> str:
    """Add or replace a single entry. Returns 'added' or 'replaced'."""
    data    = _load_index(index_path)
    stories = data.get("stories", [])
    sid     = entry["id"]
    existing = next((i for i, s in enumerate(stories) if s["id"] == sid), None)
    if existing is not None:
        stories[existing] = entry
        action = "replaced"
    else:
        stories.append(entry)
        action = "added"
    data["stories"]       = stories
    data["total_stories"] = len(stories)
    _save_index(data, index_path)
    return action


# ── Build index entry ─────────────────────────────────────────────────────────

def build_entry(sd: StoryData, title: str, author: str, story_id: str,
                source: str = "local") -> dict:
    """Build a complete index entry dict from a StoryData object."""
    full_text = format_chapters_as_html(sd.chapter_texts)
    wc        = _word_count(full_text)
    return {
        "id":            story_id,
        "title":         title,
        "author":        author,
        "author_url":    sd.author_url,
        "story_url":     sd.story_url,
        "categories":    sd.categories,
        "description":   sd.description,
        "word_count":    wc,
        "added_date":    sd.added_date,
        "updated_date":  "",
        "full_text":     full_text,
        "fetch_error":   None,
        "auto_tags":     None,
        "chapter_count": len(sd.chapter_texts),
        "chapter_urls":  [],
        "source":        source,
    }


# ── Batch folder ingestion ─────────────────────────────────────────────────────

def _group_folder_files(folder: Path) -> list[list[Path]]:
    """
    Scan a folder and group files into stories.

    Multi-part series (e.g. "Mad Monday 1.docx", "Mad Monday 2.docx") are
    grouped into a single story. Single files are their own group.
    Returns a list of groups (each group = list of Paths = one story).
    """
    children = sorted(folder.iterdir())
    supported = [c for c in children
                 if c.is_file() and c.suffix.lower() in _ALL_EXTS]

    # Group by (extension_family, series_base_name).
    # EPUBs are NEVER grouped — each EPUB is a self-contained book.
    # PDFs, DOCXs, TXTs, and images CAN be grouped when they share a series base.
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in supported:
        ext = path.suffix.lower()
        if ext in _TEXT_EXTS:    family = "txt"
        elif ext in _PDF_EXTS:   family = "pdf"
        elif ext in _EPUB_EXTS:  family = "epub"
        elif ext in _DOCX_EXTS:  family = "docx"
        elif ext in _IMAGE_EXTS: family = "img"
        else: continue
        # EPUBs use the full stem as key so each file is its own group
        base = path.stem if ext in _EPUB_EXTS else _series_base(path.stem)
        groups[(family, base)].append(path)

    # Sort files within each group by the trailing number
    result: list[list[Path]] = []
    for (family, base), paths in sorted(groups.items(), key=lambda x: x[0][1].lower()):
        paths.sort(key=lambda p: (
            int(m.group(1)) if (m := _SERIES_NUM.search(p.stem)) else 0,
            p.name,
        ))
        result.append(paths)
    return result


def ingest_folder(folder: Path, default_author: str,
                  index_path: Path, source: str,
                  dry_run: bool, verbose: bool) -> dict[str, int]:
    """
    Batch-ingest all supported files in a folder.
    Skips stories already in the index (matched by ID or title+author).
    Returns counts: added, skipped_existing, skipped_empty, failed.
    """
    data    = _load_index(index_path)
    stories = data.get("stories", [])
    existing_ids    = {s["id"] for s in stories}
    existing_ta     = {(s.get("title", "").lower(), s.get("author", "").lower())
                       for s in stories}
    # Normalised title → id: strips leading "the/a/an" to catch minor title variants.
    # Author is intentionally excluded from the key — the stored author may differ
    # from what ingest derives (e.g. extracted from SOL header vs --author flag).
    # Also registers CamelCase-split variants for stories whose title == id (wayback/local).
    existing_title_norm: dict[str, str] = {}
    for s in stories:
        raw = s.get("title", "")
        if not raw:
            continue
        sid = s["id"]
        for variant in (raw, _split_camel(raw)):
            key = _normalize_title(variant)
            if key:
                existing_title_norm.setdefault(key, sid)

    groups = _group_folder_files(folder)
    counts = {"added": 0, "skipped_existing": 0, "skipped_empty": 0, "failed": 0}

    for paths in groups:
        # Derive title from filename (first file in group)
        p0      = paths[0]
        # Prefer the supplied default_author; only fall back to filename detection
        # if no author was provided (avoids false matches like "Win-Win" → author="Win")
        fn_author = default_author or _author_from_filename(p0)
        fn_title  = _title_from_filename(p0, fn_author)

        # For multi-part groups, try to strip trailing number from title
        if len(paths) > 1:
            fn_title = _series_base(fn_title)
        fn_title = fn_title.strip()

        story_id = _make_id(fn_title or p0.stem)

        # Skip if already indexed
        if story_id in existing_ids:
            if verbose:
                print(f"  SKIP (exists): {story_id}")
            counts["skipped_existing"] += 1
            continue
        ta_key = (fn_title.lower(), fn_author.lower())
        if ta_key in existing_ta and fn_title:
            if verbose:
                print(f"  SKIP (title+author exists): {fn_title}")
            counts["skipped_existing"] += 1
            continue
        norm_title = _normalize_title(fn_title)
        norm_title_base = _normalize_title(_series_base(fn_title))
        matched = (existing_title_norm.get(norm_title)
                   or (existing_title_norm.get(norm_title_base) if norm_title_base != norm_title else None))
        if matched:
            if verbose:
                print(f"  SKIP (title variant of {matched}): {fn_title!r}")
            counts["skipped_existing"] += 1
            continue

        if verbose:
            names = ", ".join(p.name for p in paths)
            print(f"  Processing: {names}")

        # Extract
        try:
            sd = extract(paths)
        except SystemExit as e:
            print(f"  FAIL: {p0.name}: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue
        except Exception as e:
            print(f"  FAIL: {p0.name}: {e}", file=sys.stderr)
            counts["failed"] += 1
            continue

        # Prefer extracted metadata over filename-derived
        title  = sd.title  or fn_title  or p0.stem
        author = sd.author or fn_author or default_author
        if not sd.chapter_texts:
            counts["skipped_empty"] += 1
            continue

        entry = build_entry(sd, title, author, story_id, source)

        # Always update tracking sets (even in dry-run) so later files in the
        # same folder won't show as duplicates of each other.
        existing_ids.add(story_id)
        existing_ta.add((title.lower(), author.lower()))
        existing_title_norm[_normalize_title(title)] = story_id

        if dry_run:
            print(f"  WOULD ADD: {story_id}  ({entry['word_count']:,} words, "
                  f"{len(sd.chapter_texts)} ch)  — {title}")
            counts["added"] += 1
            continue

        # Patch directly into loaded data to avoid repeated file I/O
        stories.append(entry)
        print(f"  + {story_id}  ({entry['word_count']:,} words)  — {title}")
        counts["added"] += 1

    if not dry_run and counts["added"] > 0:
        data["stories"]       = stories
        data["total_stories"] = len(stories)
        _save_index(data, index_path)
        print(f"\nIndex updated → {index_path}  ({len(stories):,} stories total)")

    return counts


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Ingest local content (txt/epub/pdf/docx/image) into the MC Archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ingest.py story.txt --patch
  ingest.py book.epub --patch
  ingest.py ch1.pdf ch2.pdf --title "Long Story" --author "Jane" --patch
  ingest.py ./story_folder/ --author "Jane" --patch
  ingest.py --folder /path/to/Pan/ --author "Pan" --dry-run
  ingest.py --folder /path/to/Pan/ --author "Pan" --patch
  ingest.py scan.png --title "Scanned" --author "Jane" --patch

After patching:  make tag && make post-update
        """,
    )
    # Single-story mode
    p.add_argument("sources", nargs="*",
                   help="Source file(s) or directory")
    p.add_argument("--title",       default="",  help="Story title (auto-extracted if available)")
    p.add_argument("--author",      default="",  help="Author name")
    p.add_argument("--id",          default="",  help="Story ID (auto-derived from title)")
    p.add_argument("--description", default="",  help="Short blurb (auto-extracted if omitted)")
    p.add_argument("--source",      default=DEFAULT_SOURCE,
                   help=f"Source tag (default: {DEFAULT_SOURCE})")
    # Batch folder mode
    p.add_argument("--folder",  default="",
                   help="Batch ingest all stories in this folder")
    # Output / behaviour
    p.add_argument("--index",   default=DEFAULT_INDEX,
                   help=f"Path to stories_index.json (default: {DEFAULT_INDEX})")
    p.add_argument("--patch",   action="store_true",
                   help="Write entries into stories_index.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be added without saving")
    p.add_argument("--verbose", action="store_true",
                   help="Show per-file progress in folder mode")
    args = p.parse_args()

    index_path = Path(args.index)

    # ── Batch folder mode ──────────────────────────────────────────────────────
    if args.folder:
        folder = Path(args.folder)
        if not folder.is_dir():
            sys.exit(f"Folder not found: {folder}")
        if not index_path.exists():
            sys.exit(f"Index not found: {index_path}")
        print(f"Batch ingesting: {folder}")
        print(f"Index          : {index_path}")
        if args.dry_run:
            print("Dry run — nothing will be saved.\n")

        counts = ingest_folder(
            folder        = folder,
            default_author= args.author,
            index_path    = index_path,
            source        = args.source,
            dry_run       = args.dry_run,
            verbose       = args.verbose or args.dry_run,
        )
        print(f"\nDone:")
        print(f"  Added          : {counts['added']}")
        print(f"  Already indexed: {counts['skipped_existing']}")
        print(f"  Empty/failed   : {counts['skipped_empty'] + counts['failed']}")
        if counts["added"] > 0 and not args.dry_run:
            print("\nNext: make tag && make post-update")
        return

    # ── Single-story mode ─────────────────────────────────────────────────────
    if not args.sources:
        p.print_help()
        sys.exit(1)

    paths = resolve_paths(args.sources)
    print(f"Sources  : {[p.name for p in paths]}")

    sd = extract(paths)

    # Resolve metadata: CLI args > extracted > filename
    title  = args.title  or sd.title  or _title_from_filename(paths[0], args.author)
    author = args.author or sd.author or _author_from_filename(paths[0])
    desc   = args.description or sd.description
    sid    = args.id or _make_id(title or paths[0].stem)

    print(f"Title    : {title or '(not set)'}")
    print(f"Author   : {author or '(not set)'}")
    print(f"ID       : {sid}")
    print(f"Chapters : {len(sd.chapter_texts)}")

    entry = build_entry(sd, title, author, sid, args.source)
    entry["description"] = desc
    print(f"Words    : {entry['word_count']:,}")
    print(f"Paras    : {entry['full_text'].count('<p>')}")

    if args.dry_run:
        preview = entry["full_text"][:800]
        print(f"\n── HTML preview ─────────────────────────────────────────────")
        print(preview)
        print("─────────────────────────────────────────────────────────────")
        print("\nDry run — nothing saved.")
        return

    if args.patch:
        if not index_path.exists():
            sys.exit(f"Index not found: {index_path}")
        action = patch_index(entry, index_path)
        print(f"  Entry {action} in {index_path}")
        print("\nNext: make tag && make post-update")
        return

    # Default: print JSON snippet
    print("\n── JSON entry ───────────────────────────────────────────────────")
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    print("\nRe-run with --patch to add automatically, or --dry-run to preview.")


if __name__ == "__main__":
    main()
