"""
tests.py — test suite for MCScraper

Fast tests (no network):  pytest tests.py -m "not network"
All tests:                 pytest tests.py -v
Semantic tests:            pytest tests.py -m semantic  (requires sentence-transformers)

Run via Makefile:
  make test
  make test-fast
  make test-semantic
"""

import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the project directory is on sys.path so all modules are importable
# regardless of what directory pytest is invoked from.
PROJECT_DIR = Path(__file__).parent.resolve()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_STORY_HYPNOSIS = {
    "id":          "Authors__TestAuthor__HypnosisStory",
    "title":       "The Hypnotist's Assistant",
    "author":      "Test Author",
    "author_url":  "https://mcstories.com/Authors/TestAuthor/index.html",
    "story_url":   "https://mcstories.com/Authors/TestAuthor/HypnosisStory.html",
    "categories":  ["mc", "mf", "hm"],
    "description": "A woman visits a hypnotherapist and finds herself unable to resist.",
    "word_count":  3200,
    "added_date":  "01 Jan 2020",
    "updated_date": "",
    "full_text": """
        Sarah had never been hypnotised before. She sat in the therapist's office,
        watching the spiral pendant swing slowly back and forth. "You are feeling
        sleepy," he said. She felt her eyelids grow heavy.

        The trance came on gradually. She could hear his voice but her thoughts
        were becoming distant, floating away. The induction was gentle but
        irresistible. Post-hypnotic suggestion, he explained later, would
        ensure she returned.

        Weeks later she found herself obeying his trigger words without thinking,
        her conscious mind completely unaware of the compulsion. She had become
        his obedient subject.
    """,
    "fetch_error": None,
}

SAMPLE_STORY_OFFICE = {
    "id":          "Authors__TestAuthor__OfficeControl",
    "title":       "Corner Office",
    "author":      "Test Author",
    "author_url":  "https://mcstories.com/Authors/TestAuthor/index.html",
    "story_url":   "https://mcstories.com/Authors/TestAuthor/OfficeControl.html",
    "categories":  ["mc", "mf", "fd"],
    "description": "A new boss takes control of her staff in unexpected ways.",
    "word_count":  5800,
    "added_date":  "15 Mar 2021",
    "updated_date": "",
    "full_text": """
        The boardroom was quiet as the new manager took her seat. Her coworkers
        sat around the corporate table, uneasy. The office had changed since
        she arrived. She was dominant in ways nobody could quite explain.

        Her secretary had stopped questioning her. The manager's commands were
        simply obeyed now. She had reprogrammed them slowly, conditioning each
        employee through subtle workplace rituals. The subliminal influence was
        so gradual nobody noticed the change.

        He submitted without knowing why. She controlled the whole department.
        The corporate hierarchy meant nothing — her dominant personality
        had made them all yield to her will.
    """,
    "fetch_error": None,
}

SAMPLE_STORY_FANTASY = {
    "id":          "Authors__TestAuthor__EnchantedKingdom",
    "title":       "The Enchanted Kingdom",
    "author":      "Test Author",
    "author_url":  "https://mcstories.com/Authors/TestAuthor/index.html",
    "story_url":   "https://mcstories.com/Authors/TestAuthor/EnchantedKingdom.html",
    "categories":  ["mc", "ff", "ma", "ft"],
    "description": "A witch casts a spell over the castle.",
    "word_count":  12000,
    "added_date":  "20 Jun 2019",
    "updated_date": "",
    "full_text": """
        The wizard had warned them about the enchanted forest. Beyond the castle
        gates, a powerful witch had laid her spells across the kingdom. Dragons
        circled the towers. The tavern at the edge of the village was the last
        safe place.

        She cast her incantation at midnight. The arcane ritual was ancient,
        the potion brewed from rare herbs. The curse fell over the kingdom
        like a warm fog. Love, she whispered. Her companion kissed her
        tenderly under the moonlight. The romance felt real but had been
        conjured by the enchantment. She was warm and caring in her affection,
        gentle with her companion, sweet in every tender word she spoke.
        Their love was deep and genuine, growing more intimate with every
        passing day under the enchanted sky.
    """,
    "fetch_error": None,
}

SAMPLE_STORY_FIRST_PERSON = {
    "id":    "Authors__TestAuthor__FirstPersonStory",
    "title": "My Awakening",
    "author": "Test Author",
    "author_url": "",
    "story_url": "",
    "categories": ["mc", "mf"],
    "description": "",
    "word_count": 800,
    "added_date": "",
    "updated_date": "",
    "full_text": """
        I remember the moment it happened. I was sitting in my apartment when
        I felt the change come over me. I had no idea what was happening to me.
        My thoughts began to drift and I found myself unable to resist.
        I smiled without knowing why. My mind felt warm and distant.
        I was happy. I was content. I was his.
    """,
    "fetch_error": None,
}

SAMPLE_STORY_BROKEN = {
    "id":         "broken",
    "title":      "",
    "author":     "",
    "author_url": "",
    "story_url":  "",
    "categories": [],
    "description": "",
    "word_count": 0,
    "added_date": "",
    "updated_date": "",
    "full_text":  "",
    "fetch_error": "Failed after 3 retries",
}


def make_index(stories: list[dict]) -> dict:
    return {
        "source": "https://mcstories.com",
        "scraped_at": "2026-01-01T00:00:00Z",
        "total_stories": len(stories),
        "stories": stories,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TAGGER — REGEX LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegexTagger:

    def test_site_tag_expansion_basic(self):
        from tagger import expand_site_tags
        result = expand_site_tags(["mc", "mf", "hm"])
        assert "mind control" in result["labels"]
        assert "male/female sex" in result["labels"]
        assert "humiliation" in result["labels"]

    def test_site_tag_expansion_unknown_codes_ignored(self):
        from tagger import expand_site_tags
        result = expand_site_tags(["mc", "zz", "xx"])
        assert len(result["labels"]) == 1
        assert "mind control" in result["labels"]

    def test_site_tag_expansion_deduplicates(self):
        from tagger import expand_site_tags
        result = expand_site_tags(["mc", "mc", "mc"])
        assert result["labels"].count("mind control") == 1

    def test_site_tag_groups_populated(self):
        from tagger import expand_site_tags
        result = expand_site_tags(["mc", "mf", "hm"])
        assert "sex_act" in result["groups"]
        assert "kink" in result["groups"]

    def test_content_type_extracts_gender_dynamic(self):
        from tagger import regex_tag
        # content_type only carries the gender-dynamic codes (fd/md)
        result = regex_tag(SAMPLE_STORY_OFFICE)
        assert "female dominant" in result["content_type"]

    def test_mind_control_type_hypnosis(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_HYPNOSIS)
        assert "hypnosis" in result["mind_control_type"]

    def test_mind_control_type_subliminal(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_OFFICE)
        assert "subliminal" in result["mind_control_type"]

    def test_mind_control_type_magic(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_FANTASY)
        assert "magic" in result["mind_control_type"]

    def test_setting_office(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_OFFICE)
        assert "workplace" in result["setting"]

    def test_setting_fantasy(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_FANTASY)
        assert "fantasy / historical" in result["setting"]

    def test_theme_romance(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_FANTASY)
        assert "romance" in result["themes"]

    def test_theme_minimum_count_respected(self):
        """A single keyword hit below threshold should NOT fire the tag."""
        from tagger import regex_tag
        story = {**SAMPLE_STORY_HYPNOSIS,
                 "full_text": "Once there was an office. The end."}
        result = regex_tag(story)
        # "office" appears once but threshold is 2 — should not fire
        assert "workplace" not in result["setting"]

    def test_pov_first_person(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_FIRST_PERSON)
        assert result["pov"] == "first person"

    def test_pov_third_person(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_HYPNOSIS)
        assert result["pov"] == "third person"

    def test_pov_unknown_on_short_text(self):
        from tagger import detect_pov
        assert detect_pov("") == "unknown"
        assert detect_pov("Hello.") == "unknown"

    def test_length_flash(self):
        from tagger import classify_length
        assert classify_length(500) == "flash"

    def test_length_short_story(self):
        from tagger import classify_length
        assert classify_length(3000) == "short story"

    def test_length_novelette(self):
        from tagger import classify_length
        assert classify_length(10000) == "novelette"

    def test_length_novella(self):
        from tagger import classify_length
        assert classify_length(25000) == "novella"

    def test_length_novel(self):
        from tagger import classify_length
        assert classify_length(50000) == "novel"

    def test_length_boundaries(self):
        from tagger import classify_length
        assert classify_length(999)   == "flash"
        assert classify_length(1000)  == "short story"
        assert classify_length(7499)  == "short story"
        assert classify_length(7500)  == "novelette"
        assert classify_length(17499) == "novelette"
        assert classify_length(17500) == "novella"
        assert classify_length(39999) == "novella"
        assert classify_length(40000) == "novel"

    def test_broken_story_returns_empty_not_crash(self):
        from tagger import regex_tag
        result = regex_tag(SAMPLE_STORY_BROKEN)
        assert isinstance(result, dict)
        assert result["mind_control_type"] == []
        assert result["setting"] == []
        assert result["pov"] == "unknown"
        assert result["length_class"] == "flash"

    def test_title_weighting(self):
        """Title keyword alone (weighted 3x) should push a tag over threshold."""
        from tagger import regex_tag
        story = {
            **SAMPLE_STORY_HYPNOSIS,
            "title": "Office Office Office",
            "description": "An office story.",
            "full_text": "She walked in. He smiled. The end.",
        }
        result = regex_tag(story)
        assert "workplace" in result["setting"]


# ═══════════════════════════════════════════════════════════════════════════════
# TAGGER — MERGE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class TestMergeLayer:

    def setup_method(self):
        from tagger import regex_tag
        self.rx = regex_tag(SAMPLE_STORY_HYPNOSIS)

    def test_merge_regex_only_returns_valid_dict(self):
        from tagger import merge_tags
        merged = merge_tags(self.rx)
        assert isinstance(merged, dict)
        for dim in ("site_tags_expanded", "content_type", "mind_control_type",
                    "setting", "themes", "tone", "pov", "length_class"):
            assert dim in merged

    def test_merge_semantic_none_same_as_regex(self):
        from tagger import merge_tags
        merged = merge_tags(self.rx, semantic_result=None)
        assert merged["mind_control_type"] == self.rx["mind_control_type"]
        assert merged["setting"] == self.rx["setting"]

    def test_merge_semantic_adds_new_tags(self):
        from tagger import merge_tags
        sem = {"mind_control_type": ["subliminal"], "setting": ["gym / fitness"],
               "themes": ["slow burn"], "tone": ["psychological"]}
        merged = merge_tags(self.rx, semantic_result=sem)
        # Semantic-only tag should be appended
        assert "gym / fitness" in merged["setting"]

    def test_merge_regex_tags_come_first(self):
        from tagger import merge_tags
        rx_with_setting = {**self.rx, "setting": ["workplace"]}
        sem = {"mind_control_type": [], "setting": ["workplace", "gym / fitness"],
               "themes": [], "tone": []}
        merged = merge_tags(rx_with_setting, semantic_result=sem)
        assert merged["setting"][0] == "workplace"

    def test_merge_no_duplicates(self):
        from tagger import merge_tags
        sem = {"mind_control_type": ["hypnosis"], "setting": [],
               "themes": [], "tone": []}
        merged = merge_tags(self.rx, semantic_result=sem)
        assert merged["mind_control_type"].count("hypnosis") == 1

    def test_merge_sources_recorded(self):
        from tagger import merge_tags
        sem = {"mind_control_type": ["subliminal", "hypnosis"], "setting": [],
               "themes": [], "tone": []}
        merged = merge_tags(self.rx, semantic_result=sem)
        sources = merged["_tag_sources"]
        # hypnosis fired in both layers
        assert "regex" in sources.get("hypnosis", [])
        assert "semantic" in sources.get("hypnosis", [])
        # subliminal fired only in semantic (not in hypnosis story)
        assert sources.get("subliminal") == ["semantic"]

    def test_merge_semantic_flag_recorded(self):
        from tagger import merge_tags
        no_sem  = merge_tags(self.rx)
        with_sem = merge_tags(self.rx, {"mind_control_type": [], "setting": [],
                                        "themes": [], "tone": []})
        assert no_sem["_semantic_used"] is False
        assert with_sem["_semantic_used"] is True

    def test_regex_only_dims_never_overwritten(self):
        """site_tags_expanded, pov, length_class must always come from regex."""
        from tagger import merge_tags
        # Semantic result has no say over these dims — passing garbage should not matter
        sem = {"mind_control_type": [], "setting": [], "themes": [], "tone": [],
               "pov": "WRONG", "length_class": "WRONG",
               "site_tags_expanded": ["WRONG"]}
        merged = merge_tags(self.rx, semantic_result=sem)
        assert merged["pov"] != "WRONG"
        assert merged["length_class"] != "WRONG"
        assert "WRONG" not in merged["site_tags_expanded"]

    def test_tagger_version_present(self):
        from tagger import merge_tags
        merged = merge_tags(self.rx)
        assert merged["_tagger_version"] == "2.0"


# ═══════════════════════════════════════════════════════════════════════════════
# TAGGER — PUBLIC tag_story() API
# ═══════════════════════════════════════════════════════════════════════════════

class TestTagStoryAPI:

    def test_tag_story_regex_only(self):
        from tagger import tag_story
        result = tag_story(SAMPLE_STORY_HYPNOSIS, use_semantic=False)
        assert isinstance(result, dict)
        assert "hypnosis" in result["mind_control_type"]
        assert result["_semantic_used"] is False

    def test_tag_story_all_keys_present(self):
        from tagger import tag_story
        result = tag_story(SAMPLE_STORY_HYPNOSIS, use_semantic=False)
        for key in ("site_tags_expanded", "content_type", "mind_control_type",
                    "setting", "themes", "tone", "pov", "length_class",
                    "_tag_sources", "_tagger_version", "_semantic_used"):
            assert key in result, f"Missing key: {key}"

    def test_tag_story_broken_story_safe(self):
        from tagger import tag_story
        result = tag_story(SAMPLE_STORY_BROKEN, use_semantic=False)
        assert isinstance(result, dict)

    @pytest.mark.semantic
    def test_tag_story_semantic_layer(self):
        """Semantic layer should find tags regex misses."""
        pytest.importorskip("sentence_transformers")
        from tagger import tag_story

        # Build a story that describes an office setting without using office words
        story = {
            **SAMPLE_STORY_OFFICE,
            "full_text": (
                "She sat behind her desk, the fluorescent lights humming above. "
                "Her subordinates filed in for the morning briefing. The quarterly "
                "reports were due. Her assistant brought coffee. The conference call "
                "started at nine. She ran the division with quiet authority. "
                "Everyone followed her lead without question. The meeting ended "
                "and she returned to her corner suite on the fourteenth floor."
            ),
            "description": "",
        }
        result = tag_story(story, use_semantic=True, threshold=0.30)
        assert result["_semantic_used"] is True
        # Semantic should catch office setting even without exact trigger words
        assert "workplace" in result["setting"]

    @pytest.mark.semantic
    def test_tag_story_semantic_threshold_respected(self):
        pytest.importorskip("sentence_transformers")
        from tagger import tag_story

        # At a very high threshold nothing should fire
        result_strict = tag_story(SAMPLE_STORY_HYPNOSIS,
                                  use_semantic=True, threshold=0.99)
        # At a very low threshold many things should fire
        result_loose  = tag_story(SAMPLE_STORY_HYPNOSIS,
                                  use_semantic=True, threshold=0.01)
        total_strict = sum(len(result_strict.get(d, [])) for d in
                           ("mind_control_type", "setting", "themes", "tone"))
        total_loose  = sum(len(result_loose.get(d, [])) for d in
                           ("mind_control_type", "setting", "themes", "tone"))
        assert total_loose >= total_strict


# ═══════════════════════════════════════════════════════════════════════════════
# TAGGER — BATCH PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatchProcessor:

    def test_process_index_writes_auto_tags(self):
        from tagger import process_index
        data = make_index([SAMPLE_STORY_HYPNOSIS, SAMPLE_STORY_OFFICE])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            process_index(tmp)
            result = json.loads(tmp.read_text())
            for story in result["stories"]:
                assert story["auto_tags"] is not None
                assert "mind_control_type" in story["auto_tags"]
        finally:
            tmp.unlink()

    def test_process_index_skips_broken_stories(self):
        from tagger import process_index
        data = make_index([SAMPLE_STORY_HYPNOSIS, SAMPLE_STORY_BROKEN])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            process_index(tmp)
            result = json.loads(tmp.read_text())
            broken = next(s for s in result["stories"] if s["id"] == "broken")
            assert broken["auto_tags"] is None
        finally:
            tmp.unlink()

    def test_process_index_metadata_written(self):
        from tagger import process_index
        data = make_index([SAMPLE_STORY_HYPNOSIS])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp = Path(f.name)
        try:
            process_index(tmp)
            result = json.loads(tmp.read_text())
            assert result.get("auto_tagged") is True
            assert "tagged_count" in result
        finally:
            tmp.unlink()

    def test_process_index_custom_output_path(self):
        from tagger import process_index
        data = make_index([SAMPLE_STORY_HYPNOSIS])
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.json"
            dst = Path(tmpdir) / "output.json"
            src.write_text(json.dumps(data))
            process_index(src, output_path=dst)
            assert dst.exists()
            result = json.loads(dst.read_text())
            assert result["auto_tagged"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# TAGGER — JSONL REGENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONLRebuilder:

    def _make_tagged_index(self, tmpdir: Path) -> Path:
        from tagger import process_index
        data = make_index([SAMPLE_STORY_HYPNOSIS, SAMPLE_STORY_OFFICE, SAMPLE_STORY_BROKEN])
        idx_path = tmpdir / "stories_index.json"
        idx_path.write_text(json.dumps(data))
        process_index(idx_path)
        return idx_path

    def test_rebuild_creates_three_files(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            for fname in ("training_raw_pretraining.jsonl",
                          "training_instruction_response.jsonl",
                          "training_completion.jsonl"):
                assert (Path(tmpdir) / fname).exists()

    def test_rebuild_raw_pretraining_format(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            lines = (Path(tmpdir) / "training_raw_pretraining.jsonl").read_text().splitlines()
            assert len(lines) == 2   # 2 good stories, 1 broken skipped
            record = json.loads(lines[0])
            assert "text" in record
            assert "Title:" in record["text"]
            assert len(record["text"]) > 50

    def test_rebuild_instruction_format(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            lines = (Path(tmpdir) / "training_instruction_response.jsonl").read_text().splitlines()
            record = json.loads(lines[0])
            assert "messages" in record
            roles = [m["role"] for m in record["messages"]]
            assert roles == ["system", "user", "assistant"]

    def test_rebuild_completion_format(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            lines = (Path(tmpdir) / "training_completion.jsonl").read_text().splitlines()
            record = json.loads(lines[0])
            assert "prompt" in record
            assert "completion" in record
            assert len(record["completion"]) > 50

    def test_rebuild_skips_broken_stories(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            lines = (Path(tmpdir) / "training_raw_pretraining.jsonl").read_text().splitlines()
            # Only 2 valid stories
            assert len(lines) == 2

    def test_rebuild_jsonl_is_valid_json(self):
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            for fname in ("training_raw_pretraining.jsonl",
                          "training_instruction_response.jsonl",
                          "training_completion.jsonl"):
                for line in (Path(tmpdir) / fname).read_text().splitlines():
                    json.loads(line)   # raises if invalid

    def test_rebuild_tags_in_output(self):
        """Trusted tags appear in the meta header; low-trust tags are filtered out."""
        from tagger import rebuild_jsonl_with_tags
        story = {**SAMPLE_STORY_HYPNOSIS, "auto_tags": {
            "themes": ["trigger phrase", "romance"],   # Tier 1 / not trusted
            "mind_control_type": ["hypnosis"],         # AND-gated tag...
            "_tag_sources": {"hypnosis": ["regex"]},   # ...but only regex agreed
            "setting": ["home / domestic"],            # Tier 1
            "length_class": "short story",
        }}
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = Path(tmpdir) / "stories_index.json"
            idx.write_text(json.dumps(make_index([story])))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            out = (Path(tmpdir) / "training_raw_pretraining.jsonl").read_text()
            assert "Themes: trigger phrase" in out
            assert "Setting: home / domestic" in out
            assert "Length: short story" in out
            assert "romance" not in out
            assert "Mind Control Method" not in out


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER — HTML PARSING (no network)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScraperParsing:

    # /Titles/index.html — contains letter page links like A.html (relative)
    TITLES_INDEX_HTML = """
    <html><body>
    <a href="A.html">A</a>
    <a href="B.html">B</a>
    <a href="../Authors/index.html">Authors</a>
    <a href="../index.html">Home</a>
    </body></html>
    """

    # /Titles/A.html — contains story landing links like ../AGreatStory/index.html
    LETTER_PAGE_HTML = """
    <html><body>
    <a href="../AGreatStory/index.html">A Great Story</a>
    <a href="../AnotherTale/index.html">Another Tale</a>
    <a href="../Titles/index.html">Titles</a>
    <a href="../Authors/index.html">Authors</a>
    <a href="../Tags/index.html">Tags</a>
    </body></html>
    """

    # /AGreatStory/index.html — story landing: metadata + chapter links
    # Multi-chapter: ch1=AGreatStory.html (no number), ch2=AGreatStory2.html
    LANDING_MULTI_HTML = """
    <html><body>
    <h1>The Erotic Mind-Control Story Archive</h1>
    <h2>A Great Story</h2>
    <p>Added: 01 Jan 2022</p>
    <a href="../Authors/JohnDoe.html">John Doe</a>
    <a href="../Tags/mc.html">mc</a>
    <a href="../Tags/mf.html">mf</a>
    <p>She sat in the office and felt herself drifting into trance.</p>
    <a href="AGreatStory.html">Part 1</a>
    <a href="AGreatStory2.html">Part 2</a>
    <a href="AGreatStory3.html">Part 3</a>
    </body></html>
    """

    # Single-chapter landing: only one chapter link (no number)
    LANDING_SINGLE_HTML = """
    <html><body>
    <h2>Short Piece</h2>
    <p>Added: 15 Jun 2021</p>
    <a href="../Authors/JaneDoe.html">Jane Doe</a>
    <a href="../Tags/ff.html">ff</a>
    <a href="../Tags/hm.html">hm</a>
    <p>A brief tale of enchantment.</p>
    <a href="ShortPiece.html">Read</a>
    </body></html>
    """

    CHAPTER_HTML = """
    <html><body>
    <div id="mcstories">
        She felt her mind growing hazy. The hypnotist spoke softly.
        Her eyes closed and she drifted into a deep trance.
    </div>
    </body></html>
    """

    def _parse_letter(self, html, letter_url="https://mcstories.com/Titles/A.html"):
        from bs4 import BeautifulSoup
        from scraper import get_story_urls_from_letter
        from unittest.mock import patch
        soup = BeautifulSoup(html, "html.parser")
        with patch("scraper.get", return_value=soup):
            return get_story_urls_from_letter(letter_url)

    def _parse_landing(self, html, story_slug="AGreatStory"):
        from bs4 import BeautifulSoup
        from scraper import parse_landing
        from unittest.mock import patch
        soup = BeautifulSoup(html, "html.parser")
        url = f"https://mcstories.com/{story_slug}/index.html"
        with patch("scraper.get", return_value=soup):
            return parse_landing(url)

    def test_letter_page_finds_story_links(self):
        urls = self._parse_letter(self.LETTER_PAGE_HTML)
        assert any("AGreatStory" in u for u in urls)
        assert any("AnotherTale" in u for u in urls)

    def test_letter_page_excludes_nav(self):
        urls = self._parse_letter(self.LETTER_PAGE_HTML)
        assert not any("/Titles/" in u for u in urls)
        assert not any("/Authors/" in u for u in urls)
        assert not any("/Tags/" in u for u in urls)

    def test_letter_page_deduplicates(self):
        dup = self.LETTER_PAGE_HTML + '<a href="../AGreatStory/index.html">Dup</a>'
        urls = self._parse_letter(dup)
        assert urls.count(next(u for u in urls if "AGreatStory" in u)) == 1

    def test_landing_extracts_title(self):
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        assert result.get("title") == "A Great Story"

    def test_landing_skips_masthead_h1(self):
        # The site masthead h1 "The Erotic Mind-Control Story Archive" must not be title
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        assert "mind-control story archive" not in result.get("title", "").lower()

    def test_landing_extracts_author(self):
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        assert result.get("author") == "John Doe"

    def test_landing_extracts_categories(self):
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        cats = result.get("categories", [])
        assert "mc" in cats
        assert "mf" in cats

    def test_landing_extracts_added_date(self):
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        assert "2022" in result.get("added_date", "")

    def test_landing_multi_chapter_urls(self):
        """ch1 = StoryName.html (no number), ch2 = StoryName2.html, ch3 = StoryName3.html"""
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        chapters = result.get("chapter_urls", [])
        assert len(chapters) == 3
        assert chapters[0].endswith("AGreatStory.html")    # ch1: no number
        assert chapters[1].endswith("AGreatStory2.html")   # ch2
        assert chapters[2].endswith("AGreatStory3.html")   # ch3

    def test_landing_multi_chapter_order(self):
        result = self._parse_landing(self.LANDING_MULTI_HTML)
        chapters = result.get("chapter_urls", [])
        # Chapter 1 (no suffix) must come before chapter 2
        assert "AGreatStory.html" in chapters[0]
        assert "AGreatStory2" in chapters[1]

    def test_landing_single_chapter_url(self):
        result = self._parse_landing(self.LANDING_SINGLE_HTML, story_slug="ShortPiece")
        chapters = result.get("chapter_urls", [])
        assert len(chapters) == 1
        assert chapters[0].endswith("ShortPiece.html")     # no number

    def test_landing_fallback_on_no_chapter_links(self):
        """If landing page has no chapter links, fall back to slug/slug.html"""
        html = "<html><body><h2>Mystery</h2></body></html>"
        result = self._parse_landing(html, story_slug="Mystery")
        chapters = result.get("chapter_urls", [])
        assert len(chapters) == 1
        assert "Mystery/Mystery.html" in chapters[0]

    def test_landing_network_fail(self):
        from unittest.mock import patch
        from scraper import parse_landing
        with patch("scraper.get", return_value=None):
            result = parse_landing("https://mcstories.com/X/index.html")
        assert result.get("fetch_error") is not None

    def test_make_id(self):
        from scraper import make_id
        assert make_id("https://mcstories.com/AGreatStory/index.html") == "AGreatStory"
        assert make_id("https://mcstories.com/HomeByNine/index.html")  == "HomeByNine"


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER — TRAINING FORMAT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingFormats:

    def _make_story(self):
        from scraper import Story
        from tagger import tag_story
        s = Story(
            id="test__story",
            title="Test Story",
            author="Test Author",
            author_url="",
            story_url="",
            categories=["mc", "mf"],
            description="A test story.",
            word_count=500,
            full_text="She felt her mind going blank as his voice filled the room.",
        )
        s.auto_tags = tag_story({
            "title": s.title, "author": s.author, "categories": s.categories,
            "description": s.description, "word_count": s.word_count,
            "full_text": s.full_text,
        })
        return s

    def test_raw_pretraining_has_text_key(self):
        from scraper import to_raw_pretraining
        rec = to_raw_pretraining(self._make_story())
        assert "text" in rec
        assert len(rec["text"]) > 30

    def test_raw_pretraining_includes_title(self):
        from scraper import to_raw_pretraining
        rec = to_raw_pretraining(self._make_story())
        assert "Test Story" in rec["text"]

    def test_instruction_has_messages(self):
        from scraper import to_instruction_response
        rec = to_instruction_response(self._make_story())
        assert "messages" in rec
        assert len(rec["messages"]) == 3

    def test_instruction_roles_correct(self):
        from scraper import to_instruction_response
        rec = to_instruction_response(self._make_story())
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_instruction_assistant_is_full_text(self):
        from scraper import to_instruction_response
        s = self._make_story()
        rec = to_instruction_response(s)
        assert rec["messages"][2]["content"] == s.full_text

    def test_completion_has_prompt_and_completion(self):
        from scraper import to_completion
        rec = to_completion(self._make_story())
        assert "prompt" in rec
        assert "completion" in rec

    def test_completion_completion_is_full_text(self):
        from scraper import to_completion
        s = self._make_story()
        rec = to_completion(s)
        assert rec["completion"] == s.full_text


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearch:

    @pytest.fixture
    def stories(self):
        from tagger import tag_story
        def tagged(s):
            s = dict(s)
            s["auto_tags"] = tag_story(s)
            return s
        return [tagged(SAMPLE_STORY_HYPNOSIS),
                tagged(SAMPLE_STORY_OFFICE),
                tagged(SAMPLE_STORY_FANTASY)]

    def test_keyword_search_title(self, stories):
        from search_stories import search
        results = search(stories, query="enchanted")
        assert len(results) == 1
        assert results[0]["title"] == "The Enchanted Kingdom"

    def test_keyword_search_full_text(self, stories):
        from search_stories import search
        results = search(stories, query="boardroom")
        assert any(r["title"] == "Corner Office" for r in results)

    def test_author_filter(self, stories):
        from search_stories import search
        results = search(stories, author="Test Author")
        assert len(results) == 3

    def test_author_filter_case_insensitive(self, stories):
        from search_stories import search
        results = search(stories, author="test author")
        assert len(results) == 3

    def test_author_filter_no_match(self, stories):
        from search_stories import search
        results = search(stories, author="Nobody")
        assert len(results) == 0

    def test_category_filter(self, stories):
        from search_stories import search
        results = search(stories, categories=["ff"])
        # Only the fantasy story has ff
        assert len(results) == 1
        assert results[0]["title"] == "The Enchanted Kingdom"

    def test_min_words_filter(self, stories):
        from search_stories import search
        results = search(stories, min_words=10000)
        assert all(r["word_count"] >= 10000 for r in results)

    def test_mc_type_filter(self, stories):
        from search_stories import search
        results = search(stories, mc_type="hypnosis")
        titles = [r["title"] for r in results]
        assert "The Hypnotist's Assistant" in titles

    def test_setting_filter(self, stories):
        from search_stories import search
        results = search(stories, setting="workplace")
        titles = [r["title"] for r in results]
        assert "Corner Office" in titles

    def test_tone_filter(self, stories):
        from search_stories import search
        results = search(stories, tone="romantic")
        titles = [r["title"] for r in results]
        assert "The Enchanted Kingdom" in titles

    def test_length_class_filter(self, stories):
        from search_stories import search
        results = search(stories, length_class="novelette")
        assert all(
            r["auto_tags"]["length_class"] == "novelette"
            for r in results if r.get("auto_tags")
        )

    def test_combined_filters(self, stories):
        from search_stories import search
        # Hypnosis story matches mc_type=hypnosis, author=Test Author
        results = search(stories, mc_type="hypnosis", author="Test Author")
        assert len(results) >= 1

    def test_broken_stories_excluded(self, stories):
        from search_stories import search
        broken = {**SAMPLE_STORY_BROKEN, "auto_tags": None}
        results = search(stories + [broken])
        for r in results:
            assert not r.get("fetch_error")

    def test_empty_query_returns_all_valid(self, stories):
        from search_stories import search
        results = search(stories)
        assert len(results) == 3


class TestSearchDb:
    """build_search_db: story text is searchable, and updates re-index only what changed."""

    @staticmethod
    def _write(idx, stories):
        idx.write_text(json.dumps(make_index(stories)))
        # Make the index newer than search.db so the up-to-date guard doesn't skip.
        future = time.time() + 10
        os.utime(idx, (future, future))

    @staticmethod
    def _match(db, word):
        from services.search_service import _fts_candidate_ids
        conn = sqlite3.connect(db)
        try:
            return _fts_candidate_ids(SimpleNamespace(search_conn=conn), word, "literal")
        finally:
            conn.close()

    @staticmethod
    def _rowids(db):
        conn = sqlite3.connect(db)
        try:
            return dict(conn.execute("SELECT story_id, rowid FROM fts_docs"))
        finally:
            conn.close()

    @pytest.fixture
    def built(self, tmp_path):
        from build_search_db import build
        idx, db = tmp_path / "stories_index.json", tmp_path / "search.db"
        stories = [dict(SAMPLE_STORY_HYPNOSIS), dict(SAMPLE_STORY_OFFICE),
                   dict(SAMPLE_STORY_FANTASY)]
        idx.write_text(json.dumps(make_index(stories)))
        build(str(idx), str(db), str(tmp_path / "no_cache.db"))
        return build, idx, db, stories

    def test_story_text_is_searchable(self, built):
        _, _, db, _ = built
        assert self._match(db, "boardroom") == {SAMPLE_STORY_OFFICE["id"]}

    def test_slim_build_skips_story_text(self, built, tmp_path):
        build, idx, _, _ = built
        slim = tmp_path / "slim.db"
        build(str(idx), str(slim), str(tmp_path / "no_cache.db"), no_fulltext=True)
        assert self._match(slim, "boardroom") == frozenset()

    def test_update_reindexes_only_changed_stories(self, built, capsys):
        build, idx, db, stories = built
        before = self._rowids(db)
        hyp, office, fantasy = stories
        office["full_text"] = office["full_text"].replace("boardroom", "penthouse")
        new = {**SAMPLE_STORY_HYPNOSIS, "id": "NewStory", "title": "New Story",
               "full_text": "A lighthouse keeper and a pocket watch."}
        self._write(idx, [hyp, office, new])          # fantasy removed
        capsys.readouterr()
        build(str(idx), str(db), str(idx.parent / "no_cache.db"))

        assert "1 added  ·  1 re-indexed  ·  1 removed" in capsys.readouterr().out
        assert self._match(db, "boardroom") == frozenset()
        assert self._match(db, "penthouse") == {office["id"]}
        assert self._match(db, "lighthouse") == {"NewStory"}
        after = self._rowids(db)
        assert fantasy["id"] not in after
        assert after[hyp["id"]] == before[hyp["id"]]
        assert after[office["id"]] == before[office["id"]]

    def test_old_schema_is_rebuilt(self, tmp_path):
        from build_search_db import build
        idx, db = tmp_path / "stories_index.json", tmp_path / "search.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE VIRTUAL TABLE stories_fts USING fts5(story_id UNINDEXED, title)")
        conn.close()
        self._write(idx, [dict(SAMPLE_STORY_OFFICE)])
        build(str(idx), str(db), str(tmp_path / "no_cache.db"))
        assert self._match(db, "boardroom") == {SAMPLE_STORY_OFFICE["id"]}


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK TESTS  (skipped by default — run with pytest -m network)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.network
class TestNetwork:

    def test_titles_index_reachable(self):
        from scraper import get
        soup = get("https://mcstories.com/Titles/index.html")
        assert soup is not None, "Could not reach Titles index"

    def test_letter_page_reachable(self):
        from scraper import get, get_story_urls_from_letter
        soup = get("https://mcstories.com/Titles/A.html")
        if soup is None:
            pytest.skip("Network unavailable")

    def test_letter_page_has_story_urls(self):
        from scraper import get_story_urls_from_letter
        urls = get_story_urls_from_letter("https://mcstories.com/Titles/A.html")
        if not urls:
            pytest.skip("Network unavailable")
        assert len(urls) > 0, "No story landing pages in Titles/A.html"
        assert all("/index.html" in u for u in urls)

    def test_single_story_scrape(self):
        import time
        from scraper import get_story_urls_from_letter, fetch_full_story
        urls = get_story_urls_from_letter("https://mcstories.com/Titles/A.html")
        if not urls:
            pytest.skip("Network unavailable")
        time.sleep(1)
        result = fetch_full_story(urls[0])
        assert result.get("full_text"), "Story full text is empty"
        assert len(result["full_text"]) > 100


# ═══════════════════════════════════════════════════════════════════════════════
# BROWSER — every page renders (Flask test client, fixture corpus)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrowserPages:
    """Load every page of the browser against a small tagged corpus.

    The corpus goes through the real tagger and build_stories_db, so stories
    reach the templates the same way they do in production (auto_tags as a
    lazily parsed JSON string). Everything runs in a temp dir because the app
    reads and writes ./output/.
    """

    @pytest.fixture(scope="class")
    def corpus(self, tmp_path_factory):
        from tagger import process_index
        from build_stories_db import build

        root = tmp_path_factory.mktemp("browser")
        out  = root / "output"
        out.mkdir()

        bases = [SAMPLE_STORY_HYPNOSIS, SAMPLE_STORY_OFFICE,
                 SAMPLE_STORY_FANTASY, SAMPLE_STORY_FIRST_PERSON]
        stories = []
        for i in range(20):
            base = bases[i % len(bases)]
            stories.append({**base, "id": f"{base['id']}{i}",
                            "title": f"{base['title']} {i}",
                            "author": f"Author {i % 5}"})
        stories.append(SAMPLE_STORY_BROKEN)
        idx = out / "stories_index.json"
        idx.write_text(json.dumps(make_index(stories)))
        process_index(idx)
        build(str(idx), str(out / "stories_meta.db"), force=True)
        from build_search_db import build as build_search
        build_search(str(idx), str(out / "search.db"),
                     str(out / "full_text_cache.db"), force=True, no_fulltext=True)

        ids = [s["id"] for s in stories[:20]]
        (out / "preferences.json").write_text(json.dumps({
            "liked_stories": ids[:3], "disliked_stories": [ids[3]],
            "liked_authors": ["Author 1"], "disliked_authors": [],
            "notes": {}, "hidden_recs": [], "want_to_read": [ids[4]],
            "history": {ids[5]: {"chapter": 1, "total": 1, "ts": "2026-01-01T00:00:00"}},
        }))

        return root, ids

    @pytest.fixture
    def client(self, corpus, monkeypatch):
        # Fresh app per test: pages parse auto_tags in place, so a shared app
        # would hide bugs that only show on the first page after startup.
        from app_factory import AppState, create_app
        from services.data_store import load_data
        root, ids = corpus
        monkeypatch.chdir(root)
        state = AppState(index_path="./output/stories_index.json")
        load_data(state)
        app = create_app(state)
        app.config["TESTING"] = True
        return app.test_client(), ids

    @pytest.mark.parametrize("path", [
        "/", "/?q=hypnosis", "/search?q=office", "/authors", "/tags", "/tag-stats",
        "/stats", "/liked", "/history", "/want-to-read", "/new-stories",
        "/recommendations", "/generate", "/bibles",
    ])
    def test_page_renders(self, client, path):
        c, _ = client
        resp = c.get(path)
        assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    @pytest.mark.parametrize("template", ["/story/{id}", "/similar/{id}", "/author/Author 1"])
    def test_story_pages_render(self, client, template):
        c, ids = client
        for sid in ids[:4]:  # one of each base story
            path = template.format(id=sid)
            resp = c.get(path)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    def test_fresh_metadata_shows_no_stale_banner(self, client):
        c, _ = client
        assert b"Story metadata is out of date" not in c.get("/").data

    def test_stale_metadata_db_is_flagged(self, corpus, tmp_path, monkeypatch):
        """stories_meta.db older than the index must not be used silently."""
        import os, shutil, time
        from app_factory import AppState, create_app
        from services.data_store import load_data
        root, _ = corpus
        shutil.copytree(root / "output", tmp_path / "output")
        idx = tmp_path / "output" / "stories_index.json"
        later = time.time() + 60
        os.utime(idx, (later, later))  # index edited after the DB was built
        monkeypatch.chdir(tmp_path)
        state = AppState(index_path="./output/stories_index.json")
        load_data(state)
        assert state.data["_meta_stale"] is True
        page = create_app(state).test_client().get("/")
        assert b"Story metadata is out of date" in page.data

    def test_random_redirects_to_a_story(self, client):
        c, _ = client
        resp = c.get("/random")
        assert resp.status_code in (301, 302)
        assert "/story/" in resp.headers["Location"]


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE INVALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeEncoder:
    """Stands in for a sentence-transformers model; records what it encodes."""
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kw):
        import numpy as np
        self.encoded.extend(texts)
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


class TestRecommendationEmbeddingCache:

    def _engine(self, monkeypatch):
        from recommender import EmbeddingEngine
        eng = EmbeddingEngine()
        fake = _FakeEncoder()
        def load(self=eng):
            eng._model, eng.semantic = fake, True
        monkeypatch.setattr(eng, "_try_load_model", load)
        return eng, fake

    def _stories(self):
        return [{"id": f"S{i}", "title": f"Story {i}", "description": "d",
                 "categories": ["mc"], "full_text": "x",
                 "auto_tags": {"themes": ["romance"]}} for i in range(3)]

    def test_changed_story_is_reencoded_and_others_reused(self, tmp_path, monkeypatch):
        cache = tmp_path / "recs_embeddings.npz"
        stories = self._stories()
        eng, fake = self._engine(monkeypatch)
        eng.build_index(stories, cache_path=cache)
        assert len(fake.encoded) == 3

        stories[1]["auto_tags"]["themes"] = ["trigger phrase"]   # retagged
        eng2, fake2 = self._engine(monkeypatch)
        eng2.build_index(stories, cache_path=cache)
        assert len(fake2.encoded) == 1 and "trigger phrase" in fake2.encoded[0]

    def test_removed_story_is_dropped_from_cache(self, tmp_path, monkeypatch):
        import numpy as np
        cache = tmp_path / "recs_embeddings.npz"
        stories = self._stories()
        self._engine(monkeypatch)[0].build_index(stories, cache_path=cache)
        self._engine(monkeypatch)[0].build_index(stories[:2], cache_path=cache)
        assert sorted(np.load(cache)["ids"].tolist()) == ["S0", "S1"]

    def test_arc_tags_reach_the_signature(self):
        from recommender import EmbeddingEngine
        s = {"id": "A", "title": "t", "auto_tags": {"narrative_arc": {"ending_state": "fully converted"}}}
        assert "fully converted" in EmbeddingEngine().story_signature(s)


class TestSetFitEmbeddingCache:

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        import numpy as np
        import sentence_transformers
        import setfit_trainer as st
        model_dir = tmp_path / "setfit_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"v1")
        monkeypatch.setattr(st, "SETFIT_MODEL_DIR", str(model_dir))
        monkeypatch.setattr(st, "EMBED_CACHE", str(tmp_path / "setfit_embeddings.npz"))
        calls = []

        class FakeST:
            def __init__(self, path): pass
            def encode(self, texts, **kw):
                calls.append(len(texts))
                return np.array([[float(len(t) % 97), 1.0] for t in texts], dtype=np.float32)

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
        stories = [{"id": f"S{i}", "title": f"T{i}", "full_text": ("word " * 400) + str(i)}
                   for i in range(4)]
        return st, stories, calls, model_dir

    def _embedded(self, st, calls):
        return sum(calls) // st.N_CHUNKS

    def test_unchanged_reuses_cache(self, env):
        st, stories, calls, _ = env
        st.build_embeddings(stories)
        calls.clear()
        st.build_embeddings(stories)
        assert self._embedded(st, calls) == 0

    def test_changed_text_reembeds_only_that_story(self, env):
        st, stories, calls, _ = env
        st.build_embeddings(stories)
        calls.clear()
        stories[2]["full_text"] = "different " * 400
        st.build_embeddings(stories)
        assert self._embedded(st, calls) == 1

    def test_retrained_encoder_reembeds_everything(self, env):
        st, stories, calls, model_dir = env
        st.build_embeddings(stories)
        calls.clear()
        (model_dir / "model.safetensors").write_bytes(b"retrained weights")
        st.build_embeddings(stories)
        assert self._embedded(st, calls) == 4

    def test_old_cache_without_fingerprint_is_adopted_and_stamped(self, env):
        import numpy as np
        st, stories, calls, _ = env
        np.savez_compressed(st.EMBED_CACHE.replace(".npz", ""),
                            ids=np.array([s["id"] for s in stories]),
                            vecs=np.ones((4, 2), dtype=np.float32))
        st.build_embeddings(stories)
        assert self._embedded(st, calls) == 0
        assert "encoder_fp" in np.load(st.EMBED_CACHE, allow_pickle=True).files


# ═══════════════════════════════════════════════════════════════════════════════
# WRITE SAFETY — index_store and preferences
# ═══════════════════════════════════════════════════════════════════════════════

class TestIndexStore:

    @pytest.fixture
    def idx(self, tmp_path):
        import index_store
        p = tmp_path / "stories_index.json"
        p.write_text(json.dumps(make_index([SAMPLE_STORY_HYPNOSIS])))
        p.chmod(0o644)
        yield p
        index_store.release_write_lock(p)

    def test_save_roundtrip_keeps_permissions_and_leaves_no_temp(self, idx):
        import index_store
        data = index_store.load_index(idx, for_write=True)
        data["stories"].append({**SAMPLE_STORY_OFFICE})
        index_store.save_index(data, idx, validate=True)
        index_store.save_index(data, idx)          # a second checkpoint is fine
        assert len(json.loads(idx.read_text())["stories"]) == 2
        assert (idx.stat().st_mode & 0o777) == 0o644
        assert [p.name for p in idx.parent.iterdir() if p.suffix == ".tmp"] == []

    def test_change_on_disk_after_load_is_refused(self, idx):
        import index_store, os, time
        data = index_store.load_index(idx, for_write=True)
        idx.write_text(json.dumps(make_index([])))   # "another machine" wrote it
        later = time.time() + 5
        os.utime(idx, (later, later))
        with pytest.raises(index_store.IndexChangedError):
            index_store.save_index(data, idx)
        assert json.loads(idx.read_text())["stories"] == []   # their write survives

    def test_second_writer_is_refused_while_lock_held(self, idx):
        import index_store, subprocess, sys, textwrap
        holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
            import sys, time; sys.path.insert(0, {str(Path(__file__).parent)!r})
            import index_store
            index_store.load_index({str(idx)!r}, for_write=True)
            print("locked", flush=True); time.sleep(30)
        """)], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "locked"
            with pytest.raises(index_store.IndexLockedError, match="pid=" + str(holder.pid)):
                index_store.load_index(idx, for_write=True)
        finally:
            holder.kill(); holder.wait()

    def test_lock_left_by_crashed_process_is_cleared(self, idx):
        import index_store, socket, subprocess, sys
        dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
        Path(str(idx) + ".lock").write_text(json.dumps(
            {"host": socket.gethostname(), "pid": dead.pid, "command": "x", "started_at": "t"}))
        index_store.load_index(idx, for_write=True)   # does not raise
        assert json.loads(Path(str(idx) + ".lock").read_text())["pid"] != dead.pid


class TestRetagSingleValueDims:

    def test_length_and_pov_refreshed_but_svengali_pov_kept(self, tmp_path, monkeypatch):
        import index_store, retag_parallel
        text = "I walked home and I saw my friend. " * 30
        stale = {"pov": "third person", "length_class": "novel"}
        idx = tmp_path / "stories_index.json"
        idx.write_text(json.dumps(make_index([
            {"id": "Plain", "full_text": text, "word_count": 240, "auto_tags": dict(stale)},
            {"id": "Sv", "full_text": text, "word_count": 240,
             "auto_tags": {**stale, "_svengali_tags": ["Third-person"]}},
        ])))
        monkeypatch.setattr(retag_parallel, "INDEX_PATH", str(idx))
        try:
            retag_parallel.main()
        finally:
            index_store.release_write_lock(idx)
        at = {s["id"]: s["auto_tags"] for s in json.loads(idx.read_text())["stories"]}
        assert at["Plain"]["pov"] == "first person"
        assert at["Plain"]["length_class"] == "flash"
        assert at["Sv"]["pov"] == "third person"        # Svengali's value wins
        assert at["Sv"]["length_class"] == "flash"


class TestWaybackChapters:

    def test_each_chapter_fetched_by_its_own_number(self, monkeypatch):
        import wayback_scraper as ws
        urls = [f"https://mcstories.com/S/S{n if n > 1 else ''}.html" for n in (1, 2, 3)]
        monkeypatch.setattr(ws, "DELAY", 0)
        monkeypatch.setattr(ws, "scrape_wayback_landing",
                            lambda wb, slug: {"chapter_urls": list(urls)})
        # Like the real one, the first URL tried is built from chapter_num
        monkeypatch.setattr(ws, "scrape_wayback_chapter",
                            lambda ts, url, chapter_num=1: f"text of chapter {chapter_num}")
        story = ws.scrape_wayback_story("S", "20200101000000")
        assert story["full_text"].split("\n\n* * *\n\n") == [
            "text of chapter 1", "text of chapter 2", "text of chapter 3"]


class TestWaybackMerge:

    def _story(self, sid, words, source="wayback", **kw):
        return {"id": sid, "title": sid, "source": source, "full_text": "word " * words,
                "word_count": words, "chapter_count": 1, "fetch_error": None, **kw}

    def test_merge_adds_new_and_refreshes_shorter_wayback_copies(self, tmp_path, monkeypatch):
        import index_store, wayback_scraper as ws
        main, wb = tmp_path / "stories_index.json", tmp_path / "wayback_index.json"
        monkeypatch.setattr(ws, "MAIN_INDEX", main)
        monkeypatch.setattr(ws, "WAYBACK_INDEX", wb)
        main.write_text(json.dumps(make_index([
            self._story("Short", 100, auto_tags={"pov": "third person"}),
            self._story("Live", 100, source="mcstories"),
            self._story("Cleaned", 100),
        ])))
        # Index holds the complete copy; the sidecar a partial one
        wb.write_text(json.dumps({"stories": [
            self._story("Short", 900, chapter_count=3),
            self._story("Live", 900),
            self._story("New", 50),
            self._story("Cleaned", 105),     # raw text a few words longer than cleaned
        ]}))
        wb.with_suffix(".jsonl").write_text(json.dumps(self._story("Short", 400)) + "\n")
        try:
            ws.merge_into_main()
        finally:
            index_store.release_write_lock(main)
        stories = {s["id"]: s for s in json.loads(main.read_text())["stories"]}
        assert stories["Short"]["word_count"] == 900
        assert stories["Short"]["chapter_count"] == 3
        assert stories["Short"]["auto_tags"] == {"pov": "third person"}   # tags kept
        assert stories["Live"]["word_count"] == 100                     # live story untouched
        assert stories["Cleaned"]["word_count"] == 100                  # small gain ignored
        assert "New" in stories


class TestPreferenceConcurrency:

    def test_parallel_mutations_all_survive_as_valid_json(self, tmp_path):
        import threading
        from recommender import PreferenceStore
        path = tmp_path / "preferences.json"
        prefs = PreferenceStore(path)
        errors = []

        def like_many(t):
            try:
                for i in range(40):
                    prefs.like_story(f"T{t}-S{i}")
                    prefs.record_visit(f"T{t}-S{i}", 1, 1)
            except Exception as e:     # e.g. "dictionary changed size during iteration"
                errors.append(e)

        def save_loop():
            try:
                for _ in range(80):
                    prefs._save()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=like_many, args=(t,)) for t in range(8)]
        threads.append(threading.Thread(target=save_loop))
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []
        saved = json.loads(path.read_text())
        assert len(saved["liked_stories"]) == 8 * 40
        assert len(saved["history"]) == 8 * 40


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA — reproducibility and story-level validation split
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingDataset:

    @pytest.fixture
    def config(self, tmp_path, monkeypatch):
        import train
        body = " ".join(["She listened to the words and felt her thoughts slow."] * 220)
        stories = [{
            "id": f"Story{i}", "title": f"Story {i}", "author": f"Author {i % 4}",
            "categories": ["mc", "mf"], "word_count": 2200, "full_text": body,
            "description": "d", "source": "live",
            "auto_tags": {"themes": ["trigger phrase", "slave mentality", "romance"],
                          "tone": ["psychological"], "content_type": ["male dominant"]},
        } for i in range(40)]
        (tmp_path / "stories_index.json").write_text(json.dumps(make_index(stories)))
        for key, name in [("index_path", "stories_index.json"), ("prefs_path", "preferences.json"),
                          ("train_data_path", "train_data.jsonl"), ("val_data_path", "val_data.jsonl")]:
            monkeypatch.setitem(train.CONFIG, key, str(tmp_path / name))
        monkeypatch.setitem(train.CONFIG, "val_fraction", 0.25)
        return train, tmp_path

    def _titles(self, path):
        out = set()
        for line in path.read_text().splitlines():
            for m in json.loads(line)["messages"]:
                if m["role"] == "user":
                    out.add(re.search(r'titled "([^"]+)"', m["content"]).group(1)
                            if 'titled "' in m["content"] else m["content"][:60])
        return out

    def test_same_seed_gives_identical_files(self, config):
        train, tmp = config
        train.prepare_data()
        first = (tmp / "train_data.jsonl").read_bytes(), (tmp / "val_data.jsonl").read_bytes()
        train.prepare_data()
        assert ((tmp / "train_data.jsonl").read_bytes(), (tmp / "val_data.jsonl").read_bytes()) == first

    def test_validation_stories_never_in_training(self, config):
        train, tmp = config
        train.prepare_data()
        from dataset_utils import is_validation
        val_ids = {f"Story{i}" for i in range(40) if is_validation(f"Story{i}", 0.25, train.CONFIG["dataset_seed"])}
        assert 0 < len(val_ids) < 40
        train_text = (tmp / "train_data.jsonl").read_text()
        val_text = (tmp / "val_data.jsonl").read_text()
        for sid in val_ids:
            title = sid.replace("Story", "Story ")
            assert f'"{title}"' not in train_text.replace('\\"', '"')
            assert title in val_text

    def test_manifest_records_seed_and_hashes(self, config):
        train, tmp = config
        train.prepare_data()
        from dataset_utils import file_sha256
        m = json.loads((tmp / "train_data.manifest.json").read_text())
        assert m["settings"]["dataset_seed"] == train.CONFIG["dataset_seed"]
        assert m["outputs"]["train"]["sha256"] == file_sha256(tmp / "train_data.jsonl")
        assert m["counts"]["val_stories"] > 0 and m["code"]["tag_trust.py"]


class TestChatDataset:

    def _run(self, tmp_path, monkeypatch):
        import sys
        import extract_chatbot_data as ecd
        paras = []
        for i in range(30):
            paras.append(f'"Look at me, number {i}," she said softly. "Keep breathing."')
            paras.append(f"He watched the pendant swing for the {i}th time and felt calm.")
        text = "\n\n".join(paras)
        stories = [{"id": f"Chat{i}", "title": f"Chat {i}", "author": "A", "categories": ["mc"],
                    "word_count": 1500, "full_text": text, "auto_tags": {}} for i in range(30)]
        (tmp_path / "stories_index.json").write_text(json.dumps(make_index(stories)))
        monkeypatch.setattr(ecd, "INDEX_PATH", tmp_path / "stories_index.json")
        monkeypatch.setattr(ecd, "PREFS_PATH", tmp_path / "preferences.json")
        monkeypatch.setattr(ecd, "VAL_PATH", tmp_path / "chat_val.jsonl")
        monkeypatch.setattr(ecd, "VAL_FRACTION", 0.3)
        out = tmp_path / "chat_data.jsonl"
        monkeypatch.setattr(sys, "argv", ["extract_chatbot_data.py", "--output", str(out)])
        ecd.main()
        return out, tmp_path / "chat_val.jsonl"

    def test_reproducible_and_story_level_split(self, tmp_path, monkeypatch):
        out, val = self._run(tmp_path, monkeypatch)
        first = out.read_bytes(), val.read_bytes()
        assert first[0] and first[1], "expected both train and validation conversations"
        out, val = self._run(tmp_path, monkeypatch)
        assert (out.read_bytes(), val.read_bytes()) == first
        assert (tmp_path / "chat_data.manifest.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SETFIT — train / val / test / gold separation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetFitSplit:

    @pytest.fixture(autouse=True)
    def no_gold(self, tmp_path, monkeypatch):
        import setfit_trainer as st
        monkeypatch.setattr(st, "GOLD_PATH", str(tmp_path / "gold_labels.jsonl"))
        monkeypatch.setattr(st, "_gold_cache", None)
        return st

    def test_split_is_stable_and_proportioned(self, no_gold):
        st = no_gold
        ids = [f"Story{i}" for i in range(6000)]
        splits = [st.split_of(i) for i in ids]
        assert splits == [st.split_of(i) for i in ids]
        assert abs(splits.count("test") / 6000 - st.TEST_SHARE) < 0.03
        assert abs(splits.count("val") / 6000 - st.VAL_SHARE) < 0.03

    def test_gold_stories_form_their_own_split(self, no_gold, tmp_path, monkeypatch):
        st = no_gold
        (tmp_path / "gold_labels.jsonl").write_text(
            json.dumps({"id": "Story7", "tag": "mind_control_type:hypnosis", "label": 1}) + "\n")
        monkeypatch.setattr(st, "_gold_cache", None)
        assert st.split_of("Story7") == "gold"

    def _corpus(self, n=900):
        import numpy as np
        stories, rows = [], []
        for i in range(n):
            hyp = i % 3 == 0
            tag = "hypnosis" if hyp else "magic"
            stories.append({"id": f"Story{i}", "full_text": "text",
                            "auto_tags": {"mind_control_type": [tag], "_svengali_tags": [tag],
                                          "_tag_sources": {tag: ["svengali"]}}})
            rows.append([1.0 if hyp else -1.0, float(i % 7) / 7, float(i)])  # last col = story index
        return stories, np.array(rows, dtype=np.float32)

    def test_encoder_pool_is_train_only(self, no_gold):
        st = no_gold
        stories, _ = self._corpus()
        pool = st.encoder_training_pool(stories, {})
        assert pool and all(st.split_of(s["id"]) == "train" for s in pool)

    def test_classifiers_never_fit_on_test_stories(self, no_gold, monkeypatch):
        import setfit_trainer as st
        from sklearn.linear_model import LogisticRegression
        seen = set()

        class Recording(LogisticRegression):
            def fit(self, X, y, *a, **k):
                seen.update(int(v) for v in X[:, -1])
                return super().fit(X, y, *a, **k)

        monkeypatch.setattr(st, "LogisticRegression", Recording)
        monkeypatch.setattr(st, "MIN_POSITIVES", 10)
        stories, vecs = self._corpus()
        clfs = st.train_classifiers(stories, vecs, [("mind_control_type", "hypnosis")])
        assert "mind_control_type:hypnosis" in clfs
        assert clfs["mind_control_type:hypnosis"]["threshold_source"] == "val"
        fitted_splits = {st.split_of(stories[i]["id"]) for i in seen}
        assert fitted_splits <= {"train", "val"}
        assert any(st.split_of(stories[i]["id"]) == "test" for i in range(len(stories)))

    def test_eval_reports_test_split_only(self, no_gold, monkeypatch, capsys):
        import setfit_trainer as st
        monkeypatch.setattr(st, "MIN_POSITIVES", 10)
        stories, vecs = self._corpus()
        clfs = st.train_classifiers(stories, vecs, [("mind_control_type", "hypnosis")])
        st.eval_classifiers(stories, vecs, clfs)
        n_test = sum(st.split_of(s["id"]) == "test" for s in stories)
        assert f"Held-out TEST split: {n_test:,} stories" in capsys.readouterr().out
