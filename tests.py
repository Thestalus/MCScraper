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
import re
import sys
import tempfile
from pathlib import Path

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
        """Tags should appear in the meta header of training records."""
        from tagger import rebuild_jsonl_with_tags
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = self._make_tagged_index(Path(tmpdir))
            rebuild_jsonl_with_tags(idx, Path(tmpdir))
            lines = (Path(tmpdir) / "training_raw_pretraining.jsonl").read_text().splitlines()
            # At least one record should contain theme or MC type metadata
            full_output = "\n".join(lines)
            has_meta = ("Themes:" in full_output or
                        "Mind Control Method:" in full_output or
                        "Setting:" in full_output)
            assert has_meta


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
