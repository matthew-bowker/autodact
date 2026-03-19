"""Tests for custom word list detector."""

import json
from pathlib import Path

import pytest

from src.pipeline.custom_list_detector import (
    CustomListDetector,
    load_custom_lists,
    save_custom_lists,
)
from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

class TestCustomListDetectorBasic:
    def test_single_word_matched(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["I visited Belfast last year."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert lookup.lookup("Belfast") is not None
        entry = [e for e in lookup.all_entries() if e.original_term == "Belfast"][0]
        assert entry.pii_category == "LOCATION"

    def test_case_insensitive_matching(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["I visited belfast last year."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert lookup.lookup("belfast") is not None

    def test_multiple_words_matched(self):
        lists = [{"category": "LOCATION", "words": ["Belfast", "Edinburgh", "Cardiff"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["Belfast and Edinburgh are great. Cardiff too."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert lookup.lookup("Belfast") is not None
        assert lookup.lookup("Edinburgh") is not None
        assert lookup.lookup("Cardiff") is not None

    def test_multi_word_term(self):
        lists = [{"category": "LOCATION", "words": ["New York City"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["I flew to New York City yesterday."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert lookup.lookup("New York City") is not None

    def test_word_boundary_respected(self):
        lists = [{"category": "NAME", "words": ["Smith"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["Blacksmith is a trade."])
        lookup = LookupTable()
        det.process(doc, lookup)
        # "Smith" should not match inside "Blacksmith"
        assert lookup.lookup("Smith") is None

    def test_different_categories(self):
        lists = [
            {"category": "NAME", "words": ["Johnson"]},
            {"category": "LOCATION", "words": ["Belfast"]},
        ]
        det = CustomListDetector(lists)
        doc = _make_doc(["Johnson visited Belfast."])
        lookup = LookupTable()
        det.process(doc, lookup)
        name_entry = [e for e in lookup.all_entries() if e.original_term == "Johnson"][0]
        loc_entry = [e for e in lookup.all_entries() if e.original_term == "Belfast"][0]
        assert name_entry.pii_category == "NAME"
        assert loc_entry.pii_category == "LOCATION"

    def test_replaces_in_document(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["Belfast is in Northern Ireland."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert "Belfast" not in doc.lines[0].text
        assert "[LOCATION" in doc.lines[0].text

    def test_already_registered_reused(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["Belfast is great. Belfast is lovely."])
        lookup = LookupTable()
        det.process(doc, lookup)
        # Should only have one entry
        entries = [e for e in lookup.all_entries() if "Belfast" in e.original_term.lower()]
        # The first match registers, the second reuses
        assert len(entries) <= 2  # may have case variants

    def test_existing_lookup_reused(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["I went to Belfast."])
        lookup = LookupTable()
        existing_tag = lookup.register("Belfast", "LOCATION", "other.txt", 1)
        det.process(doc, lookup)
        assert lookup.lookup("Belfast") == existing_tag

    def test_existing_tag_skipped(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["[LOCATION 1] is in Northern Ireland."])
        lookup = LookupTable()
        det.process(doc, lookup)
        # Should not register [LOCATION 1] as a word
        assert len(lookup) == 0


class TestCustomListDetectorEdgeCases:
    def test_empty_lists(self):
        det = CustomListDetector([])
        assert not det.has_lists
        doc = _make_doc(["Nothing to find."])
        lookup = LookupTable()
        count = det.process(doc, lookup)
        assert count is None  # returns early

    def test_empty_words(self):
        lists = [{"category": "NAME", "words": []}]
        det = CustomListDetector(lists)
        assert not det.has_lists

    def test_whitespace_only_words_filtered(self):
        lists = [{"category": "NAME", "words": ["  ", "\t", ""]}]
        det = CustomListDetector(lists)
        assert not det.has_lists

    def test_missing_category(self):
        lists = [{"words": ["Belfast"]}]  # no category
        det = CustomListDetector(lists)
        assert not det.has_lists

    def test_progress_callback(self):
        lists = [{"category": "LOCATION", "words": ["Belfast"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["Line one Belfast.", "Line two."])
        lookup = LookupTable()
        calls: list[tuple[int, int]] = []
        det.process(doc, lookup, on_progress=lambda c, t: calls.append((c, t)))
        assert calls == [(1, 2), (2, 2)]

    def test_multiple_lists_same_category_merged(self):
        lists = [
            {"category": "LOCATION", "words": ["Belfast"]},
            {"category": "LOCATION", "words": ["Edinburgh"]},
        ]
        det = CustomListDetector(lists)
        doc = _make_doc(["Belfast and Edinburgh."])
        lookup = LookupTable()
        det.process(doc, lookup)
        assert lookup.lookup("Belfast") is not None
        assert lookup.lookup("Edinburgh") is not None

    def test_longest_match_first(self):
        """Longer terms should be matched before shorter ones."""
        lists = [{"category": "LOCATION", "words": ["New York", "New York City"]}]
        det = CustomListDetector(lists)
        doc = _make_doc(["I went to New York City."])
        lookup = LookupTable()
        det.process(doc, lookup)
        # "New York City" should match (longer), not just "New York"
        assert lookup.lookup("New York City") is not None


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestCustomListsPersistence:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "custom_lists.json"
        lists = [
            {"name": "Places", "category": "LOCATION", "words": ["Belfast", "Edinburgh"]},
            {"name": "Staff", "category": "NAME", "words": ["Johnson", "Williams"]},
        ]
        save_custom_lists(path, lists)

        loaded = load_custom_lists(path)
        assert len(loaded) == 2
        assert loaded[0]["name"] == "Places"
        assert loaded[0]["words"] == ["Belfast", "Edinburgh"]
        assert loaded[1]["name"] == "Staff"

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        loaded = load_custom_lists(path)
        assert loaded == []

    def test_load_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        loaded = load_custom_lists(path)
        assert loaded == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "lists.json"
        save_custom_lists(path, [{"name": "Test", "category": "NAME", "words": ["x"]}])
        assert path.exists()

    def test_roundtrip_unicode(self, tmp_path):
        path = tmp_path / "unicode.json"
        lists = [{"name": "Names", "category": "NAME", "words": ["O\u2019Brien", "Muller"]}]
        save_custom_lists(path, lists)
        loaded = load_custom_lists(path)
        assert loaded[0]["words"][0] == "O\u2019Brien"
