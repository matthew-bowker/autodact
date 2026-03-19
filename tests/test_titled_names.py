"""Tests for the title + next-word heuristic in PresidioDetector."""

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable
from src.pipeline.presidio_detector import _TITLE_NAME_RE


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


# ---------------------------------------------------------------------------
# Regex unit tests
# ---------------------------------------------------------------------------

class TestTitleNameRegex:
    """Validate the _TITLE_NAME_RE pattern in isolation."""

    @pytest.mark.parametrize("text,expected_name", [
        ("Mr Smith", "Smith"),
        ("Mr. Smith", "Smith"),
        ("Mrs Jones", "Jones"),
        ("Ms Williams", "Williams"),
        ("Miss Taylor", "Taylor"),
        ("Dr Johnson", "Johnson"),
        ("Dr. Johnson", "Johnson"),
        ("Prof Davies", "Davies"),
        ("Professor Davies", "Davies"),
        ("Rev Adams", "Adams"),
        ("Reverend Adams", "Adams"),
        ("Sir Keir", "Keir"),
        ("Dame Judi", "Judi"),
        ("Lord Byron", "Byron"),
        ("Lady Gaga", "Gaga"),
        ("Cllr Patel", "Patel"),
        ("Councillor Patel", "Patel"),
        ("Sen Warren", "Warren"),
        ("Senator Warren", "Warren"),
        ("Gov Abbott", "Abbott"),
        ("Governor Abbott", "Abbott"),
        ("Fr Thomas", "Thomas"),
        ("Father Thomas", "Thomas"),
        ("Sgt Baker", "Baker"),
        ("Sergeant Baker", "Baker"),
        ("Capt Kirk", "Kirk"),
        ("Captain Kirk", "Kirk"),
        ("Col Mustard", "Mustard"),
        ("Gen Powell", "Powell"),
        ("Insp Morse", "Morse"),
        ("Inspector Morse", "Morse"),
        ("Det Carter", "Carter"),
        ("Detective Carter", "Carter"),
    ])
    def test_single_word_name(self, text, expected_name):
        m = _TITLE_NAME_RE.search(text)
        assert m is not None, f"No match for: {text}"
        assert m.group(1).strip() == expected_name

    @pytest.mark.parametrize("text,expected_name", [
        ("Mr John Smith", "John Smith"),
        ("Dr. Jane Williams", "Jane Williams"),
        ("Prof Alan Turing", "Alan Turing"),
        ("Mrs Mary Jane Watson", "Mary Jane Watson"),
    ])
    def test_multi_word_name(self, text, expected_name):
        m = _TITLE_NAME_RE.search(text)
        assert m is not None, f"No match for: {text}"
        assert m.group(1).strip() == expected_name

    @pytest.mark.parametrize("text,expected_name", [
        ("Mr O'Brien", "O'Brien"),
        ("Dr Al-Hassan", "Al-Hassan"),
        ("Mrs O\u2019Connor", "O\u2019Connor"),  # curly apostrophe
    ])
    def test_hyphenated_and_apostrophed_names(self, text, expected_name):
        m = _TITLE_NAME_RE.search(text)
        assert m is not None, f"No match for: {text}"
        assert m.group(1).strip() == expected_name

    def test_uk_style_no_period(self):
        m = _TITLE_NAME_RE.search("Dr Smith said hello")
        assert m is not None
        assert m.group(1).strip() == "Smith"

    def test_us_style_with_period(self):
        m = _TITLE_NAME_RE.search("Dr. Smith said hello")
        assert m is not None
        assert m.group(1).strip() == "Smith"

    def test_no_match_lowercase_after_title(self):
        m = _TITLE_NAME_RE.search("Mr the quick fox")
        assert m is None

    def test_no_match_without_title(self):
        m = _TITLE_NAME_RE.search("John Smith went home")
        assert m is None

    def test_mid_sentence(self):
        m = _TITLE_NAME_RE.search("I spoke to Dr. Jane Smith about it")
        assert m is not None
        assert m.group(1).strip() == "Jane Smith"


# ---------------------------------------------------------------------------
# Integration with PresidioDetector.pre_detect_titled_names
# ---------------------------------------------------------------------------

try:
    from src.pipeline.presidio_detector import PresidioDetector
    _HAS_PRESIDIO = True
except ImportError:
    _HAS_PRESIDIO = False


@pytest.mark.skipif(not _HAS_PRESIDIO, reason="Presidio not installed")
class TestPreDetectTitledNames:
    @pytest.fixture
    def detector(self):
        # Create with a dummy analyzer — we only need pre_detect_titled_names
        # which doesn't use the analyzer.
        class _MinimalAnalyzer:
            pass
        det = PresidioDetector.__new__(PresidioDetector)
        det._analyzer = _MinimalAnalyzer()
        det._score_threshold = 0.5
        det._enabled_categories = None
        return det

    def test_single_name_registered(self, detector):
        doc = _make_doc(["I met Mr Johnson yesterday."])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        assert lookup.lookup("Johnson") is not None
        assert "[NAME" in doc.lines[0].text

    def test_multi_word_name_registered(self, detector):
        doc = _make_doc(["Dr. Jane Smith is here."])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        tag = lookup.lookup("Jane Smith")
        assert tag is not None
        # Individual words should be aliases
        assert lookup.lookup("Jane") == tag
        assert lookup.lookup("Smith") == tag

    def test_name_replaced_in_document(self, detector):
        doc = _make_doc(["Mr Williams said hello. Williams left."])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        text = doc.lines[0].text
        assert "Williams" not in text.replace("[", "").split("]")[0] or "[NAME" in text

    def test_already_tagged_skipped(self, detector):
        doc = _make_doc(["Mr [NAME 1] is here."])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        assert len(lookup) == 0

    def test_existing_lookup_reused(self, detector):
        doc = _make_doc(["Mr Smith arrived. Dr. Smith left."])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        # Should only have one entry, reused for both occurrences
        entries = [e for e in lookup.all_entries() if e.original_term == "Smith"]
        assert len(entries) == 1

    def test_multiple_titles_same_doc(self, detector):
        doc = _make_doc([
            "Dr. Adams examined the patient.",
            "Prof. Baker reviewed the results.",
        ])
        lookup = LookupTable()
        detector.pre_detect_titled_names(doc, lookup)
        assert lookup.lookup("Adams") is not None
        assert lookup.lookup("Baker") is not None

    def test_short_word_alias_skipped(self):
        """Words < 3 chars should not be registered as aliases."""
        # "Mr Li Bo" — "Li" is only 2 chars, should not get an alias
        doc = _make_doc(["Mr Li Bo arrived."])
        det = PresidioDetector.__new__(PresidioDetector)
        det._analyzer = type("A", (), {})()
        det._score_threshold = 0.5
        det._enabled_categories = None
        lookup = LookupTable()
        det.pre_detect_titled_names(doc, lookup)
        full_tag = lookup.lookup("Li Bo")
        assert full_tag is not None
        # "Li" is only 2 chars — should NOT have an alias
        assert lookup.lookup("Li") is None
        # "Bo" is also only 2 chars
        assert lookup.lookup("Bo") is None
