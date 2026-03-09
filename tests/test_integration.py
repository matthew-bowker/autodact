"""Integration tests that exercise real Presidio + NameDictionaryDetector.

These tests use the actual spaCy NER model and the generated name data files
(no mocks), ensuring the full detection pipeline works end-to-end.  The LLM
layer is still mocked since it requires a model download.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import spacy.util

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import NameDictionaryDetector
from src.pipeline.presidio_detector import PresidioDetector

# Skip the entire module if the spaCy model isn't installed (e.g. on CI).
pytestmark = pytest.mark.skipif(
    not spacy.util.is_package("en_core_web_lg"),
    reason="spaCy en_core_web_lg model not installed",
)


@pytest.fixture(scope="module")
def presidio() -> PresidioDetector:
    return PresidioDetector()


@pytest.fixture(scope="module")
def name_detector() -> NameDictionaryDetector:
    det = NameDictionaryDetector()
    if not det.loaded:
        pytest.skip("Name data files not generated — run scripts/extract_names.py")
    return det


def _make_doc(lines_text: list[str], fmt: str = "txt") -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format=fmt)


class TestEndToEnd:
    """Full pipeline through Presidio pre-passes + NER + name dictionary."""

    def test_mixed_pii_detected(
        self, presidio: PresidioDetector, name_detector: NameDictionaryDetector,
    ):
        doc = _make_doc([
            "Contact Oluwaseun Adeyemi at o.adeyemi@example.com or +44 7700 900123.",
            "His colleague Priya Krishnamurthy works in the Manchester office.",
            "NI number: AB 12 34 56 C. Postcode: M1 2AB.",
        ])
        lookup = LookupTable()

        presidio.pre_detect_emails(doc, lookup)
        presidio.pre_detect_patterns(doc, lookup)
        presidio.process(doc, lookup)
        name_detector.process(doc, lookup)

        # Email, phone, NI number, and postcode should be caught by pre-passes.
        assert lookup.lookup("o.adeyemi@example.com") is not None
        assert lookup.lookup("+44 7700 900123") is not None
        assert lookup.lookup("M1 2AB") is not None

        # SA/NA names should be caught (by Presidio NER or name dictionary).
        assert lookup.lookup("Oluwaseun") is not None or lookup.lookup("Oluwaseun Adeyemi") is not None
        assert lookup.lookup("Krishnamurthy") is not None or lookup.lookup("Priya Krishnamurthy") is not None

    def test_common_words_not_tagged(
        self, presidio: PresidioDetector, name_detector: NameDictionaryDetector,
    ):
        doc = _make_doc([
            "The park was green and the hill overlooked the street.",
            "Digital services are available for senior researchers.",
        ])
        lookup = LookupTable()

        presidio.pre_detect_emails(doc, lookup)
        presidio.pre_detect_patterns(doc, lookup)
        presidio.process(doc, lookup)
        name_detector.process(doc, lookup)

        # None of these common English words should be tagged as names.
        for word in ["park", "green", "hill", "street", "Digital",
                     "services", "senior", "researchers"]:
            assert lookup.lookup(word) is None, f"{word!r} should not be tagged"

    def test_address_words_excluded(
        self, presidio: PresidioDetector, name_detector: NameDictionaryDetector,
    ):
        doc = _make_doc([
            "Office at 42 Grafton Street, near the Avenue and Park Road.",
        ])
        lookup = LookupTable()

        presidio.pre_detect_emails(doc, lookup)
        presidio.pre_detect_patterns(doc, lookup)
        presidio.process(doc, lookup)
        name_detector.process(doc, lookup)

        # Street/road/avenue/park should never be tagged as NAME.
        for word in ["Street", "Avenue", "Road"]:
            entry = lookup.lookup(word)
            if entry is not None:
                # If tagged, must be LOCATION not NAME.
                for e in lookup.all_entries():
                    if e.original_term == word:
                        assert e.pii_category != "NAME", (
                            f"{word!r} tagged as NAME — should be LOCATION or excluded"
                        )

    def test_existing_tags_preserved(
        self, presidio: PresidioDetector, name_detector: NameDictionaryDetector,
    ):
        doc = _make_doc([
            "Already tagged [NAME 1] and [EMAIL 2] remain intact.",
        ])
        lookup = LookupTable()

        presidio.pre_detect_emails(doc, lookup)
        presidio.pre_detect_patterns(doc, lookup)
        presidio.process(doc, lookup)
        name_detector.process(doc, lookup)

        assert "[NAME 1]" in doc.lines[0].text
        assert "[EMAIL 2]" in doc.lines[0].text
