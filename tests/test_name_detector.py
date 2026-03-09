from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import NameDictionaryDetector


@pytest.fixture(scope="module")
def detector() -> NameDictionaryDetector:
    det = NameDictionaryDetector()
    if not det.loaded:
        pytest.skip("Name data files not generated — run scripts/extract_names.py")
    return det


def _make_doc(text: str) -> Document:
    lines = []
    for i, line_text in enumerate(text.splitlines(), start=1):
        lines.append(Line(cells=[Cell(text=line_text)], line_number=i))
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


class TestHighConfidenceNames:
    """Names NOT in common-words list should be tagged directly."""

    @pytest.mark.parametrize("name", [
        "Oluwaseun", "Adeyemi", "Krishnamurthy", "Subramaniam",
        "Chakraborty", "Boudiaf", "Kowalski", "Nakamura",
    ])
    def test_non_western_name_caught(self, detector: NameDictionaryDetector, name: str):
        doc = _make_doc(f"Contacted {name} about the report.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        assert lookup.lookup(name) is not None, f"{name} should be tagged"

    def test_non_western_full_name(self, detector: NameDictionaryDetector):
        doc = _make_doc("Contacted Oluwaseun Adeyemi about the report.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        assert lookup.lookup("Oluwaseun") is not None
        assert lookup.lookup("Adeyemi") is not None


class TestCommonWordFiltering:
    """Common English words that are also names should NOT be tagged alone."""

    @pytest.mark.parametrize("word", [
        "Park", "Green", "Rose", "Grace", "Will", "Lane", "Hill", "Stone",
    ])
    def test_common_word_not_tagged_alone(self, detector: NameDictionaryDetector, word: str):
        doc = _make_doc(f"The {word} was visible from the window.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        assert lookup.lookup(word) is None, f"{word} should NOT be tagged alone"


class TestPairDetection:
    """Common-word names should be tagged when adjacent to another name."""

    def test_common_first_name_with_surname(self, detector: NameDictionaryDetector):
        doc = _make_doc("Spoke with Grace Okafor yesterday.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        # "Grace" is in web2, but "Okafor" is not — pair detection tags both
        assert lookup.lookup("Okafor") is not None
        assert lookup.lookup("Grace") is not None

    def test_common_surname_with_first_name(self, detector: NameDictionaryDetector):
        doc = _make_doc("Report from Priya Singh on the data.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        assert lookup.lookup("Priya") is not None
        assert lookup.lookup("Singh") is not None


class TestExistingTagsSkipped:
    """Text already inside [TAG] brackets should be left alone."""

    def test_tagged_text_not_reprocessed(self, detector: NameDictionaryDetector):
        doc = _make_doc("Contacted [NAME 1] Adeyemi about the project.")
        lookup = LookupTable()
        detector.process(doc, lookup)
        # Adeyemi should be tagged, but [NAME 1] should not be touched
        assert lookup.lookup("Adeyemi") is not None
        output = doc.lines[0].text
        assert "[NAME 1]" in output


class TestProgressCallback:
    def test_progress_called(self, detector: NameDictionaryDetector):
        doc = _make_doc("Line one.\nLine two.")
        lookup = LookupTable()
        progress: list[tuple[int, int]] = []
        detector.process(doc, lookup, on_progress=lambda c, t: progress.append((c, t)))
        assert len(progress) == 2
        assert progress[-1] == (2, 2)
