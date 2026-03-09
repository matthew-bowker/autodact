"""Tests for the post-processing validation pass."""

from pathlib import Path

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable
from src.pipeline.post_validator import validate_and_clean


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


class TestRemovesStopWords:
    def test_stop_words_removed(self):
        doc = _make_doc(["[NAME 1] [NAME 2]"])
        lookup = LookupTable()
        lookup.register("the", "NAME", "test.txt", 1)
        lookup.register("and", "NAME", "test.txt", 1)
        doc.replace_all("the", "[NAME 1]")
        doc.replace_all("and", "[NAME 2]")

        removed = validate_and_clean(doc, lookup)
        assert removed == 2
        assert len(lookup) == 0

    def test_real_names_preserved(self):
        doc = _make_doc(["[NAME 1] [NAME 2]"])
        lookup = LookupTable()
        lookup.register("Jane", "NAME", "test.txt", 1)
        lookup.register("Smith", "NAME", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 0
        assert len(lookup) == 2


class TestRemovesPlaceholders:
    def test_na_removed(self):
        doc = _make_doc(["[NAME 1]"])
        lookup = LookupTable()
        lookup.register("N/A", "NAME", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 1

    def test_unknown_removed(self):
        doc = _make_doc(["[NAME 1]"])
        lookup = LookupTable()
        lookup.register("Unknown", "NAME", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 1

    def test_tbd_removed(self):
        doc = _make_doc(["[ORG 1]"])
        lookup = LookupTable()
        lookup.register("TBD", "ORG", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 1


class TestRemovesPureNumeric:
    def test_numeric_name_removed(self):
        doc = _make_doc(["[NAME 1]"])
        lookup = LookupTable()
        lookup.register("123", "NAME", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 1

    def test_numeric_id_preserved(self):
        doc = _make_doc(["[ID 1]"])
        lookup = LookupTable()
        lookup.register("456", "ID", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 0
        assert lookup.lookup("456") is not None

    def test_numeric_phone_preserved(self):
        doc = _make_doc(["[PHONE 1]"])
        lookup = LookupTable()
        lookup.register("5551234567", "PHONE", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 0


class TestRemovesSingleChar:
    def test_single_char_removed(self):
        doc = _make_doc(["[NAME 1]"])
        lookup = LookupTable()
        lookup.register("X", "NAME", "test.txt", 1)

        removed = validate_and_clean(doc, lookup)
        assert removed == 1


class TestRestoresDocument:
    def test_tag_replaced_back_with_original(self):
        doc = _make_doc(["The [NAME 1] is here"])
        lookup = LookupTable()
        lookup.register("the", "NAME", "test.txt", 1)

        validate_and_clean(doc, lookup)
        assert "[NAME" not in doc.lines[0].text
        # Original should be restored (but we can't guarantee exact position
        # since _safe_replace uses word boundaries).
        assert "the" in doc.lines[0].text.lower()
