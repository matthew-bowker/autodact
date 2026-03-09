from pathlib import Path

from src.pipeline.column_detector import ColumnDetector, _find_name_groups
from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable


def _make_csv_doc(headers: list[str], rows: list[list[str]]) -> Document:
    """Build a Document with a header row + data rows (CSV-style)."""
    lines: list[Line] = []
    # Header row (line_number=1)
    header_cells = [Cell(text=h, col_index=i) for i, h in enumerate(headers)]
    lines.append(Line(cells=header_cells, line_number=1))
    # Data rows
    for row_idx, row in enumerate(rows):
        cells = [Cell(text=v, col_index=i) for i, v in enumerate(row)]
        lines.append(Line(cells=cells, line_number=row_idx + 2))
    return Document(lines=lines, source_path=Path("test.csv"), source_format="csv")


def test_basic_column_tagging():
    doc = _make_csv_doc(
        ["ID", "Email"],
        [["1", "jane@example.com"], ["2", "bob@example.com"]],
    )
    lookup = LookupTable()
    mapping = {0: "Skip", 1: "EMAIL"}
    ColumnDetector().process(doc, lookup, mapping)
    # Email column should be tagged
    assert lookup.lookup("jane@example.com") is not None
    assert lookup.lookup("bob@example.com") is not None
    # ID column should be untouched
    assert doc.lines[1].cells[0].text == "1"


def test_header_row_not_tagged():
    doc = _make_csv_doc(["Name"], [["Jane"]])
    lookup = LookupTable()
    mapping = {0: "NAME"}
    ColumnDetector().process(doc, lookup, mapping)
    # Header value "Name" should NOT be registered
    assert lookup.lookup("Name") is None
    # Data value "Jane" should be registered
    assert lookup.lookup("Jane") is not None


def test_skip_and_freetext_ignored():
    doc = _make_csv_doc(
        ["ID", "Notes"],
        [["1", "Some notes here"]],
    )
    lookup = LookupTable()
    mapping = {0: "Skip", 1: "FREETEXT"}
    ColumnDetector().process(doc, lookup, mapping)
    assert len(lookup) == 0


def test_name_combining():
    """Adjacent NAME columns should be combined into a full name."""
    doc = _make_csv_doc(
        ["First Name", "Last Name", "City"],
        [["Jane", "Smith", "London"]],
    )
    lookup = LookupTable()
    mapping = {0: "NAME", 1: "NAME", 2: "LOCATION"}
    ColumnDetector().process(doc, lookup, mapping)
    # Full name registered
    assert lookup.lookup("Jane Smith") is not None
    # Individual parts also registered
    assert lookup.lookup("Jane") is not None
    assert lookup.lookup("Smith") is not None
    # Same tag for all three
    tag = lookup.lookup("Jane Smith")
    assert lookup.lookup("Jane") == tag
    assert lookup.lookup("Smith") == tag


def test_replace_all_propagates():
    """Column-detected PII should be replaced throughout the document."""
    doc = _make_csv_doc(
        ["Name", "Notes"],
        [["Jane", "Jane is a researcher."]],
    )
    lookup = LookupTable()
    mapping = {0: "NAME", 1: "FREETEXT"}
    ColumnDetector().process(doc, lookup, mapping)
    # "Jane" in the Notes cell should also be replaced
    assert "Jane" not in doc.lines[1].cells[1].text
    assert "[NAME" in doc.lines[1].cells[1].text


def test_find_name_groups():
    mapping = {0: "Skip", 1: "NAME", 2: "NAME", 3: "EMAIL", 5: "NAME"}
    groups = _find_name_groups(mapping)
    assert groups == [[1, 2], [5]]


def test_empty_cells_skipped():
    doc = _make_csv_doc(
        ["Name", "City"],
        [["", "London"]],
    )
    lookup = LookupTable()
    mapping = {0: "NAME", 1: "LOCATION"}
    ColumnDetector().process(doc, lookup, mapping)
    # Empty name cell should not be registered
    assert lookup.lookup("") is None
    assert lookup.lookup("London") is not None


def test_single_column_full_name_splits_parts():
    """A single 'Full Name' column with 'Jane Smith' should register parts."""
    doc = _make_csv_doc(
        ["Full Name", "Notes"],
        [["Jane Smith", "Jane is the lead."]],
    )
    lookup = LookupTable()
    mapping = {0: "NAME", 1: "FREETEXT"}
    ColumnDetector().process(doc, lookup, mapping)
    # Full name registered
    tag = lookup.lookup("Jane Smith")
    assert tag is not None
    # Individual parts aliased to the same tag
    assert lookup.lookup("Jane") == tag
    assert lookup.lookup("Smith") == tag
    # "Jane" in Notes cell (same row) is replaced
    assert "Jane" not in doc.lines[1].cells[1].text


def test_name_parts_scoped_to_row():
    """Name parts should not cross-contaminate other rows."""
    doc = _make_csv_doc(
        ["First Name", "Last Name", "Notes"],
        [
            ["Jane", "Smith", "Some notes"],
            ["Bob", "Jones", "Jane said hello"],
        ],
    )
    lookup = LookupTable()
    mapping = {0: "NAME", 1: "NAME", 2: "FREETEXT"}
    ColumnDetector().process(doc, lookup, mapping)
    tag_jane = lookup.lookup("Jane Smith")
    tag_bob = lookup.lookup("Bob Jones")
    assert tag_jane is not None
    assert tag_bob is not None
    assert tag_jane != tag_bob
    # Row 2 (line 3): "Jane" should NOT be replaced by column detector
    # (it's on a different row — Presidio will handle it later)
    assert "Jane" in doc.lines[2].cells[2].text
