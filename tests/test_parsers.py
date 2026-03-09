from pathlib import Path

import pytest

from src.pipeline.parsers import (
    parse_csv,
    parse_docx,
    parse_file,
    parse_txt,
    parse_xlsx,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_txt():
    doc = parse_txt(FIXTURES / "sample.txt")
    assert doc.source_format == "txt"
    assert len(doc.lines) == 5
    assert "Jane Smith" in doc.lines[0].text
    assert len(doc.lines[0].cells) == 1


def test_parse_csv():
    doc = parse_csv(FIXTURES / "sample.csv")
    assert doc.source_format == "csv"
    assert len(doc.lines) == 3  # header + 2 data rows
    assert len(doc.lines[0].cells) == 3
    assert doc.lines[1].cells[0].text == "Jane Smith"
    assert doc.lines[1].cells[1].text == "jane.smith@acme.com"


def test_parse_xlsx():
    doc = parse_xlsx(FIXTURES / "sample.xlsx")
    assert doc.source_format == "xlsx"
    # Sheet1: 3 rows + Sheet2: 2 rows = 5 lines
    assert len(doc.lines) == 5
    assert doc.lines[0].cells[0].sheet_name == "Sheet1"
    assert doc.lines[1].cells[0].text == "Jane Smith"
    assert doc.lines[3].cells[0].sheet_name == "Sheet2"


def test_parse_docx():
    doc = parse_docx(FIXTURES / "sample.docx")
    assert doc.source_format == "docx"
    assert len(doc.lines) == 3
    assert "Jane Smith" in doc.lines[0].text
    assert "Bob Jones" in doc.lines[2].text


def test_parse_file_dispatch():
    doc = parse_file(FIXTURES / "sample.txt")
    assert doc.source_format == "txt"


def test_parse_file_unsupported():
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_file(Path("test.pdf"))


def test_parse_csv_cells_have_indices():
    doc = parse_csv(FIXTURES / "sample.csv")
    row = doc.lines[1]
    assert row.cells[0].col_index == 0
    assert row.cells[1].col_index == 1
    assert row.cells[2].col_index == 2


def test_parse_xlsx_preserves_cell_metadata():
    doc = parse_xlsx(FIXTURES / "sample.xlsx")
    cell = doc.lines[1].cells[0]
    assert cell.sheet_name == "Sheet1"
    assert cell.col_index == 0
