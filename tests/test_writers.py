import csv
from pathlib import Path

from src.pipeline.document import Cell, Document, Line
from src.pipeline.parsers import parse_csv, parse_docx, parse_txt, parse_xlsx
from src.pipeline.writers import write_csv, write_docx, write_file, write_txt, write_xlsx

FIXTURES = Path(__file__).parent / "fixtures"


def _make_simple_doc(fmt: str = "txt") -> Document:
    return Document(
        lines=[
            Line(cells=[Cell(text="Hello world")], line_number=1),
            Line(cells=[Cell(text="Second line")], line_number=2),
        ],
        source_path=Path("test.txt"),
        source_format=fmt,
    )


def test_write_txt(tmp_path: Path):
    doc = _make_simple_doc()
    out = tmp_path / "out.txt"
    write_txt(doc, out)
    assert out.exists()
    content = out.read_text()
    assert "Hello world\n" in content
    assert "Second line\n" in content


def test_write_csv(tmp_path: Path):
    doc = Document(
        lines=[
            Line(cells=[Cell(text="Name"), Cell(text="Email")], line_number=1),
            Line(cells=[Cell(text="Jane"), Cell(text="jane@x.com")], line_number=2),
        ],
        source_path=Path("test.csv"),
        source_format="csv",
    )
    out = tmp_path / "out.csv"
    write_csv(doc, out)
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Name", "Email"]
    assert rows[1] == ["Jane", "jane@x.com"]


def test_write_xlsx(tmp_path: Path):
    doc = Document(
        lines=[
            Line(
                cells=[
                    Cell(text="Name", sheet_name="Sheet1", col_index=0),
                    Cell(text="Email", sheet_name="Sheet1", col_index=1),
                ],
                line_number=1,
            ),
            Line(
                cells=[
                    Cell(text="Jane", sheet_name="Sheet1", col_index=0),
                    Cell(text="jane@x.com", sheet_name="Sheet1", col_index=1),
                ],
                line_number=2,
            ),
        ],
        source_path=Path("test.xlsx"),
        source_format="xlsx",
    )
    out = tmp_path / "out.xlsx"
    write_xlsx(doc, out)
    assert out.exists()
    reparsed = parse_xlsx(out)
    assert len(reparsed.lines) == 2
    assert reparsed.lines[0].cells[0].text == "Name"


def test_write_docx(tmp_path: Path):
    doc = _make_simple_doc(fmt="docx")
    out = tmp_path / "out.docx"
    write_docx(doc, out)
    assert out.exists()
    reparsed = parse_docx(out)
    assert len(reparsed.lines) == 2
    assert reparsed.lines[0].text == "Hello world"


def test_roundtrip_txt(tmp_path: Path):
    original = parse_txt(FIXTURES / "sample.txt")
    out = tmp_path / "roundtrip.txt"
    write_txt(original, out)
    reparsed = parse_txt(out)
    assert len(reparsed.lines) == len(original.lines)
    for orig, reparse in zip(original.lines, reparsed.lines):
        assert orig.text == reparse.text


def test_roundtrip_csv(tmp_path: Path):
    original = parse_csv(FIXTURES / "sample.csv")
    out = tmp_path / "roundtrip.csv"
    write_csv(original, out)
    reparsed = parse_csv(out)
    assert len(reparsed.lines) == len(original.lines)
    for orig, reparse in zip(original.lines, reparsed.lines):
        assert len(orig.cells) == len(reparse.cells)
        for oc, rc in zip(orig.cells, reparse.cells):
            assert oc.text == rc.text


def test_write_file_preserve_format(tmp_path: Path):
    doc = _make_simple_doc(fmt="txt")
    out = tmp_path / "out.txt"
    write_file(doc, out, preserve_format=True)
    assert out.exists()


def test_write_file_force_txt(tmp_path: Path):
    doc = _make_simple_doc(fmt="csv")
    out = tmp_path / "out.txt"
    write_file(doc, out, preserve_format=False)
    assert out.exists()
    content = out.read_text()
    assert "Hello world" in content
