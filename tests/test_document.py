from pathlib import Path

from src.pipeline.document import (
    Cell,
    Document,
    Line,
    reunify_sublines,
    split_long_lines,
)


def _make_doc(lines_text: list[str], fmt: str = "txt") -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format=fmt)


def test_cell_creation():
    c = Cell(text="hello")
    assert c.text == "hello"
    assert c.sheet_name is None


def test_line_text_single_cell():
    line = Line(cells=[Cell(text="hello world")], line_number=1)
    assert line.text == "hello world"


def test_line_text_multi_cell():
    line = Line(
        cells=[Cell(text="a"), Cell(text="b"), Cell(text="c")],
        line_number=1,
    )
    assert line.text == "a | b | c"


def test_line_replace_all():
    line = Line(
        cells=[Cell(text="Jane went to Jane's house"), Cell(text="Jane said hi")],
        line_number=1,
    )
    line.replace_all("Jane", "[NAME 1]")
    assert line.cells[0].text == "[NAME 1] went to [NAME 1]'s house"
    assert line.cells[1].text == "[NAME 1] said hi"


def test_document_replace_all():
    doc = _make_doc(["Jane is here", "Jane went home", "No names"])
    doc.replace_all("Jane", "[NAME 1]")
    assert doc.lines[0].text == "[NAME 1] is here"
    assert doc.lines[1].text == "[NAME 1] went home"
    assert doc.lines[2].text == "No names"


def test_split_long_lines_short_unchanged():
    doc = _make_doc(["short line", "another short line"])
    split_long_lines(doc, max_chars=500)
    assert len(doc.lines) == 2
    assert not doc.lines[0].is_subline
    assert not doc.lines[1].is_subline


def test_split_long_lines_sentence_split():
    long_text = "First sentence. " * 40  # ~640 chars
    doc = _make_doc([long_text.strip()])
    split_long_lines(doc, max_chars=200)
    assert len(doc.lines) > 1
    assert all(line.line_number == 1 for line in doc.lines)
    assert not doc.lines[0].is_subline
    assert doc.lines[1].is_subline


def test_split_long_lines_cell_split():
    cells = [Cell(text=f"cell content {i}") for i in range(30)]
    line = Line(cells=cells, line_number=1)
    doc = Document(lines=[line], source_path=Path("test.csv"), source_format="csv")
    split_long_lines(doc, max_chars=100)
    assert len(doc.lines) > 1
    assert all(line.line_number == 1 for line in doc.lines)


def test_reunify_sublines():
    lines = [
        Line(cells=[Cell(text="part 1.")], line_number=1, is_subline=False),
        Line(cells=[Cell(text="part 2.")], line_number=1, is_subline=True),
        Line(cells=[Cell(text="separate line")], line_number=2, is_subline=False),
    ]
    doc = Document(lines=lines, source_path=Path("test.txt"), source_format="txt")
    reunify_sublines(doc)
    assert len(doc.lines) == 2
    assert doc.lines[0].text == "part 1. | part 2."
    assert doc.lines[1].text == "separate line"


def test_split_then_reunify_roundtrip():
    original_text = "First sentence. " * 40
    doc = _make_doc([original_text.strip(), "Short line"])
    original_count = len(doc.lines)
    split_long_lines(doc, max_chars=200)
    assert len(doc.lines) > original_count
    reunify_sublines(doc)
    assert len(doc.lines) == original_count


def test_split_no_sentence_boundary():
    long_text = "a" * 600
    doc = _make_doc([long_text])
    split_long_lines(doc, max_chars=200)
    assert len(doc.lines) > 1
    for line in doc.lines:
        assert len(line.text) <= 200


def test_replace_all_skips_brackets():
    """replace_all must not modify text inside [TAG N] brackets."""
    line = Line(
        cells=[Cell(text="[NAME 1] works at NAME Corp")],
        line_number=1,
    )
    line.replace_all("NAME", "[JOBTITLE 3]")
    # The "NAME" inside [NAME 1] should be untouched; only the bare one is replaced
    assert line.cells[0].text == "[NAME 1] works at [JOBTITLE 3] Corp"


def test_replace_all_preserves_multiple_tags():
    """Multiple tags in the same cell should all be preserved."""
    doc = _make_doc(["[NAME 1] met [NAME 2] at the NAME conference"])
    doc.replace_all("NAME", "[JOBTITLE 5]")
    assert doc.lines[0].text == "[NAME 1] met [NAME 2] at the [JOBTITLE 5] conference"


def test_split_oversized_single_cell():
    """A single cell > max_chars in a multi-cell row should get sentence-split."""
    long_notes = "First sentence of notes. " * 30  # ~750 chars
    cells = [Cell(text="Jane"), Cell(text="Smith"), Cell(text=long_notes.strip())]
    line = Line(cells=cells, line_number=1)
    doc = Document(lines=[line], source_path=Path("test.csv"), source_format="csv")
    split_long_lines(doc, max_chars=200)
    # The long notes cell should be split into multiple sub-lines
    assert len(doc.lines) > 1
    # Short cells should be in their own chunk
    assert "Jane" in doc.lines[0].text
