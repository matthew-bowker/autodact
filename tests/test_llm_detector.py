from pathlib import Path

from src.pipeline.document import Cell, Document, Line
from src.pipeline.llm_detector import LLMDetector
from src.pipeline.lookup_table import LookupTable


class MockEngine:
    """Mock LLM engine that returns predetermined responses."""

    def __init__(self, responses: list[list[dict[str, str]]] | None = None):
        self._responses = responses or []
        self._call_count = 0
        self.prompts: list[str] = []

    def detect_pii(self, user_prompt: str, categories: list[str]) -> list[dict[str, str]]:
        self.prompts.append(user_prompt)
        if self._call_count < len(self._responses):
            result = self._responses[self._call_count]
            self._call_count += 1
            return result
        self._call_count += 1
        return []


class FailingEngine:
    """Mock engine that always raises."""

    def detect_pii(self, user_prompt: str, categories: list[str]) -> list[dict[str, str]]:
        raise RuntimeError("LLM inference failed")


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


def test_basic_detection():
    engine = MockEngine(responses=[
        [{"original": "Jane Smith", "category": "NAME"}],
    ])
    doc = _make_doc(["Jane Smith works at Acme Corp."])
    lookup = LookupTable()
    detector = LLMDetector(engine)
    flagged = detector.process(doc, lookup)
    assert flagged == []
    assert lookup.lookup("Jane Smith") == "[NAME 1]"
    # The LLM detector registers findings but does not mutate the document;
    # replacement is handled by the orchestrator's additive replay.
    assert doc.lines[0].text == "Jane Smith works at Acme Corp."


def test_multiple_detections_per_line():
    engine = MockEngine(responses=[
        [
            {"original": "Jane Smith", "category": "NAME"},
            {"original": "Manchester", "category": "LOCATION"},
        ],
    ])
    doc = _make_doc(["Jane Smith lives in Manchester."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert lookup.lookup("Jane Smith") is not None
    assert lookup.lookup("Manchester") is not None


def test_register_across_document():
    engine = MockEngine(responses=[
        [{"original": "Jane Smith", "category": "NAME"}],
        [],  # Second line has no new PII
    ])
    doc = _make_doc([
        "Jane Smith is a researcher.",
        "We spoke with Jane Smith yesterday.",
    ])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    # Detector registers but does not mutate; document is unchanged.
    assert lookup.lookup("Jane Smith") == "[NAME 1]"
    assert doc.lines[1].text == "We spoke with Jane Smith yesterday."


def test_skip_empty_lines():
    engine = MockEngine(responses=[
        [{"original": "Jane", "category": "NAME"}],
        [],
    ])
    doc = _make_doc(["Jane is here.", "", "Another line."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    # Engine should only be called for non-empty lines
    assert len(engine.prompts) == 2


def test_hallucination_filtered():
    engine = MockEngine(responses=[
        [{"original": "HALLUCINATED_TEXT", "category": "NAME"}],
    ])
    doc = _make_doc(["Jane Smith works here."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert lookup.lookup("HALLUCINATED_TEXT") is None
    assert len(lookup) == 0


def test_invalid_category_filtered():
    engine = MockEngine(responses=[
        [{"original": "Jane", "category": "INVALID_CAT"}],
    ])
    doc = _make_doc(["Jane is here."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(lookup) == 0


def test_retry_on_failure():
    call_count = 0
    original_responses = [
        [{"original": "Jane", "category": "NAME"}],
    ]

    class RetryEngine:
        def detect_pii(self, user_prompt: str, categories: list[str]) -> list[dict[str, str]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Temporary failure")
            return original_responses[0]

    doc = _make_doc(["Jane is here."])
    lookup = LookupTable()
    flagged = LLMDetector(RetryEngine()).process(doc, lookup)
    assert flagged == []
    assert lookup.lookup("Jane") == "[NAME 1]"
    assert call_count == 2


def test_flag_line_on_double_failure():
    doc = _make_doc(["Jane is here."])
    lookup = LookupTable()
    flagged = LLMDetector(FailingEngine()).process(doc, lookup)
    assert flagged == [1]
    assert len(lookup) == 0


def test_progress_callback():
    engine = MockEngine(responses=[[], []])
    doc = _make_doc(["Line 1", "Line 2"])
    lookup = LookupTable()
    progress_calls = []
    LLMDetector(engine).process(
        doc, lookup, on_progress=lambda c, t: progress_calls.append((c, t))
    )
    assert progress_calls == [(1, 2), (2, 2)]


def test_deduplication_across_lines():
    engine = MockEngine(responses=[
        [{"original": "Jane", "category": "NAME"}],
        [{"original": "Jane", "category": "NAME"}],
    ])
    doc = _make_doc(["Jane is here.", "Jane is there."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    # Only one lookup entry since "Jane" is deduplicated
    assert len([e for e in lookup.all_entries() if e.pii_category == "NAME"]) == 1


def test_category_filtering():
    """Detections outside enabled categories should be filtered."""
    engine = MockEngine(responses=[
        [
            {"original": "Jane", "category": "NAME"},
            {"original": "CEO", "category": "JOBTITLE"},
        ],
    ])
    doc = _make_doc(["Jane is CEO here."])
    lookup = LookupTable()
    # Only enable NAME, not JOBTITLE
    LLMDetector(engine, categories=["NAME"]).process(doc, lookup)
    assert lookup.lookup("Jane") is not None
    assert lookup.lookup("CEO") is None


def _make_csv_doc(rows: list[list[str]]) -> Document:
    """Create a Document with multi-cell lines (simulating CSV data)."""
    lines = [
        Line(
            cells=[Cell(text=v, col_index=ci) for ci, v in enumerate(row)],
            line_number=i + 1,
        )
        for i, row in enumerate(rows)
    ]
    return Document(lines=lines, source_path=Path("test.csv"), source_format="csv")


def test_multi_cell_entity_detection():
    """Entity spanning multiple cells (e.g. 'Jane Smith' in CSV) should be detected."""
    engine = MockEngine(responses=[
        [{"original": "Jane Smith", "category": "NAME"}],
    ])
    doc = _make_csv_doc([["Jane", "Smith", "some text"]])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    # Entity should be registered even though it spans cells
    assert lookup.lookup("Jane Smith") == "[NAME 1]"
    # Individual words should be registered as aliases for additive replay
    assert lookup.lookup("Jane") == "[NAME 1]"
    assert lookup.lookup("Smith") == "[NAME 1]"
    # Document is not mutated by the detector
    assert doc.lines[0].cells[0].text == "Jane"
    assert doc.lines[0].cells[1].text == "Smith"


def test_multi_cell_entity_registers_across_document():
    """Multi-cell entity and its word aliases are registered for replay."""
    engine = MockEngine(responses=[
        [{"original": "Jane Smith", "category": "NAME"}],
        [],
    ])
    doc = _make_csv_doc([
        ["Jane", "Smith", "works here"],
        ["Contact", "Jane Smith", "for details"],
    ])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert lookup.lookup("Jane Smith") == "[NAME 1]"
    assert lookup.lookup("Jane") == "[NAME 1]"
    assert lookup.lookup("Smith") == "[NAME 1]"
    # Document is not mutated — orchestrator handles replacement
    assert doc.lines[1].cells[1].text == "Jane Smith"


def test_prompt_uses_distil_pii_format():
    """Prompts sent to the engine should use Distil-PII template."""
    engine = MockEngine(responses=[[]])
    doc = _make_doc(["Jane Smith works here."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert "<context>" in engine.prompts[0]
    assert "Jane Smith works here." in engine.prompts[0]
    assert "<question>" in engine.prompts[0]


def test_csv_prompt_includes_headers():
    """CSV data should have column headers prefixed in the LLM prompt."""
    engine = MockEngine(responses=[[]])
    doc = _make_csv_doc([["Jane", "Smith", "some notes"]])
    doc.headers = ["First Name", "Last Name", "Notes"]
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    prompt = engine.prompts[0]
    assert "First Name: Jane" in prompt
    assert "Last Name: Smith" in prompt
    assert "Notes: some notes" in prompt


def test_skip_fully_mapped_row():
    """Rows where every non-empty cell is in a mapped column are skipped."""
    engine = MockEngine(responses=[])  # Engine should never be called
    doc = _make_csv_doc([["Jane Smith", "jane@example.com", "555-1234"]])
    lookup = LookupTable()
    # Columns 0, 1, 2 all mapped to PII types
    mapped = {0: "NAME", 1: "EMAIL", 2: "PHONE"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 0


def test_skip_mapped_row_with_empty_cells():
    """Empty cells don't prevent a fully-mapped row from being skipped."""
    engine = MockEngine(responses=[])
    doc = _make_csv_doc([["Jane Smith", "", ""]])
    lookup = LookupTable()
    mapped = {0: "NAME", 1: "EMAIL", 2: "PHONE"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 0


def test_not_skipped_when_unmapped_column_has_content():
    """Rows with content in unmapped columns must still go to the LLM."""
    engine = MockEngine(responses=[
        [{"original": "42 Oak Road", "category": "LOCATION"}],
    ])
    doc = _make_csv_doc([["Jane Smith", "42 Oak Road"]])
    lookup = LookupTable()
    # Only column 0 mapped — column 1 is unmapped
    mapped = {0: "NAME"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 1
    assert lookup.lookup("42 Oak Road") == "[LOCATION 1]"


def test_freetext_column_not_treated_as_covered():
    """Columns mapped to 'freetext' still need LLM review."""
    engine = MockEngine(responses=[
        [{"original": "42 Oak Road", "category": "LOCATION"}],
    ])
    doc = _make_csv_doc([["Jane Smith", "Lives at 42 Oak Road"]])
    lookup = LookupTable()
    mapped = {0: "NAME", 1: "freetext"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 1


def test_prompt_omits_mapped_columns():
    """The LLM prompt should only contain unmapped cell content."""
    engine = MockEngine(responses=[[]])
    doc = _make_csv_doc([["Jane Smith", "jane@example.com", "some notes"]])
    doc.headers = ["Name", "Email", "Notes"]
    lookup = LookupTable()
    mapped = {0: "NAME", 1: "EMAIL"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    prompt = engine.prompts[0]
    # Mapped columns should be excluded from the prompt
    assert "Jane Smith" not in prompt
    assert "jane@example.com" not in prompt
    # Unmapped column should be present with its header
    assert "Notes: some notes" in prompt


def test_no_mapping_sends_full_line():
    """Without a column mapping, every line goes to the LLM unchanged."""
    engine = MockEngine(responses=[[]])
    doc = _make_doc(["Jane Smith works at Acme Corp."])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(engine.prompts) == 1
    assert "Jane Smith works at Acme Corp." in engine.prompts[0]


def test_txt_prompt_no_headers():
    """Plain text lines should not get header prefixes even if headers exist."""
    engine = MockEngine(responses=[[]])
    doc = _make_doc(["Jane Smith works here."])
    doc.headers = ["Name"]  # shouldn't be used (single-cell lines)
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    # Should use plain text, not "Name: Jane Smith works here."
    assert "Jane Smith works here." in engine.prompts[0]
    assert "Name:" not in engine.prompts[0]


# ------------------------------------------------------------------
# Trivial content short-circuit
# ------------------------------------------------------------------

def test_skip_trivial_placeholder():
    """Cells with placeholder values like 'N/A' should skip the LLM."""
    engine = MockEngine(responses=[])
    doc = _make_csv_doc([["N/A", "pending"]])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(engine.prompts) == 0


def test_skip_trivial_numeric():
    """Purely numeric cells should skip the LLM."""
    engine = MockEngine(responses=[])
    doc = _make_csv_doc([["12345", "42.5"]])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(engine.prompts) == 0


def test_skip_trivial_single_char():
    """Single-character cells should skip the LLM."""
    engine = MockEngine(responses=[])
    doc = _make_csv_doc([["Y", "-"]])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(engine.prompts) == 0


def test_not_skipped_when_one_cell_nontrivial():
    """Row with a mix of trivial and real content must go to the LLM."""
    engine = MockEngine(responses=[[]])
    doc = _make_csv_doc([["N/A", "Lives at 42 Oak Road"]])
    lookup = LookupTable()
    LLMDetector(engine).process(doc, lookup)
    assert len(engine.prompts) == 1


def test_trivial_check_with_mapped_columns():
    """Trivial check applies only to uncovered cells."""
    engine = MockEngine(responses=[])
    doc = _make_csv_doc([["Jane Smith", "N/A"]])
    lookup = LookupTable()
    # Column 0 is mapped (covered), column 1 is unmapped but trivial
    mapped = {0: "NAME"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 0


# ------------------------------------------------------------------
# Result caching
# ------------------------------------------------------------------

def test_cache_identical_freetext():
    """Rows with the same unmapped content should reuse the LLM result."""
    engine = MockEngine(responses=[
        [{"original": "42 Oak Road", "category": "LOCATION"}],
    ])
    doc = _make_csv_doc([
        ["Jane Smith", "42 Oak Road"],
        ["Bob Jones", "42 Oak Road"],
    ])
    doc.headers = ["Name", "Address"]
    lookup = LookupTable()
    mapped = {0: "NAME"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    # Engine called only once — second row reuses the cached result
    assert len(engine.prompts) == 1
    assert lookup.lookup("42 Oak Road") == "[LOCATION 1]"


def test_cache_does_not_cross_different_content():
    """Different freetext content should produce separate LLM calls."""
    engine = MockEngine(responses=[
        [{"original": "42 Oak Road", "category": "LOCATION"}],
        [{"original": "7 Elm Street", "category": "LOCATION"}],
    ])
    doc = _make_csv_doc([
        ["Jane Smith", "42 Oak Road"],
        ["Bob Jones", "7 Elm Street"],
    ])
    doc.headers = ["Name", "Address"]
    lookup = LookupTable()
    mapped = {0: "NAME"}
    LLMDetector(engine).process(doc, lookup, mapped_columns=mapped)
    assert len(engine.prompts) == 2
    assert lookup.lookup("42 Oak Road") == "[LOCATION 1]"
    assert lookup.lookup("7 Elm Street") == "[LOCATION 2]"
