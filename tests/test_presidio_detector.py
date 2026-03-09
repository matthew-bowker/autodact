"""Tests for PresidioDetector.

These tests use a real AnalyzerEngine (with spaCy) so they exercise the
full Presidio stack.  They are deliberately lightweight so the test suite
stays fast — we only check that the integration wiring works, not that
spaCy NER is perfect.
"""

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable

# Guard: skip the whole module if spaCy model is not installed.
from src.pipeline.presidio_detector import (
    _is_plausible_date,
    _is_plausible_phone,
)

# Guard: skip spaCy-dependent tests if spaCy model is not installed.
try:
    from src.pipeline.presidio_detector import PresidioDetector, _build_analyzer

    _analyzer = _build_analyzer("en_core_web_lg")
    _HAS_SPACY = True
except (ImportError, OSError):
    _HAS_SPACY = False

pytestmark = pytest.mark.skipif(not _HAS_SPACY, reason="spaCy model not installed")


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


def test_detects_email():
    doc = _make_doc(["Contact us at jane@example.com for info."])
    lookup = LookupTable()
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    assert lookup.lookup("jane@example.com") is not None


def test_detects_phone():
    doc = _make_doc(["Call +1-202-555-0173 for details."])
    lookup = LookupTable()
    # Phones use a lenient threshold (0.3) automatically — no need to override.
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    assert len(lookup) >= 1


def test_detects_person_name():
    doc = _make_doc(["John Smith is the project lead."])
    lookup = LookupTable()
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    # spaCy should detect "John Smith" as PERSON → NAME
    names = [e for e in lookup.all_entries() if e.pii_category == "NAME"]
    assert len(names) >= 1


def test_respects_enabled_categories():
    doc = _make_doc(["John Smith emailed john@example.com."])
    lookup = LookupTable()
    # Only enable EMAIL
    detector = PresidioDetector(analyzer=_analyzer, enabled_categories={"EMAIL"})
    detector.process(doc, lookup)
    # EMAIL should be detected, NAME should not
    cats = {e.pii_category for e in lookup.all_entries()}
    assert "EMAIL" in cats
    assert "NAME" not in cats


def test_skips_existing_tags():
    doc = _make_doc(["[NAME 1] emailed info@example.com."])
    lookup = LookupTable()
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    # "[NAME 1]" should not be re-registered as a detection
    assert lookup.lookup("[NAME 1]") is None


def test_progress_callback():
    doc = _make_doc(["Line 1.", "Line 2."])
    lookup = LookupTable()
    calls: list[tuple[int, int]] = []
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup, on_progress=lambda c, t: calls.append((c, t)))
    assert calls == [(1, 2), (2, 2)]


def test_email_not_corrupted_by_url_overlap():
    """URL entities inside emails (e.g. acme.com) must not be registered."""
    doc = _make_doc([
        "Contact jane@acme.com or bob@acme.com for help.",
    ])
    lookup = LookupTable()
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    # Both emails should be tagged cleanly — no stray URL tags.
    text = doc.lines[0].text
    assert "acme.com" not in text  # domain should be inside an [EMAIL] tag
    assert "[URL" not in text  # no URL tags should appear


def test_overlap_removes_url_inside_email_across_lines():
    """URL detected on line 1 must not corrupt emails on line 2."""
    doc = _make_doc([
        "Email jane@example.com for the report.",
        "Also email bob@example.com please.",
    ])
    lookup = LookupTable()
    detector = PresidioDetector(analyzer=_analyzer)
    detector.process(doc, lookup)
    for line in doc.lines:
        assert "[URL" not in line.text
        assert "example.com" not in line.text


# ── Date validation (no spaCy needed) ──────────────────────────────


class TestIsPlausibleDate:
    def test_valid_dmy(self):
        assert _is_plausible_date("31/12/2025") is True

    def test_valid_mdy(self):
        assert _is_plausible_date("12/31/2025") is True

    def test_valid_two_digit_year(self):
        assert _is_plausible_date("01-15-99") is True

    def test_short_ambiguous_valid(self):
        # 2/4/16 — could be Feb 4 or Apr 2, 2-digit year
        assert _is_plausible_date("2/4/16") is True

    def test_reject_invalid_month_and_day(self):
        assert _is_plausible_date("99/99/2024") is False

    def test_reject_month_too_high(self):
        # 13/40/2025 — neither interpretation yields month ≤12
        assert _is_plausible_date("13/40/2025") is False

    def test_reject_year_out_of_range(self):
        assert _is_plausible_date("1/1/1800") is False
        assert _is_plausible_date("1/1/2100") is False

    def test_reject_three_digit_year(self):
        assert _is_plausible_date("1/2/300") is False

    def test_reject_tiny_version_string(self):
        # 1/2/3 — year "3" is single-digit (2-digit years are OK, 1-digit not
        # matched by {2,4} in the original regex, but validate anyway)
        assert _is_plausible_date("1/2/3") is False


# ── Phone validation (no spaCy needed) ─────────────────────────────


class TestIsPlausiblePhone:
    def test_valid_international(self):
        assert _is_plausible_phone("+1-202-555-0173") is True

    def test_valid_uk(self):
        assert _is_plausible_phone("+44 7700 900123") is True

    def test_valid_local(self):
        assert _is_plausible_phone("(555) 123-4567") is True

    def test_reject_too_few_digits(self):
        assert _is_plausible_phone("123456") is False

    def test_reject_too_many_digits(self):
        assert _is_plausible_phone("1234567890123456") is False

    def test_reject_version_number_dots(self):
        assert _is_plausible_phone("1.2.3.4.5") is False

    def test_reject_semver(self):
        assert _is_plausible_phone("10.23.456") is False
