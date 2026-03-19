"""Tests for entropy-based detection of API keys, tokens, and hashes."""

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.entropy_detector import (
    EntropyDetector,
    _has_mixed_chars,
    _shannon_entropy,
)
from src.pipeline.lookup_table import LookupTable


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_empty_string(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char_repeated(self):
        assert _shannon_entropy("aaaaaaa") == 0.0

    def test_two_chars_equal_frequency(self):
        e = _shannon_entropy("ab")
        assert abs(e - 1.0) < 0.01  # exactly 1 bit

    def test_high_entropy_random(self):
        # Random-looking string should have high entropy
        e = _shannon_entropy("sk_live_abc123XYZ789")
        assert e > 3.5

    def test_low_entropy_repeated_pattern(self):
        e = _shannon_entropy("aaaa1111")
        assert e < 2.0

    def test_normal_word(self):
        # Normal English words have moderate entropy
        e = _shannon_entropy("Birmingham")
        assert e < 4.0  # words are somewhat structured


class TestHasMixedChars:
    def test_upper_lower_digits(self):
        assert _has_mixed_chars("aBc123") is True

    def test_upper_and_lower(self):
        assert _has_mixed_chars("AbCdEf") is True

    def test_lower_and_digits(self):
        assert _has_mixed_chars("abc123") is True

    def test_upper_and_digits(self):
        assert _has_mixed_chars("ABC123") is True

    def test_only_lowercase(self):
        assert _has_mixed_chars("abcdefgh") is False

    def test_only_uppercase(self):
        assert _has_mixed_chars("ABCDEFGH") is False

    def test_only_digits(self):
        assert _has_mixed_chars("12345678") is False


# ---------------------------------------------------------------------------
# Integration tests for EntropyDetector
# ---------------------------------------------------------------------------

class TestEntropyDetectorFlags:
    """Strings that SHOULD be flagged as ID."""

    @pytest.mark.parametrize("token", [
        "sk_live_abc123XYZ789",           # API key style
        "eyJhbGciOiJIUzI1NiJ9",           # JWT fragment
        "REF2024_ABX9K001",               # Reference ID
        "a3f2b8c9d1e4f5a6",               # Hex hash with digits
        "Token_Abc123Xyz456",             # Token with mixed case + digits
        "xK9mP2qL5rT8wN3v",              # Random alphanumeric
    ])
    def test_high_entropy_token_flagged(self, token):
        doc = _make_doc([f"Value: {token}"])
        lookup = LookupTable()
        det = EntropyDetector()
        count = det.process(doc, lookup)
        assert count >= 1, f"Expected {token} to be flagged"
        assert lookup.lookup(token) is not None
        entry = [e for e in lookup.all_entries() if e.original_term == token][0]
        assert entry.pii_category == "ID"


class TestEntropyDetectorSkips:
    """Strings that should NOT be flagged."""

    @pytest.mark.parametrize("token", [
        "Birmingham",          # Pure alpha — normal word
        "enterprise",          # Pure alpha — common word
        "JavaScript",          # Pure alpha (camelCase)
        "methodology",         # Pure alpha — long word
    ])
    def test_pure_alpha_skipped(self, token):
        doc = _make_doc([f"The {token} is important."])
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert lookup.lookup(token) is None, f"{token} should not be flagged"

    @pytest.mark.parametrize("token", [
        "12345678",            # Pure digit
        "987654321012",        # Pure digit
    ])
    def test_pure_digit_skipped(self, token):
        doc = _make_doc([f"Number: {token}"])
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert lookup.lookup(token) is None

    @pytest.mark.parametrize("token", [
        "date_of_birth",       # snake_case
        "first_name",          # snake_case
        "background-color",    # kebab-case
        "user-agent-string",   # kebab-case
    ])
    def test_snake_kebab_case_skipped(self, token):
        doc = _make_doc([f"Field: {token}"])
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert lookup.lookup(token) is None, f"{token} should not be flagged"

    def test_short_token_skipped(self):
        doc = _make_doc(["abc1234"])  # 7 chars, below minimum
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert len(lookup) == 0

    def test_low_entropy_skipped(self):
        doc = _make_doc(["aaaa1111bbbb"])  # Repetitive, low entropy
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert len(lookup) == 0

    def test_existing_tag_skipped(self):
        doc = _make_doc(["[NAME 1] has token abc123XY"])
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        # Should not flag [NAME 1] as high entropy
        assert lookup.lookup("[NAME 1]") is None

    def test_already_registered_skipped(self):
        doc = _make_doc(["Token sk_live_abc123XYZ appears twice: sk_live_abc123XYZ"])
        lookup = LookupTable()
        det = EntropyDetector()
        count = det.process(doc, lookup)
        assert count == 1  # Only counted once


class TestEntropyDetectorEdgeCases:
    def test_empty_document(self):
        doc = _make_doc([""])
        lookup = LookupTable()
        det = EntropyDetector()
        count = det.process(doc, lookup)
        assert count == 0

    def test_no_candidates(self):
        doc = _make_doc(["Just a normal sentence with nothing special."])
        lookup = LookupTable()
        det = EntropyDetector()
        count = det.process(doc, lookup)
        assert count == 0

    def test_custom_thresholds(self):
        # With very high entropy threshold, fewer things should match
        doc = _make_doc(["Token: abc12345XY"])
        lookup_strict = LookupTable()
        det_strict = EntropyDetector(min_entropy=5.0)
        det_strict.process(doc, lookup_strict)

        lookup_lenient = LookupTable()
        det_lenient = EntropyDetector(min_entropy=2.0)
        det_lenient.process(doc, lookup_lenient)

        assert len(lookup_lenient) >= len(lookup_strict)

    def test_multiple_tokens_per_line(self):
        doc = _make_doc(["Keys: sk_live_abc123XYZ and pk_test_789defGHI"])
        lookup = LookupTable()
        det = EntropyDetector()
        count = det.process(doc, lookup)
        assert count == 2

    def test_replaces_in_document(self):
        doc = _make_doc(["My key is sk_live_abc123XYZ789 and it works."])
        lookup = LookupTable()
        det = EntropyDetector()
        det.process(doc, lookup)
        assert "sk_live_abc123XYZ789" not in doc.lines[0].text
        assert "[ID" in doc.lines[0].text
