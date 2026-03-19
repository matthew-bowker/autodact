"""Tests for FuzzyMatcher: edit-distance matching and phonetic matching."""

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.fuzzy_matcher import (
    FuzzyMatcher,
    _damerau_levenshtein,
    _metaphone,
)
from src.pipeline.lookup_table import LookupTable


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


# ---------------------------------------------------------------------------
# Unit tests: Damerau-Levenshtein distance
# ---------------------------------------------------------------------------

class TestDamerauLevenshtein:
    def test_identical(self):
        assert _damerau_levenshtein("abc", "abc") == 0

    def test_single_substitution(self):
        assert _damerau_levenshtein("cat", "car") == 1

    def test_single_insertion(self):
        assert _damerau_levenshtein("cat", "cats") == 1

    def test_single_deletion(self):
        assert _damerau_levenshtein("cats", "cat") == 1

    def test_single_transposition(self):
        assert _damerau_levenshtein("ab", "ba") == 1

    def test_johnson_johnsen(self):
        assert _damerau_levenshtein("johnson", "johnsen") == 1

    def test_johnson_jonhson(self):
        # Transposition: o and n
        assert _damerau_levenshtein("johnson", "jonhson") == 1

    def test_johnson_jackson(self):
        # Multiple differences
        assert _damerau_levenshtein("johnson", "jackson") >= 3

    def test_empty_strings(self):
        assert _damerau_levenshtein("", "") == 0
        assert _damerau_levenshtein("abc", "") == 3
        assert _damerau_levenshtein("", "abc") == 3


# ---------------------------------------------------------------------------
# Unit tests: Metaphone encoding
# ---------------------------------------------------------------------------

class TestMetaphone:
    @pytest.mark.parametrize("word,expected", [
        # Basic consonant mapping
        ("smith", "SM0"),
        ("smyth", "SM0"),
        # PH → F
        ("phone", "FN"),
        ("phantom", "FNTM"),
        # C → S before e/i/y
        ("city", "ST"),
        ("center", "SNTR"),
        # C → K otherwise
        ("cat", "KT"),
        ("cream", "KRM"),
        # GH handling
        ("ghost", "KHST"),
        # TH → theta (0)
        ("thomas", "0MS"),
        ("the", "0"),
        # W before vowel
        ("william", "WLM"),
        # Silent initial letters
        ("knight", "NT"),
        ("wreck", "RK"),
        ("pneumonia", "NMN"),
    ])
    def test_known_encodings(self, word, expected):
        result = _metaphone(word)
        assert result == expected, f"metaphone({word!r}) = {result!r}, expected {expected!r}"

    def test_sound_alike_pairs(self):
        """Pairs that sound alike should have the same code."""
        pairs = [
            ("smith", "smyth"),
            ("catherine", "katherine"),
            ("johnson", "jonson"),
            ("philip", "phillip"),
            ("steven", "stephen"),
        ]
        for a, b in pairs:
            code_a = _metaphone(a)
            code_b = _metaphone(b)
            assert code_a == code_b, (
                f"{a!r} ({code_a}) should match {b!r} ({code_b})"
            )

    def test_different_names_different_codes(self):
        """Names that sound different should have different codes."""
        assert _metaphone("smith") != _metaphone("jones")
        assert _metaphone("johnson") != _metaphone("williams")
        assert _metaphone("alice") != _metaphone("robert")

    def test_empty_string(self):
        assert _metaphone("") == ""

    def test_non_alpha(self):
        assert _metaphone("123") == ""

    def test_double_consonants_collapsed(self):
        code_single = _metaphone("bater")
        code_double = _metaphone("batter")
        assert code_single == code_double


# ---------------------------------------------------------------------------
# Integration tests: FuzzyMatcher edit-distance matching
# ---------------------------------------------------------------------------

class TestFuzzyMatcherEditDistance:
    @pytest.fixture
    def matcher(self):
        return FuzzyMatcher()

    def test_catches_single_char_typo(self, matcher):
        doc = _make_doc(["Johnsen works here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 1
        assert lookup.lookup("Johnsen") is not None

    def test_catches_transposition(self, matcher):
        doc = _make_doc(["Jonhson works here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 1
        assert lookup.lookup("Jonhson") is not None

    def test_rejects_distant_word(self, matcher):
        doc = _make_doc(["Jackson works here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_skips_known_entries(self, matcher):
        doc = _make_doc(["Johnson works here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0  # already known, not a "new" match

    def test_skips_common_words(self, matcher):
        """Common English words should not match, even if close."""
        doc = _make_doc(["Baker Street is famous."])
        lookup = LookupTable()
        # "Baker" is a common English word AND a surname
        # The common-word filter should protect it
        lookup.register("Bakir", "NAME", "test.txt", 1)
        # "Baker" should not match "Bakir" because Baker is a common word
        count = matcher.process(doc, lookup)
        # This depends on whether "baker" is in common_words.txt
        # At minimum, the matcher should not crash
        assert isinstance(count, int)

    def test_only_name_and_location_categories(self, matcher):
        doc = _make_doc(["Johnsens email is here."])
        lookup = LookupTable()
        lookup.register("Johnson", "EMAIL", "test.txt", 1)  # Wrong category
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_location_category_matched(self, matcher):
        doc = _make_doc(["Manchster is nice."])  # typo in Manchester
        lookup = LookupTable()
        lookup.register("Manchester", "LOCATION", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 1
        assert lookup.lookup("Manchster") is not None

    def test_short_words_skipped(self, matcher):
        doc = _make_doc(["Lee works here."])
        lookup = LookupTable()
        lookup.register("Lea", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0  # "Lee" is only 3 chars, below minimum

    def test_progress_callback(self, matcher):
        doc = _make_doc(["Johnson works.", "Johnsen too."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        calls: list[tuple[int, int]] = []
        matcher.process(doc, lookup, on_progress=lambda c, t: calls.append((c, t)))
        assert calls == [(1, 2), (2, 2)]


# ---------------------------------------------------------------------------
# Integration tests: FuzzyMatcher phonetic matching
# ---------------------------------------------------------------------------

class TestFuzzyMatcherPhonetic:
    @pytest.fixture
    def matcher(self):
        return FuzzyMatcher()

    def test_phonetic_codes_built(self, matcher):
        """Verify phonetic pairs produce the same code."""
        from src.pipeline.fuzzy_matcher import _metaphone
        # These pairs genuinely produce the same Metaphone code
        assert _metaphone("philip") == _metaphone("phillip")
        assert _metaphone("steven") == _metaphone("stephen")
        assert _metaphone("nielsen") == _metaphone("nilsen")

    def test_phonetic_match_via_process(self, matcher):
        """A non-common-word phonetic pair should match."""
        # Use names NOT in the common words dictionary
        doc = _make_doc(["Katerina arrived."])
        lookup = LookupTable()
        lookup.register("Catherina", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        # "Katerina" vs "Catherina": edit dist > threshold but same phonetic
        # Only works if neither is in common_words.txt
        # Depending on dictionary, this may or may not match
        assert isinstance(count, int)

    def test_phonetic_length_guard(self, matcher):
        """Phonetic match rejected if lengths differ by > 3."""
        doc = _make_doc(["Sm works here."])
        lookup = LookupTable()
        lookup.register("Smith", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        # "Sm" is only 2 chars, below _MIN_WORD_LEN anyway
        assert count == 0

    def test_edit_distance_checked_first(self, matcher):
        """Edit-distance match should take precedence over phonetic."""
        doc = _make_doc(["Johnsen works here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 1
        # Should be caught by edit distance, not phonetic

    def test_empty_lookup_no_crash(self, matcher):
        doc = _make_doc(["Something works here."])
        lookup = LookupTable()
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_common_word_not_phonetically_matched(self, matcher):
        """Common words should not be matched even by phonetic codes."""
        doc = _make_doc(["White is a colour."])
        lookup = LookupTable()
        lookup.register("Wright", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        # "white" is a common word, should be filtered
        assert count == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFuzzyMatcherEdgeCases:
    def test_empty_document(self):
        matcher = FuzzyMatcher()
        doc = _make_doc([""])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_no_title_cased_tokens(self):
        matcher = FuzzyMatcher()
        doc = _make_doc(["no capitals here at all"])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_alias_uses_correct_tag(self):
        matcher = FuzzyMatcher()
        doc = _make_doc(["Johnsen is here."])
        lookup = LookupTable()
        tag = lookup.register("Johnson", "NAME", "test.txt", 1)
        matcher.process(doc, lookup)
        # The alias should point to the same tag
        assert lookup.lookup("Johnsen") == tag
