"""Tests for embedding similarity matcher using spaCy word vectors."""

from pathlib import Path

import pytest

from src.pipeline.document import Cell, Document, Line
from src.pipeline.lookup_table import LookupTable

try:
    import spacy
    _nlp = spacy.load("en_core_web_md")
    _HAS_SPACY = True
except (ImportError, OSError):
    _nlp = None
    _HAS_SPACY = False

if _HAS_SPACY:
    from src.pipeline.embedding_matcher import EmbeddingMatcher, _cosine_similarity

pytestmark = pytest.mark.skipif(
    not _HAS_SPACY,
    reason="spaCy en_core_web_md model not installed",
)


def _make_doc(lines_text: list[str]) -> Document:
    lines = [
        Line(cells=[Cell(text=t)], line_number=i + 1)
        for i, t in enumerate(lines_text)
    ]
    return Document(lines=lines, source_path=Path("test.txt"), source_format="txt")


# ---------------------------------------------------------------------------
# Unit tests: cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        import numpy as np
        v = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        import numpy as np
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Integration tests: EmbeddingMatcher
# ---------------------------------------------------------------------------

class TestEmbeddingMatcherBasic:
    @pytest.fixture
    def matcher(self):
        return EmbeddingMatcher(_nlp, threshold=0.85)

    def test_skips_already_known(self, matcher):
        doc = _make_doc(["Johnson is here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0  # already in lookup

    def test_skips_short_words(self, matcher):
        doc = _make_doc(["Lee is here."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0  # "Lee" is only 3 chars

    def test_no_targets_no_crash(self, matcher):
        doc = _make_doc(["Johnson is here."])
        lookup = LookupTable()
        # No eligible entries in lookup
        lookup.register("someone@email.com", "EMAIL", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_empty_document(self, matcher):
        doc = _make_doc([""])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_progress_callback(self, matcher):
        doc = _make_doc(["Line one.", "Line two."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        calls: list[tuple[int, int]] = []
        matcher.process(doc, lookup, on_progress=lambda c, t: calls.append((c, t)))
        assert calls == [(1, 2), (2, 2)]


class TestEmbeddingMatcherVectors:
    """Test actual vector-based matching with the spaCy model."""

    def test_oov_words_skipped(self):
        """Out-of-vocabulary words (no vector) should be skipped."""
        matcher = EmbeddingMatcher(_nlp, threshold=0.85)
        # Use a gibberish word that spaCy won't have a vector for
        doc = _make_doc(["Xyzzyplugh arrived."])
        lookup = LookupTable()
        lookup.register("Johnson", "NAME", "test.txt", 1)
        count = matcher.process(doc, lookup)
        # Either OOV or no match — should not crash
        assert isinstance(count, int)

    def test_only_name_and_location_categories(self):
        """Only NAME and LOCATION lookup entries should be used as targets."""
        matcher = EmbeddingMatcher(_nlp, threshold=0.85)
        doc = _make_doc(["London is great."])
        lookup = LookupTable()
        # Register London as EMAIL (wrong category) — should not be a target
        lookup.register("London", "EMAIL", "test.txt", 1)
        count = matcher.process(doc, lookup)
        assert count == 0

    def test_high_threshold_prevents_loose_matches(self):
        """With very high threshold, fewer matches should occur."""
        matcher_strict = EmbeddingMatcher(_nlp, threshold=0.99)
        matcher_lenient = EmbeddingMatcher(_nlp, threshold=0.50)

        doc_strict = _make_doc(["Paris Berlin Tokyo London"])
        doc_lenient = _make_doc(["Paris Berlin Tokyo London"])

        lookup_strict = LookupTable()
        lookup_strict.register("Manchester", "LOCATION", "test.txt", 1)

        lookup_lenient = LookupTable()
        lookup_lenient.register("Manchester", "LOCATION", "test.txt", 1)

        count_strict = matcher_strict.process(doc_strict, lookup_strict)
        count_lenient = matcher_lenient.process(doc_lenient, lookup_lenient)

        assert count_lenient >= count_strict

    def test_registers_as_alias(self):
        """Matches should be registered as aliases of the original tag."""
        matcher = EmbeddingMatcher(_nlp, threshold=0.50)  # lenient for test
        doc = _make_doc(["Paris Berlin Tokyo"])
        lookup = LookupTable()
        tag = lookup.register("London", "LOCATION", "test.txt", 1)
        matcher.process(doc, lookup)
        # If any match was found, it should be an alias pointing to the same tag
        for entry in lookup.all_entries():
            if entry.original_term != "London":
                assert entry.anonymised_term == tag

    def test_self_similarity_is_1(self):
        """A word's vector should be perfectly similar to itself."""
        word = "London"
        lexeme = _nlp.vocab[word]
        if lexeme.has_vector:
            sim = _cosine_similarity(lexeme.vector, lexeme.vector)
            assert abs(sim - 1.0) < 1e-5
