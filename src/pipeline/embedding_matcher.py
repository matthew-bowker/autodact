"""Find semantic relatives of known PII using spaCy word vectors."""

from __future__ import annotations

import logging
import math
import re
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import _load_name_set

logger = logging.getLogger(__name__)

# Matches title-cased word tokens.
_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b")

# Categories eligible for embedding matching.
_EMBED_CATEGORIES: frozenset[str] = frozenset({"NAME", "LOCATION"})

_MIN_WORD_LEN = 4


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors (numpy arrays)."""
    dot = float(a @ b)
    norm_a = float(math.sqrt(a @ a))
    norm_b = float(math.sqrt(b @ b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingMatcher:
    """Find semantic relatives of known PII using spaCy word vectors.

    Reuses the ``en_core_web_md`` vectors already loaded by Presidio
    (300-dim, ~685K vocabulary).  For each unmatched title-cased token,
    computes cosine similarity against known NAME/LOCATION entries and
    registers matches above a threshold as aliases.

    Runs after ``FuzzyMatcher`` so it skips tokens already caught by
    edit-distance or phonetic matching.
    """

    def __init__(self, nlp, threshold: float = 0.85) -> None:
        self._nlp = nlp
        self._threshold = threshold
        self._common_words = _load_name_set("common_words.txt")

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Scan for embedding-similar words. Returns new match count."""
        vocab = self._nlp.vocab

        # Build target vectors from eligible lookup entries.
        targets: list[tuple[str, object, str, str]] = []  # (word, vec, tag, cat)
        for entry in lookup.all_entries():
            if entry.pii_category not in _EMBED_CATEGORIES:
                continue
            for word in entry.original_term.split():
                if len(word) < _MIN_WORD_LEN:
                    continue
                lexeme = vocab[word]
                if not lexeme.has_vector:
                    continue
                targets.append((
                    word.lower(),
                    lexeme.vector,
                    entry.anonymised_term,
                    entry.pii_category,
                ))

        if not targets:
            return 0

        count = 0
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            for cell in line.cells:
                for match in _TOKEN_RE.finditer(cell.text):
                    word = match.group()
                    if len(word) < _MIN_WORD_LEN:
                        continue
                    if lookup.lookup(word):
                        continue
                    if word.lower() in self._common_words:
                        continue

                    lexeme = vocab[word]
                    if not lexeme.has_vector:
                        continue

                    vec = lexeme.vector
                    best_sim = 0.0
                    best_tag = ""
                    best_cat = ""

                    for _, target_vec, tag, category in targets:
                        sim = _cosine_similarity(vec, target_vec)
                        if sim > best_sim:
                            best_sim = sim
                            best_tag = tag
                            best_cat = category

                    if best_sim >= self._threshold:
                        lookup.register_alias(
                            word, best_tag, best_cat,
                            doc.source_path.name, line.line_number,
                        )
                        count += 1
            if on_progress:
                on_progress(i + 1, total)

        if count:
            logger.info(
                "EmbeddingMatcher: found %d semantic match(es)", count,
            )
        return count
