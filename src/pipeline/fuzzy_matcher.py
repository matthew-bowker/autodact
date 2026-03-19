"""Post-detection fuzzy matcher that catches misspellings of known PII."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import _load_name_set

logger = logging.getLogger(__name__)

# Matches title-cased word tokens (same pattern as name_detector).
_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b")

# Categories eligible for fuzzy matching.
_FUZZY_CATEGORIES: frozenset[str] = frozenset({"NAME", "LOCATION"})

_MIN_WORD_LEN = 4

# Maximum length difference for phonetic matching.
_PHONETIC_MAX_LEN_DIFF = 3


def _damerau_levenshtein(s: str, t: str) -> int:
    """Compute Damerau-Levenshtein distance between two strings.

    Handles insertions, deletions, substitutions, and adjacent
    transpositions.  Pure Python — no external dependency needed.
    """
    len_s = len(s)
    len_t = len(t)

    # Use a dict for the matrix to handle the "infinite" row.
    d: dict[tuple[int, int], int] = {}
    max_dist = len_s + len_t

    d[(-1, -1)] = max_dist
    for i in range(len_s + 1):
        d[(i, -1)] = max_dist
        d[(i, 0)] = i
    for j in range(len_t + 1):
        d[(-1, j)] = max_dist
        d[(0, j)] = j

    last_row: dict[str, int] = {}

    for i in range(1, len_s + 1):
        ch_s = s[i - 1]
        last_match_col = 0
        for j in range(1, len_t + 1):
            ch_t = t[j - 1]
            last_match_row = last_row.get(ch_t, 0)
            cost = 0 if ch_s == ch_t else 1

            d[(i, j)] = min(
                d[(i - 1, j)] + 1,          # deletion
                d[(i, j - 1)] + 1,          # insertion
                d[(i - 1, j - 1)] + cost,   # substitution
                d[(last_match_row - 1, last_match_col - 1)]
                + (i - last_match_row - 1) + 1
                + (j - last_match_col - 1),  # transposition
            )
            if cost == 0:
                last_match_col = j
        last_row[ch_s] = i

    return d[(len_s, len_t)]


# ---------------------------------------------------------------------------
# Metaphone phonetic encoding
# ---------------------------------------------------------------------------

_VOWELS = frozenset("AEIOU")


def _metaphone(word: str) -> str:
    """Simplified Metaphone phonetic encoding for English names.

    Returns a consonant-based code that groups words that sound alike.
    Covers the main English pronunciation rules for personal and place names.
    """
    if not word:
        return ""

    w = word.upper()
    # Strip non-alpha.
    w = "".join(c for c in w if c.isalpha())
    if not w:
        return ""

    # Drop silent initial letters.
    if len(w) >= 2:
        if w[:2] in ("AE", "GN", "KN", "PN", "WR"):
            w = w[1:]

    code: list[str] = []
    i = 0
    prev = ""

    while i < len(w):
        c = w[i]
        next_c = w[i + 1] if i + 1 < len(w) else ""
        next2 = w[i + 2] if i + 2 < len(w) else ""

        # Skip duplicate consecutive consonants.
        if c == prev and c != "C":
            i += 1
            continue

        if c in _VOWELS:
            # Keep initial vowel only.
            if i == 0:
                code.append(c)
            i += 1
            prev = c
            continue

        if c == "B":
            # Silent B after M at end (e.g., "dumb").
            if not (prev == "M" and i == len(w) - 1):
                code.append("B")

        elif c == "C":
            if next_c in ("E", "I", "Y"):
                code.append("S")
            elif next_c == "H":
                code.append("X")
                i += 1
            else:
                code.append("K")

        elif c == "D":
            if next_c == "G" and next2 in ("E", "I", "Y"):
                code.append("J")
                i += 1
            else:
                code.append("T")

        elif c == "F":
            code.append("F")

        elif c == "G":
            if next_c in ("E", "I", "Y"):
                code.append("J")
            elif next_c == "H" and i + 2 < len(w) and w[i + 2] not in _VOWELS:
                # GH silent before consonant (e.g., "night").
                i += 1
            elif not (i > 0 and next_c == "H"):
                code.append("K")

        elif c == "H":
            if next_c in _VOWELS and prev not in _VOWELS:
                code.append("H")

        elif c == "J":
            code.append("J")

        elif c == "K":
            if prev != "C":
                code.append("K")

        elif c == "L":
            code.append("L")

        elif c == "M":
            code.append("M")

        elif c == "N":
            code.append("N")

        elif c == "P":
            if next_c == "H":
                code.append("F")
                i += 1
            else:
                code.append("P")

        elif c == "Q":
            code.append("K")

        elif c == "R":
            code.append("R")

        elif c == "S":
            if next_c == "H":
                code.append("X")
                i += 1
            elif next_c == "C" and next2 in ("E", "I", "Y"):
                code.append("S")
                i += 1
            else:
                code.append("S")

        elif c == "T":
            if next_c == "H":
                code.append("0")  # theta
                i += 1
            else:
                code.append("T")

        elif c == "V":
            code.append("F")

        elif c == "W":
            if next_c in _VOWELS:
                code.append("W")

        elif c == "X":
            code.append("KS")

        elif c == "Y":
            if next_c in _VOWELS:
                code.append("Y")

        elif c == "Z":
            code.append("S")

        prev = c
        i += 1

    return "".join(code)


class FuzzyMatcher:
    """Find misspellings of known PII terms in the document.

    Scans title-cased tokens that are NOT already in the lookup table
    and compares them against known NAME and LOCATION entries using:
    1. Damerau-Levenshtein edit distance (catches typos/transpositions)
    2. Metaphone phonetic codes (catches sound-alike variants)

    Guards against false positives:
    - Only matches NAME and LOCATION categories.
    - Rejects candidates that are common English words.
    - Minimum word length of 4 characters.
    - Conservative edit distance thresholds (1 for short words, 2 for longer).
    - Phonetic matching requires lengths within 3 chars of each other.
    """

    def __init__(self, max_distance: int = 2) -> None:
        self._max_distance = max_distance
        self._common_words = _load_name_set("common_words.txt")

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Find fuzzy matches and register as aliases. Returns new match count."""
        # Build target set from eligible lookup entries.
        targets: dict[str, tuple[str, str]] = {}  # lower_word -> (tag, category)
        for entry in lookup.all_entries():
            if entry.pii_category not in _FUZZY_CATEGORIES:
                continue
            for word in entry.original_term.split():
                if len(word) >= _MIN_WORD_LEN:
                    targets[word.lower()] = (
                        entry.anonymised_term,
                        entry.pii_category,
                    )

        if not targets:
            return 0

        # Build phonetic index for fallback matching.
        phonetic_index: dict[str, list[str]] = defaultdict(list)
        for target_word in targets:
            code = _metaphone(target_word)
            if code:
                phonetic_index[code].append(target_word)

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

                    # Try edit-distance first, then phonetic fallback.
                    best = self._find_best_match(word, targets)
                    if best is None:
                        best = self._find_phonetic_match(
                            word, targets, phonetic_index,
                        )

                    if best:
                        tag, category = targets[best]
                        lookup.register_alias(
                            word, tag, category,
                            doc.source_path.name, line.line_number,
                        )
                        count += 1
            if on_progress:
                on_progress(i + 1, total)

        if count:
            logger.info("FuzzyMatcher: found %d misspelling(s)", count)
        return count

    def _find_best_match(
        self,
        word: str,
        targets: dict[str, tuple[str, str]],
    ) -> str | None:
        word_lower = word.lower()
        threshold = 1 if len(word) <= 6 else min(2, self._max_distance)

        best: str | None = None
        best_dist = threshold + 1

        for target in targets:
            # Quick length filter.
            if abs(len(word_lower) - len(target)) > threshold:
                continue
            dist = _damerau_levenshtein(word_lower, target)
            if dist <= threshold and dist < best_dist:
                best = target
                best_dist = dist

        return best

    def _find_phonetic_match(
        self,
        word: str,
        targets: dict[str, tuple[str, str]],
        phonetic_index: dict[str, list[str]],
    ) -> str | None:
        """Find a target with the same phonetic code as *word*."""
        code = _metaphone(word.lower())
        if not code or code not in phonetic_index:
            return None

        word_len = len(word)
        for candidate in phonetic_index[code]:
            if candidate in targets and abs(len(candidate) - word_len) <= _PHONETIC_MAX_LEN_DIFF:
                return candidate
        return None
