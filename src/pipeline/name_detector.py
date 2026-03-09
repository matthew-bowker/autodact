from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)


def _get_data_dir() -> Path:
    """Resolve the data directory, handling PyInstaller frozen bundles."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "src" / "pipeline" / "data"
    return Path(__file__).resolve().parent / "data"


_DATA_DIR = _get_data_dir()

# Matches title-cased word tokens, including hyphenated forms like "Al-Hassan".
_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b")

# Detects text already inside a [TAG …] bracket.
_TAG_SPAN_RE = re.compile(r"\[[^\]]*\]")

# Domain-specific words that appear in the name dataset but are NOT personal
# names.  These supplement the dictionary-based common-words file, catching
# titles, organisational terms, address words, and calendar terms that
# dictionaries miss.
_DOMAIN_EXCLUSIONS: frozenset[str] = frozenset({
    # Titles and honorifics
    "professor", "doctor", "councillor", "minister", "director", "sheikh",
    "imam", "mullah", "reverend", "bishop", "cardinal", "deacon",
    "sergeant", "lieutenant", "captain", "colonel", "general", "commander",
    "ambassador", "president", "chancellor", "chairman", "inspector",
    "superintendent", "officer", "governor", "senator", "commissioner",
    "warden", "dean", "provost", "rector", "pastor", "vicar",
    # Organisational suffixes / terms
    "ltd", "inc", "corp", "gmbh", "plc", "llc", "institut", "institute",
    "university", "college", "academy", "foundation", "council",
    "department", "ministry", "bureau", "authority", "commission",
    "hospital", "clinic", "surgery", "pharmacy",
    # Address / street words (expanded based on benchmark false positives:
    # Presidio tags synthetic Faker addresses like "Lake Christopherborough",
    # "Garcia Haven", "Santos Colonnade" as PERSON)
    "street", "road", "lane", "drive", "boulevard", "court", "place",
    "avenue", "terrace", "crescent", "close", "mews", "walk", "hill",
    "square", "row", "gardens", "park", "green", "circle", "way",
    "rue", "via", "calle", "salai", "marg", "nagar", "chowk",
    "corniche", "prospekt",
    # Additional address words (benchmark-observed FPs)
    "ridge", "path", "trail", "crossing", "hollow", "haven", "glen",
    "inlet", "heights", "manor", "meadow", "vista",
    # Common nouns / adjectives the name dataset includes erroneously
    "clinical", "completed", "certified", "applied", "advanced",
    "senior", "junior", "principal", "associate", "assistant",
    "digital", "technical", "financial", "international", "national",
    "engineering", "consulting", "analytics", "services", "solutions",
    "research", "telecom",
    # UK constituent countries / common subdivisions not in pycountry
    "england", "scotland", "wales",
    # Foreign-language country/region names
    "tunisie", "algerie", "maroc", "deutschland", "schweiz",
    # Calendar (includes "may" — common false positive)
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    # Month abbreviations (benchmark showed "Sep" as FP)
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec",
})


def _is_inflected_common(word: str, common: frozenset[str]) -> bool:
    """Return True if *word* is a simple inflection of a common English word.

    Catches plurals (Kings → king), past tense (Completed → complete),
    and -ing forms (Consulting → consult) that the dictionary misses.
    """
    if len(word) < 4:
        return False
    # Plural: -s, -es
    if word.endswith("s") and word[:-1] in common:
        return True
    if word.endswith("es") and word[:-2] in common:
        return True
    # Past tense / participle: -ed, -d
    if word.endswith("ed") and (word[:-2] in common or word[:-1] in common):
        return True
    # Gerund / progressive: -ing
    if word.endswith("ing") and (word[:-3] in common or word[:-3] + "e" in common):
        return True
    return False


def _load_name_set(filename: str) -> frozenset[str]:
    """Load a newline-delimited name file into a case-folded frozenset."""
    path = _DATA_DIR / filename
    if not path.exists():
        logger.warning("Name file not found: %s — run scripts/extract_names.py", path)
        return frozenset()
    return frozenset(
        line.lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


class NameDictionaryDetector:
    """Catch names missed by spaCy using a 1.7 M-entry name dictionary.

    Loads three flat text files produced by ``scripts/extract_names.py``:

    * ``first_names.txt``  — 693 K first names from the *names-dataset* package
    * ``last_names.txt``   — 977 K last names
    * ``common_words.txt`` — 234 K English words (Webster's 2nd International)

    **Matching strategy** (to minimise false positives):

    1. A title-cased word that is in the name dictionary but NOT in the
       common-words list is tagged directly (high confidence).
    2. A title-cased word that is in BOTH the name dictionary and the
       common-words list is only tagged when it appears adjacent to
       another name-dictionary word (pair detection).  This catches
       "Fatima Al-Hassan" even though "fatima" is in Webster's.
    """

    def __init__(self) -> None:
        self._first_names = _load_name_set("first_names.txt")
        self._last_names = _load_name_set("last_names.txt")
        self._common_words = _load_name_set("common_words.txt")
        logger.info(
            "NameDictionaryDetector loaded: %d first, %d last, %d common",
            len(self._first_names), len(self._last_names), len(self._common_words),
        )

    @property
    def loaded(self) -> bool:
        return bool(self._first_names or self._last_names)

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if not self._first_names and not self._last_names:
            return
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            self._process_line(line, doc, lookup)
            if on_progress:
                on_progress(i + 1, total)

    def _process_line(
        self,
        line: "Line",  # noqa: F821 (forward ref)
        doc: Document,
        lookup: LookupTable,
    ) -> None:
        text = line.text
        if not text.strip():
            return

        # Build a set of character ranges already inside [TAG …] brackets
        # so we can skip tokens that overlap with existing tags.
        tag_spans = [(m.start(), m.end()) for m in _TAG_SPAN_RE.finditer(text)]

        # Find all title-cased tokens.
        tokens = list(_TOKEN_RE.finditer(text))
        if not tokens:
            return

        # Classify each token.
        candidates: list[dict] = []
        for tok in tokens:
            start, end = tok.start(), tok.end()

            # Skip tokens inside existing tag brackets.
            if any(ts <= start and end <= te for ts, te in tag_spans):
                continue

            word = tok.group()
            parts = word.split("-")
            lower_parts = [p.lower() for p in parts]

            # Check each part of a (possibly hyphenated) token.
            is_name = any(
                p in self._first_names or p in self._last_names
                for p in lower_parts
            )

            # Hard exclusion: domain words (titles, org terms, address words)
            # can NEVER be tagged, even via pair detection.
            is_domain_excluded = any(
                p in _DOMAIN_EXCLUSIONS for p in lower_parts
            )

            # Soft exclusion: common English dictionary words are only
            # tagged when adjacent to a high-confidence name (pair detection).
            is_dict_common = any(
                p in self._common_words
                or _is_inflected_common(p, self._common_words)
                for p in lower_parts
            )

            if not is_name or is_domain_excluded:
                continue

            candidates.append({
                "match": tok,
                "word": word,
                "is_common": is_dict_common,
            })

        if not candidates:
            return

        # Decide which candidates to tag.
        to_tag: set[int] = set()
        for idx, cand in enumerate(candidates):
            if not cand["is_common"]:
                # High confidence: name-dict word that isn't a common English word.
                to_tag.add(idx)
            else:
                # Ambiguous: only tag if adjacent to a HIGH-CONFIDENCE name
                # (i.e. a name-dict word that is NOT a common English word).
                # Two common words cannot boost each other — prevents
                # "The Park" from being tagged as a name pair.
                for neighbour in (idx - 1, idx + 1):
                    if 0 <= neighbour < len(candidates):
                        if candidates[neighbour]["is_common"]:
                            continue
                        # Distance between the two tokens in the text.
                        n_match = candidates[neighbour]["match"]
                        c_match = cand["match"]
                        gap = max(
                            0,
                            max(c_match.start(), n_match.start())
                            - min(c_match.end(), n_match.end()),
                        )
                        if gap <= 4:
                            to_tag.add(idx)
                            to_tag.add(neighbour)
                            break

        # Register and replace.
        for idx in sorted(to_tag):
            word = candidates[idx]["word"]
            # Already registered by a previous pipeline layer?
            existing = lookup.lookup(word)
            if existing:
                doc.replace_all(word, existing)
            else:
                tag = lookup.register(
                    word, "NAME", doc.source_path.name, line.line_number,
                )
                doc.replace_all(word, tag)
