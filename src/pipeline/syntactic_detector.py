"""Detect proper nouns in meaningful syntactic roles that spaCy NER missed."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import _DOMAIN_EXCLUSIONS, _load_name_set

logger = logging.getLogger(__name__)

# Syntactic roles that indicate a meaningful grammatical position.
_MEANINGFUL_DEPS: frozenset[str] = frozenset({
    "dobj", "pobj", "iobj",       # objects
    "attr", "appos",              # attributes / appositives
    "nsubj", "nsubjpass", "conj", # subjects / conjuncts
})

# Detects text already inside a [TAG ...] bracket.
_TAG_SPAN_RE = re.compile(r"\[[^\]]*\]")

_MIN_TOKEN_LEN = 3


class SyntacticDetector:
    """Catch proper nouns in syntactic roles that NER missed.

    Reuses the spaCy ``Language`` object already loaded by Presidio
    to avoid loading the model twice.  For each line the detector:

    1. Runs spaCy to get POS tags and dependency labels.
    2. Finds tokens that are proper nouns (``PROPN``) in a meaningful
       syntactic role (subject, object, appositive, etc.).
    3. Filters out domain-specific words and common English words.
    4. Groups consecutive proper nouns into multi-token entities.
    5. Registers survivors as ``NAME`` in the lookup table.
    """

    def __init__(self, nlp) -> None:
        self._nlp = nlp
        self._common_words = _load_name_set("common_words.txt")

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            self._process_line(line, doc, lookup)
            if on_progress:
                on_progress(i + 1, total)

    def _process_line(
        self, line, doc: Document, lookup: LookupTable,
    ) -> None:
        text = line.text
        if not text.strip():
            return

        # Build tag-span set so we skip tokens inside existing brackets.
        tag_spans = [(m.start(), m.end()) for m in _TAG_SPAN_RE.finditer(text)]

        spacy_doc = self._nlp(text)

        # Collect qualifying proper-noun tokens.
        candidates: list[tuple[int, int, str]] = []  # (start, end, text)
        for token in spacy_doc:
            if token.pos_ != "PROPN":
                continue
            if token.dep_ not in _MEANINGFUL_DEPS:
                continue
            if len(token.text) < _MIN_TOKEN_LEN:
                continue
            if not token.text[0].isupper():
                continue

            start, end = token.idx, token.idx + len(token.text)

            # Skip tokens inside existing tag brackets.
            if any(ts <= start and end <= te for ts, te in tag_spans):
                continue

            word_lower = token.text.lower()
            if word_lower in _DOMAIN_EXCLUSIONS:
                continue
            if word_lower in self._common_words:
                continue
            if lookup.lookup(token.text):
                continue

            candidates.append((start, end, token.text))

        if not candidates:
            return

        # Group consecutive tokens into multi-word entities.
        groups = _group_consecutive(candidates, text)

        for entity_text in groups:
            existing = lookup.lookup(entity_text)
            if existing:
                doc.replace_all(entity_text, existing)
            else:
                tag = lookup.register(
                    entity_text, "NAME",
                    doc.source_path.name, line.line_number,
                )
                doc.replace_all(entity_text, tag)


def _group_consecutive(
    candidates: list[tuple[int, int, str]],
    full_text: str,
) -> list[str]:
    """Group consecutive candidate tokens into multi-word entities.

    Two tokens are consecutive if only whitespace separates them.
    Returns the full text span for each group (e.g. "Jane Smith").
    """
    if not candidates:
        return []

    groups: list[str] = []
    group_start = candidates[0][0]
    group_end = candidates[0][1]

    for j in range(1, len(candidates)):
        prev_end = candidates[j - 1][1]
        curr_start = candidates[j][0]
        between = full_text[prev_end:curr_start]
        if between.strip() == "":  # only whitespace between
            group_end = candidates[j][1]
        else:
            groups.append(full_text[group_start:group_end])
            group_start = candidates[j][0]
            group_end = candidates[j][1]

    groups.append(full_text[group_start:group_end])
    return groups
