"""Post-processing validation pass that removes obvious false positives."""

from __future__ import annotations

import logging

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)

# High-frequency stop words that should never be PII.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "but", "with", "or", "a", "an", "to",
    "in", "on", "at", "of", "is", "was", "are", "be", "as",
    "it", "by", "not", "no", "so", "if", "do", "up", "my",
    "we", "he", "me",
})

# Role/title words the LLM mis-tags as NAME (benchmark-observed FPs).
# Applied only to the NAME category in post-validation.
_NAME_ROLE_WORDS: frozenset[str] = frozenset({
    "applicant", "borrower", "consignor", "consignee", "claimant",
    "complainant", "defendant", "plaintiff", "respondent", "petitioner",
    "inspector", "examiner", "auditor", "investigator", "assessor",
    "omitted", "redacted", "withheld", "confidential", "restricted",
})

# Known placeholder values (checked case-insensitively).
_PLACEHOLDERS: frozenset[str] = frozenset({
    "n/a", "na", "n.a.", "unknown", "tbd", "none", "null",
    "not applicable", "not available", "to be filled in",
    "to be determined", "to be confirmed", "pending",
})

# Categories where pure-numeric values are valid.
_NUMERIC_OK: frozenset[str] = frozenset({
    "ID", "PHONE", "DOB", "POSTCODE",
})


def validate_and_clean(doc: Document, lookup: LookupTable) -> int:
    """Remove false positives from *lookup* and restore originals in *doc*.

    Returns the number of entries removed.
    """
    to_remove: list[tuple[str, str]] = []  # (original_term, anonymised_term)

    for entry in lookup.all_entries():
        original = entry.original_term
        category = entry.pii_category
        tag = entry.anonymised_term
        normalised = original.lower().strip()

        # Stop words (any category).
        if normalised in _STOP_WORDS:
            to_remove.append((original, tag))
            continue

        # Known placeholders (any category).
        if normalised in _PLACEHOLDERS:
            to_remove.append((original, tag))
            continue

        # Pure-numeric in a category where that makes no sense.
        if original.strip().isdigit() and category not in _NUMERIC_OK:
            to_remove.append((original, tag))
            continue

        # Single-character entries (any category).
        if len(original.strip()) < 2:
            to_remove.append((original, tag))
            continue

        # NAME-specific: reject role/title words the LLM mis-tags.
        if category == "NAME" and normalised in _NAME_ROLE_WORDS:
            to_remove.append((original, tag))
            continue

    for original, tag in to_remove:
        lookup.remove(original)
        # Direct str.replace — _safe_replace skips bracketed text, but
        # we need to replace the bracket tag itself back to the original.
        for line in doc.lines:
            for cell in line.cells:
                cell.text = cell.text.replace(tag, original)

    if to_remove:
        logger.info("Post-validation removed %d false positive(s)", len(to_remove))

    return len(to_remove)
