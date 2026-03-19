from __future__ import annotations

import logging
import re
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)

# Minimum length for a name variation to be worth replacing.
_MIN_VARIATION_LEN = 5

# Categories that mean "don't tag this column".
_SKIP_CATEGORIES = frozenset({"skip", "freetext"})


class ColumnDetector:
    """Tag entire CSV/XLSX columns based on a user-provided mapping."""

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        column_mapping: dict[int, str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if not column_mapping:
            return

        # Identify groups of adjacent NAME columns (for combining first+last).
        name_groups = _find_name_groups(column_mapping)

        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            # Skip the header row (line_number 1 in CSV/XLSX).
            if line.line_number == 1:
                if on_progress:
                    on_progress(i + 1, total)
                continue

            # --- Combined NAME handling ---
            for group in name_groups:
                parts = []
                for col_idx in group:
                    if col_idx < len(line.cells):
                        part = line.cells[col_idx].text.strip()
                        if part:
                            parts.append((col_idx, part))

                if not parts:
                    continue

                # Register the combined full name (replaced globally —
                # full names are specific enough to be unambiguous).
                full_name = " ".join(p[1] for p in parts)
                tag = _register(full_name, "NAME", doc, lookup, line.line_number)

                # Extract individual name words.
                # Multi-column: each cell is a word.
                # Single column with "Jane Smith": split on spaces.
                words = (
                    [p[1] for p in parts]
                    if len(parts) > 1
                    else full_name.split() if " " in full_name else []
                )

                # Register word-level aliases in the lookup (so Presidio/LLM
                # reuse the tag when they encounter the word on other rows)
                # but only replace on THIS row to avoid cross-row contamination
                # when different people share a first name.
                for word in words:
                    if word == full_name:
                        continue
                    if not lookup.lookup(word):
                        lookup.register_alias(
                            word, tag, "NAME",
                            doc.source_path.name, line.line_number,
                        )
                    line.replace_all(word, tag)

                # Generate email-local-part-style name variations
                # (e.g. james.oconnor, j.oconnor) and replace across the
                # entire document.  These are specific enough to be safe
                # for document-wide replacement, and email addresses are
                # already tagged by the pre-pass so we won't corrupt them.
                if len(parts) > 1:
                    first_text = parts[0][1]
                    last_text = parts[-1][1]
                elif " " in full_name:
                    name_words = full_name.split()
                    first_text = name_words[0]
                    last_text = name_words[-1]
                else:
                    first_text = ""
                    last_text = ""

                for variation in _name_variations(first_text, last_text):
                    if not lookup.lookup(variation):
                        lookup.register_alias(
                            variation, tag, "NAME",
                            doc.source_path.name, line.line_number,
                        )
                    doc.replace_all(variation, tag)

            # --- All other mapped columns ---
            for col_idx, category in column_mapping.items():
                if category.lower() in _SKIP_CATEGORIES or category == "NAME":
                    continue
                if col_idx >= len(line.cells):
                    continue

                cell_text = line.cells[col_idx].text.strip()
                if not cell_text:
                    continue

                _register(cell_text, category, doc, lookup, line.line_number)

            if on_progress:
                on_progress(i + 1, total)


    def cross_reference(
        self,
        doc: Document,
        lookup: LookupTable,
        column_mapping: dict[int, str],
    ) -> None:
        """Scan freetext/unmapped columns for values found in mapped columns.

        For NAME entries, only replace when the value appears title-cased
        in the target cell (avoids replacing common-word homonyms like
        lowercase "rose" in "a beautiful rose garden").
        """
        if not column_mapping:
            return

        # Identify columns that should be scanned (freetext + unmapped).
        mapped_cols = set(column_mapping.keys())
        freetext_cols = {
            i for i, cat in column_mapping.items()
            if cat.lower() in _SKIP_CATEGORIES
        }
        if doc.lines:
            all_cols = set(range(len(doc.lines[0].cells)))
            freetext_cols |= all_cols - mapped_cols

        if not freetext_cols:
            return

        # Collect entries that came from mapped columns (NAME single words).
        name_entries = [
            e for e in lookup.all_entries()
            if e.pii_category == "NAME" and " " not in e.original_term
        ]

        for line in doc.lines:
            if line.line_number == 1:
                continue
            for cell in line.cells:
                if cell.col_index not in freetext_cols:
                    continue
                for entry in name_entries:
                    original = entry.original_term
                    # Only match title-cased occurrences to avoid
                    # replacing common words like "rose", "May", etc.
                    if original in cell.text and original[0].isupper():
                        cell.text = _safe_replace(
                            cell.text, original, entry.anonymised_term,
                        )


def _safe_replace(text: str, original: str, replacement: str) -> str:
    """Replace *original* with *replacement*, skipping text inside [brackets]."""
    from src.pipeline.document import _safe_replace as doc_safe_replace
    return doc_safe_replace(text, original, replacement)


def _register(
    original: str,
    category: str,
    doc: Document,
    lookup: LookupTable,
    line_number: int,
) -> str:
    """Register *original* in the lookup and replace across the document."""
    existing = lookup.lookup(original)
    if existing:
        return existing
    tag = lookup.register(original, category, doc.source_path.name, line_number)
    doc.replace_all(original, tag)
    return tag


def _find_name_groups(column_mapping: dict[int, str]) -> list[list[int]]:
    """Return groups of adjacent column indices all mapped to NAME."""
    name_cols = sorted(ci for ci, cat in column_mapping.items() if cat == "NAME")
    if not name_cols:
        return []

    groups: list[list[int]] = [[name_cols[0]]]
    for col in name_cols[1:]:
        if col == groups[-1][-1] + 1:
            groups[-1].append(col)
        else:
            groups.append([col])
    return groups


def _normalise_name_part(name: str) -> str:
    """Lowercase and strip apostrophes / hyphens / spaces."""
    return re.sub(r"['\-\s]", "", name).lower()


def _name_variations(first: str, last: str) -> list[str]:
    """Generate email-local-part-style variations from first and last names.

    Examples for ("James", "O'Connor"):
        james.oconnor, oconnor.james, j.oconnor, jamesoconnor, joconnor
    """
    f = _normalise_name_part(first)
    la = _normalise_name_part(last)
    if not f or not la:
        return []

    initial = f[0]
    last_initial = la[0]
    variations = [
        f"{f}.{la}",            # james.oconnor
        f"{la}.{f}",            # oconnor.james
        f"{initial}.{la}",      # j.oconnor
        f"{f}.{last_initial}",  # james.o
        f"{f}{la}",             # jamesoconnor
        f"{initial}{la}",       # joconnor
    ]
    return [v for v in variations if len(v) >= _MIN_VARIATION_LEN]
