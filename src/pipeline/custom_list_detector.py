"""Detector that matches text against user-provided custom word lists."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)


def _build_pattern(words: list[str]) -> re.Pattern[str] | None:
    """Build a compiled regex that matches any word in the list.

    Words are matched with word boundaries and case-insensitively.
    Longer words are tried first so that overlapping terms like
    "New York City" match before "New York".
    """
    if not words:
        return None
    sorted_words = sorted(words, key=len, reverse=True)
    escaped = [re.escape(w) for w in sorted_words if w.strip()]
    if not escaped:
        return None
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


class CustomListDetector:
    """Tag text using user-provided word lists, each tied to a PII category."""

    def __init__(self, lists: list[dict]) -> None:
        # Group words by category and build one regex per category.
        category_words: dict[str, list[str]] = {}
        for lst in lists:
            cat = lst.get("category", "")
            words = lst.get("words", [])
            cleaned = [w.strip() for w in words if w.strip()]
            if cat and cleaned:
                category_words.setdefault(cat, []).extend(cleaned)

        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for cat, words in category_words.items():
            pattern = _build_pattern(words)
            if pattern:
                self._patterns.append((cat, pattern))
                logger.info(
                    "CustomListDetector: %d word(s) for category %s",
                    len(words), cat,
                )

    @property
    def has_lists(self) -> bool:
        return bool(self._patterns)

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        if not self._patterns:
            return
        total = len(doc.lines)
        for i, line in enumerate(doc.lines):
            for cell in line.cells:
                for category, pattern in self._patterns:
                    for match in pattern.finditer(cell.text):
                        original = match.group()
                        # Skip text already inside [TAG ...] brackets.
                        if original.startswith("[") and original.endswith("]"):
                            continue
                        existing = lookup.lookup(original)
                        if existing:
                            doc.replace_all(original, existing)
                        else:
                            tag = lookup.register(
                                original, category,
                                doc.source_path.name, line.line_number,
                            )
                            doc.replace_all(original, tag)
            if on_progress:
                on_progress(i + 1, total)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_custom_lists(path: Path) -> list[dict]:
    """Load custom word lists from a JSON file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("lists", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to parse custom lists file: %s", path)
        return []


def save_custom_lists(path: Path, lists: list[dict]) -> None:
    """Save custom word lists to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"lists": lists}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
