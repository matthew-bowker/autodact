"""Detect high-entropy alphanumeric strings (API keys, tokens, hashes)."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from src.pipeline.document import Document
from src.pipeline.lookup_table import LookupTable

logger = logging.getLogger(__name__)

# Alphanumeric tokens of 8+ chars (includes underscores and hyphens).
_CANDIDATE_RE = re.compile(r"\b[A-Za-z0-9_-]{8,}\b")

# Text inside [TAG ...] brackets.
_TAG_SPAN_RE = re.compile(r"\[[^\]]*\]")

# snake_case or kebab-case identifiers (programming field names).
_SNAKE_KEBAB_RE = re.compile(r"^[a-z]+(?:[_-][a-z]+)+$")

_MIN_LEN = 8
_MIN_ENTROPY = 3.5  # bits per character


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _has_mixed_chars(s: str) -> bool:
    """True if string contains at least 2 of: uppercase, lowercase, digits."""
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    return (has_upper + has_lower + has_digit) >= 2


class EntropyDetector:
    """Flag high-entropy alphanumeric strings as potential IDs/tokens.

    Catches API keys, session tokens, hashes, and random reference IDs
    that no other detection layer looks for.  Guards against false
    positives by requiring mixed character types, minimum length,
    and minimum Shannon entropy.
    """

    def __init__(
        self,
        min_entropy: float = _MIN_ENTROPY,
        min_length: int = _MIN_LEN,
    ) -> None:
        self._min_entropy = min_entropy
        self._min_length = min_length

    def process(self, doc: Document, lookup: LookupTable) -> int:
        """Scan for high-entropy tokens. Returns number of new detections."""
        count = 0
        for line in doc.lines:
            tag_spans = [
                (m.start(), m.end())
                for m in _TAG_SPAN_RE.finditer(line.text)
            ]
            for cell in line.cells:
                for match in _CANDIDATE_RE.finditer(cell.text):
                    token = match.group()
                    start, end = match.start(), match.end()

                    # Skip tokens inside existing [TAG] brackets.
                    if any(ts <= start and end <= te for ts, te in tag_spans):
                        continue

                    if len(token) < self._min_length:
                        continue

                    # Pure alphabetic → normal word, skip.
                    if token.isalpha():
                        continue

                    # Pure numeric → handled by structured patterns, skip.
                    if token.isdigit():
                        continue

                    # snake_case / kebab-case → programming identifier, skip.
                    if _SNAKE_KEBAB_RE.match(token):
                        continue

                    # Must contain mixed character types.
                    if not _has_mixed_chars(token):
                        continue

                    # Entropy check.
                    if _shannon_entropy(token) < self._min_entropy:
                        continue

                    # Already registered?
                    if lookup.lookup(token):
                        continue

                    tag = lookup.register(
                        token, "ID",
                        doc.source_path.name, line.line_number,
                    )
                    doc.replace_all(token, tag)
                    count += 1

        if count:
            logger.info("EntropyDetector: flagged %d high-entropy token(s)", count)
        return count
