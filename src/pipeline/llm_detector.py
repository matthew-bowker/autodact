from __future__ import annotations

import logging
from typing import Callable, Protocol

from src.pipeline.document import Document, Line
from src.pipeline.llm_engine import build_user_prompt
from src.pipeline.lookup_table import LookupTable

# Minimum entity word length for per-word replacement in multi-cell data
_MIN_WORD_LEN = 3

logger = logging.getLogger(__name__)


class LLMEngineProtocol(Protocol):
    def detect_pii(
        self, user_prompt: str, categories: list[str]
    ) -> list[dict[str, str]]: ...


class LLMDetector:
    def __init__(
        self,
        engine: LLMEngineProtocol,
        categories: list[str] | None = None,
    ) -> None:
        self._engine = engine
        self._categories = categories or ["NAME", "ORG", "LOCATION", "JOBTITLE"]

    def process(
        self,
        doc: Document,
        lookup: LookupTable,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[int]:
        flagged_lines: list[int] = []
        total = len(doc.lines)

        for i, line in enumerate(doc.lines):
            if not line.text.strip():
                if on_progress:
                    on_progress(i + 1, total)
                continue

            # Build prompt text — prefix cells with column headers for CSV/XLSX
            # so the LLM understands the data context.
            prompt_text = _header_prefixed_text(line, doc.headers)
            user_prompt = build_user_prompt(prompt_text)
            detections = self._detect_with_retry(user_prompt, line.text)

            if detections is None:
                flagged_lines.append(line.line_number)
                logger.warning(
                    "LLM parse failure on line %d, flagged for review",
                    line.line_number,
                )
            else:
                for det in detections:
                    original = det["original"]
                    category = det["category"]
                    if category not in self._categories:
                        continue
                    if _entity_in_text(original, line.text):
                        tag = lookup.register(
                            original, category, doc.source_path.name, line.line_number
                        )
                        doc.replace_all(original, tag)
                        # For CSV/XLSX: entity may span cells (e.g. "Jane Smith"
                        # with "Jane" and "Smith" in separate columns).  Replace
                        # individual words on THIS line only — global replacement
                        # of common words like "Road" or "Street" would corrupt
                        # unrelated rows.
                        if " " in original and original not in line.text:
                            for word in original.split():
                                if len(word) >= _MIN_WORD_LEN:
                                    line.replace_all(word, tag)

            if on_progress:
                on_progress(i + 1, total)

        return flagged_lines

    def _detect_with_retry(
        self,
        user_prompt: str,
        focus_text: str,
    ) -> list[dict[str, str]] | None:
        for attempt in range(2):
            try:
                detections = self._engine.detect_pii(
                    user_prompt, self._categories
                )
                return [
                    d
                    for d in detections
                    if _entity_in_text(d["original"], focus_text)
                ]
            except Exception:
                if attempt == 0:
                    logger.debug("LLM inference failed, retrying...")
                continue
        return None


def _header_prefixed_text(line: Line, headers: list[str] | None) -> str:
    """Build line text with column headers prefixed to each cell value.

    For CSV/XLSX data this produces e.g.
    ``First Name: Jane | Last Name: Smith | Notes: Joined in 2018.``
    which gives the LLM much better context than raw pipe-separated values.
    Falls back to plain ``line.text`` for non-tabular data or missing headers.
    """
    if not headers or len(line.cells) <= 1:
        return line.text
    parts: list[str] = []
    for cell in line.cells:
        idx = cell.col_index
        header = headers[idx] if idx < len(headers) else ""
        if header:
            parts.append(f"{header}: {cell.text}")
        else:
            parts.append(cell.text)
    return " | ".join(parts)


def _entity_in_text(original: str, focus_text: str) -> bool:
    """Check if an entity is present in the focus text.

    Handles CSV/XLSX rows where cells are joined with ' | ', so a
    multi-word entity like 'Jane Smith' may appear as 'Jane | Smith'.
    """
    if original in focus_text:
        return True
    # Remove cell separators and check again
    if " | " in focus_text:
        return original in focus_text.replace(" | ", " ")
    return False
