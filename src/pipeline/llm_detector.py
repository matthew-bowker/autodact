from __future__ import annotations

import logging
from typing import Callable, Protocol

from src.pipeline.document import Document, Line
from src.pipeline.llm_engine import build_user_prompt
from src.pipeline.lookup_table import LookupTable

# Minimum entity word length for per-word replacement in multi-cell data
_MIN_WORD_LEN = 3

# Column-mapping categories that still need LLM review (free-form content).
_FREETEXT_CATEGORIES = frozenset({"skip", "freetext"})

# Cell values too short or generic to contain PII — consistent with
# the post-validator's _PLACEHOLDERS / short-entity filters.
_TRIVIAL_VALUES: frozenset[str] = frozenset({
    "n/a", "na", "n.a.", "unknown", "tbd", "none", "null",
    "not applicable", "not available", "to be filled in",
    "to be determined", "to be confirmed", "pending",
    "yes", "no", "true", "false",
    "-", "--", "---", "",
})

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
        mapped_columns: dict[int, str] | None = None,
    ) -> list[int]:
        flagged_lines: list[int] = []
        total = len(doc.lines)

        # Columns whose content is already fully handled by the column
        # detector — the LLM only needs to look at unmapped / freetext
        # columns.
        covered_cols: frozenset[int] = _covered_columns(mapped_columns)

        # Cache LLM results keyed by the (trimmed) prompt text.
        # Identical freetext content across rows produces the same
        # detections — no need to call the model again.
        result_cache: dict[str, list[dict[str, str]] | None] = {}

        for i, line in enumerate(doc.lines):
            if not line.text.strip():
                if on_progress:
                    on_progress(i + 1, total)
                continue

            # For structured data: if every non-empty cell falls in a
            # column the column detector already handled, skip the row.
            if covered_cols and _line_is_covered(line, covered_cols):
                if on_progress:
                    on_progress(i + 1, total)
                continue

            # Build prompt text.  When we have a column mapping, send
            # only the unmapped cells — the mapped ones are already
            # tagged and just waste context tokens.
            prompt_text = _prompt_text_for_line(
                line, doc.headers, covered_cols,
            )

            # Short-circuit: if every uncovered cell is trivially
            # non-PII (placeholders, pure numerics, very short strings)
            # there is nothing for the model to find.
            if _all_cells_trivial(line, covered_cols):
                if on_progress:
                    on_progress(i + 1, total)
                continue

            # Re-use a previous result if the LLM has already seen
            # identical content (common in CSV freetext columns).
            if prompt_text in result_cache:
                detections = result_cache[prompt_text]
            else:
                user_prompt = build_user_prompt(prompt_text)
                detections = self._detect_with_retry(
                    user_prompt, line.text,
                )
                result_cache[prompt_text] = detections

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
                        # For CSV/XLSX: entity may span cells (e.g. "Jane Smith"
                        # with "Jane" and "Smith" in separate columns).  Register
                        # individual words as aliases so the orchestrator's
                        # additive replay can replace them.
                        if " " in original and original not in line.text:
                            for word in original.split():
                                if len(word) >= _MIN_WORD_LEN and not lookup.lookup(word):
                                    lookup.register_alias(
                                        word, tag, category,
                                        doc.source_path.name, line.line_number,
                                    )

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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _covered_columns(
    mapped_columns: dict[int, str] | None,
) -> frozenset[int]:
    """Return column indices whose PII type is already handled.

    Columns mapped to a real PII category (NAME, EMAIL, …) are fully
    processed by the column detector.  Columns mapped to ``skip`` or
    ``freetext`` still need LLM review, so they are *not* covered.
    Unmapped columns are also not covered.
    """
    if not mapped_columns:
        return frozenset()
    return frozenset(
        col for col, cat in mapped_columns.items()
        if cat.lower() not in _FREETEXT_CATEGORIES
    )


def _line_is_covered(line: Line, covered_cols: frozenset[int]) -> bool:
    """True when every non-empty cell belongs to a covered column."""
    for cell in line.cells:
        if not cell.text.strip():
            continue
        if cell.col_index not in covered_cols:
            return False
    return True


def _is_trivial(text: str) -> bool:
    """True when *text* is too short or generic to contain PII."""
    stripped = text.strip()
    if stripped.lower() in _TRIVIAL_VALUES:
        return True
    # Pure numeric (possibly with decimals, commas, or sign)
    if stripped.replace(",", "").replace(".", "").replace("-", "").isdigit():
        return True
    # Single character
    if len(stripped) <= 1:
        return True
    return False


def _all_cells_trivial(line: Line, covered_cols: frozenset[int]) -> bool:
    """True when every uncovered, non-empty cell has trivially non-PII content."""
    for cell in line.cells:
        if covered_cols and cell.col_index in covered_cols:
            continue
        text = cell.text.strip()
        if not text:
            continue
        if not _is_trivial(text):
            return False
    return True


def _prompt_text_for_line(
    line: Line,
    headers: list[str] | None,
    covered_cols: frozenset[int],
) -> str:
    """Build the text sent to the LLM for a single line.

    When *covered_cols* is non-empty, cells in those columns are omitted
    from the prompt — the column detector already handled them and
    including them wastes context tokens.  Column headers are still
    prefixed to each retained cell for CSV/XLSX data.
    """
    # Free text (single cell) or no header info — send as-is.
    if not headers or len(line.cells) <= 1:
        return line.text

    parts: list[str] = []
    for cell in line.cells:
        if covered_cols and cell.col_index in covered_cols:
            continue
        if not cell.text.strip():
            continue
        idx = cell.col_index
        header = headers[idx] if idx < len(headers) else ""
        if header:
            parts.append(f"{header}: {cell.text}")
        else:
            parts.append(cell.text)
    return " | ".join(parts) if parts else line.text


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
