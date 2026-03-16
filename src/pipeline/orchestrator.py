from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.pipeline.column_detector import ColumnDetector
from src.pipeline.document import Document, reunify_sublines, split_long_lines
from src.pipeline.llm_detector import LLMDetector
from src.pipeline.lookup_table import LookupTable
from src.pipeline.name_detector import NameDictionaryDetector
from src.pipeline.parsers import parse_file
from src.pipeline.post_validator import validate_and_clean
from src.pipeline.presidio_detector import PresidioDetector
from src.pipeline.writers import write_file


@dataclass
class PipelineResult:
    document: Document
    lookup_table: LookupTable
    flagged_lines: list[int]
    output_path: Path
    lookup_path: Path


class Orchestrator:
    def __init__(
        self,
        presidio_detector: PresidioDetector,
        llm_detector: LLMDetector,
        name_detector: NameDictionaryDetector | None = None,
        max_line_chars: int = 500,
    ) -> None:
        self._presidio = presidio_detector
        self._llm = llm_detector
        self._names = name_detector
        self._max_line_chars = max_line_chars

    def process_file(
        self,
        file_path: Path,
        lookup: LookupTable,
        output_dir: Path,
        preserve_format: bool = True,
        lookup_filename: str | None = None,
        column_mapping: dict[int, str] | None = None,
        on_column_progress: Callable[[int, int], None] | None = None,
        on_presidio_progress: Callable[[int, int], None] | None = None,
        on_names_progress: Callable[[int, int], None] | None = None,
        on_llm_progress: Callable[[int, int], None] | None = None,
    ) -> PipelineResult:
        doc = parse_file(file_path)

        # Pre-pass: tag emails before column detector so that
        # name-derived variations don't corrupt email addresses.
        self._presidio.pre_detect_emails(doc, lookup)

        # Pre-pass: tag company names by corporate suffix before other
        # layers can misclassify them (e.g. spaCy tagging "Smith Inc" as PERSON).
        self._presidio.pre_detect_orgs(doc, lookup)

        # Layer 1: Column-based detection (CSV/XLSX only)
        if column_mapping:
            ColumnDetector().process(
                doc, lookup, column_mapping, on_progress=on_column_progress,
            )

        # Pre-pass: tag structured patterns (phones, NI numbers, SSNs,
        # postcodes, IPs, URLs) before Presidio can misclassify them.
        self._presidio.pre_detect_patterns(doc, lookup)

        # Layer 2: Presidio NER + regex (on all remaining text)
        self._presidio.process(doc, lookup, on_progress=on_presidio_progress)

        # Layer 3: Name dictionary (catches names spaCy missed)
        if self._names:
            self._names.process(doc, lookup, on_progress=on_names_progress)

        # Re-parse the file so the LLM sees full unredacted content.
        # This replaces the previous in-memory snapshot approach — the
        # file is already on disk, re-reading it is far cheaper than
        # holding a duplicate of every cell text in RAM.
        doc = parse_file(file_path)

        # Split long lines before LLM pass
        split_long_lines(doc, max_chars=self._max_line_chars)

        # Layer 4: LLM pass (contextual detection on sub-lines)
        flagged = self._llm.process(
            doc, lookup,
            on_progress=on_llm_progress,
            mapped_columns=column_mapping,
        )

        # Reunify sub-lines for output
        reunify_sublines(doc)

        # Re-parse once more for a clean document, then apply ALL
        # findings additively.  This reconciles detections from every
        # layer (rule-based, Presidio, name dictionary, LLM) on the
        # pristine original text.
        doc = parse_file(file_path)
        _apply_all_entries(doc, lookup)

        # Post-processing: remove obvious false positives
        validate_and_clean(doc, lookup)

        # Write output
        suffix = file_path.suffix if preserve_format else ".txt"
        output_path = output_dir / f"{file_path.stem}_anon{suffix}"
        write_file(doc, output_path, preserve_format)

        # Write lookup CSV
        csv_name = lookup_filename or f"{file_path.stem}_lookup.csv"
        lookup_path = output_dir / csv_name
        lookup.export_csv(lookup_path)

        return PipelineResult(
            document=doc,
            lookup_table=lookup,
            flagged_lines=flagged,
            output_path=output_path,
            lookup_path=lookup_path,
        )


def _apply_all_entries(doc: Document, lookup: LookupTable) -> None:
    """Apply every lookup entry to *doc*, longest originals first.

    Replacing longer terms first prevents partial-word contamination
    (e.g. "Jane Smith" must be replaced before the alias "Jane").
    """
    entries = lookup.all_entries()
    entries.sort(key=lambda e: len(e.original_term), reverse=True)
    for entry in entries:
        doc.replace_all(entry.original_term, entry.anonymised_term)
