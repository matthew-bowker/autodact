from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class Cell:
    text: str
    sheet_name: str | None = None
    row_index: int = 0
    col_index: int = 0


_WORD_CHAR_RE = re.compile(r"\w")


@lru_cache(maxsize=512)
def _compile_safe_pattern(escaped: str) -> re.Pattern[str]:
    """Compile and cache the bracket-aware replacement regex."""
    return re.compile(r"\[[^\]]*\]|(" + escaped + ")")


def _safe_replace(text: str, original: str, replacement: str) -> str:
    """Replace *original* with *replacement*, skipping text inside [brackets].

    This prevents ``replace_all("NAME", "[JOBTITLE 3]")`` from corrupting
    an existing ``[NAME 1]`` tag into ``[[JOBTITLE 3] 1]``.

    Word boundaries are added when the original starts/ends with a word
    character to prevent partial-word contamination (e.g. "Tunis" matching
    inside "Tunisie").
    """
    escaped = re.escape(original)
    if original and _WORD_CHAR_RE.match(original[0]):
        escaped = r"\b" + escaped
    if original and _WORD_CHAR_RE.match(original[-1]):
        escaped = escaped + r"\b"
    pattern = _compile_safe_pattern(escaped)
    return pattern.sub(
        lambda m: replacement if m.group(1) else m.group(0),
        text,
    )


@dataclass
class Line:
    cells: list[Cell]
    line_number: int
    is_subline: bool = False

    @property
    def text(self) -> str:
        return " | ".join(c.text for c in self.cells)

    def replace_all(self, original: str, replacement: str) -> None:
        for cell in self.cells:
            cell.text = _safe_replace(cell.text, original, replacement)


@dataclass
class Document:
    lines: list[Line]
    source_path: Path
    source_format: str
    headers: list[str] | None = None

    def replace_all(self, original: str, replacement: str) -> None:
        for line in self.lines:
            line.replace_all(original, replacement)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "source_path": str(self.source_path),
            "source_format": self.source_format,
            "headers": self.headers,
            "lines": [
                {
                    "line_number": line.line_number,
                    "is_subline": line.is_subline,
                    "cells": [
                        {
                            "text": cell.text,
                            "sheet_name": cell.sheet_name,
                            "row_index": cell.row_index,
                            "col_index": cell.col_index,
                        }
                        for cell in line.cells
                    ],
                }
                for line in self.lines
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        """Deserialize from dictionary."""
        lines = [
            Line(
                line_number=line_data["line_number"],
                is_subline=line_data["is_subline"],
                cells=[
                    Cell(
                        text=cell_data["text"],
                        sheet_name=cell_data.get("sheet_name"),
                        row_index=cell_data.get("row_index", 0),
                        col_index=cell_data.get("col_index", 0),
                    )
                    for cell_data in line_data["cells"]
                ],
            )
            for line_data in data["lines"]
        ]
        return cls(
            lines=lines,
            source_path=Path(data["source_path"]),
            source_format=data["source_format"],
            headers=data.get("headers"),
        )


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_long_lines(doc: Document, max_chars: int = 500) -> None:
    new_lines: list[Line] = []
    for line in doc.lines:
        if len(line.text) <= max_chars:
            new_lines.append(line)
            continue
        if len(line.cells) > 1:
            chunks = _chunk_cells(line.cells, max_chars)
        else:
            chunks = _split_at_sentences(line.cells[0].text, max_chars)
        for i, chunk in enumerate(chunks):
            new_lines.append(Line(cells=chunk, line_number=line.line_number, is_subline=(i > 0)))
    doc.lines = new_lines


def reunify_sublines(doc: Document) -> None:
    if not doc.lines:
        return
    merged: list[Line] = []
    for line in doc.lines:
        if line.is_subline and merged and merged[-1].line_number == line.line_number:
            merged[-1].cells.extend(line.cells)
        else:
            line.is_subline = False
            merged.append(line)
    doc.lines = merged


def _chunk_cells(cells: list[Cell], max_chars: int) -> list[list[Cell]]:
    chunks: list[list[Cell]] = []
    current_chunk: list[Cell] = []
    current_len = 0
    for cell in cells:
        cell_len = len(cell.text)
        separator_len = 3 if current_chunk else 0  # " | "
        if current_chunk and current_len + separator_len + cell_len > max_chars:
            chunks.append(current_chunk)
            current_chunk = [cell]
            current_len = cell_len
        else:
            current_chunk.append(cell)
            current_len += separator_len + cell_len
    if current_chunk:
        chunks.append(current_chunk)

    # If any chunk is a single oversized cell, sentence-split it
    final: list[list[Cell]] = []
    for chunk in chunks:
        chunk_len = sum(len(c.text) for c in chunk) + 3 * (len(chunk) - 1)
        if chunk_len > max_chars and len(chunk) == 1:
            final.extend(_split_at_sentences(chunk[0].text, max_chars))
        else:
            final.append(chunk)
    return final


def _split_at_sentences(text: str, max_chars: int) -> list[list[Cell]]:
    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks: list[list[Cell]] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append([Cell(text=current)])
            current = sentence
        else:
            current = f"{current} {sentence}".strip() if current else sentence
    if current:
        chunks.append([Cell(text=current)])

    # If any chunk is still too long, hard-split at last space before max_chars
    final: list[list[Cell]] = []
    for chunk in chunks:
        text = chunk[0].text
        while len(text) > max_chars:
            split_pos = text.rfind(" ", 0, max_chars)
            if split_pos == -1:
                split_pos = max_chars
            final.append([Cell(text=text[:split_pos])])
            text = text[split_pos:].lstrip()
        if text:
            final.append([Cell(text=text)])
    return final
