from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


_APOSTROPHE_RE = re.compile(r"['\u2019]")  # straight + curly
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_key(s: str) -> str:
    """Normalize a lookup key so punctuation variants share one entry.

    - Lowercases
    - Removes apostrophes  (O'Connor → OConnor → oconnor)
    - Replaces hyphens with spaces  (Al-Hassan → Al Hassan → al hassan)
    - Collapses whitespace
    """
    s = s.lower()
    s = _APOSTROPHE_RE.sub("", s)
    s = s.replace("-", " ")
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


@dataclass
class LookupEntry:
    original_term: str
    anonymised_term: str
    pii_category: str
    source_file: str
    first_seen_line: int


class LookupTable:
    def __init__(self) -> None:
        self._entries: dict[str, LookupEntry] = {}
        self._counters: dict[str, int] = defaultdict(int)

    def register(
        self,
        original: str,
        category: str,
        source_file: str,
        line_number: int,
    ) -> str:
        key = _normalize_key(original)
        if key in self._entries:
            return self._entries[key].anonymised_term
        self._counters[category] += 1
        tag = f"[{category} {self._counters[category]}]"
        self._entries[key] = LookupEntry(
            original_term=original,
            anonymised_term=tag,
            pii_category=category,
            source_file=source_file,
            first_seen_line=line_number,
        )
        return tag

    def register_alias(
        self,
        original: str,
        existing_tag: str,
        category: str,
        source_file: str,
        line_number: int,
    ) -> str:
        """Register *original* as an alias pointing to an existing tag."""
        key = _normalize_key(original)
        if key in self._entries:
            return self._entries[key].anonymised_term
        self._entries[key] = LookupEntry(
            original_term=original,
            anonymised_term=existing_tag,
            pii_category=category,
            source_file=source_file,
            first_seen_line=line_number,
        )
        return existing_tag

    def lookup(self, original: str) -> str | None:
        entry = self._entries.get(_normalize_key(original))
        return entry.anonymised_term if entry else None

    def reset(self) -> None:
        self._entries.clear()
        self._counters.clear()

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "original_term",
                "anonymised_term",
                "pii_category",
                "source_file",
                "first_seen_line",
            ])
            for entry in self._entries.values():
                writer.writerow([
                    entry.original_term,
                    entry.anonymised_term,
                    entry.pii_category,
                    entry.source_file,
                    entry.first_seen_line,
                ])

    def all_entries(self) -> list[LookupEntry]:
        return list(self._entries.values())

    def remove(self, original: str) -> None:
        key = _normalize_key(original)
        self._entries.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entry in self._entries.values():
            counts[entry.pii_category] += 1
        return dict(counts)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "entries": {
                key: {
                    "original_term": entry.original_term,
                    "anonymised_term": entry.anonymised_term,
                    "pii_category": entry.pii_category,
                    "source_file": entry.source_file,
                    "first_seen_line": entry.first_seen_line,
                }
                for key, entry in self._entries.items()
            },
            "counters": dict(self._counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LookupTable:
        """Deserialize from dictionary."""
        table = cls()
        table._entries = {
            key: LookupEntry(
                original_term=entry_data["original_term"],
                anonymised_term=entry_data["anonymised_term"],
                pii_category=entry_data["pii_category"],
                source_file=entry_data["source_file"],
                first_seen_line=entry_data["first_seen_line"],
            )
            for key, entry_data in data["entries"].items()
        }
        table._counters = defaultdict(int, data["counters"])
        return table
