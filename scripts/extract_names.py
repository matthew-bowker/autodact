#!/usr/bin/env python3
"""Extract name strings from names-dataset into flat text files.

This is a BUILD-TIME script.  It generates the text files that the
NameDictionaryDetector loads at runtime.  The ``names-dataset``,
``english-words``, and ``pycountry`` packages are NOT required at
runtime — only the output text files are.

Usage:
    python scripts/extract_names.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Build-time dependencies only.
try:
    from names_dataset import NameDataset
except ImportError:
    print("Install names-dataset first:  pip install names-dataset")
    sys.exit(1)

try:
    from english_words import get_english_words_set
except ImportError:
    print("Install english-words first:  pip install english-words")
    sys.exit(1)

try:
    import pycountry
except ImportError:
    print("Install pycountry first:  pip install pycountry")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "pipeline" / "data"


def _country_words() -> set[str]:
    """Collect country names, common names, and demonyms from pycountry."""
    words: set[str] = set()
    for country in pycountry.countries:
        # Full name and common name
        for attr in ("name", "common_name", "official_name"):
            val = getattr(country, attr, None)
            if val:
                for part in val.replace(",", " ").replace("(", " ").replace(")", " ").split():
                    if len(part) >= 3:
                        words.add(part.lower())
    # Historic country names
    for hc in pycountry.historic_countries:
        for attr in ("name", "common_name"):
            val = getattr(hc, attr, None)
            if val:
                for part in val.replace(",", " ").split():
                    if len(part) >= 3:
                        words.add(part.lower())
    return words


def extract() -> None:
    print("Loading names-dataset (this takes ~10 s and ~1.7 GB RAM) …")
    nd = NameDataset()

    first_names: set[str] = set()
    last_names: set[str] = set()

    for name in nd.first_names:
        # Only keep names with >= 3 characters to reduce noise.
        if len(name) >= 3:
            first_names.add(name)

    for name in nd.last_names:
        if len(name) >= 3:
            last_names.add(name)

    # --- Build the common-words exclusion set ---------------------------------
    print("Loading English dictionaries …")

    # Start with the broadest English dictionary combination.
    common_words: set[str] = set()
    for source in ["web2", "gcide"]:
        common_words |= get_english_words_set([source], lower=True)
        common_words |= get_english_words_set([source], lower=True, alpha=True)

    # Add country/territory names from pycountry.
    country_words = _country_words()
    common_words |= country_words
    print(f"  Added {len(country_words):,} country-derived words")

    # --- Write output files ---------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    first_path = OUTPUT_DIR / "first_names.txt"
    last_path = OUTPUT_DIR / "last_names.txt"
    common_path = OUTPUT_DIR / "common_words.txt"

    first_path.write_text("\n".join(sorted(first_names)) + "\n", encoding="utf-8")
    last_path.write_text("\n".join(sorted(last_names)) + "\n", encoding="utf-8")
    common_path.write_text("\n".join(sorted(common_words)) + "\n", encoding="utf-8")

    print(f"First names:  {len(first_names):>10,} → {first_path}")
    print(f"Last names:   {len(last_names):>10,} → {last_path}")
    print(f"Common words: {len(common_words):>10,} → {common_path}")

    first_mb = first_path.stat().st_size / 1024 / 1024
    last_mb = last_path.stat().st_size / 1024 / 1024
    common_mb = common_path.stat().st_size / 1024 / 1024
    print(f"Disk: {first_mb:.1f} + {last_mb:.1f} + {common_mb:.1f} = "
          f"{first_mb + last_mb + common_mb:.1f} MB")


if __name__ == "__main__":
    extract()
