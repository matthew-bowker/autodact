import csv
from pathlib import Path

from src.pipeline.lookup_table import LookupEntry, LookupTable


def test_register_new_term():
    lt = LookupTable()
    tag = lt.register("Jane Smith", "NAME", "test.txt", 3)
    assert tag == "[NAME 1]"
    assert len(lt) == 1


def test_register_duplicate_returns_existing():
    lt = LookupTable()
    tag1 = lt.register("Jane Smith", "NAME", "test.txt", 3)
    tag2 = lt.register("Jane Smith", "NAME", "test.txt", 10)
    assert tag1 == tag2 == "[NAME 1]"
    assert len(lt) == 1


def test_register_case_insensitive():
    lt = LookupTable()
    tag1 = lt.register("Jane Smith", "NAME", "test.txt", 3)
    tag2 = lt.register("jane smith", "NAME", "test.txt", 5)
    assert tag1 == tag2


def test_register_independent_counters():
    lt = LookupTable()
    name_tag = lt.register("Jane", "NAME", "test.txt", 1)
    org_tag = lt.register("Acme", "ORG", "test.txt", 2)
    name_tag2 = lt.register("Bob", "NAME", "test.txt", 3)
    assert name_tag == "[NAME 1]"
    assert org_tag == "[ORG 1]"
    assert name_tag2 == "[NAME 2]"


def test_lookup_existing():
    lt = LookupTable()
    lt.register("Jane", "NAME", "test.txt", 1)
    assert lt.lookup("Jane") == "[NAME 1]"
    assert lt.lookup("jane") == "[NAME 1]"


def test_lookup_missing():
    lt = LookupTable()
    assert lt.lookup("Unknown") is None


def test_reset():
    lt = LookupTable()
    lt.register("Jane", "NAME", "test.txt", 1)
    lt.reset()
    assert len(lt) == 0
    tag = lt.register("Jane", "NAME", "test.txt", 1)
    assert tag == "[NAME 1]"


def test_export_csv(tmp_path: Path):
    lt = LookupTable()
    lt.register("Jane Smith", "NAME", "test.txt", 3)
    lt.register("jane.smith@acme.com", "EMAIL", "test.txt", 12)
    out = tmp_path / "lookup.csv"
    lt.export_csv(out)
    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == [
        "original_term", "anonymised_term", "pii_category",
        "source_file", "first_seen_line",
    ]
    assert len(rows) == 3
    assert rows[1][0] == "Jane Smith"
    assert rows[1][1] == "[NAME 1]"


def test_all_entries():
    lt = LookupTable()
    lt.register("Jane", "NAME", "test.txt", 1)
    lt.register("Acme", "ORG", "test.txt", 2)
    entries = lt.all_entries()
    assert len(entries) == 2
    assert all(isinstance(e, LookupEntry) for e in entries)


def test_remove():
    lt = LookupTable()
    lt.register("Jane", "NAME", "test.txt", 1)
    lt.remove("Jane")
    assert len(lt) == 0
    assert lt.lookup("Jane") is None


def test_category_counts():
    lt = LookupTable()
    lt.register("Jane", "NAME", "test.txt", 1)
    lt.register("Bob", "NAME", "test.txt", 2)
    lt.register("Acme", "ORG", "test.txt", 3)
    counts = lt.category_counts()
    assert counts == {"NAME": 2, "ORG": 1}


# ── Fuzzy key normalization ────────────────────────────────────────


def test_normalize_apostrophe():
    lt = LookupTable()
    tag1 = lt.register("O'Connor", "NAME", "test.txt", 1)
    tag2 = lt.register("OConnor", "NAME", "test.txt", 2)
    assert tag1 == tag2
    assert len(lt) == 1
    # Original preserves first-seen form.
    assert lt.all_entries()[0].original_term == "O'Connor"


def test_normalize_hyphen():
    lt = LookupTable()
    tag1 = lt.register("Al-Hassan", "NAME", "test.txt", 1)
    tag2 = lt.register("Al Hassan", "NAME", "test.txt", 2)
    assert tag1 == tag2
    assert len(lt) == 1


def test_normalize_whitespace():
    lt = LookupTable()
    tag1 = lt.register("John  Smith", "NAME", "test.txt", 1)
    tag2 = lt.register("John Smith", "NAME", "test.txt", 2)
    assert tag1 == tag2


def test_lookup_with_different_punctuation():
    lt = LookupTable()
    lt.register("O'Brien", "NAME", "test.txt", 1)
    assert lt.lookup("OBrien") == "[NAME 1]"
    assert lt.lookup("O'Brien") == "[NAME 1]"


def test_remove_with_different_punctuation():
    lt = LookupTable()
    lt.register("O'Malley", "NAME", "test.txt", 1)
    lt.remove("OMalley")
    assert len(lt) == 0


def test_emails_not_collapsed():
    """Periods are NOT stripped — different emails stay separate."""
    lt = LookupTable()
    tag1 = lt.register("jane.smith@acme.com", "EMAIL", "test.txt", 1)
    tag2 = lt.register("janesmith@acme.com", "EMAIL", "test.txt", 2)
    assert tag1 != tag2
    assert len(lt) == 2
