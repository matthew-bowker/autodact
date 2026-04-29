"""Unit tests for the DeBERTa detector's span-merging and label-mapping
logic. These exercise the parts of ``DebertaDetector`` that do not require
torch/transformers to be installed.
"""
from __future__ import annotations

from src.pipeline.deberta_detector import (
    _LABEL_TO_CATEGORY,
    DebertaDetector,
    _spans_to_pairs,
)


class _FakeDetector:
    """Minimal stand-in exposing just the attributes ``_merge_chunk_spans``
    needs, so we can unit-test it without loading a real model."""

    def __init__(
        self,
        id2label: dict[int, str],
        confidence: float = 0.5,
    ) -> None:
        self._id2label = id2label
        self._confidence = confidence

    _merge_chunk_spans = DebertaDetector._merge_chunk_spans


def test_label_map_covers_piranha_schema():
    expected = {
        "GIVENNAME", "SURNAME", "EMAIL", "TELEPHONENUM",
        "DATEOFBIRTH", "CITY", "STREET", "ZIPCODE",
        "SOCIALNUM", "CREDITCARDNUMBER",
    }
    missing = expected - _LABEL_TO_CATEGORY.keys()
    assert not missing, f"Missing Piranha labels: {missing}"


def test_merge_consecutive_io_tokens_into_one_span():
    # Piranha-style I-only labels: two adjacent I-GIVENNAME tokens for "Jane Smith"
    det = _FakeDetector(id2label={0: "O", 1: "I-GIVENNAME"})
    out: list[tuple[int, int, str]] = []
    # Mimic [CLS] Jane Smith [SEP] with offsets like (0,0)(0,4)(4,10)(0,0)
    det._merge_chunk_spans(
        preds=[0, 1, 1, 0],
        scores=[0.9, 0.99, 0.99, 0.9],
        offsets=[[0, 0], [0, 4], [4, 10], [0, 0]],
        out=out,
    )
    assert out == [(0, 10, "NAME")]


def test_b_tag_starts_a_new_entity():
    # BIO scheme: "Jane" (B-) "Smith" (I-) but second name "Bob" (B-) starts new
    det = _FakeDetector(id2label={
        0: "O", 1: "B-GIVENNAME", 2: "I-GIVENNAME",
    })
    out: list[tuple[int, int, str]] = []
    det._merge_chunk_spans(
        preds=[0, 1, 2, 0, 1, 0],
        scores=[0.9] * 6,
        offsets=[[0, 0], [0, 4], [4, 10], [10, 15], [15, 18], [0, 0]],
        out=out,
    )
    assert out == [(0, 10, "NAME"), (15, 18, "NAME")]


def test_low_confidence_breaks_span():
    det = _FakeDetector(
        id2label={0: "O", 1: "I-GIVENNAME"}, confidence=0.5,
    )
    out: list[tuple[int, int, str]] = []
    # Middle token has score 0.3 → treated as O, splitting the span
    det._merge_chunk_spans(
        preds=[1, 1, 1],
        scores=[0.9, 0.3, 0.9],
        offsets=[[0, 4], [4, 5], [5, 10]],
        out=out,
    )
    assert out == [(0, 4, "NAME"), (5, 10, "NAME")]


def test_unmapped_label_is_dropped():
    det = _FakeDetector(id2label={0: "O", 1: "I-WEIRD_LABEL"})
    out: list[tuple[int, int, str]] = []
    det._merge_chunk_spans(
        preds=[1, 1],
        scores=[0.99, 0.99],
        offsets=[[0, 4], [4, 10]],
        out=out,
    )
    assert out == []


def test_category_changes_flush_span():
    det = _FakeDetector(id2label={
        0: "O", 1: "I-GIVENNAME", 2: "I-CITY",
    })
    out: list[tuple[int, int, str]] = []
    det._merge_chunk_spans(
        preds=[1, 1, 2],
        scores=[0.9, 0.9, 0.9],
        offsets=[[0, 4], [4, 10], [10, 18]],
        out=out,
    )
    assert out == [(0, 10, "NAME"), (10, 18, "LOCATION")]


def test_spans_to_pairs_dedups_and_trims():
    text = " Jane Smith   "
    spans = [(0, 11, "NAME"), (0, 11, "NAME"), (6, 11, "NAME")]
    pairs = _spans_to_pairs(text, spans)
    # First two are duplicates → collapsed; third has different offsets so
    # it stays; whitespace is trimmed.
    assert pairs == [("Jane Smith", "NAME"), ("Smith", "NAME")]


def test_special_tokens_close_open_span():
    det = _FakeDetector(id2label={0: "O", 1: "I-GIVENNAME"})
    out: list[tuple[int, int, str]] = []
    # CLS, name, SEP, name, SEP — sep should split the two names
    det._merge_chunk_spans(
        preds=[0, 1, 0, 1, 0],
        scores=[0.9] * 5,
        offsets=[[0, 0], [0, 4], [0, 0], [5, 10], [0, 0]],
        out=out,
    )
    assert out == [(0, 4, "NAME"), (5, 10, "NAME")]
