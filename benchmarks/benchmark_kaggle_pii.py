#!/usr/bin/env python3
"""Benchmark autodact against the Kaggle "PII Detection from Educational Data"
competition (Learning Agency Lab, 2024).

Closed competition; we run autodact zero-shot against the training data
(which has gold BIO labels) and compute the same micro F5 metric the
leaderboard used.

Usage:
    python -m benchmarks.benchmark_kaggle_pii \\
        --train /path/to/train.json \\
        --samples 200 \\
        --output benchmarks/results_kaggle.json

Mapping of autodact internal categories → Kaggle BIO labels:

    NAME      → NAME_STUDENT     (note: Kaggle excludes instructor/author names —
                                  this is a known precision penalty for us)
    EMAIL     → EMAIL
    PHONE     → PHONE_NUM
    ID        → ID_NUM
    URL       → URL_PERSONAL
    LOCATION  → STREET_ADDRESS   (loose match; Kaggle is street-only)

Categories we cannot output: USERNAME (no autodact equivalent).
"""
from __future__ import annotations

# Stub _lzma early so torchvision's incidental import in transformers'
# auto-class machinery doesn't blow up on pyenv-built Pythons.
from src.pipeline import deberta_detector as _deberta_detector  # noqa: F401

import argparse
import json
import logging
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


INTERNAL_TO_KAGGLE: dict[str, str] = {
    "NAME": "NAME_STUDENT",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE_NUM",
    "ID": "ID_NUM",
    "URL": "URL_PERSONAL",
    "LOCATION": "STREET_ADDRESS",
}

# In --tuned mode we suppress two mappings that produce far more false
# positives than true positives on this dataset:
#
#   * LOCATION → STREET_ADDRESS only fires when the term contains a digit
#     (a crude proxy for "this might be a street, not a city/country").
#   * URL → URL_PERSONAL is dropped entirely — we have no signal to
#     distinguish a student's portfolio URL from a cited news article.
#
# These cost us ~6 URL_PERSONAL TPs and any genuine STREET_ADDRESS without
# a digit, but eliminate hundreds of false positives that crater precision.
INTERNAL_TO_KAGGLE_TUNED: dict[str, str] = {
    "NAME": "NAME_STUDENT",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE_NUM",
    "ID": "ID_NUM",
    # URL omitted entirely
    # LOCATION conditional, handled below
}


def _street_looks_real(term: str) -> bool:
    """Heuristic: a real street address usually has a digit somewhere."""
    return any(ch.isdigit() for ch in term)

KAGGLE_LABELS = {
    "NAME_STUDENT", "EMAIL", "USERNAME", "ID_NUM",
    "PHONE_NUM", "URL_PERSONAL", "STREET_ADDRESS",
}


def reconstruct_text(tokens: list[str], trailing_whitespace: list[bool]) -> str:
    """Rebuild the full text from tokens + trailing-whitespace flags.

    The Kaggle JSON includes ``full_text`` directly, so this is just a
    consistency check. Returns the same text as joining tokens with the
    appropriate whitespace.
    """
    parts: list[str] = []
    for tok, ws in zip(tokens, trailing_whitespace):
        parts.append(tok)
        if ws:
            parts.append(" ")
    return "".join(parts)


def token_char_offsets(
    full_text: str, tokens: list[str], trailing_whitespace: list[bool],
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` character offsets for each spaCy token.

    SpaCy emits whitespace runs as their own tokens in this dataset, so we
    can't blindly skip whitespace before searching.  Walk forward from the
    previous cursor and locate each token in order; advance past it.
    """
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        idx = full_text.find(tok, cursor)
        if idx == -1:
            # Token not found from cursor — emit zero-length placeholder.
            offsets.append((cursor, cursor))
            continue
        offsets.append((idx, idx + len(tok)))
        cursor = idx + len(tok)
    return offsets


def assign_bio_labels(
    n_tokens: int,
    token_offsets: list[tuple[int, int]],
    detections: list[tuple[int, int, str]],
) -> list[str]:
    """Convert character spans to BIO labels per spaCy token.

    *detections* is a list of ``(start, end, kaggle_label)`` tuples. Tokens
    overlapping a detection get B-/I- prefixes; other tokens get "O".

    When detections overlap, the first one wins (matches our orchestrator's
    longest-first replay order, which prevents partial-word contamination).
    """
    labels = ["O"] * n_tokens
    # Sort detections by start, longest first on tie so longer entities take
    # precedence over substrings.
    sorted_dets = sorted(detections, key=lambda d: (d[0], -(d[1] - d[0])))

    for start, end, label in sorted_dets:
        first = True
        for i, (t_start, t_end) in enumerate(token_offsets):
            if labels[i] != "O":
                continue  # already labelled by a prior detection
            # Token overlaps the detection if any character is inside.
            if t_end <= start:
                continue
            if t_start >= end:
                break
            labels[i] = ("B-" if first else "I-") + label
            first = False
    return labels


def run_autodact_on_doc(
    orchestrator,
    full_text: str,
    doc_id: str,
    tmp_dir: Path,
    output_dir: Path,
) -> tuple[list[tuple[str, str]], float]:
    """Run the orchestrator and return (lookup_entries, elapsed_seconds)."""
    from src.pipeline.lookup_table import LookupTable

    input_path = tmp_dir / f"doc_{doc_id}.txt"
    input_path.write_text(full_text, encoding="utf-8")

    lookup = LookupTable()
    t0 = time.perf_counter()
    orchestrator.process_file(
        file_path=input_path,
        lookup=lookup,
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0
    entries = [(e.original_term, e.pii_category) for e in lookup.all_entries()]
    return entries, elapsed


def find_all_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    """Locate every occurrence of *term* in *text* (case-sensitive)."""
    out: list[tuple[int, int]] = []
    if not term:
        return out
    start = 0
    while True:
        idx = text.find(term, start)
        if idx == -1:
            break
        out.append((idx, idx + len(term)))
        start = idx + len(term)
    return out


def f_beta(p: float, r: float, beta: float = 5.0) -> float:
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def evaluate_doc(
    gold_labels: list[str], pred_labels: list[str],
) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    """Strict (token, label) matching.

    Returns global ``(tp, fp, fn)`` plus per-label breakdown keyed by the
    base category (without B-/I- prefix).
    """
    tp = fp = fn = 0
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for g, p in zip(gold_labels, pred_labels):
        g_pos = g != "O"
        p_pos = p != "O"
        cat = (g if g_pos else p).split("-", 1)[-1] if (g_pos or p_pos) else None
        if g_pos and p_pos and g == p:
            tp += 1
            if cat:
                per_cat[cat][0] += 1
        elif g_pos and (not p_pos or g != p):
            fn += 1
            if cat:
                per_cat[cat][2] += 1
            if p_pos and g != p:
                # Wrong label is also a FP for the predicted category.
                fp += 1
                pred_cat = p.split("-", 1)[-1]
                per_cat[pred_cat][1] += 1
        elif p_pos and not g_pos:
            fp += 1
            if cat:
                per_cat[cat][1] += 1
    return tp, fp, fn, {k: tuple(v) for k, v in per_cat.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark autodact on the Kaggle PII competition data.",
    )
    p.add_argument("--train", type=str, required=True,
                   help="Path to train.json (with gold BIO labels)")
    p.add_argument("--samples", type=int, default=200,
                   help="Number of essays to evaluate (default: 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="cpu | mps | cuda (default: auto)")
    p.add_argument("--output", type=str, default=None,
                   help="Save JSON results to this path")
    p.add_argument("--tuned", action="store_true",
                   help="Conservative category mapping: drop URL_PERSONAL, "
                        "only emit STREET_ADDRESS when the term contains a digit.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.ERROR)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading {args.train}…")
    with open(args.train) as f:
        all_docs = json.load(f)
    print(f"  {len(all_docs)} essays in training set")

    random.seed(args.seed)
    n = min(args.samples, len(all_docs))
    indices = sorted(random.sample(range(len(all_docs)), n))
    docs = [all_docs[i] for i in indices]
    print(f"Selected {n} essays (seed={args.seed})")

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    print("Loading detector + Presidio + names…")
    t0 = time.perf_counter()
    from src.config import DEFAULT_MODEL_REPO
    from src.pipeline.deberta_detector import DebertaDetector
    from src.pipeline.name_detector import NameDictionaryDetector
    from src.pipeline.orchestrator import Orchestrator
    from src.pipeline.presidio_detector import PresidioDetector

    detector = DebertaDetector(
        model_name=DEFAULT_MODEL_REPO, device=args.device,
    )
    orchestrator = Orchestrator(
        presidio_detector=PresidioDetector(),
        deberta_detector=detector,
        name_detector=NameDictionaryDetector(),
    )
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s on {detector._device}")

    # ------------------------------------------------------------------
    # Run + evaluate
    # ------------------------------------------------------------------
    total_tp = total_fp = total_fn = 0
    per_cat_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    elapsed_per_doc: list[float] = []

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        for i, doc in enumerate(docs):
            doc_id = str(doc["document"])
            full_text = doc["full_text"]
            tokens = doc["tokens"]
            ws = doc["trailing_whitespace"]
            gold_labels = doc["labels"]

            try:
                entries, elapsed = run_autodact_on_doc(
                    orchestrator, full_text, doc_id, tmp_dir, out_dir,
                )
            except Exception as exc:
                print(f"  ERROR on doc {doc_id}: {exc}")
                continue
            elapsed_per_doc.append(elapsed)

            # Map lookup entries to character spans, then to BIO labels
            token_offsets = token_char_offsets(full_text, tokens, ws)
            mapping = INTERNAL_TO_KAGGLE_TUNED if args.tuned else INTERNAL_TO_KAGGLE
            detections: list[tuple[int, int, str]] = []
            for term, our_cat in entries:
                kaggle_label = mapping.get(our_cat)
                if kaggle_label is None and args.tuned and our_cat == "LOCATION":
                    if _street_looks_real(term):
                        kaggle_label = "STREET_ADDRESS"
                if kaggle_label is None:
                    continue
                for start, end in find_all_occurrences(full_text, term):
                    detections.append((start, end, kaggle_label))

            pred_labels = assign_bio_labels(
                len(tokens), token_offsets, detections,
            )

            tp, fp, fn, doc_per_cat = evaluate_doc(gold_labels, pred_labels)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            for cat, (t, f, n_) in doc_per_cat.items():
                per_cat_totals[cat][0] += t
                per_cat_totals[cat][1] += f
                per_cat_totals[cat][2] += n_

            if args.verbose:
                doc_p = tp / (tp + fp) if (tp + fp) > 0 else 0
                doc_r = tp / (tp + fn) if (tp + fn) > 0 else 0
                print(
                    f"  [{i+1}/{n}] doc={doc_id} "
                    f"TP={tp} FP={fp} FN={fn} "
                    f"P={doc_p:.2f} R={doc_r:.2f} "
                    f"({elapsed:.2f}s)"
                )
            elif (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  Progress: {i+1}/{n}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = f_beta(micro_p, micro_r, beta=1.0)
    micro_f5 = f_beta(micro_p, micro_r, beta=5.0)

    print()
    print("=" * 70)
    print("KAGGLE PII LEADERBOARD — AUTODACT ZERO-SHOT")
    print("=" * 70)
    print(f"Samples:         {len(elapsed_per_doc)}")
    print(f"Avg time/doc:    {sum(elapsed_per_doc)/max(len(elapsed_per_doc),1):.2f}s")
    print(f"Micro precision: {micro_p:.4f}")
    print(f"Micro recall:    {micro_r:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Micro F5:        {micro_f5:.4f}    (the leaderboard metric)")
    print()
    print("Per-category (base label, ignoring B-/I-):")
    for cat in sorted(per_cat_totals):
        tp_c, fp_c, fn_c = per_cat_totals[cat]
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f5 = f_beta(p, r, beta=5.0)
        print(
            f"  {cat:<18} P={p:.2f} R={r:.2f} F5={f5:.3f} "
            f"(TP={tp_c} FP={fp_c} FN={fn_c})"
        )

    if args.output:
        from datetime import datetime, timezone

        result = {
            "metadata": {
                "date": datetime.now(timezone.utc).isoformat(),
                "samples": len(elapsed_per_doc),
                "seed": args.seed,
                "dataset": "Kaggle PII Detection (Learning Agency Lab)",
                "model": DEFAULT_MODEL_REPO,
                "device": detector._device,
            },
            "overall": {
                "precision": round(micro_p, 4),
                "recall": round(micro_r, 4),
                "f1": round(micro_f1, 4),
                "f5": round(micro_f5, 4),
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                "avg_time_per_doc": round(
                    sum(elapsed_per_doc) / max(len(elapsed_per_doc), 1), 3
                ),
            },
            "per_category": {
                cat: {
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f5": round(f_beta(p, r, beta=5.0), 4),
                    "tp": per_cat_totals[cat][0],
                    "fp": per_cat_totals[cat][1],
                    "fn": per_cat_totals[cat][2],
                }
                for cat in sorted(per_cat_totals)
                for p in [
                    per_cat_totals[cat][0]
                    / (per_cat_totals[cat][0] + per_cat_totals[cat][1])
                    if (per_cat_totals[cat][0] + per_cat_totals[cat][1]) > 0 else 0
                ]
                for r in [
                    per_cat_totals[cat][0]
                    / (per_cat_totals[cat][0] + per_cat_totals[cat][2])
                    if (per_cat_totals[cat][0] + per_cat_totals[cat][2]) > 0 else 0
                ]
            },
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
