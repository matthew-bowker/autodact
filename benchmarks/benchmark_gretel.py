#!/usr/bin/env python3
"""Benchmark the autodact PII pipeline against the Gretel PII dataset.

Runs the full pipeline (Presidio + Name dictionary + LLM) on samples from
the gretelai/gretel-pii-masking-en-v1 test split, comparing detected PII
against ground-truth annotations.

Uses huggingface_hub + pyarrow (already app dependencies) to fetch the
dataset — no extra installs required.

Usage:
    python -m benchmarks.benchmark_gretel [OPTIONS]

    --samples N       Number of test rows to evaluate (default: 100)
    --models MODEL    "standard", "fast", or "both" (default: "both")
    --seed SEED       Random seed for sample selection (default: 42)
    --verbose         Print per-sample details
    --output PATH     Save JSON results to file
    --relaxed         Category-relaxed matching (text only, ignore category)
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Gretel → internal category mapping
# ---------------------------------------------------------------------------

GRETEL_TO_INTERNAL: dict[str, str] = {
    # Names
    "name": "NAME",
    "first_name": "NAME",
    "last_name": "NAME",
    "user_name": "NAME",
    # Contact
    "email": "EMAIL",
    "phone_number": "PHONE",
    # Identifiers
    "ssn": "ID",
    "account_number": "ID",
    "credit_card_number": "ID",
    "employee_id": "ID",
    "customer_id": "ID",
    "unique_identifier": "ID",
    "medical_record_number": "ID",
    "health_plan_beneficiary_number": "ID",
    "certificate_license_number": "ID",
    "national_id": "ID",
    "tax_id": "ID",
    "license_plate": "ID",
    "vehicle_identifier": "ID",
    "device_identifier": "ID",
    "biometric_identifier": "ID",
    "api_key": "ID",
    "password": "ID",
    "swift_bic": "ID",
    "bank_routing_number": "ID",
    "cvv": "ID",
    "pin": "ID",
    "account_id": "ID",
    # Location
    "address": "LOCATION",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "coordinate": "LOCATION",
    "zip_code": "POSTCODE",
    "postcode": "POSTCODE",
    # Organisation
    "company_name": "ORG",
    # Job
    "job_title": "JOBTITLE",
    "job_area": "JOBTITLE",
    # Dates
    "date_of_birth": "DOB",
    "date": "DOB",
    "date_time": "DOB",
    "time": "DOB",
    # Network
    "ipv4": "IP",
    "ipv6": "IP",
    "url": "URL",
}

MIN_ENTITY_LEN = 3  # skip ground-truth entities shorter than this


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    sample_id: str
    elapsed: float
    n_gt: int
    n_det: int
    true_positives: int
    false_positives: int
    false_negatives: int
    category_tp: dict[str, int] = field(default_factory=dict)
    category_fp: dict[str, int] = field(default_factory=dict)
    category_fn: dict[str, int] = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    model_name: str
    total_samples: int
    total_time: float
    avg_time: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_gretel_entities(entities_raw: str | list) -> list[tuple[str, str]]:
    """Parse Gretel entities into (text, internal_category) pairs.

    Skips entities whose Gretel type has no internal mapping and entities
    shorter than MIN_ENTITY_LEN.
    """
    if isinstance(entities_raw, str):
        # Dataset uses Python literal syntax (single quotes), not JSON.
        raw = ast.literal_eval(entities_raw)
    else:
        raw = entities_raw

    results: list[tuple[str, str]] = []
    for item in raw:
        entity_text = item["entity"]
        if len(entity_text.strip()) < MIN_ENTITY_LEN:
            continue
        for gretel_type in item["types"]:
            internal_cat = GRETEL_TO_INTERNAL.get(gretel_type)
            if internal_cat:
                results.append((entity_text.strip(), internal_cat))
                break  # use first matching type
    return results


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def text_match(gt_text: str, det_text: str) -> bool:
    """Case-insensitive bidirectional substring match."""
    gt = gt_text.lower().strip()
    det = det_text.lower().strip()
    return gt in det or det in gt


def evaluate_sample(
    gt_entities: list[tuple[str, str]],
    det_entries: list,
    relaxed: bool = False,
) -> SampleResult:
    """Compare ground truth against pipeline detections for one sample."""
    det_entities = [(e.original_term, e.pii_category) for e in det_entries]

    gt_matched = [False] * len(gt_entities)
    det_matched = [False] * len(det_entities)

    # Greedy matching
    for gi, (gt_text, gt_cat) in enumerate(gt_entities):
        for di, (det_text, det_cat) in enumerate(det_entities):
            if det_matched[di]:
                continue
            cat_ok = relaxed or gt_cat == det_cat
            if cat_ok and text_match(gt_text, det_text):
                gt_matched[gi] = True
                det_matched[di] = True
                break

    cat_tp: dict[str, int] = defaultdict(int)
    cat_fp: dict[str, int] = defaultdict(int)
    cat_fn: dict[str, int] = defaultdict(int)

    for gi, matched in enumerate(gt_matched):
        cat = gt_entities[gi][1]
        if matched:
            cat_tp[cat] += 1
        else:
            cat_fn[cat] += 1

    for di, matched in enumerate(det_matched):
        if not matched:
            cat = det_entities[di][1]
            cat_fp[cat] += 1

    tp = sum(gt_matched)
    fn = len(gt_matched) - tp
    fp = sum(1 for m in det_matched if not m)

    return SampleResult(
        sample_id="",
        elapsed=0.0,
        n_gt=len(gt_entities),
        n_det=len(det_entities),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        category_tp=dict(cat_tp),
        category_fp=dict(cat_fp),
        category_fn=dict(cat_fn),
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def aggregate_metrics(
    model_name: str, results: list[SampleResult],
) -> AggregateMetrics:
    total_tp = sum(r.true_positives for r in results)
    total_fp = sum(r.false_positives for r in results)
    total_fn = sum(r.false_negatives for r in results)

    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Per-category
    all_cats: set[str] = set()
    cat_tp_t: dict[str, int] = defaultdict(int)
    cat_fp_t: dict[str, int] = defaultdict(int)
    cat_fn_t: dict[str, int] = defaultdict(int)
    for r in results:
        for c, v in r.category_tp.items():
            cat_tp_t[c] += v
            all_cats.add(c)
        for c, v in r.category_fp.items():
            cat_fp_t[c] += v
            all_cats.add(c)
        for c, v in r.category_fn.items():
            cat_fn_t[c] += v
            all_cats.add(c)

    per_category: dict[str, dict[str, float]] = {}
    for cat in sorted(all_cats):
        tp = cat_tp_t[cat]
        fp = cat_fp_t[cat]
        fn = cat_fn_t[cat]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f = _safe_div(2 * p * r, p + r)
        per_category[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    total_time = sum(r.elapsed for r in results)
    return AggregateMetrics(
        model_name=model_name,
        total_samples=len(results),
        total_time=round(total_time, 2),
        avg_time=round(_safe_div(total_time, len(results)), 2),
        overall_precision=round(precision, 4),
        overall_recall=round(recall, 4),
        overall_f1=round(f1, 4),
        per_category=per_category,
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_on_text(
    orchestrator,
    text: str,
    sample_id: str,
    tmp_dir: Path,
    output_dir: Path,
):
    """Run the full pipeline on a text string.

    Returns (lookup_entries, elapsed_seconds).
    """
    from src.pipeline.lookup_table import LookupTable

    input_path = tmp_dir / f"sample_{sample_id}.txt"
    input_path.write_text(text, encoding="utf-8")

    lookup = LookupTable()

    t0 = time.perf_counter()
    orchestrator.process_file(
        file_path=input_path,
        lookup=lookup,
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - t0

    return lookup.all_entries(), elapsed


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_comparison_table(metrics_list: list[AggregateMetrics]) -> None:
    """Print side-by-side comparison table to stdout."""
    names = [m.model_name for m in metrics_list]
    col_w = max(24, *(len(n) + 2 for n in names))
    header_w = 24

    print()
    print("=" * (header_w + col_w * len(metrics_list) + 4))
    print("GRETEL PII BENCHMARK RESULTS")
    print(f"Samples: {metrics_list[0].total_samples}")
    print("=" * (header_w + col_w * len(metrics_list) + 4))

    # Overall metrics
    print()
    print("OVERALL METRICS")
    print("-" * (header_w + col_w * len(metrics_list) + 4))
    header = f"{'Metric':<{header_w}}"
    for m in metrics_list:
        header += f"{m.model_name:>{col_w}}"
    print(header)
    print("-" * (header_w + col_w * len(metrics_list) + 4))

    for label, attr in [
        ("Precision", "overall_precision"),
        ("Recall", "overall_recall"),
        ("F1 Score", "overall_f1"),
        ("Total Time (s)", "total_time"),
        ("Avg Time/Sample (s)", "avg_time"),
    ]:
        row = f"{label:<{header_w}}"
        for m in metrics_list:
            val = getattr(m, attr)
            row += f"{val:>{col_w}}"
        print(row)

    print()

    # Per-category
    all_cats = sorted(
        set().union(*(m.per_category.keys() for m in metrics_list))
    )
    if not all_cats:
        return

    print("PER-CATEGORY BREAKDOWN")
    print("-" * (header_w + col_w * len(metrics_list) + 4))
    header = f"{'Category':<{header_w}}"
    for m in metrics_list:
        header += f"{m.model_name + ' P/R/F1':>{col_w}}"
    print(header)
    print("-" * (header_w + col_w * len(metrics_list) + 4))

    for cat in all_cats:
        row = f"{cat:<{header_w}}"
        for m in metrics_list:
            if cat in m.per_category:
                d = m.per_category[cat]
                cell = f"{d['precision']:.2f}/{d['recall']:.2f}/{d['f1']:.2f}"
            else:
                cell = "  -  "
            row += f"{cell:>{col_w}}"
        print(row)

    print()


def save_json_results(
    metrics_list: list[AggregateMetrics],
    path: str,
    args: argparse.Namespace,
) -> None:
    result = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "samples": args.samples,
            "seed": args.seed,
            "dataset": "gretelai/gretel-pii-masking-en-v1",
            "split": "test",
            "relaxed_matching": args.relaxed,
        },
        "models": [asdict(m) for m in metrics_list],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark autodact PII pipeline against Gretel PII dataset.",
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of test rows to evaluate (default: 100)",
    )
    parser.add_argument(
        "--models", choices=["standard", "fast", "both"], default="both",
        help='Which model(s) to benchmark (default: "both")',
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-sample details",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save JSON results to this file path",
    )
    parser.add_argument(
        "--relaxed", action="store_true",
        help="Category-relaxed matching (text only, ignore category)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.ERROR)

    # 1. Load dataset via huggingface_hub + pyarrow (avoids lzma dependency)
    print("Loading Gretel PII dataset (test split)...")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        repo_id="gretelai/gretel-pii-masking-en-v1",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    ds = table.to_pydict()
    total_rows = len(ds["uid"])
    print(f"Loaded {total_rows} rows from test split")

    # 2. Sample rows
    random.seed(args.seed)
    n = min(args.samples, total_rows)
    indices = sorted(random.sample(range(total_rows), n))
    samples = [{k: ds[k][i] for k in ds} for i in indices]
    print(f"Selected {n} samples (seed={args.seed})")

    # 3. Determine models
    model_ids = (
        ["standard", "fast"] if args.models == "both" else [args.models]
    )

    # 4. Initialise shared components
    print("Initialising Presidio + name dictionary...")
    from src.pipeline.presidio_detector import PresidioDetector
    from src.pipeline.name_detector import NameDictionaryDetector

    presidio = PresidioDetector()
    name_detector = NameDictionaryDetector()

    from src.config import AVAILABLE_MODELS, get_models_dir
    from src.pipeline.llm_engine import LLMEngine
    from src.pipeline.llm_detector import LLMDetector
    from src.pipeline.orchestrator import Orchestrator

    all_metrics: list[AggregateMetrics] = []

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_dir = tmp_dir / "output"
        output_dir.mkdir()

        for model_id in model_ids:
            model_info = next(
                (m for m in AVAILABLE_MODELS if m.id == model_id), None,
            )
            if model_info is None:
                print(f"SKIP: Unknown model ID '{model_id}'")
                continue

            model_path = get_models_dir() / model_info.local_name
            if not model_path.exists():
                print(f"SKIP: {model_info.name} not found at {model_path}")
                continue

            print()
            print("=" * 70)
            print(f"Benchmarking: {model_info.name}")
            print(f"Model file:   {model_path}")
            print("=" * 70)

            engine = LLMEngine(str(model_path), n_threads=8)
            llm_detector = LLMDetector(engine)
            orchestrator = Orchestrator(
                presidio_detector=presidio,
                llm_detector=llm_detector,
                name_detector=name_detector,
            )

            sample_results: list[SampleResult] = []

            for idx, row in enumerate(samples):
                text = row["text"]
                entities_raw = row["entities"]
                uid = row["uid"]

                gt_entities = parse_gretel_entities(entities_raw)
                if not gt_entities:
                    continue

                try:
                    det_entries, elapsed = run_pipeline_on_text(
                        orchestrator, text, uid, tmp_dir, output_dir,
                    )
                except Exception as exc:
                    print(f"  ERROR on {uid[:8]}: {exc}")
                    continue

                result = evaluate_sample(
                    gt_entities, det_entries, relaxed=args.relaxed,
                )
                result.sample_id = uid
                result.elapsed = elapsed
                sample_results.append(result)

                if args.verbose:
                    print(
                        f"  [{idx+1}/{n}] {uid[:8]}… "
                        f"GT={result.n_gt} Det={result.n_det} "
                        f"TP={result.true_positives} "
                        f"FP={result.false_positives} "
                        f"FN={result.false_negatives} "
                        f"({elapsed:.2f}s)"
                    )
                elif (idx + 1) % 10 == 0 or idx + 1 == n:
                    print(f"  Progress: {idx+1}/{n}")

            metrics = aggregate_metrics(model_info.name, sample_results)
            all_metrics.append(metrics)

    # Output
    if all_metrics:
        print_comparison_table(all_metrics)
        if args.output:
            save_json_results(all_metrics, args.output, args)
    else:
        print("No models were benchmarked. Ensure model files are downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
