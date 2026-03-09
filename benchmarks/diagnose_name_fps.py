"""Diagnostic: find NAME false positives and attribute them to pipeline layers.

Runs each layer incrementally on benchmark samples to see which layer
produces which NAME detections, then compares against ground truth.
"""
from __future__ import annotations

import ast
import random
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmarks.benchmark_gretel import (
    GRETEL_TO_INTERNAL,
    MIN_ENTITY_LEN,
    text_match,
)


def parse_gt(entities_raw):
    if isinstance(entities_raw, str):
        raw = ast.literal_eval(entities_raw)
    else:
        raw = entities_raw
    results = []
    for item in raw:
        text = item["entity"]
        if len(text.strip()) < MIN_ENTITY_LEN:
            continue
        for t in item["types"]:
            cat = GRETEL_TO_INTERNAL.get(t)
            if cat:
                results.append((text.strip(), cat))
                break
    return results


def main():
    import logging
    logging.basicConfig(level=logging.ERROR)

    # Load dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        repo_id="gretelai/gretel-pii-masking-en-v1",
        filename="data/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    ds = table.to_pydict()

    random.seed(42)
    indices = sorted(random.sample(range(len(ds["uid"])), 100))
    samples = [{k: ds[k][i] for k in ds} for i in indices]

    # Initialise components
    from src.pipeline.presidio_detector import PresidioDetector
    from src.pipeline.name_detector import NameDictionaryDetector
    from src.pipeline.llm_engine import LLMEngine
    from src.pipeline.llm_detector import LLMDetector
    from src.pipeline.lookup_table import LookupTable
    from src.pipeline.document import Document, split_long_lines, reunify_sublines
    from src.pipeline.parsers import parse_file
    from src.pipeline.post_validator import validate_and_clean
    from src.config import get_models_dir, AVAILABLE_MODELS

    presidio = PresidioDetector()
    name_det = NameDictionaryDetector()

    model_info = next(m for m in AVAILABLE_MODELS if m.id == "standard")
    model_path = get_models_dir() / model_info.local_name
    engine = LLMEngine(str(model_path), n_threads=8)
    llm_det = LLMDetector(engine)

    # Track per-layer NAME detections
    layer_fps = defaultdict(list)   # layer -> list of false positive terms
    layer_tps = defaultdict(list)   # layer -> list of true positive terms
    layer_counts = defaultdict(int) # layer -> total NAME detections

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        out_dir = tmp_dir / "out"
        out_dir.mkdir()

        for idx, row in enumerate(samples):
            text = row["text"]
            uid = row["uid"]
            gt = parse_gt(row["entities"])
            gt_names = [(t, c) for t, c in gt if c == "NAME"]

            input_path = tmp_dir / f"s_{uid}.txt"
            input_path.write_text(text, encoding="utf-8")

            # Run layers incrementally, snapshotting NAME detections after each
            doc = parse_file(input_path)
            lookup = LookupTable()

            # Layer 0: email pre-pass (no NAME)
            presidio.pre_detect_emails(doc, lookup)

            # Layer 1: pattern pre-pass (no NAME)
            presidio.pre_detect_patterns(doc, lookup)
            
            # Snapshot before presidio NER
            before_presidio = {e.original_term for e in lookup.all_entries() if e.pii_category == "NAME"}

            # Layer 2: Presidio NER
            presidio.process(doc, lookup)
            after_presidio = {e.original_term for e in lookup.all_entries() if e.pii_category == "NAME"}
            presidio_names = after_presidio - before_presidio

            # Layer 3: Name dictionary
            name_det.process(doc, lookup)
            after_names = {e.original_term for e in lookup.all_entries() if e.pii_category == "NAME"}
            dict_names = after_names - after_presidio

            # Layer 4: LLM
            split_long_lines(doc, max_chars=500)
            llm_det.process(doc, lookup)
            reunify_sublines(doc)
            after_llm = {e.original_term for e in lookup.all_entries() if e.pii_category == "NAME"}
            llm_names = after_llm - after_names

            # Post-validation
            validate_and_clean(doc, lookup)
            after_post = {e.original_term for e in lookup.all_entries() if e.pii_category == "NAME"}

            # Now attribute each remaining NAME detection to its layer
            for term in after_post:
                if term in presidio_names:
                    layer = "presidio"
                elif term in dict_names:
                    layer = "namedict"
                elif term in llm_names:
                    layer = "llm"
                else:
                    layer = "unknown"

                layer_counts[layer] += 1

                # Is this a true positive?
                is_tp = any(text_match(gt_t, term) for gt_t, _ in gt_names)
                if is_tp:
                    layer_tps[layer].append(term)
                else:
                    layer_fps[layer].append(term)

            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx+1}/100")

    # Print results
    print("\n" + "=" * 70)
    print("NAME DETECTION ATTRIBUTION (100 samples)")
    print("=" * 70)
    
    for layer in ["presidio", "namedict", "llm", "unknown"]:
        total = layer_counts[layer]
        tp = len(layer_tps[layer])
        fp = len(layer_fps[layer])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"\n--- {layer.upper()} ---")
        print(f"  Total NAME detections: {total}")
        print(f"  True positives: {tp}  |  False positives: {fp}  |  Precision: {prec:.2%}")
        if layer_fps[layer]:
            # Show most common FP terms
            from collections import Counter
            fp_counts = Counter(layer_fps[layer]).most_common(30)
            print(f"  Top false positive terms:")
            for term, cnt in fp_counts:
                print(f"    {cnt:3d}x  {term!r}")

    # Also show removed-by-post-validation stats
    print("\n--- SUMMARY ---")
    total_all = sum(layer_counts.values())
    total_tp = sum(len(v) for v in layer_tps.values())
    total_fp = sum(len(v) for v in layer_fps.values())
    print(f"Total NAME detections (after post-validation): {total_all}")
    print(f"Overall NAME TP: {total_tp}  |  FP: {total_fp}  |  Precision: {total_tp/(total_tp+total_fp):.2%}" if (total_tp+total_fp) > 0 else "No NAME detections")


if __name__ == "__main__":
    main()
