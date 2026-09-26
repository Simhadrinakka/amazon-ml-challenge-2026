"""
Scale and Invariant Validation for Preprocessing on Real Datasets.

Validates normalization throughput, invariants, missing value handling,
and memory safety on chunks of real dataset TSV files without loading entire datasets.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Ensure safe console printing on Windows platforms with non-ASCII dataset characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure parent directory is accessible for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocessing import normalize_dataframe_chunk


def validate_file_preprocessing(
    file_path: Path,
    num_chunks: int = 3,
    chunk_size: int = 100_000,
    sample_examples_count: int = 5,
) -> Dict[str, Any]:
    """
    Validate chunk-based preprocessing on a real TSV file.

    Parameters
    ----------
    file_path : Path
        Path to the TSV file.
    num_chunks : int, default 3
        Number of chunks to read and process.
    chunk_size : int, default 100000
        Number of rows per chunk.
    sample_examples_count : int, default 5
        Number of original -> normalized examples to collect.

    Returns
    -------
    Dict[str, Any]
        Validation statistics and verification results.
    """
    file_name = file_path.name
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    total_rows = 0
    total_processing_time = 0.0
    chunks_processed = 0

    raw_missing_counts = Counter()
    norm_missing_counts = Counter()
    empty_norm_names = 0
    empty_norm_addrs = 0
    empty_norm_countries = 0

    raw_countries_counter = Counter()
    norm_countries_counter = Counter()

    name_examples: List[Tuple[str, str]] = []
    addr_examples: List[Tuple[str, str]] = []

    # Invariant verification flags
    invariant_failures: List[str] = []

    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
    )

    for chunk_idx, raw_chunk in enumerate(reader, 1):
        if chunk_idx > num_chunks:
            break

        chunks_processed += 1
        num_rows = len(raw_chunk)
        total_rows += num_rows

        # Capture pre-normalization snapshots for invariants
        raw_entity_ids = raw_chunk["entity_id"].copy() if "entity_id" in raw_chunk.columns else None
        raw_names = raw_chunk["business_name"].copy() if "business_name" in raw_chunk.columns else None
        raw_addrs = raw_chunk["business_address"].copy() if "business_address" in raw_chunk.columns else None
        raw_countries = raw_chunk["country"].copy() if "country" in raw_chunk.columns else None

        # Measure pure normalization execution time
        t_start = time.perf_counter()
        try:
            norm_chunk = normalize_dataframe_chunk(raw_chunk, inplace=False)
        except Exception as e:
            invariant_failures.append(f"Chunk {chunk_idx}: Exception during normalize_dataframe_chunk: {e}")
            break
        t_elapsed = time.perf_counter() - t_start
        total_processing_time += t_elapsed

        # --- Invariant Verifications ---
        # 1. Row count preservation
        if len(norm_chunk) != num_rows:
            invariant_failures.append(f"Chunk {chunk_idx}: Row count mismatch ({len(norm_chunk)} vs {num_rows})")

        # 2. entity_id unchanged
        if raw_entity_ids is not None:
            if not (raw_entity_ids.fillna("<NULL>") == norm_chunk["entity_id"].fillna("<NULL>")).all():
                invariant_failures.append(f"Chunk {chunk_idx}: entity_id was modified by normalization")

        # 3. Original raw columns untouched in non-inplace mode
        if raw_names is not None:
            if not (raw_names.fillna("<NULL>") == norm_chunk["business_name"].fillna("<NULL>")).all():
                invariant_failures.append(f"Chunk {chunk_idx}: Original business_name column was mutated")

        if raw_addrs is not None:
            if not (raw_addrs.fillna("<NULL>") == norm_chunk["business_address"].fillna("<NULL>")).all():
                invariant_failures.append(f"Chunk {chunk_idx}: Original business_address column was mutated")

        if raw_countries is not None:
            if not (raw_countries.fillna("<NULL>") == norm_chunk["country"].fillna("<NULL>")).all():
                invariant_failures.append(f"Chunk {chunk_idx}: Original country column was mutated")

        # 4. Normalized columns data types
        for col_name in ["business_name_norm", "business_address_norm", "country_norm"]:
            if col_name in norm_chunk.columns:
                non_str_count = sum(not isinstance(val, str) for val in norm_chunk[col_name])
                if non_str_count > 0:
                    invariant_failures.append(
                        f"Chunk {chunk_idx}: Column {col_name} contains {non_str_count} non-string entries"
                    )

        # --- Statistical Aggregations ---
        # Missing values in raw columns
        for col in raw_chunk.columns:
            raw_missing_counts[col] += int(raw_chunk[col].isna().sum())

        # Empty strings in normalized columns
        if "business_name_norm" in norm_chunk.columns:
            empty_norm_names += int((norm_chunk["business_name_norm"] == "").sum())
        if "business_address_norm" in norm_chunk.columns:
            empty_norm_addrs += int((norm_chunk["business_address_norm"] == "").sum())
        if "country_norm" in norm_chunk.columns:
            empty_norm_countries += int((norm_chunk["country_norm"] == "").sum())

        # Country distributions before & after
        if "country" in raw_chunk.columns and "country_norm" in norm_chunk.columns:
            raw_c_dist = raw_chunk["country"].fillna("<MISSING>").value_counts().to_dict()
            raw_countries_counter.update(raw_c_dist)

            norm_c_dist = norm_chunk["country_norm"].replace({"": "<EMPTY>"}).value_counts().to_dict()
            norm_countries_counter.update(norm_c_dist)

        # Collect diverse sample examples
        if len(name_examples) < sample_examples_count and "business_name" in raw_chunk.columns:
            mask = raw_chunk["business_name"].notna() & (raw_chunk["business_name"].str.len() > 3)
            sample_df = raw_chunk[mask].head(sample_examples_count - len(name_examples))
            for idx in sample_df.index:
                orig = raw_chunk.loc[idx, "business_name"]
                norm = norm_chunk.loc[idx, "business_name_norm"]
                name_examples.append((str(orig), str(norm)))

        if len(addr_examples) < sample_examples_count and "business_address" in raw_chunk.columns:
            mask = raw_chunk["business_address"].notna() & (raw_chunk["business_address"].str.len() > 10)
            sample_df = raw_chunk[mask].head(sample_examples_count - len(addr_examples))
            for idx in sample_df.index:
                orig = raw_chunk.loc[idx, "business_address"]
                norm = norm_chunk.loc[idx, "business_address_norm"]
                addr_examples.append((str(orig), str(norm)))

    throughput_rows_per_sec = total_rows / total_processing_time if total_processing_time > 0 else 0.0

    return {
        "file_name": file_name,
        "file_size_mb": round(file_size_mb, 2),
        "chunks_processed": chunks_processed,
        "total_rows": total_rows,
        "processing_time_sec": round(total_processing_time, 3),
        "throughput_rows_per_sec": round(throughput_rows_per_sec, 1),
        "raw_missing_counts": dict(raw_missing_counts),
        "empty_norm_names": empty_norm_names,
        "empty_norm_names_pct": round(empty_norm_names / total_rows * 100, 2) if total_rows > 0 else 0.0,
        "empty_norm_addrs": empty_norm_addrs,
        "empty_norm_addrs_pct": round(empty_norm_addrs / total_rows * 100, 2) if total_rows > 0 else 0.0,
        "empty_norm_countries": empty_norm_countries,
        "raw_top_countries": raw_countries_counter.most_common(5),
        "norm_top_countries": norm_countries_counter.most_common(5),
        "name_examples": name_examples,
        "addr_examples": addr_examples,
        "invariant_failures": invariant_failures,
        "passed": len(invariant_failures) == 0,
    }


def print_validation_report(res: Dict[str, Any]) -> None:
    """Print formatted validation report for a single file."""
    status_str = "[PASS]" if res["passed"] else "[FAIL]"
    print("\n" + "=" * 80)
    print(f" VALIDATION REPORT: {res['file_name']} {status_str}")
    print("=" * 80)
    print(f"File Size:            {res['file_size_mb']:,} MB")
    print(f"Chunks Processed:     {res['chunks_processed']}")
    print(f"Total Rows Sampled:   {res['total_rows']:,}")
    print(f"Processing Time:      {res['processing_time_sec']} seconds")
    print(f"Processing Speed:     {res['throughput_rows_per_sec']:,.1f} rows/sec")

    print("\n--- Missing / Empty Value Analysis ---")
    print("Raw Column Missing Counts:")
    for col, cnt in res["raw_missing_counts"].items():
        print(f"  - {col:<20}: {cnt:>8,} missing ({cnt / res['total_rows'] * 100:>5.2f}%)")
    print("Normalized Empty Counts (from missing or zero-length):")
    print(f"  - business_name_norm  : {res['empty_norm_names']:>8,} empty ({res['empty_norm_names_pct']:>5.2f}%)")
    print(f"  - business_address_norm: {res['empty_norm_addrs']:>8,} empty ({res['empty_norm_addrs_pct']:>5.2f}%)")

    print("\n--- Country Normalization Distribution (Top 5) ---")
    print("  Raw Countries:")
    for c, cnt in res["raw_top_countries"]:
        print(f"    * {repr(c):<20}: {cnt:>8,} ({cnt / res['total_rows'] * 100:>5.2f}%)")
    print("  Normalized Countries:")
    for c, cnt in res["norm_top_countries"]:
        print(f"    * {repr(c):<20}: {cnt:>8,} ({cnt / res['total_rows'] * 100:>5.2f}%)")

    print("\n--- Sample Business Name Transformations ---")
    for orig, norm in res["name_examples"]:
        print(f"  Original:   {repr(orig)}")
        print(f"  Normalized: {repr(norm)}")
        print("  " + "-" * 50)

    print("\n--- Sample Address Transformations ---")
    for orig, norm in res["addr_examples"]:
        print(f"  Original:   {repr(orig)}")
        print(f"  Normalized: {repr(norm)}")
        print("  " + "-" * 50)

    print("\n--- Invariant Verification Checks ---")
    if res["passed"]:
        print("  [OK] entity_id remains untouched and identical.")
        print("  [OK] Original business_name, business_address, country columns preserved.")
        print("  [OK] All normalized columns are valid non-null string objects.")
        print("  [OK] Row counts perfectly preserved across all chunks.")
        print("  [OK] Zero exceptions raised during chunk processing.")
    else:
        print("  [FAIL] Invariant violations detected:")
        for failure in res["invariant_failures"]:
            print(f"    - {failure}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate preprocessing scalability and invariants on real dataset TSVs."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing dataset TSV files (e.g. .../dataset/train).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"],
        help="List of TSV files to validate (default: train_source1.tsv, train_source2.tsv, train_source3.tsv).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=3,
        help="Number of chunks to process per file (default: 3).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk (default: 100000).",
    )
    parser.add_argument(
        "--sample-examples",
        type=int,
        default=5,
        help="Number of original vs normalized sample examples to display (default: 5).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "#" * 80)
    print(" PREPROCESSING SCALE & INVARIANT VALIDATION")
    print(f" Data Directory: {data_dir}")
    print(f" Target Files:   {args.files}")
    print(f" Target Chunks:  {args.chunks} chunks x {args.chunk_size:,} rows/chunk")
    print("#" * 80)

    overall_passed = True
    total_sampled_rows = 0
    total_bench_time = 0.0

    for file_name in args.files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"\nWarning: File not found: {file_path}, skipping...", file=sys.stderr)
            overall_passed = False
            continue

        print(f"\n>>> Validating '{file_name}' ({args.chunks} chunks x {args.chunk_size:,} rows)...")
        res = validate_file_preprocessing(
            file_path=file_path,
            num_chunks=args.chunks,
            chunk_size=args.chunk_size,
            sample_examples_count=args.sample_examples,
        )
        print_validation_report(res)

        total_sampled_rows += res["total_rows"]
        total_bench_time += res["processing_time_sec"]
        if not res["passed"]:
            overall_passed = False

    avg_speed = total_sampled_rows / total_bench_time if total_bench_time > 0 else 0.0

    print("\n" + "#" * 80)
    print(" OVERALL VALIDATION SUMMARY")
    print("#" * 80)
    print(f"Status:             {'PASSED (All Invariants Satisfied)' if overall_passed else 'FAILED'}")
    print(f"Total Rows Tested:  {total_sampled_rows:,}")
    print(f"Total Process Time: {total_bench_time:.2f} seconds")
    print(f"Average Throughput: {avg_speed:,.1f} rows/second")
    print("#" * 80 + "\n")

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
