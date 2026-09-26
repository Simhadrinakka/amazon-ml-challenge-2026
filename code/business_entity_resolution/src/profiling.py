"""
Dataset Profiling Utility for Amazon ML Challenge 2026: Business Entity Resolution.

Memory-efficient dataset profiler that processes large TSV files in chunks
without loading entire multi-million-row datasets into memory.
"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class StreamingStats:
    """Computes streaming statistics (min, max, mean, std, null/empty counts) in O(1) memory."""

    def __init__(self, name: str):
        self.name = name
        self.count: int = 0
        self.null_count: int = 0
        self.empty_count: int = 0
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self.sum_val: float = 0.0
        self.sum_sq_val: float = 0.0

    def update(self, series: pd.Series) -> None:
        """Update statistics with a chunk of data."""
        # Null count
        nulls = series.isna().sum()
        self.null_count += int(nulls)

        # Valid non-null values
        valid = series.dropna().astype(str)
        if valid.empty:
            return

        # Empty string count
        lengths = valid.str.len()
        self.empty_count += int((lengths == 0).sum())

        # Update min, max, sums
        n = len(lengths)
        self.count += n
        min_l = lengths.min()
        max_l = lengths.max()
        sum_l = lengths.sum()
        sum_sq_l = (lengths**2).sum()

        if min_l < self.min_val:
            self.min_val = float(min_l)
        if max_l > self.max_val:
            self.max_val = float(max_l)
        self.sum_val += float(sum_l)
        self.sum_sq_val += float(sum_sq_l)

    @property
    def mean(self) -> float:
        return self.sum_val / self.count if self.count > 0 else 0.0

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        variance = (self.sum_sq_val - (self.sum_val**2) / self.count) / (self.count - 1)
        return math.sqrt(max(0.0, variance))

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "valid_count": self.count,
            "null_count": self.null_count,
            "empty_string_count": self.empty_count,
            "min_length": int(self.min_val) if self.count > 0 else 0,
            "max_length": int(self.max_val) if self.count > 0 else 0,
            "mean_length": round(self.mean, 2),
            "std_length": round(self.std, 2),
        }


def profile_source_tsv(
    file_path: Path,
    chunk_size: int = 100_000,
    top_n_countries: int = 15,
) -> Dict[str, Any]:
    """
    Profile a source TSV file (Source 1, Source 2, Source 3) using chunked streaming.
    """
    file_name = file_path.name
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    total_rows = 0
    column_names: List[str] = []
    missing_counts: Counter = Counter()
    country_counter: Counter = Counter()
    entity_id_set = set()
    duplicate_entity_id_count = 0

    name_stats = StreamingStats("business_name_char_len")
    name_words_stats = StreamingStats("business_name_word_len")
    addr_stats = StreamingStats("business_address_char_len")
    addr_words_stats = StreamingStats("business_address_word_len")

    start_time = time.time()
    chunk_count = 0

    # Stream file in chunks
    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
    )

    for chunk in reader:
        chunk_count += 1
        num_chunk_rows = len(chunk)
        total_rows += num_chunk_rows

        if not column_names:
            column_names = list(chunk.columns)

        # Missing values per column
        for col in chunk.columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Entity ID unique & duplicate tracking
        if "entity_id" in chunk.columns:
            ids = chunk["entity_id"].dropna().tolist()
            for eid in ids:
                if eid in entity_id_set:
                    duplicate_entity_id_count += 1
                else:
                    entity_id_set.add(eid)

        # Country distribution
        if "country" in chunk.columns:
            countries = chunk["country"].fillna("<MISSING>").value_counts().to_dict()
            country_counter.update(countries)

        # Business name statistics
        if "business_name" in chunk.columns:
            name_stats.update(chunk["business_name"])
            # Word count stats
            non_null_names = chunk["business_name"].dropna().astype(str)
            word_counts = non_null_names.str.split().str.len()
            if not word_counts.empty:
                name_words_stats.count += len(word_counts)
                min_w = word_counts.min()
                max_w = word_counts.max()
                sum_w = word_counts.sum()
                sum_sq_w = (word_counts**2).sum()
                if min_w < name_words_stats.min_val:
                    name_words_stats.min_val = float(min_w)
                if max_w > name_words_stats.max_val:
                    name_words_stats.max_val = float(max_w)
                name_words_stats.sum_val += float(sum_w)
                name_words_stats.sum_sq_val += float(sum_sq_w)

        # Business address statistics
        if "business_address" in chunk.columns:
            addr_stats.update(chunk["business_address"])
            # Word count stats
            non_null_addrs = chunk["business_address"].dropna().astype(str)
            addr_word_counts = non_null_addrs.str.split().str.len()
            if not addr_word_counts.empty:
                addr_words_stats.count += len(addr_word_counts)
                min_w = addr_word_counts.min()
                max_w = addr_word_counts.max()
                sum_w = addr_word_counts.sum()
                sum_sq_w = (addr_word_counts**2).sum()
                if min_w < addr_words_stats.min_val:
                    addr_words_stats.min_val = float(min_w)
                if max_w > addr_words_stats.max_val:
                    addr_words_stats.max_val = float(max_w)
                addr_words_stats.sum_val += float(sum_w)
                addr_words_stats.sum_sq_val += float(sum_sq_w)

    elapsed_time = time.time() - start_time

    unique_entity_count = len(entity_id_set)
    # Free memory
    del entity_id_set

    return {
        "file_name": file_name,
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": total_rows,
        "chunks_processed": chunk_count,
        "elapsed_seconds": round(elapsed_time, 2),
        "columns": column_names,
        "unique_entity_ids": unique_entity_count,
        "duplicate_entity_ids": duplicate_entity_id_count,
        "missing_counts": dict(missing_counts),
        "top_countries": country_counter.most_common(top_n_countries),
        "total_unique_countries": len(country_counter),
        "business_name_char_stats": name_stats.summary_dict(),
        "business_name_word_stats": name_words_stats.summary_dict(),
        "business_address_char_stats": addr_stats.summary_dict(),
        "business_address_word_stats": addr_words_stats.summary_dict(),
    }


def profile_ground_truth_tsv(
    file_path: Path,
    chunk_size: int = 100_000,
) -> Dict[str, Any]:
    """
    Profile train_ground_truth.tsv using chunked streaming.
    """
    file_name = file_path.name
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    total_rows = 0
    column_names: List[str] = []
    missing_counts: Counter = Counter()
    s1_ids_set = set()
    duplicate_s1_count = 0

    empty_matches_count = 0
    match_count_distribution: Counter = Counter()
    source2_matches_count = 0
    source3_matches_count = 0
    other_source_matches_count = 0
    total_matched_ids_count = 0

    min_matches_per_s1 = float("inf")
    max_matches_per_s1 = float("-inf")
    sum_matches = 0

    start_time = time.time()
    chunk_count = 0

    reader = pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
    )

    for chunk in reader:
        chunk_count += 1
        num_chunk_rows = len(chunk)
        total_rows += num_chunk_rows

        if not column_names:
            column_names = list(chunk.columns)

        for col in chunk.columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Source 1 ID tracking
        if "source1_entity_id" in chunk.columns:
            for s1_id in chunk["source1_entity_id"].dropna():
                if s1_id in s1_ids_set:
                    duplicate_s1_count += 1
                else:
                    s1_ids_set.add(s1_id)

        # Matched entity IDs analysis
        if "matched_entity_ids" in chunk.columns:
            for val in chunk["matched_entity_ids"]:
                if pd.isna(val) or not str(val).strip() or str(val).strip().lower() == "nan":
                    empty_matches_count += 1
                    match_count_distribution[0] += 1
                    min_matches_per_s1 = min(min_matches_per_s1, 0)
                    max_matches_per_s1 = max(max_matches_per_s1, 0)
                else:
                    matched_list = [m.strip() for m in str(val).split(",") if m.strip()]
                    num_matches = len(matched_list)
                    if num_matches == 0:
                        empty_matches_count += 1
                        match_count_distribution[0] += 1
                    else:
                        match_count_distribution[num_matches] += 1
                        total_matched_ids_count += num_matches
                        sum_matches += num_matches
                        min_matches_per_s1 = min(min_matches_per_s1, num_matches)
                        max_matches_per_s1 = max(max_matches_per_s1, num_matches)

                        # Check source prefix distribution
                        for m_id in matched_list:
                            if m_id.startswith("S2-"):
                                source2_matches_count += 1
                            elif m_id.startswith("S3-"):
                                source3_matches_count += 1
                            else:
                                other_source_matches_count += 1

    elapsed_time = time.time() - start_time
    unique_s1_count = len(s1_ids_set)
    del s1_ids_set

    avg_matches_per_s1 = sum_matches / total_rows if total_rows > 0 else 0.0

    return {
        "file_name": file_name,
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": total_rows,
        "chunks_processed": chunk_count,
        "elapsed_seconds": round(elapsed_time, 2),
        "columns": column_names,
        "unique_source1_entities": unique_s1_count,
        "duplicate_source1_entities": duplicate_s1_count,
        "missing_counts": dict(missing_counts),
        "empty_matches_count": empty_matches_count,
        "empty_matches_percentage": round((empty_matches_count / total_rows * 100), 2) if total_rows > 0 else 0.0,
        "total_matched_ids": total_matched_ids_count,
        "source2_matches_count": source2_matches_count,
        "source3_matches_count": source3_matches_count,
        "other_source_matches_count": other_source_matches_count,
        "min_matches_per_s1": int(min_matches_per_s1) if total_rows > 0 else 0,
        "max_matches_per_s1": int(max_matches_per_s1) if total_rows > 0 else 0,
        "mean_matches_per_s1": round(avg_matches_per_s1, 3),
        "match_count_distribution": dict(sorted(match_count_distribution.items())),
    }


def print_source_report(report: Dict[str, Any]) -> None:
    """Print formatted profiling report for a source TSV."""
    print("=" * 80)
    print(f" FILE PROFILING REPORT: {report['file_name']}")
    print("=" * 80)
    print(f"File Size:            {report['file_size_mb']:,} MB")
    print(f"Total Rows:           {report['total_rows']:,}")
    print(f"Chunks Processed:     {report['chunks_processed']} (Streaming)")
    print(f"Execution Time:       {report['elapsed_seconds']} seconds")
    print(f"Column Names:         {', '.join(report['columns'])}")
    print(f"Unique entity_id:     {report['unique_entity_ids']:,}")
    print(f"Duplicate entity_id:  {report['duplicate_entity_ids']:,}")

    print("\n--- Missing Value Count per Column ---")
    for col, count in report["missing_counts"].items():
        pct = (count / report["total_rows"] * 100) if report["total_rows"] > 0 else 0.0
        print(f"  - {col:<20}: {count:>10,} missing ({pct:>6.2f}%)")

    print(f"\n--- Country Distribution (Total Unique: {report['total_unique_countries']}) ---")
    for country, count in report["top_countries"]:
        pct = (count / report["total_rows"] * 100) if report["total_rows"] > 0 else 0.0
        print(f"  - {country:<20}: {count:>10,} ({pct:>6.2f}%)")

    print("\n--- Business Name Statistics ---")
    c_stat = report["business_name_char_stats"]
    w_stat = report["business_name_word_stats"]
    print(f"  - Character Length: Min={c_stat['min_length']}, Max={c_stat['max_length']}, Mean={c_stat['mean_length']}, Std={c_stat['std_length']}")
    print(f"  - Word Count:       Min={w_stat['min_length']}, Max={w_stat['max_length']}, Mean={w_stat['mean_length']}, Std={w_stat['std_length']}")
    print(f"  - Empty Strings:    {c_stat['empty_string_count']:,}")

    print("\n--- Business Address Statistics ---")
    c_stat = report["business_address_char_stats"]
    w_stat = report["business_address_word_stats"]
    print(f"  - Character Length: Min={c_stat['min_length']}, Max={c_stat['max_length']}, Mean={c_stat['mean_length']}, Std={c_stat['std_length']}")
    print(f"  - Word Count:       Min={w_stat['min_length']}, Max={w_stat['max_length']}, Mean={w_stat['mean_length']}, Std={w_stat['std_length']}")
    print(f"  - Empty Strings:    {c_stat['empty_string_count']:,}")
    print()


def print_ground_truth_report(report: Dict[str, Any]) -> None:
    """Print formatted profiling report for train_ground_truth.tsv."""
    print("=" * 80)
    print(f" GROUND TRUTH PROFILING REPORT: {report['file_name']}")
    print("=" * 80)
    print(f"File Size:                     {report['file_size_mb']:,} MB")
    print(f"Total Rows:                    {report['total_rows']:,}")
    print(f"Execution Time:                {report['elapsed_seconds']} seconds")
    print(f"Column Names:                  {', '.join(report['columns'])}")
    print(f"Unique Source 1 Entities:      {report['unique_source1_entities']:,}")
    print(f"Duplicate Source 1 Entities:   {report['duplicate_source1_entities']:,}")
    print(f"Entities with 0 Matches:       {report['empty_matches_count']:,} ({report['empty_matches_percentage']}%)")
    print(f"Total Matched IDs Linked:      {report['total_matched_ids']:,}")
    print(f"  - Source 2 (S2) Matches:     {report['source2_matches_count']:,}")
    print(f"  - Source 3 (S3) Matches:     {report['source3_matches_count']:,}")
    if report["other_source_matches_count"] > 0:
        print(f"  - Other Matches:             {report['other_source_matches_count']:,}")

    print(f"\nMatches per Source 1 Entity Stats:")
    print(f"  - Min Matches:               {report['min_matches_per_s1']}")
    print(f"  - Max Matches:               {report['max_matches_per_s1']}")
    print(f"  - Mean Matches:              {report['mean_matches_per_s1']}")

    print("\n--- Match Count Distribution ---")
    for match_count, freq in report["match_count_distribution"].items():
        pct = (freq / report["total_rows"] * 100) if report["total_rows"] > 0 else 0.0
        print(f"  - {match_count} match(es): {freq:>10,} entities ({pct:>6.2f}%)")

    print("\n--- Missing Value Count per Column ---")
    for col, count in report["missing_counts"].items():
        pct = (count / report["total_rows"] * 100) if report["total_rows"] > 0 else 0.0
        print(f"  - {col:<25}: {count:>10,} missing ({pct:>6.2f}%)")
    print()


def find_tsv_files(data_dir: Path, split: str = "all") -> List[Path]:
    """Find TSV files in given data directory and subdirectories."""
    tsv_files: List[Path] = []

    if data_dir.is_file() and data_dir.suffix.lower() == ".tsv":
        return [data_dir]

    if not data_dir.exists():
        print(f"Error: Directory or file not found: {data_dir}", file=sys.stderr)
        return []

    # Check if target directory directly has TSVs
    direct_tsvs = sorted(list(data_dir.glob("*.tsv")))
    if direct_tsvs:
        tsv_files.extend(direct_tsvs)

    # Check train/test subdirectories if data_dir is the root dataset folder
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if split in ("train", "all") and train_dir.is_dir():
        tsv_files.extend(sorted(list(train_dir.glob("*.tsv"))))

    if split in ("test", "all") and test_dir.is_dir():
        tsv_files.extend(sorted(list(test_dir.glob("*.tsv"))))

    # Remove duplicates preserving order
    seen = set()
    unique_files = []
    for f in tsv_files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique_files.append(f)

    return unique_files


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient Dataset Profiler for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset directory (or subfolder like dataset/train or specific TSV file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Dataset split to profile if pointing to root dataset directory (default: all).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows per chunk for streaming processing (default: 100000).",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=15,
        help="Number of top countries to display in distribution (default: 15).",
    )

    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()
    tsv_files = find_tsv_files(data_path, split=args.split)

    if not tsv_files:
        print(f"No TSV files found for path: {data_path} (split={args.split})", file=sys.stderr)
        sys.exit(1)

    print("\n" + "#" * 80)
    print(" AMAZON ML CHALLENGE 2026 - DATASET PROFILING")
    print(f" Target Path: {data_path}")
    print(f" Files Found ({len(tsv_files)}): {[f.name for f in tsv_files]}")
    print(f" Chunk Size:  {args.chunk_size:,} rows per iteration")
    print("#" * 80 + "\n")

    for file_path in tsv_files:
        print(f">>> Processing '{file_path.name}' ({file_path.stat().st_size / (1024 * 1024):.1f} MB)...")
        if "ground_truth" in file_path.name.lower():
            report = profile_ground_truth_tsv(file_path, chunk_size=args.chunk_size)
            print_ground_truth_report(report)
        else:
            report = profile_source_tsv(
                file_path,
                chunk_size=args.chunk_size,
                top_n_countries=args.top_countries,
            )
            print_source_report(report)

    print("#" * 80)
    print(" PROFILING COMPLETED SUCCESSFULLY")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
