"""
Evaluation Script for Candidate Blocking (Member 2).

Evaluates candidate blocking performance against train_ground_truth.tsv:
- Blocking Pair Recall (True candidate pairs retrieved / Total ground truth pairs).
- Total ground truth pairs vs retrieved true pairs.
- Candidate count statistics (Average, Median, P95, Max per S1).
- Total candidate pairs and Reduction Ratio.
- Per-pass marginal and cumulative recall (Pass 1, Pass 2, Pass 3, Pass 4).
- Detailed Missed-Match analysis report.
- Clearly distinguishes genuine no-match S1 entities from S1 entities with missed matches.
"""

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure safe console printing on Windows platforms with non-ASCII dataset characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# Ensure parent directory is in path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from blocking import (
    BlockingConfig,
    MultiPassBlocker,
    extract_all_blocking_keys,
)
from preprocessing import (
    is_missing,
    normalize_address,
    normalize_business_name,
    normalize_country,
)


def evaluate_blocking_on_sample(
    data_dir: Path,
    sample_size: int = 2_000,
    distractor_size: int = 200_000,
    chunk_size: int = 100_000,
    max_candidates: int = 200,
    max_block_size: int = 1_000,
    partition_by_country: bool = True,
    missed_report_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run blocking evaluation on a controlled sample of S1 entities against ground truth.

    Parameters
    ----------
    data_dir : Path
        Dataset directory containing train_source*.tsv and train_ground_truth.tsv.
    sample_size : int, default 2000
        Number of S1 entities from ground truth to evaluate.
    distractor_size : int, default 200000
        Number of additional target records to index per source file as realistic negative distractors.
    chunk_size : int, default 100000
        Chunk size for reading TSVs.
    max_candidates : int, default 200
        Candidate cap per S1 entity.
    max_block_size : int, default 1000
        Super-block frequency limit.
    partition_by_country : bool, default True
        Whether to partition blocking keys by country.
    missed_report_file : Optional[Path]
        Optional file path to output missed-match TSV report.

    Returns
    -------
    Dict[str, Any]
        Dictionary of comprehensive evaluation metrics.
    """
    t_start = time.time()

    gt_file = data_dir / "train_ground_truth.tsv"
    s1_file = data_dir / "train_source1.tsv"
    s2_file = data_dir / "train_source2.tsv"
    s3_file = data_dir / "train_source3.tsv"

    # Step 1: Read validation slice of ground truth
    print(f">>> Reading first {sample_size:,} rows from '{gt_file.name}'...")
    gt_df = pd.read_csv(gt_file, sep="\t", nrows=sample_size, dtype=str)

    ground_truth_map: Dict[str, Set[str]] = {}
    s1_needed: Set[str] = set()
    s2_needed: Set[str] = set()
    s3_needed: Set[str] = set()

    no_match_s1_count = 0
    matched_s1_count = 0
    total_gt_pairs = 0

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        s1_needed.add(s1_id)
        m_val = row["matched_entity_ids"]

        if is_missing(m_val) or not str(m_val).strip():
            ground_truth_map[s1_id] = set()
            no_match_s1_count += 1
        else:
            m_list = [x.strip() for x in str(m_val).split(",") if x.strip()]
            if not m_list:
                ground_truth_map[s1_id] = set()
                no_match_s1_count += 1
            else:
                ground_truth_map[s1_id] = set(m_list)
                matched_s1_count += 1
                total_gt_pairs += len(m_list)
                for mid in m_list:
                    if mid.startswith("S2-"):
                        s2_needed.add(mid)
                    elif mid.startswith("S3-"):
                        s3_needed.add(mid)

    print(f"    Sampled {len(gt_df):,} S1 entities:")
    print(f"    - Entities with >=1 true matches: {matched_s1_count:,}")
    print(f"    - Genuine no-match entities:     {no_match_s1_count:,} ({no_match_s1_count/len(gt_df)*100:.2f}%)")
    print(f"    - Total ground truth pairs:       {total_gt_pairs:,}")
    print(f"    - True targets needed: S2={len(s2_needed):,}, S3={len(s3_needed):,}")

    # Step 2: Initialize Blocker
    config = BlockingConfig(
        data_dir=data_dir,
        chunk_size=chunk_size,
        max_candidates=max_candidates,
        max_block_size=max_block_size,
        partition_by_country=partition_by_country,
    )
    blocker = MultiPassBlocker(config)

    # Step 3: Stream and index Source 2 & Source 3 (Indexing needed targets + distractors)
    print(f"\n>>> Indexing Source 2 (ensuring all {len(s2_needed):,} needed targets + {distractor_size:,} distractors)...")
    s2_indexed = 0
    s2_metadata: Dict[str, Dict[str, str]] = {}

    reader_s2 = pd.read_csv(s2_file, sep="\t", chunksize=chunk_size, dtype=str)
    for chunk in reader_s2:
        eids = chunk["entity_id"].values
        names = chunk["business_name"].values
        addrs = chunk["business_address"].values
        ctrys = chunk["country"].values

        for i in range(len(eids)):
            eid = eids[i]
            is_needed = eid in s2_needed
            if is_needed or s2_indexed < distractor_size:
                raw_c = ctrys[i]
                norm_c = normalize_country(raw_c) if raw_c and not is_missing(raw_c) else ""
                raw_n = names[i]
                norm_n = normalize_business_name(raw_n) if raw_n and not is_missing(raw_n) else ""
                raw_a = addrs[i]
                norm_a = normalize_address(raw_a) if raw_a and not is_missing(raw_a) else ""

                if is_needed:
                    s2_metadata[eid] = {
                        "name": norm_n,
                        "addr": norm_a,
                        "country": norm_c,
                        "source": "S2",
                    }

                keys_dict = extract_all_blocking_keys(
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    country=norm_c,
                    partition_by_country=partition_by_country,
                )
                blocker.index.add_target(
                    entity_id=eid,
                    source="S2",
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    keys_dict=keys_dict,
                )
                s2_indexed += 1

        # Stop early if distractors met and all needed targets found
        if s2_indexed >= distractor_size and len(s2_metadata) >= len(s2_needed):
            break

    print(f"    Indexed {s2_indexed:,} S2 entities ({len(s2_metadata)}/{len(s2_needed)} true targets found)")

    print(f"\n>>> Indexing Source 3 (ensuring all {len(s3_needed):,} needed targets + {distractor_size:,} distractors)...")
    s3_indexed = 0
    s3_metadata: Dict[str, Dict[str, str]] = {}

    reader_s3 = pd.read_csv(s3_file, sep="\t", chunksize=chunk_size, dtype=str)
    for chunk in reader_s3:
        eids = chunk["entity_id"].values
        names = chunk["business_name"].values
        addrs = chunk["business_address"].values
        ctrys = chunk["country"].values

        for i in range(len(eids)):
            eid = eids[i]
            is_needed = eid in s3_needed
            if is_needed or s3_indexed < distractor_size:
                raw_c = ctrys[i]
                norm_c = normalize_country(raw_c) if raw_c and not is_missing(raw_c) else ""
                raw_n = names[i]
                norm_n = normalize_business_name(raw_n) if raw_n and not is_missing(raw_n) else ""
                raw_a = addrs[i]
                norm_a = normalize_address(raw_a) if raw_a and not is_missing(raw_a) else ""

                if is_needed:
                    s3_metadata[eid] = {
                        "name": norm_n,
                        "addr": norm_a,
                        "country": norm_c,
                        "source": "S3",
                    }

                keys_dict = extract_all_blocking_keys(
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    country=norm_c,
                    partition_by_country=partition_by_country,
                )
                blocker.index.add_target(
                    entity_id=eid,
                    source="S3",
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    keys_dict=keys_dict,
                )
                s3_indexed += 1

        if s3_indexed >= distractor_size and len(s3_metadata) >= len(s3_needed):
            break

    total_targets_indexed = s2_indexed + s3_indexed
    print(f"    Indexed {s3_indexed:,} S3 entities ({len(s3_metadata)}/{len(s3_needed)} true targets found)")
    print(f"    Total targets indexed: {total_targets_indexed:,}, Index keys: {blocker.index.num_keys():,}")
    print(f"    Super-blocks flagged and capped: {len(blocker.index.super_blocks):,}")

    # Step 4: Stream S1 slice and query candidates
    print(f"\n>>> Querying {len(s1_needed):,} Source 1 entities...")
    s1_rows_collected: List[Dict[str, Any]] = []
    reader_s1 = pd.read_csv(s1_file, sep="\t", chunksize=chunk_size, dtype=str)

    for chunk in reader_s1:
        sub = chunk[chunk["entity_id"].isin(s1_needed)]
        if not sub.empty:
            s1_rows_collected.extend(sub.to_dict(orient="records"))
        if len(s1_rows_collected) >= len(s1_needed):
            break

    s1_df_sample = pd.DataFrame(s1_rows_collected)
    candidates = blocker.query_source1_chunk(s1_df_sample)
    total_candidate_pairs = len(candidates)

    # Step 5: Evaluate Recall & Statistics
    candidate_map: Dict[str, Set[str]] = defaultdict(set)
    pass_provenance_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    candidate_counts_per_s1: List[int] = []

    for c in candidates:
        s1_id = c["source1_entity_id"]
        cand_id = c["candidate_entity_id"]
        candidate_map[s1_id].add(cand_id)
        passes = set(c["blocking_passes"].split(";"))
        pass_provenance_map[(s1_id, cand_id)].update(passes)

    # Track candidate counts for all evaluated S1 entities
    for s1_id in s1_needed:
        candidate_counts_per_s1.append(len(candidate_map.get(s1_id, set())))

    # Global Recall & Per-pass Marginal Recall
    retrieved_true_pairs = 0
    p1_only_true_pairs = 0
    p2_only_true_pairs = 0
    p3_only_true_pairs = 0
    p4_only_true_pairs = 0

    cum_p1_true = 0
    cum_p12_true = 0
    cum_p123_true = 0
    cum_all_true = 0

    missed_matches: List[Dict[str, Any]] = []

    # Map S1 data for missed match inspection
    s1_dict_map = {r["entity_id"]: r for r in s1_rows_collected}
    all_target_meta = {**s2_metadata, **s3_metadata}

    for s1_id, true_targets in ground_truth_map.items():
        if not true_targets:
            continue

        retrieved_set = candidate_map.get(s1_id, set())
        s1_row = s1_dict_map.get(s1_id, {})

        for target_id in true_targets:
            pair_key = (s1_id, target_id)
            if target_id in retrieved_set:
                retrieved_true_pairs += 1
                passes = pass_provenance_map[pair_key]

                # Standalone pass recalls
                if "pass1" in passes:
                    p1_only_true_pairs += 1
                if "pass2" in passes:
                    p2_only_true_pairs += 1
                if "pass3" in passes:
                    p3_only_true_pairs += 1
                if "pass4" in passes:
                    p4_only_true_pairs += 1

                # Cumulative pass recalls
                if "pass1" in passes:
                    cum_p1_true += 1
                if passes & {"pass1", "pass2"}:
                    cum_p12_true += 1
                if passes & {"pass1", "pass2", "pass3"}:
                    cum_p123_true += 1
                cum_all_true += 1
            else:
                # Missed Match!
                t_meta = all_target_meta.get(target_id, {})
                missed_matches.append({
                    "source1_entity_id": s1_id,
                    "target_entity_id": target_id,
                    "target_source": t_meta.get("source", "Unknown"),
                    "s1_name": s1_row.get("business_name", ""),
                    "target_name": t_meta.get("name", ""),
                    "s1_addr": s1_row.get("business_address", ""),
                    "target_addr": t_meta.get("addr", ""),
                    "s1_country": s1_row.get("country", ""),
                    "target_country": t_meta.get("country", ""),
                })

    blocking_pair_recall = (
        (retrieved_true_pairs / total_gt_pairs * 100.0)
        if total_gt_pairs > 0 else 0.0
    )

    # Reduction Ratio: 1 - (candidates / (S1 * Targets))
    cartesian_space = len(s1_needed) * total_targets_indexed
    reduction_ratio = (
        (1.0 - (total_candidate_pairs / cartesian_space)) * 100.0
        if cartesian_space > 0 else 100.0
    )

    # Distribution stats
    cand_counts_arr = np.array(candidate_counts_per_s1)
    avg_cands = float(np.mean(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0.0
    median_cands = float(np.median(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0.0
    p95_cands = float(np.percentile(cand_counts_arr, 95)) if len(cand_counts_arr) > 0 else 0.0
    max_cands = int(np.max(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0

    elapsed_total = time.time() - t_start

    # Save missed report if requested
    if missed_report_file and missed_matches:
        missed_df = pd.DataFrame(missed_matches)
        missed_report_file.parent.mkdir(parents=True, exist_ok=True)
        missed_df.to_csv(missed_report_file, sep="\t", index=False)
        print(f"\n[Saved missed match report: {missed_report_file.name} ({len(missed_matches)} missed pairs)]")

    return {
        "sample_size": len(gt_df),
        "total_s1_evaluated": len(s1_needed),
        "matched_s1_count": matched_s1_count,
        "no_match_s1_count": no_match_s1_count,
        "total_ground_truth_pairs": total_gt_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "missed_true_pairs": total_gt_pairs - retrieved_true_pairs,
        "blocking_pair_recall_pct": round(blocking_pair_recall, 2),
        "total_candidate_pairs": total_candidate_pairs,
        "total_targets_indexed": total_targets_indexed,
        "reduction_ratio_pct": round(reduction_ratio, 4),
        "avg_candidates_per_s1": round(avg_cands, 2),
        "median_candidates_per_s1": round(median_cands, 2),
        "p95_candidates_per_s1": round(p95_cands, 2),
        "max_candidates_per_s1": max_cands,
        "pass_recalls": {
            "pass1_alone": round(p1_only_true_pairs / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "pass2_alone": round(p2_only_true_pairs / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "pass3_alone": round(p3_only_true_pairs / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "pass4_alone": round(p4_only_true_pairs / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "cumulative_pass1": round(cum_p1_true / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "cumulative_pass1_2": round(cum_p12_true / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "cumulative_pass1_2_3": round(cum_p123_true / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
            "cumulative_all_passes": round(cum_all_true / total_gt_pairs * 100, 2) if total_gt_pairs else 0.0,
        },
        "missed_matches_sample": missed_matches[:10],
        "elapsed_seconds": round(elapsed_total, 2),
    }


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 80)
    print(" CANDIDATE BLOCKING EVALUATION REPORT")
    print("=" * 80)
    print(f"Total S1 Entities Evaluated:      {results['total_s1_evaluated']:,}")
    print(f"  - S1 with >= 1 true matches:     {results['matched_s1_count']:,}")
    print(f"  - Genuine no-match S1 entities:  {results['no_match_s1_count']:,} ({results['no_match_s1_count']/results['total_s1_evaluated']*100:.2f}%)")
    print(f"Total Targets in Search Space:    {results['total_targets_indexed']:,}")
    print(f"Execution Time:                   {results['elapsed_seconds']} seconds")

    print("\n" + "-" * 80)
    print(" RECALL PERFORMANCE (GROUND TRUTH)")
    print("-" * 80)
    print(f"Total Ground Truth Pairs:         {results['total_ground_truth_pairs']:,}")
    print(f"Retrieved True Match Pairs:       {results['retrieved_true_pairs']:,}")
    print(f"Missed True Match Pairs:          {results['missed_true_pairs']:,}")
    print(f">>> CANDIDATE PAIR RECALL:        {results['blocking_pair_recall_pct']:.2f}% <<<")

    print("\n" + "-" * 80)
    print(" PER-PASS RECALL CONTRIBUTION")
    print("-" * 80)
    pr = results["pass_recalls"]
    print(f"  - Pass 1 Alone (Name Prefix):    {pr['pass1_alone']:>6.2f}%")
    print(f"  - Pass 2 Alone (Address Anchor): {pr['pass2_alone']:>6.2f}%")
    print(f"  - Pass 3 Alone (Sorted Tokens):  {pr['pass3_alone']:>6.2f}%")
    print(f"  - Pass 4 Alone (Domain Stem):    {pr['pass4_alone']:>6.2f}%")
    print("  Cumulative Union:")
    print(f"  * Pass 1:                        {pr['cumulative_pass1']:>6.2f}%")
    print(f"  * Pass 1 + 2:                    {pr['cumulative_pass1_2']:>6.2f}%")
    print(f"  * Pass 1 + 2 + 3:                {pr['cumulative_pass1_2_3']:>6.2f}%")
    print(f"  * Pass 1 + 2 + 3 + 4 (All):      {pr['cumulative_all_passes']:>6.2f}%")

    print("\n" + "-" * 80)
    print(" CANDIDATE VOLUME & REDUCTION RATIO")
    print("-" * 80)
    print(f"Total Candidate Pairs Emitted:    {results['total_candidate_pairs']:,}")
    print(f"Reduction Ratio:                  {results['reduction_ratio_pct']:.4f}%")
    print(f"Average Candidates per S1:        {results['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates per S1:         {results['median_candidates_per_s1']:.2f}")
    print(f"95th Percentile Candidates:       {results['p95_candidates_per_s1']:.2f}")
    print(f"Maximum Candidates for any S1:    {results['max_candidates_per_s1']:,}")

    if results["missed_matches_sample"]:
        print("\n" + "-" * 80)
        print(" SAMPLE MISSED MATCHES (FOR ERROR ANALYSIS)")
        print("-" * 80)
        for idx, m in enumerate(results["missed_matches_sample"][:5], 1):
            print(f"[{idx}] S1: {m['source1_entity_id']} | Target: {m['target_entity_id']} ({m['target_source']})")
            print(f"    S1 Name:     {repr(m['s1_name'])}")
            print(f"    Target Name: {repr(m['target_name'])}")
            print(f"    S1 Addr:     {repr(m['s1_addr'])}")
            print(f"    Target Addr: {repr(m['target_addr'])}")
            print(f"    Country:     {m['s1_country']} vs {m['target_country']}")
            print()
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Candidate Blocking Evaluator for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing train TSV files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2_000,
        help="Number of S1 entities from ground truth to evaluate (default: 2000).",
    )
    parser.add_argument(
        "--distractor-size",
        type=int,
        default=100_000,
        help="Number of target records to index per source as distractors (default: 100000).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Candidate cap per S1 entity (default: 200).",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=1_000,
        help="Super-block limit (default: 1000).",
    )
    parser.add_argument(
        "--disable-country-partition",
        action="store_true",
        help="Disable country partitioning.",
    )
    parser.add_argument(
        "--missed-report",
        type=str,
        default="missed_matches_report.tsv",
        help="Path to save missed match report TSV (default: missed_matches_report.tsv).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    missed_report_path = Path(args.missed_report).resolve() if args.missed_report else None

    results = evaluate_blocking_on_sample(
        data_dir=data_dir,
        sample_size=args.sample_size,
        distractor_size=args.distractor_size,
        max_candidates=args.max_candidates,
        max_block_size=args.max_block_size,
        partition_by_country=not args.disable_country_partition,
        missed_report_file=missed_report_path,
    )

    print_evaluation_report(results)


if __name__ == "__main__":
    main()
