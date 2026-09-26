"""
Evaluation Script for Candidate Blocking V3 Experimental (Member 2).

Evaluates candidate blocking performance against train_ground_truth.tsv:
- Blocking Pair Recall (True candidate pairs retrieved / Total ground truth pairs).
- Per-pass marginal and cumulative recall (Pass 1, Pass 2, Pass 2 Locpair, Pass 3, Pass 4, Pass 5 Distinctive, Pass 6 Compound).
- Reduction ratio, average, median, p95, max candidates per S1.
- Comparison V1 vs V2 vs V3.
- Tracks recovery from original 123 misses and from 70 V2 misses.
- Outputs missed_matches_report_v3.tsv.
"""

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from blocking_v3 import (
    BlockingConfig,
    MultiPassBlocker,
    extract_all_blocking_keys,
    normalize_business_name_v3,
)
from preprocessing import (
    is_missing,
    normalize_address,
    normalize_country,
)


def evaluate_blocking_v3_on_sample(
    data_dir: Path,
    sample_size: int = 1_000,
    distractor_size: int = 50_000,
    chunk_size: int = 100_000,
    max_candidates: int = 200,
    max_block_size: int = 1_000,
    partition_by_country: bool = True,
    missed_report_file: Optional[Path] = None,
    v1_missed_report_file: Optional[Path] = None,
    v2_missed_report_file: Optional[Path] = None,
) -> Dict[str, Any]:
    t_start = time.time()

    gt_file = data_dir / "train_ground_truth.tsv"
    s1_file = data_dir / "train_source1.tsv"
    s2_file = data_dir / "train_source2.tsv"
    s3_file = data_dir / "train_source3.tsv"

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

    config = BlockingConfig(
        data_dir=data_dir,
        chunk_size=chunk_size,
        max_candidates=max_candidates,
        max_block_size=max_block_size,
        partition_by_country=partition_by_country,
    )
    blocker = MultiPassBlocker(config)

    # Index Source 2
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
                norm_n = normalize_business_name_v3(raw_n) if raw_n and not is_missing(raw_n) else ""
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

        if s2_indexed >= distractor_size and len(s2_metadata) >= len(s2_needed):
            break

    print(f"    Indexed {s2_indexed:,} S2 entities ({len(s2_metadata)}/{len(s2_needed)} true targets found)")

    # Index Source 3
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
                norm_n = normalize_business_name_v3(raw_n) if raw_n and not is_missing(raw_n) else ""
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

    # Query S1 entities
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

    # Evaluate Recall & Statistics
    candidate_map: Dict[str, Set[str]] = defaultdict(set)
    pass_provenance_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    candidate_counts_per_s1: List[int] = []

    for c in candidates:
        s1_id = c["source1_entity_id"]
        cand_id = c["candidate_entity_id"]
        candidate_map[s1_id].add(cand_id)
        passes = set(c["blocking_passes"].split(";"))
        pass_provenance_map[(s1_id, cand_id)].update(passes)

    for s1_id in s1_needed:
        candidate_counts_per_s1.append(len(candidate_map.get(s1_id, set())))

    retrieved_true_pairs = 0
    pass_hits_counter: Counter = Counter()

    missed_matches: List[Dict[str, Any]] = []
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
                for p in passes:
                    pass_hits_counter[p] += 1
            else:
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

    cartesian_space = len(s1_needed) * total_targets_indexed
    reduction_ratio = (
        (1.0 - (total_candidate_pairs / cartesian_space)) * 100.0
        if cartesian_space > 0 else 100.0
    )

    cand_counts_arr = np.array(candidate_counts_per_s1)
    avg_cands = float(np.mean(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0.0
    median_cands = float(np.median(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0.0
    p95_cands = float(np.percentile(cand_counts_arr, 95)) if len(cand_counts_arr) > 0 else 0.0
    max_cands = int(np.max(cand_counts_arr)) if len(cand_counts_arr) > 0 else 0

    elapsed_total = time.time() - t_start

    # Save new missed matches report
    if missed_report_file and missed_matches:
        missed_df = pd.DataFrame(missed_matches)
        missed_report_file.parent.mkdir(parents=True, exist_ok=True)
        missed_df.to_csv(missed_report_file, sep="\t", index=False)
        print(f"\n[Saved V3 missed match report: {missed_report_file.name} ({len(missed_matches)} missed pairs)]")

    # Comparisons
    v3_missed_set = set(zip([m["source1_entity_id"] for m in missed_matches],
                            [m["target_entity_id"] for m in missed_matches]))

    recovered_from_v1: List[Tuple[str, str]] = []
    if v1_missed_report_file and v1_missed_report_file.exists():
        v1_df = pd.read_csv(v1_missed_report_file, sep="\t")
        v1_pairs = set(zip(v1_df["source1_entity_id"], v1_df["target_entity_id"]))
        for pair in v1_pairs:
            if pair not in v3_missed_set:
                recovered_from_v1.append(pair)

    recovered_from_v2: List[Tuple[str, str]] = []
    if v2_missed_report_file and v2_missed_report_file.exists():
        v2_df = pd.read_csv(v2_missed_report_file, sep="\t")
        v2_pairs = set(zip(v2_df["source1_entity_id"], v2_df["target_entity_id"]))
        for pair in v2_pairs:
            if pair not in v3_missed_set:
                recovered_from_v2.append(pair)

    return {
        "sample_size": len(gt_df),
        "total_s1_evaluated": len(s1_needed),
        "matched_s1_count": matched_s1_count,
        "no_match_s1_count": no_match_s1_count,
        "total_ground_truth_pairs": total_gt_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "missed_true_pairs": len(missed_matches),
        "blocking_pair_recall_pct": round(blocking_pair_recall, 2),
        "total_candidate_pairs": total_candidate_pairs,
        "total_targets_indexed": total_targets_indexed,
        "reduction_ratio_pct": round(reduction_ratio, 4),
        "avg_candidates_per_s1": round(avg_cands, 2),
        "median_candidates_per_s1": round(median_cands, 2),
        "p95_candidates_per_s1": round(p95_cands, 2),
        "max_candidates_per_s1": max_cands,
        "num_super_blocks": len(blocker.index.super_blocks),
        "pass_hits": dict(pass_hits_counter),
        "recovered_from_v1_count": len(recovered_from_v1),
        "recovered_from_v2_count": len(recovered_from_v2),
        "elapsed_seconds": round(elapsed_total, 2),
    }


def print_v3_report(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(" CANDIDATE BLOCKING V3 EXPERIMENTAL EVALUATION REPORT")
    print("=" * 80)
    print(f"Total S1 Entities Evaluated:      {results['total_s1_evaluated']:,}")
    print(f"  - S1 with >= 1 true matches:     {results['matched_s1_count']:,}")
    print(f"  - Genuine no-match S1 entities:  {results['no_match_s1_count']:,} ({results['no_match_s1_count']/results['total_s1_evaluated']*100:.2f}%)")
    print(f"Total Targets Indexed:            {results['total_targets_indexed']:,}")
    print(f"Super-blocks Flagged and Capped:  {results['num_super_blocks']:,}")
    print(f"Execution Time:                   {results['elapsed_seconds']} seconds")

    print("\n" + "-" * 80)
    print(" RECALL PERFORMANCE COMPARISON (V1 vs V2 vs V3)")
    print("-" * 80)
    print(f"Total Ground Truth Pairs:         {results['total_ground_truth_pairs']:,}")
    print(f"Retrieved True Match Pairs:       {results['retrieved_true_pairs']:,} (V1: 3,321 | V2: 3,374)")
    print(f"Missed True Match Pairs:          {results['missed_true_pairs']:,} (V1: 123 | V2: 70)")
    print(f">>> V3 CANDIDATE PAIR RECALL:     {results['blocking_pair_recall_pct']:.2f}% (V1: 96.43% | V2: 97.97%) <<<")
    print(f"Original 123 Misses Recovered:    {results['recovered_from_v1_count']} / 123 ({results['recovered_from_v1_count']/123*100:.2f}%)")
    print(f"70 V2 Misses Recovered:           {results['recovered_from_v2_count']} / 70 ({results['recovered_from_v2_count']/70*100:.2f}%)")

    print("\n" + "-" * 80)
    print(" CANDIDATE DISTRIBUTION & REDUCTION RATIO")
    print("-" * 80)
    print(f"Total Candidate Pairs:            {results['total_candidate_pairs']:,} (V1: 43,936 | V2: 90,685)")
    print(f"Average Candidates per S1:        {results['avg_candidates_per_s1']:.2f} (V1: 43.94 | V2: 90.69)")
    print(f"Median Candidates per S1:         {results['median_candidates_per_s1']:.2f}")
    print(f"95th Percentile Candidates:       {results['p95_candidates_per_s1']:.2f}")
    print(f"Maximum Candidates for any S1:    {results['max_candidates_per_s1']} (Cap: 200)")
    print(f"Reduction Ratio:                  {results['reduction_ratio_pct']:.4f}%")

    print("\n" + "-" * 80)
    print(" PER-PASS RETRIEVALS (TRUE MATCHES FOUND)")
    print("-" * 80)
    for p, cnt in sorted(results["pass_hits"].items()):
        print(f"  - {p:<25}: {cnt:,}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Candidate Blocking V3 Experimental.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"C:\Users\MANOJ KUMAR\Downloads\ml hack dataset\student_resource\dataset\train"),
        help="Path to dataset/train folder",
    )
    parser.add_argument("--sample-size", type=int, default=1_000, help="S1 sample size")
    parser.add_argument("--distractor-size", type=int, default=50_000, help="Distractors to index")
    parser.add_argument("--max-candidates", type=int, default=200, help="Candidate cap")
    parser.add_argument("--max-block-size", type=int, default=1_000, help="Super block threshold")
    parser.add_argument(
        "--missed-report",
        type=Path,
        default=Path(r"C:\Users\MANOJ KUMAR\amazon-ml-challenge-2026\missed_matches_report_v3.tsv"),
        help="Output path for V3 missed matches report",
    )
    parser.add_argument(
        "--v1-missed-report",
        type=Path,
        default=Path(r"C:\Users\MANOJ KUMAR\amazon-ml-challenge-2026\missed_matches_report.tsv"),
        help="Path to baseline V1 missed matches report",
    )
    parser.add_argument(
        "--v2-missed-report",
        type=Path,
        default=Path(r"C:\Users\MANOJ KUMAR\amazon-ml-challenge-2026\missed_matches_report_v2.tsv"),
        help="Path to baseline V2 missed matches report",
    )

    args = parser.parse_args()

    results = evaluate_blocking_v3_on_sample(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        distractor_size=args.distractor_size,
        max_candidates=args.max_candidates,
        max_block_size=args.max_block_size,
        missed_report_file=args.missed_report,
        v1_missed_report_file=args.v1_missed_report,
        v2_missed_report_file=args.v2_missed_report,
    )

    print_v3_report(results)
