# Candidate Blocking Handoff Report: Member 2 to Member 3

**Target Audience:** Member 3 (Matching Model & Pair Scoring)  
**Author:** Member 2 (Candidate Generation & Blocking)  
**Git Branch:** `member2-blocking`  
**Recommended Generator:** **V3 Experimental Candidate Blocker** (`blocking_v3.py`)

---

## 1. Executive Summary & Progression (V1 vs V2 vs V3)

Across three iterative, evidence-backed improvements evaluated against the exact same 1,000-Source-1 benchmark (103,411 target search space, `max_candidates=200`, `max_block_size=1000`, country partitioning enabled), blocking candidate pair recall improved from **96.43%** to **98.43%**, achieving a **56.10% total error reduction** while maintaining a high **99.87% reduction ratio**.

| Metric | V1 Baseline | V2 Experimental | V3 Recommended | Net Improvement (V1 $\rightarrow$ V3) |
|---|:---:|:---:|:---:|:---:|
| **Candidate Pair Recall** | **96.43%** | **97.97%** | **98.43%** | **+2.00% absolute (+56.10% error reduction)** |
| **Total Ground Truth Pairs** | 3,444 | 3,444 | 3,444 | — |
| **Retrieved True Pairs** | 3,321 | 3,374 | **3,390** | **+69 net true matches recovered** |
| **Missed True Pairs** | 123 | 70 | **54** | **Down from 123 to 54 misses** |
| **Original 123 Misses Recovered** | 0 / 123 | 69 / 123 | **92 / 123** | **74.80% overall recovery rate** |
| **Total Candidate Pairs** | 43,936 | 90,685 | 133,675 | Evidence-bounded candidate pool |
| **Average Candidates / S1** | 43.94 | 90.69 | 133.68 | Bounded by hard cap |
| **Median Candidates / S1** | 7.00 | 62.50 | 200.00 | Higher candidate density on multi-match pools |
| **95th Percentile Candidates** | 200.00 | 200.00 | 200.00 | Strict cap enforced |
| **Maximum Candidates per S1** | 200 | 200 | **200** | Strict hard cap enforced |
| **Reduction Ratio** | 99.9575% | 99.9123% | **99.8707%** | **Search space reduced by 99.87%** |
| **Super-Blocks Flagged & Capped** | 0 | 1 | 2 | Safely suppressed |
| **Execution Runtime** | 38.8 s | 38.4 s | **39.92 s** | High-throughput streaming |

---

## 2. Inverted Index Configurations & Safety Limits

- **`max_candidates = 200`:** Hard candidate cap per Source-1 entity. When candidate hits exceed 200, evidence-based ranking ranks candidates by multi-pass agreement, pass priority, and lexical Jaccard overlap before truncating.
- **`max_block_size = 1,000`:** Super-block pruning limit. Any inverted index key exceeding 1,000 entities is automatically deactivated from candidate generation to avoid quadratic Cartesian explosions.
- **`partition_by_country = True`:** Every blocking key is namespaced by ISO-2 country code (`US:`, `IN:`, etc.). 100% of benchmark ground-truth pairs share identical country codes.
- **Memory Safety:** Streaming chunk-based architecture (`chunk_size=100,000`), compact 32-bit unsigned integer arrays (`array.array('I')`) for row references, ensuring under 2.5 GB peak RAM on the 16 GB machine.

---

## 3. All Blocking Passes / Key Families in V3

V3 combines 7 complementary blocking passes to guarantee high coverage across cross-script, missing-address, and typographical variations:

| Pass Identifier | Name & Mechanism | Key Format | Primary Target / Purpose |
|---|---|---|---|
| `pass1` | **Core Name Prefix Bigram** | `{country}:p1_{tok1}_{tok2}` | Matches standard business names on the first two significant tokens. |
| `pass2` | **Address Number + Street / Locality Anchor** | `{country}:p2_{num}_{street/locality}` | Pairs building numbers (up to top 4) with street, locality, and last-segment city tokens. |
| `pass2_locpair` | **Address Locality Pair Fallback** | `{country}:p2_locpair_{loc1}_{loc2}` | Connects cross-script entities with descriptive addresses (no numeric digits). |
| `pass3` | **Sorted Token Fingerprint** | `{country}:p3_{tok_a}_{tok_b}` | Invariable to word order permutations (`Print Ventures EFS` vs `EFS Print Ventures`). |
| `pass4` | **Compressed Domain Stem** | `{country}:p4_{stem12}` | Aligns websites, space-separated domains, and names (`bryansquare com` vs `Bryan Square LLC`). |
| `pass5_distinctive` | **Distinctive Token & 5-Gram Prefix** | `{country}:p_dist_{tok}`, `{country}:p_pfx_{pfx5}` | Captures rare brand names with missing addresses, tolerating trailing typos. |
| `pass6_compound` | **Compound Adjacent Brand Tokens** | `{country}:p_comp_{w0w1}` | Connects collapsed trade names (`First Seven Exports` vs `firstseven`). |

---

## 4. Candidate Provenance Fields

Each generated candidate pair contains complete provenance metadata for Member 3's feature extraction:

1. **`source1_entity_id`** (`str`): Unique entity ID for Source 1 record (e.g. `S1-785847572`).
2. **`candidate_entity_id`** (`str`): Unique entity ID for candidate Target record (e.g. `S3-408212372`).
3. **`candidate_source`** (`str`): Source indicator (`S2` or `S3`).
4. **`blocking_passes`** (`str`): Semicolon-delimited list of passes that generated this pair (e.g. `pass1;pass3;pass5_distinctive`).
5. **`num_passes`** (`int`): Count of distinct matching passes (1 to 7). Highly correlated with true match probability.

---

## 5. How Member 3 Should Call Candidate Generation

### Option A: Python API (Recommended)
```python
from pathlib import Path
from business_entity_resolution.src.blocking_v3 import BlockingConfig, MultiPassBlocker

# 1. Initialize configuration
config = BlockingConfig(
    data_dir=Path("path/to/dataset/train"),  # or dataset/test
    chunk_size=100_000,
    max_candidates=200,
    max_block_size=1_000,
    partition_by_country=True,
)
blocker = MultiPassBlocker(config)

# 2. Stream and index Target files (S2 and S3)
blocker.index_target_file(config.data_dir / "train_source2.tsv", source_label="S2")
blocker.index_target_file(config.data_dir / "train_source3.tsv", source_label="S3")

# 3. Stream Source 1 and write candidate pairs to TSV
total_s1, total_pairs = blocker.generate_candidates(
    s1_file=config.data_dir / "train_source1.tsv",
    output_file=Path("candidate_pairs.tsv"),
)
print(f"Generated {total_pairs:,} candidate pairs across {total_s1:,} S1 entities.")
```

### Option B: Command-Line Interface (CLI)
```bash
python code/business_entity_resolution/src/blocking_v3.py \
    --data-dir "C:\Users\MANOJ KUMAR\Downloads\ml hack dataset\student_resource\dataset\train" \
    --output-file "candidate_pairs.tsv" \
    --chunk-size 100000 \
    --max-candidates 200 \
    --max-block-size 1000
```

---

## 6. Output TSV Schema

The generated candidate file is tab-separated (`\t`), UTF-8 encoded, with the following header:

```tsv
source1_entity_id	candidate_entity_id	candidate_source	blocking_passes	num_passes
S1-925783039	S2-415733261	S2	pass5_distinctive	1
S1-785847572	S3-408212372	S3	pass2;pass2_locpair	2
S1-785847572	S2-508602797	S2	pass1;pass3;pass4	3
```

---

## 7. Known Limitations: The 54 Remaining Benchmark Misses

Member 3 should be aware of the remaining 54 benchmark misses documented in [`missed_matches_report_v3.tsv`](file:///C:/Users/MANOJ%20KUMAR/amazon-ml-challenge-2026/missed_matches_report_v3.tsv):

1. **Address / Locality Key Failure (23 pairs / 42.6%):**
   - E.g. `9236 Meadowmont View Dr` vs `9238 meadowmont view dr` where 1-digit address discrepancies occur on high-frequency names (`Physical Therapy Clinic`).
2. **Cross-Script / Multilingual with Landmark Divergence (16 pairs / 29.6%):**
   - Latin vs Indic (Devanagari, Telugu, Bengali) names where address text differs heavily in transliteration or district division (`At Shahagad` vs `at shahgad`).
3. **Missing Target Address with Multi-Token Corruptions (10 pairs / 18.5%):**
   - Empty target address combined with multi-word typos (`Red Consultants` vs `red pvt ltd center`, `Odonnell Cambridge` vs `odonnell camrie`).
4. **Name Alias / Trade Names (4 pairs / 7.4%):**
   - Acronyms or corporate aliases (`American Legacy Islands` vs `veragildsyn`, `Gopal Yatra` vs `wexsol formerly gopal yatra`).
5. **Multi-Token Typo Corruption (1 pair / 1.9%):**
   - Severe garbling (`feancis and sze pllc` vs `Francis & Sze PLLC`).

---

## 8. Artifact Preservation Status

All baseline and experimental implementations are strictly preserved in the repository:
- **V1 Baseline:** [`code/business_entity_resolution/src/blocking.py`](file:///C:/Users/MANOJ%20KUMAR/amazon-ml-challenge-2026/code/business_entity_resolution/src/blocking.py) (96.43% recall)
- **V2 Experimental:** [`code/business_entity_resolution/src/blocking_v2.py`](file:///C:/Users/MANOJ%20KUMAR/amazon-ml-challenge-2026/code/business_entity_resolution/src/blocking_v2.py) (97.97% recall)
- **V3 Recommended:** [`code/business_entity_resolution/src/blocking_v3.py`](file:///C:/Users/MANOJ%20KUMAR/amazon-ml-challenge-2026/code/business_entity_resolution/src/blocking_v3.py) (98.43% recall)
- **Unit Tests:** All unit test suites pass (`test_preprocessing.py`, `test_blocking.py`, `test_blocking_v2.py`, `test_blocking_v3.py`).

---

## 9. Next Steps: Member 3 Responsibilities

Member 3 takes over from this handoff point. Member 3 is responsible for:
- Generating the candidate pairs for the training / evaluation / test sets using `blocking_v3.py`.
- Computing pairwise lexical, phonological, geographic, and neural/TF-IDF similarity features.
- Training the entity matching / pair classification model.
- Producing the final challenge evaluation artifact `matching_results.tsv` conforming to submission requirements.
