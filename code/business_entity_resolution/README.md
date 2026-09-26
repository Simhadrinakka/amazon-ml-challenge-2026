# Business Entity Resolution - Amazon ML Challenge 2026

## Overview
This package contains the source code, pipeline scripts, and environment configuration for the **Business Entity Resolution** task in Amazon ML Challenge 2026.

---

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py              # Member 1: Unicode-safe cleaning & normalization
│   ├── profiling.py                  # Member 1: Streaming dataset profiler
│   ├── test_preprocessing.py         # Member 1: Unit tests for preprocessing
│   ├── validate_preprocessing_scale.py # Member 1: Invariant & scale verification
│   ├── blocking.py                   # Member 2: Baseline multi-pass candidate blocking (V1: 96.43% recall)
│   ├── test_blocking.py              # Member 2: Unit tests for baseline blocking V1
│   ├── evaluate_blocking.py          # Member 2: Baseline evaluation benchmark (V1)
│   ├── blocking_v2.py                # Member 2: V2 experimental candidate blocker (97.97% recall)
│   ├── test_blocking_v2.py           # Member 2: Unit tests for blocking V2
│   ├── evaluate_blocking_v2.py       # Member 2: Evaluation benchmark for V2
│   ├── blocking_v3.py                # Member 2: V3 recommended candidate blocker (98.43% recall)
│   ├── test_blocking_v3.py           # Member 2: Unit tests for blocking V3
│   └── evaluate_blocking_v3.py       # Member 2: Evaluation benchmark for V3
├── MEMBER3_BLOCKING_HANDOFF.md       # Member 2 -> Member 3 Blocking Handoff Guide
├── requirements.txt                  # Python dependencies
└── README.md                         # Setup and execution instructions
```

---

## Setup & Installation

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   .\.venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Testing & Verification

1. **Run Preprocessing Tests (Member 1):**
   ```bash
   python src/test_preprocessing.py
   ```

2. **Run Blocking Tests (Member 2):**
   ```bash
   python src/test_blocking.py
   ```

---

## Candidate Blocking Execution & Evaluation (Member 2)

### 1. Evaluate Candidate Blocking V3 (Recommended)
Evaluates candidate pair recall, candidate counts per S1, and reduction ratio against `train_ground_truth.tsv`:
```bash
python src/evaluate_blocking_v3.py \
    --data-dir "<path_to_dataset>/dataset/train" \
    --sample-size 1000 \
    --distractor-size 50000 \
    --max-candidates 200 \
    --max-block-size 1000
```

### 2. Generate Candidate Pairs for Matching (Member 3 Pipeline)
Using the recommended V3 blocker (98.43% candidate recall):
```bash
python src/blocking_v3.py \
    --data-dir "<path_to_dataset>/dataset/train" \
    --output-file "candidate_pairs.tsv" \
    --chunk-size 100000 \
    --max-candidates 200 \
    --max-block-size 1000
```
Output candidate TSV schema:
- `source1_entity_id`: ID of query entity in Source 1
- `candidate_entity_id`: ID of matched entity in Source 2 or Source 3
- `candidate_source`: Source of target candidate (`S2` or `S3`)
- `blocking_passes`: Semicolon-delimited list of passes that matched (e.g. `pass1;pass2;pass5_distinctive`)
- `num_passes`: Number of distinct blocking passes that produced the candidate

See [MEMBER3_BLOCKING_HANDOFF.md](file:///C:/Users/MANOJ%20KUMAR/amazon-ml-challenge-2026/code/business_entity_resolution/MEMBER3_BLOCKING_HANDOFF.md) for full documentation.
