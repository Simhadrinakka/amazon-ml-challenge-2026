# Business Entity Resolution - Amazon ML Challenge 2026

## Overview
This package contains the source code, pipeline scripts, and environment configuration for the **Business Entity Resolution** task in Amazon ML Challenge 2026.

---

## Directory Structure

```
business_entity_resolution/
├── src/                # Core source code modules (preprocessing, blocking, modeling, inference)
├── requirements.txt    # Python dependencies
└── README.md           # Setup and execution instructions
```

---

## Setup & Installation

1. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. **Place datasets in the designated data directory** (outside version control).
2. **Execute inference / pipeline:**
   ```bash
   python src/main.py
   ```
3. **Check output:**
   The output prediction files will be generated in the root `output/` folder.
