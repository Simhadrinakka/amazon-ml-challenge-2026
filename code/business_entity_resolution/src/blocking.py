"""
Candidate Blocking Module for Amazon ML Challenge 2026: Business Entity Resolution.

High-recall, memory-bounded multi-pass candidate generation supporting multi-million row datasets.
Features:
- 4 complementary blocking passes (Name Prefix, Address Anchor, Sorted Fingerprint, Compressed Domain Stem).
- Preserves full Unicode and multilingual scripts (Indic, Latin, CJK, etc.) including vowel matras.
- Compact memory-efficient inverted indexing using 32-bit integer arrays (array.array('I')).
- Super-block frequency capping to prevent quadratic Cartesian explosion.
- Evidence-based candidate truncation when candidate pools are exceptionally large.
- Full candidate provenance tracking (recording all matching passes per candidate).
- Memory-safe streaming in chunks for both indexing (S2/S3) and candidate generation (S1).
"""

import array
import argparse
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Ensure parent directory is in path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from preprocessing import (
    basic_clean_text,
    is_missing,
    normalize_address,
    normalize_business_name,
    normalize_country,
)

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns and Stopword Sets for Blocking
# ---------------------------------------------------------------------------

# Legal / Corporate stopwords that offer little distinction for entity indexing
LEGAL_STOPWORDS: Set[str] = {
    # English legal / corporate tokens
    "inc", "corp", "corporation", "incorporated", "llc", "llp", "ltd", "limited",
    "pvt", "private", "plc", "co", "company", "gmbh", "sa", "enterprises",
    "enterprise", "services", "service", "solutions", "solution", "tech",
    "technologies", "technology", "intl", "international", "the", "and", "of",
    "in", "group", "holdings", "proprietorship", "prop", "associates",
    "consulting", "consultants", "management", "industries", "trading",
    # Indic transliterated legal suffixes (Hindi/Devanagari, Tamil, etc.)
    "प्राइवेट", "लिमिटेड", "एलएलपी", "कंपनी", "एंटरप्राइजेज", "सर्विस",
    "लिமிடெட்", "பிரைவேட்", "எல்எல்பி",
}

# Generic address tokens to exclude when building address anchor keys
ADDR_STOPWORDS: Set[str] = {
    "st", "rd", "ave", "dr", "ln", "blvd", "unit", "apt", "ste", "null",
    "floor", "fl", "bldg", "room", "dept", "kh", "no", "plot", "door", "h",
    "sec", "blk", "phase", "post", "box", "pobox", "po", "street", "road",
    "avenue", "drive", "lane", "boulevard", "suite", "apartment", "building",
    "near", "opp", "opposite", "behind", "beside", "main", "cross",
}

# Regex for extracting digits / house / plot / flat numbers
RE_NUM = re.compile(r"\b\d+(?:[-/]\d+)?\b")

# Regex for domain extensions
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|in|co|io|biz|info|edu|gov|eu|uk|fr|de)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Unicode-Safe Word Tokenization & Character Cleaning Helpers
# ---------------------------------------------------------------------------

def clean_token(token: str) -> str:
    """Strip leading and trailing punctuation while preserving letters, numbers, and marks."""
    return token.strip(" ,.-/#()[]{}|\\\"'`~:;!?<>@#$%^&*")


def strip_latin_diacritics(text: str) -> str:
    """
    Remove accent marks / diacritics ONLY from Latin base characters (e.g. 'bóral' -> 'boral',
    'énterprises' -> 'enterprises', 'gríll' -> 'grill') while preserving Indic matras
    and non-Latin combining marks intact.
    """
    if not text:
        return ""
    result = []
    for c in unicodedata.normalize("NFD", text):
        if unicodedata.category(c) == "Mn":
            # Strip combining mark only if preceding character is Latin letter
            if result and "a" <= result[-1].lower() <= "z":
                continue
        result.append(c)
    return unicodedata.normalize("NFC", "".join(result))


def tokenize_unicode_words(text: str) -> List[str]:
    """
    Split text into words preserving all Unicode letters, numbers, and combining vowel marks (matras).
    Crucial for non-Latin and Indic scripts (Devanagari, Tamil, Telugu, etc.).
    """
    if not text:
        return []
    cleaned_latin = strip_latin_diacritics(text)
    words: List[str] = []
    for token in cleaned_latin.split():
        clean_tok = clean_token(token)
        if clean_tok:
            words.append(clean_tok)
    return words


def clean_alphanumeric_unicode(text: str) -> str:
    """
    Preserve all Unicode letters (L), numbers (N), and combining marks/matras (M).
    Used for domain stems and compressed alphanumeric fingerprints.
    """
    if not text:
        return ""
    cleaned_latin = strip_latin_diacritics(text)
    return "".join(c for c in cleaned_latin if unicodedata.category(c)[0] in ("L", "N", "M"))


# ---------------------------------------------------------------------------
# Key Extraction Functions for the 4 Blocking Passes
# ---------------------------------------------------------------------------

def extract_pass1_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 1: Normalized Core Name Prefix / Significant Token Blocking.

    Extracts the first 2 significant tokens from the normalized business name.
    If only 1 significant token exists, uses that single token.
    Essential for matching records when address is missing or empty (~3.5% of S2/S3).
    """
    if not norm_name or is_missing(norm_name):
        return []

    all_words = tokenize_unicode_words(norm_name)
    words = [
        w for w in all_words
        if w.lower() not in LEGAL_STOPWORDS and len(w) >= 2
    ]
    if not words:
        words = [w for w in all_words if len(w) >= 2]
    if not words:
        return []

    prefix = f"{country}:" if country else ""
    if len(words) >= 2:
        return [f"{prefix}p1_{words[0]}_{words[1]}"]
    return [f"{prefix}p1_{words[0]}"]


def extract_pass2_keys(norm_addr: str, country: str = "") -> List[str]:
    """
    Pass 2: Address Number + Locality / Street Anchor Blocking.

    Extracts street/building/plot numbers (normalized without leading zeros) paired with:
    1) Primary street token from the first segment.
    2) Locality / City tokens from subsequent comma-separated segments.
    Also emits a number-independent street + locality anchor to catch street-number typos.
    Essential for cross-script matches (e.g. Tamil name vs English name)
    and trade-name aliases where business name differs but physical location matches.
    """
    if not norm_addr or is_missing(norm_addr):
        return []

    # Extract all numbers, stripping leading zeros and trailing characters (e.g. '0684' -> '684', '1056c' -> '1056')
    raw_num_matches = re.findall(r"\b[a-zA-Z]*-?0*(\d+)[a-zA-Z]*\b", norm_addr)
    unique_nums = list(dict.fromkeys(raw_num_matches))[:2]  # Top 2 numbers

    prefix = f"{country}:" if country else ""
    keys: List[str] = []

    parts = [p.strip() for p in norm_addr.split(",") if p.strip()]

    # Extract street tokens from part 0
    p0_words = []
    if parts:
        p0_words = [
            clean_token(w) for w in parts[0].split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]

    # Extract locality tokens from subsequent comma-separated parts
    subsequent_words = []
    for part in parts[1:]:
        p_words = [
            clean_token(w) for w in part.split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]
        subsequent_words.extend(p_words)

    if unique_nums:
        for num in unique_nums:
            # 1) Number + Street token
            if p0_words:
                keys.append(f"{prefix}p2_{num}_{p0_words[0]}")
            # 2) Number + Locality tokens (check first 3 and last token)
            target_locs = list(subsequent_words[:3])
            if subsequent_words and subsequent_words[-1] not in target_locs:
                target_locs.append(subsequent_words[-1])
            for tok in target_locs:
                keys.append(f"{prefix}p2_{num}_{tok}")
    else:
        # Fallback when no digits exist in address: combine first two significant words
        all_words = [
            clean_token(w) for w in norm_addr.split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
        ]
        if len(all_words) >= 2:
            keys.append(f"{prefix}p2_{all_words[0]}_{all_words[1]}")

    # 3) Number-independent street + locality anchor (catches street number typos like 870 vs 8706)
    if p0_words and subsequent_words:
        keys.append(f"{prefix}p2_loc_{p0_words[0]}_{subsequent_words[0]}")

    # Deduplicate while preserving insertion order
    return list(dict.fromkeys(keys))


def extract_pass3_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 3: Sorted Significant-Name-Token Fingerprint Blocking.

    Sorts significant alphanumeric tokens alphabetically and joins top tokens.
    Invariable to word-order permutations (e.g. 'Print Ventures EFS' vs 'EFS Print Ventures')
    and prefix/suffix relocations.

    Parameters
    ----------
    norm_name : str
        Normalized business name.
    country : str, optional
        Normalized ISO-2 country code for namespacing.

    Returns
    -------
    List[str]
        List of generated Pass 3 keys.
    """
    if not norm_name or is_missing(norm_name):
        return []

    all_words = tokenize_unicode_words(norm_name)
    sig_words = [
        w for w in all_words
        if w.lower() not in LEGAL_STOPWORDS and len(w) >= 3
    ]
    if not sig_words:
        sig_words = [w for w in all_words if len(w) >= 3]
    if not sig_words:
        return []

    prefix = f"{country}:" if country else ""
    sorted_words = sorted(sig_words)

    if len(sorted_words) >= 2:
        return [f"{prefix}p3_{sorted_words[0]}_{sorted_words[1]}"]
    return [f"{prefix}p3_{sorted_words[0]}"]


def extract_pass4_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 4: Compressed Alphanumeric Name / Domain Stem Blocking.

    Strips domain endings (.com, .org, etc.), removes all non-alphanumerics,
    and extracts a 12-character prefix stem while preserving all Unicode marks.
    Matches website URLs to trade names (e.g. 'maurewilliamscolombier.com' -> 'maurewilliam')
    and handles hyphenated or joined business names across scripts.

    Parameters
    ----------
    norm_name : str
        Normalized business name.
    country : str, optional
        Normalized ISO-2 country code for namespacing.

    Returns
    -------
    List[str]
        List of generated Pass 4 keys.
    """
    if not norm_name or is_missing(norm_name):
        return []

    s_clean = RE_DOMAIN_SUFFIX.sub("", norm_name)
    alphanum = clean_alphanumeric_unicode(s_clean)
    if len(alphanum) < 4:
        return []

    prefix = f"{country}:" if country else ""
    stem = alphanum[:12]
    return [f"{prefix}p4_{stem}"]


def extract_all_blocking_keys(
    norm_name: str,
    norm_addr: str,
    country: str = "",
    partition_by_country: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate all 4 blocking pass keys for a given record.

    Parameters
    ----------
    norm_name : str
        Normalized business name.
    norm_addr : str
        Normalized business address.
    country : str, default ""
        Country code.
    partition_by_country : bool, default True
        Whether to namespace keys with country code.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping pass names ('pass1', 'pass2', 'pass3', 'pass4') to key lists.
    """
    c_code = country if partition_by_country else ""
    return {
        "pass1": extract_pass1_keys(norm_name, country=c_code),
        "pass2": extract_pass2_keys(norm_addr, country=c_code),
        "pass3": extract_pass3_keys(norm_name, country=c_code),
        "pass4": extract_pass4_keys(norm_name, country=c_code),
    }


# ---------------------------------------------------------------------------
# Compact Inverted Index Structure
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Memory-efficient inverted index storing target entity references as unsigned 32-bit integers.
    Uses array.array('I') to consume only 4 bytes per indexed reference.
    Enforces super-block frequency caps to prevent quadratic pair explosion.
    """

    def __init__(self, max_block_size: int = 1_000):
        self.max_block_size = max_block_size
        self.index: Dict[str, array.array] = {}
        self.super_blocks: Set[str] = set()
        self.target_ids: List[str] = []
        self.target_sources: List[str] = []
        # Store compact normalized representation for evidence-based ranking if capping occurs
        self.target_names: List[str] = []
        self.target_addrs: List[str] = []

    def add_target(
        self,
        entity_id: str,
        source: str,
        norm_name: str,
        norm_addr: str,
        keys_dict: Dict[str, List[str]],
    ) -> int:
        """
        Add a target entity (from Source 2 or Source 3) to the index.

        Parameters
        ----------
        entity_id : str
            Entity ID (e.g. 'S2-12345').
        source : str
            Source identifier ('S2' or 'S3').
        norm_name : str
            Normalized business name.
        norm_addr : str
            Normalized business address.
        keys_dict : Dict[str, List[str]]
            Blocking keys for each pass.

        Returns
        -------
        int
            Internal integer row index assigned to this target entity.
        """
        target_idx = len(self.target_ids)
        self.target_ids.append(entity_id)
        self.target_sources.append(source)
        self.target_names.append(norm_name)
        self.target_addrs.append(norm_addr)

        for pass_name, keys in keys_dict.items():
            for key in keys:
                if key in self.super_blocks:
                    continue

                if key not in self.index:
                    self.index[key] = array.array("I", [target_idx])
                else:
                    arr = self.index[key]
                    if len(arr) >= self.max_block_size:
                        # Reached super-block limit: freeze and mark
                        self.super_blocks.add(key)
                        del self.index[key]
                    else:
                        arr.append(target_idx)

        return target_idx

    def query(
        self,
        query_keys_dict: Dict[str, List[str]],
    ) -> Dict[int, List[str]]:
        """
        Query the index with an S1 entity's pass keys.

        Returns
        -------
        Dict[int, List[str]]
            Mapping of target integer index -> list of matched pass names (e.g. ['pass1', 'pass3']).
        """
        candidate_hits: Dict[int, List[str]] = defaultdict(list)

        for pass_name, keys in query_keys_dict.items():
            for key in keys:
                if key in self.super_blocks or key not in self.index:
                    continue
                arr = self.index[key]
                for tidx in arr:
                    candidate_hits[tidx].append(pass_name)

        return candidate_hits

    def clear(self) -> None:
        """Clear all indexed structures to release memory."""
        self.index.clear()
        self.super_blocks.clear()
        self.target_ids.clear()
        self.target_sources.clear()
        self.target_names.clear()
        self.target_addrs.clear()

    def size(self) -> int:
        return len(self.target_ids)

    def num_keys(self) -> int:
        return len(self.index)

    def memory_usage_mb(self) -> float:
        """Estimate index memory in megabytes."""
        bytes_total = sys.getsizeof(self.index)
        for k, v in self.index.items():
            bytes_total += sys.getsizeof(k) + v.buffer_info()[1] * v.itemsize
        return bytes_total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Evidence-Based Ranking & Scoring for Truncation
# ---------------------------------------------------------------------------

def fast_jaccard_similarity(s1: str, s2: str) -> float:
    """Compute fast token-level Jaccard similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def rank_candidates_by_evidence(
    candidate_hits: Dict[int, List[str]],
    s1_name: str,
    s1_addr: str,
    index: InvertedIndex,
    max_candidates: int,
) -> List[Tuple[int, List[str]]]:
    """
    Evidence-based candidate ranking when candidate set exceeds max_candidates.

    Prioritizes:
    1. Multi-pass agreement (number of distinct matching passes).
    2. Pass priority (Pass 1 / Pass 2 agreement).
    3. Fast lexical token similarity between names and addresses.

    Never discards candidates based purely on arbitrary insertion order.
    """
    scored_candidates = []
    for tidx, passes in candidate_hits.items():
        unique_passes = list(dict.fromkeys(passes))
        num_passes = len(unique_passes)

        # Base score driven by multi-pass support
        score = num_passes * 100.0

        # Pass 1 and Pass 2 carry high precision signal
        if "pass1" in unique_passes:
            score += 20.0
        if "pass2" in unique_passes:
            score += 25.0

        # Fast string overlap verification
        cand_name = index.target_names[tidx] if tidx < len(index.target_names) else ""
        cand_addr = index.target_addrs[tidx] if tidx < len(index.target_addrs) else ""

        name_sim = fast_jaccard_similarity(s1_name, cand_name)
        addr_sim = fast_jaccard_similarity(s1_addr, cand_addr)
        score += name_sim * 10.0 + addr_sim * 5.0

        scored_candidates.append((score, tidx, unique_passes))

    # Sort descending by evidence score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    return [(tidx, passes) for _, tidx, passes in scored_candidates[:max_candidates]]


# ---------------------------------------------------------------------------
# High-Level MultiPassBlocker Class
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    """Configuration parameters for candidate blocking."""
    data_dir: Path
    output_dir: Optional[Path] = None
    chunk_size: int = 100_000
    max_candidates: int = 200
    max_block_size: int = 1_000
    partition_by_country: bool = True


class MultiPassBlocker:
    """
    Manages building inverted indexes across target sources (S2/S3)
    and streaming candidate pairs for query entities (S1).
    """

    def __init__(self, config: BlockingConfig):
        self.config = config
        self.index = InvertedIndex(max_block_size=config.max_block_size)

    def index_target_file(
        self,
        file_path: Path,
        source_label: str,
        target_country: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> int:
        """
        Stream a target TSV file (Source 2 or Source 3) in chunks and add to inverted index.

        Parameters
        ----------
        file_path : Path
            Path to train_source2.tsv or train_source3.tsv.
        source_label : str
            'S2' or 'S3'.
        target_country : Optional[str]
            If specified, only records matching this normalized country will be indexed.
        max_rows : Optional[int]
            Optional cap on rows for testing/validation.

        Returns
        -------
        int
            Total target rows indexed.
        """
        indexed_rows = 0

        reader = pd.read_csv(
            file_path,
            sep="\t",
            chunksize=self.config.chunk_size,
            dtype=str,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="warn",
        )

        for chunk in reader:
            if max_rows and indexed_rows >= max_rows:
                break

            # Fast numpy-backed array iteration
            eids = chunk["entity_id"].values
            names = chunk["business_name"].values
            addrs = chunk["business_address"].values
            ctrys = chunk["country"].values

            num_in_chunk = len(eids)
            for i in range(num_in_chunk):
                if max_rows and indexed_rows >= max_rows:
                    break

                raw_c = ctrys[i]
                norm_c = normalize_country(raw_c) if raw_c and not is_missing(raw_c) else ""

                if target_country and norm_c != target_country:
                    continue

                raw_n = names[i]
                norm_n = normalize_business_name(raw_n) if raw_n and not is_missing(raw_n) else ""

                raw_a = addrs[i]
                norm_a = normalize_address(raw_a) if raw_a and not is_missing(raw_a) else ""

                keys_dict = extract_all_blocking_keys(
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    country=norm_c,
                    partition_by_country=self.config.partition_by_country,
                )

                self.index.add_target(
                    entity_id=eids[i],
                    source=source_label,
                    norm_name=norm_n,
                    norm_addr=norm_a,
                    keys_dict=keys_dict,
                )
                indexed_rows += 1

        return indexed_rows

    def query_source1_chunk(
        self,
        df_chunk: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Process an S1 DataFrame chunk and generate deduplicated candidate pairs.

        Returns
        -------
        List[Dict[str, Any]]
            List of candidate pair dicts containing:
            - source1_entity_id
            - candidate_entity_id
            - candidate_source
            - blocking_passes
            - num_passes
        """
        candidate_rows: List[Dict[str, Any]] = []

        eids = df_chunk["entity_id"].values
        names = df_chunk["business_name"].values
        addrs = df_chunk["business_address"].values
        ctrys = df_chunk["country"].values

        for i in range(len(eids)):
            s1_id = eids[i]
            raw_c = ctrys[i]
            norm_c = normalize_country(raw_c) if raw_c and not is_missing(raw_c) else ""

            raw_n = names[i]
            norm_n = normalize_business_name(raw_n) if raw_n and not is_missing(raw_n) else ""

            raw_a = addrs[i]
            norm_a = normalize_address(raw_a) if raw_a and not is_missing(raw_a) else ""

            keys_dict = extract_all_blocking_keys(
                norm_name=norm_n,
                norm_addr=norm_a,
                country=norm_c,
                partition_by_country=self.config.partition_by_country,
            )

            hits = self.index.query(keys_dict)
            if not hits:
                continue

            # Deduplication and evidence-based capping
            if len(hits) <= self.config.max_candidates:
                selected_cands = [
                    (tidx, list(dict.fromkeys(passes)))
                    for tidx, passes in hits.items()
                ]
            else:
                selected_cands = rank_candidates_by_evidence(
                    candidate_hits=hits,
                    s1_name=norm_n,
                    s1_addr=norm_a,
                    index=self.index,
                    max_candidates=self.config.max_candidates,
                )

            for tidx, passes in selected_cands:
                cand_id = self.index.target_ids[tidx]
                cand_source = self.index.target_sources[tidx]
                candidate_rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "candidate_source": cand_source,
                    "blocking_passes": ";".join(passes),
                    "num_passes": len(passes),
                })

        return candidate_rows

    def generate_candidates(
        self,
        s1_file: Path,
        output_file: Optional[Path] = None,
        max_s1_rows: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Stream S1 file and write candidates to output TSV.

        Parameters
        ----------
        s1_file : Path
            Path to train_source1.tsv or test_source1.tsv.
        output_file : Optional[Path]
            Destination path for candidate TSV.
        max_s1_rows : Optional[int]
            Optional cap on S1 rows processed.

        Returns
        -------
        Tuple[int, int]
            (total_s1_processed, total_candidate_pairs_generated)
        """
        total_s1 = 0
        total_pairs = 0
        header_written = False

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            out_f = open(output_file, "w", encoding="utf-8")
        else:
            out_f = None

        try:
            reader = pd.read_csv(
                s1_file,
                sep="\t",
                chunksize=self.config.chunk_size,
                dtype=str,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="warn",
            )

            for chunk in reader:
                if max_s1_rows and total_s1 >= max_s1_rows:
                    break

                if max_s1_rows and (total_s1 + len(chunk)) > max_s1_rows:
                    chunk = chunk.head(max_s1_rows - total_s1)

                candidates = self.query_source1_chunk(chunk)
                total_s1 += len(chunk)
                total_pairs += len(candidates)

                if out_f and candidates:
                    cand_df = pd.DataFrame(candidates)
                    cand_df.to_csv(
                        out_f,
                        sep="\t",
                        index=False,
                        header=not header_written,
                    )
                    header_written = True

        finally:
            if out_f:
                out_f.close()

        return total_s1, total_pairs


# ---------------------------------------------------------------------------
# CLI Entrypoint for Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Candidate Blocking Generator for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory containing train or test TSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="candidates.tsv",
        help="Output candidate pairs TSV file path (default: candidates.tsv).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for streaming processing (default: 100000).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200,
        help="Conservative evidence-based candidate cap per S1 entity (default: 200).",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=1_000,
        help="Super-block frequency limit before a key is pruned (default: 1000).",
    )
    parser.add_argument(
        "--disable-country-partition",
        action="store_true",
        help="Disable country-based key namespacing.",
    )
    parser.add_argument(
        "--max-s1-rows",
        type=int,
        default=None,
        help="Optional row limit on Source 1 for benchmarking/testing.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    s1_file = data_dir / "train_source1.tsv"
    s2_file = data_dir / "train_source2.tsv"
    s3_file = data_dir / "train_source3.tsv"

    if not s1_file.exists():
        # Check if test split
        s1_file = data_dir / "test_source1.tsv"
        s2_file = data_dir / "test_source2.tsv"
        s3_file = data_dir / "test_source3.tsv"

    if not s1_file.exists() or not s2_file.exists() or not s3_file.exists():
        print(f"Error: Missing source TSV files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    config = BlockingConfig(
        data_dir=data_dir,
        chunk_size=args.chunk_size,
        max_candidates=args.max_candidates,
        max_block_size=args.max_block_size,
        partition_by_country=not args.disable_country_partition,
    )

    blocker = MultiPassBlocker(config)

    print("\n" + "=" * 80)
    print(" CANDIDATE BLOCKING PIPELINE")
    print(f" Data Directory:       {data_dir}")
    print(f" Chunk Size:           {config.chunk_size:,}")
    print(f" Max Candidates / S1:  {config.max_candidates}")
    print(f" Super-block Limit:    {config.max_block_size}")
    print(f" Country Partitioning: {config.partition_by_country}")
    print("=" * 80 + "\n")

    # Step 1: Index Source 2
    t0 = time.time()
    print(f">>> Indexing '{s2_file.name}'...")
    s2_count = blocker.index_target_file(s2_file, source_label="S2")
    t1 = time.time()
    print(f"    Indexed {s2_count:,} Source 2 entities in {t1 - t0:.2f}s ({s2_count/(t1-t0):,.1f} rows/s)")
    print(f"    Current index keys: {blocker.index.num_keys():,}, Estimated RAM: {blocker.index.memory_usage_mb():.1f} MB")

    # Step 2: Index Source 3
    t2 = time.time()
    print(f"\n>>> Indexing '{s3_file.name}'...")
    s3_count = blocker.index_target_file(s3_file, source_label="S3")
    t3 = time.time()
    print(f"    Indexed {s3_count:,} Source 3 entities in {t3 - t2:.2f}s ({s3_count/(t3-t2):,.1f} rows/s)")
    print(f"    Total index keys: {blocker.index.num_keys():,}, Estimated RAM: {blocker.index.memory_usage_mb():.1f} MB")
    print(f"    Super-blocks flagged and pruned: {len(blocker.index.super_blocks):,}")

    # Step 3: Query Source 1 and emit candidate pairs
    out_path = Path(args.output_file).resolve()
    t4 = time.time()
    print(f"\n>>> Querying '{s1_file.name}' and writing candidates to '{out_path.name}'...")
    total_s1, total_pairs = blocker.generate_candidates(
        s1_file=s1_file,
        output_file=out_path,
        max_s1_rows=args.max_s1_rows,
    )
    t5 = time.time()

    avg_cands = total_pairs / total_s1 if total_s1 > 0 else 0.0
    print("\n" + "=" * 80)
    print(" BLOCKING RUN SUMMARY")
    print(f" Source 1 Entities Processed:   {total_s1:,}")
    print(f" Total Candidate Pairs Created: {total_pairs:,}")
    print(f" Average Candidates per S1:     {avg_cands:.2f}")
    print(f" Candidate Generation Time:     {t5 - t4:.2f} seconds ({total_s1/(t5-t4):,.1f} S1/s)")
    print(f" Output File Size:              {out_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
