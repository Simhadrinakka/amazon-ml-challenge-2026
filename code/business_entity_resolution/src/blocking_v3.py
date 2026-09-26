"""
Candidate Blocking Module (V3 Experimental) for Amazon ML Challenge 2026: Business Entity Resolution.

Builds upon V2 while targeting the remaining 70 misses with evidence-backed enhancements:
1. Cross-script / Multilingual:
   - Conservative address locality fallback using pairs of distinct meaningful locality/city tokens
     for descriptive addresses without numeric digits.
2. Missing Target Address:
   - 5-gram prefix key for distinctive business-name tokens (len >= 6) to tolerate minor typos.
   - Emits distinctive tokens across all significant words without single-token truncation.
3. Compound Brand / Acronym Alignment:
   - Concatenates adjacent significant tokens (e.g., 'First Seven Exports' -> 'firstseven')
     to match collapsed trade-names and brand compounds.
4. Country Partitioning, Generic Stopword Filtering, and Super-Block Protection strictly enforced.
5. Preserves candidate cap (max_candidates=200).
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
# Pre-compiled Regex Patterns and Stopword Sets
# ---------------------------------------------------------------------------

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

HONORIFIC_PREFIXES: Set[str] = {
    "dr", "dr.", "smt", "smt.", "shri", "shri.", "sri", "sri.", "mr", "mr.", "mrs", "mrs.", "ms", "ms."
}

ADDR_STOPWORDS: Set[str] = {
    "st", "rd", "ave", "dr", "ln", "blvd", "unit", "apt", "ste", "null",
    "floor", "fl", "bldg", "room", "dept", "kh", "no", "plot", "door", "h",
    "sec", "blk", "phase", "post", "box", "pobox", "po", "street", "road",
    "avenue", "drive", "lane", "boulevard", "suite", "apartment", "building",
    "near", "opp", "opposite", "behind", "beside", "main", "cross",
}

GENERIC_DISTINCTIVE_STOPWORDS: Set[str] = LEGAL_STOPWORDS | {
    "center", "centre", "group", "global", "direct", "market", "marketing",
    "service", "services", "management", "industries", "industry", "trading",
    "national", "federal", "general", "standard", "universal", "commercial",
    "financial", "finance", "capital", "logistics", "properties", "property",
    "holdings", "holding", "ventures", "venture", "systems", "system",
    "products", "product", "company", "eastern", "western", "northern",
    "southern", "corporation", "enterprises", "enterprise", "solutions",
    "solution", "international", "associates", "consulting", "consultants",
    "technologies", "technology", "limited", "private", "public", "producer",
    "foundation", "society", "hospital", "clinic", "medical", "health",
    "healthcare", "pharma", "pharmaceuticals", "construction", "developers",
    "development", "project", "projects", "engineering", "security",
    "communications", "telecom", "agency", "agencies", "supply", "supplies",
    "distributors", "distribution", "energy", "power", "infra", "infrastructure",
    "export", "exports", "surgical", "surgery", "therapy", "physical",
    "food", "foods", "india", "american", "america", "united", "states",
}

RE_NUM = re.compile(r"\b\d+(?:[-/]\d+)?\b")

RE_DOMAIN_SUFFIX = re.compile(
    r"(?:[\.\s]+)(?:com|org|net|in|co|io|biz|info|edu|gov|eu|uk|fr|de)\b",
    re.IGNORECASE
)

RE_CORP_DESIGNATOR = re.compile(
    r"\b(?:llc|inc|ltd|corp|co|pvt|gmbh|sa)\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Unicode-Safe Word Tokenization & Character Cleaning Helpers
# ---------------------------------------------------------------------------

def clean_token(token: str) -> str:
    return token.strip(" ,.-/#()[]{}|\\\"'`~:;!?<>@#$%^&*")


def strip_latin_diacritics(text: str) -> str:
    if not text:
        return ""
    result = []
    for c in unicodedata.normalize("NFD", text):
        if unicodedata.category(c) == "Mn":
            if result and "a" <= result[-1].lower() <= "z":
                continue
        result.append(c)
    return unicodedata.normalize("NFC", "".join(result))


def tokenize_unicode_words(text: str) -> List[str]:
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
    if not text:
        return ""
    cleaned_latin = strip_latin_diacritics(text)
    return "".join(c for c in cleaned_latin if unicodedata.category(c)[0] in ("L", "N", "M"))


def normalize_leetspeak_token(token: str) -> str:
    """Conservative leetspeak normalization for tokens with letters."""
    if not token or not any(c.isalpha() for c in token):
        return token
    res = []
    for c in token:
        if c == '5':
            res.append('s')
        elif c == '0':
            res.append('o')
        elif c == '1':
            res.append('l')
        else:
            res.append(c)
    return "".join(res)


def normalize_business_name_v3(name: str) -> str:
    """Enhanced business name normalization for V3."""
    if not name or is_missing(name):
        return ""
    base_norm = normalize_business_name(name)
    tokens = base_norm.split()
    while tokens and tokens[0].lower().rstrip(".") in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    cleaned_tokens = [normalize_leetspeak_token(t) for t in tokens]
    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# Key Extraction Functions for V3 Blocking Passes
# ---------------------------------------------------------------------------

def extract_pass1_keys(norm_name: str, country: str = "") -> List[str]:
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
    """Address Number + Street / Locality Anchor Blocking (from V2)."""
    if not norm_addr or is_missing(norm_addr):
        return []

    raw_num_matches = re.findall(r"\b[a-zA-Z]*-?0*(\d+)[a-zA-Z]*\b", norm_addr)
    unique_nums = list(dict.fromkeys(raw_num_matches))[:4]

    prefix = f"{country}:" if country else ""
    keys: List[str] = []

    parts = [p.strip() for p in norm_addr.split(",") if p.strip()]

    p0_words = []
    if parts:
        p0_words = [
            clean_token(w) for w in parts[0].split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]

    subsequent_words = []
    for part in parts[1:]:
        p_words = [
            clean_token(w) for w in part.split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]
        subsequent_words.extend(p_words)

    last_words = []
    if len(parts) >= 2:
        if len(parts) >= 3:
            p_prev = [
                clean_token(w).lower() for w in parts[-2].split()
                if clean_token(w).lower() not in ADDR_STOPWORDS
                and len(clean_token(w)) >= 3
                and not any(c.isdigit() for c in clean_token(w))
            ]
            last_words.extend(p_prev)
        p_last = [
            clean_token(w).lower() for w in parts[-1].split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]
        last_words.extend(p_last)

    if unique_nums:
        for num in unique_nums:
            if p0_words:
                keys.append(f"{prefix}p2_{num}_{p0_words[0]}")
            target_locs = list(subsequent_words[:3])
            if subsequent_words and subsequent_words[-1] not in target_locs:
                target_locs.append(subsequent_words[-1])
            for tok in target_locs:
                keys.append(f"{prefix}p2_{num}_{tok}")
            for tok in last_words[:2]:
                keys.append(f"{prefix}p2_{num}_{tok}")
    else:
        all_words = [
            clean_token(w) for w in norm_addr.split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
        ]
        if len(all_words) >= 2:
            keys.append(f"{prefix}p2_{all_words[0]}_{all_words[1]}")

    if p0_words and subsequent_words:
        keys.append(f"{prefix}p2_loc_{p0_words[0]}_{subsequent_words[0]}")
    if p0_words and last_words and last_words[-1] not in subsequent_words[:1]:
        keys.append(f"{prefix}p2_loc_{p0_words[0]}_{last_words[-1]}")

    return list(dict.fromkeys(keys))


def extract_pass2_locality_pair_keys(norm_addr: str, country: str = "") -> List[str]:
    """
    Pass 2B (V3): Conservative Address Locality Fallback.
    Forms pairs of distinct meaningful locality tokens for non-numeric/descriptive addresses.
    Connects cross-script records where numbers are missing but city/district text aligns.
    """
    if not norm_addr or is_missing(norm_addr):
        return []
    parts = [p.strip() for p in norm_addr.split(",") if p.strip()]
    if len(parts) < 2:
        return []

    loc_tokens: List[str] = []
    for p in parts[1:]:
        for w in p.split():
            cw = clean_token(w).lower()
            if cw not in ADDR_STOPWORDS and len(cw) >= 4 and not any(c.isdigit() for c in cw):
                if cw not in loc_tokens:
                    loc_tokens.append(cw)

    if len(loc_tokens) < 2:
        return []

    prefix = f"{country}:" if country else ""
    keys: List[str] = []

    # Emit pair of top 2 locality tokens (alphabetically sorted)
    t1 = loc_tokens[0]
    t2 = loc_tokens[1]
    s1 = sorted([t1, t2])
    keys.append(f"{prefix}p2_locpair_{s1[0]}_{s1[1]}")

    # If 3 or more locality tokens, also pair first with last
    if len(loc_tokens) >= 3:
        t3 = loc_tokens[-1]
        if t3 != t2:
            s2 = sorted([t1, t3])
            keys.append(f"{prefix}p2_locpair_{s2[0]}_{s2[1]}")

    return list(dict.fromkeys(keys))


def extract_pass3_keys(norm_name: str, country: str = "") -> List[str]:
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
    if not norm_name or is_missing(norm_name):
        return []
    s_clean = RE_DOMAIN_SUFFIX.sub("", norm_name)
    s_clean = RE_CORP_DESIGNATOR.sub("", s_clean)
    alphanum = clean_alphanumeric_unicode(s_clean)
    if len(alphanum) < 4:
        return []
    prefix = f"{country}:" if country else ""
    stem = alphanum[:12]
    return [f"{prefix}p4_{stem}"]


def extract_distinctive_token_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 5 (V3): Distinctive Business-Name Token & Prefix Blocking.
    Emits full token and 5-gram prefix key for rare tokens (len >= 6) to tolerate minor typos.
    """
    if not norm_name or is_missing(norm_name):
        return []

    words = tokenize_unicode_words(norm_name)
    distinctive = [
        w for w in words
        if len(w) >= 5
        and w.lower() not in GENERIC_DISTINCTIVE_STOPWORDS
        and not any(c.isdigit() for c in w)
    ]
    if not distinctive:
        return []

    prefix = f"{country}:" if country else ""
    keys: List[str] = []

    # Emit up to top 2 distinctive tokens
    sorted_by_len = sorted(distinctive, key=lambda x: len(x), reverse=True)
    for tok in sorted_by_len[:2]:
        keys.append(f"{prefix}p_dist_{tok.lower()}")
        # For tokens of length >= 6, emit 5-character prefix key to tolerate trailing typos
        if len(tok) >= 6:
            keys.append(f"{prefix}p_pfx_{tok[:5].lower()}")

    return list(dict.fromkeys(keys))


def extract_compound_brand_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 6 (V3): Compound Brand / Acronym Alignment.
    Connects 'First Seven Exports' -> 'firstseven', 'Bn Technologies' -> 'bntechnologies'.
    """
    if not norm_name or is_missing(norm_name):
        return []

    words = tokenize_unicode_words(norm_name)
    # Exclude only strictly formal legal endings (ltd, inc, llc, pvt)
    core_words = [
        w for w in words
        if w.lower() not in {"ltd", "limited", "pvt", "private", "inc", "corp", "llc", "llp", "plc", "co", "company"}
        and len(w) >= 2
    ]
    if not core_words:
        return []

    prefix = f"{country}:" if country else ""
    keys: List[str] = []

    # If already a single compound word (length >= 6)
    if len(core_words) == 1 and len(core_words[0]) >= 6:
        keys.append(f"{prefix}p_comp_{core_words[0].lower()}")
    # If 2 or more words, concatenate adjacent words[0] + words[1]
    elif len(core_words) >= 2:
        comp = f"{core_words[0]}{core_words[1]}".lower()
        if 6 <= len(comp) <= 18:
            keys.append(f"{prefix}p_comp_{comp}")

    return list(dict.fromkeys(keys))


def extract_all_blocking_keys(
    norm_name: str,
    norm_addr: str,
    country: str = "",
    partition_by_country: bool = True,
) -> Dict[str, List[str]]:
    """Generate all blocking pass keys for a given record in V3."""
    c_prefix = country if partition_by_country else ""

    p1 = extract_pass1_keys(norm_name, c_prefix)
    p2 = extract_pass2_keys(norm_addr, c_prefix)
    p2_locpair = extract_pass2_locality_pair_keys(norm_addr, c_prefix)
    p3 = extract_pass3_keys(norm_name, c_prefix)
    p4 = extract_pass4_keys(norm_name, c_prefix)
    p5 = extract_distinctive_token_keys(norm_name, c_prefix)
    p6 = extract_compound_brand_keys(norm_name, c_prefix)

    return {
        "pass1": p1,
        "pass2": p2,
        "pass2_locpair": p2_locpair,
        "pass3": p3,
        "pass4": p4,
        "pass5_distinctive": p5,
        "pass6_compound": p6,
    }


# ---------------------------------------------------------------------------
# Inverted Index Data Structure
# ---------------------------------------------------------------------------

class InvertedIndex:
    def __init__(self, max_block_size: int = 1_000):
        self.index: Dict[str, array.array] = {}
        self.super_blocks: Set[str] = set()
        self.max_block_size = max_block_size
        self.target_ids: List[str] = []
        self.target_sources: List[str] = []
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
        target_idx = len(self.target_ids)
        self.target_ids.append(entity_id)
        self.target_sources.append(source)
        self.target_names.append(norm_name)
        self.target_addrs.append(norm_addr)

        all_keys: Set[str] = set()
        for k_list in keys_dict.values():
            all_keys.update(k_list)

        for k in all_keys:
            if k in self.super_blocks:
                continue
            if k not in self.index:
                self.index[k] = array.array("I", [target_idx])
            else:
                arr = self.index[k]
                if len(arr) >= self.max_block_size:
                    self.super_blocks.add(k)
                    del self.index[k]
                else:
                    arr.append(target_idx)

        return target_idx

    def query(self, keys_dict: Dict[str, List[str]]) -> Dict[int, List[str]]:
        hits: Dict[int, List[str]] = defaultdict(list)
        for pass_name, k_list in keys_dict.items():
            for k in k_list:
                if k in self.super_blocks:
                    continue
                arr = self.index.get(k)
                if arr is not None:
                    for target_idx in arr:
                        hits[target_idx].append(pass_name)
        return hits

    def size(self) -> int:
        return len(self.target_ids)

    def num_keys(self) -> int:
        return len(self.index)


# ---------------------------------------------------------------------------
# Evidence-Based Ranking & Scoring for Truncation
# ---------------------------------------------------------------------------

def fast_jaccard_similarity(s1: str, s2: str) -> float:
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
    scored_candidates = []
    for tidx, passes in candidate_hits.items():
        unique_passes = list(dict.fromkeys(passes))
        num_passes = len(unique_passes)

        # Base score driven by multi-pass support
        score = num_passes * 100.0

        # Pass priority weights
        if "pass1" in unique_passes: score += 20.0
        if "pass2" in unique_passes: score += 25.0
        if "pass4" in unique_passes: score += 15.0
        if "pass6_compound" in unique_passes: score += 15.0
        if "pass5_distinctive" in unique_passes: score += 10.0
        if "pass2_locpair" in unique_passes: score += 12.0

        # Fast string overlap verification
        cand_name = index.target_names[tidx] if tidx < len(index.target_names) else ""
        cand_addr = index.target_addrs[tidx] if tidx < len(index.target_addrs) else ""

        name_sim = fast_jaccard_similarity(s1_name, cand_name)
        addr_sim = fast_jaccard_similarity(s1_addr, cand_addr)
        score += name_sim * 10.0 + addr_sim * 5.0

        scored_candidates.append((score, tidx, unique_passes))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [(tidx, passes) for _, tidx, passes in scored_candidates[:max_candidates]]


# ---------------------------------------------------------------------------
# High-Level MultiPassBlocker Class (V3)
# ---------------------------------------------------------------------------

@dataclass
class BlockingConfig:
    data_dir: Path
    output_dir: Optional[Path] = None
    chunk_size: int = 100_000
    max_candidates: int = 200
    max_block_size: int = 1_000
    partition_by_country: bool = True


class MultiPassBlocker:
    """Manages inverted index building and candidate generation for V3."""

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
                norm_n = normalize_business_name_v3(raw_n) if raw_n and not is_missing(raw_n) else ""

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

    def query_source1_chunk(self, s1_df: pd.DataFrame) -> List[Dict[str, Any]]:
        candidate_rows: List[Dict[str, Any]] = []

        eids = s1_df["entity_id"].values
        names = s1_df["business_name"].values
        addrs = s1_df["business_address"].values
        ctrys = s1_df["country"].values

        for i in range(len(eids)):
            s1_id = eids[i]
            raw_c = ctrys[i]
            norm_c = normalize_country(raw_c) if raw_c and not is_missing(raw_c) else ""

            raw_n = names[i]
            norm_n = normalize_business_name_v3(raw_n) if raw_n and not is_missing(raw_n) else ""

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
        Stream S1 file and write candidates to output TSV in chunks.
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
# CLI Entrypoint for Execution (V3)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Candidate Blocking Generator V3 for Amazon ML Challenge 2026."
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
    parser.add_argument(
        "--max-target-rows",
        type=int,
        default=None,
        help="Optional row limit per target file for testing.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    s1_file = data_dir / "train_source1.tsv"
    s2_file = data_dir / "train_source2.tsv"
    s3_file = data_dir / "train_source3.tsv"

    if not s1_file.exists():
        s1_file = data_dir / "test_source1.tsv"
        s2_file = data_dir / "test_source2.tsv"
        s3_file = data_dir / "test_source3.tsv"

    if not s1_file.exists() or not s2_file.exists() or not s3_file.exists():
        print(f"Error: Missing source TSV files in {data_dir}", file=sys.stderr)
        sys.exit(1)

    config = BlockingConfig(
        data_dir=data_dir,
        output_dir=Path(args.output_file).parent,
        chunk_size=args.chunk_size,
        max_candidates=args.max_candidates,
        max_block_size=args.max_block_size,
        partition_by_country=not args.disable_country_partition,
    )

    blocker = MultiPassBlocker(config)

    print(f">>> Indexing Source 2 from {s2_file.name}...")
    n_s2 = blocker.index_target_file(s2_file, "S2", max_rows=args.max_target_rows)
    print(f"    Indexed {n_s2:,} Source 2 entities.")

    print(f">>> Indexing Source 3 from {s3_file.name}...")
    n_s3 = blocker.index_target_file(s3_file, "S3", max_rows=args.max_target_rows)
    print(f"    Indexed {n_s3:,} Source 3 entities.")
    print(f"    Total target entities in index: {blocker.index.size():,}")
    print(f"    Total unique blocking keys:    {blocker.index.num_keys():,}")
    print(f"    Super-blocks pruned:           {len(blocker.index.super_blocks):,}")

    out_path = Path(args.output_file).resolve()
    print(f"\n>>> Generating candidate pairs for Source 1 from {s1_file.name}...")
    n_s1, n_pairs = blocker.generate_candidates(
        s1_file=s1_file,
        output_file=out_path,
        max_s1_rows=args.max_s1_rows,
    )

    print(f"\nCandidate Blocking Complete!")
    print(f"  Source 1 Processed:       {n_s1:,}")
    print(f"  Candidate Pairs Written:  {n_pairs:,}")
    if n_s1 > 0:
        print(f"  Average Candidates / S1:  {n_pairs / n_s1:.2f}")
    print(f"  Output File:              {out_path}")


if __name__ == "__main__":
    main()

