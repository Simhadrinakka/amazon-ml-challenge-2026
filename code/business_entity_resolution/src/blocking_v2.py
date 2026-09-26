"""
Candidate Blocking Module (V2 Experimental) for Amazon ML Challenge 2026: Business Entity Resolution.

Preserves the baseline blocking architecture while implementing evidence-backed enhancements:
1. Address Blocking:
   - Preserves all baseline Pass 2 keys.
   - Expands address number extraction from top-2 to top-4 numbers.
   - Extracts locality/city anchor from the LAST meaningful comma-separated segment.
   - Emits number + last-locality combinations.
2. Name Normalization:
   - Conservative leetspeak normalization (5->s, 0->o, 1->l) strictly for tokens containing letters.
   - Honorific prefix stripping (dr, smt, shri, sri, mr, mrs, ms).
3. Distinctive Token Fallback:
   - Single distinctive business-name token of length >= 5 not in generic/legal vocabulary.
   - Namespaced by country.
4. Domain Stem Enhancement:
   - Matches space-separated domain extensions (.com, com, .org, org).
   - Strips trailing corporate designators for stem alignment (e.g. 'bryansquare com' vs 'Bryan Square LLC').
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

# Honorific prefixes to strip from business names
HONORIFIC_PREFIXES: Set[str] = {
    "dr", "dr.", "smt", "smt.", "shri", "shri.", "sri", "sri.", "mr", "mr.", "mrs", "mrs.", "ms", "ms."
}

# Generic address tokens to exclude when building address anchor keys
ADDR_STOPWORDS: Set[str] = {
    "st", "rd", "ave", "dr", "ln", "blvd", "unit", "apt", "ste", "null",
    "floor", "fl", "bldg", "room", "dept", "kh", "no", "plot", "door", "h",
    "sec", "blk", "phase", "post", "box", "pobox", "po", "street", "road",
    "avenue", "drive", "lane", "boulevard", "suite", "apartment", "building",
    "near", "opp", "opposite", "behind", "beside", "main", "cross",
}

# Generic industry/business words excluded from distinctive token fallback
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

# Regex for extracting digits / house / plot / flat numbers
RE_NUM = re.compile(r"\b\d+(?:[-/]\d+)?\b")

# Enhanced regex matching domain extensions preceded by dot, space, or at end
RE_DOMAIN_SUFFIX = re.compile(
    r"(?:[\.\s]+)(?:com|org|net|in|co|io|biz|info|edu|gov|eu|uk|fr|de)\b",
    re.IGNORECASE
)

# Trailing corporate entity types to strip for domain stem alignment
RE_CORP_DESIGNATOR = re.compile(
    r"\b(?:llc|inc|ltd|corp|co|pvt|gmbh|sa)\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Unicode-Safe Word Tokenization & Character Cleaning Helpers
# ---------------------------------------------------------------------------

def clean_token(token: str) -> str:
    """Strip leading and trailing punctuation while preserving letters, numbers, and marks."""
    return token.strip(" ,.-/#()[]{}|\\\"'`~:;!?<>@#$%^&*")


def strip_latin_diacritics(text: str) -> str:
    """Remove accent marks from Latin characters while preserving Indic marks."""
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
    """Split text into words preserving all Unicode letters, numbers, and combining vowel marks."""
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
    """Preserve all Unicode letters (L), numbers (N), and combining marks/matras (M)."""
    if not text:
        return ""
    cleaned_latin = strip_latin_diacritics(text)
    return "".join(c for c in cleaned_latin if unicodedata.category(c)[0] in ("L", "N", "M"))


# ---------------------------------------------------------------------------
# Evidence-Backed Name Normalization Helpers (V2)
# ---------------------------------------------------------------------------

def normalize_leetspeak_token(token: str) -> str:
    """
    Conservative leetspeak normalization:
    5 -> s, 0 -> o, 1 -> l
    ONLY when the token contains at least one alphabetic letter.
    Pure numeric tokens (e.g. '500', '10') are preserved intact.
    """
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


def normalize_business_name_v2(name: str) -> str:
    """
    Enhanced business name normalization:
    1. Member 1 baseline normalization.
    2. Strips leading honorific prefixes (dr, smt, shri, etc.).
    3. Conservative leetspeak conversion on alphanumeric tokens.
    """
    if not name or is_missing(name):
        return ""
    base_norm = normalize_business_name(name)
    tokens = base_norm.split()
    # Strip leading honorifics
    while tokens and tokens[0].lower().rstrip(".") in HONORIFIC_PREFIXES:
        tokens = tokens[1:]
    # Apply conservative leetspeak
    cleaned_tokens = [normalize_leetspeak_token(t) for t in tokens]
    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# Key Extraction Functions for the 5 Blocking Passes (V2)
# ---------------------------------------------------------------------------

def extract_pass1_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 1: Normalized Core Name Prefix / Significant Token Blocking.
    Extracts the first 2 significant tokens from the normalized business name.
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
    Pass 2 (V2): Address Number + Street / Locality Anchor Blocking.
    Preserves all baseline Pass 2 keys, expands numbers to top-4,
    and adds locality/city anchors derived from the LAST meaningful segment.
    """
    if not norm_addr or is_missing(norm_addr):
        return []

    raw_num_matches = re.findall(r"\b[a-zA-Z]*-?0*(\d+)[a-zA-Z]*\b", norm_addr)
    # Expanded from top 2 to top 4 numbers
    unique_nums = list(dict.fromkeys(raw_num_matches))[:4]

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

    # Extract tokens from the LAST meaningful comma-separated segments (city and state/region)
    last_words = []
    if len(parts) >= 2:
        # Check second-to-last part (typically City/Locality)
        if len(parts) >= 3:
            p_prev = [
                clean_token(w).lower() for w in parts[-2].split()
                if clean_token(w).lower() not in ADDR_STOPWORDS
                and len(clean_token(w)) >= 3
                and not any(c.isdigit() for c in clean_token(w))
            ]
            last_words.extend(p_prev)
        # Check last part (typically State/Region/City)
        p_last = [
            clean_token(w).lower() for w in parts[-1].split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
            and not any(c.isdigit() for c in clean_token(w))
        ]
        last_words.extend(p_last)

    if unique_nums:
        for num in unique_nums:
            # 1) Number + Street token (part 0)
            if p0_words:
                keys.append(f"{prefix}p2_{num}_{p0_words[0]}")
            # 2) Number + Locality tokens (subsequent words)
            target_locs = list(subsequent_words[:3])
            if subsequent_words and subsequent_words[-1] not in target_locs:
                target_locs.append(subsequent_words[-1])
            for tok in target_locs:
                keys.append(f"{prefix}p2_{num}_{tok}")
            # 3) Additional: Number + LAST locality segment tokens
            for tok in last_words[:2]:
                keys.append(f"{prefix}p2_{num}_{tok}")
    else:
        # Fallback when no digits exist: combine first two significant words
        all_words = [
            clean_token(w) for w in norm_addr.split()
            if clean_token(w).lower() not in ADDR_STOPWORDS
            and len(clean_token(w)) >= 3
        ]
        if len(all_words) >= 2:
            keys.append(f"{prefix}p2_{all_words[0]}_{all_words[1]}")

    # Number-independent street + locality anchors
    if p0_words and subsequent_words:
        keys.append(f"{prefix}p2_loc_{p0_words[0]}_{subsequent_words[0]}")
    if p0_words and last_words and last_words[-1] not in subsequent_words[:1]:
        keys.append(f"{prefix}p2_loc_{p0_words[0]}_{last_words[-1]}")

    return list(dict.fromkeys(keys))


def extract_pass3_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 3: Sorted Significant-Name-Token Fingerprint Blocking.
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
    Pass 4 (V2): Compressed Alphanumeric Name / Domain Stem Blocking.
    Enhanced to strip space-separated domain extensions and trailing corporate designators.
    """
    if not norm_name or is_missing(norm_name):
        return []

    # Strip domain suffix (.com, com, .org, org)
    s_clean = RE_DOMAIN_SUFFIX.sub("", norm_name)
    # Strip trailing corporate designator (llc, inc, etc.) for domain stem alignment
    s_clean = RE_CORP_DESIGNATOR.sub("", s_clean)
    alphanum = clean_alphanumeric_unicode(s_clean)
    if len(alphanum) < 4:
        return []

    prefix = f"{country}:" if country else ""
    stem = alphanum[:12]
    return [f"{prefix}p4_{stem}"]


def extract_distinctive_token_keys(norm_name: str, country: str = "") -> List[str]:
    """
    Pass 5 (V2): Distinctive Significant Business-Name Token Fallback.
    Selects rare, distinctive tokens (length >= 5) that are not generic business words.
    Crucial for matching records with missing target addresses (~3.5% of S2/S3).
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
    # Sort by length descending to pick the most specific brand token
    sorted_by_len = sorted(distinctive, key=lambda x: len(x), reverse=True)
    return [f"{prefix}p_dist_{sorted_by_len[0]}"]


def extract_all_blocking_keys(
    norm_name: str,
    norm_addr: str,
    country: str = "",
    partition_by_country: bool = True,
) -> Dict[str, List[str]]:
    """
    Generate all 5 blocking pass keys for a given record in V2.
    """
    c_prefix = country if partition_by_country else ""

    p1 = extract_pass1_keys(norm_name, c_prefix)
    p2 = extract_pass2_keys(norm_addr, c_prefix)
    p3 = extract_pass3_keys(norm_name, c_prefix)
    p4 = extract_pass4_keys(norm_name, c_prefix)
    p5 = extract_distinctive_token_keys(norm_name, c_prefix)

    return {
        "pass1": p1,
        "pass2": p2,
        "pass3": p3,
        "pass4": p4,
        "pass5_distinctive": p5,
    }


# ---------------------------------------------------------------------------
# Inverted Index Data Structure
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Memory-compact inverted index mapping blocking keys to target entity row indices.
    """

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

        # High precision signals
        if "pass1" in unique_passes:
            score += 20.0
        if "pass2" in unique_passes:
            score += 25.0
        if "pass4" in unique_passes:
            score += 15.0
        if "pass5_distinctive" in unique_passes:
            score += 10.0

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
# High-Level MultiPassBlocker Class (V2)
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
    """Manages inverted index building and candidate generation for V2."""

    def __init__(self, config: BlockingConfig):
        self.config = config
        self.index = InvertedIndex(max_block_size=config.max_block_size)

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
            norm_n = normalize_business_name_v2(raw_n) if raw_n and not is_missing(raw_n) else ""

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
