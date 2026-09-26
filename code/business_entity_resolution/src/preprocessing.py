"""
Preprocessing and Normalization Utilities for Amazon ML Challenge 2026: Business Entity Resolution.

Memory-efficient normalization functions designed for chunk-based streaming and vectorized
processing of business names, addresses, and country identifiers.
"""

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Pre-compiled Regex Patterns for High Performance
# ---------------------------------------------------------------------------

# Whitespace and unicode cleaning
RE_WHITESPACE = re.compile(r"\s+")
RE_AMPERSAND = re.compile(r"\s*&\s*|\s*\+\s*")
RE_APOSTROPHE = re.compile(r"['`’]")

# Legal and business suffixes mapping (standardizing to canonical tokens)
LEGAL_SUFFIXES_MAP = {
    r"\b(?:incorporated|inc\.?)\b": "inc",
    r"\b(?:corporation|corp\.?)\b": "corp",
    r"\b(?:limited\s+liability\s+company|l[\.\s]*l[\.\s]*c\.?|llc)\b": "llc",
    r"\b(?:limited\s+liability\s+partnership|l[\.\s]*l[\.\s]*p\.?|llp)\b": "llp",
    r"\b(?:private\s+limited|pvt[\.\s]*ltd\.?|p[\.\s]*ltd\.?)\b": "pvt ltd",
    r"\b(?:public\s+limited\s+company|p[\.\s]*l[\.\s]*c\.?|plc)\b": "plc",
    r"\b(?:limited|ltd\.?)\b": "ltd",
    r"\b(?:company|co\.?)\b": "co",
    r"\b(?:gesellschaft\s+mit\s+beschrankter\s+haftung|gmbh)\b": "gmbh",
    r"\b(?:societe\s+anonyme|s[\.\s]*a\.?)\b": "sa",
    r"\b(?:proprietorship|prop\.?)\b": "prop",
    r"\b(?:enterprises|enterprise|entr\.?)\b": "enterprises",
    r"\b(?:services|serv\.?)\b": "services",
    r"\b(?:solutions|soln\.?|sol\.?)\b": "solutions",
    r"\b(?:technologies|tech\.?|technology)\b": "tech",
    r"\b(?:international|intl\.?|int'l)\b": "intl",
}

COMPILED_LEGAL_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in LEGAL_SUFFIXES_MAP.items()
]

# Common address abbreviations mapping (standardizing to canonical tokens)
ADDRESS_ABBREVIATIONS_MAP = {
    r"\b(?:street|str\.?|st\.?)\b": "st",
    r"\b(?:avenue|ave\.?|av\.?)\b": "ave",
    r"\b(?:boulevard|blvd\.?|boul\.?)\b": "blvd",
    r"\b(?:drive|dr\.?|drv\.?)\b": "dr",
    r"\b(?:road|rd\.?)\b": "rd",
    r"\b(?:lane|ln\.?)\b": "ln",
    r"\b(?:court|ct\.?)\b": "ct",
    r"\b(?:circle|cir\.?)\b": "cir",
    r"\b(?:place|pl\.?)\b": "pl",
    r"\b(?:square|sq\.?)\b": "sq",
    r"\b(?:highway|hwy\.?)\b": "hwy",
    r"\b(?:parkway|pkwy\.?|pky\.?)\b": "pkwy",
    r"\b(?:terrace|ter\.?)\b": "ter",
    r"\b(?:way|wy\.?)\b": "way",
    r"\b(?:suite|ste\.?)\b": "ste",
    r"\b(?:apartment|apt\.?)\b": "apt",
    r"\b(?:building|bldg\.?)\b": "bldg",
    r"\b(?:floor|fl\.?|flr\.?)\b": "fl",
    r"\b(?:room|rm\.?)\b": "rm",
    r"\b(?:department|dept\.?)\b": "dept",
    r"\b(?:post\s+office\s+box|p[\.\s]*o[\.\s]*box|pobox)\b": "po box",
    r"\b(?:khata\s+no\.?|kh\s+no\.?|khasra\s+no\.?)\b": "kh no",
    r"\b(?:plot\s+no\.?|plt\s+no\.?)\b": "plot no",
    r"\b(?:house\s+no\.?|h\s+no\.?|hno\.?)\b": "h no",
    r"\b(?:shop\s+no\.?|sco\s+no\.?)\b": "shop no",
    r"\b(?:sector|sec\.?)\b": "sec",
    r"\b(?:phase|ph\.?)\b": "phase",
    r"\b(?:block|blk\.?)\b": "blk",
}

COMPILED_ADDR_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in ADDRESS_ABBREVIATIONS_MAP.items()
]

# Country standardization mapping to ISO-2 codes
COUNTRY_SYNONYMS = {
    # United States
    "us": "US",
    "usa": "US",
    "united states": "US",
    "united states of america": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "america": "US",
    # India
    "in": "IN",
    "ind": "IN",
    "india": "IN",
    "bharat": "IN",
    # United Kingdom
    "uk": "GB",
    "gb": "GB",
    "gbr": "GB",
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    # Canada
    "ca": "CA",
    "can": "CA",
    "canada": "CA",
    # Germany
    "de": "DE",
    "deu": "DE",
    "germany": "DE",
    "deutschland": "DE",
    # France
    "fr": "FR",
    "fra": "FR",
    "france": "FR",
    # Australia
    "au": "AU",
    "aus": "AU",
    "australia": "AU",
    # Japan
    "jp": "JP",
    "jpn": "JP",
    "japan": "JP",
    # China
    "cn": "CN",
    "chn": "CN",
    "china": "CN",
    # Brazil
    "br": "BR",
    "bra": "BR",
    "brazil": "BR",
    "brasil": "BR",
    # Mexico
    "mx": "MX",
    "mex": "MX",
    "mexico": "MX",
    # Singapore
    "sg": "SG",
    "sgp": "SG",
    "singapore": "SG",
    # United Arab Emirates
    "ae": "AE",
    "are": "AE",
    "uae": "AE",
    "united arab emirates": "AE",
}


# ---------------------------------------------------------------------------
# Core Helper Functions
# ---------------------------------------------------------------------------


def is_missing(val: Any) -> bool:
    """
    Check if a value is null, NaN, None, empty string, or text representation of null.

    Parameters
    ----------
    val : Any
        Value to test.

    Returns
    -------
    bool
        True if missing/empty, False otherwise.
    """
    if val is None:
        return True
    if isinstance(val, float) and (pd.isna(val) or val != val):
        return True
    val_str = str(val).strip().lower()
    return val_str in ("", "nan", "none", "null", "<missing>", "n/a", "na", "\\n")


def basic_clean_text(text: Any) -> str:
    """
    Apply foundational text sanitization preserving all Unicode scripts:
    - Unicode NFKC normalization (Canonical Decomposition + Canonical Composition)
    - Removal of unprintable control characters and corrupted tokens
    - Lowercasing across all Unicode alphabets
    - Standardizing '&' and '+' to 'and'
    - Normalizing apostrophes
    - Collapsing whitespace

    Parameters
    ----------
    text : Any
        Input text.

    Returns
    -------
    str
        Cleaned lowercase Unicode text.
    """
    if is_missing(text):
        return ""

    text_str = str(text)

    # Unicode NFKC normalization: preserves base characters, matras, and compatible forms
    normalized = unicodedata.normalize("NFKC", text_str)

    # Remove non-printable control characters (Cc, Cf, Cs, Cn) and replacement char \ufffd
    # while preserving Letters (L), Numbers (N), Combining Marks (M), Symbols (S), Punctuation (P), Spaces (Zs)
    cleaned = "".join(
        c if (unicodedata.category(c)[0] != "C" and c != "\ufffd") else " "
        for c in normalized
    )

    # Lowercase (handles Unicode lowercasing across all scripts)
    cleaned = cleaned.lower()

    # Standardize ampersand and plus signs to 'and'
    cleaned = RE_AMPERSAND.sub(" and ", cleaned)

    # Normalize apostrophes
    cleaned = RE_APOSTROPHE.sub("", cleaned)

    # Collapse whitespace
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()

    return cleaned


def clean_unicode_punctuation_name(text: str) -> str:
    """
    Filter punctuation in business names while preserving all Unicode letters (L),
    numbers (N), combining vowel marks/matras (M), whitespace, hyphens, and slashes.
    """
    return "".join(
        c if (unicodedata.category(c)[0] in ("L", "N", "M") or c in " \t\n-/") else " "
        for c in text
    )


def clean_unicode_punctuation_address(text: str) -> str:
    """
    Filter punctuation in addresses while preserving all Unicode letters (L),
    numbers (N), combining vowel marks/matras (M), whitespace, and address delimiters (-,/#.).
    """
    return "".join(
        c if (unicodedata.category(c)[0] in ("L", "N", "M") or c in " \t\n-,/#.") else " "
        for c in text
    )


# ---------------------------------------------------------------------------
# Business Name Normalization
# ---------------------------------------------------------------------------


def normalize_business_name(name: Any, standardize_legal_suffix: bool = True) -> str:
    """
    Normalize business name while preserving distinctive alphanumeric tokens across all Unicode scripts.

    Transformation Pipeline:
    1. Base Unicode & whitespace normalization (NFKC).
    2. Normalize '&' / '+' to 'and'.
    3. Standardize legal/corporate suffixes (e.g., 'incorporated' -> 'inc', 'l.l.c.' -> 'llc').
    4. Clean non-word punctuation while preserving Unicode letters, marks (matras), numbers, hyphens, slashes.
    5. Clean redundant leading/trailing symbols and whitespaces.

    Parameters
    ----------
    name : Any
        Raw business name.
    standardize_legal_suffix : bool, default True
        Whether to standardize common legal suffixes to canonical short forms.

    Returns
    -------
    str
        Normalized business name.
    """
    if is_missing(name):
        return ""

    cleaned = basic_clean_text(name)
    if not cleaned:
        return ""

    # Standardize legal suffixes before stripping periods
    if standardize_legal_suffix:
        for compiled_re, replacement in COMPILED_LEGAL_PATTERNS:
            cleaned = compiled_re.sub(f" {replacement} ", cleaned)

    # Unicode punctuation filtering (preserves all letters, matras, and numbers across scripts)
    cleaned = clean_unicode_punctuation_name(cleaned)

    # Clean multiple spaces again
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip()

    return cleaned


# ---------------------------------------------------------------------------
# Business Address Normalization
# ---------------------------------------------------------------------------


def normalize_address(address: Any, standardize_abbreviations: bool = True) -> str:
    """
    Normalize business address while preserving street numbers, localities, and postal codes across all scripts.

    Transformation Pipeline:
    1. Base Unicode & whitespace normalization (NFKC).
    2. Normalize hash symbols and unit prefixes (e.g. '# 101' -> 'ste 101', 'Suite #400' -> 'ste 400').
    3. Standardize common street/unit abbreviations (e.g. 'avenue' -> 'ave', 'street' -> 'st').
    4. Clean formatting around numbers and hyphenated unit numbers (e.g. '570 - 13' -> '570-13').
    5. Unicode punctuation filtering preserving all Unicode characters, marks, numbers, and address tokens.
    6. Clean periods inside abbreviations and normalize whitespace.

    Parameters
    ----------
    address : Any
        Raw business address string.
    standardize_abbreviations : bool, default True
        Whether to standardize street and unit abbreviations.

    Returns
    -------
    str
        Normalized address.
    """
    if is_missing(address):
        return ""

    cleaned = basic_clean_text(address)
    if not cleaned:
        return ""

    # Normalize hash and suite combinations: 'Suite # 400' -> 'ste 400', '#101' -> 'ste 101'
    cleaned = re.sub(r"\b(?:suite|ste|apt|unit|room|fl|floor)\s*#\s*(\d+)", r"ste \1", cleaned)
    cleaned = re.sub(r"#\s*(\d+)", r"ste \1", cleaned)

    # Standardize abbreviations if enabled (before removing punctuation)
    if standardize_abbreviations:
        for compiled_re, replacement in COMPILED_ADDR_PATTERNS:
            cleaned = compiled_re.sub(f" {replacement} ", cleaned)

    # Deduplicate repeated unit/street tokens like 'ste ste' -> 'ste'
    cleaned = re.sub(r"\b(ste|apt|st|rd|dr|ave|ln|ct|blvd)\s+\1\b", r"\1", cleaned)

    # Clean leading dash/hyphen before numbers e.g. " -570/13" -> " 570/13"
    cleaned = re.sub(r"(?<=\s)-\s*(?=\d)", "", cleaned)

    # Clean numbers with hyphens/slashes spacing: e.g. "570 - 13" -> "570-13"
    cleaned = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1-\2", cleaned)
    cleaned = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1/\2", cleaned)

    # Unicode punctuation filtering (preserves letters, marks, digits, and address separators)
    cleaned = clean_unicode_punctuation_address(cleaned)

    # Clean periods remaining inside abbreviations
    cleaned = re.sub(r"\.(?!\d)", " ", cleaned)

    # Normalize multiple commas or trailing commas
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r",(\s*,)+", ",", cleaned)

    # Collapse whitespace
    cleaned = RE_WHITESPACE.sub(" ", cleaned).strip(" ,.-/")

    return cleaned


# ---------------------------------------------------------------------------
# Country Normalization
# ---------------------------------------------------------------------------


def normalize_country(country: Any) -> str:
    """
    Standardize country representations into consistent canonical ISO-2 country codes.

    Parameters
    ----------
    country : Any
        Raw country string.

    Returns
    -------
    str
        Standardized uppercase country code (e.g. 'US', 'IN', 'GB') or empty string if missing.
    """
    if is_missing(country):
        return ""

    clean_c = basic_clean_text(country)
    clean_c = re.sub(r"[^\w\s]", "", clean_c).strip()

    if not clean_c:
        return ""

    # Lookup canonical mapping
    if clean_c in COUNTRY_SYNONYMS:
        return COUNTRY_SYNONYMS[clean_c]

    # Fallback: if already 2-character alphabetic code, return uppercase
    if len(clean_c) == 2 and clean_c.isalpha():
        return clean_c.upper()

    # Fallback for unrecognized longer country strings: return cleaned title/upper
    return clean_c.upper()


# ---------------------------------------------------------------------------
# Pandas Series & Chunk-Based DataFrame Helpers
# ---------------------------------------------------------------------------


def normalize_series(
    series: pd.Series,
    normalizer_fn: Callable[[Any], str],
) -> pd.Series:
    """
    Apply a normalizer function over a pandas Series safely.

    Parameters
    ----------
    series : pd.Series
        Input pandas Series.
    normalizer_fn : Callable[[Any], str]
        Normalization function to apply.

    Returns
    -------
    pd.Series
        Series with normalized strings.
    """
    return series.apply(normalizer_fn)


def normalize_dataframe_chunk(
    df: pd.DataFrame,
    name_col: Optional[str] = "business_name",
    addr_col: Optional[str] = "business_address",
    country_col: Optional[str] = "country",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply normalization to relevant columns in a DataFrame chunk.
    Creates new normalized columns with `_norm` suffix without mutating original data
    unless inplace is True.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chunk from TSV reader.
    name_col : Optional[str], default 'business_name'
        Column name for business name.
    addr_col : Optional[str], default 'business_address'
        Column name for business address.
    country_col : Optional[str], default 'country'
        Column name for country.
    inplace : bool, default False
        Whether to overwrite existing columns or append '_norm' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame chunk with normalized fields.
    """
    target_df = df if inplace else df.copy()

    if name_col and name_col in target_df.columns:
        dest_col = name_col if inplace else f"{name_col}_norm"
        target_df[dest_col] = target_df[name_col].apply(normalize_business_name)

    if addr_col and addr_col in target_df.columns:
        dest_col = addr_col if inplace else f"{addr_col}_norm"
        target_df[dest_col] = target_df[addr_col].apply(normalize_address)

    if country_col and country_col in target_df.columns:
        dest_col = country_col if inplace else f"{country_col}_norm"
        target_df[dest_col] = target_df[country_col].apply(normalize_country)

    return target_df


# ---------------------------------------------------------------------------
# Self-Verification Demonstrations (for Testing / Module Verification)
# ---------------------------------------------------------------------------


def run_demonstration() -> Dict[str, List[Tuple[str, str]]]:
    """
    Run small verification demonstration on hardcoded sample strings.
    """
    sample_names = [
        ("Orelee's Barbershop", normalize_business_name("Orelee's Barbershop")),
        ("AT&T Inc.", normalize_business_name("AT&T Inc.")),
        ("Zephay Labs, Inc.", normalize_business_name("Zephay Labs, Inc.")),
        ("Tata Consultancy Services Ltd.", normalize_business_name("Tata Consultancy Services Ltd.")),
        ("McDonald's & Sons, L.L.C.", normalize_business_name("McDonald's & Sons, L.L.C.")),
        ("Walmart Supercenter #1234", normalize_business_name("Walmart Supercenter #1234")),
        (None, normalize_business_name(None)),
    ]

    sample_addrs = [
        (
            "1795 Westchester Drive, High Point, NC 27262",
            normalize_address("1795 Westchester Drive, High Point, NC 27262"),
        ),
        (
            "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
            normalize_address("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ),
        (
            "Suite # 400, 5th Avenue, New York, NY",
            normalize_address("Suite # 400, 5th Avenue, New York, NY"),
        ),
        (
            "P.O. Box 12345, North Main Street",
            normalize_address("P.O. Box 12345, North Main Street"),
        ),
        (
            "Plot No. 42, Sector 18, Phase 4, Gurgaon",
            normalize_address("Plot No. 42, Sector 18, Phase 4, Gurgaon"),
        ),
        (
            float("nan"),
            normalize_address(float("nan")),
        ),
    ]

    sample_countries = [
        ("US", normalize_country("US")),
        ("United States", normalize_country("United States")),
        ("U.S.A.", normalize_country("U.S.A.")),
        ("India", normalize_country("India")),
        ("IND", normalize_country("IND")),
        ("Great Britain", normalize_country("Great Britain")),
        ("Unknown Country", normalize_country("Unknown Country")),
        (None, normalize_country(None)),
    ]

    return {
        "names": sample_names,
        "addresses": sample_addrs,
        "countries": sample_countries,
    }


if __name__ == "__main__":
    demo_results = run_demonstration()

    print("=" * 70)
    print(" PREPROCESSING & NORMALIZATION DEMO")
    print("=" * 70)

    print("\n[1] BUSINESS NAME NORMALIZATION:")
    for raw, norm in demo_results["names"]:
        print(f"  Raw:  {repr(raw):<40} -> Norm: {repr(norm)}")

    print("\n[2] ADDRESS NORMALIZATION:")
    for raw, norm in demo_results["addresses"]:
        print(f"  Raw:  {repr(raw):<50} -> Norm: {repr(norm)}")

    print("\n[3] COUNTRY NORMALIZATION:")
    for raw, norm in demo_results["countries"]:
        print(f"  Raw:  {repr(raw):<40} -> Norm: {repr(norm)}")

    print("\n" + "=" * 70)
