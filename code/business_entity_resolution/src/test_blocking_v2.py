"""
Unit Tests for V2 Experimental Candidate Blocking Module.
Validates all 5 evidence-backed improvements:
1. Address Blocking (last locality anchor + expanded numbers).
2. Name Normalization (conservative leetspeak + honorific stripping).
3. Distinctive Token Fallback.
4. Domain Stem Enhancement (space/dot domain suffix + corporate designator).
5. Candidate Cap and Multi-pass indexing.
"""

import unittest
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from blocking_v2 import (
    normalize_leetspeak_token,
    normalize_business_name_v2,
    extract_pass1_keys,
    extract_pass2_keys,
    extract_pass3_keys,
    extract_pass4_keys,
    extract_distinctive_token_keys,
    extract_all_blocking_keys,
    InvertedIndex,
    BlockingConfig,
    MultiPassBlocker,
)
import pandas as pd


class TestBlockingV2(unittest.TestCase):

    def test_conservative_leetspeak(self):
        """Pure numbers preserved; words with letters and 5/0/1 normalized."""
        self.assertEqual(normalize_leetspeak_token("500"), "500")
        self.assertEqual(normalize_leetspeak_token("12"), "12")
        self.assertEqual(normalize_leetspeak_token("08"), "08")
        self.assertEqual(normalize_leetspeak_token("5terling"), "sterling")
        self.assertEqual(normalize_leetspeak_token("r0yal"), "royal")
        self.assertEqual(normalize_leetspeak_token("f1etcher"), "fletcher")
        self.assertEqual(normalize_leetspeak_token("5haan"), "shaan")

    def test_honorific_prefix_stripping(self):
        """Leading honorifics (dr, smt, shri, sri) stripped; core name preserved."""
        self.assertEqual(normalize_business_name_v2("Dr Solana Silos"), "solana silos")
        self.assertEqual(normalize_business_name_v2("Smt Vision Industries"), "vision industries")
        self.assertEqual(normalize_business_name_v2("Shri Tag India"), "tag india")
        self.assertEqual(normalize_business_name_v2("Dr. Champion & Brothers"), "champion and brothers")

    def test_domain_stem_alignment(self):
        """bryansquare com, bryansquare.com, and Bryan Square LLC produce matching stem."""
        k1 = extract_pass4_keys("bryansquare com", "US")
        k2 = extract_pass4_keys("bryansquare.com", "US")
        k3 = extract_pass4_keys("bryan square llc", "US")
        self.assertEqual(k1, ["US:p4_bryansquare"])
        self.assertEqual(k2, ["US:p4_bryansquare"])
        self.assertEqual(k3, ["US:p4_bryansquare"])

    def test_address_last_locality_and_expanded_numbers(self):
        """Extracts last locality anchor and up to 4 numbers."""
        addr = "Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad, Uttar Pradesh"
        keys = extract_pass2_keys(addr, "IN")
        # Check that 684 is paired with ghaziabad (last meaningful locality segment)
        self.assertIn("IN:p2_684_ghaziabad", keys)
        # Check that 989 is also captured
        self.assertIn("IN:p2_989_ghaziabad", keys)

    def test_distinctive_token_fallback(self):
        """Distinctive brand token extracted, generic words skipped."""
        k_rare = extract_distinctive_token_keys("zander blue", "US")
        self.assertEqual(k_rare, ["US:p_dist_zander"])

        # Generic words should not produce distinctive token keys
        k_generic = extract_distinctive_token_keys("Global Trading Services LLC", "US")
        self.assertEqual(k_generic, [])

    def test_multi_pass_candidate_hits(self):
        """Verify inverted index indexes and queries V2 keys accurately."""
        index = InvertedIndex(max_block_size=1000)
        keys_target = extract_all_blocking_keys(
            norm_name="solana silos",
            norm_addr="hn 37/2, keshav complex, pune",
            country="IN",
        )
        index.add_target("S2-100", "S2", "solana silos", "hn 37/2, keshav complex, pune", keys_target)

        # Query with S1 having honorific prefix
        keys_s1 = extract_all_blocking_keys(
            norm_name=normalize_business_name_v2("Dr Solana Silos Private Limited"),
            norm_addr="s no 37/2, pune city, maharashtra",
            country="IN",
        )
        hits = index.query(keys_s1)
        self.assertIn(0, hits)
        self.assertIn("pass1", hits[0])


if __name__ == "__main__":
    unittest.main()
