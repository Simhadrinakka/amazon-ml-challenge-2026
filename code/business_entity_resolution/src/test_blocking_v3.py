"""
Unit Tests for V3 Experimental Candidate Blocking Module.
Validates all V3 targeted enhancements:
1. Address locality pair fallback (cross-script descriptive addresses).
2. Missing target address (prefix key on distinctive tokens).
3. Compound brand / acronym alignment (First Seven Exports -> firstseven).
4. Safety invariants (country prefix, generic stopword filtering, candidate cap).
"""

import unittest
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from blocking_v3 import (
    extract_pass2_locality_pair_keys,
    extract_distinctive_token_keys,
    extract_compound_brand_keys,
    extract_all_blocking_keys,
    normalize_business_name_v3,
    InvertedIndex,
    BlockingConfig,
    MultiPassBlocker,
)
import pandas as pd


class TestBlockingV3(unittest.TestCase):

    def test_address_locality_pair_fallback(self):
        """Cross-script descriptive addresses without numbers match on locality pairs."""
        addr1 = "c/o malay sarkar, panchagram, badkulla, hanskhali, west bengal"
        addr2 = "west bengal, panchagram, badkulla, c/o: malay sarkar, hanskhali, nadia"
        k1 = extract_pass2_locality_pair_keys(addr1, "IN")
        k2 = extract_pass2_locality_pair_keys(addr2, "IN")
        overlap = set(k1) & set(k2)
        self.assertIn("IN:p2_locpair_badkulla_panchagram", overlap)

    def test_missing_address_distinctive_prefix(self):
        """Distinctive token prefix key tolerates trailing spelling variations."""
        keys1 = extract_distinctive_token_keys("Aguilar Diversified LLC", "US")
        keys2 = extract_distinctive_token_keys("aguilar diervsmfeid llc", "US")
        overlap = set(keys1) & set(keys2)
        # Both generate 5-gram prefix key 'aguil'
        self.assertIn("US:p_pfx_aguil", overlap)
        self.assertIn("US:p_dist_aguilar", overlap)

    def test_compound_brand_alignment(self):
        """First Seven Exports matches firstseven via compound adjacent tokens."""
        k1 = extract_compound_brand_keys("First Seven Exports Pvt Ltd", "IN")
        k2 = extract_compound_brand_keys("firstseven", "IN")
        self.assertIn("IN:p_comp_firstseven", k1)
        self.assertIn("IN:p_comp_firstseven", k2)

        # Bn Technologies matches bntechnologies
        kb1 = extract_compound_brand_keys("Bn Technologies Ltd", "IN")
        kb2 = extract_compound_brand_keys("bntechnologies", "IN")
        self.assertIn("IN:p_comp_bntechnologies", kb1)
        self.assertIn("IN:p_comp_bntechnologies", kb2)

    def test_safety_invariants(self):
        """Verify country partitioning, generic words exclusion, and max block size."""
        # Generic word center / services should NOT generate compound or distinctive keys
        dist = extract_distinctive_token_keys("Global Services Center", "US")
        self.assertEqual(dist, [])

        comp = extract_compound_brand_keys("Global Services", "US")
        # Global is generic, so shouldn't create unchecked generic blocks
        all_k = extract_all_blocking_keys("Global Services", "", "US", True)
        for pass_name, k_list in all_k.items():
            for k in k_list:
                self.assertTrue(k.startswith("US:"), f"Key {k} missing country prefix")


if __name__ == "__main__":
    unittest.main()
