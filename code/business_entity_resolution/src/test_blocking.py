"""
Unit and Integration Tests for Candidate Blocking Module (Member 2).

Validates:
- All 4 blocking pass key extractors (Pass 1, Pass 2, Pass 3, Pass 4).
- Unicode and multilingual scripts (Devanagari, Tamil, mixed scripts).
- Missing address handling (graceful empty key generation).
- Deduplication and pass provenance tracking across multiple passes.
- Multiple true matches per S1 entity across S2 and S3.
- Super-block frequency thresholding.
- Configurable evidence-based candidate capping.
- Source provenance tagging (S2 vs S3).
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

# Ensure parent directory is in path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from blocking import (
    BlockingConfig,
    InvertedIndex,
    MultiPassBlocker,
    extract_all_blocking_keys,
    extract_pass1_keys,
    extract_pass2_keys,
    extract_pass3_keys,
    extract_pass4_keys,
    fast_jaccard_similarity,
    rank_candidates_by_evidence,
)
from preprocessing import (
    normalize_address,
    normalize_business_name,
    normalize_country,
)


class TestBlockingKeys(unittest.TestCase):
    """Test key generation logic across all 4 individual blocking passes."""

    def test_pass1_name_prefix(self):
        # Two or more significant tokens
        keys1 = extract_pass1_keys("maure williams colombier inc", country="US")
        self.assertEqual(keys1, ["US:p1_maure_williams"])

        # Stopwords removed
        keys2 = extract_pass1_keys("the coffee club llc", country="US")
        self.assertEqual(keys2, ["US:p1_coffee_club"])

        # Single word name
        keys3 = extract_pass1_keys("summit inc", country="US")
        self.assertEqual(keys3, ["US:p1_summit"])

        # Empty / missing name
        self.assertEqual(extract_pass1_keys("", country="US"), [])
        self.assertEqual(extract_pass1_keys(None, country="US"), [])

    def test_pass2_address_anchor(self):
        # Standard US address with house number and city
        keys1 = extract_pass2_keys("85 wayne ave, ticonderoga, ny", country="US")
        self.assertIn("US:p2_85_wayne", keys1)
        self.assertIn("US:p2_85_ticonderoga", keys1)

        # Complex Indian address with door/unit number and locality
        keys2 = extract_pass2_keys("6 29, c i t colony, 2nd main rd mylapore, chennai, tamil nadu", country="IN")
        self.assertIn("US:p2_6_colony".replace("US", "IN"), keys2)
        self.assertIn("IN:p2_6_chennai", keys2)

        # Missing / NaN address returns empty list without error
        self.assertEqual(extract_pass2_keys("", country="US"), [])
        self.assertEqual(extract_pass2_keys(None, country="US"), [])

    def test_pass3_sorted_token_fingerprint(self):
        # Inversion invariance: word order does not affect the key
        keys_a = extract_pass3_keys("efs print ventures ltd", country="IN")
        keys_b = extract_pass3_keys("print ventures efs", country="IN")
        self.assertEqual(keys_a, keys_b)
        self.assertEqual(keys_a, ["IN:p3_efs_print"])

        # Standard legal suffixes filtered
        keys_c = extract_pass3_keys("dahlia power reliable scientific llc", country="US")
        self.assertEqual(keys_c, ["US:p3_dahlia_power"])

        # Empty / missing name
        self.assertEqual(extract_pass3_keys("", country="US"), [])
        self.assertEqual(extract_pass3_keys(None, country="US"), [])

    def test_pass4_compressed_domain_stem(self):
        # Domain name with .com suffix
        keys_domain = extract_pass4_keys("maurewilliamscolombier.com", country="US")
        self.assertEqual(keys_domain, ["US:p4_maurewilliam"])

        # Spaced legal name compresses to identical stem
        keys_legal = extract_pass4_keys("maure williams colombier inc", country="US")
        self.assertEqual(keys_legal, ["US:p4_maurewilliam"])

        # Very short names (< 4 chars) return empty
        self.assertEqual(extract_pass4_keys("ab", country="US"), [])

        # Missing name
        self.assertEqual(extract_pass4_keys(None, country="US"), [])


class TestUnicodeAndMultilingualSupport(unittest.TestCase):
    """Test full preservation of non-ASCII and Indic scripts across passes."""

    def test_devanagari_keys(self):
        name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        norm_name = normalize_business_name(name)
        keys_p1 = extract_pass1_keys(norm_name, country="IN")
        self.assertEqual(keys_p1, ["IN:p1_राम_मार्केटिंग"])

        keys_p3 = extract_pass3_keys(norm_name, country="IN")
        self.assertEqual(keys_p3, ["IN:p3_मार्केटिंग_राम"])

        keys_p4 = extract_pass4_keys(norm_name, country="IN")
        self.assertTrue(len(keys_p4) > 0)
        self.assertIn("राम", keys_p4[0])

    def test_tamil_keys_and_address_anchor(self):
        tamil_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
        norm_name = normalize_business_name(tamil_name)
        keys_p1 = extract_pass1_keys(norm_name, country="IN")
        self.assertEqual(keys_p1, ["IN:p1_ராஜ்_இன்வெஸ்ட்மெண்ட்ஸ்"])

        # Address with Tamil script locality
        addr = "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு"
        norm_addr = normalize_address(addr)
        keys_p2 = extract_pass2_keys(norm_addr, country="IN")
        self.assertIn("IN:p2_6_chennai", keys_p2)
        self.assertIn("IN:p2_6_தமிழ்நாடு", keys_p2)


class TestInvertedIndexAndProvenance(unittest.TestCase):
    """Test InvertedIndex behavior: deduplication, pass provenance, and super-block capping."""

    def setUp(self):
        self.index = InvertedIndex(max_block_size=5)

    def test_add_and_query_provenance(self):
        # Add target entity matching on pass1 and pass3
        keys_dict = {
            "pass1": ["US:p1_alpha_beta"],
            "pass2": [],
            "pass3": ["US:p3_alpha_beta"],
            "pass4": ["US:p4_alphabeta"],
        }
        idx = self.index.add_target(
            entity_id="S2-1001",
            source="S2",
            norm_name="alpha beta corp",
            norm_addr="100 main st",
            keys_dict=keys_dict,
        )
        self.assertEqual(idx, 0)
        self.assertEqual(self.index.size(), 1)

        # Query with S1 having pass1 and pass2
        query_keys = {
            "pass1": ["US:p1_alpha_beta"],
            "pass2": ["US:p2_different_addr"],
            "pass3": [],
            "pass4": [],
        }
        hits = self.index.query(query_keys)
        self.assertIn(0, hits)
        # Should record that pass1 matched
        self.assertEqual(hits[0], ["pass1"])

        # Query with S1 having both pass1 and pass4
        query_keys_multi = {
            "pass1": ["US:p1_alpha_beta"],
            "pass2": [],
            "pass3": [],
            "pass4": ["US:p4_alphabeta"],
        }
        hits_multi = self.index.query(query_keys_multi)
        self.assertIn(0, hits_multi)
        self.assertEqual(sorted(hits_multi[0]), ["pass1", "pass4"])

    def test_super_block_threshold(self):
        # Max block size is 5
        key = "US:p1_generic_store"
        for i in range(5):
            self.index.add_target(
                entity_id=f"S2-{i}",
                source="S2",
                norm_name="generic store",
                norm_addr="",
                keys_dict={"pass1": [key], "pass2": [], "pass3": [], "pass4": []},
            )

        self.assertIn(key, self.index.index)
        self.assertEqual(len(self.index.index[key]), 5)

        # 6th addition should trigger super-block threshold
        self.index.add_target(
            entity_id="S2-6",
            source="S2",
            norm_name="generic store",
            norm_addr="",
            keys_dict={"pass1": [key], "pass2": [], "pass3": [], "pass4": []},
        )
        self.assertIn(key, self.index.super_blocks)
        self.assertNotIn(key, self.index.index)

        # Subsequent queries on this super-block return empty
        hits = self.index.query({"pass1": [key]})
        self.assertEqual(hits, {})


class TestMultiMatchAndCapping(unittest.TestCase):
    """Test candidate union, multiple true matches per S1, and evidence-based capping."""

    def setUp(self):
        self.config = BlockingConfig(
            data_dir=Path("."),
            chunk_size=100,
            max_candidates=3,  # Set small cap to test evidence-based truncation
            max_block_size=100,
            partition_by_country=True,
        )
        self.blocker = MultiPassBlocker(self.config)

    def test_multiple_true_candidates_across_s2_and_s3(self):
        # Index 2 entities in S2 and 1 in S3 matching the same S1
        # Target 1 (S2): Matches on Pass 1
        self.blocker.index.add_target(
            entity_id="S2-001",
            source="S2",
            norm_name="acme tools inc",
            norm_addr="",
            keys_dict=extract_all_blocking_keys("acme tools inc", "", "US"),
        )
        # Target 2 (S2): Matches on Pass 2
        self.blocker.index.add_target(
            entity_id="S2-002",
            source="S2",
            norm_name="alias manufacturing",
            norm_addr="500 industrial pkwy, dallas, tx",
            keys_dict=extract_all_blocking_keys("alias manufacturing", "500 industrial pkwy, dallas, tx", "US"),
        )
        # Target 3 (S3): Matches on Pass 1 & Pass 4
        self.blocker.index.add_target(
            entity_id="S3-001",
            source="S3",
            norm_name="acmetools.com",
            norm_addr="500 industrial pkwy, dallas, tx",
            keys_dict=extract_all_blocking_keys("acmetools.com", "500 industrial pkwy, dallas, tx", "US"),
        )

        # Query S1
        s1_df = pd.DataFrame([{
            "entity_id": "S1-999",
            "business_name": "Acme Tools Inc",
            "business_address": "500 Industrial Parkway, Dallas, TX",
            "country": "US",
        }])

        candidates = self.blocker.query_source1_chunk(s1_df)

        cand_ids = [c["candidate_entity_id"] for c in candidates]
        self.assertIn("S2-001", cand_ids)
        self.assertIn("S2-002", cand_ids)
        self.assertIn("S3-001", cand_ids)

        # Verify sources
        sources = {c["candidate_entity_id"]: c["candidate_source"] for c in candidates}
        self.assertEqual(sources["S2-001"], "S2")
        self.assertEqual(sources["S2-002"], "S2")
        self.assertEqual(sources["S3-001"], "S3")

        # Verify provenance
        passes = {c["candidate_entity_id"]: c["blocking_passes"] for c in candidates}
        self.assertIn("pass1", passes["S2-001"])
        self.assertIn("pass2", passes["S2-002"])

    def test_evidence_based_capping_preserves_strongest_candidates(self):
        # Create 5 targets matching S1 with varying evidence levels
        # T1: 3 passes agreement (P1, P3, P4)
        self.blocker.index.add_target(
            entity_id="S2-STRONG-1",
            source="S2",
            norm_name="apex logistics solutions",
            norm_addr="",
            keys_dict={"pass1": ["US:p1_apex_logistics"], "pass2": [], "pass3": ["US:p3_apex_logistics"], "pass4": ["US:p4_apexlogistic"]},
        )
        # T2: 2 passes agreement (P1, P2)
        self.blocker.index.add_target(
            entity_id="S2-STRONG-2",
            source="S2",
            norm_name="apex logistics inc",
            norm_addr="100 broad st",
            keys_dict={"pass1": ["US:p1_apex_logistics"], "pass2": ["US:p2_100_broad"], "pass3": [], "pass4": []},
        )
        # T3: 2 passes agreement (P1, P3)
        self.blocker.index.add_target(
            entity_id="S3-STRONG-3",
            source="S3",
            norm_name="apex logistics co",
            norm_addr="",
            keys_dict={"pass1": ["US:p1_apex_logistics"], "pass2": [], "pass3": ["US:p3_apex_logistics"], "pass4": []},
        )
        # T4 & T5: only 1 weak pass
        self.blocker.index.add_target(
            entity_id="S2-WEAK-4",
            source="S2",
            norm_name="apex random",
            norm_addr="",
            keys_dict={"pass1": ["US:p1_apex_logistics"], "pass2": [], "pass3": [], "pass4": []},
        )
        self.blocker.index.add_target(
            entity_id="S2-WEAK-5",
            source="S2",
            norm_name="apex other",
            norm_addr="",
            keys_dict={"pass1": ["US:p1_apex_logistics"], "pass2": [], "pass3": [], "pass4": []},
        )

        s1_df = pd.DataFrame([{
            "entity_id": "S1-CAP-TEST",
            "business_name": "Apex Logistics Solutions",
            "business_address": "100 Broad St",
            "country": "US",
        }])

        # Max candidates is 3; all 5 match on at least one pass.
        # Evidence-based capping should prioritize S2-STRONG-1, S2-STRONG-2, S3-STRONG-3!
        candidates = self.blocker.query_source1_chunk(s1_df)

        self.assertEqual(len(candidates), 3)
        retrieved_ids = [c["candidate_entity_id"] for c in candidates]
        self.assertIn("S2-STRONG-1", retrieved_ids)
        self.assertIn("S2-STRONG-2", retrieved_ids)
        self.assertIn("S3-STRONG-3", retrieved_ids)
        self.assertNotIn("S2-WEAK-4", retrieved_ids)
        self.assertNotIn("S2-WEAK-5", retrieved_ids)

    def test_genuine_no_match_entity(self):
        # Query S1 that has no matching targets in index
        s1_df = pd.DataFrame([{
            "entity_id": "S1-NO-MATCH",
            "business_name": "Unheard Unique Name Never In Index",
            "business_address": "99999 Remote Lunar Crater, Moon",
            "country": "US",
        }])

        candidates = self.blocker.query_source1_chunk(s1_df)
        self.assertEqual(candidates, [])


def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
