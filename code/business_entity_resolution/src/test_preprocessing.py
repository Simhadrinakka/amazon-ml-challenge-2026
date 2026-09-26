"""
Unit and Integration Tests for Preprocessing and Normalization Module.

Executable directly via `python <path_to_file>` or via `pytest`.
Validates missing values, business names, addresses, countries, and DataFrame chunk transformations.
"""

import sys
import unittest
from pathlib import Path

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
    normalize_dataframe_chunk,
    normalize_series,
)


class TestMissingValues(unittest.TestCase):
    """Test missing value detection across varied input representations."""

    def test_none_is_missing(self):
        self.assertTrue(is_missing(None))

    def test_nan_is_missing(self):
        self.assertTrue(is_missing(float("nan")))
        self.assertTrue(is_missing(np.nan))

    def test_empty_string_is_missing(self):
        self.assertTrue(is_missing(""))
        self.assertTrue(is_missing("   "))
        self.assertTrue(is_missing("\t\n\r  "))

    def test_string_sentinels_are_missing(self):
        self.assertTrue(is_missing("null"))
        self.assertTrue(is_missing("NULL"))
        self.assertTrue(is_missing("None"))
        self.assertTrue(is_missing("NaN"))
        self.assertTrue(is_missing("<missing>"))
        self.assertTrue(is_missing("N/A"))
        self.assertTrue(is_missing("na"))

    def test_valid_strings_are_not_missing(self):
        self.assertFalse(is_missing("Apple Inc"))
        self.assertFalse(is_missing("0"))
        self.assertFalse(is_missing("US"))


class TestBusinessNameNormalization(unittest.TestCase):
    """Test normalization of business entity names."""

    def test_case_insensitivity(self):
        name1 = normalize_business_name("MICROSOFT CORPORATION")
        name2 = normalize_business_name("microsoft corporation")
        name3 = normalize_business_name("Microsoft Corporation")
        self.assertEqual(name1, name2)
        self.assertEqual(name2, name3)
        self.assertEqual(name1, "microsoft corp")

    def test_ampersand_and_plus(self):
        name1 = normalize_business_name("Johnson & Johnson")
        name2 = normalize_business_name("Johnson and Johnson")
        name3 = normalize_business_name("Johnson + Johnson")
        self.assertEqual(name1, name2)
        self.assertEqual(name2, name3)
        self.assertEqual(name1, "johnson and johnson")

    def test_legal_suffixes_standardization(self):
        # Inc / Incorporated
        self.assertEqual(normalize_business_name("Alpha Incorporated"), "alpha inc")
        self.assertEqual(normalize_business_name("Alpha Inc."), "alpha inc")
        self.assertEqual(normalize_business_name("Alpha Inc"), "alpha inc")

        # LLC / Limited Liability Company
        self.assertEqual(normalize_business_name("Beta LLC"), "beta llc")
        self.assertEqual(normalize_business_name("Beta L.L.C."), "beta llc")
        self.assertEqual(normalize_business_name("Beta Limited Liability Company"), "beta llc")

        # Ltd / Limited / Pvt Ltd
        self.assertEqual(normalize_business_name("Gamma Limited"), "gamma ltd")
        self.assertEqual(normalize_business_name("Gamma Ltd."), "gamma ltd")
        self.assertEqual(normalize_business_name("Delta Pvt. Ltd."), "delta pvt ltd")
        self.assertEqual(normalize_business_name("Delta Private Limited"), "delta pvt ltd")

        # Corp / Corporation
        self.assertEqual(normalize_business_name("Omega Corp."), "omega corp")
        self.assertEqual(normalize_business_name("Omega Corporation"), "omega corp")

    def test_apostrophes_handling(self):
        self.assertEqual(normalize_business_name("Orelee's Barbershop"), "orelees barbershop")
        self.assertEqual(normalize_business_name("McDonald`s Restaurant"), "mcdonalds restaurant")
        self.assertEqual(normalize_business_name("Wendy’s"), "wendys")

    def test_names_with_numbers(self):
        self.assertEqual(normalize_business_name("7-Eleven"), "7-eleven")
        self.assertEqual(normalize_business_name("3M Company"), "3m co")
        self.assertEqual(normalize_business_name("Store #104"), "store 104")
        self.assertEqual(normalize_business_name("Studio 54"), "studio 54")

    def test_names_without_aggressive_word_removal(self):
        # Meaningful distinguishing words like 'First', 'National', 'The', 'Club' should be preserved
        self.assertEqual(normalize_business_name("The Coffee Club"), "the coffee club")
        self.assertEqual(normalize_business_name("First National Bank"), "first national bank")
        self.assertEqual(normalize_business_name("General Electric Co."), "general electric co")

    def test_missing_name_returns_empty_string(self):
        self.assertEqual(normalize_business_name(None), "")
        self.assertEqual(normalize_business_name(float("nan")), "")
        self.assertEqual(normalize_business_name("   "), "")


class TestAddressNormalization(unittest.TestCase):
    """Test normalization of business address strings."""

    def test_case_and_whitespace(self):
        addr1 = normalize_address("123  MAIN   STREET ,   SUITE  100 ")
        self.assertEqual(addr1, "123 main st, ste 100")

    def test_common_street_abbreviations(self):
        self.assertEqual(normalize_address("100 Main Street"), "100 main st")
        self.assertEqual(normalize_address("100 Main St."), "100 main st")
        self.assertEqual(normalize_address("200 Park Avenue"), "200 park ave")
        self.assertEqual(normalize_address("200 Park Ave."), "200 park ave")
        self.assertEqual(normalize_address("300 Grand Boulevard"), "300 grand blvd")
        self.assertEqual(normalize_address("400 Ocean Drive"), "400 ocean dr")
        self.assertEqual(normalize_address("500 Country Club Road"), "500 country club rd")
        self.assertEqual(normalize_address("600 Sunset Lane"), "600 sunset ln")
        self.assertEqual(normalize_address("700 Interstate Highway 10"), "700 interstate hwy 10")

    def test_suite_apartment_variations(self):
        self.assertEqual(normalize_address("123 Main St, Suite 400"), "123 main st, ste 400")
        self.assertEqual(normalize_address("123 Main St, Ste. 400"), "123 main st, ste 400")
        self.assertEqual(normalize_address("123 Main St, Suite # 400"), "123 main st, ste 400")
        self.assertEqual(normalize_address("123 Main St, #400"), "123 main st, ste 400")
        self.assertEqual(normalize_address("123 Main St, Apartment 2B"), "123 main st, apt 2b")
        self.assertEqual(normalize_address("123 Main St, Apt. 2B"), "123 main st, apt 2b")

    def test_preservation_of_house_and_building_numbers(self):
        self.assertEqual(normalize_address("1795 Westchester Drive"), "1795 westchester dr")
        self.assertEqual(normalize_address("KH NO. -570/13, NEW DELHI"), "kh no 570/13, new delhi")
        self.assertEqual(normalize_address("Plot No. 42, Sector 18, Phase 4"), "plot no 42, sec 18, phase 4")
        self.assertEqual(normalize_address("Building 5, Floor 3"), "bldg 5, fl 3")
        self.assertEqual(normalize_address("P.O. Box 12345"), "po box 12345")

    def test_preservation_of_locality_and_postal_code(self):
        addr = normalize_address("1795 Westchester Drive, High Point, NC 27262")
        self.assertEqual(addr, "1795 westchester dr, high point, nc 27262")
        self.assertIn("high point", addr)
        self.assertIn("nc", addr)
        self.assertIn("27262", addr)

    def test_missing_address_returns_empty_string(self):
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(normalize_address(float("nan")), "")
        self.assertEqual(normalize_address("   "), "")


class TestCountryNormalization(unittest.TestCase):
    """Test normalization of country strings."""

    def test_united_states_variations(self):
        expected = "US"
        self.assertEqual(normalize_country("US"), expected)
        self.assertEqual(normalize_country("us"), expected)
        self.assertEqual(normalize_country("USA"), expected)
        self.assertEqual(normalize_country("usa"), expected)
        self.assertEqual(normalize_country("United States"), expected)
        self.assertEqual(normalize_country("United States of America"), expected)
        self.assertEqual(normalize_country("U.S.A."), expected)
        self.assertEqual(normalize_country("u.s."), expected)

    def test_india_variations(self):
        expected = "IN"
        self.assertEqual(normalize_country("India"), expected)
        self.assertEqual(normalize_country("india"), expected)
        self.assertEqual(normalize_country("IND"), expected)
        self.assertEqual(normalize_country("ind"), expected)
        self.assertEqual(normalize_country("IN"), expected)
        self.assertEqual(normalize_country("in"), expected)
        self.assertEqual(normalize_country("Bharat"), expected)

    def test_other_major_countries(self):
        self.assertEqual(normalize_country("United Kingdom"), "GB")
        self.assertEqual(normalize_country("UK"), "GB")
        self.assertEqual(normalize_country("Canada"), "CA")
        self.assertEqual(normalize_country("Germany"), "DE")

    def test_missing_country_returns_empty_string(self):
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(float("nan")), "")
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country("   "), "")


class TestUnicodeAndIndicNormalization(unittest.TestCase):
    """Test full Unicode support, specifically Indic and multi-script business names/addresses."""

    def test_pure_devanagari_business_names(self):
        # Full Devanagari names with matras, halants, and conjuncts must be preserved
        name1 = "राम मार्केटिंग प्राइवेट लिमिटेड"
        self.assertEqual(normalize_business_name(name1), "राम मार्केटिंग प्राइवेट लिमिटेड")

        name2 = "आदित्य प्रॉपर्टीज एलएलपी"
        self.assertEqual(normalize_business_name(name2), "आदित्य प्रॉपर्टीज एलएलपी")

        name3 = "श्री गणेश एंटरप्राइजेज"
        self.assertEqual(normalize_business_name(name3), "श्री गणेश एंटरप्राइजेज")

    def test_mixed_latin_and_indic_names(self):
        # Mixed script names should normalize Latin components (lowercased, suffixes) while preserving Indic script
        self.assertEqual(normalize_business_name("Tata मोटर्स Limited"), "tata मोटर्स ltd")
        self.assertEqual(normalize_business_name("Reliance डिजिटल Store"), "reliance डिजिटल store")
        self.assertEqual(normalize_business_name("Infosys टेक्नोलॉजीज Inc."), "infosys टेक्नोलॉजीज inc")

    def test_other_indic_scripts(self):
        # Kannada, Tamil, Bengali, Telugu scripts preserved
        self.assertEqual(normalize_business_name("ಕರ್ನಾಟಕ ಬ್ಯಾಂಕ್"), "ಕರ್ನಾಟಕ ಬ್ಯಾಂಕ್")
        self.assertEqual(normalize_business_name("தமிழ்நாடு டெக்ஸ்டைல்ஸ்"), "தமிழ்நாடு டெக்ஸ்டைல்ஸ்")
        self.assertEqual(normalize_business_name("বাংলা ট্রেডার্স"), "বাংলা ট্রেডার্স")

    def test_indic_with_punctuation(self):
        # Punctuation around Indic names should be safely removed without affecting combining marks
        raw = "राम मार्केटिंग, (प्राइवेट लिमिटेड)! #10"
        norm = normalize_business_name(raw)
        self.assertEqual(norm, "राम मार्केटिंग प्राइवेट लिमिटेड 10")

    def test_indic_address_normalization(self):
        addr = "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ - 560041"
        norm = normalize_address(addr)
        self.assertIn("ಕರ್ನಾಟಕ", norm)
        self.assertIn("560041", norm)
        self.assertIn("9th blk jayanagar", norm)

    def test_nfkc_unicode_fullwidth_normalization(self):
        # Full-width Unicode characters normalized cleanly via NFKC
        self.assertEqual(normalize_business_name("ＡＢＣ　Ｉｎｃ．"), "abc inc")



class TestSeriesAndDataFrameTransformations(unittest.TestCase):
    """Test chunk-level pandas transformations."""

    def setUp(self):
        self.sample_data = {
            "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
            "business_name": [
                "Orelee's Barbershop",
                "AT&T Inc.",
                "Tata Consultancy Services Ltd.",
                None,
            ],
            "business_address": [
                "1795 Westchester Drive, High Point, NC 27262",
                "Suite # 400, 5th Avenue, New York, NY",
                "Plot No. 42, Sector 18, Gurgaon",
                float("nan"),
            ],
            "country": ["United States", "USA", "India", None],
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_normalize_series(self):
        res = normalize_series(self.df["business_name"], normalize_business_name)
        self.assertEqual(res.iloc[0], "orelees barbershop")
        self.assertEqual(res.iloc[1], "at and t inc")
        self.assertEqual(res.iloc[2], "tata consultancy services ltd")
        self.assertEqual(res.iloc[3], "")

    def test_normalize_dataframe_chunk_non_inplace(self):
        original_names = list(self.df["business_name"])
        original_addrs = list(self.df["business_address"])
        original_countries = list(self.df["country"])

        norm_df = normalize_dataframe_chunk(self.df, inplace=False)

        # Verify original DataFrame was NOT modified
        self.assertEqual(list(self.df["business_name"]), original_names)
        self.assertEqual(list(self.df["business_address"]), original_addrs)
        self.assertEqual(list(self.df["country"]), original_countries)
        self.assertNotIn("business_name_norm", self.df.columns)

        # Verify normalized DataFrame has new columns
        self.assertIn("business_name_norm", norm_df.columns)
        self.assertIn("business_address_norm", norm_df.columns)
        self.assertIn("country_norm", norm_df.columns)

        # Verify normalized values
        self.assertEqual(norm_df["business_name_norm"].iloc[0], "orelees barbershop")
        self.assertEqual(norm_df["business_address_norm"].iloc[0], "1795 westchester dr, high point, nc 27262")
        self.assertEqual(norm_df["country_norm"].iloc[0], "US")
        self.assertEqual(norm_df["country_norm"].iloc[2], "IN")
        self.assertEqual(norm_df["country_norm"].iloc[3], "")

    def test_normalize_dataframe_chunk_inplace(self):
        df_copy = self.df.copy()
        norm_df = normalize_dataframe_chunk(df_copy, inplace=True)

        self.assertEqual(norm_df["business_name"].iloc[0], "orelees barbershop")
        self.assertEqual(norm_df["business_address"].iloc[0], "1795 westchester dr, high point, nc 27262")
        self.assertEqual(norm_df["country"].iloc[0], "US")


def run_all_tests():
    """Run tests and return summary statistics."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
