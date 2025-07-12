"""
Unit tests for configuration management functionalities,
including writing configuration templates and reading existing configuration files.
"""

import unittest
from pathlib import Path

import polars as pl

from dmqclib.interface.stats import get_summary_stats, format_summary_stats


class TestSummaryStats(unittest.TestCase):
    """
    Tests for verifying that configuration templates can be correctly
    written to disk for 'prepare' (dataset) and 'train' modules.
    """

    def setUp(self):
        """
        Set up test environment by defining sample file path
        for input file
        """
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_get_profile_summary_stats(self):
        """
        Check that test_ds_profile_summary_stats returns a data frame with correct data summary.
        """
        ds = get_summary_stats(self.test_data_file, "profiles")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_get_global_summary_stats(self):
        """
        Check that test_ds_profile_summary_stats returns a data frame with correct data summary.
        """
        ds = get_summary_stats(self.test_data_file, "all")
        self.assertIsInstance(ds, pl.DataFrame)

    def test_format_profile_summary_stats(self):
        """
        Check that test_ds_profile_summary_stats returns a data frame with correct data summary.
        """
        ds = get_summary_stats(self.test_data_file, "profiles")

        stats_dict = format_summary_stats(ds)
        self.assertIsInstance(stats_dict, str)
        self.assertIn("psal", stats_dict)
        self.assertIn("pct25", stats_dict)

        stats_dict = format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
        self.assertIn("pct25", stats_dict)

        stats_dict = format_summary_stats(ds, ["pres", "temp"], ["mean"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
        self.assertNotIn("pct25", stats_dict)

    def test_format_global_summary_stats(self):
        """
        Check that test_format_global_summary_stats returns a data frame with correct data summary.
        """
        ds = get_summary_stats(self.test_data_file, "all")

        stats_dict = format_summary_stats(ds)
        self.assertIsInstance(stats_dict, str)
        self.assertIn("psal", stats_dict)

        stats_dict = format_summary_stats(ds, ["pres", "temp"])
        self.assertIsInstance(stats_dict, str)
        self.assertNotIn("psal", stats_dict)
