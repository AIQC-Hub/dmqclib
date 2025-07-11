"""
This module contains unit tests for the ExtractDataSetA class,
verifying its functionality in processing and extracting features from
various intermediate datasets (input, summary, select, and locate)
generated in previous data quality control steps.
"""

import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.loader.dataset_loader import (
    load_step1_input_dataset,
    load_step2_summary_dataset,
    load_step3_select_dataset,
    load_step4_locate_dataset,
)
from dmqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA


class TestExtractDataSetA(unittest.TestCase):
    """
    A suite of tests verifying that the ExtractDataSetA class gathers
    and outputs extracted features from multiple prior steps (input, summary,
    select, locate).
    """

    def setUp(self):
        """Set up test environment and load input, summary, select, and locate data."""
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_step2_summary_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_step4_locate_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

    def test_output_file_names(self):
        """Check that the output file names are set according to the configuration."""
        ds = ExtractDataSetA(self.config)
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_temp.parquet",
            str(ds.output_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_psal.parquet",
            str(ds.output_file_names["psal"]),
        )

    def test_step_name(self):
        """Ensure the step name is set to 'extract'."""
        ds = ExtractDataSetA(self.config)
        self.assertEqual(ds.step_name, "extract")

    def test_init_arguments(self):
        """Validate that input data, selected profiles, target rows, and summary stats are set."""
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 2520)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 10683)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 128)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 140)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["pres"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["pres"].shape[0], 122)
        self.assertEqual(ds.selected_rows["pres"].shape[1], 9)

    def test_location_features(self):
        """Check that features are correctly processed for temp and psal targets."""
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        ds.process_targets()

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 128)
        self.assertEqual(ds.target_features["temp"].shape[1], 43)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 140)
        self.assertEqual(ds.target_features["psal"].shape[1], 43)

        self.assertIsInstance(ds.target_features["pres"], pl.DataFrame)
        self.assertEqual(ds.target_features["pres"].shape[0], 122)
        self.assertEqual(ds.target_features["pres"].shape[1], 43)

    def test_write_target_features(self):
        """Confirm that target features are written to parquet files as expected."""
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        data_path = Path(__file__).resolve().parent / "data" / "extract"
        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_pres.parquet"
        )

        ds.process_targets()
        ds.write_target_features()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])

    def test_write_no_target_features(self):
        """Check that calling write_target_features with empty features raises ValueError."""
        ds = ExtractDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        with self.assertRaises(ValueError):
            ds.write_target_features()
