import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
)
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll


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
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_classify_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_classify_step2_summary_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_classify_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_classify_step4_locate_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

    def test_output_file_names(self):
        """Check that the output file names are set according to the configuration."""
        ds = ExtractDataSetAll(self.config)
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_classify_temp.parquet",
            str(ds.output_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/extract/extracted_features_classify_psal.parquet",
            str(ds.output_file_names["psal"]),
        )

    def test_step_name(self):
        """Ensure the step name is set to 'extract'."""
        ds = ExtractDataSetAll(self.config)
        self.assertEqual(ds.step_name, "extract")

    def test_init_arguments(self):
        """Validate that input data, selected profiles, target rows, and summary stats are set."""
        ds = ExtractDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 595)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)

    def test_location_features(self):
        """Check that features are correctly processed for temp and psal targets."""
        ds = ExtractDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        ds.process_targets()

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 19480)
        self.assertEqual(ds.target_features["temp"].shape[1], 41)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 19480)
        self.assertEqual(ds.target_features["psal"].shape[1], 41)

    def test_write_target_features(self):
        """Confirm that target features are written to parquet files as expected."""
        ds = ExtractDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        data_path = Path(__file__).resolve().parent / "data" / "extract"
        ds.output_file_names["temp"] = str(
            data_path / "temp_extracted_features_classify_temp.parquet"
        )
        ds.output_file_names["psal"] = str(
            data_path / "temp_extracted_features_classify_psal.parquet"
        )
        ds.output_file_names["pres"] = str(
            data_path / "temp_extracted_features_classify_pres.parquet"
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
        ds = ExtractDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )

        with self.assertRaises(ValueError):
            ds.write_target_features()
