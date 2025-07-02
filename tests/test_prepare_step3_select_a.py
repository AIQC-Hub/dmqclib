import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step3_select.dataset_a import SelectDataSetA


class TestSelectDataSetA(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input dataset."""
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
        self.ds = load_step1_input_dataset(self.config)
        self.ds.input_file_name = str(self.test_data_file)
        self.ds.read_input_data()

    def test_step_name(self):
        """Ensure the step name is set correctly."""
        ds = SelectDataSetA(self.config)
        self.assertEqual(ds.step_name, "select")

    def test_output_file_name(self):
        """Ensure output file name is set correctly."""
        ds = SelectDataSetA(self.config)
        self.assertEqual(
            "/path/to/select_1/nrt_bo_001/select_folder_1/selected_profiles.parquet",
            str(ds.output_file_name),
        )

    def test_default_output_file_name(self):
        """Ensure output file name is set correctly."""
        config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        config = DataSetConfig(config_file_path)
        config.select("NRT_BO_001")

        ds = SelectDataSetA(config)
        self.assertEqual(
            "/path/to/data_1/nrt_bo_001/select/selected_profiles.parquet",
            str(ds.output_file_name),
        )

    def test_input_data(self):
        """Ensure input data is set correctly."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_positive_profiles(self):
        """Ensure positive profiles are selected correctly."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        self.assertIsInstance(ds.pos_profile_df, pl.DataFrame)
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 7)

    def test_negative_profiles(self):
        """Ensure negative profiles are selected correctly."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        self.assertIsInstance(ds.neg_profile_df, pl.DataFrame)
        self.assertEqual(ds.neg_profile_df.shape[0], 478)
        self.assertEqual(ds.neg_profile_df.shape[1], 7)

    def test_find_profile_pairs(self):
        """Ensure profile pairs are found correctly."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 8)
        self.assertEqual(ds.neg_profile_df.shape[0], 19)
        self.assertEqual(ds.neg_profile_df.shape[1], 8)

    def test_label_profiles(self):
        """Ensure profiles are labeled correctly in the dataset."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.label_profiles()
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_write_selected_profiles(self):
        """Ensure selected profiles are written to parquet file correctly."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.output_file_name = (
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles.parquet"
        )

        ds.label_profiles()
        ds.write_selected_profiles()
        self.assertTrue(os.path.exists(ds.output_file_name))
        os.remove(ds.output_file_name)

    def test_write_empty_selected_profiles(self):
        """ "Ensure ValueError is raised for empty profiles."""
        ds = SelectDataSetA(self.config, input_data=self.ds.input_data)
        ds.output_file_name = (
            Path(__file__).resolve().parent
            / "data"
            / "select"
            / "temp_selected_profiles.parquet"
        )

        with self.assertRaises(ValueError):
            ds.write_selected_profiles()
