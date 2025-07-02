import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.common.loader.dataset_loader import load_step2_summary_dataset
from dmqclib.common.loader.dataset_loader import load_step3_select_dataset
from dmqclib.common.loader.dataset_loader import load_step4_locate_dataset
from dmqclib.common.loader.dataset_loader import load_step5_extract_dataset
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step6_split.dataset_a import SplitDataSetA


class TestSplitDataSetA(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
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

        self.ds_extract = load_step5_extract_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            target_rows=self.ds_locate.target_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        self.ds_extract.process_targets()

    def test_step_name(self):
        """Ensure the step name is set correctly."""
        ds = SplitDataSetA(self.config)
        self.assertEqual(ds.step_name, "split")

    def test_output_file_names(self):
        """Ensure output file names are set correctly."""
        ds = SplitDataSetA(self.config)

        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/temp_train.parquet",
            str(ds.output_file_names["train"]["temp"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/psal_train.parquet",
            str(ds.output_file_names["train"]["psal"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/temp_test.parquet",
            str(ds.output_file_names["test"]["temp"]),
        )
        self.assertEqual(
            "/path/to/split_1/nrt_bo_001/split_folder_1/psal_test.parquet",
            str(ds.output_file_names["test"]["psal"]),
        )

    def test_target_features_data(self):
        """Ensure input data and selected profiles are read correctly."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        self.assertIsInstance(ds.target_features["temp"], pl.DataFrame)
        self.assertEqual(ds.target_features["temp"].shape[0], 128)
        self.assertEqual(ds.target_features["temp"].shape[1], 43)

        self.assertIsInstance(ds.target_features["psal"], pl.DataFrame)
        self.assertEqual(ds.target_features["psal"].shape[0], 140)
        self.assertEqual(ds.target_features["psal"].shape[1], 43)

    def test_split_features_data(self):
        """Ensure input data and selected profiles are read correctly."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 38)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_default_test_set_fraction(self):
        """Ensure set_fraction is set to default value when no config entry is found."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["test_set_fraction"] = None

        test_set_fraction = ds.get_test_set_fraction()
        self.assertEqual(test_set_fraction, 0.1)

    def test_default_k_fold(self):
        """Ensure k_fold is set to default value when no config entry is found."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)
        ds.config.data["step_param_set"]["steps"]["split"]["k_fold"] = None

        k_fold = ds.get_k_fold()
        self.assertEqual(k_fold, 10)

    def test_write_training_sets(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["train"]["temp"] = data_path / "temp_temp_train.parquet"
        ds.output_file_names["train"]["psal"] = data_path / "temp_psal_train.parquet"
        ds.output_file_names["train"]["pres"] = data_path / "temp_pres_train.parquet"

        ds.process_targets()
        ds.write_training_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])

    def test_write_test_sets(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["test"]["temp"] = data_path / "temp_temp_test.parquet"
        ds.output_file_names["test"]["psal"] = data_path / "temp_psal_test.parquet"
        ds.output_file_names["test"]["pres"] = data_path / "temp_pres_test.parquet"

        ds.process_targets()
        ds.write_test_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])

    def test_write_data_sets(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = SplitDataSetA(self.config, target_features=self.ds_extract.target_features)

        ds.process_targets()

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["train"]["temp"] = data_path / "temp_temp_train.parquet"
        ds.output_file_names["train"]["psal"] = data_path / "temp_psal_train.parquet"
        ds.output_file_names["train"]["pres"] = data_path / "temp_pres_train.parquet"
        ds.output_file_names["test"]["temp"] = data_path / "temp_temp_test.parquet"
        ds.output_file_names["test"]["psal"] = data_path / "temp_psal_test.parquet"
        ds.output_file_names["test"]["pres"] = data_path / "temp_pres_test.parquet"

        ds.process_targets()
        ds.write_data_sets()

        self.assertTrue(os.path.exists(ds.output_file_names["train"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["train"]["pres"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["test"]["pres"]))

        os.remove(ds.output_file_names["train"]["temp"])
        os.remove(ds.output_file_names["train"]["psal"])
        os.remove(ds.output_file_names["train"]["pres"])
        os.remove(ds.output_file_names["test"]["temp"])
        os.remove(ds.output_file_names["test"]["psal"])
        os.remove(ds.output_file_names["test"]["pres"])
