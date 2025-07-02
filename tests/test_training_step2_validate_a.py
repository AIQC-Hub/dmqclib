import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.step2_validate.kfold_validation import KFoldValidation


class TestKFoldValidation(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        data_path = Path(__file__).resolve().parent / "data" / "training"
        self.input_file_names = {
            "train": {
                "temp": data_path / "temp_train.parquet",
                "psal": data_path / "psal_train.parquet",
                "pres": data_path / "pres_train.parquet",
            },
            "test": {
                "temp": data_path / "temp_test.parquet",
                "psal": data_path / "psal_test.parquet",
                "pres": data_path / "pres_test.parquet",
            },
        }

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_step_name(self):
        """Ensure the step name is set correctly."""
        ds = KFoldValidation(self.config)
        self.assertEqual(ds.step_name, "validate")

    def test_output_file_names(self):
        """Ensure output file names are set correctly."""
        ds = KFoldValidation(self.config)
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/temp_validation_result.tsv",
            str(ds.output_file_names["result"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/psal_validation_result.tsv",
            str(ds.output_file_names["result"]["psal"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/pres_validation_result.tsv",
            str(ds.output_file_names["result"]["pres"]),
        )

    def test_base_model(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 38)

    def test_default_k_fold(self):
        """Ensure k_fold is set to default value when no config entry is found."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        ds.config.data["step_param_set"]["steps"]["validate"]["k_fold"] = None

        k_fold = ds.get_k_fold()
        self.assertEqual(k_fold, 10)

    def test_xgboost(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        ds.process_targets()

        self.assertIsInstance(ds.results["temp"], pl.DataFrame)
        self.assertEqual(ds.results["temp"].shape[0], 12)
        self.assertEqual(ds.results["temp"].shape[1], 7)

        self.assertIsInstance(ds.results["psal"], pl.DataFrame)
        self.assertEqual(ds.results["psal"].shape[0], 12)
        self.assertEqual(ds.results["psal"].shape[1], 7)

    def test_write_results(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["result"]["temp"] = (
            data_path / "temp_temp_validation_result.tsv"
        )
        ds.output_file_names["result"]["psal"] = (
            data_path / "temp_psal_validation_result.tsv"
        )
        ds.output_file_names["result"]["pres"] = (
            data_path / "temp_pres_validation_result.tsv"
        )

        ds.process_targets()
        ds.write_results()

        self.assertTrue(os.path.exists(ds.output_file_names["result"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["result"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["result"]["pres"]))

        os.remove(ds.output_file_names["result"]["temp"])
        os.remove(ds.output_file_names["result"]["psal"])
        os.remove(ds.output_file_names["result"]["pres"])
