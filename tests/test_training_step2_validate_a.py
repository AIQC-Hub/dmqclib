import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.step2_validate_model.kfold_validation import KFoldValidation


class TestKFoldValidation(unittest.TestCase):
    """
    A suite of tests ensuring that KFoldValidation correctly captures
    configurations, splits training data, applies the XGBoost model,
    and writes validation results.
    """

    def setUp(self):
        """
        Prepare the test environment by loading a training configuration
        and input training data. The input file names for train/test sets
        are defined here for subsequent model validation tests.
        """
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
                "temp": str(data_path / "train_set_temp.parquet"),
                "psal": str(data_path / "train_set_psal.parquet"),
                "pres": str(data_path / "train_set_pres.parquet"),
            },
            "test": {
                "temp": str(data_path / "test_set_temp.parquet"),
                "psal": str(data_path / "test_set_psal.parquet"),
                "pres": str(data_path / "test_set_pres.parquet"),
            },
        }

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_step_name(self):
        """Check that the step name is correctly identified as 'validate'."""
        ds = KFoldValidation(self.config)
        self.assertEqual(ds.step_name, "validate")

    def test_output_file_names(self):
        """Verify that the default output file names are correctly resolved."""
        ds = KFoldValidation(self.config)
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_temp.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_psal.tsv",
            str(ds.output_file_names["report"]["psal"]),
        )
        self.assertEqual(
            "/path/to/validate_1/nrt_bo_001/validate_folder_1/validation_report_pres.tsv",
            str(ds.output_file_names["report"]["pres"]),
        )

    def test_base_model(self):
        """Ensure the base model is an XGBoost instance, as defined by the config."""
        ds = KFoldValidation(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """Check that training data is properly loaded into the KFoldValidation instance."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 42)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 42)

    def test_default_k_fold(self):
        """Confirm that k_fold defaults to 10 if no config entry is present."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        ds.config.data["step_param_set"]["steps"]["validate"]["k_fold"] = None

        k_fold = ds.get_k_fold()
        self.assertEqual(k_fold, 10)

    def test_xgboost(self):
        """Check that the XGBoost model processes the training sets and populates results."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)
        ds.process_targets()

        self.assertIsInstance(ds.reports["temp"], pl.DataFrame)
        self.assertEqual(ds.reports["temp"].shape[0], 12)
        self.assertEqual(ds.reports["temp"].shape[1], 7)

        self.assertIsInstance(ds.reports["psal"], pl.DataFrame)
        self.assertEqual(ds.reports["psal"].shape[0], 12)
        self.assertEqual(ds.reports["psal"].shape[1], 7)

    def test_write_results(self):
        """Ensure validation results are written to files as expected."""
        ds = KFoldValidation(self.config, training_sets=self.ds_input.training_sets)

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["report"]["temp"] = str(
            data_path / "temp_temp_validation_report.tsv"
        )
        ds.output_file_names["report"]["psal"] = str(
            data_path / "temp_psal_validation_report.tsv"
        )
        ds.output_file_names["report"]["pres"] = str(
            data_path / "temp_pres_validation_report.tsv"
        )

        ds.process_targets()
        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])
