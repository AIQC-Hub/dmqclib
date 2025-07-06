import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.step4_build.build_model import BuildModel


class TestBuildModel(unittest.TestCase):
    """
    A suite of tests ensuring that building, testing, and saving XGBoost models
    via BuildModel follows the expected configuration and data flows.
    """

    def setUp(self):
        """
        Prepare a test training configuration and load input data.
        Train/test file paths are defined for subsequent tests.
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
        """Check that the BuildModel step name is correctly assigned."""
        ds = BuildModel(self.config)
        self.assertEqual(ds.step_name, "build")

    def test_output_file_names(self):
        """Verify that default output file names (model and reports) are as expected."""
        ds = BuildModel(self.config)

        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/temp_model.joblib",
            str(ds.model_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/psal_model.joblib",
            str(ds.model_file_names["psal"]),
        )

        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/temp_test_report.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/psal_test_report.tsv",
            str(ds.output_file_names["report"]["psal"]),
        )

    def test_base_model(self):
        """Ensure that the configured base model is an XGBoost instance."""
        ds = BuildModel(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """Check that training and test sets are loaded into BuildModel correctly."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 38)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_train_with_xgboost(self):
        """Confirm that building models populates the 'models' dictionary with XGBoost instances."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

    def test_test_with_xgboost(self):
        """Check that testing sets after model building populates the result columns."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        ds.build_targets()
        ds.test_targets()

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_test_without_model(self):
        """Ensure that testing without building models raises a ValueError."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_reports(self):
        """Check that the test reports are correctly written to file."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["report"]["temp"] = data_path / "temp_temp_test_report.tsv"
        ds.output_file_names["report"]["psal"] = data_path / "temp_psal_test_report.tsv"
        ds.output_file_names["report"]["pres"] = data_path / "temp_pres_test_report.tsv"

        ds.build_targets()
        ds.test_targets()
        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    def test_write_no_results(self):
        """Ensure ValueError is raised if write_results is called with no results available."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_reports()

    def test_write_no_models(self):
        """Ensure ValueError is raised if write_models is called without built models."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        with self.assertRaises(ValueError):
            ds.write_models()

    def test_write_models(self):
        """Check that the trained models are serialized to files correctly."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = data_path / "temp_temp_model.joblib"
        ds.model_file_names["psal"] = data_path / "temp_psal_model.joblib"
        ds.model_file_names["pres"] = data_path / "temp_pres_model.joblib"

        ds.build_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.model_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.model_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.model_file_names["pres"]))

        os.remove(ds.model_file_names["temp"])
        os.remove(ds.model_file_names["psal"])
        os.remove(ds.model_file_names["pres"])

    def test_read_models(self):
        """Verify that existing models can be reloaded from disk and used for testing."""
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = data_path / "temp_model.joblib"
        ds.model_file_names["psal"] = data_path / "psal_model.joblib"
        ds.model_file_names["pres"] = data_path / "pres_model.joblib"

        ds.read_models()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

        ds.test_targets()

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_read_models_no_file(self):
        """Check that FileNotFoundError is raised if model files are missing."""
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = data_path / "model.joblib"
        ds.model_file_names["psal"] = data_path / "model.joblib"
        ds.model_file_names["pres"] = data_path / "model.joblib"

        with self.assertRaises(FileNotFoundError):
            ds.read_models()
