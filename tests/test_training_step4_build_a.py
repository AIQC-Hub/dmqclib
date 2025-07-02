import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.step4_build.build_model import BuildModel


class TestBuildModel(unittest.TestCase):
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
        ds = BuildModel(self.config)
        self.assertEqual(ds.step_name, "build")

    def test_output_file_names(self):
        """Ensure output file names are set correctly."""
        ds = BuildModel(self.config)
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/temp_model.joblib",
            str(ds.output_file_names["model"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/psal_model.joblib",
            str(ds.output_file_names["model"]["psal"]),
        )

        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/temp_test_result.tsv",
            str(ds.output_file_names["result"]["temp"]),
        )
        self.assertEqual(
            "/path/to/build_1/nrt_bo_001/build_folder_1/psal_test_result.tsv",
            str(ds.output_file_names["result"]["psal"]),
        )

    def test_base_model(self):
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """Verify the config file is correctly set in the member variable."""
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
        """Verify the config file is correctly set in the member variable."""
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
        """Verify the config file is correctly set in the member variable."""
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
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_results(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["result"]["temp"] = data_path / "temp_temp_test_result.tsv"
        ds.output_file_names["result"]["psal"] = data_path / "temp_psal_test_result.tsv"
        ds.output_file_names["result"]["pres"] = data_path / "temp_pres_test_result.tsv"

        ds.build_targets()
        ds.test_targets()
        ds.write_results()

        self.assertTrue(os.path.exists(ds.output_file_names["result"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["result"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["result"]["pres"]))

        os.remove(ds.output_file_names["result"]["temp"])
        os.remove(ds.output_file_names["result"]["psal"])
        os.remove(ds.output_file_names["result"]["pres"])

    def test_write_no_results(self):
        """Ensure ValueError is raised for an empty result list."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.write_results()

    def test_write_no_models(self):
        """Ensure ValueError is raised for an empty model list."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.write_models()

    def test_write_models(self):
        """Ensure models are saved correctly."""
        ds = BuildModel(
            self.config,
            training_sets=self.ds_input.training_sets,
            test_sets=self.ds_input.test_sets,
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["model"]["temp"] = data_path / "temp_temp_model.joblib"
        ds.output_file_names["model"]["psal"] = data_path / "temp_psal_model.joblib"
        ds.output_file_names["model"]["pres"] = data_path / "temp_pres_model.joblib"

        ds.build_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.output_file_names["model"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["model"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["model"]["pres"]))

        os.remove(ds.output_file_names["model"]["temp"])
        os.remove(ds.output_file_names["model"]["psal"])
        os.remove(ds.output_file_names["model"]["pres"])

    def test_read_models(self):
        """Ensure models are loaded correctly."""
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["model"]["temp"] = data_path / "temp_model.joblib"
        ds.output_file_names["model"]["psal"] = data_path / "psal_model.joblib"
        ds.output_file_names["model"]["pres"] = data_path / "pres_model.joblib"

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
        """ "Ensure FileNotFoundError is raised for an invalid file name."""
        ds = BuildModel(
            self.config, training_sets=None, test_sets=self.ds_input.test_sets
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["model"]["temp"] = data_path / "model.joblib"
        ds.output_file_names["model"]["psal"] = data_path / "model.joblib"
        ds.output_file_names["model"]["pres"] = data_path / "model.joblib"

        with self.assertRaises(FileNotFoundError):
            ds.read_models()
