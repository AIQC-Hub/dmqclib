import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.training.models.xgboost import XGBoost
from dmqclib.training.step4_build.build_model import BuildModel


class TestBuildModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
        self.config_file_path = str(
            Path(__file__).resolve().parent / "data" / "config" / "training.yaml"
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        self.input_file_names = {
            "temp": {
                "train": str(data_path / "temp_train.parquet"),
                "test": str(data_path / "temp_test.parquet"),
            },
            "psal": {
                "train": str(data_path / "psal_train.parquet"),
                "test": str(data_path / "psal_test.parquet"),
            },
        }

        self.ds_input = load_step1_input_training_set(
            "NRT_BO_002", str(self.config_file_path)
        )
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_init_valid_dataset_name(self):
        """Ensure ExtractDataSetA constructs correctly with a valid label."""
        ds = BuildModel("NRT_BO_002", str(self.config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_BO_002")

    def test_init_invalid_dataset_name(self):
        """Ensure ValueError is raised for an invalid label."""
        with self.assertRaises(ValueError):
            BuildModel("NON_EXISTENT_LABEL", str(self.config_file_path))

    def test_config_file(self):
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel("NRT_BO_002", str(self.config_file_path))
        self.assertTrue("training.yaml" in ds.config_file_name)

    def test_output_file_names(self):
        """Ensure output file names are set correctly."""
        ds = BuildModel("NRT_BO_002", str(self.config_file_path))
        self.assertEqual(
            "/path/to/data/nrt_bo_002/training/temp_model.joblib",
            str(ds.output_file_names["temp"]["model"]),
        )
        self.assertEqual(
            "/path/to/data/nrt_bo_002/training/psal_model.joblib",
            str(ds.output_file_names["psal"]["model"]),
        )

        self.assertEqual(
            "/path/to/data/nrt_bo_002/training/temp_test_result.tsv",
            str(ds.output_file_names["temp"]["result"]),
        )
        self.assertEqual(
            "/path/to/data/nrt_bo_002/training/psal_test_result.tsv",
            str(ds.output_file_names["psal"]["result"]),
        )

    def test_base_model(self):
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel("NRT_BO_002", str(self.config_file_path))
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_training_sets(self):
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel(
            "NRT_BO_002",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
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
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        ds.build_targets()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)

    def test_test_with_xgboost(self):
        """Verify the config file is correctly set in the member variable."""
        ds = BuildModel(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
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
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_results(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = BuildModel(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["temp"]["result"] = data_path / "temp_temp_test_result.tsv"
        ds.output_file_names["psal"]["result"] = data_path / "temp_psal_test_result.tsv"

        ds.build_targets()
        ds.test_targets()
        ds.write_results()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]["result"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]["result"]))
        os.remove(ds.output_file_names["temp"]["result"])
        os.remove(ds.output_file_names["psal"]["result"])

    def test_write_no_results(self):
        """Ensure ValueError is raised for an empty result list."""
        ds = BuildModel(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.write_results()

    def test_write_no_models(self):
        """Ensure ValueError is raised for an empty model list."""
        ds = BuildModel(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        with self.assertRaises(ValueError):
            ds.write_models()

    def test_write_models(self):
        """Ensure models are saved correctly."""
        ds = BuildModel(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.training_sets,
            self.ds_input.test_sets,
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["temp"]["model"] = data_path / "temp_temp_model.joblib"
        ds.output_file_names["psal"]["model"] = data_path / "temp_psal_model.joblib"

        ds.build_targets()
        ds.write_models()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]["model"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]["model"]))
        os.remove(ds.output_file_names["temp"]["model"])
        os.remove(ds.output_file_names["psal"]["model"])

    def test_read_models(self):
        """Ensure models are loaded correctly."""
        ds = BuildModel(
            "NRT_BO_001", str(self.config_file_path), None, self.ds_input.test_sets
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["temp"]["model"] = data_path / "temp_model.joblib"
        ds.output_file_names["psal"]["model"] = data_path / "psal_model.joblib"

        ds.read_models()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)

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
            "NRT_BO_001", str(self.config_file_path), None, self.ds_input.test_sets
        )

        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.output_file_names["temp"]["model"] = data_path / "model.joblib"
        ds.output_file_names["psal"]["model"] = data_path / "model.joblib"

        with self.assertRaises(FileNotFoundError):
            ds.read_models()
