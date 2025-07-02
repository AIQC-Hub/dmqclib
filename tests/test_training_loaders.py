import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.common.loader.training_loader import load_step2_model_validation_class
from dmqclib.common.loader.training_loader import load_step4_build_model_class
from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.step1_input.dataset_a import InputTrainingSetA
from dmqclib.train.step2_validate.kfold_validation import KFoldValidation
from dmqclib.train.step4_build.build_model import BuildModel


class TestTrainingInputClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_dataset_valid_config(self):
        """
        Test that load_dataset returns an instance of InputTrainingSetA for the known label.
        """
        ds = load_step1_input_training_set(self.config)
        self.assertIsInstance(ds, InputTrainingSetA)
        self.assertEqual(ds.step_name, "input")


class TestModelValidationClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
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

    def test_load_dataset_valid_config(self):
        """
        Test that load_dataset returns an instance of KFoldValidation for the known label.
        """
        ds = load_step2_model_validation_class(self.config)
        self.assertIsInstance(ds, KFoldValidation)
        self.assertEqual(ds.step_name, "validate")

    def test_training_set_data(self):
        """
        Test that load_dataset returns an instance of KFoldValidation with correct input_data.
        """

        ds = load_step2_model_validation_class(self.config, self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 38)


class TestBuildModelClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
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

    def test_load_dataset_valid_config(self):
        """
        Test that load_dataset returns an instance of BuildModel for the known label.
        """
        ds = load_step4_build_model_class(self.config)
        self.assertIsInstance(ds, BuildModel)
        self.assertEqual(ds.step_name, "build")

    def test_training_and_test_sets(self):
        """
        Test that load_dataset returns an instance of SummaryDataSetA with correct input_data.
        """

        ds = load_step4_build_model_class(
            self.config, self.ds_input.training_sets, self.ds_input.test_sets
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
