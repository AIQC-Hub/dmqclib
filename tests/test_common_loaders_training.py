import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.training_loader import (
    load_step1_input_training_set,
    load_step2_model_validation_class,
    load_step4_build_model_class,
)
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.step1_read_input.dataset_a import InputTrainingSetA
from dmqclib.train.step2_validate_model.kfold_validation import KFoldValidation
from dmqclib.train.step4_build_model.build_model import BuildModel


class TestTrainingInputClassLoader(unittest.TestCase):
    """
    Tests for verifying that the correct input training class
    (InputTrainingSetA) is loaded from the config.
    """

    def setUp(self):
        """
        Initialize a training configuration object and select a dataset
        for input loading tests.
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
        Check that load_step1_input_training_set returns an InputTrainingSetA
        instance with the expected step name.
        """
        ds = load_step1_input_training_set(self.config)
        self.assertIsInstance(ds, InputTrainingSetA)
        self.assertEqual(ds.step_name, "input")


class TestModelValidationClassLoader(unittest.TestCase):
    """
    Tests confirming that the correct model validation class
    (KFoldValidation) is loaded and configured.
    """

    def setUp(self):
        """
        Initialize a training configuration and load input data for testing
        the validation class loader.
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
                "temp": data_path / "train_set_temp.parquet",
                "psal": data_path / "train_set_psal.parquet",
                "pres": data_path / "train_set_pres.parquet",
            },
            "test": {
                "temp": data_path / "test_set_temp.parquet",
                "psal": data_path / "test_set_psal.parquet",
                "pres": data_path / "test_set_pres.parquet",
            },
        }

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_load_dataset_valid_config(self):
        """
        Check that load_step2_model_validation_class returns a KFoldValidation
        instance with the expected step name.
        """
        ds = load_step2_model_validation_class(self.config)
        self.assertIsInstance(ds, KFoldValidation)
        self.assertEqual(ds.step_name, "validate")

    def test_training_set_data(self):
        """
        Ensure that when training_sets are provided, KFoldValidation
        is instantiated with the correct Polars DataFrame sizes.
        """
        ds = load_step2_model_validation_class(self.config, self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 39)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 39)


class TestBuildModelClassLoader(unittest.TestCase):
    """
    Tests verifying that the correct build model class (BuildModel)
    is loaded from the config.
    """

    def setUp(self):
        """
        Initialize a training configuration and load input data for testing
        the build model class loader.
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
                "temp": data_path / "train_set_temp.parquet",
                "psal": data_path / "train_set_psal.parquet",
                "pres": data_path / "train_set_pres.parquet",
            },
            "test": {
                "temp": data_path / "test_set_temp.parquet",
                "psal": data_path / "test_set_psal.parquet",
                "pres": data_path / "test_set_pres.parquet",
            },
        }

        self.ds_input = load_step1_input_training_set(self.config)
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_load_dataset_valid_config(self):
        """
        Check that load_step4_build_model_class returns a BuildModel instance
        with the expected step name.
        """
        ds = load_step4_build_model_class(self.config)
        self.assertIsInstance(ds, BuildModel)
        self.assertEqual(ds.step_name, "build")

    def test_training_and_test_sets(self):
        """
        Ensure that BuildModel receives the correct training and test sets
        when they are provided.
        """
        ds = load_step4_build_model_class(
            self.config, self.ds_input.training_sets, self.ds_input.test_sets
        )

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 39)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 39)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 38)
