import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.step1_read_input.dataset_a import InputTrainingSetA


class TestInputTrainingSetA(unittest.TestCase):
    """
    A suite of tests ensuring that the InputTrainingSetA class
    correctly loads and processes training data sets.
    """

    def setUp(self):
        """
        Prepare a TrainingConfig instance pointing to a test YAML file with
        specified training sets. Also defines file paths where training
        and test data can be loaded from.
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

    def test_step_name(self):
        """Check that the step name in the InputTrainingSetA instance is 'input'."""
        ds = InputTrainingSetA(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_names(self):
        """Verify that file names for training and test sets are correctly resolved."""
        ds = InputTrainingSetA(self.config)
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/train_set_temp.parquet",
            str(ds.input_file_names["train"]["temp"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/train_set_psal.parquet",
            str(ds.input_file_names["train"]["psal"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_set_temp.parquet",
            str(ds.input_file_names["test"]["temp"]),
        )
        self.assertEqual(
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_set_psal.parquet",
            str(ds.input_file_names["test"]["psal"]),
        )

    def test_read_files(self):
        """
        Confirm that training and test data sets are loaded into
        Polars DataFrame objects, with expected shapes for each variable.
        """
        ds = InputTrainingSetA(self.config)
        ds.input_file_names = self.input_file_names

        ds.process_targets()

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 42)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 12)
        self.assertEqual(ds.test_sets["temp"].shape[1], 41)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 42)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 14)
        self.assertEqual(ds.test_sets["psal"].shape[1], 41)
