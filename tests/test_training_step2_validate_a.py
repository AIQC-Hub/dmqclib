import unittest
from pathlib import Path

import polars as pl

from dmqclib.training.step2_validate.kfold_validation import KFoldValidation
from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.training.models.empty_model import EmptyModel


class TestKFoldValidation(unittest.TestCase):
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

        self.ds_input = load_step1_input_training_set("NRT_BO_002", str(self.config_file_path))
        self.ds_input.input_file_names = self.input_file_names
        self.ds_input.process_targets()

    def test_init_valid_dataset_name(self):
        """Ensure ExtractDataSetA constructs correctly with a valid label."""
        ds = KFoldValidation("NRT_BO_002", str(self.config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_BO_002")

    def test_init_invalid_dataset_name(self):
        """Ensure ValueError is raised for an invalid label."""
        with self.assertRaises(ValueError):
            KFoldValidation("NON_EXISTENT_LABEL", str(self.config_file_path))

    def test_config_file(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation("NRT_BO_002", str(self.config_file_path))
        self.assertTrue("training.yaml" in ds.config_file_name)

    def test_base_model(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation("NRT_BO_002", str(self.config_file_path))
        self.assertIsInstance(ds.base_model, EmptyModel)

    def test_training_sets(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation("NRT_BO_002", str(self.config_file_path),
                             self.ds_input.training_sets)

        self.assertIsInstance(ds.training_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.training_sets["temp"].shape[0], 116)
        self.assertEqual(ds.training_sets["temp"].shape[1], 38)

        self.assertIsInstance(ds.training_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.training_sets["psal"].shape[0], 126)
        self.assertEqual(ds.training_sets["psal"].shape[1], 38)

    def test_process_targets(self):
        """Verify the config file is correctly set in the member variable."""
        ds = KFoldValidation("NRT_BO_002", str(self.config_file_path),
                             self.ds_input.training_sets)

        ds.process_targets()

        self.assertEqual(ds.built_models, {'temp': [1, 2, 3], 'psal': [1, 2, 3]})
        self.assertEqual(ds.results, {'temp': [10, 11, 12], 'psal': [10, 11, 12]})
        self.assertEqual(ds.summary,  {'temp': 100, 'psal': 100})
