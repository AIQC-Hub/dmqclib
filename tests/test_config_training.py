import unittest
from pathlib import Path

from dmqclib.config.training_config import TrainingConfig


class TestTrainingConfig(unittest.TestCase):
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

    def test_valid_config(self):
        """
        Test valid config
        """
        ds = TrainingConfig(str(self.config_file_path))
        msg = ds.validate()
        self.assertIn("valid", msg)

    def test_invalid_config(self):
        """
        Test invalid config
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_invalid.yaml"
        )
        ds = TrainingConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        ds = TrainingConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 5)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_invalid_dataset_name(self):
        ds = TrainingConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.load_dataset_config("INVALID_NAME")
