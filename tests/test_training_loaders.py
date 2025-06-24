import unittest
from pathlib import Path

from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.training.step1_input.dataset_a import InputTrainingSetA


class TestTrainingInputClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "training.yaml"
        )

    def test_load_dataset_valid_label(self):
        """
        Test that load_dataset returns an instance of InputDataSetA for the known label.
        """
        ds = load_step1_input_training_set("NRT_BO_001", str(self.config_file_path))
        self.assertIsInstance(ds, InputTrainingSetA)
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_load_dataset_invalid_label(self):
        """
        Test that calling load_dataset with an invalid label raises a ValueError.
        """
        with self.assertRaises(ValueError):
            load_step1_input_training_set(
                "NON_EXISTENT_LABEL", str(self.config_file_path)
            )
