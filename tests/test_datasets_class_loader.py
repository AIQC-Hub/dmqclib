import unittest
from pathlib import Path
from dmqclib.datasets.class_loader import load_input_dataset
from dmqclib.datasets.input.dataset_a import InputDataSetA


class TestInputClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )

    def test_load_dataset_valid_label(self):
        """
        Test that load_dataset returns an instance of InputDataSetA for the known label.
        """
        ds = load_input_dataset("NRT_BO_001", str(self.explicit_config_file_path))
        self.assertIsInstance(ds, InputDataSetA)
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_load_dataset_invalid_label(self):
        """
        Test that calling load_dataset with an invalid label raises a ValueError.
        """
        with self.assertRaises(ValueError):
            load_input_dataset(
                "NON_EXISTENT_LABEL", str(self.explicit_config_file_path)
            )
