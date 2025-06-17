import unittest
from pathlib import Path
from dmqclib.utils.config import read_config


class TestReadConfig(unittest.TestCase):

    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )

    def test_read_config_with_explicit_file(self):
        """
        Test when config_file is explicitly specified.
        """
        data = read_config(config_file=str(self.explicit_config_file_path))
        self.assertIsNotNone(data, "Data should not be None")
        self.assertIn("path_info", data, "Key 'path_info' should be in the YAML")
        self.assertEqual(data["path_info"]["input_path"], "/path/to/data")
        self.assertEqual(data["path_info"]["input_folder"], "input")
        self.assertEqual(data["path_info"]["data_path"], "/path/to/data")
        self.assertEqual(data["path_info"]["train_folder"], "train")
        self.assertEqual(data["path_info"]["validate_folder"], "validate")
        self.assertEqual(data["path_info"]["test_folder"], "test")

    def test_read_config_with_config_name(self):
        """
        Test when only config_name is specified (check that the file can be
        found by traversing up parent_level directories).
        Adjust parent_level if your directory structure is different.
        """
        data = read_config(config_name="datasets.yaml", parent_level=3)
        self.assertIsNotNone(data, "Data should not be None")
        self.assertIn("path_info", data, "Key 'path_info' should be in the YAML")

    def test_read_config_no_params_raises_error(self):
        """
        Test that ValueError is raised if neither config_file nor config_name is provided.
        """
        with self.assertRaises(ValueError):
            read_config()

    def test_read_config_nonexistent_file(self):
        """
        Test that FileNotFoundError is raised if a non-existent file is specified.
        """
        with self.assertRaises(FileNotFoundError):
            read_config(config_file="non_existent.yaml")
