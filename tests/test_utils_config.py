import unittest
from pathlib import Path
from dmqclib.utils.config import read_config


class TestReadConfig(unittest.TestCase):
    def setUp(self):
        """
        Set the test data config file.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )

    def test_read_config_with_explicit_file(self):
        """
        Test when config_file is explicitly specified.
        """
        config = read_config(config_file=str(self.explicit_config_file_path))
        self.assertIsNotNone(config, "Data should not be None")
        self.assertIn("path_info", config, "Key 'path_info' should be in the YAML")
        self.assertEqual(config["path_info"]["input"]["base_path"], "/path/to/data")
        self.assertEqual(config["path_info"]["input"]["folder_name"], "input")
        self.assertEqual(config["path_info"]["train"]["base_path"], "/path/to/data")
        self.assertEqual(config["path_info"]["train"]["folder_name"], "train")

    def test_read_config_with_config_name(self):
        """
        Test when only config_name is specified (check that the file can be
        found by traversing up parent_level directories).
        """
        config = read_config(config_file_name="datasets.yaml", parent_level=3)
        self.assertIsNotNone(config, "Data should not be None")
        self.assertIn("path_info", config, "Key 'path_info' should be in the YAML")

    def test_read_config_no_params_raises_error(self):
        """
        Test that ValueError is raised if neither config_file nor config_file_name is provided.
        """
        with self.assertRaises(ValueError):
            read_config()

    def test_read_config_nonexistent_file(self):
        """
        Test that FileNotFoundError is raised if a non-existent file is specified.
        """
        with self.assertRaises(FileNotFoundError):
            read_config(config_file="non_existent.yaml")
