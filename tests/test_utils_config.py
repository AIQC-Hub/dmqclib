import unittest
from pathlib import Path

from dmqclib.utils.config import read_config


class TestReadConfig(unittest.TestCase):
    """
    A suite of tests verifying proper functionality of the read_config function
    under various usage scenarios (explicit file path, config name only,
    missing arguments, non-existent file).
    """

    def setUp(self):
        """
        Set the test data configuration file path for use in multiple tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_read_config_with_explicit_file(self):
        """
        Verify that read_config can load a specific YAML file path, ensuring
        certain keys ('data_sets', 'path_info_sets') are present in the result.
        """
        config = read_config(config_file=str(self.config_file_path))
        self.assertIsNotNone(config, "Data should not be None")
        self.assertIn("data_sets", config, "Key 'data_sets' should be in the YAML")
        self.assertIn(
            "path_info_sets", config, "Key 'path_info_sets' should be in the YAML"
        )

    def test_read_config_with_config_name(self):
        """
        Verify that read_config can find and load a YAML file from
        multiple parent directories when only config_file_name is specified.
        """
        config = read_config(
            config_file_name="prepare_config_template.yaml", parent_level=3
        )
        self.assertIsNotNone(config, "Data should not be None")
        self.assertIn("data_sets", config, "Key 'data_sets' should be in the YAML")
        self.assertIn(
            "path_info_sets", config, "Key 'path_info_sets' should be in the YAML"
        )

    def test_read_config_no_params_raises_error(self):
        """
        Check that ValueError is raised if neither config_file nor config_file_name is provided.
        """
        with self.assertRaises(ValueError):
            read_config()

    def test_read_config_nonexistent_file(self):
        """
        Ensure that FileNotFoundError is raised for a file path that does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            read_config(config_file="non_existent.yaml")
