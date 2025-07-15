"""
This module contains unit tests for the ClassificationConfig class,
ensuring its ability to load, validate, and select configuration settings
for classification tasks, as well as generate correct file paths.
"""

import unittest
from pathlib import Path

from dmqclib.common.config.classify_config import ClassificationConfig


class TestClassificationConfig(unittest.TestCase):
    """
    A suite of tests ensuring DataSetConfig can validate configurations,
    select datasets correctly, and generate file/folder paths as expected.
    """

    def setUp(self):
        """
        Set up references to valid and template configuration files
        to be used in subsequent tests.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "config_classify_set_template.yaml"
        )

    def test_valid_config(self):
        """
        Verify that validating a well-formed configuration reports it as valid.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        msg = ds.validate()
        self.assertIn("valid", msg)

    def test_invalid_config(self):
        """
        Verify that validating a malformed configuration reports it as invalid.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_invalid.yaml"
        )
        ds = ClassificationConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        """
        Check that the correct sections (path_info, target_set, etc.)
        are loaded from a valid configuration.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 8)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["feature_set"]), 2)
        self.assertEqual(len(ds.data["feature_param_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_load_dataset_config_twice(self):
        """
        Confirm that calling select() multiple times does not break anything.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")

    def test_invalid_dataset_name(self):
        """
        Check that attempting to select an unavailable dataset name
        raises a ValueError.
        """
        ds = ClassificationConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.select("INVALID_NAME")

    def test_input_folder(self):
        """
        Verify that input folder paths are generated as expected using template config.
        """
        ds = ClassificationConfig(str(self.template_file))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name(
            "input",
            ds.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        self.assertEqual(input_file_name, "/path/to/input/nrt_cora_bo_4.parquet")

    def test_summary_folder(self):
        """
        Confirm that files placed in a 'summary' folder are resolved correctly.
        """
        ds = ClassificationConfig(str(self.template_file))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name("summary", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/dataset_0001/summary/test.txt")

    def test_classify_folder(self):
        """
        Confirm that files placed in a 'classify' folder are resolved correctly.
        """
        ds = ClassificationConfig(str(self.template_file))
        ds.select("classification_0001")
        input_file_name = ds.get_full_file_name("classify", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/dataset_0001/classify/test.txt")
