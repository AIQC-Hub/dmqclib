"""
Unit tests for the ConfigBase class in dmqclib.common.base.config_base
This module verifies the correct functionality of ConfigBase's methods.
"""

import unittest
from pathlib import Path

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.config.training_config import TrainingConfig


class ConfigBaseWithExpectedName(ConfigBase):
    """
    DataSetWithExpectedName is used to test methods and variables in ConfigBase
    """

    expected_class_name: str = "ConfigBaseWithExpectedName"

    def __init__(self, section_name: str, config_file: str) -> None:
        super().__init__(section_name, config_file)


class TestDatasetBaseMethods(unittest.TestCase):
    """
    A suite of tests that verify the correctness of methods
    within the DataSetBase.
    """

    def setUp(self):
        """
        Set up a reference to the test configuration file (test_dataset_001.yaml)
        to be used by all subsequent tests in this class.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

    def test_common_base_path(self):
        """
        Ensure that undefined expected_class_name raises a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            _ = ConfigBase("data_sets", self.config_file_path)

    def test_section_name(self):
        """
        Ensure that unmatched section name raises a ValueError.
        """
        with self.assertRaises(ValueError):
            _ = ConfigBaseWithExpectedName(
                "invalid_section_name", self.config_file_path
            )

    def test_represented_str(self):
        """
        Ensure that the instance returns a correct string representation.
        """

        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        self.assertEqual(str(ds), "ConfigBase(section_name=data_sets)")

    def test_validation_error_with_select(self):
        """
        Ensure that incorrect yaml config contents raise a ValueError.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        ds.full_config = ""
        with self.assertRaises(ValueError):
            ds.select("NRT_BO_001")

    def test_representing_str(self):
        """
        Ensure that the instance returns a correct string representation.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        self.assertEqual(str(ds), "ConfigBase(section_name=data_sets)")

    def test_no_base_name(self):
        """
        Ensure that the instance returns a correct string representation.
        """
        ds = ConfigBaseWithExpectedName("data_sets", self.config_file_path)
        ds.select("NRT_BO_001")
        ds.data["path_info"]["common"]["base_path"] = None
        with self.assertRaises(ValueError):
            ds.get_base_path("invalid_step_name")


class TestConfigTemplates(unittest.TestCase):
    def test_read_datasets_template(self):
        conf = DataSetConfig("template:data_sets")
        self.assertIsNotNone(conf.full_config)

        self.assertIsNone(conf.data)
        conf.select("NRT_BO_001")
        self.assertIsNotNone(conf.data)
        print(conf.data)

    def test_read_training_template(self):
        conf = TrainingConfig("template:training_sets")
        self.assertIsNotNone(conf.full_config)

        self.assertIsNone(conf.data)
        conf.select("NRT_BO_001")
        self.assertIsNotNone(conf.data)

    def test_read_classification_template(self):
        conf = ClassificationConfig("template:classification_sets")
        self.assertIsNotNone(conf.full_config)

        self.assertIsNone(conf.data)
        conf.select("NRT_BO_001")
        self.assertIsNotNone(conf.data)
