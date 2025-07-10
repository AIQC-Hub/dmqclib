"""
Unit tests for verifying the correct loading and initialization of model classes
at various processing steps, using common loader functions.
"""

import unittest
from pathlib import Path

from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost


class TestModelClassLoader(unittest.TestCase):
    """
    Tests related to loading the Model class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_model_valid_config(self):
        """
        Check that load_model_class returns an XGBoost
        """
        ds = load_model_class(self.config)
        self.assertIsInstance(ds, XGBoost)

    def test_load_model_invalid_config(self):
        """
        Ensure that invalid model name raises a ValueError.

        """
        self.config.data["step_class_set"]["steps"]["model"] = "invalid_model_name"
        with self.assertRaises(ValueError):
            _ = load_model_class(self.config)
