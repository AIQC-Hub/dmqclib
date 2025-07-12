"""
Unit tests for verifying the correct loading and initialization of model classes
at various processing steps, using common loader functions.
"""

import unittest
from pathlib import Path

from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.common.loader.model_loader import load_model_class
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

    def test_build_model_empty_training_set(self):
        """
        Ensure that build raises a ValueError when training set is not set.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.build()

    def test_predict_model_empty_test_set(self):
        """
        Ensure that predict raises a ValueError when test set is not set.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.predict()

    def test_create_report_empty_test_set(self):
        """
        Ensure that create_report raises a ValueError when test set is not set.
        """
        ds = load_model_class(self.config)
        with self.assertRaises(ValueError):
            ds.create_report()

    def test_create_report_empty_predictions(self):
        """
        Ensure that create_report raises a ValueError when predictions are not set.
        """
        ds = load_model_class(self.config)
        ds.test_set = {}
        with self.assertRaises(ValueError):
            ds.create_report()
