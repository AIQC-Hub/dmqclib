"""
Unit tests for concrete model classes (XGBoost, LogisticRegression).

This module verifies that specific model implementations correctly integrate
with the dmqclib configuration system and initialize with expected parameters.
"""

import unittest
from pathlib import Path

import xgboost as xgb
from sklearn.linear_model import LogisticRegression as SklearnLR

from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.models.logistic_regression import LogisticRegression


class TestXGBoost(unittest.TestCase):
    """
    Tests for the XGBoost model wrapper.
    """

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_init_class(self):
        """Verify initialization and class naming."""
        ds = XGBoost(self.config)
        self.assertEqual(ds.expected_class_name, "XGBoost")
        self.assertEqual(ds.k, 0)

        # Verify the underlying model class getter
        self.assertEqual(ds._get_model_class(), xgb.XGBClassifier)

    def test_default_params(self):
        """Verify default parameters are set correctly."""
        ds = XGBoost(self.config)
        self.assertEqual(ds.model_params.get("n_estimators"), 100)
        self.assertEqual(ds.model_params.get("n_jobs"), -1)

    def test_config_params_override(self):
        """Verify configuration overrides default parameters."""
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "max_depth": 10,
            "scale_pos_weight": 5,
            "n_jobs": 4
        }
        ds = XGBoost(self.config)

        self.assertEqual(ds.model_params["max_depth"], 10)
        self.assertEqual(ds.model_params["scale_pos_weight"], 5)
        self.assertEqual(ds.model_params["n_jobs"], 4)


class TestLogisticRegression(unittest.TestCase):
    """
    Tests for the Logistic Regression model wrapper.
    """

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        # We need to temporarily spoof the config to expect "LogisticRegression"
        # otherwise ModelBase.__init__ will raise a ValueError.
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        # Inject the class name expectation into the config data
        self.config.data["step_class_set"]["steps"]["model"] = "LogisticRegression"

    def test_init_class(self):
        """Verify initialization and class naming."""
        ds = LogisticRegression(self.config)
        self.assertEqual(ds.expected_class_name, "LogisticRegression")

        # Verify the underlying model class getter
        self.assertEqual(ds._get_model_class(), SklearnLR)

    def test_default_params(self):
        """Verify default parameters are set correctly."""
        ds = LogisticRegression(self.config)
        self.assertEqual(ds.model_params.get("l1_ratio"), 0)
        self.assertEqual(ds.model_params.get("solver"), "lbfgs")

    def test_config_params_override(self):
        """Verify configuration overrides default parameters."""
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "C": 0.5,
            "max_iter": 500
        }
        ds = LogisticRegression(self.config)

        self.assertEqual(ds.model_params["C"], 0.5)
        self.assertEqual(ds.model_params["max_iter"], 500)
