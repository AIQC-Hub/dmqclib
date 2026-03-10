"""
Unit tests for the SklearnModelBase class in dmqclib.common.base.scikit_learn_model_base.
This module verifies the correct functionality of the common Scikit-Learn API wrapper methods.
"""

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, ClassifierMixin

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.scikit_learn_model_base import SklearnModelBase
from dmqclib.common.config.training_config import TrainingConfig


class MockSklearnClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple mock classifier compatible with Scikit-Learn API for testing purposes.
    """

    def __init__(self, **kwargs):
        self.params = kwargs
        self.n_jobs = kwargs.get("n_jobs", 1)

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Return dummy predictions (all 0s)
        return np.zeros(X.shape[0])

    def predict_proba(self, X):
        # Return dummy probabilities (all 0.5s for class 1)
        # Shape: (n_samples, 2)
        n = X.shape[0]
        return np.column_stack((np.full(n, 0.5), np.full(n, 0.5)))


class ConcreteSklearnModel(SklearnModelBase):
    """
    Concrete implementation of SklearnModelBase for testing.
    """

    expected_class_name: str = (
        "XGBoost"  # Reusing a valid class name from config for simplicity
    )

    def __init__(self, config: ConfigBase) -> None:
        super().__init__(config)
        self.model_params = {"n_jobs": 1}

    def _get_model_class(self) -> Any:
        return MockSklearnClassifier


class TestSklearnModelBase(unittest.TestCase):
    """
    A suite of tests that verify the correctness of methods within SklearnModelBase.
    """

    def setUp(self):
        """
        Set up configuration and a concrete model instance.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.model_wrapper = ConcreteSklearnModel(self.config)

    def test_build(self):
        """
        Ensure build converts data and fits the underlying model.
        """
        # Setup dummy training data
        self.model_wrapper.training_set = pl.DataFrame(
            {"feature1": [1.0, 2.0, 3.0], "label": [0, 1, 0]}
        )

        self.model_wrapper.build()

        self.assertIsInstance(self.model_wrapper.model, MockSklearnClassifier)

    def test_build_empty_training_set(self):
        """
        Ensure build raises ValueError if training_set is missing.
        """
        self.model_wrapper.training_set = None
        with self.assertRaisesRegex(ValueError, "training_set"):
            self.model_wrapper.build()

    def test_predict(self):
        """
        Ensure predict generates predictions and scores in the correct format.
        """
        # We need a fitted model (or just an instance for the mock)
        self.model_wrapper.model = MockSklearnClassifier()
        self.model_wrapper.test_set = pl.DataFrame(
            {"feature1": [1.0, 2.0], "label": [0, 1]}
        )

        self.model_wrapper.predict()

        self.assertIsNotNone(self.model_wrapper.predictions)
        self.assertEqual(self.model_wrapper.predictions.shape, (2, 2))
        self.assertListEqual(self.model_wrapper.predictions.columns, ["class", "score"])
        # Based on MockSklearnClassifier logic:
        self.assertEqual(self.model_wrapper.predictions["class"][0], 0.0)
        self.assertEqual(self.model_wrapper.predictions["score"][0], 0.5)

    def test_predict_empty_test_set(self):
        """
        Ensure predict raises ValueError if test_set is missing.
        """
        with self.assertRaisesRegex(ValueError, "test_set"):
            self.model_wrapper.predict()

    def test_create_report(self):
        """
        Ensure create_report generates a DataFrame with metrics.
        """
        self.model_wrapper.k = 1
        self.model_wrapper.test_set = pl.DataFrame({"label": [0, 1, 0, 1]})
        self.model_wrapper.predictions = pl.DataFrame(
            {
                "class": [0, 1, 0, 0],  # One error
                "score": [0.5, 0.5, 0.5, 0.5],
            }
        )

        self.model_wrapper.create_report()

        self.assertIsNotNone(self.model_wrapper.report)
        self.assertIsInstance(self.model_wrapper.report, pl.DataFrame)
        self.assertIn("k", self.model_wrapper.report.columns)
        self.assertIn("metric_type", self.model_wrapper.report.columns)
        self.assertIn(
            "value", self.model_wrapper.report.columns
        )  # For accuracy/balanced

        # Verify specific rows exist
        metrics = self.model_wrapper.report["metric_type"].unique().to_list()
        self.assertIn("overall_accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("classification_report", metrics)

    def test_test_workflow(self):
        """
        Ensure the test method calls predict, create_report, and update_contingency_table.
        """
        # Mock the internal methods to verify flow
        self.model_wrapper.predict = MagicMock()
        self.model_wrapper.create_report = MagicMock()
        self.model_wrapper.update_contingency_table = MagicMock()

        self.model_wrapper.test()

        self.model_wrapper.predict.assert_called_once()
        self.model_wrapper.create_report.assert_called_once()
        self.model_wrapper.update_contingency_table.assert_called_once()

    def test_update_nthreads(self):
        """
        Ensure update_nthreads updates the underlying model's n_jobs.
        """
        self.model_wrapper.model = MockSklearnClassifier(n_jobs=1)
        self.model_wrapper.model_params = {"n_jobs": 4}

        self.model_wrapper.update_nthreads(self.model_wrapper)

        self.assertEqual(self.model_wrapper.model.n_jobs, 4)
