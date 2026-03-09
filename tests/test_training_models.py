"""
Unit tests for concrete model classes (XGBoost, LogisticRegression, SVM, etc.).

This module verifies that specific model implementations correctly integrate
with the dmqclib configuration system and initialize with expected parameters.
"""

import unittest
from pathlib import Path

import xgboost as xgb
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from sklearn.svm import SVC as SklearnSVC
from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.neural_network import MLPClassifier as SklearnMLP

from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.models.logistic_regression import LogisticRegression
from dmqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from dmqclib.train.models.svm import SVM
from dmqclib.train.models.decision_tree import DecisionTree
from dmqclib.train.models.random_forest import RandomForest
from dmqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from dmqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from dmqclib.train.models.mlp import MLP


class TestXGBoost(unittest.TestCase):
    """Tests for the XGBoost model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        # XGBoost is usually the default in the test config, but we ensure it here
        self.config.data["step_class_set"]["steps"]["model"] = "XGBoost"

    def test_init_class(self):
        ds = XGBoost(self.config)
        self.assertEqual(ds.expected_class_name, "XGBoost")
        self.assertEqual(ds._get_model_class(), xgb.XGBClassifier)

    def test_multi_flag(self):
        ds = XGBoost(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = XGBoost(self.config)
        self.assertEqual(ds.model_params.get("n_estimators"), 100)
        self.assertEqual(ds.model_params.get("n_jobs"), -1)

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "max_depth": 10,
            "n_jobs": 4
        }
        ds = XGBoost(self.config)
        self.assertEqual(ds.model_params["max_depth"], 10)
        self.assertEqual(ds.model_params["n_jobs"], 4)


class TestLogisticRegression(unittest.TestCase):
    """Tests for the Logistic Regression model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "LogisticRegression"

    def test_init_class(self):
        ds = LogisticRegression(self.config)
        self.assertEqual(ds.expected_class_name, "LogisticRegression")
        self.assertEqual(ds._get_model_class(), SklearnLR)

    def test_multi_flag(self):
        ds = LogisticRegression(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = LogisticRegression(self.config)
        self.assertEqual(ds.model_params.get("penalty"), "l2")

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "C": 0.5,
            "max_iter": 500
        }
        ds = LogisticRegression(self.config)
        self.assertEqual(ds.model_params["C"], 0.5)
        self.assertEqual(ds.model_params["max_iter"], 500)


class TestLDA(unittest.TestCase):
    """Tests for the Linear Discriminant Analysis model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "LinearDiscriminantAnalysis"

    def test_init_class(self):
        ds = LinearDiscriminantAnalysis(self.config)
        self.assertEqual(ds.expected_class_name, "LinearDiscriminantAnalysis")
        self.assertEqual(ds._get_model_class(), SklearnLDA)

    def test_multi_flag(self):
        ds = LinearDiscriminantAnalysis(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = LinearDiscriminantAnalysis(self.config)
        self.assertEqual(ds.model_params.get("solver"), "svd")
        # Ensure n_jobs is NOT present (sklearn LDA doesn't support it)
        self.assertNotIn("n_jobs", ds.model_params)

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "tol": 1.0e-5
        }
        ds = LinearDiscriminantAnalysis(self.config)
        self.assertEqual(ds.model_params["tol"], 1.0e-5)


class TestSVM(unittest.TestCase):
    """Tests for the SVM model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "SVM"

    def test_init_class(self):
        ds = SVM(self.config)
        self.assertEqual(ds.expected_class_name, "SupportVectorMachine")
        self.assertEqual(ds._get_model_class(), SklearnSVC)

    def test_multi_flag(self):
        ds = SVM(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = SVM(self.config)
        self.assertEqual(ds.model_params.get("kernel"), "linear")
        # Probability must be True for the base class predict_proba logic
        self.assertTrue(ds.model_params.get("probability"))

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "C": 0.5,
            "kernel": "rbf"
        }
        ds = SVM(self.config)
        self.assertEqual(ds.model_params["C"], 0.5)
        self.assertEqual(ds.model_params["kernel"], "rbf")


class TestDecisionTree(unittest.TestCase):
    """Tests for the Decision Tree model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "DecisionTree"

    def test_init_class(self):
        ds = DecisionTree(self.config)
        self.assertEqual(ds.expected_class_name, "DecisionTree")
        self.assertEqual(ds._get_model_class(), SklearnDT)

    def test_multi_flag(self):
        ds = DecisionTree(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = DecisionTree(self.config)
        self.assertEqual(ds.model_params.get("criterion"), "gini")
        self.assertEqual(ds.model_params.get("splitter"), "best")

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "max_depth": 5,
            "min_samples_split": 4
        }
        ds = DecisionTree(self.config)
        self.assertEqual(ds.model_params["max_depth"], 5)
        self.assertEqual(ds.model_params["min_samples_split"], 4)


class TestRandomForest(unittest.TestCase):
    """Tests for the Random Forest model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "RandomForest"

    def test_init_class(self):
        ds = RandomForest(self.config)
        self.assertEqual(ds.expected_class_name, "RandomForest")
        self.assertEqual(ds._get_model_class(), SklearnRF)

    def test_multi_flag(self):
        ds = RandomForest(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = RandomForest(self.config)
        self.assertEqual(ds.model_params.get("n_estimators"), 100)
        self.assertEqual(ds.model_params.get("n_jobs"), -1)

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "n_estimators": 50,
            "max_features": "log2"
        }
        ds = RandomForest(self.config)
        self.assertEqual(ds.model_params["n_estimators"], 50)
        self.assertEqual(ds.model_params["max_features"], "log2")


class TestKNN(unittest.TestCase):
    """Tests for the K-Nearest Neighbors model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "KNearestNeighbors"

    def test_init_class(self):
        ds = KNearestNeighbors(self.config)
        self.assertEqual(ds.expected_class_name, "KNearestNeighbors")
        self.assertEqual(ds._get_model_class(), SklearnKNN)

    def test_multi_flag(self):
        ds = KNearestNeighbors(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = KNearestNeighbors(self.config)
        self.assertEqual(ds.model_params.get("n_neighbors"), 5)
        self.assertEqual(ds.model_params.get("n_jobs"), -1)

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "n_neighbors": 10,
            "weights": "distance"
        }
        ds = KNearestNeighbors(self.config)
        self.assertEqual(ds.model_params["n_neighbors"], 10)
        self.assertEqual(ds.model_params["weights"], "distance")


class TestGaussianNaiveBayes(unittest.TestCase):
    """Tests for the Gaussian Naive Bayes model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "GaussianNaiveBayes"

    def test_init_class(self):
        ds = GaussianNaiveBayes(self.config)
        self.assertEqual(ds.expected_class_name, "GaussianNaiveBayes")
        self.assertEqual(ds._get_model_class(), SklearnGNB)

    def test_multi_flag(self):
        ds = GaussianNaiveBayes(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = GaussianNaiveBayes(self.config)
        self.assertEqual(ds.model_params.get("var_smoothing"), 1e-9)

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "var_smoothing": 1e-5
        }
        ds = GaussianNaiveBayes(self.config)
        self.assertEqual(ds.model_params["var_smoothing"], 1e-5)


class TestMLP(unittest.TestCase):
    """Tests for the Multi-layer Perceptron model wrapper."""

    def setUp(self):
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["step_class_set"]["steps"]["model"] = "MLP"

    def test_init_class(self):
        ds = MLP(self.config)
        self.assertEqual(ds.expected_class_name, "MultilayerPerceptron")
        self.assertEqual(ds._get_model_class(), SklearnMLP)

    def test_multi_flag(self):
        ds = MLP(self.config)
        self.assertFalse(ds.multi)

    def test_default_params(self):
        ds = MLP(self.config)
        self.assertEqual(ds.model_params.get("hidden_layer_sizes"), (100,))
        self.assertEqual(ds.model_params.get("activation"), "relu")

    def test_config_params_override(self):
        self.config.data["step_param_set"]["steps"]["model"]["model_params"] = {
            "hidden_layer_sizes": (50, 50),
            "learning_rate_init": 0.01
        }
        ds = MLP(self.config)
        self.assertEqual(ds.model_params["hidden_layer_sizes"], (50, 50))
        self.assertEqual(ds.model_params["learning_rate_init"], 0.01)
