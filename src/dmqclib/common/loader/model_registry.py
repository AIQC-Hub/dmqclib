"""
This module provides a registry of model classes that can be used
during training or inference steps. Each key in the dictionary
corresponds to a model name (string), and each value is the class
constructor for that model.
"""

from typing import Dict, Type

from dmqclib.common.base.model_base import ModelBase
from dmqclib.train.models.decision_tree import DecisionTree
from dmqclib.train.models.random_forest import RandomForest
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.train.models.logistic_regression import LogisticRegression
from dmqclib.train.models.linear_discriminant_analysis import LinearDiscriminantAnalysis
from dmqclib.train.models.svm import SVM
from dmqclib.train.models.k_nearest_neighbors import KNearestNeighbors
from dmqclib.train.models.gaussian_naive_bayes import GaussianNaiveBayes
from dmqclib.train.models.mlp import MLP

#: A dictionary mapping model names to their corresponding Python classes.
#:
#: The keys are strings (e.g., "XGBoost"), and the values are class objects
#: that inherit from :class:`dmqclib.common.base.model_base.ModelBase`.
#:
#: :type: Dict[str, Type[ModelBase]]
MODEL_REGISTRY: Dict[str, Type[ModelBase]] = {
    "DecisionTree": DecisionTree,
    "DT": DecisionTree,
    "RandomForest": RandomForest,
    "RF": RandomForest,
    "XGBoost": XGBoost,
    "XGB": XGBoost,
    "LogisticRegression": LogisticRegression,
    "Logit": LogisticRegression,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "LDA": LinearDiscriminantAnalysis,
    "SupportVectorMachine": SVM,
    "SVM": SVM,
    "KNearestNeighbors": KNearestNeighbors,
    "KNN": KNearestNeighbors,
    "GaussianNaiveBayes": GaussianNaiveBayes,
    "GNB": GaussianNaiveBayes,
    "MultilayerPerceptron": MLP,
    "MLP": MLP,
}
