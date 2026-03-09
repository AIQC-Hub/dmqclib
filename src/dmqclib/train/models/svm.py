"""
This module provides a Support Vector Machine (SVM) model wrapper, inheriting
from `dmqclib.common.base.scikit_learn_model_base.SklearnModelBase`.

It facilitates training, prediction, and evaluation of an SVM classifier using
Polars DataFrames, converting them to Pandas for compatibility with the
`sklearn` library.
"""

from typing import Dict, Any

from sklearn.svm import SVC as SklearnSVC

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.scikit_learn_model_base import SklearnModelBase


class SVM(SklearnModelBase):
    """
    A Support Vector Machine (SVM) model wrapper class for training and testing.

    Inherits from :class:`SklearnModelBase` to reuse common Scikit-Learn API logic.

    Features include:
    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Uses ``sklearn.svm.SVC``.
    - Enforces ``probability=True`` by default to support the generation of
      contingency tables and ROC/PR curves used in the parent class.
    - Uses a linear kernel by default.

    .. note::
       This class sets :attr:`expected_class_name` to ``"SVM"``.
       Standard SVM implementations do not support the ``n_jobs`` parameter directly.
    """

    expected_class_name: str = "SupportVectorMachine"
    short_name: str = "SVM"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the SVM model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        self.model_params: Dict[str, Any] = {
            "C": 1.0,
            "kernel": "linear",
            "gamma": "scale",
            "probability": True,  # Required for predict_proba used in base class
            "tol": 1e-3,
            "max_iter": -1,
            "random_state": None,
        }
        # Update model parameters with config step parameters
        model_params = self.config.get_model_params(self.expected_class_name, self.short_name)
        self.model_params.update(model_params)

    def _get_model_class(self) -> Any:
        """
        Return the Scikit-Learn SVC class.

        :return: The SVC class.
        """
        return SklearnSVC
