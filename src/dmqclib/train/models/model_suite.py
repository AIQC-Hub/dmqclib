"""
This module provides a ModelSuite class, inheriting
from `dmqclib.common.base.model_base.ModelBase`.

It facilitates training, prediction, and evaluation with multiple ML methods.
"""

import copy

from typing import Dict, Any, List, Self

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.model_base import ModelBase

from dmqclib.common.loader.single_model_loader import (
    load_single_model_class_with_class_name,
)


class ModelSuite(ModelBase):
    """
    A model suite class for training and testing.

    Inherits from :class:`ModelBase` to use the common model class interface.

    Features include:
    - Automatic application of ``model_params`` from the YAML config, if defined;

    .. note::
       This class sets :attr:`expected_class_name` to ``"ModelSuite"``.
    """

    expected_class_name: str = "ModelSuite"
    short_name: str = "MS"
    multi = True

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the Decision Tree model with default or user-specified parameters.

        :param config: A configuration object providing model parameters.
        :type config: ConfigBase
        """
        super().__init__(config=config)

        self.default_methods: List[str] = [
            "Logit",
            "LDA",
            "SVM",
            "DT",
            "XGB",
            "RF",
            "GNB",
            "KNN",
            "MLP",
        ]
        self.methods = self.config.get_step_params("model").get(
            "methods", self.default_methods
        )

        self.method_objs: Dict[str, Any] = {
            m: self._load_model_class_with_method_name(config, m) for m in self.methods
        }

    def _load_model_class_with_method_name(
        self, config: ConfigBase, method: str
    ) -> Any:
        config_method = copy.deepcopy(config)
        config_method.set_base_class("model", method)
        return load_single_model_class_with_class_name(config_method, method)

    def _get_model_class(self) -> Any:
        """
        :return: None.
        """
        return None  # pragma: no cover

    def set_enable_shap(self, enable_shap: bool):
        """
        Set all shap flag in all models.
        """
        for method_obj in self.method_objs.values():
            method_obj.enable_shap = enable_shap

    def build(self) -> None:
        """
        Build the model architecture or pipeline.
        """
        pass  # pragma: no cover

    def test(self) -> None:
        """
        Evaluate the model performance on a provided test set or validation data.
        """
        pass  # pragma: no cover

    def update_nthreads(self, model: Self) -> Self:
        """
        Update the number of threads set in the model.

        :param model: The model needs to be updated.
        :type model: Self
        """
        pass  # pragma: no cover
