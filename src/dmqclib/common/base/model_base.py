import os
from abc import ABC, abstractmethod

from joblib import dump, load

from dmqclib.common.base.config_base import ConfigBase


class ModelBase(ABC):
    """
    Base class to model
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, config: ConfigBase):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = config.get_base_class("model")
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        model_params = config.data["step_param_set"]["steps"]["model"].get(
            "model_params", {}
        )

        self.config = config
        self.model_params = model_params

        self.training_set = None
        self.test_set = None
        self.model = None
        self.result = None
        self.k = 0

    @abstractmethod
    def build(self):
        """
        Build model
        """
        pass  # pragma: no cover

    @abstractmethod
    def test(self):
        """
        Test model.
        """
        pass  # pragma: no cover

    def load_model(self, file_name: str):
        """
        Read model.
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")

        self.model = load(file_name)

    def save_model(self, file_name: str):
        """
        Write model.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        dump(self.model, file_name)
