import os
from abc import ABC, abstractmethod

from joblib import dump, load

from dmqclib.utils.config import read_config


class ModelBase(ABC):
    """
    Base class to model
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        config_file_name: str = "training.yaml",
    ):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        config = read_config(config_file, config_file_name)
        if dataset_name not in config:
            raise ValueError(
                f"Dataset name '{dataset_name}' not found in config file '{config_file}'"
            )
        dataset_info = config[dataset_name]

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = dataset_info["base_class"].get("model")
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        model_params = dataset_info.get("model_params", {})

        self.dataset_name = dataset_name
        self.config_file_name = config.get("config_file_name")
        self.base_class_name = base_class
        self.dataset_info = dataset_info
        self.path_info = config.get("path_info")
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
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")

        self.model = load(file_name)

    def save_model(self, file_name: str):
        """
        Write model.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        dump(self.model, file_name)
