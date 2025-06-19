from abc import ABC
from dmqclib.utils.config import read_config


class DataSetBase(ABC):
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, dataset_type: str, dataset_name: str, config_file: str = None):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        config = read_config(config_file, "datasets.yaml")
        if dataset_name not in config:
            raise ValueError(
                f"Dataset name '{dataset_name}' not found in config file '{config_file}'"
            )
        dataset_info = config[dataset_name]

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = dataset_info[dataset_type].get("base_class")
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        # Set member variables
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.config_file_name = config.get("config_file_name")
        self.base_class_name = base_class
        self.dataset_info = dataset_info
        self.path_info = config.get("path_info")

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_type}(dataset=={self.dataset_name}, class={self.base_class_name})"
