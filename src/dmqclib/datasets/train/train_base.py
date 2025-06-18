from abc import ABC, abstractmethod
import polars as pl
from dmqclib.utils.config import read_config


class TrainingDataSetBin1Base(ABC):
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
    ):
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
        base_class = dataset_info["train"].get("base_class")
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        # Set member variables
        self.dataset_name = dataset_name
        self.config_file_name = config.get("config_file_name")
        self.base_class_name = base_class
        self.dataset_info = dataset_info
        self.path_info = config.get("path_info")
        self.input_data = input_data
        self.profiles = None

    @abstractmethod
    def label_profiles(self):
        """
        Label profiles in terms of positive and negative candidates
        """
        pass

    @abstractmethod
    def filter_profiles(self):
        """
        Filter profiles based on the labels
        """
        pass

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_name}(class={self.base_class_name})"
