from abc import ABC
import polars as pl
from dmqclib.utils.config import read_config
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_locate_path


class LocatePositionBase(ABC):
    """
    Base class for identifying position classes like LocateDataSetA, LocateDataSetB, LocateDataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
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
        base_class = dataset_info["locate"].get("base_class")
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
        self.__build_output_file_names()
        self.input_data = input_data
        self.selected_profiles = selected_profiles

    def __build_output_file_names(self):
        """
        Set the input file from configuration entries to the member variable 'self.input_file_name'.
        """
        folder_name = self.dataset_info["locate"].get("folder_name", "")

        self.output_file_names = {k: build_full_locate_path(
            self.path_info, folder_name, get_file_name_from_config(v, self.config_file_name)
        ) for k, v in self.dataset_info["locate"]["targets"].items()}

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_name}(class={self.base_class_name})"
