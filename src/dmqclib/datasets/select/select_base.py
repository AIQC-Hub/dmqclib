from abc import ABC, abstractmethod
import polars as pl
from dmqclib.utils.config import read_config
from dmqclib.utils.dataset_path import build_full_select_path


class ProfileSelectionBase(ABC):
    """
    Base class for profile selection data set classes like SelectDataSetA, SelectDataSetB, SelectDataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
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
        base_class = dataset_info["select"].get("base_class")
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
        self.__build_output_file_name()
        self.input_data = input_data
        self.selected_profiles = None

    def __build_output_file_name(self):
        """
        Set the input file from configuration entries to the member variable 'self.input_file_name'.
        """
        folder_name = self.dataset_info["select"].get("folder_name", "")
        file_name = self.dataset_info["select"].get("file_name", "")
        if file_name is None or file_name == "":
            raise ValueError(
                f"'input_file' not found or set to None in config file '{self.config_file_name}'"
            )

        self.output_file_name = build_full_select_path(
            self.path_info, folder_name, file_name
        )

    @abstractmethod
    def label_profiles(self):
        """
        Label profiles in terms of positive and negative candidates
        """
        pass

    def write_selected_profiles(self):
        """
        Write selected profiles to parquet file
        """
        if self.selected_profiles is None:
            raise ValueError("'selected_profiles' is empty.")

        self.selected_profiles.write_parquet(self.output_file_name)

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_name}(class={self.base_class_name})"
