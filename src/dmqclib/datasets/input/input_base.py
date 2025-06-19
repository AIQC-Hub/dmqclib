from abc import ABC, abstractmethod
from dmqclib.utils.config import read_config
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_input_path
from dmqclib.utils.file_io import read_input_file


class InputDataSetBase(ABC):
    """
    Base class for data set classes like InputDataSetA, InputDataSetB, InputDataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, dataset_name: str, config_file: str = None):
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
        base_class = dataset_info["input"].get("base_class")
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
        self.__build_input_file_name()
        self.input_data = None

    def __build_input_file_name(self):
        """
        Set the input file from configuration entries to the member variable 'self.input_file_name'.
        """
        folder_name = self.dataset_info["input"].get("folder_name", "")
        file_name = get_file_name_from_config(self.dataset_info["input"], self.config_file_name)

        self.input_file_name = build_full_input_path(
            self.path_info, folder_name, file_name
        )

    def read_input_data(self):
        """
        Reads the input data from self.input_file_name using read_input_file,
        with file type and options derived from self.dataset_info.
        If either is missing or None, appropriate defaults (None for file_type,
        and empty dict for options) are used. The resulting DataFrame is stored
        in self.input_data.
        """
        input_file = self.input_file_name
        file_type = self.dataset_info["input"].get("file_type") or None
        options = self.dataset_info["input"].get("options") or {}

        self.input_data = read_input_file(input_file, file_type, options)

    @abstractmethod
    def select(self):
        """
        Selects columns of the data frame in self.input_data
        """
        pass

    @abstractmethod
    def filter(self):
        """
        Filter rows of the data frame in self.input_data
        """
        pass

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_name}(class={self.base_class_name})"
