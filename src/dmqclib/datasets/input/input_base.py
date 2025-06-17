from dmqclib.utils.config import read_config
from dmqclib.utils.dataset_path import build_full_input_path


class InputDataSetBase:
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, dataset_name: str, config_file: str = None):
        data = read_config(config_file, "datasets.yaml")

        if dataset_name not in data:
            raise ValueError(
                f"Dataset name '{dataset_name}' not found in config file '{config_file}'"
            )

        dataset_config = data[dataset_name]

        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        if dataset_config.get("input_class") != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{dataset_config.get('input_class')}'"
            )

        # Set member variables
        self.dataset_name = dataset_name
        self.config_file = data.get("config_file")
        self.dataset_config = dataset_config
        self.path_info = data.get("path_info")
        self.__build_input_file_name()

    def __build_input_file_name(self):
        """
        Set the input file from configuration entries to the member variable 'self.input_file_name'.
        """
        input_folder = self.dataset_config.get("input_folder", "")
        file_name = self.dataset_config.get("input_file", "")
        if file_name is None or file_name == "":
            raise ValueError(
                f"'input_file' not found or set to None in config file '{self.config_file}'"
            )

        self.input_file_name = build_full_input_path(
            self.path_info, input_folder, file_name
        )

    def __repr__(self):
        # Provide a simple representation
        return f"{self.dataset_name}(class={self.expected_class_name})"
