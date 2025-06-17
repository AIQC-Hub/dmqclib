from dmqclib.utils.config import read_config


class InputDataSetBase:
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, label: str, config_file: str = None):
        data = read_config(config_file, "datasets.yaml")

        if label not in data:
            raise ValueError(
                f"Label '{label}' not found in config file '{config_file}'"
            )

        dataset_config = data[label]

        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        if dataset_config.get("input_class") != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{dataset_config.get('class')}'"
            )

        # Store common fields or config data
        self.label = label
        self.file = dataset_config.get("input_file")

    def __repr__(self):
        # Provide a simple representation; you could customize further if needed
        return f"{self.expected_class_name}(file={self.file})"
