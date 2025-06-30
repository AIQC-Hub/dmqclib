from abc import ABC
import jsonschema
from jsonschema import validate

from dmqclib.utils.config import read_config


class ConfigBase(ABC):
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self,
        module_name: str,
        config_file: str = None,
        config_file_name: str = None,
    ):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Set member variables
        yaml_schemas = {"DataSet": "config_data_set_schema.yaml",
                        "Training": "config_training_schema.yaml"}

        self.module_name = module_name
        self.yaml_schema = read_config(config_file_name=yaml_schemas[module_name],
                                       add_config_file_name = False)
        self.config = read_config(config_file, config_file_name,
                                       add_config_file_name = False)
        self.dataset_name = None
        self.valid_yaml = False

    def validate(self) -> str:
        try:
            validate(instance=self.config, schema=self.yaml_schema)
            self.valid_yaml = True
            return ("YAML file is valid")
        except jsonschema.exceptions.ValidationError as e:
            self.valid_yaml = False
            return(f"YAML file is invalid: {e.message}")

    def __repr__(self):
        # Provide a simple representation
        return f"ConfigBase(module_name={self.module_name})"
