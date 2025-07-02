from abc import ABC

from dmqclib.common.base.config_base import ConfigBase


class DataSetBase(ABC):
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, step_name: str, config: ConfigBase):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Validate that the YAML's "class" matches the child's declared class name
        base_class = config.get_base_class(step_name)
        if base_class != self.expected_class_name:
            raise ValueError(
                f"Configuration mismatch: expected class '{self.expected_class_name}' "
                f"but got '{base_class}'"
            )

        # Set member variables
        self.step_name = step_name
        self.config = config

    def __repr__(self):
        # Provide a simple representation
        return f"DataSetBase(step={self.step_name}, class={self.base_class_name})"
