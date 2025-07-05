from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.utils.config import get_config_item


class ClassificationConfig(ConfigBase):
    """
    A configuration class for retrieving and organizing dataset-related 
    configurations specific to classification tasks.

    Extends :class:`ConfigBase` by adding logic to select datasets 
    from YAML-based configuration files. The selected dataset references 
    various sub-configurations (e.g., target sets, feature sets, and 
    step class definitions). These references are resolved and stored 
    within :attr:`data`.
    """

    expected_class_name: str = "ClassificationConfig"
    """
    The class name expected by this configuration to validate it 
    aligns with the YAML definition. Used by :class:`ConfigBase`.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initialize a new :class:`ClassificationConfig`.

        :param config_file: The path to the YAML file containing 
                            classification datasets and their sub-configurations.
        :type config_file: str
        :raises ValueError: If the YAML is invalid or missing the 
                            "data_sets" section.
        """
        super().__init__("classification_sets", config_file=config_file)

    def select(self, dataset_name: str) -> None:
        """
        Choose a dataset by name and load its sub-configuration items 
        (e.g., target sets, feature sets) into :attr:`data`.

        This method retrieves multiple related configurations by calling 
        :func:`dmqclib.common.utils.config.get_config_item` on relevant 
        sections of the YAML file.

        :param dataset_name: The name (key) of the desired dataset 
                             in the YAML's "data_sets" dictionary.
        :raises KeyError: If ``dataset_name`` is not present in `data_sets`.
        """
        super().select(dataset_name)
        self.data["target_set"] = get_config_item(
            self.full_config, "target_sets", self.data["target_set"]
        )
        self.data["feature_set"] = get_config_item(
            self.full_config, "feature_sets", self.data["feature_set"]
        )
        self.data["feature_param_set"] = get_config_item(
            self.full_config, "feature_param_sets", self.data["feature_param_set"]
        )
        self.data["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.data["step_class_set"]
        )
        self.data["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.data["step_param_set"]
        )
