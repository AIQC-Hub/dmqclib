from dmqclib.common.base.config_base import ConfigBase
from dmqclib.utils.config import get_config_item


class DataSetConfig(ConfigBase):
    """
    A configuration class that provides dataset-related configuration interfaces.

    This class extends :class:`ConfigBase` with handling for one or more
    dataset-specific YAML sections, mapping them to container dictionaries
    within :attr:`data`. The selected dataset name is used to look up
    configurations for target sets, feature sets, step classes, etc.

    .. note::

       :attr:`expected_class_name` must match the YAML's ``base_class``
       if instantiated directly.
    """

    expected_class_name: str = "DataSetConfig"
    """
    The class name expected by the configuration. Used by 
    :class:`ConfigBase` to validate consistency with the YAML data.
    """

    def __init__(self, config_file: str) -> None:
        """
        Initialize a new :class:`DataSetConfig` instance.

        :param config_file: The path to the YAML configuration file.
        :type config_file: str
        :raises ValueError: If the YAML structure is invalid or the
                            file does not contain `data_sets` section.
        """
        super().__init__("data_sets", config_file=config_file)

    def select(self, dataset_name: str) -> None:
        """
        Select a dataset entry by name from :attr:`data_sets` in the YAML config,
        then retrieve related configuration items (e.g., target_set, feature_set, etc.).

        This method populates :attr:`data` with relevant sub-configurations by
        calling :func:`dmqclib.utils.config.get_config_item` on specified fields.

        :param dataset_name: The key name of the dataset to select from the YAML.
        :type dataset_name: str
        :raises KeyError: If the dataset name does not exist in the YAML's
                          `data_sets` dictionary.
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
