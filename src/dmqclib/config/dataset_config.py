from dmqclib.common.base.config_base import ConfigBase
from dmqclib.utils.config import get_config_item


class DataSetConfig(ConfigBase):
    """
    DataSetConfig provides dataset config interfaces
    """

    expected_class_name = "DataSetConfig"

    def __init__(
        self,
        config_file: str = None,
        config_file_name: str = None,
    ):
        super().__init__("DataSet", config_file=config_file, config_file_name=config_file_name)

        self.dataset_config = None
        self.path_info = None
        self.target_set = None
        self.feature_set = None
        self.feature_param_set = None
        self.step_class_set = None
        self.step_param_set = None

    def load_dataset_config(self, dataset_name:str):
        self.validate()
        if not self.valid_yaml:
            raise ValueError("YAML file is invalid")

        self.dataset_config = get_config_item(self.config, "data_sets", dataset_name)
        self.path_info = get_config_item(self.config, "path_info_sets", self.dataset_config["path_info"])
        self.target_set = get_config_item(self.config, "target_sets", self.dataset_config["target_set"])
        self.feature_set = get_config_item(self.config, "feature_sets", self.dataset_config["feature_set"])
        self.feature_param_set = get_config_item(self.config, "feature_param_sets", self.dataset_config["feature_param_set"])
        self.step_class_set = get_config_item(self.config, "step_class_sets", self.dataset_config["step_class_set"])
        self.step_param_set = get_config_item(self.config, "step_param_sets", self.dataset_config["step_param_set"])

        self.dataset_name = dataset_name
