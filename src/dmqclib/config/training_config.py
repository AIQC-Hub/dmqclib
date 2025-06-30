from dmqclib.common.base.config_base import ConfigBase
from dmqclib.utils.config import get_config_item


class TrainingConfig(ConfigBase):
    """
    TrainingConfig provides training config interfaces
    """

    expected_class_name = "TrainingConfig"

    def __init__(
        self,
        config_file: str = None,
        config_file_name: str = None,
    ):
        super().__init__(
            "training_sets", config_file=config_file, config_file_name=config_file_name
        )

    def load_dataset_config(self, dataset_name: str):
        super().load_dataset_config(dataset_name)
        self.config["target_set"] = get_config_item(
            self.full_config, "target_sets", self.config["target_set"]
        )
        self.config["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.config["step_class_set"]
        )
        self.config["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.config["step_param_set"]
        )
