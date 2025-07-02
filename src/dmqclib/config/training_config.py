from dmqclib.common.base.config_base import ConfigBase
from dmqclib.utils.config import get_config_item


class TrainingConfig(ConfigBase):
    """
    TrainingConfig provides training config interfaces
    """

    expected_class_name = "TrainingConfig"

    def __init__(self, config_file: str = None):
        super().__init__("training_sets", config_file=config_file)

    def select(self, dataset_name: str):
        super().select(dataset_name)
        self.data["target_set"] = get_config_item(
            self.full_config, "target_sets", self.data["target_set"]
        )
        self.data["step_class_set"] = get_config_item(
            self.full_config, "step_class_sets", self.data["step_class_set"]
        )
        self.data["step_param_set"] = get_config_item(
            self.full_config, "step_param_sets", self.data["step_param_set"]
        )
