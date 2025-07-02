import os

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig
from dmqclib.utils.config import get_config_file
from dmqclib.common.config.yaml_templates import (
    get_train_config_template,
    get_prepare_config_template,
)


def write_config_template(file_name: str, module: str) -> None:
    function_registry = {
        "prepare": get_prepare_config_template,
        "train": get_train_config_template,
    }
    if module not in function_registry:
        raise ValueError(f"Module {module} is not supported.")

    yaml_text = function_registry.get(module)()
    if not os.path.exists(os.path.dirname(file_name)):
        raise IOError(f"Directory '{os.path.dirname(file_name)}' does not exist.")

    with open(file_name, "w") as yaml_file:
        yaml_file.write(yaml_text)


def read_config(file_name: str, module: str) -> ConfigBase:
    config_classes = {
        "prepare": DataSetConfig,
        "train": TrainingConfig,
    }
    if module not in config_classes:
        raise ValueError(f"Module {module} is not supported.")

    config_file_name = get_config_file(file_name)

    return config_classes.get(module)(config_file_name)
