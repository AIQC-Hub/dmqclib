import os
import shutil

from dmqclib.utils.config import get_config_file
from dmqclib.common.base.config_base import ConfigBase
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig


def write_config_template(file_name: str, module: str) -> None:
    source_files = {
        "prepare": "prepare_config_template.yaml",
        "train": "train_config_template.yaml",
    }
    if module not in source_files:
        raise ValueError(f"Module {module} is not supported.")

    source_name = get_config_file(config_file_name=source_files.get(module))
    if not os.path.exists(os.path.dirname(file_name)):
        raise IOError(f"Directory '{os.path.dirname(file_name)}' does not exist.")

    shutil.copyfile(source_name, file_name)


def read_config(file_name: str, module: str) -> ConfigBase:
    config_classes = {
        "prepare": DataSetConfig,
        "train": TrainingConfig,
    }
    if module not in config_classes:
        raise ValueError(f"Module {module} is not supported.")

    config_file_name = get_config_file(file_name)

    return config_classes.get(module)(config_file_name)
