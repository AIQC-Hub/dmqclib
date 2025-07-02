import os
import shutil

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig
from dmqclib.utils.config import get_config_file


def create_training_data_set(dataset_name: str, config: ConfigBase) -> None:
    config.load_dataset_config(dataset_name)