"""
Module providing utilities for writing YAML configuration templates and
reading them as instantiated configuration objects. Supports "prepare",
"train", and "classify" stages using corresponding registry lookups.
"""

import os
from typing import Optional

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.common.config.yaml_templates import (
    get_config_train_set_template,
    get_config_data_set_template,
    get_config_classify_set_template,
)
from dmqclib.common.utils.config import get_config_file
from dmqclib.common.utils.config import read_config as utils_read_config

def write_config_template(file_name: str, stage: str) -> None:
    """
    Write a YAML configuration template for the specified stage
    ("prepare", "train", or "classify") to a file.

    This function:

      1. Chooses a template generator (from get_config_data_set_template,
         get_config_train_set_template, or get_config_classify_set_template)
         based on the ``stage`` argument.
      2. Validates that the directory for ``file_name`` exists.
      3. Writes the generated YAML template text to the specified file.

    :param file_name: The path (including filename) where the YAML file will be written.
    :type file_name: str
    :param stage: Determines which template to write; must be one of "prepare", "train", or "classify".
    :type stage: str
    :raises ValueError: If the specified stage is not supported ("prepare", "train", or "classify" only).
    :raises IOError: If the directory of the specified file path does not exist.
    """
    function_registry = {
        "prepare": get_config_data_set_template,
        "train": get_config_train_set_template,
        "classify": get_config_classify_set_template,
    }
    if stage not in function_registry:
        raise ValueError(f"Module {stage} is not supported.")

    yaml_text = function_registry[stage]()
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path) and dir_path != "":
        raise IOError(f"Directory '{dir_path}' does not exist.")

    with open(file_name, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_text)


def read_config(file_name: str, set_name: Optional[str]=None,
                auto_select: bool=True) -> ConfigBase:
    """
    Read a YAML configuration file as a :class:`ConfigBase` object,
    automatically selecting the appropriate subclass based on the given stage.

    This function:

      1. Resolves the file path by calling :meth:`get_config_file`.
      2. Read the specified YAML file to pick up the main key ("data_sets", "training_sets", or "classification_sets")
         to the corresponding configuration class (DataSetConfig, TrainingConfig,
         or ClassificationConfig).
      3. Instantiates and returns the matched configuration class with the resolved path.

    :param file_name: The path (including filename) to the YAML file.
    :type file_name: str
    :param set_name: The name (key) of the desired dataset in the YAML's dictionary.
    :type set_name: str
    :param auto_select: Select the data set name automatically if set to True
    :type auto_select: bool
    :raises ValueError: If the set name is not supported.
    :return: An instantiated configuration object (either DataSetConfig, TrainingConfig, or ClassificationConfig).
    :rtype: ConfigBase
    """
    config_file_name = get_config_file(file_name)
    config = utils_read_config(config_file_name)

    config_classes = {
        "data_sets": DataSetConfig,
        "training_sets": TrainingConfig,
        "classification_sets": ClassificationConfig,
    }
    matching_key = next((key for key in config_classes.keys() if key in config), None)
    if matching_key is None:
        raise ValueError("No valid 'set' name found in the provided YAML file.")

    config = config_classes[matching_key](config_file_name, auto_select)

    if set_name is not None:
        config.select(set_name)

    return config
