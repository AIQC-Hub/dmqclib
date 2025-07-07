"""
Module providing utilities for writing YAML configuration templates and
reading them as instantiated configuration objects. Supports both
"prepare" and "train" modules using corresponding registry lookups.
"""

import os

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.common.utils.config import get_config_file
from dmqclib.common.config.yaml_templates import (
    get_config_train_set_template,
    get_config_data_set_template,
)


def write_config_template(file_name: str, module: str) -> None:
    """
    Write a YAML configuration template for the specified module
    ("prepare" or "train") to a file.

    This function:

      1. Chooses a template generator (from get_config_data_set_template
         or get_config_train_set_template) based on the ``module`` argument.
      2. Validates that the directory for ``file_name`` exists.
      3. Writes the generated YAML template text to the specified file.

    :param file_name: The path (including filename) where the YAML file will be written.
    :type file_name: str
    :param module: Determines which template to write; must be either "prepare" or "train".
    :type module: str
    :raises ValueError: If the specified module is not supported ("prepare" or "train" only).
    :raises IOError: If the directory of the specified file path does not exist.
    """
    function_registry = {
        "prepare": get_config_data_set_template,
        "train": get_config_train_set_template,
    }
    if module not in function_registry:
        raise ValueError(f"Module {module} is not supported.")

    yaml_text = function_registry[module]()
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path) and dir_path != "":
        raise IOError(f"Directory '{dir_path}' does not exist.")

    with open(file_name, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_text)


def read_config(file_name: str, module: str) -> ConfigBase:
    """
    Read a YAML configuration file as a :class:`ConfigBase` object,
    automatically selecting the appropriate subclass based on the given module.

    This function:
      1. Matches the specified module (either "prepare" or "train")
         to the corresponding configuration class (DataSetConfig or TrainingConfig).
      2. Resolves the file path by calling :meth:`get_config_file`.
      3. Instantiates and returns the matched configuration class with the resolved path.

    :param file_name: The path (including filename) to the YAML file.
    :type file_name: str
    :param module: Determines which configuration class to instantiate;
                   must be either "prepare" or "train".
    :type module: str
    :raises ValueError: If the module is not supported.
    :return: An instantiated configuration object (either DataSetConfig or TrainingConfig).
    :rtype: ConfigBase
    """
    config_classes = {
        "prepare": DataSetConfig,
        "train": TrainingConfig,
    }
    if module not in config_classes:
        raise ValueError(f"Module {module} is not supported.")

    config_file_name = get_config_file(file_name)
    return config_classes[module](config_file_name)
