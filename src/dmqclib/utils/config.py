import os
from pathlib import Path
from typing import Dict

import yaml


def get_config_file(
    config_file: str = None, config_file_name: str = None, parent_level: int = 3
) -> str:
    if config_file is None:
        if config_file_name is None:
            raise ValueError(
                "'config_file_name' cannot be None when 'config_file' is None"
            )
        config_file = (
            Path(__file__).resolve().parents[parent_level] / "config" / config_file_name
        )

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File '{config_file}' does not exist.")

    return config_file


def read_config(
    config_file: str = None,
    config_file_name: str = None,
    parent_level: int = 3,
    add_config_file_name=True,
) -> Dict:
    """
    Reads either a YAML configuration file specified in config_file
    or a file named config_file_name after traversing parent_level directories
    upward from this file.

    :param config_file: The full path name of the config file.
    :param config_file_name: The name of the config file (e.g., "datasets.yaml").
    :param parent_level: Number of directories to go up from __file__.
    :param add_config_file_name: Add the file name when set to True.

    :return: A dictionary representing the parsed YAML file content.
    """

    config_file = get_config_file(config_file, config_file_name, parent_level)

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if add_config_file_name:
        data["config_file_name"] = config_file

    return data


def get_config_item(config: Dict, section: str, name: str) -> Dict:
    """
    Get a specific item from a given section in a YAML configuration file.

    :param config: config dictionary.
    :param section: The section in the YAML file to retrieve data from.
    :param name: The name of the item to find within the section.
    :return: A dictionary representing the item, or None if not found.
    """

    for item in config[section]:
        if item.get("name") == name:
            return item

    raise ValueError(f"Item with name '{name}' not found in section '{section}'.")
