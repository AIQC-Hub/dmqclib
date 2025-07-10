"""
This module provides utility functions for locating, reading, and parsing
configuration files, typically in YAML format.

It supports flexible methods for locating config files based on relative paths
or explicit paths, and facilitates easy retrieval of specific items within
the parsed configuration data.
"""

import os
from typing import Dict, Optional, Any

import yaml


def get_config_file(config_file: Optional[str] = None) -> str:
    """
    Determine the file path for a configuration file.

    If the resulting file path does not exist,
    a :class:`FileNotFoundError` is raised.

    :param config_file: The full path of the configuration file, if already known.
    :type config_file: str, optional
    :raises ValueError: If ``config_file`` is missing.
    :raises FileNotFoundError: If the resolved ``config_file`` path does not exist.
    :return: A string representing the resolved absolute path to the configuration file.
    :rtype: str
    """
    if config_file is None:
        raise ValueError("'config_file' cannot be None")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File '{config_file}' does not exist.")

    return str(config_file)


def read_config(
    config_file: Optional[str] = None, add_config_file_name: bool = True
) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file, returning its contents as a dictionary.

    Depending on the arguments, this function either:
      1. Uses the provided ``config_file`` path directly.
      2. Constructs a path by moving upward ``parent_level`` directories,
         then down into a "config" directory, naming the file ``config_file_name``.

    After reading the YAML file, if requested, the absolute path of the
    configuration file is added to the returned dictionary under the key
    "config_file_name".

    :param config_file: Full path to the config file, if already known.
    :type config_file: str, optional
    :param add_config_file_name: If True, the absolute path of the configuration file
                                 is added to the returned dictionary under
                                 the key "config_file_name". Defaults to True.
    :type add_config_file_name: bool
    :raises ValueError: If ``config_file`` is missing (propagated from :func:`get_config_file`).
    :raises FileNotFoundError: If no file is found at the resolved path (propagated from :func:`get_config_file`).
    :raises yaml.YAMLError: If the configuration file is not valid YAML.
    :return: A dictionary representing the parsed YAML config.
    :rtype: dict
    """
    resolved_file = get_config_file(config_file)

    with open(resolved_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if add_config_file_name:
        data["config_file_name"] = resolved_file

    return data


def get_config_item(config: Dict[str, Any], section: str, name: str) -> Dict[str, Any]:
    """
    Retrieve a named item from a specified section of a configuration dictionary.

    Iterates through the list of items in ``config[section]`` to find
    one whose "name" key matches the provided ``name``. If no matching
    item is found, a :class:`ValueError` is raised.

    :param config: The parsed configuration dictionary (as returned by :func:`read_config`).
    :type config: dict
    :param section: The section (key) in ``config`` whose list is searched.
    :type section: str
    :param name: The "name" field of the desired item in that section.
    :type name: str
    :raises KeyError: If the specified ``section`` does not exist in the configuration dictionary.
    :raises TypeError: If the value associated with the ``section`` key is not an iterable (e.g., not a list of dictionaries).
    :raises ValueError: If no matching item with the specified ``name`` is found in the given ``section``.
    :return: A dictionary representing the desired item from the config.
    :rtype: dict
    """
    for item in config[section]:
        if item.get("name") == name:
            return item

    raise ValueError(f"Item with name '{name}' not found in section '{section}'.")
