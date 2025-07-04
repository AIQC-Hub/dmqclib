import os
from pathlib import Path
from typing import Dict, Optional, Any

import yaml


def get_config_file(
    config_file: Optional[str] = None,
    config_file_name: Optional[str] = None,
    parent_level: int = 4,
) -> str:
    """
    Determine the file path for a configuration file.

    If ``config_file`` is not provided, this function constructs a path
    by traversing ``parent_level`` directories upward from the location
    of this file, then descending into a "config" directory using
    ``config_file_name``. If the resulting file path does not exist,
    a :class:`FileNotFoundError` is raised.

    :param config_file: The full path of the configuration file, if already known.
    :type config_file: str, optional
    :param config_file_name: The name of the configuration file (e.g., "datasets.yaml").
                            Required if ``config_file`` is not given.
    :type config_file_name: str, optional
    :param parent_level: Number of directories to go up from this file's location.
    :type parent_level: int
    :raises ValueError: If both ``config_file`` and ``config_file_name`` are missing.
    :raises FileNotFoundError: If the resolved ``config_file`` path does not exist.
    :return: A string representing the resolved path to the configuration file.
    :rtype: str
    """
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

    return str(config_file)


def read_config(
    config_file: Optional[str] = None,
    config_file_name: Optional[str] = None,
    parent_level: int = 3,
    add_config_file_name: bool = True,
) -> Dict[str, Any]:
    """
    Read and parse a YAML configuration file, returning its contents as a dictionary.

    Depending on the arguments, this function either:
      1. Uses the provided ``config_file`` path directly.
      2. Constructs a path by moving upward ``parent_level`` directories,
         then down into a "config" directory, naming the file ``config_file_name``.

    After reading the YAML file, if requested, the absolute path is added
    to the returned dictionary under the key "config_file_name".

    :param config_file: Full path to the config file, if already known.
    :type config_file: str, optional
    :param config_file_name: Name of the config file (e.g., "datasets.yaml"),
                             required if ``config_file`` is not given.
    :type config_file_name: str, optional
    :param parent_level: Number of directories to go up from this file's location
                         before going down into "config" (used if constructing the path).
    :type parent_level: int
    :param add_config_file_name: If True, the path of the configuration file
                                 is added to the returned dictionary under
                                 the key "config_file_name". Defaults to True.
    :type add_config_file_name: bool
    :raises ValueError: If both ``config_file`` and ``config_file_name`` are missing.
    :raises FileNotFoundError: If no file is found at the resolved path.
    :return: A dictionary representing the parsed YAML config.
    :rtype: dict
    """
    resolved_file = get_config_file(config_file, config_file_name, parent_level)

    with open(resolved_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if add_config_file_name:
        data["config_file_name"] = resolved_file

    return data


def get_config_item(config: Dict[str, Any], section: str, name: str) -> Dict[str, Any]:
    """
    Retrieve a named item from a specified section of a configuration dictionary.

    Iterates through the list of items in ``config[section]`` to find
    one whose "name" key matches the provided ``name``. If none match,
    a :class:`ValueError` is raised.

    :param config: The parsed configuration dictionary (as returned by :func:`read_config`).
    :type config: dict
    :param section: The section (key) in ``config`` whose list is searched.
    :type section: str
    :param name: The "name" field of the desired item in that section.
    :type name: str
    :raises ValueError: If no matching item is found in the given section.
    :return: A dictionary representing the desired item from the config.
    :rtype: dict
    """
    for item in config[section]:
        if item.get("name") == name:
            return item

    raise ValueError(f"Item with name '{name}' not found in section '{section}'.")
