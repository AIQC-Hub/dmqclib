import os
import yaml
from pathlib import Path
from typing import Dict, Optional


def read_config(
    config_file: str = None, config_name: str = None, parent_level: int = 3
) -> Dict:
    """
    Reads a YAML configuration file named config_name after traversing
    parent_level directories upward from this file, then returning the 'config'
    directory.

    :param config_file: The full path name of the config file.
    :param config_name: The name of the config file (e.g., "datasets.yaml").
    :param parent_level: Number of directories to go up from __file__.
                    Adjust this based on your project structure.

    :return: A dictionary representing the parsed YAML file content.
    """

    if config_file is None:
        if config_name is None:
            raise ValueError("config_name cannot be None when config_file is None")
        config_file = (
            Path(__file__).resolve().parents[parent_level] / "config" / config_name
        )

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data
