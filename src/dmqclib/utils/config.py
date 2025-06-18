import yaml
from pathlib import Path
from typing import Dict


def read_config(
    config_file: str = None, config_file_name: str = None, parent_level: int = 3
) -> Dict:
    """
    Reads either a YAML configuration file specified in config_file
    or a file named config_file_name after traversing parent_level directories
    upward from this file.

    :param config_file: The full path name of the config file.
    :param config_file_name: The name of the config file (e.g., "datasets.yaml").
    :param parent_level: Number of directories to go up from __file__.

    :return: A dictionary representing the parsed YAML file content.
    """

    if config_file is None:
        if config_file_name is None:
            raise ValueError(
                "'config_file_name' cannot be None when 'config_file' is None"
            )
        config_file = (
            Path(__file__).resolve().parents[parent_level] / "config" / config_file_name
        )

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["config_file_name"] = config_file

    return data
