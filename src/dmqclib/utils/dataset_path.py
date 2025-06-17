import os
from typing import Dict, Optional


def build_full_input_path(
    config: Dict, input_folder: Optional[str], file_name: str
) -> str:
    """
    Build a full input path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        config["path_info"]["input_path"] +
        "/" + config["path_info"]["input_folder"] +
        "/" + input_folder +
        "/" + file_name

    Both config["path_info"]["input_folder"] and the input_folder argument
    can be None or an empty string. They can also include "..".

    :param config: A dictionary that must contain:
                   config["path_info"]["input_path"] (str)
                   config["path_info"]["input_folder"] (str or None)
    :param input_folder: The input folder name (can be None or empty string).
    :param file_name: The name of the file to append at the end of the path.
    :return: A string representing the full path.
    """
    base_path = config["path_info"]["input_path"]

    # config's optional input_folder
    config_input_folder = config["path_info"].get("input_folder", "")
    if config_input_folder is None:
        config_input_folder = ""

    # Function argument input_folder
    if input_folder is None:
        input_folder = ""

    # Use os.path.join to properly handle slashes and special cases
    path_with_config_folder = (
        os.path.join(base_path, config_input_folder)
        if config_input_folder
        else base_path
    )
    path_with_input_folder = (
        os.path.join(path_with_config_folder, input_folder)
        if input_folder
        else path_with_config_folder
    )

    # Finally join with the file_name
    full_input_path = os.path.normpath(os.path.join(path_with_input_folder, file_name))

    return full_input_path


def build_full_data_path(
    config: Dict, data_folder: str, data_type: str, file_name: str
) -> str:
    """
    Build a full data path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        config["path_info"]["data_path"] +
        "/" + data_folder +
        "/" + config["path_info"][data_type + "_folder"] +
        "/" + file_name

    Both config["path_info"][data_type] and the data_folder argument
    can be None or an empty string. They can also include "..".

    :param config: A dictionary that must contain:
                   config["path_info"]["data_path"] (str)
                   config["path_info"][data_type + "_folder"] (str or None)
    :param data_folder: The data folder name (can be None or empty string).
    :param data_type: The data type (e.g., "train", "validate", "test").
    :param file_name: The name of the file to append at the end of the path.
    :return: A string representing the full path.
    """
    base_path = config["path_info"]["data_path"]

    # data_folder can be None or empty
    if not data_folder:
        data_folder = ""

    # config["path_info"][data_type] can be None or empty
    data_type_folder = config["path_info"].get(data_type + "_folder", "")
    if not data_type_folder:
        data_type_folder = ""

    # Use os.path.join to properly handle directory separators and relative segments
    first_level_path = (
        os.path.join(base_path, data_folder) if data_folder else base_path
    )
    second_level_path = (
        os.path.join(first_level_path, data_type_folder)
        if data_type_folder
        else first_level_path
    )

    # Finally, join the filename
    full_data_path = os.path.normpath(os.path.join(second_level_path, file_name))

    return full_data_path
