import os
from typing import Dict, Optional


def build_full_input_path(
    path_info: Dict, input_folder: Optional[str], file_name: str
) -> str:
    """
    Build a full input path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        path_info["input_path"] +
        "/" + path_info["input_folder"] +
        "/" + input_folder +
        "/" + file_name

    Both path_info["input_folder"] and the input_folder argument
    can be None or an empty string. They can also include "..".

    :param path_info: A dictionary that must contain:
                      path_info["input_path"] (str)
                      path_info["input_folder"] (str or None)
    :param input_folder: The input folder name (can be None or empty string).
    :param file_name: The name of the file to append at the end of the path.
    :return: A string representing the full path.
    """
    base_path = path_info["input_path"]

    # config's optional input_folder
    config_input_folder = path_info.get("input_folder", "")
    if config_input_folder is None:
        config_input_folder = ""

    # Function argument input_folder
    if input_folder is None:
        input_folder = ""

    # Use os.path.join to properly handle slashes and special cases
    first_level_path = (
        os.path.join(base_path, config_input_folder)
        if config_input_folder
        else base_path
    )
    second_level_path = (
        os.path.join(first_level_path, input_folder)
        if input_folder
        else first_level_path
    )

    # Finally join with the file_name
    full_input_path = os.path.normpath(os.path.join(second_level_path, file_name))

    return full_input_path


def build_full_data_path(
    path_info: Dict, data_folder: str, data_type: str, file_name: str
) -> str:
    """
    Build a full data path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        path_info["data_path"] +
        "/" + data_folder +
        "/" + path_info[data_type + "_folder"] +
        "/" + file_name

    Both path_info[data_type] and the data_folder argument
    can be None or an empty string. They can also include "..".

    :param path_info: A dictionary that must contain:
                   path_info["data_path"] (str)
                   path_info[data_type + "_folder"] (str or None)
    :param data_folder: The data folder name (can be None or empty string).
    :param data_type: The data type (e.g., "train", "validate", "test").
    :param file_name: The name of the file to append at the end of the path.
    :return: A string representing the full path.
    """
    base_path = path_info["data_path"]

    # data_folder can be None or empty
    if not data_folder:
        data_folder = ""

    # path_info[data_type] can be None or empty
    data_type_folder = path_info.get(data_type + "_folder", "")
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
