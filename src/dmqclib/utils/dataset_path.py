import os
from typing import Dict, Optional


def build_full_input_path(
    path_info: Dict, folder_name2: Optional[str], file_name: str
) -> str:
    """
    Build a full input path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        path_info["input"]["base_path"] +
        "/" + path_info["input"]["folder_name"] +
        "/" + folder_name2 +
        "/" + file_name

    Both path_info["input"]["folder_name"] and folder_name2
    can be None or an empty string. They can also include "..".

    :param path_info: A dictionary that must contain:
                      path_info["input"]["base_path"] (str)
                      path_info["input"]["folder_name"] (str or None)
    :param folder_name2: The input folder name (can be None or empty string).
    :param file_name: The name of the file to append at the end of the path.
    :return: A string representing the full path.
    """
    base_path = path_info["input"]["base_path"]
    folder_name1 = path_info["input"].get("folder_name", "") or ""
    folder_name2 = folder_name2 or ""

    return os.path.normpath(
        os.path.join(base_path, folder_name1, folder_name2, file_name)
    )


def _build_full_data_path(
    path_info: Dict, data_type: str, folder_name1: Optional[str], file_name: str
) -> str:
    """
    Build a full data path based on the fields in the 'config' dictionary and
    the provided arguments. The assembled path is:
        path_info[data_type]["base_path"] +
        "/" + folder_name1 +
        "/" + path_info[data_type]["folder_name"] +
        "/" + file_name

    Both folder_name1 and path_info[data_type]["folder_name"]
    can be None or an empty string. They can also include "..".

    :param path_info: A dictionary that must contain:
                   path_info[data_type]["base_path"] (str)
                   path_info[data_type]["folder_name"] (str or None)
    :param folder_name1: The data folder name (can be None or empty string).
    :param file_name: The name of the file to append at the end of the path.

    :return: A string representing the full path.
    """
    base_path = path_info[data_type]["base_path"]
    folder_name1 = folder_name1 or ""
    folder_name2 = path_info[data_type].get("folder_name", "") or ""

    return os.path.normpath(
        os.path.join(base_path, folder_name1, folder_name2, file_name)
    )


def build_full_select_path(
    path_info: Dict, folder_name1: Optional[str], file_name: str
) -> str:
    """
    Returns the full path with the specified file name for the select module.
    """
    return _build_full_data_path(path_info, "select", folder_name1, file_name)


def build_full_locate_path(
    path_info: Dict, folder_name1: Optional[str], file_name: str
) -> str:
    """
    Returns the full path with the specified file name for the locate module.
    """
    return _build_full_data_path(path_info, "locate", folder_name1, file_name)
