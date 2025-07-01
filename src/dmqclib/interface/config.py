import os
import shutil

from dmqclib.utils.config import get_config_file

def write_config_template(file_name: str, module: str) -> None:
    source_files = {
        "prepare": "prepare_config_template.yaml",
        "train": "",
    }
    if module not in source_files:
        raise ValueError(f"Module {module} is not supported.")
    
    source_name = get_config_file(config_file_name=source_files.get(module))
    if not os.path.exists(os.path.dirname(file_name)):
        raise IOError(f"Directory {os.path.dirname(file_name)} does not exist.")
    
    shutil.copyfile(source_name, file_name)
