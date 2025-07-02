import os
from abc import ABC
from typing import List, Dict

import jsonschema
import yaml
from jsonschema import validate

from dmqclib.common.config.yaml_schema import (
    get_data_set_config_schema,
    get_training_config_schema,
)
from dmqclib.utils.config import get_config_item
from dmqclib.utils.config import read_config


class ConfigBase(ABC):
    """
    Base class for data set classes like DataSetA, DataSetB, DataSetC, etc.
    Child classes must define an 'expected_class_name' attribute, which is
    validated against the YAML entry's 'base_class' field.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(
        self,
        section_name: str,
        config_file: str,
    ):
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Set member variables
        yaml_schemas = {
            "data_sets": get_data_set_config_schema,
            "training_sets": get_training_config_schema,
        }
        if section_name not in yaml_schemas:
            raise ValueError(f"Section name {section_name} is not supported.")

        self.section_name = section_name
        self.yaml_schema = yaml.safe_load(yaml_schemas.get(section_name)())
        self.full_config = read_config(config_file, add_config_file_name=False)
        self.valid_yaml = False
        self.data = None
        self.dataset_name = None

    def validate(self) -> str:
        try:
            validate(instance=self.full_config, schema=self.yaml_schema)
            self.valid_yaml = True
            return "YAML file is valid"
        except jsonschema.exceptions.ValidationError as e:
            self.valid_yaml = False
            return f"YAML file is invalid: {e.message}"

    def select(self, dataset_name: str):
        self.validate()
        if not self.valid_yaml:
            raise ValueError("YAML file is invalid")

        self.data = get_config_item(
            self.full_config, self.section_name, dataset_name
        ).copy()
        self.data["path_info"] = get_config_item(
            self.full_config, "path_info_sets", self.data["path_info"]
        )
        self.dataset_name = dataset_name

    def get_base_path(self, step_name: str):
        if step_name not in self.data["path_info"] or (
            step_name in self.data["path_info"]
            and "base_path" not in self.data["path_info"][step_name]
        ):
            step_name = "common"
        base_path = self.data["path_info"][step_name].get("base_path")

        if base_path is None:
            raise ValueError(
                "'base_path' for '{step_name}' not found or set to None in the config file"
            )

        return base_path

    def get_step_params(self, step_name: str) -> Dict:
        return self.data["step_param_set"]["steps"][step_name]

    def get_dataset_folder_name(self, step_name: str) -> str:
        dataset_folder_name = self.data.get("dataset_folder_name", "")

        if (
            step_name in self.data["step_param_set"]["steps"]
            and "dataset_folder_name" in self.data["step_param_set"]["steps"][step_name]
        ):
            dataset_folder_name = self.get_step_params(step_name).get(
                "dataset_folder_name", ""
            )

        return dataset_folder_name

    def get_step_folder_name(self, step_name: str, folder_name_auto=True) -> str:
        orig_step_name = step_name
        if step_name not in self.data["path_info"] or (
            step_name in self.data["path_info"]
            and "step_folder_name" not in self.data["path_info"][step_name]
        ):
            step_name = "common"
        step_folder_name = self.data["path_info"][step_name].get("step_folder_name")

        if step_folder_name is None:
            step_folder_name = orig_step_name if folder_name_auto else ""

        return step_folder_name

    def get_file_name(self, step_name: str, default_name: str = None) -> str:
        file_name = default_name
        if (
            step_name in self.data["step_param_set"]["steps"]
            and "file_name" in self.data["step_param_set"]["steps"][step_name]
        ):
            file_name = self.data["step_param_set"]["steps"][step_name].get(
                "file_name", ""
            )

        if file_name is None:
            raise ValueError(
                f"'file_name' for '{step_name}' not found or set to None in the config file"
            )

        return file_name

    def get_full_file_name(
        self,
        step_name: str,
        default_file_name: str = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ) -> str:
        base_path = self.get_base_path(step_name)
        dataset_folder_name = (
            self.get_dataset_folder_name(step_name) if use_dataset_folder else ""
        )
        folder_name = self.get_step_folder_name(step_name, folder_name_auto)
        file_name = self.get_file_name(step_name, default_file_name)

        return os.path.normpath(
            os.path.join(base_path, dataset_folder_name, folder_name, file_name)
        )

    def get_base_class(self, step_name: str) -> str:
        return self.data["step_class_set"]["steps"][step_name]

    def get_target_variables(self) -> List:
        return self.data["target_set"]["variables"]

    def get_target_names(self) -> List:
        return [x["name"] for x in self.get_target_variables()]

    def get_target_dict(self) -> List:
        return {x["name"]: x for x in self.get_target_variables()}

    def get_target_file_names(
        self,
        step_name: str,
        default_file_name: str = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ):
        full_file_name = self.get_full_file_name(
            step_name, default_file_name, use_dataset_folder, folder_name_auto
        )

        return {
            x: full_file_name.format(target_name=x) for x in self.get_target_names()
        }

    def __repr__(self):
        # Provide a simple representation
        return f"ConfigBase(section_name={self.section_name})"
