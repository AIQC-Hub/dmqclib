import os
from abc import ABC
from typing import List, Dict, Optional

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
    Base class for dataset or training configuration classes.

    Subclasses must define an ``expected_class_name`` attribute, which is
    used to validate that the YAML configuration matches the intended
    class signature.

    .. note::

       This class extends :class:`abc.ABC` to indicate that it is an
       abstract base class and should not be instantiated directly.
    """

    expected_class_name = None  # Must be overridden by child classes

    def __init__(self, section_name: str, config_file: str) -> None:
        """
        Initialize the configuration object for a specific section in the
        YAML configuration file.

        :param section_name: The name of the configuration section to load,
                             for example "data_sets" or "training_sets".
        :type section_name: str
        :param config_file: The path to the YAML configuration file.
        :type config_file: str
        :raises NotImplementedError: If no ``expected_class_name`` is set
                                     by a child class.
        :raises ValueError: If the provided ``section_name`` is not among
                            the supported ones (i.e., "data_sets" or
                            "training_sets").
        """
        if not self.expected_class_name:
            raise NotImplementedError(
                "Child class must define 'expected_class_name' attribute"
            )

        # Associates each section with a schema-retrieval function
        yaml_schemas = {
            "data_sets": get_data_set_config_schema,
            "training_sets": get_training_config_schema,
        }
        if section_name not in yaml_schemas:
            raise ValueError(f"Section name {section_name} is not supported.")

        self.section_name: str = section_name
        self.yaml_schema: Dict = yaml.safe_load(yaml_schemas.get(section_name)())
        self.full_config: Dict = read_config(config_file, add_config_file_name=False)
        self.valid_yaml: bool = False
        self.data: Optional[Dict] = None
        self.dataset_name: Optional[str] = None

    def validate(self) -> str:
        """
        Validate the loaded configuration against the expected YAML schema.

        :return: A message indicating success or the error encountered during validation.
        :rtype: str
        """
        try:
            validate(instance=self.full_config, schema=self.yaml_schema)
            self.valid_yaml = True
            return "YAML file is valid"
        except jsonschema.exceptions.ValidationError as e:
            self.valid_yaml = False
            return f"YAML file is invalid: {e.message}"

    def select(self, dataset_name: str) -> None:
        """
        Validate and select a specific dataset configuration entry from
        the loaded YAML.

        :param dataset_name: The identifier of the dataset to be selected
                             (must appear in the YAML under the
                             configured ``section_name``).
        :type dataset_name: str
        :raises ValueError: If the YAML configuration is found invalid
                            during validation.
        """
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

    def get_base_path(self, step_name: str) -> str:
        """
        Retrieve the base path for the given ``step_name`` from the configuration.

        :param step_name: The name of the step for which the base path is requested.
        :type step_name: str
        :return: A valid filesystem path (as a string).
        :rtype: str
        :raises ValueError: If the ``base_path`` is missing or set to None
                            in the configuration for this step.
        """
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
        """
        Return the parameter dictionary for a specified step.

        :param step_name: The name of the step to retrieve parameters for.
        :type step_name: str
        :return: A dictionary containing parameters for the specified step.
        :rtype: Dict
        """
        return self.data["step_param_set"]["steps"][step_name]

    def get_dataset_folder_name(self, step_name: str) -> str:
        """
        Return the folder name for the dataset associated with a particular step.

        :param step_name: The name of the step to retrieve the folder name for.
        :type step_name: str
        :return: The designated folder name for the dataset if defined, otherwise an empty string.
        :rtype: str
        """
        dataset_folder_name = self.data.get("dataset_folder_name", "")

        if (
            step_name in self.data["step_param_set"]["steps"]
            and "dataset_folder_name" in self.data["step_param_set"]["steps"][step_name]
        ):
            dataset_folder_name = self.get_step_params(step_name).get(
                "dataset_folder_name", ""
            )

        return dataset_folder_name

    def get_step_folder_name(
        self, step_name: str, folder_name_auto: bool = True
    ) -> str:
        """
        Determine the folder name for a particular step. If none is available
        in the config and ``folder_name_auto`` is True, the function falls
        back to using the original step name.

        :param step_name: The name of the step for which the folder is requested.
        :type step_name: str
        :param folder_name_auto: If True, automatically use the step name if
                                 no folder name is defined in the config.
        :type folder_name_auto: bool
        :return: A step folder name, either derived or explicitly defined.
        :rtype: str
        """
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

    def get_file_name(self, step_name: str, default_name: Optional[str] = None) -> str:
        """
        Retrieve the file name for a given step. If the step defines no
        file name in the config and ``default_name`` is also None, an error
        is raised.

        :param step_name: The step for which the file name is requested.
        :type step_name: str
        :param default_name: A fallback file name to use if none is in the config.
        :type default_name: str, optional
        :return: The file name for the specified step.
        :rtype: str
        :raises ValueError: If no file name can be found in the config or default.
        """
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
        default_file_name: Optional[str] = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ) -> str:
        """
        Construct a full file path by combining base path, optional dataset folder,
        step folder, and the file name. If placeholders are used, the function
        allows for partial dynamic formatting (see also :meth:`get_target_file_names`).

        :param step_name: The step name used to gather folder/file information.
        :type step_name: str
        :param default_file_name: A default file name if none is found.
        :type default_file_name: str, optional
        :param use_dataset_folder: Include the dataset folder in the path if True.
        :type use_dataset_folder: bool
        :param folder_name_auto: Derive a folder name from the step name if none is found.
        :type folder_name_auto: bool
        :return: A normalized file system path combining all components.
        :rtype: str
        """
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
        """
        Retrieve a class name for the specified step, as declared in the config.

        :param step_name: The name of the step for which the class is requested.
        :type step_name: str
        :return: The base class name designated by the YAML configuration.
        :rtype: str
        """
        return self.data["step_class_set"]["steps"][step_name]

    def get_target_variables(self) -> List:
        """
        Return the list of target variable dictionaries from the config.

        :return: A list of dictionaries describing each target variable.
        :rtype: List
        """
        return self.data["target_set"]["variables"]

    def get_target_names(self) -> List[str]:
        """
        Extract the target variable names from the list of target variable definitions.

        :return: A list of target variable names.
        :rtype: List[str]
        """
        return [x["name"] for x in self.get_target_variables()]

    def get_target_dict(self) -> Dict[str, Dict]:
        """
        Convert the list of target variable definitions into a dictionary
        keyed by variable name.

        :return: A dictionary mapping target variable names to their definitions.
        :rtype: Dict[str, Dict]
        """
        return {x["name"]: x for x in self.get_target_variables()}

    def get_target_file_names(
        self,
        step_name: str,
        default_file_name: Optional[str] = None,
        use_dataset_folder: bool = True,
        folder_name_auto: bool = True,
    ) -> Dict[str, str]:
        """
        Build a dictionary of full file names for each target variable,
        formatting placeholders in the file name with each target name.

        :param step_name: The step name used to build the file path.
        :type step_name: str
        :param default_file_name: A default file name if none is found in config.
        :type default_file_name: str, optional
        :param use_dataset_folder: If True, include the dataset folder in the path.
        :type use_dataset_folder: bool
        :param folder_name_auto: If True, derive the folder name from the step
                                 name if not explicitly stated in the config.
        :type folder_name_auto: bool
        :return: A dictionary mapping each target name (key) to a formatted file path.
        :rtype: Dict[str, str]
        """
        full_file_name = self.get_full_file_name(
            step_name, default_file_name, use_dataset_folder, folder_name_auto
        )
        return {
            x: full_file_name.format(target_name=x) for x in self.get_target_names()
        }

    def __repr__(self) -> str:
        """
        Return a simple string representation of the ConfigBase instance.

        :return: A description including the ``section_name`` used.
        :rtype: str
        """
        return f"ConfigBase(section_name={self.section_name})"
