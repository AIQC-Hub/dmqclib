from abc import abstractmethod
from typing import Dict
import polars as pl
from dmqclib.datasets.base.dataset_base import DataSetBase
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_locate_path


class LocatePositionBase(DataSetBase):
    """
    Base class to identify training data rows
    """

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
    ):
        super().__init__("locate", dataset_name, config_file=config_file)

        # Set member variables
        self._build_output_file_names()
        self.input_data = input_data
        self.selected_profiles = selected_profiles
        self.target_rows = {}

    def _build_output_file_names(self):
        """
        Set the output files based on configuration entries.
        """
        folder_name = self.dataset_info["locate"].get("folder_name", "")

        self.output_file_names = {
            k: build_full_locate_path(
                self.path_info,
                folder_name,
                get_file_name_from_config(v, self.config_file_name),
            )
            for k, v in self.dataset_info["locate"]["targets"].items()
        }

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k, v in self.dataset_info["locate"]["targets"].items():
            self.locate_target_rows(k, v)

    @abstractmethod
    def locate_target_rows(self, target_name: str, target_value: Dict):
        """
        Locate training data rows.
        """
        pass

    def write_target_rows(self):
        """
        Write target_rows to parquet files
        """
        if self.target_rows is None:
            raise ValueError("Member variable 'target_rows' must not be empty.")

        for k, v in self.target_rows.items():
            v.write_parquet(self.output_file_names[k])
