from abc import abstractmethod
from typing import Dict
import polars as pl
from dmqclib.datasets.base.dataset_base import DataSetBase
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_locate_path


class LocatePositionBase(DataSetBase):
    """
    Base class to identify training data chunks
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
        self.target_locations = None

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
        Iterate all targets to locate training data chunks.
        """
        self.target_locations = {
            k: self.locate_target_chunks(k, v)
            for k, v in self.dataset_info["locate"]["targets"].items()
        }

    @abstractmethod
    def locate_target_chunks(self, target_name: str, target_value: Dict):
        """
        Locate training data chunks.
        """
        pass

    def write_target_locations(self):
        """
        Write target_locations to parquet files
        """
        if self.target_locations is None:
            raise ValueError("Member variable 'target_locations' must not be empty.")

        for k, v in self.target_locations.items():
            v.write_parquet(self.output_file_names[k])
