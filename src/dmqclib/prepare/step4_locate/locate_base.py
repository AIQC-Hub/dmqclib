import os
from abc import abstractmethod
from typing import Dict

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class LocatePositionBase(DataSetBase):
    """
    Base class to identify training data rows
    """

    def __init__(
        self,
        config: DataSetConfig,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
    ):
        super().__init__("locate", config)

        # Set member variables
        self.default_file_name = "{target_name}_rows.parquet"
        self.output_file_names = self.config.get_target_file_names(
            "locate", self.default_file_name
        )
        self.input_data = input_data
        self.selected_profiles = selected_profiles
        self.target_rows = {}

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k, v in self.config.get_target_dict().items():
            self.locate_target_rows(k, v)

    @abstractmethod
    def locate_target_rows(self, target_name: str, target_value: Dict):
        """
        Locate training data rows.
        """
        pass  # pragma: no cover

    def write_target_rows(self):
        """
        Write target_rows to parquet files
        """
        if len(self.target_rows) == 0:
            raise ValueError("Member variable 'target_rows' must not be empty.")

        for k, v in self.target_rows.items():
            os.makedirs(os.path.dirname(self.output_file_names[k]), exist_ok=True)
            v.write_parquet(self.output_file_names[k])
