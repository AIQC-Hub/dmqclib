import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class ProfileSelectionBase(DataSetBase):
    """
    Base class for profile selection and group labeling classes
    """

    def __init__(self, config: DataSetConfig, input_data: pl.DataFrame = None):
        super().__init__("select", config)

        # Set member variables
        self.default_file_name = "selected_profiles.parquet"
        self.output_file_name = self.config.get_full_file_name(
            "select", self.default_file_name
        )
        self.input_data = input_data
        self.selected_profiles = None

    @abstractmethod
    def label_profiles(self):
        """
        Label profiles to identify positive and negative groups.
        """
        pass  # pragma: no cover

    def write_selected_profiles(self):
        """
        Write selected profiles to parquet file
        """
        if self.selected_profiles is None:
            raise ValueError("Member variable 'selected_profiles' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.selected_profiles.write_parquet(self.output_file_name)
